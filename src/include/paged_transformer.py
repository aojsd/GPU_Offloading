import torch
import torch.nn as nn
from vllm import _custom_ops as ops

class TransformerArgs:
    def __init__(self, dim, n_heads, n_layers, norm_eps=1e-5):
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.norm_eps = norm_eps

class PagedTransformerData:
    def __init__(
        self, 
        batch_size: int,
        seq_len: int,
        max_num_blocks: int, 
        num_layers: int, 
        num_heads: int, 
        head_dim: int, 
        block_size: int = 16, 
        dtype: torch.dtype = torch.float16, 
        device: str = "cuda"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len # Store this for graph-safe access later
        
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.x = 16 // element_size
        
        # 1. Allocate Heaps
        self.key_heap = torch.randn(
            (max_num_blocks, num_heads, head_dim // self.x, block_size, self.x),
            dtype=dtype, device=device
        )
        self.val_heap = torch.randn(
            (max_num_blocks, num_heads, head_dim, block_size),
            dtype=dtype, device=device
        )
        
        # 2. Block Tables
        needed_logical_blocks = (seq_len + 1 + block_size - 1) // block_size
        total_blocks_needed = num_layers * batch_size * needed_logical_blocks
        
        if total_blocks_needed > max_num_blocks:
            raise RuntimeError(f"OOM: Heap size {max_num_blocks} too small. Needed {total_blocks_needed}.")
        
        self.context_lens = torch.full(
            (batch_size,), seq_len + 1, dtype=torch.int32, device=device
        )
        
        self.block_tables = torch.zeros(
            (num_layers, batch_size, needed_logical_blocks), 
            dtype=torch.int32, device=device
        )
        
        physical_ids = torch.arange(0, total_blocks_needed, dtype=torch.int32, device=device)
        physical_ids = physical_ids.view(num_layers, batch_size, needed_logical_blocks)
        self.block_tables.copy_(physical_ids)

        # 3. Slot Mapping
        target_token_idx = seq_len 
        logical_block_idx = target_token_idx // block_size
        block_offset = target_token_idx % block_size
        
        target_physical_blocks = self.block_tables[:, :, logical_block_idx]
        self.slot_mapping = (target_physical_blocks.long() * block_size) + block_offset

        # 4. Scratchpads
        self._partition_size = 512
        max_num_partitions = ((seq_len + 1) + self._partition_size - 1) // self._partition_size
        
        self.exp_sums = torch.empty(
            (batch_size, num_heads, max_num_partitions),
            dtype=torch.float32, device=device
        )
        self.max_logits = torch.empty_like(self.exp_sums)
        self.tmp_output = torch.empty(
            (batch_size, num_heads, max_num_partitions, head_dim),
            dtype=dtype, device=device
        )

    def get_layer_data(self, layer_idx: int):
        return (self.block_tables[layer_idx], self.slot_mapping[layer_idx])


class PagedTransformer(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.n_layers = model_args.n_layers
        self.head_dim = self.dim // self.n_heads
        
        self.layers = nn.ModuleList([
            PagedTransformerBlock(model_args) for _ in range(self.n_layers)
        ])
        self.norm = nn.RMSNorm(self.dim, eps=model_args.norm_eps)
        
        self.scale = float(1.0 / (self.head_dim ** 0.5))
        self.register_buffer("k_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("v_scale", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor, data: PagedTransformerData):
        # Calculate max_seq_len as a Python int to avoid graph breaks
        # We know seq_len from init, plus 1 for the new token.
        max_seq_len_scalar = data.seq_len + 1
        
        for i, layer in enumerate(self.layers):
            block_table, slot_mapping = data.get_layer_data(i)
            
            x = layer(
                x, 
                data.key_heap, 
                data.val_heap, 
                block_table, 
                slot_mapping,
                data.context_lens,
                data.exp_sums,
                data.max_logits,
                data.tmp_output,
                self.scale,
                self.k_scale,
                self.v_scale,
                max_seq_len_scalar # Pass the scalar explicitly
            )
            
        return self.norm(x)

class PagedTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.norm1 = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.norm2 = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.w1 = nn.Linear(args.dim, 4 * args.dim, bias=False)
        self.w2 = nn.Linear(4 * args.dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, 4 * args.dim, bias=False)

    def forward(
        self, 
        x, 
        key_heap, 
        val_heap, 
        block_table,
        slot_mapping,
        context_lens,
        exp_sums,
        max_logits,
        tmp_output,
        scale,
        k_scale,
        v_scale,
        max_seq_len: int # New Argument
    ):
        residual = x
        x = self.norm1(x)
        
        # 1. Projections
        # x is [Batch, 1, Dim]
        q = self.wq(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        k = self.wk(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        v = self.wv(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        
        # 2. WRITE PHASE
        k_flat = k.view(-1, self.n_heads, self.head_dim)
        v_flat = v.view(-1, self.n_heads, self.head_dim)
        
        ops.reshape_and_cache(
            k_flat, v_flat, key_heap, val_heap, slot_mapping,
            "auto", k_scale, v_scale
        )
        
        # 3. READ PHASE
        # Reshape Query to 3D [Batch, Heads, Head Dim]
        # vLLM expects [num_seqs, num_heads, head_size]
        q_for_attn = q.transpose(1, 2).squeeze(2).contiguous()
        
        attn_out = torch.empty_like(q_for_attn)
        
        ops.paged_attention_v2(
            attn_out,
            exp_sums,
            max_logits,
            tmp_output,
            q_for_attn, # Passed correctly as 3D tensor
            key_heap,
            val_heap,
            self.n_heads,
            scale,
            block_table,
            context_lens,
            key_heap.shape[-2],
            max_seq_len, # Use the python scalar
            None, "auto", k_scale, v_scale,
            0, 0, 0, 0, 0
        )
        
        # Reshape output back to [Batch, 1, Dim]
        attn_out = attn_out.unsqueeze(2).transpose(1, 2).reshape(x.shape[0], 1, -1)
        x = self.wo(attn_out) + residual
        
        # 4. MLP
        residual = x
        x = self.norm2(x)
        x = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        x = x + residual
        
        return x