import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from typing import List

class TransformerArgs:
    def __init__(self, dim, n_heads, n_layers, norm_eps=1e-5):
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.norm_eps = norm_eps

class PagedTransformerData:
    """
    Manages the Key/Value cache for arbitrary batch sizes and sequence lengths.
    """
    def __init__(
        self, 
        sequence_lengths: List[int],
        max_num_blocks: int, 
        num_layers: int, 
        num_heads: int, 
        head_dim: int, 
        block_size: int = 16, 
        dtype: torch.dtype = torch.float16, 
        device: str = "cuda",
        allocation_mode: str = "contiguous"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        
        self.batch_size = len(sequence_lengths)
        self.sequence_lengths = sequence_lengths
        self.max_seq_len = max(sequence_lengths)
        
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.x = 16 // element_size
        
        # 1. Allocate Heaps
        self.key_heap = torch.zeros(
            (max_num_blocks, num_heads, head_dim // self.x, block_size, self.x),
            dtype=dtype, device=device
        )
        self.val_heap = torch.zeros(
            (max_num_blocks, num_heads, head_dim, block_size),
            dtype=dtype, device=device
        )
        
        # 2. Block Tables
        needed_logical_blocks = [(l + 1 + block_size - 1) // block_size for l in sequence_lengths]
        max_logical_blocks = max(needed_logical_blocks)
        total_blocks_needed = num_layers * sum(needed_logical_blocks)
        
        if total_blocks_needed > max_num_blocks:
            raise RuntimeError(f"OOM: Heap size {max_num_blocks} too small. Needed {total_blocks_needed}.")
        
        self.context_lens = torch.tensor(
            [l + 1 for l in sequence_lengths], dtype=torch.int32, device=device
        )
        
        self.block_tables = torch.zeros(
            (num_layers, self.batch_size, max_logical_blocks), 
            dtype=torch.int32, device=device
        )
        
        if allocation_mode == "contiguous":
            current_id = 0
            for b in range(self.batch_size):
                num_blocks = needed_logical_blocks[b]
                for l in range(num_layers):
                    ids = torch.arange(current_id, current_id + num_blocks, device=device, dtype=torch.int32)
                    self.block_tables[l, b, :num_blocks] = ids
                    current_id += num_blocks
        elif allocation_mode == "random":
            available_ids = torch.randperm(max_num_blocks, device=device, dtype=torch.int32)
            current_idx = 0
            for b in range(self.batch_size):
                num_blocks = needed_logical_blocks[b]
                for l in range(num_layers):
                    ids = available_ids[current_idx : current_idx + num_blocks]
                    self.block_tables[l, b, :num_blocks] = ids
                    current_idx += num_blocks

        # 3. Slot Mapping
        target_indices = torch.tensor(sequence_lengths, device=device, dtype=torch.long)
        logical_block_indices = target_indices // block_size
        block_offsets = target_indices % block_size
        
        gather_indices = logical_block_indices.view(1, self.batch_size, 1).expand(num_layers, -1, -1)
        physical_blocks = torch.gather(self.block_tables.long(), 2, gather_indices).squeeze(2)
        
        self.slot_mapping = (physical_blocks * block_size) + block_offsets.view(1, self.batch_size)
        self.slot_mapping = self.slot_mapping.long()

        # 4. Scratchpads
        self._partition_size = 512
        max_total_len = self.max_seq_len + 1
        max_num_partitions = (max_total_len + self._partition_size - 1) // self._partition_size
        
        self.exp_sums = torch.empty(
            (self.batch_size, num_heads, max_num_partitions),
            dtype=torch.float32, device=device
        )
        self.max_logits = torch.empty_like(self.exp_sums)
        self.tmp_output = torch.empty(
            (self.batch_size, num_heads, max_num_partitions, head_dim),
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
        max_seq_len_scalar = data.max_seq_len + 1
        
        for i, layer in enumerate(self.layers):
            block_table, slot_mapping = data.get_layer_data(i)
            
            x = layer(
                x, data.key_heap, data.val_heap, block_table, slot_mapping,
                data.context_lens, data.exp_sums, data.max_logits, data.tmp_output,
                self.scale, self.k_scale, self.v_scale, max_seq_len_scalar
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

    # --- Decomposed Operations (Hooks for Subclasses) ---

    def _op_qkv_write(self, x, key_heap, val_heap, slot_mapping, k_scale, v_scale):
        """Part 1: Projections & Write to Cache"""
        residual = x
        x = self.norm1(x)
        
        q = self.wq(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        k = self.wk(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        v = self.wv(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        
        ops.reshape_and_cache(
            k.view(-1, self.n_heads, self.head_dim),
            v.view(-1, self.n_heads, self.head_dim),
            key_heap, val_heap, slot_mapping,
            "auto", k_scale, v_scale
        )
        return residual, q

    def _op_attn(self, q, key_heap, val_heap, block_table, context_lens, 
                 exp_sums, max_logits, tmp_output, scale, k_scale, v_scale, max_seq_len):
        """Part 2: Paged Attention"""
        q_attn = q.transpose(1, 2).squeeze(2).contiguous()
        attn_out = torch.empty_like(q_attn)
        
        ops.paged_attention_v2(
            attn_out, exp_sums, max_logits, tmp_output, q_attn,
            key_heap, val_heap, self.n_heads, scale,
            block_table, context_lens,
            key_heap.shape[-2], max_seq_len,
            None, "auto", k_scale, v_scale,
            0, 0, 0, 0, 0
        )
        return attn_out

    def _op_mlp(self, attn_out, residual, x_shape_0):
        """Part 3: MLP"""
        attn_out = attn_out.unsqueeze(2).transpose(1, 2).reshape(x_shape_0, 1, -1)
        x = self.wo(attn_out) + residual
        
        residual = x
        x = self.norm2(x)
        x = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        x = x + residual
        return x

    # --- Main Forward (Classic) ---
    def forward(
        self, x, key_heap, val_heap, block_table, slot_mapping,
        context_lens, exp_sums, max_logits, tmp_output,
        scale, k_scale, v_scale, max_seq_len
    ):
        residual, q = self._op_qkv_write(x, key_heap, val_heap, slot_mapping, k_scale, v_scale)
        
        attn_out = self._op_attn(
            q, key_heap, val_heap, block_table, context_lens,
            exp_sums, max_logits, tmp_output, scale, k_scale, v_scale, max_seq_len
        )
        
        x = self._op_mlp(attn_out, residual, x.shape[0])
        return x