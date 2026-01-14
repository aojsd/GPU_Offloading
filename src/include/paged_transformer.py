import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from typing import List, Optional

class PagedTransformerData:
    """
    Manages the Key/Value cache for arbitrary batch sizes and sequence lengths.
    Supports both contiguous and randomized physical block layout strategies.
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
        allocation_mode: str = "contiguous"  # "contiguous" or "random"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        
        # Batch Metadata
        self.batch_size = len(sequence_lengths)
        self.sequence_lengths = sequence_lengths
        self.max_seq_len = max(sequence_lengths)
        
        # vLLM optimization constant (16 bytes / element_size)
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.x = 16 // element_size
        
        # ------------------------------------------------------------------
        # 1. Allocate the Giant KV Heaps (Monolithic)
        # ------------------------------------------------------------------
        self.key_heap = torch.zeros(
            (max_num_blocks, num_heads, head_dim // self.x, block_size, self.x),
            dtype=dtype, device=device
        )
        self.val_heap = torch.zeros(
            (max_num_blocks, num_heads, head_dim, block_size),
            dtype=dtype, device=device
        )
        
        # ------------------------------------------------------------------
        # 2. Allocate & Setup Block Tables
        # ------------------------------------------------------------------
        # Calculate blocks needed per sequence (history + 1 new token)
        needed_logical_blocks = [(l + 1 + block_size - 1) // block_size for l in sequence_lengths]
        max_logical_blocks = max(needed_logical_blocks)
        total_blocks_needed = num_layers * sum(needed_logical_blocks)
        
        if total_blocks_needed > max_num_blocks:
            raise RuntimeError(f"OOM: Heap size {max_num_blocks} too small. Needed {total_blocks_needed}.")
        
        # Context Lens: Each sequence length + 1
        self.context_lens = torch.tensor(
            [l + 1 for l in sequence_lengths], dtype=torch.int32, device=device
        )
        
        # Block Tables: [Num_Layers, Batch, Max_Blocks]
        self.block_tables = torch.zeros(
            (num_layers, self.batch_size, max_logical_blocks), 
            dtype=torch.int32, device=device
        )
        
        # --- Allocation Strategy ---
        if allocation_mode == "contiguous":
            # Linear assignment: [Seq0_L0, Seq0_L1... Seq1_L0...]
            # This keeps blocks for a single sequence/layer contiguous in memory.
            current_id = 0
            for b in range(self.batch_size):
                num_blocks = needed_logical_blocks[b]
                for l in range(num_layers):
                    ids = torch.arange(current_id, current_id + num_blocks, device=device, dtype=torch.int32)
                    self.block_tables[l, b, :num_blocks] = ids
                    current_id += num_blocks
                    
        elif allocation_mode == "random":
            # Random assignment: Permute available heap blocks and slice.
            # This simulates extreme fragmentation.
            available_ids = torch.randperm(max_num_blocks, device=device, dtype=torch.int32)
            current_idx = 0
            
            for b in range(self.batch_size):
                num_blocks = needed_logical_blocks[b]
                for l in range(num_layers):
                    # Take random blocks from the pool
                    ids = available_ids[current_idx : current_idx + num_blocks]
                    self.block_tables[l, b, :num_blocks] = ids
                    current_idx += num_blocks
        else:
            raise ValueError(f"Unknown allocation_mode: {allocation_mode}")

        # ------------------------------------------------------------------
        # 3. Vectorized Slot Mapping Calculation
        # ------------------------------------------------------------------
        # Where does the NEW token (at index sequence_lengths[b]) go?
        # Target Index = sequence_lengths[b]
        
        # [Batch]
        target_indices = torch.tensor(sequence_lengths, device=device, dtype=torch.long)
        logical_block_indices = target_indices // block_size
        block_offsets = target_indices % block_size
        
        # We need to look up physical blocks for these logical indices across all layers.
        # self.block_tables: [Layers, Batch, Max_Logical_Blocks]
        # logical_block_indices: [Batch] -> Expand to [Layers, Batch, 1] for gather
        
        # 1. Expand indices for gather
        gather_indices = logical_block_indices.view(1, self.batch_size, 1).expand(num_layers, -1, -1)
        
        # 2. Gather Physical Block IDs: [Layers, Batch, 1]
        # block_tables is int32, must be converted to long for calculation or keep int32
        physical_blocks = torch.gather(self.block_tables.long(), 2, gather_indices).squeeze(2)
        
        # 3. Calculate linear slot index: physical_block * block_size + offset
        # Result: [Layers, Batch]
        self.slot_mapping = (physical_blocks * block_size) + block_offsets.view(1, self.batch_size)
        
        # Ensure int32/long consistency for kernel
        self.slot_mapping = self.slot_mapping.long()

        # ------------------------------------------------------------------
        # 4. Initialize v2 Scratchpads
        # ------------------------------------------------------------------
        # Scratchpads must be sized for the longest sequence in the batch
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
        """Returns tuple of (block_table, slot_mapping) for a specific layer."""
        return (
            self.block_tables[layer_idx], 
            self.slot_mapping[layer_idx]
        )


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
        """
        x: Input tokens [Batch, 1, Dim]
        data: PagedTransformerData object
        """
        # FIX: Calculate max_seq_len scalar from batch max
        max_seq_len_scalar = data.max_seq_len + 1
        
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
                max_seq_len_scalar
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
        max_seq_len: int
    ):
        residual = x
        x = self.norm1(x)
        
        # 1. Projections
        q = self.wq(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        k = self.wk(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        v = self.wv(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        
        # 2. WRITE PHASE (reshape_and_cache)
        ops.reshape_and_cache(
            k.view(-1, self.n_heads, self.head_dim),
            v.view(-1, self.n_heads, self.head_dim),
            key_heap, val_heap, slot_mapping,
            "auto", k_scale, v_scale
        )
        
        # 3. READ PHASE (paged_attention_v2)
        q_for_attn = q.transpose(1, 2).squeeze(2).contiguous()
        attn_out = torch.empty_like(q_for_attn)
        
        ops.paged_attention_v2(
            attn_out,
            exp_sums,
            max_logits,
            tmp_output,
            q_for_attn, 
            key_heap,
            val_heap,
            self.n_heads,
            scale,
            block_table,
            context_lens,
            key_heap.shape[-2],
            max_seq_len,
            None, "auto", k_scale, v_scale,
            0, 0, 0, 0, 0
        )
        
        attn_out = attn_out.unsqueeze(2).transpose(1, 2).reshape(x.shape[0], 1, -1)
        x = self.wo(attn_out) + residual
        
        # 4. MLP
        residual = x
        x = self.norm2(x)
        x = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        x = x + residual
        
        return x