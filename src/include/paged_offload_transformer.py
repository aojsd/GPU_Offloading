import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from typing import List, Optional
from .transformer_common import compile_if_needed, PositionalEncoding

# ==========================================
# DATA CLASS
# ==========================================

class PagedOffloadTransformerData:
    """
    Manages a Paged KV Cache with CPU Offloading for Dynamic Batches.
    
    Architecture:
    1. Resident Pool: A pool of GPU blocks for 'Recent' history (L1..N) and 'Full' history (L0).
    2. Scratchpad Regions: Two reserved contiguous regions at the end of the heap. 
       Each is large enough to hold the sum of all offloaded blocks for the current batch.
       
    Allocation Strategy:
    - Layer 0: Fully Resident.
    - Layers 1..N: 
      - Logical [0..Offload]: Mapped to specific slice of Scratchpad.
      - Logical [Offload..End]: Mapped to blocks from Resident Pool.
    """
    def __init__(
        self, 
        sequence_lengths: List[int],
        num_layers: int, 
        num_heads: int, 
        head_dim: int, 
        kv_offload_ratio: float,
        block_size: int = 16, 
        dtype: torch.dtype = torch.float16, 
        device: str = "cuda",
        allocation_mode: str = "contiguous" # "contiguous" or "random"
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

        # ------------------------------------------------------------------
        # 1. Calculate Sizes Per Sequence
        # ------------------------------------------------------------------
        # Total tokens = history + 1 new token
        total_tokens_per_seq = [l + 1 for l in sequence_lengths]
        
        # Total logical blocks needed per sequence
        total_blocks_per_seq = [(t + block_size - 1) // block_size for t in total_tokens_per_seq]
        
        # Split into Offload vs Resident per sequence
        self.offload_counts = []  # Number of blocks offloaded per seq
        self.resident_counts = [] # Number of blocks resident per seq
        
        for tb in total_blocks_per_seq:
            n_off = int(tb * kv_offload_ratio)
            n_res = tb - n_off
            self.offload_counts.append(n_off)
            self.resident_counts.append(n_res)
            
        # Total sizes for the Batch
        self.total_offload_blocks_batch = sum(self.offload_counts)
        self.total_resident_blocks_L1N = sum(self.resident_counts) * (num_layers - 1)
        self.total_resident_blocks_L0 = sum(total_blocks_per_seq) # L0 is fully resident
        
        # ------------------------------------------------------------------
        # 2. Allocate GPU Heap
        # ------------------------------------------------------------------
        # We need space for:
        # - All Resident Blocks (L0 + L1..N)
        # - 2 Shared Scratchpads (Each sized to hold the batch's total offload)
        
        total_resident_pool_size = self.total_resident_blocks_L0 + self.total_resident_blocks_L1N
        heap_blocks_needed = total_resident_pool_size + 2 * self.total_offload_blocks_batch
        
        # Add a small safety buffer for alignment/padding
        heap_blocks_needed += 128
        
        self.key_heap = torch.zeros(
            (heap_blocks_needed, num_heads, head_dim // self.x, block_size, self.x),
            dtype=dtype, device=device
        )
        self.val_heap = torch.zeros(
            (heap_blocks_needed, num_heads, head_dim, block_size),
            dtype=dtype, device=device
        )
        
        # ------------------------------------------------------------------
        # 3. CPU Offload Buffers (Pinned Memory)
        # ------------------------------------------------------------------
        # Holds the "Old" blocks for Layers 1..N.
        # Allocated as one dense buffer per layer.
        # Format: [Seq0_Offload_Data, Seq1_Offload_Data, ... SeqN_Offload_Data]
        self.k_offload_cpu = []
        self.v_offload_cpu = []
        
        for _ in range(num_layers - 1):
            if self.total_offload_blocks_batch > 0:
                k_buf = torch.randn(
                    (self.total_offload_blocks_batch, num_heads, head_dim // self.x, block_size, self.x),
                    dtype=dtype, device='cpu'
                ).pin_memory()
                v_buf = torch.randn(
                    (self.total_offload_blocks_batch, num_heads, head_dim, block_size),
                    dtype=dtype, device='cpu'
                ).pin_memory()
            else:
                k_buf, v_buf = None, None
                
            self.k_offload_cpu.append(k_buf)
            self.v_offload_cpu.append(v_buf)

        # ------------------------------------------------------------------
        # 4. Construct Block Tables
        # ------------------------------------------------------------------
        max_logical_blocks = max(total_blocks_per_seq)
        
        self.context_lens = torch.tensor(total_tokens_per_seq, dtype=torch.int32, device=device)
        self.block_tables = torch.zeros(
            (num_layers, self.batch_size, max_logical_blocks), 
            dtype=torch.int32, device=device
        )
        
        # --- Identify Scratchpad Regions ---
        # The last part of the heap is reserved for scratchpads
        sp0_start = heap_blocks_needed - 2 * self.total_offload_blocks_batch
        sp0_end   = heap_blocks_needed - self.total_offload_blocks_batch
        
        sp1_start = heap_blocks_needed - self.total_offload_blocks_batch
        sp1_end   = heap_blocks_needed
        
        self.scratchpad_ranges = [(sp0_start, sp0_end), (sp1_start, sp1_end)]
        
        # --- Prepare Resident Pool ---
        # We assign resident blocks from the beginning of the heap
        if allocation_mode == "random":
            # Shuffle all available resident indices
            resident_pool_ids = torch.randperm(total_resident_pool_size, device=device, dtype=torch.int32)
        else:
            # Contiguous
            resident_pool_ids = torch.arange(total_resident_pool_size, device=device, dtype=torch.int32)
            
        pool_ptr = 0
        
        # Calculate Scratchpad Offsets per Sequence (Prefix Sum)
        # If Seq0 has 5 offload blocks, it uses SP[0..5]
        # Seq1 has 3 offload blocks, it uses SP[5..8]
        sp_offsets = [0]
        curr = 0
        for c in self.offload_counts:
            curr += c
            sp_offsets.append(curr)
            
        # Fill Tables
        for b in range(self.batch_size):
            # Info for this sequence
            n_total = total_blocks_per_seq[b]
            n_off   = self.offload_counts[b]
            n_res   = self.resident_counts[b]
            sp_seq_offset = sp_offsets[b] # Start index in the batch scratchpad
            
            # --- Layer 0: Fully Resident ---
            # Takes n_total blocks from resident pool
            ids_l0 = resident_pool_ids[pool_ptr : pool_ptr + n_total]
            pool_ptr += n_total
            self.block_tables[0, b, :n_total] = ids_l0
            
            # --- Layers 1..N: Offloaded ---
            for l_idx in range(1, num_layers):
                # 1. Scratchpad IDs (Logic: i % 2)
                sp_set_idx = l_idx % 2
                sp_base = self.scratchpad_ranges[sp_set_idx][0]
                
                # The physical IDs for this sequence's offloaded data in the scratchpad
                # Range: [Base + Seq_Offset, Base + Seq_Offset + N_Off]
                seq_sp_start = sp_base + sp_seq_offset
                sp_ids = torch.arange(seq_sp_start, seq_sp_start + n_off, device=device, dtype=torch.int32)
                
                # 2. Resident IDs (Recent history)
                # Takes n_res blocks from resident pool
                res_ids = resident_pool_ids[pool_ptr : pool_ptr + n_res]
                pool_ptr += n_res
                
                # 3. Concatenate: [Old (SP), Recent (Res)]
                # Attention sees a contiguous logical sequence, but physically split
                full_row = torch.cat([sp_ids, res_ids])
                self.block_tables[l_idx, b, :n_total] = full_row

        # ------------------------------------------------------------------
        # 5. Slot Mapping (Write Phase)
        # ------------------------------------------------------------------
        # Calculate where the NEW token (at index seq_len) goes.
        # Logic: New token is at end of seq. End of seq is in Resident part.
        # Vectorized lookup.
        
        # [Batch]
        target_indices = torch.tensor(sequence_lengths, device=device, dtype=torch.long)
        logical_block_indices = target_indices // block_size
        block_offsets = target_indices % block_size
        
        # Expand for Gather: [Layers, Batch, 1]
        gather_indices = logical_block_indices.view(1, self.batch_size, 1).expand(num_layers, -1, -1)
        
        # Lookup Physical Block ID
        physical_blocks = torch.gather(self.block_tables.long(), 2, gather_indices).squeeze(2)
        
        # Calculate Linear Address
        self.slot_mapping = (physical_blocks * block_size) + block_offsets.view(1, self.batch_size)
        self.slot_mapping = self.slot_mapping.long()

        # ------------------------------------------------------------------
        # 6. v2 Scratchpads
        # ------------------------------------------------------------------
        self._partition_size = 512
        max_total_len = self.max_seq_len + 1
        max_num_partitions = (max_total_len + self._partition_size - 1) // self._partition_size
        
        self.exp_sums = torch.empty((self.batch_size, num_heads, max_num_partitions), dtype=torch.float32, device=device)
        self.max_logits = torch.empty_like(self.exp_sums)
        self.tmp_output = torch.empty((self.batch_size, num_heads, max_num_partitions, head_dim), dtype=dtype, device=device)

    def get_layer_data(self, layer_idx: int):
        return (self.block_tables[layer_idx], self.slot_mapping[layer_idx])


# ==========================================
# TRANSFORMER CLASS
# ==========================================

class PagedOffloadTransformer(nn.Module):
    def __init__(self, model_args, compile_mode=None):
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.n_layers = model_args.n_layers
        self.head_dim = self.dim // self.n_heads
        
        self.layers = nn.ModuleList([
            PagedOffloadTransformerBlock(model_args) for _ in range(self.n_layers)
        ])
        self.norm = nn.RMSNorm(self.dim, eps=model_args.norm_eps)
        
        self.scale = float(1.0 / (self.head_dim ** 0.5))
        self.register_buffer("k_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("v_scale", torch.tensor(1.0, dtype=torch.float32))
        
        # I/O Async Infrastructure
        self.IO_stream = torch.cuda.Stream()
        self.attn_data_ready_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.n_layers - 1)]
        self.attn_finish_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.n_layers - 1)]
        
        # Compile compute
        self.forward_compute = compile_if_needed(self.forward_compute_, compile_mode)

    def forward_IO(self, data: PagedOffloadTransformerData):
        """
        Streams offloaded KV data from CPU to GPU.
        Copies the ENTIRE batch's offload buffer for a layer in one go.
        """
        # If no offloading happened (e.g. short seqs or ratio 0), skip
        if data.total_offload_blocks_batch == 0:
            return

        with torch.cuda.stream(self.IO_stream):
            for layer_idx in range(1, self.n_layers):
                if layer_idx >= 3:
                    self.IO_stream.wait_event(self.attn_finish_events[layer_idx - 3])
                
                # Identify Scratchpad Region
                sp_idx = layer_idx % 2
                sp_start, sp_end = data.scratchpad_ranges[sp_idx]
                
                # Source Data (CPU)
                k_src = data.k_offload_cpu[layer_idx - 1]
                v_src = data.v_offload_cpu[layer_idx - 1]
                
                # Bulk Copy (Non-blocking)
                # We copy [Batch_Total_Blocks] into the Scratchpad.
                # The block tables correctly map each sequence to its slice of this region.
                data.key_heap[sp_start:sp_end].copy_(k_src, non_blocking=True)
                data.val_heap[sp_start:sp_end].copy_(v_src, non_blocking=True)
                
                self.IO_stream.record_event(self.attn_data_ready_events[layer_idx - 1])

    def forward_compute_(self, x, data: PagedOffloadTransformerData, max_seq_len_scalar: int):
        
        # Layer 0
        block_table_0, slot_mapping_0 = data.get_layer_data(0)
        x = self.layers[0](
            x, data.key_heap, data.val_heap, block_table_0, slot_mapping_0,
            data.context_lens, data.exp_sums, data.max_logits, data.tmp_output,
            self.scale, self.k_scale, self.v_scale, max_seq_len_scalar
        )

        # Layers 1..N
        for i in range(1, self.n_layers):
            block_table, slot_mapping = data.get_layer_data(i)
            
            # Wait for IO if we actually have offloaded data
            if data.total_offload_blocks_batch > 0:
                torch.cuda.current_stream().wait_event(self.attn_data_ready_events[i-1])
            
            x = self.layers[i](
                x, data.key_heap, data.val_heap, block_table, slot_mapping,
                data.context_lens, data.exp_sums, data.max_logits, data.tmp_output,
                self.scale, self.k_scale, self.v_scale, max_seq_len_scalar
            )
            
            if data.total_offload_blocks_batch > 0:
                torch.cuda.current_stream().record_event(self.attn_finish_events[i-1])
        
        return self.norm(x)

    def forward(self, x: torch.Tensor, data: PagedOffloadTransformerData):
        self.forward_IO(data)
        
        # Calc max seq len for kernel
        max_seq_len_scalar = data.max_seq_len + 1
        return self.forward_compute(x, data, max_seq_len_scalar)


class PagedOffloadTransformerBlock(nn.Module):
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
        self, x, key_heap, val_heap, block_table, slot_mapping,
        context_lens, exp_sums, max_logits, tmp_output,
        scale, k_scale, v_scale, max_seq_len
    ):
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
        
        attn_out = attn_out.unsqueeze(2).transpose(1, 2).reshape(x.shape[0], 1, -1)
        x = self.wo(attn_out) + residual
        
        residual = x
        x = self.norm2(x)
        x = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        x = x + residual
        return x