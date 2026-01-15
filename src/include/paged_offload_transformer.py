import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from typing import List
from .transformer_common import compile_if_needed
from .paged_transformer import PagedTransformer, PagedTransformerBlock

# ==========================================
# DATA CLASS
# ==========================================

class PagedOffloadTransformerData:
    """
    Manages a Paged KV Cache with CPU Offloading.
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

        # 1. Sizes
        total_tokens_per_seq = [l + 1 for l in sequence_lengths]
        total_blocks_per_seq = [(t + block_size - 1) // block_size for t in total_tokens_per_seq]
        
        self.offload_counts = []  
        self.resident_counts = [] 
        
        for tb in total_blocks_per_seq:
            n_off = int(tb * kv_offload_ratio)
            n_res = tb - n_off
            self.offload_counts.append(n_off)
            self.resident_counts.append(n_res)
            
        self.total_offload_blocks_batch = sum(self.offload_counts)
        self.total_resident_blocks_L1N = sum(self.resident_counts) * (num_layers - 1)
        self.total_resident_blocks_L0 = sum(total_blocks_per_seq) 
        
        # 2. Heap
        total_resident_pool_size = self.total_resident_blocks_L0 + self.total_resident_blocks_L1N
        heap_blocks_needed = total_resident_pool_size + 2 * self.total_offload_blocks_batch + 128
        
        self.key_heap = torch.zeros(
            (heap_blocks_needed, num_heads, head_dim // self.x, block_size, self.x),
            dtype=dtype, device=device
        )
        self.val_heap = torch.zeros(
            (heap_blocks_needed, num_heads, head_dim, block_size),
            dtype=dtype, device=device
        )
        
        # 3. CPU Buffers
        self.k_offload_cpu = []
        self.v_offload_cpu = []
        
        # Ensure buffers exist even if size is 0
        off_size = self.total_offload_blocks_batch
        for _ in range(num_layers - 1):
            k_buf = torch.randn(
                (off_size, num_heads, head_dim // self.x, block_size, self.x),
                dtype=dtype, device='cpu'
            ).pin_memory()
            v_buf = torch.randn(
                (off_size, num_heads, head_dim, block_size),
                dtype=dtype, device='cpu'
            ).pin_memory()
            self.k_offload_cpu.append(k_buf)
            self.v_offload_cpu.append(v_buf)

        # 4. Block Tables
        max_logical_blocks = max(total_blocks_per_seq)
        self.context_lens = torch.tensor(total_tokens_per_seq, dtype=torch.int32, device=device)
        self.block_tables = torch.zeros(
            (num_layers, self.batch_size, max_logical_blocks), 
            dtype=torch.int32, device=device
        )
        
        sp0_start = heap_blocks_needed - 2 * off_size
        sp0_end   = heap_blocks_needed - off_size
        sp1_start = heap_blocks_needed - off_size
        sp1_end   = heap_blocks_needed
        
        self.scratchpad_ranges = [(sp0_start, sp0_end), (sp1_start, sp1_end)]
        
        if allocation_mode == "random":
            resident_pool_ids = torch.randperm(total_resident_pool_size, device=device, dtype=torch.int32)
        else:
            resident_pool_ids = torch.arange(total_resident_pool_size, device=device, dtype=torch.int32)
            
        pool_ptr = 0
        sp_offsets = [0]
        curr = 0
        for c in self.offload_counts:
            curr += c
            sp_offsets.append(curr)
            
        for b in range(self.batch_size):
            n_total = total_blocks_per_seq[b]
            n_off   = self.offload_counts[b]
            n_res   = self.resident_counts[b]
            sp_seq_offset = sp_offsets[b]
            
            # Layer 0
            ids_l0 = resident_pool_ids[pool_ptr : pool_ptr + n_total]
            pool_ptr += n_total
            self.block_tables[0, b, :n_total] = ids_l0
            
            # Layers 1..N
            for l_idx in range(1, num_layers):
                sp_set_idx = l_idx % 2
                sp_base = self.scratchpad_ranges[sp_set_idx][0]
                seq_sp_start = sp_base + sp_seq_offset
                sp_ids = torch.arange(seq_sp_start, seq_sp_start + n_off, device=device, dtype=torch.int32)
                
                res_ids = resident_pool_ids[pool_ptr : pool_ptr + n_res]
                pool_ptr += n_res
                
                full_row = torch.cat([sp_ids, res_ids])
                self.block_tables[l_idx, b, :n_total] = full_row

        # 5. Slot Mapping
        target_indices = torch.tensor(sequence_lengths, device=device, dtype=torch.long)
        logical_block_indices = target_indices // block_size
        block_offsets = target_indices % block_size
        
        gather_indices = logical_block_indices.view(1, self.batch_size, 1).expand(num_layers, -1, -1)
        physical_blocks = torch.gather(self.block_tables.long(), 2, gather_indices).squeeze(2)
        
        self.slot_mapping = (physical_blocks * block_size) + block_offsets.view(1, self.batch_size)
        self.slot_mapping = self.slot_mapping.long()

        # 6. v2 Scratchpads
        self._partition_size = 512
        max_total_len = self.max_seq_len + 1
        max_num_partitions = (max_total_len + self._partition_size - 1) // self._partition_size
        
        self.exp_sums = torch.empty((self.batch_size, num_heads, max_num_partitions), dtype=torch.float32, device=device)
        self.max_logits = torch.empty_like(self.exp_sums)
        self.tmp_output = torch.empty((self.batch_size, num_heads, max_num_partitions, head_dim), dtype=dtype, device=device)

    def get_layer_data(self, layer_idx: int):
        return (self.block_tables[layer_idx], self.slot_mapping[layer_idx])


# ==========================================
# POLYMORPHIC BLOCK
# ==========================================

class PagedOffloadTransformerBlock(PagedTransformerBlock):
    """
    Inherits from PagedTransformerBlock.
    Overrides forward_compute to inject synchronization events.
    """
    def forward_offload(
        self, x, key_heap, val_heap, block_table, slot_mapping,
        context_lens, exp_sums, max_logits, tmp_output,
        scale, k_scale, v_scale, max_seq_len,
        attn_data_ready_event, attn_finish_event
    ):
        # 1. Write (Compute Bound - overlaps with I/O)
        residual, q = self._op_qkv_write(x, key_heap, val_heap, slot_mapping, k_scale, v_scale)
        
        # --- SYNC: Wait for I/O Data to arrive ---
        torch.cuda.current_stream().wait_event(attn_data_ready_event)
        
        # 2. Attn (Compute Bound - overlaps with Next Layer I/O)
        attn_out = self._op_attn(
            q, key_heap, val_heap, block_table, context_lens,
            exp_sums, max_logits, tmp_output, scale, k_scale, v_scale, max_seq_len
        )

        # --- SYNC: Signal that this Scratchpad is now free ---
        torch.cuda.current_stream().record_event(attn_finish_event)
        
        # 3. MLP (Compute Bound)
        x = self._op_mlp(attn_out, residual, x.shape[0])
        return x


# ==========================================
# TRANSFORMER CLASS
# ==========================================

class PagedOffloadTransformer(PagedTransformer):
    def __init__(self, model_args, data=None, compile_mode=None):
        super().__init__(model_args)
        
        # Replace Layers 1..N with PagedOffloadTransformerBlocks
        for i in range(1, self.n_layers):
            self.layers[i] = PagedOffloadTransformerBlock(model_args)

        # I/O Async Infrastructure
        self.IO_stream = torch.cuda.Stream()
        self.attn_data_ready_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.n_layers - 1)]
        self.attn_finish_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.n_layers - 1)]
        
        # ------------------------------------------------------------------
        # Two-Path Compilation Strategy
        # ------------------------------------------------------------------
        
        # Path A: Resident (Standard)
        # We compile the wrapper that calls super().forward()
        # This traces the standard loop without any I/O events.
        self.forward_resident_impl = compile_if_needed(self._forward_resident_wrapper, compile_mode)
        
        # Path B: Offloaded
        # We compile the custom loop that includes event handling.
        self.forward_offload_impl = compile_if_needed(self._forward_offload_loop, compile_mode)

    def forward_IO(self, data: PagedOffloadTransformerData):
        if data.total_offload_blocks_batch == 0:
            return

        with torch.cuda.stream(self.IO_stream):
            def load_data(layer_idx):
                sp_idx = layer_idx % 2
                sp_start, sp_end = data.scratchpad_ranges[sp_idx]
                
                k_src = data.k_offload_cpu[layer_idx - 1]
                v_src = data.v_offload_cpu[layer_idx - 1]
                
                data.key_heap[sp_start:sp_end].copy_(k_src, non_blocking=True)
                data.val_heap[sp_start:sp_end].copy_(v_src, non_blocking=True)
                
                self.IO_stream.record_event(self.attn_data_ready_events[layer_idx - 1])
            
            # First two transfers can happen immediately
            load_data(1)
            load_data(2)
            for layer_idx in range(3, self.n_layers):
                self.IO_stream.wait_event(self.attn_finish_events[layer_idx - 3])
                load_data(layer_idx)

    def _forward_resident_wrapper(self, x, data):
        """Wrapper to allow compiling super().forward"""
        return super().forward(x, data)

    def _forward_offload_loop(self, x, data, max_seq_len_scalar):
        """
        Specialized loop for Offloading.
        Uses forward() for Layer 0, forward_offload() for Layers 1..N.
        """
        # --- Layer 0 (Resident) ---
        block_table_0, slot_mapping_0 = data.get_layer_data(0)
        
        # Use base forward() - no events needed
        x = self.layers[0](
            x, data.key_heap, data.val_heap, block_table_0, slot_mapping_0,
            data.context_lens, data.exp_sums, data.max_logits, data.tmp_output,
            self.scale, self.k_scale, self.v_scale, max_seq_len_scalar
        )

        # --- Layers 1..N (Offloaded) ---
        for i in range(1, self.n_layers):
            block_table, slot_mapping = data.get_layer_data(i)
            
            # Use forward_offload() with events
            x = self.layers[i].forward_offload(
                x, data.key_heap, data.val_heap, block_table, slot_mapping,
                data.context_lens, data.exp_sums, data.max_logits, data.tmp_output,
                self.scale, self.k_scale, self.v_scale, max_seq_len_scalar,
                attn_data_ready_event=self.attn_data_ready_events[i-1],
                attn_finish_event=self.attn_finish_events[i-1]
            )
        
        return self.norm(x)

    def forward(self, x: torch.Tensor, data: PagedOffloadTransformerData):
        # 1. Start I/O
        self.forward_IO(data)
        
        # 2. Dispatch
        if data.total_offload_blocks_batch == 0:
            # Use the clean, event-free loop
            return self.forward_resident_impl(x, data)
        else:
            # Use the specialized offload loop
            max_seq_len_scalar = data.max_seq_len + 1
            return self.forward_offload_impl(x, data, max_seq_len_scalar)