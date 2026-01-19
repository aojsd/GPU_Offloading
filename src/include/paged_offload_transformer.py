import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from typing import List, Optional
from .transformer_common import compile_if_needed
from .paged_transformer import PagedTransformer, PagedTransformerBlock

# ==========================================
# DATA CLASS
# ==========================================

class PagedOffloadTransformerData:
    """
    Manages a Paged KV Cache with CPU Offloading for Mixed Batches.
    
    Scratchpad Layout (Per Layer):
    [ Decode_Read_Buffer (History) | Prefill_Write_Buffer (Eviction) ]
    """
    def __init__(
        self, 
        decode_lengths: List[int],
        prefill_lengths: List[int],
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
        
        self.decode_lengths = decode_lengths if decode_lengths is not None else []
        self.prefill_lengths = prefill_lengths if prefill_lengths is not None else []
        
        self.num_prefills = len(self.prefill_lengths)
        self.num_decodes = len(self.decode_lengths)
        self.batch_size = self.num_prefills + self.num_decodes
        
        self.num_prefill_tokens = sum(self.prefill_lengths)
        self.num_decode_tokens = self.num_decodes
        self.total_tokens = self.num_prefill_tokens + self.num_decode_tokens
        
        max_prefill = max(self.prefill_lengths) if self.prefill_lengths else 0
        max_decode = max(self.decode_lengths) if self.decode_lengths else 0
        self.max_seq_len = max(max_prefill, max_decode)
        
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.x = 16 // element_size

        # ------------------------------------------------------------------
        # 1. Calculate Sizes & Offload Splits
        # ------------------------------------------------------------------
        # A. Prefill (Total Length)
        prefill_blocks_per_seq = [(l + block_size - 1) // block_size for l in self.prefill_lengths]
        # B. Decode (History + 1)
        decode_blocks_per_seq = [(l + 1 + block_size - 1) // block_size for l in self.decode_lengths]
        
        all_blocks_per_seq = prefill_blocks_per_seq + decode_blocks_per_seq
        
        self.offload_counts = []
        self.resident_counts = []
        
        self.total_prefill_offload_blocks = 0
        self.total_decode_offload_blocks = 0
        
        for i, tb in enumerate(all_blocks_per_seq):
            n_off = int(tb * kv_offload_ratio)
            n_res = tb - n_off
            self.offload_counts.append(n_off)
            self.resident_counts.append(n_res)
            
            if i < self.num_prefills:
                self.total_prefill_offload_blocks += n_off
            else:
                self.total_decode_offload_blocks += n_off
                
        self.total_offload_blocks_batch = self.total_prefill_offload_blocks + self.total_decode_offload_blocks
        self.total_resident_blocks_L1N = sum(self.resident_counts) * (num_layers - 1)
        self.total_resident_blocks_L0 = sum(all_blocks_per_seq)
        
        # ------------------------------------------------------------------
        # 2. Heap Allocation (GPU)
        # ------------------------------------------------------------------
        total_resident_pool_size = self.total_resident_blocks_L0 + self.total_resident_blocks_L1N
        # Heap must hold resident blocks + 2 sets of scratchpads
        heap_blocks_needed = total_resident_pool_size + 2 * self.total_offload_blocks_batch + 128
        
        self.key_heap = torch.zeros(
            (heap_blocks_needed, num_heads, head_dim // self.x, block_size, self.x),
            dtype=dtype, device=device
        )
        self.val_heap = torch.zeros(
            (heap_blocks_needed, num_heads, head_dim, block_size),
            dtype=dtype, device=device
        )
        
        # ------------------------------------------------------------------
        # 3. CPU Buffers (Pinned)
        # ------------------------------------------------------------------
        self.k_offload_cpu = [] # For Decode H2D
        self.v_offload_cpu = []
        self.k_prefill_cpu = [] # For Prefill D2H
        self.v_prefill_cpu = []
        
        dec_off_size = self.total_decode_offload_blocks
        pre_off_size = self.total_prefill_offload_blocks
        
        for _ in range(num_layers - 1):
            # Decode buffers (Read from CPU)
            if dec_off_size > 0:
                self.k_offload_cpu.append(torch.randn((dec_off_size, num_heads, head_dim // self.x, block_size, self.x), dtype=dtype, device='cpu').pin_memory())
                self.v_offload_cpu.append(torch.randn((dec_off_size, num_heads, head_dim, block_size), dtype=dtype, device='cpu').pin_memory())
            else:
                self.k_offload_cpu.append(torch.empty(0, dtype=dtype, device='cpu').pin_memory())
                self.v_offload_cpu.append(torch.empty(0, dtype=dtype, device='cpu').pin_memory())
            
            # Prefill buffers (Write to CPU)
            if pre_off_size > 0:
                self.k_prefill_cpu.append(torch.empty((pre_off_size, num_heads, head_dim // self.x, block_size, self.x), dtype=dtype, device='cpu').pin_memory())
                self.v_prefill_cpu.append(torch.empty((pre_off_size, num_heads, head_dim, block_size), dtype=dtype, device='cpu').pin_memory())
            else:
                self.k_prefill_cpu.append(torch.empty(0, dtype=dtype, device='cpu').pin_memory())
                self.v_prefill_cpu.append(torch.empty(0, dtype=dtype, device='cpu').pin_memory())

        # ------------------------------------------------------------------
        # 4. Block Tables (Mapping)
        # ------------------------------------------------------------------
        max_logical_blocks = max(all_blocks_per_seq) if all_blocks_per_seq else 0
        
        dec_lens = [l + 1 for l in self.decode_lengths]
        self.decode_context_lens = torch.tensor(dec_lens, dtype=torch.int32, device=device)
        
        self.block_tables = torch.zeros(
            (num_layers, self.batch_size, max_logical_blocks), 
            dtype=torch.int32, device=device
        )
        
        # Identify Scratchpad Regions [Decode | Prefill]
        off_size = self.total_offload_blocks_batch
        
        sp0_start = heap_blocks_needed - 2 * off_size
        sp0_end   = heap_blocks_needed - off_size
        sp1_start = heap_blocks_needed - off_size
        sp1_end   = heap_blocks_needed
        
        self.scratchpad_ranges = [(sp0_start, sp0_end), (sp1_start, sp1_end)]
        
        # Offsets relative to scratchpad start
        decode_base_offset = 0
        prefill_base_offset = self.total_decode_offload_blocks
        
        sp_seq_offsets = []
        curr_dec = decode_base_offset
        curr_pre = prefill_base_offset
        
        for i in range(self.batch_size):
            count = self.offload_counts[i]
            if i < self.num_prefills:
                sp_seq_offsets.append(curr_pre)
                curr_pre += count
            else:
                sp_seq_offsets.append(curr_dec)
                curr_dec += count
                
        # Resident Allocation
        if allocation_mode == "random":
            resident_pool_ids = torch.randperm(total_resident_pool_size, device=device, dtype=torch.int32)
        else:
            resident_pool_ids = torch.arange(total_resident_pool_size, device=device, dtype=torch.int32)
            
        pool_ptr = 0
        
        for b in range(self.batch_size):
            n_total = all_blocks_per_seq[b]
            n_off   = self.offload_counts[b]
            n_res   = self.resident_counts[b]
            sp_offset = sp_seq_offsets[b]
            
            # Layer 0
            ids_l0 = resident_pool_ids[pool_ptr : pool_ptr + n_total]
            pool_ptr += n_total
            self.block_tables[0, b, :n_total] = ids_l0
            
            # Layers 1..N
            for l_idx in range(1, num_layers):
                sp_set_idx = l_idx % 2
                sp_base = self.scratchpad_ranges[sp_set_idx][0]
                
                # Offload Part (Scratchpad)
                seq_sp_start = sp_base + sp_offset
                sp_ids = torch.arange(seq_sp_start, seq_sp_start + n_off, device=device, dtype=torch.int32)
                
                # Resident Part (Heap)
                res_ids = resident_pool_ids[pool_ptr : pool_ptr + n_res]
                pool_ptr += n_res
                
                full_row = torch.cat([sp_ids, res_ids])
                self.block_tables[l_idx, b, :n_total] = full_row

        # ------------------------------------------------------------------
        # 5. Slot Mapping
        # ------------------------------------------------------------------
        # block_tables logic ensures:
        # [0..N_off] -> Scratchpad (Prefill writes here, Decode reads here)
        # [N_off..End] -> Heap (Resident)
        
        flat_batch_indices = []
        flat_logical_indices = []
        
        # A. Prefills (All tokens)
        for b_idx in range(self.num_prefills):
            l = self.prefill_lengths[b_idx]
            flat_logical_indices.append(torch.arange(l, device=device, dtype=torch.long))
            flat_batch_indices.append(torch.full((l,), b_idx, device=device, dtype=torch.long))
            
        # B. Decodes (Only new token)
        for i in range(self.num_decodes):
            b_idx = self.num_prefills + i
            hist_len = self.decode_lengths[i]
            flat_logical_indices.append(torch.tensor([hist_len], device=device, dtype=torch.long))
            flat_batch_indices.append(torch.tensor([b_idx], device=device, dtype=torch.long))
            
        if self.total_tokens > 0:
            flat_logical_indices = torch.cat(flat_logical_indices)
            flat_batch_indices = torch.cat(flat_batch_indices)
            logical_blk = flat_logical_indices // block_size
            blk_offset = flat_logical_indices % block_size
            
            self.slot_mapping = torch.zeros((num_layers, self.total_tokens), dtype=torch.long, device=device)
            for l in range(num_layers):
                phys_ids = self.block_tables[l][flat_batch_indices, logical_blk]
                self.slot_mapping[l] = (phys_ids.long() * block_size) + blk_offset
        else:
            self.slot_mapping = torch.empty((num_layers, 0), dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # 6. Metadata
        # ------------------------------------------------------------------
        if self.num_prefills > 0:
            cu_seqlens = [0]
            cwd = 0
            for l in self.prefill_lengths:
                cwd += l
                cu_seqlens.append(cwd)
            self.prefill_cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
            self.prefill_max_seqlen = max(self.prefill_lengths)
        else:
            self.prefill_cu_seqlens = None
            self.prefill_max_seqlen = 0

        self._partition_size = 512
        max_num_partitions = (self.max_seq_len + self._partition_size - 1) // self._partition_size
        
        if self.num_decodes > 0:
            self.exp_sums = torch.empty((self.num_decodes, num_heads, max_num_partitions), dtype=torch.float32, device=device)
            self.max_logits = torch.empty_like(self.exp_sums)
            self.tmp_output = torch.empty((self.num_decodes, num_heads, max_num_partitions, head_dim), dtype=dtype, device=device)
        else:
            self.exp_sums = None
            self.max_logits = None
            self.tmp_output = None

    def get_layer_data(self, layer_idx: int):
        return (self.block_tables[layer_idx], self.slot_mapping[layer_idx])
    
    def get_decode_block_table(self, layer_idx: int):
        return self.block_tables[layer_idx, self.num_prefills : ]


# ==========================================
# POLYMORPHIC BLOCK
# ==========================================

class PagedOffloadTransformerBlock(PagedTransformerBlock):
    """
    Inherits from PagedTransformerBlock.
    Overrides forward_offload to inject synchronization events.
    """
    def forward_offload(
        self, x, key_heap, val_heap, block_table, slot_mapping,
        # Decode
        decode_block_table, decode_context_lens, 
        exp_sums, max_logits, tmp_output, decode_max_seq_len,
        # Prefill
        num_prefill_tokens, prefill_cu_seqlens, prefill_max_seqlen,
        # Sync
        h2d_ready_event, prefill_ready_event, compute_done_event,
        # Const
        scale, k_scale, v_scale
    ):
        # 1. Write (Compute Bound - overlaps with H2D and D2H)
        # Writes all tokens.
        # Prefill offload parts -> Scratchpad[Prefill]
        # Decode -> Heap
        residual, q, k, v = self._op_qkv_write(x, key_heap, val_heap, slot_mapping, k_scale, v_scale)
        
        # 2. Split Attention
        attn_outputs = []
        
        # A. Prefill (Compute Bound)
        # Use LOCAL data just written. Run BEFORE waiting for H2D.
        if num_prefill_tokens > 0:
            q_p = q[:num_prefill_tokens]
            k_p = k[:num_prefill_tokens]
            v_p = v[:num_prefill_tokens]
            
            attn_p = self._op_attn_prefill(q_p, k_p, v_p, prefill_cu_seqlens, prefill_max_seqlen, scale)
            attn_outputs.append(attn_p)
            
            # SIGNAL: Prefill Compute Done. D2H can start now.
            torch.cuda.current_stream().record_event(prefill_ready_event)
            
        # --- SYNC: Wait for H2D (Decode History) ---
        # Decode needs history from CPU.
        torch.cuda.current_stream().wait_event(h2d_ready_event)
        
        # B. Decode
        if decode_context_lens is not None:
            q_d = q[num_prefill_tokens:]
            
            attn_d = self._op_attn_decode(
                q_d, key_heap, val_heap, decode_block_table, decode_context_lens,
                exp_sums, max_logits, tmp_output, scale, k_scale, v_scale, decode_max_seq_len
            )
            attn_outputs.append(attn_d)

        # SIGNAL: Compute Done. Scratchpad[Decode] is now free.
        torch.cuda.current_stream().record_event(compute_done_event)
        
        # 3. Merge & MLP
        attn_combined = torch.cat(attn_outputs, dim=0)
        x = self._op_mlp(attn_combined, residual, x.shape[0])
        return x


# ==========================================
# TRANSFORMER CLASS
# ==========================================

class PagedOffloadTransformer(PagedTransformer):
    def __init__(self, model_args, data=None, compile_mode=None):
        super().__init__(model_args)
        
        for i in range(1, self.n_layers):
            self.layers[i] = PagedOffloadTransformerBlock(model_args)

        # I/O Async Infrastructure
        self.H2D_stream = torch.cuda.Stream()
        self.D2H_stream = torch.cuda.Stream()
        
        # Event Arrays
        self.h2d_ready_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.n_layers - 1)]
        self.prefill_ready_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.n_layers - 1)]
        self.compute_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.n_layers - 1)]
        self.d2h_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.n_layers - 1)]
        
        # Compilation
        self.forward_resident_impl = compile_if_needed(self._forward_resident_wrapper, compile_mode)
        self.forward_offload = compile_if_needed(self.forward_offload_, compile_mode)

    def _forward_resident_wrapper(self, x, data):
        return super().forward(x, data)

    def load_data_H2D(self, data: PagedOffloadTransformerData, layer_idx):
        # [Decode] CPU -> GPU Scratchpad
        sp_idx = layer_idx % 2
        sp_start, sp_end = data.scratchpad_ranges[sp_idx]
        
        k_src = data.k_offload_cpu[layer_idx - 1]
        v_src = data.v_offload_cpu[layer_idx - 1]
        decode_size = data.total_decode_offload_blocks
        
        # Wait for buffer to be free
        # Needs: compute_done[i-2] (Previous usage of this scratchpad portion)
        if layer_idx >= 3:
            self.H2D_stream.wait_event(self.compute_done_events[layer_idx - 3])
            
        with torch.cuda.stream(self.H2D_stream):
            if decode_size > 0:
                data.key_heap[sp_start : sp_start + decode_size].copy_(k_src, non_blocking=True)
                data.val_heap[sp_start : sp_start + decode_size].copy_(v_src, non_blocking=True)
            self.H2D_stream.record_event(self.h2d_ready_events[layer_idx - 1])

    def evict_data_D2H(self, data: PagedOffloadTransformerData, layer_idx):
        # [Prefill] GPU Scratchpad -> CPU
        sp_idx = layer_idx % 2
        sp_start, sp_end = data.scratchpad_ranges[sp_idx]
        
        # Prefill region is AFTER decode region
        prefill_start = sp_start + data.total_decode_offload_blocks
        prefill_end = sp_end
        copy_size = prefill_end - prefill_start
        
        k_dst = data.k_prefill_cpu[layer_idx - 1]
        v_dst = data.v_prefill_cpu[layer_idx - 1]
        
        # Wait for Compute to finish writing Prefill data
        self.D2H_stream.wait_event(self.prefill_ready_events[layer_idx - 1])
        
        with torch.cuda.stream(self.D2H_stream):
            if copy_size > 0:
                k_src = data.key_heap[prefill_start:prefill_end]
                v_src = data.val_heap[prefill_start:prefill_end]
                k_dst.copy_(k_src, non_blocking=True)
                v_dst.copy_(v_src, non_blocking=True)
            self.D2H_stream.record_event(self.d2h_done_events[layer_idx - 1])

    def forward_onload_H2D(self, data: PagedOffloadTransformerData):
        if self.n_layers > 1: self.load_data_H2D(data, 1)
        if self.n_layers > 2: self.load_data_H2D(data, 2)
        for layer_idx in range(3, self.n_layers):
            self.load_data_H2D(data, layer_idx)

    def forward_offload_D2H(self, data: PagedOffloadTransformerData):
        # Starts from layer 1
        for layer_idx in range(1, self.n_layers):
            self.evict_data_D2H(data, layer_idx)

    def forward_offload_(self, x, data, max_seq_len_scalar):
        # Layer 0 (Resident)
        block_table_0, slot_mapping_0 = data.get_layer_data(0)
        x = self.layers[0](
            x, data.key_heap, data.val_heap, slot_mapping_0,
            data.get_decode_block_table(0), data.decode_context_lens, 
            data.exp_sums, data.max_logits, data.tmp_output, max_seq_len_scalar,
            data.num_prefill_tokens, data.prefill_cu_seqlens, data.prefill_max_seqlen,
            self.scale, self.k_scale, self.v_scale
        )

        # Layers 1..N (Offloaded)
        for i in range(1, self.n_layers):
            block_table, slot_mapping = data.get_layer_data(i)
            x = self.layers[i].forward_offload(
                x, data.key_heap, data.val_heap, block_table, slot_mapping,
                data.get_decode_block_table(i), data.decode_context_lens, 
                data.exp_sums, data.max_logits, data.tmp_output, max_seq_len_scalar,
                data.num_prefill_tokens, data.prefill_cu_seqlens, data.prefill_max_seqlen,
                self.h2d_ready_events[i-1], 
                self.prefill_ready_events[i-1], 
                self.compute_done_events[i-1],
                self.scale, self.k_scale, self.v_scale
            )
        return self.norm(x)

    def forward(self, x: torch.Tensor, data: PagedOffloadTransformerData):
        if data.total_offload_blocks_batch == 0:
            return self.forward_resident_impl(x, data)

        # 1. Launch H2D Pipeline (Decode)
        if data.total_decode_offload_blocks > 0:
            self.forward_onload_H2D(data)
            
        # 2. Launch D2H Pipeline (Prefill)
        # Dependent on compute events, will block on GPU until ready
        if data.total_prefill_offload_blocks > 0:
            self.forward_offload_D2H(data)
        
        # 3. Dispatch Compute
        max_seq_len_scalar = 0
        if data.num_decodes > 0:
            max_seq_len_scalar = max(data.decode_lengths) + 1
            
        return self.forward_offload(x, data, max_seq_len_scalar)