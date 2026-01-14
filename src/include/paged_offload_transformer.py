import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from .transformer_common import compile_if_needed, PositionalEncoding

# ==========================================
# ARGUMENTS CLASS
# ==========================================

class TransformerArgs:
    def __init__(self, dim: int, n_heads: int, n_layers: int, norm_eps: float = 1e-5):
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.norm_eps = norm_eps

# ==========================================
# DATA CLASS
# ==========================================

class PagedOffloadTransformerData:
    """
    Manages a Paged KV Cache with CPU Offloading.
    
    Logical Layout (Seen by Attention):
    [ 0 ........ Offload Cutoff ........ Seq Len ]
    |<-  Mapped to Scratchpad ->|<- Mapped to Resident ->|
    """
    def __init__(
        self, 
        batch_size: int,
        seq_len: int,
        num_layers: int, 
        num_heads: int, 
        head_dim: int, 
        kv_offload_ratio: float,
        block_size: int = 16, 
        dtype: torch.dtype = torch.float16, 
        device: str = "cuda"
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.x = 16 // element_size
        
        # 1. Size Calculations
        total_tokens = seq_len + 1
        total_logical_blocks = (total_tokens + block_size - 1) // block_size
        
        # Split point (Block Aligned)
        self.num_offload_blocks = int(total_logical_blocks * kv_offload_ratio)
        self.num_resident_blocks = total_logical_blocks - self.num_offload_blocks
        
        # Verify sizes match
        assert (self.num_offload_blocks + self.num_resident_blocks) == total_logical_blocks
        
        # 2. Allocate GPU Heap
        # We need space for:
        # - Layer 0 (Fully Resident)
        # - Layers 1..N (Only the Resident/Recent part)
        # - 2 Shared Scratchpads (Size of the Offloaded part)
        heap_blocks_needed = (
            total_logical_blocks +                        # Layer 0
            (num_layers - 1) * self.num_resident_blocks + # Layers 1..N Resident
            2 * self.num_offload_blocks                   # Scratchpads
        )
        
        self.key_heap = torch.zeros(
            (heap_blocks_needed, num_heads, head_dim // self.x, block_size, self.x),
            dtype=dtype, device=device
        )
        self.val_heap = torch.zeros(
            (heap_blocks_needed, num_heads, head_dim, block_size),
            dtype=dtype, device=device
        )
        
        # 3. CPU Offload Buffers (Pinned Memory)
        self.k_offload_cpu = []
        self.v_offload_cpu = []
        
        for _ in range(num_layers - 1):
            k_buf = torch.randn(
                (self.num_offload_blocks, num_heads, head_dim // self.x, block_size, self.x),
                dtype=dtype, device='cpu'
            ).pin_memory()
            v_buf = torch.randn(
                (self.num_offload_blocks, num_heads, head_dim, block_size),
                dtype=dtype, device='cpu'
            ).pin_memory()
            self.k_offload_cpu.append(k_buf)
            self.v_offload_cpu.append(v_buf)

        # 4. Construct Block Tables
        self.context_lens = torch.full((batch_size,), total_tokens, dtype=torch.int32, device=device)
        self.block_tables = torch.zeros(
            (num_layers, batch_size, total_logical_blocks), 
            dtype=torch.int32, device=device
        )
        
        current_heap_idx = 0
        
        # --- Layer 0: Fully Resident ---
        # Maps Logical 0..Total -> Physical 0..Total
        l0_ids = torch.arange(current_heap_idx, current_heap_idx + total_logical_blocks, device=device)
        self.block_tables[0] = l0_ids.unsqueeze(0).expand(batch_size, -1)
        current_heap_idx += total_logical_blocks
        
        # --- Identify Scratchpad Physical IDs ---
        sp0_start = heap_blocks_needed - 2 * self.num_offload_blocks
        sp0_end   = heap_blocks_needed - self.num_offload_blocks
        sp0_ids   = torch.arange(sp0_start, sp0_end, device=device)
        
        sp1_start = heap_blocks_needed - self.num_offload_blocks
        sp1_end   = heap_blocks_needed
        sp1_ids   = torch.arange(sp1_start, sp1_end, device=device)
        
        self.scratchpad_ranges = [(sp0_start, sp0_end), (sp1_start, sp1_end)]
        self.scratchpad_block_ids = [sp0_ids, sp1_ids]
        
        # --- Layers 1..N: Offloaded ---
        # Logical Order: [Scratchpad (Old), Resident (Recent)]
        for i in range(1, num_layers):
            # Allocate Resident part for this layer
            res_ids = torch.arange(current_heap_idx, current_heap_idx + self.num_resident_blocks, device=device)
            current_heap_idx += self.num_resident_blocks
            
            # Select Scratchpad (Alternating logic)
            sp_ids = self.scratchpad_block_ids[i % 2]
            
            # Concatenate
            full_row = torch.cat([sp_ids, res_ids])
            self.block_tables[i] = full_row.unsqueeze(0).expand(batch_size, -1)

        # 5. Slot Mapping (Write Phase)
        # Target: The last logical token (seq_len)
        target_token_idx = seq_len 
        logical_block_idx = target_token_idx // block_size
        block_offset = target_token_idx % block_size
        
        # This will point to a physical block in the 'Resident' section
        target_physical_blocks = self.block_tables[:, :, logical_block_idx]
        self.slot_mapping = (target_physical_blocks.long() * block_size) + block_offset

        # 6. v2 Scratchpads
        self._partition_size = 512
        max_num_partitions = (total_tokens + self._partition_size - 1) // self._partition_size
        self.exp_sums = torch.empty((batch_size, num_heads, max_num_partitions), dtype=torch.float32, device=device)
        self.max_logits = torch.empty_like(self.exp_sums)
        self.tmp_output = torch.empty((batch_size, num_heads, max_num_partitions, head_dim), dtype=dtype, device=device)

    def get_layer_data(self, layer_idx: int):
        return (self.block_tables[layer_idx], self.slot_mapping[layer_idx])


# ==========================================
# TRANSFORMER CLASS
# ==========================================

class PagedOffloadTransformer(nn.Module):
    def __init__(self, model_args, compile_mode="default"):
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
        
        # Compile the main compute loop
        self.forward_compute = compile_if_needed(self.forward_compute_, compile_mode)

    def forward_IO(self, data: PagedOffloadTransformerData):
        """
        Streams offloaded KV data from CPU to the GPU scratchpads.
        Runs asynchronously on self.IO_stream.
        """
        with torch.cuda.stream(self.IO_stream):
            for layer_idx in range(1, self.n_layers):
                # We reuse scratchpads. L(i) uses SP(i%2).
                # L(i) shares buffer with L(i-2).
                # Wait for L(i-2) to finish.
                if layer_idx >= 3:
                    self.IO_stream.wait_event(self.attn_finish_events[layer_idx - 1 - 2])
                
                # Identify Scratchpad Region
                sp_idx = layer_idx % 2
                sp_start, sp_end = data.scratchpad_ranges[sp_idx]
                
                # Source Data (CPU)
                k_src = data.k_offload_cpu[layer_idx - 1]
                v_src = data.v_offload_cpu[layer_idx - 1]
                
                # Copy (Non-blocking) into Contiguous Heap Slice
                data.key_heap[sp_start:sp_end].copy_(k_src, non_blocking=True)
                data.val_heap[sp_start:sp_end].copy_(v_src, non_blocking=True)
                
                # Signal that data is ready for Layer(i)
                self.IO_stream.record_event(self.attn_data_ready_events[layer_idx - 1])

    def forward_compute_(self, x, data: PagedOffloadTransformerData, max_seq_len_scalar: int):
        """
        The compiled compute loop. 
        Iterates through all layers, syncing with I/O events.
        """
        
        # --- Layer 0 (Fully Resident) ---
        block_table_0, slot_mapping_0 = data.get_layer_data(0)
        x = self.layers[0](
            x, data.key_heap, data.val_heap, block_table_0, slot_mapping_0,
            data.context_lens, data.exp_sums, data.max_logits, data.tmp_output,
            self.scale, self.k_scale, self.v_scale, max_seq_len_scalar
        )

        # --- Layers 1..N (Offloaded) ---
        for i in range(1, self.n_layers):
            block_table, slot_mapping = data.get_layer_data(i)
            
            # Wait for offloaded data to arrive in Scratchpad
            # PyTorch compile often creates a graph break here, which is expected/fine
            torch.cuda.current_stream().wait_event(self.attn_data_ready_events[i-1])
            
            x = self.layers[i](
                x, data.key_heap, data.val_heap, block_table, slot_mapping,
                data.context_lens, data.exp_sums, data.max_logits, data.tmp_output,
                self.scale, self.k_scale, self.v_scale, max_seq_len_scalar
            )
            
            # Record that we are done using the Scratchpad
            torch.cuda.current_stream().record_event(self.attn_finish_events[i-1])
        
        return self.norm(x)

    def forward(self, x: torch.Tensor, data: PagedOffloadTransformerData):
        # 1. Kick off I/O
        self.forward_IO(data)
        
        # 2. Grab scalar for graph safety
        max_seq_len_scalar = data.context_lens[0].item()

        # 3. Compute
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
        
        # Projections
        q = self.wq(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        k = self.wk(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        v = self.wv(x).view(x.shape[0], 1, self.n_heads, self.head_dim)
        
        # Write Phase: Write new token to RESIDENT heap
        ops.reshape_and_cache(
            k.view(-1, self.n_heads, self.head_dim),
            v.view(-1, self.n_heads, self.head_dim),
            key_heap, val_heap, slot_mapping,
            "auto", k_scale, v_scale
        )
        
        # Read Phase: Read [Scratchpad (Old) ... Resident (Recent)]
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
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        x = x + residual
        return x