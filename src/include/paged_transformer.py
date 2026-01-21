import torch
import torch.nn as nn
from torch.nn import functional as F
from vllm import _custom_ops as ops
from typing import List, Optional
from flash_attn import flash_attn_varlen_func

class PagedTransformerData:
    """
    Manages the Key/Value cache for Mixed Batches (Prefill + Decode).
    
    Batch Layout:
    [ Prefill_Seq_0_Tokens | ... | Prefill_Seq_M_Tokens | Decode_Seq_0_Token | ... | Decode_Seq_N_Token ]
    
    The corresponding Block Tables and Metadata follow this order:
    Indices [0 ... M-1] -> Prefill Sequences
    Indices [M ... M+N-1] -> Decode Sequences
    """
    def __init__(
        self, 
        decode_lengths: List[int],          # History lengths for decode sequences (generating 1 token)
        prefill_lengths: List[int],         # Total lengths for prefill sequences (generating L tokens)
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
        
        # 1. Parse Inputs
        self.decode_lengths = decode_lengths if decode_lengths is not None else []
        self.prefill_lengths = prefill_lengths if prefill_lengths is not None else []
        
        self.num_prefills = len(self.prefill_lengths)
        self.num_decodes = len(self.decode_lengths)
        self.batch_size = self.num_prefills + self.num_decodes
        
        # Calculate Flattened Sizes
        self.num_prefill_tokens = sum(self.prefill_lengths)
        self.num_decode_tokens = self.num_decodes # 1 token per decode seq
        self.total_tokens = self.num_prefill_tokens + self.num_decode_tokens
        
        # Sequence Max (for scratchpad sizing)
        max_prefill = max(self.prefill_lengths) if self.prefill_lengths else 0
        max_decode = max(self.decode_lengths) if self.decode_lengths else 0
        self.max_seq_len = max(max_prefill, max_decode)
        
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.x = 16 // element_size
        
        # ------------------------------------------------------------------
        # 2. Block Table Allocation (Exact Size)
        # ------------------------------------------------------------------
        # Determine blocks needed per sequence
        needed_blocks = []
        # Part A: Prefills (Need blocks for `len` tokens)
        for l in self.prefill_lengths:
            needed_blocks.append((l + block_size - 1) // block_size)
        # Part B: Decodes (Need blocks for `history + 1` tokens)
        for l in self.decode_lengths:
            needed_blocks.append((l + 1 + block_size - 1) // block_size)
            
        max_logical_blocks = max(needed_blocks) if needed_blocks else 0
        
        # Calculate EXACT total blocks needed across all layers
        self.total_blocks = num_layers * sum(needed_blocks)
        
        # Allocate Table: [Num_Layers, Batch_Size, Max_Blocks]
        self.block_tables = torch.zeros(
            (num_layers, self.batch_size, max_logical_blocks), 
            dtype=torch.int32, device=device
        )
        
        # ------------------------------------------------------------------
        # 3. Allocate Heaps (Exact Size)
        # ------------------------------------------------------------------
        self.key_heap = torch.empty(
            (self.total_blocks, num_heads, head_dim // self.x, block_size, self.x),
            dtype=dtype, device=device
        )
        self.val_heap = torch.empty(
            (self.total_blocks, num_heads, head_dim, block_size),
            dtype=dtype, device=device
        )
        
        # Assign physical blocks (Linear Strategy)
        if allocation_mode == "contiguous":
            current_id = 0
            for b in range(self.batch_size):
                n_blocks = needed_blocks[b]
                for l in range(num_layers):
                    ids = torch.arange(current_id, current_id + n_blocks, device=device, dtype=torch.int32)
                    self.block_tables[l, b, :n_blocks] = ids
                    current_id += n_blocks
        elif allocation_mode == "random":
            available_ids = torch.randperm(self.total_blocks, device=device, dtype=torch.int32)
            current_idx = 0
            for b in range(self.batch_size):
                n_blocks = needed_blocks[b]
                for l in range(num_layers):
                    ids = available_ids[current_idx : current_idx + n_blocks]
                    self.block_tables[l, b, :n_blocks] = ids
                    current_idx += n_blocks

        # ------------------------------------------------------------------
        # 4. Slot Mapping Construction (Unified)
        # ------------------------------------------------------------------
        flat_batch_indices = []
        flat_logical_indices = []
        
        # Part A: Prefills
        for b_idx in range(self.num_prefills):
            l = self.prefill_lengths[b_idx]
            flat_logical_indices.append(torch.arange(l, device=device, dtype=torch.long))
            flat_batch_indices.append(torch.full((l,), b_idx, device=device, dtype=torch.long))
            
        # Part B: Decodes
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
        # 5. Attention Metadata
        # ------------------------------------------------------------------
        # A. Prefill Metadata
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

        # B. Decode Metadata
        if self.num_decodes > 0:
            dec_lens = [l + 1 for l in self.decode_lengths]
            self.decode_context_lens = torch.tensor(dec_lens, dtype=torch.int32, device=device)
        else:
            self.decode_context_lens = None

        # v2 Scratchpads
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

    def get_layer_slot_mapping(self, layer_idx: int):
        return self.slot_mapping[layer_idx]
    
    def get_decode_block_table(self, layer_idx: int):
        return self.block_tables[layer_idx, self.num_prefills : ]


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
        x: Flattened input [Total_Tokens, Dim]. 
           Contains prefill tokens first, then decode tokens.
        """
        # We need max_seq_len for the PagedAttention kernel (decode only)
        # For the kernel scalar, we use the max context len of the decode batch
        decode_max_seq_len = 0
        if data.num_decodes > 0:
            decode_max_seq_len = max(data.decode_lengths) + 1

        for i, layer in enumerate(self.layers):
            slot_mapping = data.get_layer_slot_mapping(i)
            
            # For PagedAttention (Decode), we need the sliced block table
            decode_block_table = None
            if data.num_decodes > 0:
                decode_block_table = data.get_decode_block_table(i)
            
            x = layer(
                x, 
                data.key_heap, 
                data.val_heap, 
                slot_mapping,
                # Decode Specifics
                decode_block_table,
                data.decode_context_lens,
                data.exp_sums,
                data.max_logits,
                data.tmp_output,
                decode_max_seq_len,
                # Prefill Specifics
                data.num_prefill_tokens,
                data.prefill_cu_seqlens,
                data.prefill_max_seqlen,
                # Constants
                self.scale,
                self.k_scale,
                self.v_scale
            )
            
        return self.norm(x)


class PagedTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        # NOTE: Fusing QKV into a single linear layer ensures that 
        # q, k, and v for the same head/token are adjacent in physical memory (Packed Layout).
        # Otherwise, DRAM sector reads may be inefficient due to reading from 3 disjoint buffers.
        self.w_qkv = nn.Linear(args.dim, 3 * args.n_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.norm1 = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.norm2 = nn.RMSNorm(args.dim, eps=args.norm_eps)
        
        # NOTE: Similarly, fusing the Gate (w1) and Up (w3) projections 
        # improves memory throughput during the MLP phase.
        self.w_gate_up = nn.Linear(args.dim, 2 * 4 * args.dim, bias=False)
        self.w2 = nn.Linear(4 * args.dim, args.dim, bias=False)

    def _op_qkv_write(self, x, key_heap, val_heap, slot_mapping, k_scale, v_scale):
        """Unified QKV Projection and Cache Write for ALL tokens."""
        residual = x
        x = self.norm1(x)
        
        # Project all tokens at once.
        # Layout becomes [Batch, 3, Heads, HeadDim] which interleaves Q, K, V in memory.
        qkv = self.w_qkv(x).view(x.shape[0], 3, self.n_heads, self.head_dim)
        
        # Create views for q, k, v. 
        # Note: These are slices of the packed tensor. While they have a stride on the 
        # second dimension, the last dimension (HeadDim) remains contiguous (stride 1),
        # satisfying FlashAttention requirements while maximizing L2/DRAM locality.
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]
        
        # Write all tokens to Paged Cache (Prefill + Decode) using the unified slot map
        ops.reshape_and_cache(
            k, v,
            key_heap, val_heap, slot_mapping,
            "auto", k_scale, v_scale
        )
        return residual, q, k, v

    def _op_attn_prefill(self, q, k, v, cu_seqlens, max_seqlen, scale):
        """Flash Attention for Prefill (Self-Attention on new tokens)"""
        # q, k, v are [Total_Prefill_Tokens, Heads, Dim]
        # We use PyTorch SDPA with FlashAttention backend
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=True
        )

    def _op_attn_decode(self, q, key_heap, val_heap, block_table, context_lens, 
                        exp_sums, max_logits, tmp_output, scale, k_scale, v_scale, max_seq_len):
        """Paged Attention for Decode"""
        # q is [Batch, Heads, Dim]
        
        # Ensure q is contiguous for the paged_attention kernel if the slice caused gaps.
        # This is a low-cost operation (often a no-op if allocator aligned it well), 
        # but technically required if the kernel assumes packed density.
        # Given the strided view from _op_qkv_write, this might trigger a copy, 
        # but it is necessary for correctness on some custom kernels.
        q_attn = q.contiguous() 
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

    def _op_mlp(self, attn_out, residual, total_tokens):
        """Unified MLP"""
        # attn_out: [Total_Tokens, Heads, Dim] -> Flatten to [Total_Tokens, Hidden]
        x = attn_out.reshape(total_tokens, -1)
        x = self.wo(x) + residual
        
        residual = x
        x = self.norm2(x)
        
        # Fused Gate/Up projection
        gate_up = self.w_gate_up(x)
        x_gate, x_up = gate_up.chunk(2, dim=-1)
        
        x = torch.nn.functional.silu(x_gate) * x_up
        x = self.w2(x)
        x = x + residual
        return x

    def forward(
        self, x, key_heap, val_heap, slot_mapping,
        # Decode Args
        decode_block_table, decode_context_lens, 
        exp_sums, max_logits, tmp_output, decode_max_seq_len,
        # Prefill Args
        num_prefill_tokens, prefill_cu_seqlens, prefill_max_seqlen,
        # Constants
        scale, k_scale, v_scale
    ):
        # 1. Unified QKV & Write
        residual, q, k, v = self._op_qkv_write(x, key_heap, val_heap, slot_mapping, k_scale, v_scale)
        
        # 2. Split Attention
        attn_outputs = []
        
        # A. Prefill
        if num_prefill_tokens > 0:
            q_p = q[:num_prefill_tokens]
            k_p = k[:num_prefill_tokens]
            v_p = v[:num_prefill_tokens]
            
            # Run Flash Attention (Self-Attn)
            attn_p = self._op_attn_prefill(q_p, k_p, v_p, prefill_cu_seqlens, prefill_max_seqlen, scale)
            attn_outputs.append(attn_p)
            
        # B. Decode
        if decode_context_lens is not None: # We have decodes
            q_d = q[num_prefill_tokens:]
            
            # Run Paged Attention (History-Attn)
            attn_d = self._op_attn_decode(
                q_d, key_heap, val_heap, decode_block_table, decode_context_lens,
                exp_sums, max_logits, tmp_output, scale, k_scale, v_scale, decode_max_seq_len
            )
            attn_outputs.append(attn_d)
            
        # 3. Merge
        attn_combined = torch.cat(attn_outputs, dim=0)
        
        # 4. Unified MLP
        x = self._op_mlp(attn_combined, residual, x.shape[0])
        return x