from .misc import compile_if_needed
from .transformer_common import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

# Manages all weights and KV cache of a split offloaded transformer
class SplitOffloadedTransformerData(OffloadedTransformerData):
    def __init__(self, batch_size, hidden_dim, seq_len, num_heads, num_layers, dev_gpu,
                 mm_offload_ratio, kv_offload_ratio):
        super(SplitOffloadedTransformerData, self).__init__(
            batch_size, hidden_dim, seq_len, num_heads, num_layers, dev_gpu)
        
        # Offloaded rows calculation
        self.qkv_offload_rows = int(3*hidden_dim * mm_offload_ratio)
        self.kv_offload_rows = int(seq_len * kv_offload_ratio)
        self.ffn1_offload_rows = int(4 * hidden_dim * mm_offload_ratio)
        self.ffn2_offload_rows = int(hidden_dim * mm_offload_ratio)
        self.output_offload_rows = int(hidden_dim * mm_offload_ratio)

        # Split projection weights, KV cache, and FFNs into resident and offloaded portions
        # QKV Projections
        self.qkv_resident = [self.qkv_full[i][:3 * hidden_dim - self.qkv_offload_rows, :]
                             for i in range(num_layers)]
        self.qkv_offloaded = [torch.zeros(self.qkv_offload_rows, hidden_dim, device='cpu', dtype=torch.float16).pin_memory()
                              for _ in range(num_layers)]
        # KV Cache
        self.k_resident = [self.k_full[i][:, :, :seq_len - self.kv_offload_rows, :]
                           for i in range(num_layers)]
        self.v_resident = [self.v_full[i][:, :, :seq_len - self.kv_offload_rows, :]
                           for i in range(num_layers)]
        self.k_offloaded = [torch.zeros(batch_size, num_heads, self.kv_offload_rows + 1, hidden_dim // num_heads, device='cpu', dtype=torch.float16).pin_memory()
                            for _ in range(num_layers)]
        self.v_offloaded = [torch.zeros(batch_size, num_heads, self.kv_offload_rows + 1, hidden_dim // num_heads, device='cpu', dtype=torch.float16).pin_memory()
                            for _ in range(num_layers)]
        # FFN Layers        
        self.ffn1_resident = [self.ffn1_full[i][:4 * hidden_dim - self.ffn1_offload_rows, :]
                             for i in range(num_layers)]
        self.ffn2_resident = [self.ffn2_full[i][:hidden_dim - self.ffn2_offload_rows, :]
                             for i in range(num_layers)]
        self.ffn1_offloaded = [torch.zeros(self.ffn1_offload_rows, hidden_dim, device='cpu', dtype=torch.float16).pin_memory()
                              for _ in range(num_layers)]
        self.ffn2_offloaded = [torch.zeros(self.ffn2_offload_rows, 4 * hidden_dim, device='cpu', dtype=torch.float16).pin_memory()
                              for _ in range(num_layers)]
        # Output Linear
        self.output_resident = self.output_full[:hidden_dim - self.output_offload_rows, :]
        self.output_offloaded = torch.zeros(self.output_offload_rows, hidden_dim, device='cpu', dtype=torch.float16).pin_memory()

        # Copy offloaded portions to CPU
        for i in range(num_layers):
            self.qkv_offloaded[i].copy_(self.qkv_full[i][3 * hidden_dim - self.qkv_offload_rows:, :])
            # Only copy the history part of the KV cache.
            # The last slot (seq_len) is reserved for the new token and shouldn't overwrite fresh computations.
            self.k_offloaded[i][..., :-1, :].copy_(self.k_full[i][:, :, seq_len - self.kv_offload_rows : seq_len, :])
            self.v_offloaded[i][..., :-1, :].copy_(self.v_full[i][:, :, seq_len - self.kv_offload_rows : seq_len, :])
            
            self.ffn1_offloaded[i].copy_(self.ffn1_full[i][4 * hidden_dim - self.ffn1_offload_rows:, :])
            self.ffn2_offloaded[i].copy_(self.ffn2_full[i][hidden_dim - self.ffn2_offload_rows:, :])
            self.output_offloaded.copy_(self.output_full[hidden_dim - self.output_offload_rows:, :])

        # Scratchpad space for offloaded portions on GPU (1 per operation per layer)
        self.qkv_scratch = torch.zeros(self.qkv_offload_rows, hidden_dim, device=dev_gpu, dtype=torch.float16)
        self.k_scratch = torch.zeros(batch_size, num_heads, self.kv_offload_rows + 1, hidden_dim // num_heads, device=dev_gpu, dtype=torch.float16)
        self.v_scratch = torch.zeros(batch_size, num_heads, self.kv_offload_rows + 1, hidden_dim // num_heads, device=dev_gpu, dtype=torch.float16)
        self.ffn1_scratch = torch.zeros(self.ffn1_offload_rows, hidden_dim, device=dev_gpu, dtype=torch.float16)
        self.ffn2_scratch = torch.zeros(self.ffn2_offload_rows, 4 * hidden_dim, device=dev_gpu, dtype=torch.float16)
        self.output_scratch = torch.zeros(self.output_offload_rows, hidden_dim, device=dev_gpu, dtype=torch.float16)

    # Calculate the total percentage of data offloaded
    # 1 - (resident_data + scratchpad_data) / total_data
    def offload_ratio(self):
        resident_elements = self.num_layers * (
            self.qkv_resident[0].numel() +
            self.k_resident[0].numel() +
            self.v_resident[0].numel() +
            self.ffn1_resident[0].numel() +
            self.ffn2_resident[0].numel()
        ) + self.output_resident.numel()

        scratchpad_elements = self.qkv_scratch.numel() + \
            self.k_scratch.numel() + \
            self.v_scratch.numel() + \
            self.ffn1_scratch.numel() + \
            self.ffn2_scratch.numel() + \
            self.output_scratch.numel()
        
        return 1 - (resident_elements + scratchpad_elements) / self.total_elements


# Transformer where each compute step has a portion of data offloaded originally
class SplitOffloadedTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, compile_mode=None):
        super(SplitOffloadedTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.heads_dim = hidden_dim // num_heads
        
        # One-time or shared functions
        self.pos_encoder = compile_if_needed(PositionalEncoding(hidden_dim), compile_mode)
        self.layer_norm = compile_if_needed(nn.LayerNorm(hidden_dim), compile_mode)
        self.softmax = compile_if_needed(nn.Softmax(dim=-1), compile_mode)

        # Larger compilation
        self.forward_compute = compile_if_needed(self.forward_compute_, compile_mode)

        # I/O Stream events
        self.IO_stream = torch.cuda.Stream()
        self.layer_start_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.qkv_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.attn_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.ffn1_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.ffn2_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.output_event = torch.cuda.Event(enable_timing=False)

    def offloaded_attention(self, q, k_A, v_A, k_B, v_B, 
                            event_transfer_done: torch.cuda.Event):
        out_A, lse_A = flex_attention(q, k_A, v_A, return_lse=True)
        torch.cuda.current_stream().wait_event(event_transfer_done)
        out_B, lse_B = flex_attention(q, k_B, v_B, return_lse=True)
        out, _ = merge_attention_states(out_A, lse_A, out_B, lse_B)
        return out
    
    def offloaded_mm(self, x, weight_A, weight_B,
                    event_transfer_done: torch.cuda.Event):
        out_A = F.linear(x, weight_A)
        torch.cuda.current_stream().wait_event(event_transfer_done)
        out_B = F.linear(x, weight_B)
        out = torch.cat([out_A, out_B], dim=-1)
        return out
    
    def forward_compute_(self, x, data: SplitOffloadedTransformerData, current_pos: int):
        # Positional Encoding
        batch_size, seq_len, _ = x.size()
        x = self.pos_encoder(x, start_pos=current_pos)

        # Iterate through layers
        for i in range(self.num_layers):
            # Signal layer start
            torch.cuda.current_stream().record_event(self.layer_start_events[i])

            # Layer Norm
            residual = x
            x_norm = self.layer_norm(x)
            
            # QKV Projection (offloaded)
            qkv = self.offloaded_mm(
                x_norm,
                data.qkv_resident[i],
                data.qkv_scratch,
                self.qkv_events[i],
            )
            qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.heads_dim)
            qkv = qkv.transpose(1, 2) # (batch, heads, seq, dim)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            # Append new k, v to KV cache (GPU scratchpad)
            data.k_scratch[:, :, -1, :] = k[:, :, -1, :]
            data.v_scratch[:, :, -1, :] = v[:, :, -1, :]

            # Attention (offloaded KV)
            attn_output = self.offloaded_attention(
                q,
                data.k_resident[i],
                data.v_resident[i],
                data.k_scratch,
                data.v_scratch,
                self.attn_events[i],
            )
            attn_output = attn_output.transpose(1, 2) # (batch_size, seq_len, num_heads, heads_dim)
            attn_output = attn_output.contiguous().view(batch_size, seq_len, self.hidden_dim)

            # Add & Norm
            x = residual + attn_output
            residual = x
            x_norm = self.layer_norm(x)

            # Feed Forward Network (GELU) (offloaded)
            ffn_output = self.offloaded_mm(
                x_norm,
                data.ffn1_resident[i],
                data.ffn1_scratch,
                self.ffn1_events[i],
            )
            ffn_output = F.gelu(ffn_output)
            ffn_output = self.offloaded_mm(
                ffn_output,
                data.ffn2_resident[i],
                data.ffn2_scratch,
                self.ffn2_events[i],
            )
            x = residual + ffn_output

        # Final Norm
        x = self.layer_norm(x)            

        # Final Linear and softmax (offloaded)
        x = self.offloaded_mm(
            x,
            data.output_resident,
            data.output_scratch,
            self.output_event,
        )
        x = self.softmax(x)
        return x
    
    def forward_IO(self, data: SplitOffloadedTransformerData):
        with torch.cuda.stream(self.IO_stream):
            for i in range(self.num_layers):
                # Wait for layer compute to start
                self.IO_stream.wait_event(self.layer_start_events[i])

                # QKV
                data.qkv_scratch.copy_(data.qkv_offloaded[i], non_blocking=True)
                self.IO_stream.record_event(self.qkv_events[i])

                # KV cache
                # IMPORTANT: Slice to [:-1] to preserve the new token computed on GPU (at index -1)
                data.k_scratch[..., :-1, :].copy_(data.k_offloaded[i][..., :-1, :], non_blocking=True)
                data.v_scratch[..., :-1, :].copy_(data.v_offloaded[i][..., :-1, :], non_blocking=True)
                self.IO_stream.record_event(self.attn_events[i])

                # FFN1
                data.ffn1_scratch.copy_(data.ffn1_offloaded[i], non_blocking=True)
                self.IO_stream.record_event(self.ffn1_events[i])
                # FFN2
                data.ffn2_scratch.copy_(data.ffn2_offloaded[i], non_blocking=True)
                self.IO_stream.record_event(self.ffn2_events[i])

            # Output
            data.output_scratch.copy_(data.output_offloaded, non_blocking=True)
            self.IO_stream.record_event(self.output_event)

    def forward(self, x, data: SplitOffloadedTransformerData, current_pos: int):
        self.forward_IO(data)
        out = self.forward_compute(x, data, current_pos=current_pos)
        return out