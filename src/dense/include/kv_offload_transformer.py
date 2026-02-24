from .misc import compile_if_needed
from .transformer_common import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Allocates offloaded KV cache of this transformer
class CacheOffloadedTransformerData(OffloadedTransformerData):
    def __init__(self, batch_size, hidden_dim, seq_len, num_heads, num_layers, dev_gpu, kv_offload_ratio):
        super().__init__(batch_size, hidden_dim, seq_len, num_heads, num_layers, dev_gpu)
        self.BLOCK_SIZE = 32
        self.offload_rows = int((seq_len + 1) * kv_offload_ratio)
        self.resident_rows = seq_len - self.offload_rows + 1 # +1 for new token during decode

        # Allocate a single tensor for the resident KV for all layers except the first
        # Also allocate scratchpad space for the offloaded portion of a single layer.
        #  - The first layer's KV cache will be entirely resident to simplify code
        #  - We need two scratchpads to overlap data movement with the attention compute
        self.physical_len = seq_len + 1 + (num_layers - 1) * self.resident_rows \
                            + 2 * self.offload_rows
        self.k_heap = torch.zeros(batch_size, num_heads, self.physical_len, hidden_dim // num_heads,
                            device=dev_gpu, dtype=torch.float16)
        self.v_heap = torch.zeros(batch_size, num_heads, self.physical_len, hidden_dim // num_heads,
                            device=dev_gpu, dtype=torch.float16)
        self.scratchpad_start = self.physical_len - 2 * self.offload_rows

        # Allocate CPU space for the offloaded KV portions
        self.k_offload_cpu = [torch.randn(batch_size, num_heads, self.offload_rows,
                                          hidden_dim // num_heads, device='cpu',
                                          dtype=torch.float16)
                             for _ in range(num_layers-1)]
        self.v_offload_cpu = [torch.randn(batch_size, num_heads, self.offload_rows,
                                          hidden_dim // num_heads, device='cpu',
                                          dtype=torch.float16)
                             for _ in range(num_layers-1)]
        
        # Copy offloaded portions to CPU
        for i in range(num_layers-1):
            self.k_offload_cpu[i].copy_(self.k_full[i+1][:, :, self.resident_rows:, :])
            self.v_offload_cpu[i].copy_(self.v_full[i+1][:, :, self.resident_rows:, :])

        # Copy resident portions to the heap
        # Each layer's resident portion is stored in a contiguous block
        self.k_heap[:, :, :seq_len, :].copy_(self.k_full[0][:, :, :seq_len, :])
        self.v_heap[:, :, :seq_len, :].copy_(self.v_full[0][:, :, :seq_len, :])
        for i in range(num_layers-1):
            start = i * self.resident_rows + seq_len + 1
            end   = start + self.resident_rows
            self.k_heap[:, :, start:end, :].copy_(self.k_full[i+1][:, :, :self.resident_rows, :])
            self.v_heap[:, :, start:end, :].copy_(self.v_full[i+1][:, :, :self.resident_rows, :])
        
        # Assert that scratchpad regions are still zero
        assert self.k_heap[:, :, -2*self.offload_rows:, :].abs().sum() == 0
        assert self.v_heap[:, :, -2*self.offload_rows:, :].abs().sum() == 0

        # Create multiple "views" of the heap
        # - Sole purpose is to "trick" the compiler into not marking the entire heap as being used
        #   in every layer, which kills performance (skips CUDA graphs)
        self.k_views = []
        self.v_views = []
        for i in range(num_layers):
            self.k_views.append(self.k_heap.view(self.k_heap.shape))
            self.v_views.append(self.v_heap.view(self.v_heap.shape))
    
    # Calculate the total percentage of data offloaded
    # 1 - (resident_data + scratchpad_data) / total_data
    # resident_data = total_data - offloaded_data
    def offload_ratio(self):
        resident_elements = self.total_elements - (self.num_layers - 1) * (
            self.k_offload_cpu[0].numel() +
            self.v_offload_cpu[0].numel()
        )
        scratchpad_elements = 2 * (
            self.k_offload_cpu[0].numel() +
            self.v_offload_cpu[0].numel()
        )
        return 1 - (resident_elements + scratchpad_elements) / self.total_elements


# Transformer where only the KV cache is offloaded
class CacheOffloadedTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, data: CacheOffloadedTransformerData,
                 compile_mode=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.heads_dim = hidden_dim // num_heads
        
        # One-time or shared functions
        self.pos_encoder = compile_if_needed(PositionalEncoding(hidden_dim), compile_mode)
        self.layer_norm = compile_if_needed(nn.LayerNorm(hidden_dim), compile_mode)
        self.softmax = compile_if_needed(nn.Softmax(dim=-1), compile_mode)

        # I/O Stream events
        self.IO_stream = torch.cuda.Stream()
        self.attn_data_ready_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers-1)]
        self.attn_finish_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers-1)]

        # Compiled forward compute function
        self.forward_compute = compile_if_needed(self.forward_compute_, compile_mode)

        # Precompute block masks for each layer with offloaded KV
        self.block_masks = []
        for layer_idx in range(1, num_layers):
            block_mask = create_block_mask(
                self.get_layer_mask_mod(layer_idx, data),
                B=data.batch_size, H=self.num_heads, Q_LEN=1, KV_LEN=data.physical_len,
                device=data.dev_gpu,
                BLOCK_SIZE=data.BLOCK_SIZE)
            self.block_masks.append(block_mask)

    # Gets the mask modification function for a specific layer
    # Should only be used for layers with offloaded KV (all except the first)
    #  - Even layers should use the first scratchpad region (e.g., -2*offload_rows:-offload_rows)
    #  - Odd layers should use the second scratchpad region (e.g., -offload_rows:offload_rows)
    def get_layer_mask_mod(self, layer_idx, data: CacheOffloadedTransformerData):
        # Resident part for this layer
        res_start = (layer_idx - 1) * data.resident_rows
        res_end   = res_start + data.resident_rows

        # Scratchpad for this layer (alternating between the two scratchpad regions)
        scratchpad_start = data.scratchpad_start + (layer_idx % 2) * data.offload_rows
        scratchpad_end   = scratchpad_start + data.offload_rows

        # Create mask mod function
        def mask_mod(b, h, q_idx, kv_idx):
            is_resident = (kv_idx >= res_start) & (kv_idx < res_end)
            is_scratchpad = (kv_idx >= scratchpad_start) & (kv_idx < scratchpad_end)
            return is_resident | is_scratchpad
        return mask_mod
    
    # Compute portion of a forward pass
    def forward_compute_(self, x, data: CacheOffloadedTransformerData, current_pos: int):
        # Positional Encoding
        x = self.pos_encoder(x, start_pos=current_pos)
        batch_size, seq_len, _ = x.size()
        history_len = data.seq_len

        # The first layer is fully resident
        residual = x
        x = self.layer_norm(x)

        # QKV
        qkv = F.linear(x, data.qkv_full[0])
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.heads_dim)
        qkv = qkv.transpose(1, 2) # (batch, heads, seq, dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Append to cache and attention
        data.k_views[0][:, :, history_len, :] = k[:, :, -1, :]
        data.v_views[0][:, :, history_len, :] = v[:, :, -1, :]
        attn_output = flex_attention(q, data.k_views[0][:, :, :history_len + 1, :],
                                        data.v_views[0][:, :, :history_len + 1, :])
        attn_output = attn_output.transpose(1, 2) # (batch_size, seq_len, num_heads, heads_dim)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Add, Norm, FFN, GeLU, Add
        x = residual + attn_output
        residual = x
        x_norm = self.layer_norm(x)
        ffn_output = F.linear(F.gelu(F.linear(x_norm, data.ffn1_full[0])), data.ffn2_full[0])
        x = residual + ffn_output

        # Iterate through remaining layers with offloaded KV
        for layer_idx in range(1, self.num_layers):
            # Beginning steps are identical
            residual = x
            x_norm = self.layer_norm(x)
            qkv = F.linear(x_norm, data.qkv_full[layer_idx])
            qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.heads_dim)
            qkv = qkv.transpose(1, 2) # (batch, heads, seq, dim)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            # Append to cache (end of this layer's resident portion)
            cache_start_idx = history_len + 1 + (layer_idx - 1) * data.resident_rows
            cache_end_idx = cache_start_idx + data.resident_rows

            # Use views to trick the compiler into thinking we are accessing a separate tensor
            # in every layer (which we effectively are, since each layer has its own region)
            # Without views, the compiler might skip CUDA graphs due to perceived data dependencies
            data.k_views[layer_idx][:, :, cache_end_idx - 1, :] = (k[:, :, -1, :])
            data.v_views[layer_idx][:, :, cache_end_idx - 1, :] = (v[:, :, -1, :])

            # Attention
            # Wait for offloaded KV to be ready
            torch.cuda.current_stream().wait_event(self.attn_data_ready_events[layer_idx-1])
            attn_output = flex_attention(q, data.k_views[layer_idx], data.v_views[layer_idx],
                                         block_mask=self.block_masks[layer_idx-1])

            # Mark attention as finished
            torch.cuda.current_stream().record_event(self.attn_finish_events[layer_idx-1])
            attn_output = attn_output.transpose(1, 2) # (batch_size, seq_len, num_heads, heads_dim)
            attn_output = attn_output.contiguous().view(batch_size, seq_len, self.hidden_dim)

            # Add, Norm, FFN, GeLU, Add
            x = residual + attn_output
            residual = x
            x_norm = self.layer_norm(x)
            ffn_output = F.linear(F.gelu(F.linear(x_norm, data.ffn1_full[layer_idx])), data.ffn2_full[layer_idx])
            x = residual + ffn_output
        
        # Final Steps
        x = self.layer_norm(x)
        x = F.linear(x, data.output_full)
        x = self.softmax(x)
        return x
    
    # IO portion of a forward pass
    def forward_IO(self, data: CacheOffloadedTransformerData):
        with torch.cuda.stream(self.IO_stream):
            for layer_idx in range(1, self.num_layers):
                # We can begin the copy when this scratchpad region is free
                # - when attention is finished for layer_idx - 2 or
                #   immediately for layer_idx == 1 or layer_idx == 2
                if layer_idx >= 3:
                    self.IO_stream.wait_event(self.attn_finish_events[layer_idx-2 - 1])
                
                # Copy this layer's offloaded data into the scratchpad
                k_off_src = data.k_offload_cpu[layer_idx-1]
                v_off_src = data.v_offload_cpu[layer_idx-1]
                s_start = data.scratchpad_start + data.offload_rows * (layer_idx % 2)
                s_end   = s_start + data.offload_rows
                data.k_heap[:, :, s_start:s_end, :].copy_(k_off_src)
                data.v_heap[:, :, s_start:s_end, :].copy_(v_off_src)

                # Mark this attention's data as ready
                self.IO_stream.record_event(self.attn_data_ready_events[layer_idx-1])

    # Full forward pass
    def forward(self, x, data: CacheOffloadedTransformerData, current_pos: int):
        # Launch IO operations
        self.forward_IO(data)

        # Launch compute operations
        output = self.forward_compute(x, data, current_pos)
        return output