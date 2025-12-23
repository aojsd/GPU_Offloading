import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from offload_kv_attn import merge_attention_states
from include.misc import compile_if_needed

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, start_pos=0):
        # Slice from start_pos to start_pos + seq_len (which is usually 1 during decode)
        seq_len = x.size(1)
        return x + self.pe[:, start_pos : start_pos + seq_len]


# Dummy parent class for all transformer data objects
class OffloadedTransformerData():
    def __init__(self):
        return

# Manages all weights and KV cache of a split offloaded transformer
class SplitOffloadedTransformerData(OffloadedTransformerData):
    def __init__(self, batch_size, hidden_dim, seq_len, num_heads, num_layers, dev_gpu,
                 mm_offload_ratio, kv_offload_ratio):
        super(SplitOffloadedTransformerData, self).__init__()
        
        # Offloaded rows calculation
        self.qkv_offload_rows = int(3*hidden_dim * mm_offload_ratio)
        self.kv_offload_rows = int(seq_len * kv_offload_ratio)
        self.ffn1_offload_rows = int(4 * hidden_dim * mm_offload_ratio)
        self.ffn2_offload_rows = int(hidden_dim * mm_offload_ratio)
        self.output_offload_rows = int(hidden_dim * mm_offload_ratio)

        # Initialize QKV projection weights
        self.qkv_full = [torch.randn(3 * hidden_dim, hidden_dim, device=dev_gpu, dtype=torch.float16)
                         for _ in range(num_layers)]
        self.qkv_resident = [self.qkv_full[i][:3 * hidden_dim - self.qkv_offload_rows, :]
                             for i in range(num_layers)]
        self.qkv_offloaded = [torch.zeros(self.qkv_offload_rows, hidden_dim, device='cpu', dtype=torch.float16).pin_memory()
                              for _ in range(num_layers)]

        # Initialize KV cache for each layer and head
        # Allocate an extra seq_len for a new token during decode
        self.seq_len = seq_len
        self.k_full = [torch.randn(batch_size, num_heads, seq_len + 1, hidden_dim // num_heads, device=dev_gpu, dtype=torch.float16)
                       for _ in range(num_layers)]
        self.v_full = [torch.randn(batch_size, num_heads, seq_len + 1, hidden_dim // num_heads, device=dev_gpu, dtype=torch.float16)
                       for _ in range(num_layers)]
        self.k_resident = [self.k_full[i][:, :, :seq_len - self.kv_offload_rows, :]
                           for i in range(num_layers)]
        self.v_resident = [self.v_full[i][:, :, :seq_len - self.kv_offload_rows, :]
                           for i in range(num_layers)]
        self.k_offloaded = [torch.zeros(batch_size, num_heads, self.kv_offload_rows + 1, hidden_dim // num_heads, device='cpu', dtype=torch.float16).pin_memory()
                            for _ in range(num_layers)]
        self.v_offloaded = [torch.zeros(batch_size, num_heads, self.kv_offload_rows + 1, hidden_dim // num_heads, device='cpu', dtype=torch.float16).pin_memory()
                            for _ in range(num_layers)]

        # Initialize FFN weights
        self.ffn1_full = [torch.randn(4* hidden_dim, hidden_dim, device=dev_gpu, dtype=torch.float16)
                          for _ in range(num_layers)]
        self.ffn2_full = [torch.randn(hidden_dim, 4 * hidden_dim, device=dev_gpu, dtype=torch.float16)
                          for _ in range(num_layers)]
        self.ffn1_resident = [self.ffn1_full[i][:4 * hidden_dim - self.ffn1_offload_rows, :]
                             for i in range(num_layers)]
        self.ffn2_resident = [self.ffn2_full[i][:hidden_dim - self.ffn2_offload_rows, :]
                             for i in range(num_layers)]
        self.ffn1_offloaded = [torch.zeros(self.ffn1_offload_rows, hidden_dim, device='cpu', dtype=torch.float16).pin_memory()
                              for _ in range(num_layers)]
        self.ffn2_offloaded = [torch.zeros(self.ffn2_offload_rows, 4 * hidden_dim, device='cpu', dtype=torch.float16).pin_memory()
                              for _ in range(num_layers)]
        
        # Initialize output linear weights
        self.output_full = torch.randn(hidden_dim, hidden_dim, device=dev_gpu, dtype=torch.float16)
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
