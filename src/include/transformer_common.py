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
    def __init__(self, batch_size, hidden_dim, seq_len, num_heads, num_layers, dev_gpu):
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dev_gpu = dev_gpu

        # Initialize QKV projection weights
        self.qkv_full = [torch.randn(3 * hidden_dim, hidden_dim, device=dev_gpu, dtype=torch.float16)
                         for _ in range(num_layers)]
        
        # Initialize KV cache for each layer and head
        # Allocate an extra seq_len for a new token during decode
        self.seq_len = seq_len
        self.k_full = [torch.randn(batch_size, num_heads, seq_len + 1, hidden_dim // num_heads, device=dev_gpu, dtype=torch.float16)
                       for _ in range(num_layers)]
        self.v_full = [torch.randn(batch_size, num_heads, seq_len + 1, hidden_dim // num_heads, device=dev_gpu, dtype=torch.float16)
                       for _ in range(num_layers)]
        
        # Initialize FFN weights
        self.ffn1_full = [torch.randn(4* hidden_dim, hidden_dim, device=dev_gpu, dtype=torch.float16)
                          for _ in range(num_layers)]
        self.ffn2_full = [torch.randn(hidden_dim, 4 * hidden_dim, device=dev_gpu, dtype=torch.float16)
                          for _ in range(num_layers)]
        
        # Initialize output linear weights
        self.output_full = torch.randn(hidden_dim, hidden_dim, device=dev_gpu, dtype=torch.float16)