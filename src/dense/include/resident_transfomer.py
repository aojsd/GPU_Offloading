from .misc import compile_if_needed
from .transformer_common import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

# Regular transformer where all matrices are resident in GPU memory
class ResidentTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, compile_mode=None):
        super(ResidentTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.heads_dim = hidden_dim // num_heads
        
        # Compute functions
        self.pos_encoder = compile_if_needed(PositionalEncoding(hidden_dim), compile_mode)
        self.layer_norm = compile_if_needed(nn.LayerNorm(hidden_dim), compile_mode)
        self.softmax = compile_if_needed(nn.Softmax(dim=-1), compile_mode)

        # Forward function
        self.forward = compile_if_needed(self.forward_, compile_mode)

    # Decode only
    def forward_(self, x, data: OffloadedTransformerData, current_pos: int):
        x = self.pos_encoder(x, start_pos=current_pos)
        batch_size, seq_len, _ = x.size()

        for i in range(self.num_layers):
            # Layer Norm
            residual = x
            x_norm = self.layer_norm(x)

            # QKV Projection
            # Use functional linear with shared weights
            qkv = F.linear(x_norm, data.qkv_full[i])
            qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.heads_dim)
            qkv = qkv.transpose(1, 2) # (batch, heads, seq, dim)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            # Append new k, v to KV cache
            data.k_full[i][:, :, -1, :] = k[:, :, -1, :]
            data.v_full[i][:, :, -1, :] = v[:, :, -1, :]

            # Attention
            attn_output = flex_attention(q, data.k_full[i], data.v_full[i])
            attn_output = attn_output.transpose(1, 2) # (batch_size, seq_len, num_heads, heads_dim)
            attn_output = attn_output.contiguous().view(batch_size, seq_len, self.hidden_dim)

            # Add & Norm
            x = residual + attn_output
            residual = x
            x_norm = self.layer_norm(x)

            # Feed Forward Network (GELU)
            ffn_output = F.linear(x_norm, data.ffn1_full[i])
            ffn_output = F.gelu(ffn_output)
            ffn_output = F.linear(ffn_output, data.ffn2_full[i])
            x = residual + ffn_output

        # Final Norm
        x = self.layer_norm(x)            

        # Final Linear and softmax
        x = F.linear(x, data.output_full)
        x = self.softmax(x)
        return x
    