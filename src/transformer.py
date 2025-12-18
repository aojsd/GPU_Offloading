import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import logging
import argparse
from misc.suppress_output import suppress_output

from offload_kv_attn import merge_attention_states

# Set logging to none
torch._logging.set_logs(all=logging.CRITICAL)

def compile_if_needed(module, compile_mode):
    if compile_mode is None:
        return module
    else:
        return torch.compile(module, mode=compile_mode)

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

# Manages all weights and KV cache of an offloaded transformer
class OffloadedTransformerData():
    def __init__(self, batch_size, hidden_dim, seq_len, num_heads, num_layers, dev_gpu,
                 mm_offload_ratio, kv_offload_ratio):
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
    
# Transformer where each compute step has a portion of data offloaded originally
class OffloadedTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, compile_mode=None):
        super(OffloadedTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.heads_dim = hidden_dim // num_heads
        
        # One-time or shared functions
        self.pos_encoder = compile_if_needed(PositionalEncoding(hidden_dim), compile_mode)
        self.layer_norm = compile_if_needed(nn.LayerNorm(hidden_dim), compile_mode)
        self.softmax = compile_if_needed(nn.Softmax(dim=-1), compile_mode)

        # Larger compilation
        self.offloaded_attention = compile_if_needed(self.offloaded_attention_, compile_mode)
        self.offloaded_mm = compile_if_needed(self.offloaded_mm_, compile_mode)
        self.forward_compute = compile_if_needed(self.forward_compute_, compile_mode)

        # I/O Stream events
        self.IO_stream = torch.cuda.Stream()
        self.layer_start_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.qkv_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.attn_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.ffn1_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.ffn2_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.output_event = torch.cuda.Event(enable_timing=False)

    def offloaded_attention_(self, q, k_A, v_A, k_B, v_B, 
                            event_transfer_done: torch.cuda.Event):
        out_A, lse_A = flex_attention(q, k_A, v_A, return_lse=True)
        torch.cuda.current_stream().wait_event(event_transfer_done)
        out_B, lse_B = flex_attention(q, k_B, v_B, return_lse=True)
        out, _ = merge_attention_states(out_A, lse_A, out_B, lse_B)
        return out
    
    def offloaded_mm_(self, x, weight_A, weight_B,
                    event_transfer_done: torch.cuda.Event):
        out_A = F.linear(x, weight_A)
        torch.cuda.current_stream().wait_event(event_transfer_done)
        out_B = F.linear(x, weight_B)
        out = torch.cat([out_A, out_B], dim=-1)
        return out
    
    def forward_compute_(self, x, data: OffloadedTransformerData, current_pos: int):
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
    
    def forward_IO(self, data: OffloadedTransformerData):
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

    def forward(self, x, data: OffloadedTransformerData, current_pos: int):
        self.forward_IO(data)
        out = self.forward_compute(x, data, current_pos=current_pos)
        return out



# ====================================================================================
# Driver Code for Benchmarking
# ====================================================================================
def run_benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires CUDA GPU.")
        
    device = torch.device("cuda")
    
    # --- Configuration ---
    H_dim = args.hidden_dim
    L = args.num_layers
    Heads = args.num_heads
    Seq_Len = args.seq_length
    B = args.batch_size
    
    mm_ratio = args.w_offload
    kv_ratio = args.kv_offload
    
    if args.compile_mode == 0:
        compile_mode = None
    elif args.compile_mode == 1:
        compile_mode = "reduce-overhead"
    else:
        compile_mode = "max-autotune"

    print("="*85)
    print(f"{'Full Offloaded Transformer Benchmark':^85}")
    print("="*85)
    print(f"{'Batch Size':<25} : {B}")
    print(f"{'Hidden Dim':<25} : {H_dim}")
    print(f"{'Layers':<25} : {L}")
    print(f"{'Heads':<25} : {Heads}")
    print(f"{'Seq Length (Cache)':<25} : {Seq_Len}")
    print("-" * 85)
    print(f"{'Weight Offload Ratio':<25} : {mm_ratio:.2f}")
    print(f"{'KV Offload Ratio':<25} : {kv_ratio:.2f}")
    print(f"{'Compile Mode':<25} : {compile_mode if compile_mode else 'Disabled'}")
    print("=" * 85)

    # Weights Calculation
    weight_params = L * (3*H_dim*H_dim + 4*H_dim*H_dim + 4*H_dim*H_dim) + (H_dim*H_dim)
    weight_bytes = weight_params * 2 # FP16
    
    # KV Cache Calculation (Read)
    head_dim = H_dim // Heads
    kv_bytes = 2 * L * B * Heads * Seq_Len * head_dim * 2
    
    total_bytes_per_step = weight_bytes + kv_bytes
    print(f"{'Total Model Size':<25} : {weight_bytes / 1024**3:.2f} GB")
    print(f"{'Total KV Size':<25} : {kv_bytes / 1024**3:.2f} GB")
    print(f"{'Data Processed/Step':<25} : {total_bytes_per_step / 1024**3:.4f} GB")
    print("-" * 85)

    # Input token (batch_size, 1, hidden_dim)
    x = torch.randn(B, 1, H_dim, device=device, dtype=torch.float16)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    num_iters = 20

    # Initialize Data Container (Weights + Cache)
    print("Initializing model weights and cache...")
    data = OffloadedTransformerData(B, H_dim, Seq_Len, Heads, L, device, 
                                    mm_offload_ratio=mm_ratio, 
                                    kv_offload_ratio=kv_ratio)
    
    # -------------------------------------------------------
    # 1. Resident Transformer
    # -------------------------------------------------------
    print("  -> Benchmarking Resident Transformer...")
    with suppress_output():
        res_model = ResidentTransformer(H_dim, Heads, L, compile_mode=compile_mode).to(device).to(torch.float16)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            torch.compiler.cudagraph_mark_step_begin()
            _ = res_model(x, data, current_pos=Seq_Len)
    torch.cuda.synchronize()

    # Measure
    with torch.no_grad():
        start_event.record()
        for _ in range(num_iters):
            torch.compiler.cudagraph_mark_step_begin()
            _ = res_model(x, data, current_pos=Seq_Len)
        end_event.record()
        torch.cuda.synchronize()
    
    time_res = start_event.elapsed_time(end_event) / num_iters
    bw_res = (total_bytes_per_step / 1e9) / (time_res / 1000.0)
    
    # -------------------------------------------------------
    # 2. Offloaded Transformer
    # -------------------------------------------------------
    print("  -> Benchmarking Offloaded Transformer...")
    with suppress_output():
        off_model = OffloadedTransformer(H_dim, Heads, L, compile_mode=compile_mode).to(device).to(torch.float16)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            torch.compiler.cudagraph_mark_step_begin()
            _ = off_model(x, data, current_pos=Seq_Len)
    torch.cuda.synchronize()

    # Measure
    with torch.no_grad():
        start_event.record()
        for _ in range(num_iters):
            torch.compiler.cudagraph_mark_step_begin()
            _ = off_model(x, data, current_pos=Seq_Len)
        end_event.record()
        torch.cuda.synchronize()

    time_off = start_event.elapsed_time(end_event) / num_iters
    bw_off = (total_bytes_per_step / 1e9) / (time_off / 1000.0)

    # -------------------------------------------------------
    # 3. Results
    # -------------------------------------------------------
    def calc_diff(time_val):
        val = ((time_val - time_res) / time_res) * 100
        return f"{val:+.2f}%"

    print("\n" + "=" * 85)
    print(f"{'Benchmark Results':^85}")
    print("=" * 85)
    print(f"{'Metric':<30} | {'Time (ms)':<12} | {'Eff. BW':<18} | {'Note':<15}")
    print("-" * 85)
    print(f"{'1. Resident GPU':<30} | {time_res:<12.4f} | {bw_res:<7.2f} GB/s {'':<8} | {'Baseline':<15}")
    print(f"{'2. Offloaded Pipeline':<30} | {time_off:<12.4f} | {bw_off:<7.2f} GB/s {'':<8} | {calc_diff(time_off):<15}")
    print("-" * 85)
    print("Note: Bandwidth = (Weights + KV_Read) / Time. Ignores small intermediate writes.")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Offloaded Transformer Pipeline")
    
    parser.add_argument("-H", "--hidden-dim", type=int, default=8192, help="Hidden Dimension")
    parser.add_argument("-L", "--num-layers", type=int, default=4, help="Number of Layers")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch Size")
    parser.add_argument("-A", "--num-heads", type=int, default=32, help="Number of Heads")
    parser.add_argument("-S", "--seq-length", type=int, default=8192, help="Sequence Length (Cache Size)")
    
    parser.add_argument("--w_offload", type=float, default=0.05, help="Ratio of weights offloaded to CPU")
    parser.add_argument("--kv_offload", type=float, default=0.05, help="Ratio of KV cache offloaded to CPU")
    
    parser.add_argument("-C", "--compile-mode", type=int, default=0,
                        choices=[0, 1, 2], help="Torch Compile Mode (0=none, 1=reduce-overhead, 2=max-autotune)")
    
    args = parser.parse_args()
    
    run_benchmark(args)