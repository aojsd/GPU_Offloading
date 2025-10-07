import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
from typing import Tuple

# --- JIT-compiled CUDA Kernel for high-precision timing ---
import cupy
_record_timestamp_kernel = cupy.RawKernel(r'''
extern "C" __global__
void recordTimestamp(unsigned long long* timestamp_out) {
    asm("mov.u64 %0, %%globaltimer;" : "=l"(*timestamp_out));
}
''', 'recordTimestamp')

# --- 1. Transformer Layer Definition ---
# We define the layer as an nn.Module, which is the standard PyTorch convention.
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn1 = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.ffn2 = nn.Linear(4 * hidden_dim, hidden_dim, bias=False)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor], index: int) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input token of shape (1, 1, H)
            kv_cache (Tuple[torch.Tensor, torch.Tensor]): Tuple of Key and Value cache tensors,
                                                          each shape (1, num_heads, L_max, head_dim)
            index (int): The current index in the sequence to write the new K/V pair.
        """
        residual = x
        x_norm = self.ln_1(x)

        qkv = self.qkv_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        # ✅ FIX 1: Use the robust view().transpose() pattern to ensure correct shape
        q = q.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # In-place update of the KV Cache at the current index
        past_k, past_v = kv_cache
        # ✅ FIX 2: Squeeze the seq_len dimension of k and v before assignment
        past_k[:, :, index, :] = k.squeeze(2)
        past_v[:, :, index, :] = v.squeeze(2)

        # Attention is performed on the cache up to the current index
        attn_output = F.scaled_dot_product_attention(q, past_k[:, :, :index+1, :], past_v[:, :, :index+1, :])
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(1, 1, self.hidden_dim)
        x = residual + self.out_proj(attn_output)

        residual = x
        x_norm = self.ln_2(x)
        
        ffn_out = self.ffn2(F.gelu(self.ffn1(x_norm)))
        x = residual + ffn_out
        
        return x

# --- 2. Main Benchmark Function ---
def run_baseline_decode(L: int, H: int, num_layers: int, trials: int):
    print("\n--- Running Non-Offloaded Baseline Transformer Decode Test (PyTorch) ---")
    
    device = torch.device("cuda")
    dtype = torch.float16
    
    num_heads = H // 128
    head_dim = H // num_heads
    
    with torch.no_grad(), torch.device(device): # ✅ Direct-to-GPU initialization
        # Create model
        model = nn.ModuleList([TransformerLayer(H, num_heads).to(dtype) for _ in range(num_layers)])
        model.eval()

        # Create dummy input token and KV cache
        input_token = torch.randn(1, 1, H, dtype=dtype)
        # ✅ Pre-allocate cache to full size (L past tokens + 1 new token)
        kv_cache = torch.randn(num_layers, 2, 1, num_heads, L + 1, head_dim, dtype=dtype)

        print(f"Model and KV Cache created directly on '{device}'. Running benchmark...")

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        
        # Warm-up runs
        for _ in range(10):
            temp_x = input_token.clone()
            for i, layer in enumerate(model):
                # We pass the full cache, and the layer modifies it in-place
                temp_x = layer(temp_x, (kv_cache[i, 0], kv_cache[i, 1]), L)
        torch.cuda.synchronize()

        print("Capturing CUDA Graph...")
        with torch.cuda.stream(s):
            g.capture_begin()
            
            graph_x = input_token.clone()
            # The graph captures the in-place modification of the kv_cache
            for i, layer in enumerate(model):
                graph_x = layer(graph_x, (kv_cache[i, 0], kv_cache[i, 1]), L)

            g.capture_end()
        torch.cuda.synchronize()
        print("Graph captured successfully.")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        timings = []

        print(f"Running {trials} timed trials...")
        for _ in range(trials):
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
        print("Done.")

    # --- Reporting (unchanged) ---
    avg_forward_pass_ms = sum(timings) / len(timings)
    avg_per_layer_ms = avg_forward_pass_ms / num_layers
    total_params = sum(p.numel() for p in model.parameters())
    total_weights_size_gb = total_params * 2 / 1.0e9 # 2 bytes for float16
    observed_gpu_throughput = total_weights_size_gb / (avg_forward_pass_ms / 1000.0)

    print("\n--- Baseline Performance (No Offloading) ---")
    print(f"{'Avg. Per-Layer Time:':<25} {avg_per_layer_ms:8.4f} ms")
    print(f"{'Avg. Forward Pass Time:':<25} {avg_forward_pass_ms:8.4f} ms")
    print(f"{'Observed GPU Throughput:':<25} {observed_gpu_throughput:8.2f} GB/s")


# --- 4. Argument Parsing and Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Transformer Decode Benchmark using PyTorch and CUDA Graphs")
    parser.add_argument("-L", "--length", type=int, default=1023, help="Sequence length of the KV Cache for decode.")
    parser.add_argument("-H", "--hidden_dim", type=int, default=12288, help="Hidden dimension of the model.")
    parser.add_argument("-N", "--num_layers", type=int, default=10, help="Number of Transformer layers.")
    parser.add_argument("-t", "--trials", type=int, default=1000, help="Number of timed trials to run.")
    
    args = parser.parse_args()

    # Simple validation
    if args.hidden_dim % 128 != 0:
        print("Error: Hidden dimension must be divisible by 128.")
    else:
        run_baseline_decode(args.length, args.hidden_dim, args.num_layers, args.trials)