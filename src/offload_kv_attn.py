import torch
from torch.nn.attention.flex_attention import flex_attention
import argparse
import time
import torch.compiler

# Set all PyTorch subsystems to only report ERRORS
import logging
torch._logging.set_logs(all=logging.ERROR)

# Check for GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("This benchmark requires a CUDA GPU.")

def compile_if_needed(func, compile_mode):
    if compile_mode is None:
        return func
    else:
        return torch.compile(func, mode=compile_mode)

# ==========================================
# Helper: Merge Logic
# ==========================================
def merge_attention_states(out1, lse1, out2, lse2):
    # Upcast to FP32 for numerical stability
    out1 = out1.float()
    out2 = out2.float()
    
    lse1 = lse1.float().unsqueeze(-1)
    lse2 = lse2.float().unsqueeze(-1)
    
    # Online Softmax Merge Logic
    new_lse = torch.logaddexp(lse1, lse2)
    w1 = torch.exp(lse1 - new_lse)
    w2 = torch.exp(lse2 - new_lse)
    
    new_out = out1 * w1 + out2 * w2
    
    return new_out.to(torch.float16), new_lse.squeeze(-1)

# ==========================================
# Baseline Attention
# ==========================================
class BaselineAttention(torch.nn.Module):
    def forward(self, q, k, v):
        # Standard full attention over the entire sequence
        return flex_attention(q, k, v, return_lse=True)

# ==========================================
# Offloaded Attention
# ==========================================
class HybridOffloadAttention(torch.nn.Module):
    def __init__(self, compile_mode=None):
        super().__init__()
        self.transfer_stream = torch.cuda.Stream()
        self.compute_A = compile_if_needed(self._compute_A_impl, compile_mode)
        self.compute_B_merge = compile_if_needed(self._compute_B_merge_impl, compile_mode)

    def _compute_A_impl(self, q, k_A, v_A):
        return flex_attention(q, k_A, v_A, return_lse=True)

    def _compute_B_merge_impl(self, q, k_B, v_B, out_A, lse_A):
        out_B, lse_B = flex_attention(q, k_B, v_B, return_lse=True)
        return merge_attention_states(out_A, lse_A, out_B, lse_B)
    
    def start_transfers(self, k_B_gpu, v_B_gpu, k_B_host, v_B_host):
        with torch.cuda.stream(self.transfer_stream):
            k_B_gpu.copy_(k_B_host, non_blocking=True)
            v_B_gpu.copy_(v_B_host, non_blocking=True)

    def forward(self, q, k_A, v_A, k_B_gpu, v_B_gpu, k_B_host, v_B_host):
        # Start Transfers
        self.start_transfers(k_B_gpu, v_B_gpu, k_B_host, v_B_host)

        # Start Compute A (Compiled Graph Dispatch)
        out_A, lse_A = self.compute_A(q, k_A, v_A)
        
        # Synchronization (CPU tells GPU Stream 0 to wait for Stream 1)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        
        # Compute B & Merge (Compiled Graph Dispatch)
        return self.compute_B_merge(q, k_B_gpu, v_B_gpu, out_A, lse_A)

# ==========================================
# Result Sanity Check
# ==========================================
class MergedAttention(torch.nn.Module):
    def __init__(self, compile_mode=None):
        super().__init__()
        self.flex_attention = compile_if_needed(flex_attention, compile_mode)
        self.compute_B_merge = compile_if_needed(self.compute_B_merge_, compile_mode)
    
    def compute_B_merge_(self, q, k_B, v_B, out_A, lse_A):
        out_B, lse_B = flex_attention(q, k_B, v_B, return_lse=True)
        return merge_attention_states(out_A, lse_A, out_B, lse_B)

    def forward(self, q, k_A, v_A, k_B, v_B):
        out_A, lse_A = self.flex_attention(q, k_A, v_A, return_lse=True)
        return self.compute_B_merge(q, k_B, v_B, out_A, lse_A)
    
# ==========================================
# Benchmarking
# ==========================================
def benchmark(args):
    # --- Configuration from Args ---
    B_BATCH = args.batch_size
    H = args.heads
    D_HEAD = args.head_dim
    Q_len = args.q_len
    if args.compile_mode == 0:
        compile_mode = None
    elif args.compile_mode == 1:
        compile_mode = "reduce-overhead"
    else:
        compile_mode = "max-autotune"
    
    total_kv = args.seq_length
    offload_ratio = args.offload_ratio
    
    # Calculate Splits
    len_B = int(total_kv * offload_ratio) # Block B (Offloaded)
    len_A = total_kv - len_B              # Block A (On-Chip)

    D_MODEL = H * D_HEAD
    dtype = torch.float16
    device = torch.device("cuda")

    # --- Size Calculations for Throughput ---
    element_size = 2 # FP16 = 2 bytes
    # Total KV size in bytes (K + V)
    total_kv_bytes = B_BATCH * H * total_kv * D_HEAD * element_size * 2 
    # Offloaded KV size in bytes (K + V for Block B only)
    transfer_bytes = B_BATCH * H * len_B * D_HEAD * element_size * 2

    print("=" * 75)
    print(f"{'Concurrent Offloading Benchmark Configuration':^75}")
    print("=" * 75)
    print(f"{'Batch Size (B)':<25} : {B_BATCH}")
    print(f"{'Attention Heads (H)':<25} : {H}")
    print(f"{'Head Dimension (D_head)':<25} : {D_HEAD}")
    print(f"{'Total KV Length':<25} : {total_kv}")
    print("-" * 75)
    print(f"{'Offload Ratio':<25} : {offload_ratio:.2f}")
    print(f"{'KV Block A (On-Chip)':<25} : {len_A}")
    print(f"{'KV Block B (Off-Chip)':<25} : {len_B}")
    print(f"{'Total KV Size':<25} : {total_kv_bytes / 1024**3:.2f} GB")
    print(f"{'Transfer Size (Block B)':<25} : {transfer_bytes / 1024**3:.2f} GB")
    print("=" * 75)

    if compile_mode is not None:
        print(f"\nStarting Warmup and Compilation ({compile_mode})...\n")
    else:
        print(f"\nStarting Warmup (No Compilation)...\n")
    
    # --- Allocations ---
    q = torch.randn(B_BATCH, H, Q_len, D_HEAD, device=device, dtype=dtype)
    k_full = torch.randn(B_BATCH, H, total_kv, D_HEAD, device=device, dtype=dtype)
    v_full = torch.randn(B_BATCH, H, total_kv, D_HEAD, device=device, dtype=dtype)

    # Split Views
    k_A = k_full[:, :, :len_A, :].contiguous()
    v_A = v_full[:, :, :len_A, :].contiguous()
    k_B = k_full[:, :, len_A:, :].contiguous()
    v_B = v_full[:, :, len_A:, :].contiguous()

    # Host Memory (Pinned)
    k_B_host = k_B.to("cpu").pin_memory()
    v_B_host = v_B.to("cpu").pin_memory()
    
    # GPU Buffer for B
    k_B_gpu = torch.zeros(B_BATCH, H, len_B, D_HEAD, device=device, dtype=dtype)
    v_B_gpu = torch.zeros(B_BATCH, H, len_B, D_HEAD, device=device, dtype=dtype)

    # Timing Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    num_iters = 100

    # ==========================================
    # Phase 0: Profiling Transfer Bandwidth
    # ==========================================
    print("  -> Profiling PCIe Transfer Bandwidth (H2D)...")
    if len_B > 0:
        # Warmup
        for _ in range(10):
            k_B_gpu.copy_(k_B_host, non_blocking=True)
            v_B_gpu.copy_(v_B_host, non_blocking=True)
        torch.cuda.synchronize()

        # Measure
        start_event.record()
        for _ in range(num_iters):
            k_B_gpu.copy_(k_B_host, non_blocking=True)
            v_B_gpu.copy_(v_B_host, non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()
        
        avg_transfer_time = start_event.elapsed_time(end_event) / num_iters
        transfer_bw_gbps = (transfer_bytes / 1e9) / (avg_transfer_time / 1000.0)
    else:
        avg_transfer_time = 0.0
        transfer_bw_gbps = 0.0

    # ==========================================
    # Phase 1: Baseline (Full GPU Attention)
    # ==========================================
    print("  -> Benchmarking Baseline (Full GPU, Compiled)...")
    baseline_model = compile_if_needed(BaselineAttention(), compile_mode).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            torch.compiler.cudagraph_mark_step_begin()
            baseline_model(q, k_full, v_full)
    torch.cuda.synchronize()

    # Benchmark
    with torch.no_grad():
        start_event.record()
        for _ in range(num_iters):
            torch.compiler.cudagraph_mark_step_begin()
            baseline_model(q, k_full, v_full)
        end_event.record()
        torch.cuda.synchronize()
    
    avg_baseline_time = start_event.elapsed_time(end_event) / num_iters
    baseline_throughput = (total_kv_bytes / 1e9) / (avg_baseline_time / 1000.0)
    
    # Validation Capture
    with torch.no_grad():
        out_baseline, _ = baseline_model(q, k_full, v_full)
        out_gt = out_baseline.clone()

    del baseline_model
    torch.cuda.empty_cache()

    # ==========================================
    # Phase 2: Hybrid (Concurrent Offload)
    # ==========================================
    print("  -> Benchmarking Hybrid (Concurrent Offload)...")
    
    hybrid_model = HybridOffloadAttention(compile_mode=compile_mode).to(device)

    # Warmup 
    with torch.no_grad():
        for _ in range(20):
            torch.compiler.cudagraph_mark_step_begin()
            hybrid_model(q, k_A, v_A, k_B_gpu, v_B_gpu, k_B_host, v_B_host)
    torch.cuda.synchronize()

    # Benchmark
    with torch.no_grad():
        start_event.record()
        for _ in range(num_iters):
            torch.compiler.cudagraph_mark_step_begin()
            hybrid_model(q, k_A, v_A, k_B_gpu, v_B_gpu, k_B_host, v_B_host)
        end_event.record()
        torch.cuda.synchronize()

    avg_hybrid_time = start_event.elapsed_time(end_event) / num_iters
    hybrid_throughput = (total_kv_bytes / 1e9) / (avg_hybrid_time / 1000.0)

    # Validation Capture
    k_B_gpu.zero_()
    v_B_gpu.zero_()
    with torch.no_grad():
        out_hybrid, _ = hybrid_model(q, k_A, v_A, k_B_gpu, v_B_gpu, k_B_host, v_B_host)

    # ==========================================
    # Results Output
    # ==========================================
    def calc_diff(time_val):
        if time_val == avg_baseline_time:
            return "Baseline"
        # Negative overhead implies speedup
        val = ((time_val - avg_baseline_time) / avg_baseline_time) * 100
        return f"{val:+.2f}%"

    print("\n" + "=" * 85)
    print(f"{'Benchmark Results (Avg of ' + str(num_iters) + ' runs)':^85}")
    print("=" * 85)
    print(f"{'Metric':<30} | {'Time (ms)':<12} | {'Throughput':<18} | {'Note':<15}")
    print("-" * 85)
    
    # PCIe Stats
    if len_B > 0:
        print(f"{'PCIe Transfer (H2D)':<30} | {avg_transfer_time:<12.4f} | {transfer_bw_gbps:<7.2f} GB/s {'':<8} | {'Pure Copy':<15}")
    else:
        print(f"{'PCIe Transfer (H2D)':<30} | {'N/A':<12} | {'N/A':<18} | {'No Offload':<15}")
        
    print("-" * 85)
    
    # Attention Stats
    print(f"{'1. Full Attention (GPU)':<30} | {avg_baseline_time:<12.4f} | {baseline_throughput:<7.2f} GB/s {'':<8} | {'Baseline':<15}")
    print(f"{'2. Hybrid Offload':<30} | {avg_hybrid_time:<12.4f} | {hybrid_throughput:<7.2f} GB/s {'':<8} | {calc_diff(avg_hybrid_time):<15}")
    print("-" * 85)
    
    # Validation
    max_diff = 1e-3
    if not args.default_baseline:
        merged_attn = MergedAttention(compile_mode).to(device)
        with torch.no_grad():
            out_gt, _ = merged_attn(q, k_A, v_A, k_B, v_B)
        max_diff = 0.0
    diff = (out_hybrid - out_gt).abs().max()
    
    print(f"\nValidation Delta (Max Diff): {diff.item():.6e}")
    if diff.item() <= max_diff:
        print("SUCCESS: Hybrid attention matches Full attention within tolerance.")
    else:
        print("WARNING: Numerical divergence detected.")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Concurrent KV Offloading")
    # Dimensions
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("-H", "--heads", type=int, default=48, help="Number of attention heads (default: 48)")
    parser.add_argument("-D", "--head-dim", type=int, default=256, help="Dimension per attention head (default: 256)")
    parser.add_argument("-q", "--q-len", type=int, default=1, help="Query sequence length (default: 1 for decoding)")
    
    # KV Cache Config
    parser.add_argument("-S", "--seq-length", type=int, default=10000, help="Total KV sequence length (default: 10000)")
    parser.add_argument("-r", "--offload-ratio", type=float, default=0.1, 
                        help="Fraction of KV cache offloaded/split (0.0 to 1.0). Example: 0.5 = 50%% split.")
    
    # Compilation and validation
    parser.add_argument("--default-baseline", action="store_true", help="Use default GEMM for baseline.", default=False)
    parser.add_argument("-C", "--compile-mode", type=int, default=0,
                        choices=[0, 1, 2], help="Torch Compile Mode (0=none, 1=reduce-overhead, 2=max-autotune)")
    
    args = parser.parse_args()
    benchmark(args)