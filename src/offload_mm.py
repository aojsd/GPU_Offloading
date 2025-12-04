import torch
import argparse
import time
import logging

# Set logging to error only
torch._logging.set_logs(all=logging.ERROR)

if not torch.cuda.is_available():
    raise RuntimeError("This benchmark requires a CUDA GPU.")

# ====================================================================================
# 1. Baseline: Standard Compiled GEMM
# ====================================================================================
class BaselineGEMM(torch.nn.Module):
    def __init__(self, compile_mode="max-autotune"):
        super().__init__()
        # Compiled Subroutines
        if compile_mode:
            self.forward_default = torch.compile(self.forward_default_, mode=compile_mode)
            self.forward_split = torch.compile(self.forward_split_, mode=compile_mode)
        else:
            self.forward_default = self.forward_default_
            self.forward_split = self.forward_split_

    def forward_default_(self, W, x):
        return torch.mm(W, x)
    
    def forward_split_(self, W_A, W_B, x):
        return torch.cat([torch.mm(W_A, x), torch.mm(W_B, x)], dim=0)

    def forward(self, W_A, x, W_B=None):
        if W_B is not None:
            return self.forward_split(W_A, W_B, x)
        else:
            return self.forward_default(W_A, x)

# ====================================================================================
# 2. Optimized: Hybrid Row-Wise GEMM
# ====================================================================================
class HybridRowOffloadGEMM(torch.nn.Module):
    def __init__(self, compile_mode="max-autotune"):
        super().__init__()
        self.transfer_stream = torch.cuda.Stream()
        
        # Compiled Subroutines
        if compile_mode:
            self.compute_A = torch.compile(self._compute_A_impl, mode=compile_mode)
            self.compute_B_concat = torch.compile(self._compute_B_concat_impl, mode=compile_mode)
        else:
            self.compute_A = self._compute_A_impl
            self.compute_B_concat = self._compute_B_concat_impl

    def _compute_A_impl(self, W_A, x):
        # Compute Top Half (GPU Resident)
        return torch.mm(W_A, x)

    def _compute_B_concat_impl(self, W_B, x, y_A):
        # Compute Bottom Half (Streamed from CPU)
        y_B = torch.mm(W_B, x)
        # Concatenate: [Top Result, Bottom Result]
        return torch.cat([y_A, y_B], dim=0)

    def forward(self, W_A, W_B_gpu, W_B_host, x):
        """
        W_A:      Top Rows of Weights (Resident on GPU)
        W_B_gpu:  Buffer for Bottom Rows (Resident on GPU)
        W_B_host: Bottom Rows of Weights (Resident on CPU, Pinned)
        x:        Full Input Vector (Resident on GPU)
        """
        # 1. Start Transfer of Bottom Rows (Block B) on Side Stream
        with torch.cuda.stream(self.transfer_stream):
            W_B_gpu.copy_(W_B_host, non_blocking=True)
        
        # 2. Compute Top Rows (Block A) on Main Stream
        # Use the full vector x
        y_A = self.compute_A(W_A, x)
        
        # 3. Synchronization
        # Wait for bottom rows to arrive
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        
        # 4. Compute Bottom Rows & Concatenate
        return self.compute_B_concat(W_B_gpu, x, y_A)

# ====================================================================================
# 3. Benchmarking Infrastructure
# ====================================================================================
def benchmark(args):
    # --- Configuration ---
    H = args.hidden_dim
    b = args.batch_size
    if args.compile_mode == 0:
        compile_mode = None
    elif args.compile_mode == 1:
        compile_mode = "reduce-overhead"
    else:
        compile_mode = "max-autotune"
    
    # Offload ratio applies to ROWS (M)
    offload_ratio = args.offload_ratio
    
    # Calculate Splits
    M_B = int(H * offload_ratio) # Rows on Host (Block B)
    M_A = H - M_B                # Rows on GPU (Block A)

    dtype = torch.float16
    device = torch.device("cuda")
    element_size = 2 # FP16 bytes

    # --- Size Calculations ---
    total_weights_bytes = H * H * element_size
    transfer_bytes = M_B * H * element_size

    print("=" * 75)
    print(f"{'Row-Wise Concurrent GEMM Offloading':^75}")
    print("=" * 75)
    print(f"{'Batch Size (b)':<25} : {b}")
    print(f"{'Hidden Dim (H)':<25} : {H}")
    print(f"{'Compile Mode':<25} : {compile_mode if compile_mode else 'Disabled'}")
    print("-" * 75)
    print(f"{'Offload Ratio (Rows)':<25} : {offload_ratio:.2f}")
    print(f"{'Rows On-Chip (M_A)':<25} : {M_A}")
    print(f"{'Rows Off-Chip (M_B)':<25} : {M_B}")
    print(f"{'Total Weight Size':<25} : {total_weights_bytes / 1024**3:.2f} GB")
    print(f"{'Transfer Size (Block B)':<25} : {transfer_bytes / 1024**3:.2f} GB")
    print("=" * 75)
    
    # --- Allocations ---
    # Full Weights on Host
    W_full_host = torch.randn(H, H, dtype=dtype)
    x = torch.randn(H, b, device=device, dtype=dtype) # Full vector on GPU
    
    # Split Views (Rows)
    W_A_host = W_full_host[:M_A, :].contiguous()
    W_B_host_orig = W_full_host[M_A:, :].contiguous()
    
    # Move permanent GPU data
    W_A = W_A_host.to(device)
    
    # Pinned Memory for Stream source
    W_B_host = W_B_host_orig.pin_memory()
    
    # GPU Buffer for Stream destination
    W_B_gpu = torch.zeros(M_B, H, device=device, dtype=dtype)
    
    # Baseline Full GPU reference
    W_full_gpu = W_full_host.to(device)

    # Timing Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    num_iters = 100

    # ====================================================================================
    # Phase 0: Profiling Transfer Bandwidth
    # ====================================================================================
    print("  -> Profiling PCIe Transfer Bandwidth (H2D)...")
    if M_B > 0:
        # Warmup
        for _ in range(10):
            W_B_gpu.copy_(W_B_host, non_blocking=True)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(num_iters):
            W_B_gpu.copy_(W_B_host, non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()
        
        avg_transfer_time = start_event.elapsed_time(end_event) / num_iters
        transfer_bw_gbps = (transfer_bytes / 1e9) / (avg_transfer_time / 1000.0)
    else:
        avg_transfer_time = 0.0
        transfer_bw_gbps = 0.0

    # ====================================================================================
    # Phase 1: Baseline (Full GPU GEMM)
    # ====================================================================================
    print("  -> Benchmarking Baseline (Full GPU, Compiled)...")
    baseline_model = BaselineGEMM(compile_mode=compile_mode).to(device)
    default_baseline = args.default_baseline
    if not default_baseline:
        print("     Using Split GEMM for Baseline.")
        baseline = lambda x: baseline_model(W_A, x, W_B=W_B_gpu)
    else:
        print("     Using Full GEMM for Baseline.")
        baseline = lambda x: baseline_model(W_full_gpu, x)
    
    # Warmup
    W_B_gpu.copy_(W_B_host, non_blocking=False)  # Ensure buffer is ready
    with torch.no_grad():
        for _ in range(20):
            torch.compiler.cudagraph_mark_step_begin()
            baseline(x)
    torch.cuda.synchronize()

    # Benchmark
    with torch.no_grad():
        start_event.record()
        for _ in range(num_iters):
            torch.compiler.cudagraph_mark_step_begin()
            baseline(x)
        end_event.record()
        torch.cuda.synchronize()
    
    avg_baseline_time = start_event.elapsed_time(end_event) / num_iters
    baseline_bw = (total_weights_bytes / 1e9) / (avg_baseline_time / 1000.0)
    
    # Capture Ground Truth
    with torch.no_grad():
        torch.compiler.cudagraph_mark_step_begin()
        out = baseline(x)
        out_gt = out.clone()

    # ====================================================================================
    # Phase 2: Hybrid (Row Concurrent Offload)
    # ====================================================================================
    print("  -> Benchmarking Hybrid (Row Concurrent Offload)...")
    hybrid_model = HybridRowOffloadGEMM(compile_mode=compile_mode).to(device)

    # Warmup 
    with torch.no_grad():
        for _ in range(20):
            torch.compiler.cudagraph_mark_step_begin()
            hybrid_model(W_A, W_B_gpu, W_B_host, x)
    torch.cuda.synchronize()

    # Benchmark
    with torch.no_grad():
        start_event.record()
        for _ in range(num_iters):
            torch.compiler.cudagraph_mark_step_begin()
            hybrid_model(W_A, W_B_gpu, W_B_host, x)
        end_event.record()
        torch.cuda.synchronize()

    avg_hybrid_time = start_event.elapsed_time(end_event) / num_iters
    hybrid_bw = (total_weights_bytes / 1e9) / (avg_hybrid_time / 1000.0)

    # Validation
    W_B_gpu.zero_()
    with torch.no_grad():
        torch.compiler.cudagraph_mark_step_begin()
        out_hybrid = hybrid_model(W_A, W_B_gpu, W_B_host, x).clone()

    # ====================================================================================
    # Results
    # ====================================================================================
    def calc_diff(time_val):
        val = ((time_val - avg_baseline_time) / avg_baseline_time) * 100
        return f"{val:+.2f}%"

    print("\n" + "=" * 85)
    print(f"{'Benchmark Results':^85}")
    print("=" * 85)
    print(f"{'Metric':<30} | {'Time (ms)':<12} | {'Eff. BW':<18} | {'Note':<15}")
    print("-" * 85)
    
    if M_B > 0:
        print(f"{'PCIe Transfer (H2D)':<30} | {avg_transfer_time:<12.4f} | {transfer_bw_gbps:<7.2f} GB/s {'':<8} | {'Raw Copy':<15}")
    
    print("-" * 85)
    print(f"{'1. Full GPU GEMM':<30} | {avg_baseline_time:<12.4f} | {baseline_bw:<7.2f} GB/s {'':<8} | {'Baseline':<15}")
    print(f"{'2. Hybrid Offload':<30} | {avg_hybrid_time:<12.4f} | {hybrid_bw:<7.2f} GB/s {'':<8} | {calc_diff(avg_hybrid_time):<15}")
    print("-" * 85)
    
    # Validation
    if (out_gt - out_hybrid).sum() == 0:
        print("SUCCESS: Hybrid matches Baseline.")
    else:
        print("ERROR: Numerical Divergence.")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Concurrent GEMM Offloading")
    
    parser.add_argument("-H", "--hidden-dim", type=int, default=16384, help="Square Matrix Dimension (H)")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch Size (b)")
    parser.add_argument("-r", "--offload-ratio", type=float, default=0.2, 
                        help="Fraction of matrix rows offloaded to CPU.")
    parser.add_argument("--default-baseline", action="store_true", help="Use default GEMM for baseline.", default=False)
    parser.add_argument("-C", "--compile-mode", type=int, default=0,
                        choices=[0, 1, 2], help="Torch Compile Mode (0=none, 1=reduce-overhead, 2=max-autotune)")
    args = parser.parse_args()
    benchmark(args)