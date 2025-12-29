import argparse
import torch
from include.transformer_common import *
from include.split_offloaded_transformer import *
from include.resident_transfomer import ResidentTransformer

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
    if args.ratio > 0:
        mm_ratio = args.ratio
        kv_ratio = args.ratio
    
    if args.compile_mode == 0:
        compile_mode = None
    elif args.compile_mode == 1:
        compile_mode = "reduce-overhead"
    else:
        compile_mode = "max-autotune"

    print("="*85)
    print(f"{'Offloaded Transformer Benchmark':^85}")
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
    data = SplitOffloadedTransformerData(B, H_dim, Seq_Len, Heads, L, device, 
                                    mm_offload_ratio=mm_ratio, 
                                    kv_offload_ratio=kv_ratio)
    
    # -------------------------------------------------------
    # 1. Resident Transformer
    # -------------------------------------------------------
    print("  -> Benchmarking Resident Transformer...")
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
    off_model = SplitOffloadedTransformer(H_dim, Heads, L, compile_mode=compile_mode).to(device).to(torch.float16)
    
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
    print(f"{'Fully-Resident':<30} | {time_res:<12.4f} | {bw_res:<7.2f} GB/s {'':<8} | {'Baseline':<15}")
    print(f"{'Offloaded Transformer':<30} | {time_off:<12.4f} | {bw_off:<7.2f} GB/s {'':<8} | {calc_diff(time_off):<15}")
    print("=" * 85)
    print(f"{'True Offload Ratio: ' + f'{data.offload_ratio():.2%}'}")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Offloaded Transformer")
    
    parser.add_argument("-H", "--hidden-dim", type=int, default=8192, help="Hidden Dimension")
    parser.add_argument("-L", "--num-layers", type=int, default=4, help="Number of Layers")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch Size")
    parser.add_argument("-A", "--num-heads", type=int, default=32, help="Number of Heads")
    parser.add_argument("-S", "--seq-length", type=int, default=8192, help="Sequence Length (Cache Size)")
    
    parser.add_argument("--w_offload", type=float, default=0.05, help="Ratio of weights offloaded to CPU")
    parser.add_argument("--kv_offload", type=float, default=0.05, help="Ratio of KV cache offloaded to CPU")
    parser.add_argument("--ratio", type=float, default=-1.0, help="Offload ratio for both weights and KV (overrides individual settings if > 0)")

    parser.add_argument("-C", "--compile-mode", type=int, default=0,
                        choices=[0, 1, 2], help="Torch Compile Mode (0=none, 1=reduce-overhead, 2=max-autotune)")
    
    args = parser.parse_args()
    
    run_benchmark(args)