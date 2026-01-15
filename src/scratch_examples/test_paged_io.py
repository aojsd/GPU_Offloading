import torch
import time
import argparse
import sys

# ==========================================
# HARDCODED SEQUENCE LENGTHS
# ==========================================
SEQUENCE_LENGTHS = [8192, 16384, 1024, 32768, 100000]

def main():
    parser = argparse.ArgumentParser(description="Benchmark Paged KV Cache I/O (Dynamic Batch)")
    
    # Configuration
    parser.add_argument("--dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=25, help="Number of layers")
    parser.add_argument("--block_size", type=int, default=16, help="PagedAttention block size")
    
    # Offloading
    parser.add_argument("--offload_ratio", type=float, default=0.5, 
                        help="Ratio of KV cache to offload to CPU")
    
    # Benchmarking
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--trials", type=int, default=50, help="Number of benchmark trials")
    
    args = parser.parse_args()
    
    # Constants
    DTYPE = torch.float16
    DEVICE = "cuda"
    BATCH_SIZE = len(SEQUENCE_LENGTHS)
    
    # vLLM Block Layout Constants
    element_size = torch.tensor([], dtype=DTYPE).element_size()
    x = 16 // element_size
    head_dim = args.dim // args.heads

    print(f"\n=== Paged I/O Benchmark (Dynamic Batch) ===")
    print(f"Batch Size:    {BATCH_SIZE}")
    print(f"Total Tokens:  {sum(SEQUENCE_LENGTHS)}")
    print(f"Layers (IO):   {args.layers - 1} (Layer 0 is skipped)")
    print(f"Offload Ratio: {args.offload_ratio*100}%")

    # ------------------------------------------------------------------
    # 1. Calculate Exact Offload Size
    # ------------------------------------------------------------------
    total_tokens_per_seq = [l + 1 for l in SEQUENCE_LENGTHS]
    total_blocks_per_seq = [(t + args.block_size - 1) // args.block_size for t in total_tokens_per_seq]
    
    offload_counts = []
    for tb in total_blocks_per_seq:
        n_off = int(tb * args.offload_ratio)
        offload_counts.append(n_off)
        
    total_offload_blocks_batch = sum(offload_counts)
    
    if total_offload_blocks_batch == 0:
        print("Error: offload_ratio resulted in 0 blocks. Increase ratio or seq_len.")
        sys.exit(1)

    print(f"Total Offloaded Blocks (Batch): {total_offload_blocks_batch}")

    # ------------------------------------------------------------------
    # 2. Allocate Buffers
    # ------------------------------------------------------------------
    # Shape: [blocks, heads, head_dim/x, block_size, x]
    buffer_shape = (total_offload_blocks_batch, args.heads, head_dim // x, args.block_size, x)
    
    # Calculate Data Size
    # Elements * 2 (K+V) * 2 (Bytes/FP16) * Layers
    elements_per_buffer = total_offload_blocks_batch * args.heads * (head_dim // x) * args.block_size * x
    bytes_per_layer = elements_per_buffer * 2 * element_size
    total_transfer_bytes = bytes_per_layer * (args.layers - 1)
    
    print(f"Transfer per Step: {total_transfer_bytes / 1024**3:.4f} GB")
    
    # CPU Pinned Memory (Source)
    print("Allocating Pinned CPU Buffers...")
    cpu_buffers_k = []
    cpu_buffers_v = []
    for _ in range(args.layers - 1):
        cpu_buffers_k.append(torch.randn(buffer_shape, dtype=DTYPE, device='cpu').pin_memory())
        cpu_buffers_v.append(torch.randn(buffer_shape, dtype=DTYPE, device='cpu').pin_memory())
        
    # GPU Memory (Destination)
    print("Allocating GPU Scratchpad...")
    gpu_scratchpad_k = torch.zeros(buffer_shape, dtype=DTYPE, device=DEVICE)
    gpu_scratchpad_v = torch.zeros(buffer_shape, dtype=DTYPE, device=DEVICE)
    
    io_stream = torch.cuda.Stream()

    # ------------------------------------------------------------------
    # 3. Benchmark Loop
    # ------------------------------------------------------------------
    def run_transfer_step():
        with torch.cuda.stream(io_stream):
            for i in range(args.layers - 1):
                gpu_scratchpad_k.copy_(cpu_buffers_k[i], non_blocking=True)
                gpu_scratchpad_v.copy_(cpu_buffers_v[i], non_blocking=True)

    print("\n--- Starting Benchmark ---")
    
    # Warmup
    for _ in range(args.warmup):
        run_transfer_step()
    torch.cuda.synchronize()
    
    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Use NVTX range for Nsight Systems analysis
    torch.cuda.nvtx.range_push("IO_Benchmark_Trials")
    
    # Record start event ON THE IO STREAM
    start_event.record(io_stream)
    
    for _ in range(args.trials):
        run_transfer_step()
    
    # Record end event ON THE IO STREAM
    end_event.record(io_stream)
    
    # Wait for GPU to finish
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / args.trials
    gb_s = (total_transfer_bytes / 1e9) / (avg_ms / 1000.0)
    
    print(f"\nResults:")
    print(f"Avg Time: {avg_ms:.4f} ms")
    print(f"PCIe BW:  {gb_s:.2f} GB/s")

if __name__ == "__main__":
    main()