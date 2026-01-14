import torch
import time
import argparse
import sys
import os
from include.paged_transformer import PagedTransformer, PagedTransformerData
from include.paged_offload_transformer import PagedOffloadTransformer, PagedOffloadTransformerData, TransformerArgs

def main():
    parser = argparse.ArgumentParser(description="Benchmark Paged and Paged-Offload Transformers")
    
    # Configuration
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-s", "--seq_len", type=int, default=8192, help="Sequence length (history)")
    parser.add_argument("-H", "--dim", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=64, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--block_size", type=int, default=16, help="PagedAttention block size")
    
    # Offloading
    parser.add_argument("-r", "--offload_ratio", type=float, default=0.0, 
                        help="Ratio of KV cache to offload to CPU (0.0 = Fully Resident)")
    
    # Compilation & Benchmarking
    parser.add_argument("-C", "--compile_mode", type=str, default="default", 
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--trials", type=int, default=50, help="Number of benchmark trials")
    
    args = parser.parse_args()
    
    # Derived Constants
    DTYPE = torch.float16
    DEVICE = "cuda"
    
    print(f"\n==========================================")
    print(f"       Transformer Benchmark Config       ")
    print(f"==========================================")
    print(f"Mode:          {'Offloaded' if args.offload_ratio > 0 else 'Fully Resident'}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Seq Len:       {args.seq_len}")
    print(f"Model Dim:     {args.dim}")
    print(f"Layers:        {args.layers}")
    print(f"Heads:         {args.heads}")
    print(f"Offload Ratio: {args.offload_ratio * 100:.1f}%")
    print(f"Compile Mode:  {args.compile_mode}")
    print(f"==========================================\n")

    # ------------------------------------------------------------------
    # 1. Initialize Model & Data
    # ------------------------------------------------------------------
    model_args = TransformerArgs(dim=args.dim, n_heads=args.heads, n_layers=args.layers)
    
    if args.offload_ratio > 0:
        # --- Offloaded Setup ---
        model = PagedOffloadTransformer(model_args).to(DEVICE).to(DTYPE)
        
        print(f"Allocating Offloaded Data...")
        data = PagedOffloadTransformerData(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_layers=args.layers,
            num_heads=args.heads,
            head_dim=args.dim // args.heads,
            kv_offload_ratio=args.offload_ratio,
            block_size=args.block_size,
            dtype=DTYPE,
            device=DEVICE
        )
        print(f"  > Resident Blocks: {data.num_resident_blocks} per sequence")
        print(f"  > Offload Blocks:  {data.num_offload_blocks} per sequence")
        
    else:
        # --- Resident Setup ---
        model = PagedTransformer(model_args).to(DEVICE).to(DTYPE)
        
        # Estimate blocks needed
        blocks_per_seq = (args.seq_len + 1 + args.block_size - 1) // args.block_size
        total_blocks = args.layers * args.batch_size * blocks_per_seq + 1024
        
        print(f"Allocating Resident Data (Heap size: {total_blocks} blocks)...")
        data = PagedTransformerData(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_num_blocks=total_blocks,
            num_layers=args.layers,
            num_heads=args.heads,
            head_dim=args.dim // args.heads,
            block_size=args.block_size,
            dtype=DTYPE,
            device=DEVICE
        )

    model.eval()
    
    # Input Tensor (Single Decode Step)
    x_input = torch.randn((args.batch_size, 1, args.dim), dtype=DTYPE, device=DEVICE)

    # ------------------------------------------------------------------
    # 2. Compilation
    # ------------------------------------------------------------------
    print(f"Compiling model (mode='{args.compile_mode}')...")
    
    # Note: PagedOffloadTransformer compiles internal layers in __init__
    # PagedTransformer needs explicit compile here.
    if args.offload_ratio == 0:
        compiled_model = torch.compile(model, mode=args.compile_mode)
    else:
        # Offload transformer manages its own compilation of sub-components
        compiled_model = model 
        
    # Trigger compilation with a warmup run
    with torch.no_grad():
        compiled_model(x_input, data)
    torch.cuda.synchronize()
    print("Compilation finished.")

    # ------------------------------------------------------------------
    # 3. Benchmarking
    # ------------------------------------------------------------------
    print("\n--- Running Benchmark ---")
    
    def run_step():
        with torch.no_grad():
            compiled_model(x_input, data)

    # Warmup
    for _ in range(args.warmup):
        run_step()
    torch.cuda.synchronize()

    # Timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.trials)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.trials)]

    for i in range(args.trials):
        start_events[i].record()
        run_step()
        end_events[i].record()

    torch.cuda.synchronize()
    
    # Stats
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times) / len(times)

    # ------------------------------------------------------------------
    # 4. Bandwidth Calculations
    # ------------------------------------------------------------------
    element_size = torch.tensor([], dtype=DTYPE).element_size() # 2 bytes
    
    # A. Weights Load (Read for every step)
    # 16 matrices of size D^2 per layer
    weights_bytes = args.layers * (16 * args.dim * args.dim) * element_size
    
    # B. KV Cache Write (New Token)
    # We write 1 token to all layers
    kv_write_bytes = args.layers * args.batch_size * 1 * args.dim * 2 * element_size
    
    # C. KV Cache Read (History)
    # We read the full history (Resident + Offloaded) for all layers
    total_tokens = args.batch_size * args.seq_len
    kv_read_bytes = args.layers * total_tokens * args.dim * 2 * element_size
    
    # D. PCIe Traffic (Offloaded Data Movement)
    # Only applies to Layers 1..N
    pcie_bytes = 0
    if args.offload_ratio > 0:
        # Data class stores exact number of offloaded blocks per seq
        offloaded_tokens_per_seq = data.num_offload_blocks * args.block_size
        bytes_per_layer = args.batch_size * offloaded_tokens_per_seq * args.dim * 2 * element_size
        pcie_bytes = bytes_per_layer * (args.layers - 1)

    # Metrics
    total_mem_bytes = weights_bytes + kv_write_bytes + kv_read_bytes
    effective_bw = (total_mem_bytes / 1e9) / (avg_ms / 1000.0)
    pcie_bw = (pcie_bytes / 1e9) / (avg_ms / 1000.0)

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print(f"\n--- Results ---")
    print(f"Average Step Time:   {avg_ms:.4f} ms")
    print(f"--------------------------------------------------")
    print(f"Data Volumes per Step:")
    print(f"  Weights Loaded:    {weights_bytes / 1024**3:.4f} GB")
    print(f"  KV Cache Read:     {kv_read_bytes / 1024**3:.4f} GB")
    print(f"  KV Cache Written:  {kv_write_bytes / 1024**3:.6f} GB")
    print(f"  Total Memory IO:   {total_mem_bytes / 1024**3:.4f} GB")
    if pcie_bytes > 0:
        print(f"  PCIe Transfer:     {pcie_bytes / 1024**3:.4f} GB")
    print(f"--------------------------------------------------")
    print(f"Effective Memory BW: {effective_bw:.2f} GB/s")
    if pcie_bytes > 0:
        print(f"Effective PCIe BW:   {pcie_bw:.2f} GB/s")

if __name__ == "__main__":
    main()