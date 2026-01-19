import torch
import time
import argparse
import sys
import os
from include.transformer_common import TransformerArgs
from include.paged_transformer import PagedTransformer, PagedTransformerData
from include.paged_offload_transformer import PagedOffloadTransformer, PagedOffloadTransformerData

# ==========================================
# HARDCODED SEQUENCE LENGTHS (Mixed Batch)
# ==========================================
DECODE_LENGTHS = [8192, 16384, 1024, 32768]
PREFILL_LENGTHS = [512, 256, 128]

def main():
    parser = argparse.ArgumentParser(description="Benchmark Paged Transformer (Mixed Batch)")
    
    # Configuration
    parser.add_argument("--dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=25, help="Number of layers")
    parser.add_argument("--block_size", type=int, default=16, help="PagedAttention block size")
    
    # Offloading
    parser.add_argument("-r", "--offload_ratio", type=float, default=0.0, 
                        help="Ratio of KV cache to offload to CPU (0.0 = Fully Resident)")
    
    # Randomization
    parser.add_argument("--randomize_blocks", action="store_true", 
                        help="Randomize physical block layout to test fragmentation")
    
    # Compilation & Benchmarking
    parser.add_argument("--compile_mode", type=str, default="default", 
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--trials", type=int, default=50, help="Number of benchmark trials")
    
    args = parser.parse_args()
    
    # Allocation Mode
    allocation_mode = "random" if args.randomize_blocks else "contiguous"

    # Derived Constants
    DTYPE = torch.float16
    DEVICE = "cuda"
    
    print(f"\n==========================================")
    print(f"       Transformer Benchmark Config       ")
    print(f"==========================================")
    print(f"Mode:          {'Offloaded' if args.offload_ratio > 0 else 'Fully Resident'}")
    print(f"Decode Seqs:   {len(DECODE_LENGTHS)} {DECODE_LENGTHS}")
    print(f"Prefill Seqs:  {len(PREFILL_LENGTHS)} {PREFILL_LENGTHS}")
    print(f"Model Dim:     {args.dim}")
    print(f"Layers:        {args.layers}")
    print(f"Heads:         {args.heads}")
    print(f"Offload Ratio: {args.offload_ratio * 100:.1f}%")
    print(f"Alloc Mode:    {allocation_mode}")
    print(f"Compile Mode:  {args.compile_mode}")
    print(f"==========================================\n")

    # ------------------------------------------------------------------
    # 1. Initialize Model & Data
    # ------------------------------------------------------------------
    model_args = TransformerArgs(dim=args.dim, n_heads=args.heads, n_layers=args.layers)
    
    if args.offload_ratio > 0:
        # --- Offloaded Setup ---
        model = PagedOffloadTransformer(model_args, compile_mode=args.compile_mode).to(DEVICE).to(DTYPE)
        
        print(f"Allocating Offloaded Data...")
        data = PagedOffloadTransformerData(
            decode_lengths=DECODE_LENGTHS,
            prefill_lengths=PREFILL_LENGTHS, 
            num_layers=args.layers,
            num_heads=args.heads,
            head_dim=args.dim // args.heads,
            kv_offload_ratio=args.offload_ratio,
            block_size=args.block_size,
            dtype=DTYPE,
            device=DEVICE,
            allocation_mode=allocation_mode
        )
        print(f"  > Total Resident Blocks (Pool): {data.total_resident_blocks_L0 + data.total_resident_blocks_L1N}")
        print(f"  > Total Offload Blocks (Batch): {data.total_offload_blocks_batch}")
        
    else:
        # --- Resident Setup ---
        model = PagedTransformer(model_args).to(DEVICE).to(DTYPE)
        
        print(f"Allocating Resident Data (Exact Size)...")
        data = PagedTransformerData(
            decode_lengths=DECODE_LENGTHS,
            prefill_lengths=PREFILL_LENGTHS,
            num_layers=args.layers,
            num_heads=args.heads,
            head_dim=args.dim // args.heads,
            block_size=args.block_size,
            dtype=DTYPE,
            device=DEVICE,
            allocation_mode=allocation_mode
        )

    model.eval()
    
    # Input Tensor (Combined Prefill + Decode tokens)
    # [Total_Tokens, Dim]
    total_tokens = data.total_tokens
    x_input = torch.randn((total_tokens, args.dim), dtype=DTYPE, device=DEVICE)

    # ------------------------------------------------------------------
    # 2. Compilation
    # ------------------------------------------------------------------
    print(f"Compiling model (mode='{args.compile_mode}')...")
    
    if args.offload_ratio == 0:
        compiled_model = torch.compile(model, mode=args.compile_mode)
    else:
        # Offload transformer manages its own compilation
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
    weights_bytes = args.layers * (16 * args.dim * args.dim) * element_size
    
    # B. KV Cache Write (New Tokens)
    total_new_tokens = data.total_tokens
    kv_write_bytes = args.layers * total_new_tokens * args.dim * 2 * element_size
    
    # C. KV Cache Read
    total_history_tokens = sum(DECODE_LENGTHS) + sum(PREFILL_LENGTHS)
    kv_read_bytes = args.layers * total_history_tokens * args.dim * 2 * element_size
    
    # D. PCIe Traffic
    pcie_bytes = 0
    if args.offload_ratio > 0:
        # Note: In this benchmark, we only H2D copy the DECODE portion.
        # Ideally we should also count D2H prefill eviction bytes if we implemented it.
        # We stick to the bytes MOVED in the step.
        total_offload_blocks = data.total_decode_offload_blocks # Only decode is moved H2D
        bytes_per_layer = total_offload_blocks * args.block_size * args.dim * 2 * element_size
        pcie_bytes = bytes_per_layer * (args.layers - 1)

    # Metrics
    total_mem_bytes = weights_bytes + kv_write_bytes + kv_read_bytes
    effective_bw = (total_mem_bytes / 1e9) / (avg_ms / 1000.0)
    tokens_per_sec = total_new_tokens / (avg_ms / 1000.0)

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
    print(f"Effective Tokens/s:  {tokens_per_sec:.2f} tok/s")

if __name__ == "__main__":
    main()