import torch
import time
import argparse
import sys
import os
import json
from include.transformer_common import TransformerArgs
from include.paged_transformer import PagedTransformer, PagedTransformerData
from include.paged_offload_transformer import PagedOffloadTransformer, PagedOffloadTransformerData

def parse_batch_distribution(arg_list):
    """
    Parses a list of strings into a flat list of integers.
    Supports:
      - Simple integers: "1024" -> [1024]
      - Distributions:   "4:2048" -> [2048, 2048, 2048, 2048]
    """
    if not arg_list:
        return []
    
    result = []
    for item in arg_list:
        if ':' in item:
            try:
                count_str, len_str = item.split(':')
                count = int(count_str)
                length = int(len_str)
                result.extend([length] * count)
            except ValueError:
                print(f"Error parsing batch argument '{item}'. Expected format 'count:length'")
                sys.exit(1)
        else:
            try:
                result.append(int(item))
            except ValueError:
                print(f"Error parsing batch argument '{item}'. Expected integer.")
                sys.exit(1)
    return result

def load_batch_from_file(filepath):
    """
    Loads decode and prefill lengths from a JSON file.
    Expected format:
    {
      "decode_lengths": [1024, "4:2048", ...],
      "prefill_lengths": [128, ...]
    }
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            # Get raw lists (default to empty)
            dec_raw = data.get("decode_lengths", [])
            pre_raw = data.get("prefill_lengths", [])
            
            # Ensure inputs are strings for the parser (handles mixed int/string JSON arrays)
            dec_str = [str(x) for x in dec_raw]
            pre_str = [str(x) for x in pre_raw]
            
            return parse_batch_distribution(dec_str), parse_batch_distribution(pre_str)
    except Exception as e:
        print(f"Error loading batch file '{filepath}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Benchmark Paged Transformer (Mixed Batch)")
    
    # --- Input Options ---
    # Option 1 & 2: CLI Args with Distribution Syntax
    parser.add_argument("--decode", nargs='+', type=str, default=["1024", "8192", "32768", "100000"],
                        help="Decode sequences. Format: 'len' or 'count:len'. Example: --decode 16:2048 4:8192")
    parser.add_argument("--prefill", nargs='+', type=str, default=["1024"],
                        help="Prefill sequences. Format: 'len' or 'count:len'. Example: --prefill 4:512")
    
    # Option 3: File-based Config
    parser.add_argument("--batch_file", type=str, default=None,
                        help="Path to JSON file containing 'decode_lengths' and 'prefill_lengths'")

    # Model Configuration
    parser.add_argument("--dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=25, help="Number of layers")
    parser.add_argument("--block_size", type=int, default=16, help="PagedAttention block size")
    
    # Offloading
    parser.add_argument("-r", "--offload_ratio", type=float, default=0.0, 
                        help="Ratio of KV cache to offload to CPU (0.0 = Fully Resident)")
    
    # Optimization
    parser.add_argument("--randomize_blocks", action="store_true", 
                        help="Randomize physical block layout to test fragmentation")
    parser.add_argument("-C", "--compile_mode", type=str, default="default", 
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    
    # Runtime
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--trials", type=int, default=50, help="Number of benchmark trials")
    
    args = parser.parse_args()
    
    # ------------------------------------------------------------------
    # 1. Resolve Batch Inputs
    # ------------------------------------------------------------------
    decode_lengths = []
    prefill_lengths = []

    # Priority: CLI Args > Batch File > Default
    
    # Load from file first if provided
    if args.batch_file:
        print(f"Loading batch config from {args.batch_file}...")
        file_decode, file_prefill = load_batch_from_file(args.batch_file)
        decode_lengths.extend(file_decode)
        prefill_lengths.extend(file_prefill)

    # Append/Override with CLI args
    if args.decode is not None:
        if args.batch_file: print("Overriding/Appending Decode lengths from CLI...")
        decode_lengths = parse_batch_distribution(args.decode)
        
    if args.prefill is not None:
        if args.batch_file: print("Overriding/Appending Prefill lengths from CLI...")
        prefill_lengths = parse_batch_distribution(args.prefill)

    # Default fallback if absolutely nothing provided
    if not decode_lengths and not prefill_lengths:
        print("No batch specified. Using default benchmark workload.")
        decode_lengths = [8192, 4096, 2048]
        prefill_lengths = [512, 256]

    # Derived Constants
    allocation_mode = "random" if args.randomize_blocks else "contiguous"
    DTYPE = torch.float16
    DEVICE = "cuda"
    
    print(f"\n==========================================")
    print(f"       Transformer Benchmark Config       ")
    print(f"==========================================")
    print(f"Mode:          {'Offloaded' if args.offload_ratio > 0 else 'Fully Resident'}")
    print(f"Decode Batch:  {len(decode_lengths)} seqs (Total tokens: {sum(decode_lengths)})")
    if 0 < len(decode_lengths) < 20: print(f"  > Lengths: {decode_lengths}")
    print(f"Prefill Batch: {len(prefill_lengths)} seqs (Total tokens: {sum(prefill_lengths)})")
    if 0 < len(prefill_lengths) < 20: print(f"  > Lengths: {prefill_lengths}")
    print(f"Model:         {args.layers}L, {args.heads}H, {args.dim}D")
    print(f"Offload:       {args.offload_ratio * 100:.1f}%")
    print(f"Alloc Mode:    {allocation_mode}")
    print(f"==========================================\n")

    # ------------------------------------------------------------------
    # 2. Initialize Model & Data
    # ------------------------------------------------------------------
    model_args = TransformerArgs(dim=args.dim, n_heads=args.heads, n_layers=args.layers)
    
    try:
        if args.offload_ratio > 0:
            # --- Offloaded Setup ---
            model = PagedOffloadTransformer(model_args, compile_mode=args.compile_mode).to(DEVICE).to(DTYPE)
            
            print(f"Allocating Offloaded Data...")
            data = PagedOffloadTransformerData(
                decode_lengths=decode_lengths,
                prefill_lengths=prefill_lengths,
                num_layers=args.layers,
                num_heads=args.heads,
                head_dim=args.dim // args.heads,
                kv_offload_ratio=args.offload_ratio,
                block_size=args.block_size,
                dtype=DTYPE,
                device=DEVICE,
                allocation_mode=allocation_mode
            )
            print(f"  > Decode Read (H2D) Blocks: {data.total_decode_offload_blocks}")
            print(f"  > Prefill Write (D2H) Blocks: {data.total_prefill_offload_blocks}")
            
        else:
            # --- Resident Setup ---
            model = PagedTransformer(model_args).to(DEVICE).to(DTYPE)
            
            print(f"Allocating Resident Data...")
            data = PagedTransformerData(
                decode_lengths=decode_lengths,
                prefill_lengths=prefill_lengths,
                num_layers=args.layers,
                num_heads=args.heads,
                head_dim=args.dim // args.heads,
                block_size=args.block_size,
                dtype=DTYPE,
                device=DEVICE,
                allocation_mode=allocation_mode
            )
    except RuntimeError as e:
        print(f"\n[Error] Allocation Failed: {e}")
        print("Try reducing batch size or sequence lengths.")
        sys.exit(1)

    model.eval()
    
    # Input Tensor (Combined Prefill + Decode tokens)
    # [Total_Tokens, Dim]
    total_tokens = data.total_tokens
    if total_tokens == 0:
        print("Error: No tokens to process (empty batch).")
        sys.exit(1)
        
    x_input = torch.randn((total_tokens, args.dim), dtype=DTYPE, device=DEVICE)

    # ------------------------------------------------------------------
    # 3. Compilation
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
    # 4. Benchmarking
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
    # 5. Bandwidth Calculations
    # ------------------------------------------------------------------
    element_size = torch.tensor([], dtype=DTYPE).element_size() # 2 bytes
    
    # A. Weights Load (Read for every step)
    weights_bytes = args.layers * (16 * args.dim * args.dim) * element_size
    
    # B. KV Cache Write (New Tokens)
    total_new_tokens = data.total_tokens
    kv_write_bytes = args.layers * total_new_tokens * args.dim * 2 * element_size
    
    # C. KV Cache Read
    total_history_tokens = sum(decode_lengths) + sum(prefill_lengths)
    kv_read_bytes = args.layers * total_history_tokens * args.dim * 2 * element_size
    
    # D. PCIe Traffic
    pcie_bytes = 0
    if args.offload_ratio > 0:
        # We move decode history H2D and prefill result D2H
        # Note: Depending on the pipeline impl, D2H might happen post-compute or overlapped
        moved_blocks = data.total_decode_offload_blocks + data.total_prefill_offload_blocks
        bytes_per_layer = moved_blocks * args.block_size * args.dim * 2 * element_size
        pcie_bytes = bytes_per_layer * (args.layers - 1)

    # Metrics
    total_mem_bytes = weights_bytes + kv_write_bytes + kv_read_bytes
    effective_bw = (total_mem_bytes / 1e9) / (avg_ms / 1000.0)
    tokens_per_sec = total_new_tokens / (avg_ms / 1000.0)

    # ------------------------------------------------------------------
    # 6. Report
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