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
      - Explicit Empty:  "0" or "none" -> []
      - Simple integers: "1024" -> [1024]
      - Distributions:   "4:2048" -> [2048, 2048, 2048, 2048]
    """
    if not arg_list:
        return []
    
    # Handle explicit empty request (e.g. "--prefill 0")
    if len(arg_list) == 1 and arg_list[0].lower() in ['0', 'none']:
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
                val = int(item)
                # Filter out accidental 0s if mixed with other numbers, though usually 0 means "stop/empty"
                if val > 0:
                    result.append(val)
            except ValueError:
                print(f"Error parsing batch argument '{item}'. Expected integer.")
                sys.exit(1)
    return result

def load_batch_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            dec_str = [str(x) for x in data.get("decode_lengths", [])]
            pre_str = [str(x) for x in data.get("prefill_lengths", [])]
            return parse_batch_distribution(dec_str), parse_batch_distribution(pre_str)
    except Exception as e:
        print(f"Error loading batch file '{filepath}': {e}"); sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Benchmark Paged Transformer (Mixed Batch)")
    
    # --- Input Options ---
    # Default Batches updated as requested
    parser.add_argument("--decode", nargs='+', type=str, 
                        default=["1024", "8192", "32768", "100000"],
                        help="Decode sequences. Pass '0' to disable.")
    
    parser.add_argument("--prefill", nargs='+', type=str, 
                        default=["1024"],
                        help="Prefill sequences. Pass '0' to disable.")
    
    parser.add_argument("--batch_file", type=str, default=None,
                        help="Path to JSON file containing 'decode_lengths' and 'prefill_lengths'")

    # Model Configuration
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--layers", type=int, default=25)
    parser.add_argument("--block_size", type=int, default=16)
    
    # Offloading
    parser.add_argument("-r", "--offload_ratio", type=float, default=0.0)
    
    # Optimization
    parser.add_argument("--randomize_blocks", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default")
    
    # Runtime
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    
    args = parser.parse_args()
    
    # ------------------------------------------------------------------
    # 1. Resolve Batch Inputs
    # ------------------------------------------------------------------
    decode_lengths = []
    prefill_lengths = []

    # If batch file is present, load it
    if args.batch_file:
        print(f"Loading batch config from {args.batch_file}...")
        file_decode, file_prefill = load_batch_from_file(args.batch_file)
        decode_lengths.extend(file_decode)
        prefill_lengths.extend(file_prefill)

    # CLI args override/append to file args if present, or use defaults
    # Note: If batch_file is NOT used, args.decode/prefill will contain the defaults
    if args.decode is not None:
        if args.batch_file: print("Overriding/Appending Decode lengths from CLI...")
        # If user passed "0", this returns []
        decode_lengths = parse_batch_distribution(args.decode)
        
    if args.prefill is not None:
        if args.batch_file: print("Overriding/Appending Prefill lengths from CLI...")
        # If user passed "0", this returns []
        prefill_lengths = parse_batch_distribution(args.prefill)

    # ------------------------------------------------------------------
    # 2. Setup
    # ------------------------------------------------------------------
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

    model_args = TransformerArgs(dim=args.dim, n_heads=args.heads, n_layers=args.layers)
    
    try:
        if args.offload_ratio > 0:
            model = PagedOffloadTransformer(model_args, compile_mode=args.compile_mode).to(DEVICE).to(DTYPE)
            print("Allocating Offloaded Data...")
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
            print(f"  > Decode (H2D) Blocks: {data.total_decode_offload_blocks} | Resident: {data.total_decode_resident_blocks}")
            print(f"  > Prefill (D2H) Blocks: {data.total_prefill_offload_blocks} | Resident: {data.total_prefill_resident_blocks}")
            total_resident_blocks = data.total_resident_blocks_L0 + data.total_resident_blocks_L1N
            
        else:
            model = PagedTransformer(model_args).to(DEVICE).to(DTYPE)
            print("Allocating Resident Data...")
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
            total_resident_blocks = data.total_blocks
            
    except RuntimeError as e:
        print(f"\n[Error] Allocation Failed: {e}")
        sys.exit(1)

    model.eval()
    
    # Input Tensor
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
        compiled_model = model 
        
    with torch.no_grad():
        compiled_model(x_input, data)
    torch.cuda.synchronize()
    print("Compilation finished.")

    # ------------------------------------------------------------------
    # 4. Benchmarking
    # ------------------------------------------------------------------
    print("\n--- Running Benchmark ---")
    
    torch.cuda.reset_peak_memory_stats()
    
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
    
    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()

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
    print(f"Memory Stats:")
    print(f"  Total Resident Blocks: {total_resident_blocks}")
    print(f"  Peak GPU Mem (Alloc):  {peak_allocated_bytes / 1024**3:.2f} GB")
    print(f"  Peak GPU Mem (Resv):   {peak_reserved_bytes / 1024**3:.2f} GB")
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