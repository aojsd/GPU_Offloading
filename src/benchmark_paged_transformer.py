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
    if not arg_list: return []
    result = []
    for item in arg_list:
        if ':' in item:
            try:
                count_str, len_str = item.split(':')
                result.extend([int(len_str)] * int(count_str))
            except:
                print(f"Error parsing '{item}'"); sys.exit(1)
        else:
            try: result.append(int(item))
            except: print(f"Error parsing '{item}'"); sys.exit(1)
    return result

def load_batch_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            dec_str = [str(x) for x in data.get("decode_lengths", [])]
            pre_str = [str(x) for x in data.get("prefill_lengths", [])]
            return parse_batch_distribution(dec_str), parse_batch_distribution(pre_str)
    except Exception as e:
        print(f"Error loading batch file: {e}"); sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode", nargs='+', type=str, default=["1024", "8192", "32768", "100000"],
                        help="Decode sequences. Format: 'len' or 'count:len'. Example: --decode 16:2048 4:8192")
    parser.add_argument("--prefill", nargs='+', type=str, default=["1024"],
                        help="Prefill sequences. Format: 'len' or 'count:len'. Example: --prefill 4:512")
    parser.add_argument("--batch_file", type=str, default=None)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--layers", type=int, default=25)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("-r", "--offload_ratio", type=float, default=0.0)
    parser.add_argument("--randomize_blocks", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    
    decode_lengths, prefill_lengths = [], []
    if args.batch_file:
        d, p = load_batch_from_file(args.batch_file)
        decode_lengths.extend(d); prefill_lengths.extend(p)
    if args.decode: decode_lengths = parse_batch_distribution(args.decode)
    if args.prefill: prefill_lengths = parse_batch_distribution(args.prefill)
    if not decode_lengths and not prefill_lengths:
        decode_lengths = [8192, 4096, 2048]; prefill_lengths = [512, 256]

    allocation_mode = "random" if args.randomize_blocks else "contiguous"
    DTYPE = torch.float16; DEVICE = "cuda"
    
    print(f"\n=== Configuration ===")
    print(f"Mode:          {'Offloaded' if args.offload_ratio > 0 else 'Resident'}")
    print(f"Decode Batch:  {len(decode_lengths)} seqs (Total tokens: {sum(decode_lengths)})")
    print(f"Prefill Batch: {len(prefill_lengths)} seqs (Total tokens: {sum(prefill_lengths)})")
    
    model_args = TransformerArgs(dim=args.dim, n_heads=args.heads, n_layers=args.layers)
    
    try:
        if args.offload_ratio > 0:
            model = PagedOffloadTransformer(model_args, compile_mode=args.compile_mode).to(DEVICE).to(DTYPE)
            print("Allocating Offloaded Data...")
            data = PagedOffloadTransformerData(decode_lengths, prefill_lengths, args.layers, args.heads, args.dim//args.heads, args.offload_ratio, args.block_size, DTYPE, DEVICE, allocation_mode)
            print(f"  > Decode (H2D) Blocks: {data.total_decode_offload_blocks} | Resident: {data.total_decode_resident_blocks}")
            print(f"  > Prefill (D2H) Blocks: {data.total_prefill_offload_blocks} | Resident: {data.total_prefill_resident_blocks}")
            total_resident = data.total_resident_blocks_L0 + data.total_resident_blocks_L1N
        else:
            model = PagedTransformer(model_args).to(DEVICE).to(DTYPE)
            print("Allocating Resident Data...")
            data = PagedTransformerData(decode_lengths, prefill_lengths, args.layers, args.heads, args.dim//args.heads, args.block_size, DTYPE, DEVICE, allocation_mode)
            total_resident = data.total_blocks
    except RuntimeError as e: print(f"Allocation Error: {e}"); sys.exit(1)

    model.eval()
    x_input = torch.randn((data.total_tokens, args.dim), dtype=DTYPE, device=DEVICE)
    
    print(f"Compiling ({args.compile_mode})...")
    if args.offload_ratio == 0: model = torch.compile(model, mode=args.compile_mode)
    with torch.no_grad(): model(x_input, data)
    torch.cuda.synchronize()
    print("Compilation Done.")
    
    print("\n--- Running Benchmark ---")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup): 
        with torch.no_grad(): model(x_input, data)
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.trials)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.trials)]
    
    for i in range(args.trials):
        start_events[i].record()
        with torch.no_grad(): model(x_input, data)
        end_events[i].record()
    torch.cuda.synchronize()
    
    avg_ms = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)]) / args.trials
    peak_mem = torch.cuda.max_memory_reserved()
    
    # Metrics
    element_size = torch.tensor([], dtype=DTYPE).element_size()
    weights_bytes = args.layers * (16 * args.dim * args.dim) * element_size
    kv_write_bytes = args.layers * data.total_tokens * args.dim * 2 * element_size
    kv_read_bytes = args.layers * (sum(decode_lengths)+sum(prefill_lengths)) * args.dim * 2 * element_size
    pcie_bytes = 0
    if args.offload_ratio > 0:
        moved = data.total_decode_offload_blocks + data.total_prefill_offload_blocks
        pcie_bytes = moved * args.block_size * args.dim * 2 * element_size * (args.layers - 1)
        
    total_mem_bytes = weights_bytes + kv_write_bytes + kv_read_bytes
    effective_bw = (total_mem_bytes / 1e9) / (avg_ms / 1000.0)
    tokens_per_sec = data.total_tokens / (avg_ms / 1000.0)
    
    print(f"\n--- Results ---")
    print(f"Average Step Time:   {avg_ms:.4f} ms")
    print(f"--------------------------------------------------")
    print(f"Memory Stats:")
    print(f"  Total Resident Blocks: {total_resident}")
    print(f"  Peak GPU Mem (Resv):   {peak_mem / 1024**3:.2f} GB")
    print(f"Data Volumes per Step:")
    print(f"  Weights Loaded:    {weights_bytes / 1024**3:.4f} GB")
    print(f"  KV Cache Read:     {kv_read_bytes / 1024**3:.4f} GB")
    print(f"  KV Cache Written:  {kv_write_bytes / 1024**3:.6f} GB")
    print(f"  Total Memory IO:   {total_mem_bytes / 1024**3:.4f} GB")
    if pcie_bytes > 0: print(f"  PCIe Transfer:     {pcie_bytes / 1024**3:.4f} GB")
    print(f"--------------------------------------------------")
    print(f"Effective Memory BW: {effective_bw:.2f} GB/s")
    print(f"Effective Tokens/s:  {tokens_per_sec:.2f} tok/s")

if __name__ == "__main__":
    main()