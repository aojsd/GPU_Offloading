"""Measure CUDA graph memory per size for piecewise capture.

Creates an engine with minimal KV allocation and captures one graph size at a
time, recording torch.cuda.memory_allocated() delta per size. This gives the
actual per-size graph memory cost (including torch.compile Triton kernels and
CUDA graph command buffers), which is needed to calibrate the overhead in
compute_kv_budget_from_cache().

Usage:
    python profiling/measure_graph_memory.py \
        --model ../../models/Mixtral-8x7B --pp 2
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MOE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOE_DIR))

import torch

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import moe_engine as _moe_engine_mod  # noqa: F401 — glibc patches
from moe_engine import MoEEngine


GRAPH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192,
               224, 256, 288, 320, 352, 384, 448, 512]


def measure(model_path, pp=2, max_seqs=8, max_seq_len=256,
            use_torch_compile=True):
    """Measure per-size graph memory with minimal KV allocation.

    Args:
        model_path: Path to model directory.
        pp: Pipeline parallel size (2 = split model across GPUs).
        max_seqs: Minimal seq count (just enough for warmup).
        max_seq_len: Minimal seq length (just enough for warmup).
        use_torch_compile: Whether to use torch.compile for graph stages.

    Returns:
        List of per-size measurement dicts.
    """
    compile_str = "torch.compile=True" if use_torch_compile else "no compile"
    print(f"Creating engine: PP={pp}, max_seqs={max_seqs}, "
          f"max_seq_len={max_seq_len}, {compile_str}")
    engine = MoEEngine(
        model_path,
        max_seqs=max_seqs,
        max_seq_len=max_seq_len,
        pipeline_parallel_size=pp,
        use_torch_compile=use_torch_compile,
    )
    torch.cuda.synchronize()

    # Use mem_get_info (free, total) — includes CUDA graph command buffers,
    # driver allocations, and everything else that memory_allocated() misses.
    def gpu_used_gb(dev_idx):
        free, total = torch.cuda.mem_get_info(dev_idx)
        return (total - free) / 1024**3

    baseline = [gpu_used_gb(i) for i in range(pp)]
    print(f"\nBaseline GPU usage (model + KV + driver):")
    for i in range(pp):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {baseline[i]:.2f} / {total / 1024**3:.1f} GB "
              f"(free: {free / 1024**3:.2f} GB)")

    print(f"\nCapturing {len(GRAPH_SIZES)} piecewise graph sizes...")
    print(f"{'N':>6s}  {'Total delta MB':>14s}  {'Per-GPU delta':>20s}  "
          f"{'Cumulative GB':>14s}")
    print("-" * 70)

    results = []
    prev_used = [gpu_used_gb(i) for i in range(pp)]

    for gs in GRAPH_SIZES:
        try:
            engine.capture_cuda_graphs(total_token_sizes=[gs])
            torch.cuda.synchronize()
            curr = [gpu_used_gb(i) for i in range(pp)]
            delta_gb = [curr[i] - prev_used[i] for i in range(pp)]
            total_delta_mb = sum(delta_gb) * 1024
            cumul_gb = sum(curr[i] - baseline[i] for i in range(pp))
            results.append({
                'N': gs,
                'total_delta_mb': round(total_delta_mb, 1),
                'per_gpu_mb': [round(d * 1024, 1) for d in delta_gb],
                'cumulative_gb': round(cumul_gb, 2),
            })
            gpu_str = ', '.join(f'GPU{i}:+{d*1024:.0f}'
                                for i, d in enumerate(delta_gb))
            print(f"  {gs:4d}  {total_delta_mb:14.1f}  {gpu_str:>20s}  "
                  f"{cumul_gb:14.2f}")
            prev_used = curr
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {gs:4d}  OOM — stopped")
            break

    # Summary
    if not results:
        print("\nNo graphs captured — engine too large for GPU memory.")
        return results

    total_mb = sum(r['total_delta_mb'] for r in results)
    avg_mb = total_mb / len(results)
    avg_per_gpu = avg_mb / pp

    print(f"\n{'='*70}")
    print(f"Captured: {len(results)} / {len(GRAPH_SIZES)} sizes")
    print(f"Total graph memory: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
    print(f"Average per size (total): {avg_mb:.1f} MB")
    print(f"Average per size per GPU: {avg_per_gpu:.1f} MB")
    print(f"\nFor compute_kv_budget_from_cache() (single-GPU estimate):")
    single_gpu_per_size = avg_per_gpu * pp
    print(f"  graph_mb_per_size ≈ {single_gpu_per_size:.0f} MB  "
          f"({engine.num_layers} layers, 3 stages/layer)")
    print(f"  20 sizes ≈ {20 * single_gpu_per_size / 1024:.1f} GB")

    # Final memory state
    print(f"\nFinal GPU memory:")
    for i in range(pp):
        used = gpu_used_gb(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {used:.2f} / {total / 1024**3:.1f} GB "
              f"(free: {free / 1024**3:.2f} GB)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure CUDA graph memory per size")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--pp", type=int, default=2,
                        help="Pipeline parallel size (default: 2)")
    parser.add_argument("--max-seqs", type=int, default=8,
                        help="Minimal max_seqs for engine (default: 8)")
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help="Minimal max_seq_len for engine (default: 256)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile for graph stages")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    results = measure(args.model, pp=args.pp,
                      max_seqs=args.max_seqs, max_seq_len=args.max_seq_len,
                      use_torch_compile=not args.no_compile)

    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
