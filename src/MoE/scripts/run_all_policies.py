"""Run policy simulations for cache × prefetch policy combinations.

Simulation-only tool: generates GPUReplayTrace files consumed by
batched_replay.py for GPU replay. For GPU replay, use batched_replay.py
directly or via 03_gpu_replay.sh.

Usage:
    # Simulate all policies for all cache%:
    python scripts/run_all_policies.py

    # Parallel (one process per cache%):
    python scripts/run_all_policies.py --parallel

    # Single cache%:
    python scripts/run_all_policies.py --cache-pct 85
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, ".")

from gpu_replay_trace import ActivationTrace, GPUReplayTrace
from policy_simulator import (
    LRU, LFU, Belady, StaticFreq, NoPrefetch, OraclePrefetch, simulate,
)

TRACE_BASE = "datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b"


def _auto_detect_cache_pcts():
    """Auto-detect cache percentages from existing trace directories."""
    import glob
    pcts = []
    for d in sorted(glob.glob(f"{TRACE_BASE}/cache*pct")):
        base = os.path.basename(d)
        # Extract number from e.g. "cache70pct"
        try:
            pct = int(base.replace("cache", "").replace("pct", ""))
            if os.path.exists(os.path.join(d, "batched_trace.json")):
                pcts.append(pct)
        except ValueError:
            continue
    return sorted(pcts, reverse=True)  # highest first


# Cache percentages to simulate. Auto-detected from Phase 1 output dirs,
# or overridden via --cache-pct CLI arg.
CACHE_PCTS = _auto_detect_cache_pcts() or [80, 70, 60]


def _read_cache_size(pct):
    """Read cache_size from the batched trace's scheduling config."""
    trace_path = f"{TRACE_BASE}/cache{pct}pct/batched_trace.json"
    with open(trace_path) as f:
        d = json.load(f)
    cs = d.get('scheduling', {}).get('cache_size')
    if cs is None:
        raise ValueError(
            f"cache_size not found in {trace_path} scheduling config. "
            f"Re-run collect_batched_traces.py with --cache-fraction.")
    return cs

CACHE_POLICIES = [
    ("LRU", LRU),
    ("LFU", LFU),
    ("Belady", Belady),
    ("StaticFreq", StaticFreq),
]

PREFETCH_POLICIES = [
    ("None", NoPrefetch),
    ("Oracle", OraclePrefetch),
    ("Oracle(1)", lambda: OraclePrefetch(max_per_layer=1)),
]


def _simulate_one_cache_pct(pct):
    """Simulate all policies for a single cache% and save GPUReplayTrace files.

    Returns list of (policy_name, summary_dict) tuples.
    """
    cs = _read_cache_size(pct)
    trace_dir = f"{TRACE_BASE}/cache{pct}pct"
    trace_path = f"{trace_dir}/batched_trace.json"
    at = ActivationTrace.load(trace_path)
    results = []

    for cp_name, cp_cls in CACHE_POLICIES:
        for pf_name, pf_cls in PREFETCH_POLICIES:
            name = f"{cp_name}-{pf_name}"
            dm = simulate(cp_cls(), pf_cls(), at, cache_size=cs)
            s = dm.summary()

            # Save GPUReplayTrace to cache%pct directory
            out_path = f"{trace_dir}/{name}.json"
            dm.save(out_path)

            results.append((name, {
                "cache_pct": pct,
                "cache_size": cs,
                "steps": len(at.steps),
                "demands": s["demand_loads"],
                "prefetches": s["prefetches"],
                "total": s["total_transfers"],
            }))

    return pct, cs, results


def run_simulations(parallel=False):
    """Run all policy simulations and save GPUReplayTrace files.

    Args:
        parallel: If True, run each cache% in a separate process.
    """
    all_results = {}

    if parallel:
        with ProcessPoolExecutor(max_workers=len(CACHE_PCTS)) as pool:
            futures = {
                pool.submit(_simulate_one_cache_pct, pct): pct
                for pct in CACHE_PCTS
            }
            for future in as_completed(futures):
                pct, cs, results = future.result()
                all_results[pct] = (cs, results)
    else:
        for pct in CACHE_PCTS:
            _, _, results = _simulate_one_cache_pct(pct)
            all_results[pct] = (_read_cache_size(pct), results)

    # Print summary table (sorted by cache%)
    for pct in sorted(all_results):
        cs, results = all_results[pct]
        print(f"\n=== cache{pct}pct (CS={cs}, "
              f"{results[0][1]['steps']} steps) ===")
        print(f"  {'Policy':<20s}  {'Demands':>8s}  {'Prefetch':>8s}  "
              f"{'Total':>8s}")
        print("  " + "-" * 50)
        for name, info in results:
            print(f"  {name:<20s}  {info['demands']:8d}  "
                  f"{info['prefetches']:8d}  {info['total']:8d}")

    # Save flat summary
    flat = {}
    for pct, (cs, results) in all_results.items():
        for name, info in results:
            flat[f"{name}_cache{pct}pct"] = info
    summary_path = f"{TRACE_BASE}/sim_summary.json"
    with open(summary_path, "w") as f:
        json.dump(flat, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Policy simulation (CPU-only). For GPU replay, use "
                    "batched_replay.py or 03_gpu_replay.sh.")
    parser.add_argument("--parallel", action="store_true",
                        help="Run cache%% simulations in parallel processes")
    parser.add_argument("--cache-pct", type=int, default=None,
                        help="Run only this cache%% (default: all)")
    args = parser.parse_args()

    global CACHE_PCTS
    if args.cache_pct is not None:
        CACHE_PCTS = [args.cache_pct]

    run_simulations(parallel=args.parallel)


if __name__ == "__main__":
    main()
