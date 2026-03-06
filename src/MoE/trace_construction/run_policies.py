"""Run policy simulations on a batched trace and save results.

Uses the orthogonal CachePolicy x PrefetchPolicy decomposition.
Default policies: StaticFreq-NoPrefetch, StaticFreq-Oracle,
StaticFreq-Oracle(max_per_layer=1).

Usage:
    python run_policies.py \
        --trace ../datasets/.../batched.json \
        --output-dir ../datasets/.../cache50pct/
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MOE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOE_DIR))

from data_movement_trace import ActivationTrace
from policy_simulator import (
    StaticFreq, NoPrefetch, OraclePrefetch, simulate,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run policy simulations on batched trace")
    parser.add_argument("--trace", type=str, required=True,
                        help="Path to batched trace JSON")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for policy results")
    parser.add_argument("--cache-size", type=int, default=None,
                        help="Expert cache size (default: from trace metadata)")
    args = parser.parse_args()

    # Load trace
    print(f"Loading trace: {args.trace}")
    with open(args.trace) as f:
        trace_data = json.load(f)

    activation_trace = ActivationTrace.from_flat_trace(trace_data)
    print(f"  {len(activation_trace.steps)} steps, "
          f"{activation_trace.num_layers}L, {activation_trace.num_experts}E")

    # Determine cache size from args or trace metadata
    if args.cache_size is not None:
        cache_size = args.cache_size
    elif 'memory' in trace_data:
        cache_size = trace_data['memory']['expert_cache_size']
    elif 'memory_budget' in trace_data:
        cache_size = trace_data['memory_budget']['expert_cache_size']
    else:
        raise ValueError(
            "No cache_size specified and no memory/memory_budget in trace. "
            "Use --cache-size or regenerate trace with --cache-fraction.")

    total_experts = activation_trace.num_layers * activation_trace.num_experts
    print(f"  Cache size: {cache_size} / {total_experts} "
          f"({cache_size / total_experts * 100:.1f}%)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Policy configurations: (name, CachePolicy, PrefetchPolicy)
    configs = [
        ("staticfreq_none", StaticFreq(), NoPrefetch()),
        ("staticfreq_oracle", StaticFreq(), OraclePrefetch()),
        ("staticfreq_oracle1", StaticFreq(), OraclePrefetch(max_per_layer=1)),
    ]

    results_summary = {}

    for name, cache_policy, prefetch_policy in configs:
        print(f"\nRunning {name}...")
        t0 = time.time()
        dm_trace = simulate(cache_policy, prefetch_policy,
                            activation_trace, cache_size=cache_size)
        elapsed = time.time() - t0

        # Validate
        errors = dm_trace.validate()
        if errors:
            print(f"  VALIDATION ERRORS:")
            for e in errors[:5]:
                print(f"    {e}")
        else:
            print(f"  Validation: OK")

        # Summary
        summary = dm_trace.summary()
        summary['elapsed_s'] = round(elapsed, 2)
        summary['cache_size'] = cache_size
        summary['policy'] = name
        results_summary[name] = summary

        print(f"  Steps: {summary['steps']}")
        print(f"  Total transfers: {summary['total_transfers']}")
        print(f"  Prefetches: {summary['prefetches']}")
        print(f"  Demand loads: {summary['demand_loads']}")
        print(f"  Evictions: {summary['evictions']}")
        print(f"  Time: {elapsed:.2f}s")

        # Save
        out_path = os.path.join(args.output_dir, f"{name}.json")
        dm_trace.save(out_path)
        print(f"  Saved: {out_path}")

    # Save combined summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"Policy Comparison (cache_size={cache_size}/{total_experts}, "
          f"{cache_size / total_experts * 100:.1f}%)")
    print("=" * 80)
    print(f"{'Policy':<25s} {'Transfers':>10s} {'Prefetch':>10s} "
          f"{'Demand':>10s} {'Evictions':>10s} {'Time':>8s}")
    print("-" * 80)
    for name, s in results_summary.items():
        print(f"{name:<25s} {s['total_transfers']:>10d} {s['prefetches']:>10d} "
              f"{s['demand_loads']:>10d} {s['evictions']:>10d} "
              f"{s['elapsed_s']:>7.2f}s")


if __name__ == "__main__":
    main()
