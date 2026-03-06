"""Benchmark replay runtime for different policy configurations.

Loads pre-computed DataMovementTraces (or generates them on the fly),
replays each on GPU hardware via ReplayController, and reports wall-clock
timing.

Usage:
    python tests/bench_replay_policies.py [--model MODEL_PATH] [--steps N]
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

from data_movement_trace import ActivationTrace, DataMovementTrace
from moe_engine import MoEEngine
from policy_simulator import (
    StaticFreq, NoPrefetch, OraclePrefetch, simulate,
)
from replay_controller import ReplayController


def generate_traces(activation_trace_path, cache_size):
    """Generate DataMovementTraces for all four configurations."""
    at = ActivationTrace.load(activation_trace_path)
    print(f"Activation trace: {len(at.steps)} steps, "
          f"{at.num_layers}L x {at.num_experts}E, cache_size={cache_size}")

    configs = [
        ("StaticFreq-None",      StaticFreq(), NoPrefetch()),
        ("StaticFreq-Oracle",    StaticFreq(), OraclePrefetch()),
        ("StaticFreq-Oracle(1)", StaticFreq(), OraclePrefetch(max_per_layer=1)),
        ("StaticFreq-Oracle(2)", StaticFreq(), OraclePrefetch(max_per_layer=2)),
    ]

    traces = {}
    for name, cp, pp in configs:
        dm = simulate(cp, pp, at, cache_size=cache_size)
        errors = dm.validate()
        assert not errors, f"{name} validation errors: {errors}"
        s = dm.summary()
        print(f"  {name:<25s}  demands={s['demand_loads']:5d}  "
              f"prefetches={s['prefetches']:5d}  total={s['total_transfers']:5d}")
        traces[name] = dm

    return traces, at


def run_replay(engine, dm_trace, max_steps=None, n_warmup=2, n_trials=5):
    """Replay a DataMovementTrace and return median step time in ms.

    Drives the engine with decode-only steps (1 token per step) since
    the trace was collected from a batched workload where each step
    processes a batch. We use a single sequence for simplicity — the
    IO timing (which dominates) is independent of batch composition.
    """
    n_steps = len(dm_trace.steps)
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    all_times = []

    for trial in range(n_warmup + n_trials):
        engine.reset()
        controller = ReplayController(engine, dm_trace)
        controller.setup()
        engine.replay_controller = controller

        # Seed KV cache: set seq_lens so decode attention works.
        # We don't need real KV data — just a nonzero length so
        # FlashInfer plan() produces valid page tables.
        engine._seq_lens_cpu[0] = 1
        engine.seq_lens[0] = 1

        torch.cuda.synchronize()
        is_timed = trial >= n_warmup

        if is_timed:
            start_evt.record()

        with torch.inference_mode():
            for step in range(n_steps):
                # Single decode token — the replay controller handles
                # all expert loading/prefetching from the trace
                token = torch.tensor([1], device=engine.device)
                engine.mixed_step([0], token, [], [])

        if is_timed:
            end_evt.record()
            torch.cuda.synchronize()
            total_ms = start_evt.elapsed_time(end_evt)
            all_times.append(total_ms)

        engine.replay_controller = None

    # Report median
    all_times.sort()
    median_ms = all_times[len(all_times) // 2]
    per_step_ms = median_ms / n_steps
    return median_ms, per_step_ms, n_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/Mixtral-8x7B")
    parser.add_argument("--activation-trace", type=str,
                        default="datasets/ShareGPT_Vicuna/expert_traces/"
                        "mixtral-8x7b/cache90pct_short/batched.json")
    parser.add_argument("--cache-size", type=int, default=200,
                        help="Total expert cache slots. Max ~226 on single "
                        "H100 (88%% of 256, overhead=2.5GB). Default 200 "
                        "(78%%) leaves room for KV cache.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Max steps to replay (default: all)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    # Generate traces
    print("=== Generating DataMovementTraces ===")
    traces, at = generate_traces(args.activation_trace, args.cache_size)

    # Create engine — minimize KV cache to leave room for expert buffers
    print(f"\n=== Loading model: {args.model} ===")
    engine = MoEEngine(args.model, cache_size=args.cache_size,
                       max_batch_size=1, max_seq_len=256,
                       use_torch_compile=False)

    # Capture piecewise CUDA graphs (needed for replay)
    print("Capturing piecewise CUDA graphs...")
    engine.capture_mixed_cuda_graphs(total_token_sizes=[1])

    # Run benchmarks
    print(f"\n=== Replay Benchmark ({args.warmup} warmup + "
          f"{args.trials} trials) ===")
    print(f"{'Policy':<25s}  {'Total ms':>10s}  {'ms/step':>8s}  "
          f"{'Steps':>5s}  {'Demands':>7s}  {'Prefetch':>8s}")
    print("-" * 75)

    for name, dm in traces.items():
        s = dm.summary()
        total_ms, per_step_ms, n_steps = run_replay(
            engine, dm, max_steps=args.steps,
            n_warmup=args.warmup, n_trials=args.trials)
        print(f"{name:<25s}  {total_ms:10.1f}  {per_step_ms:8.2f}  "
              f"{n_steps:5d}  {s['demand_loads']:7d}  {s['prefetches']:8d}")


if __name__ == "__main__":
    main()
