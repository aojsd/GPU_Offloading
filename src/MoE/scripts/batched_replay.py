"""Batched GPU replay with multi-sequence scheduling and expert offloading.

Step 03 of the trace pipeline: replay a GPUReplayTrace on real GPU hardware
with all batched requests running real computation. Loads precomputed policy
traces from step 02 (run_all_policies.py) when available, falling back to
on-the-fly simulation. Consumes step_scheduling metadata to manage KV cache
slots, prefill/decode/continuation dispatch, and request lifecycle.

Usage:
    # Run replay for a specific cache% with all policies:
    python scripts/batched_replay.py \
        --model models/Mixtral-8x7B \
        --trace-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache70pct \
        --cache-size 179

    # Single policy:
    python scripts/batched_replay.py \
        --model models/Mixtral-8x7B \
        --trace-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache70pct \
        --cache-size 179 \
        --policies LRU-Oracle
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
sys.path.insert(0, str(MOE_DIR / 'trace_construction'))

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import moe_engine as _moe_engine_mod  # noqa: F401 — glibc patches
from moe_engine import MoEEngine
from replay_controller import ReplayController
from gpu_replay_trace import ActivationTrace, GPUReplayTrace
from build_trace import load_traces, ConversationTrace
from policy_simulator import (
    LRU, LFU, Belady, StaticFreq, NoPrefetch, OraclePrefetch, simulate,
)
from scheduler import Scheduler


CACHE_POLICIES = [
    ("LRU", lambda: LRU()),
    ("LFU", lambda: LFU()),
    ("Belady", lambda: Belady()),
    ("StaticFreq", lambda: StaticFreq()),
]

PREFETCH_POLICIES = [
    ("None", lambda: NoPrefetch()),
    ("Oracle", lambda: OraclePrefetch()),
    ("Oracle(1)", lambda: OraclePrefetch(max_per_layer=1)),
]


def replay_trace(engine, dm_trace, per_conv_traces, full_token_seqs,
                 max_graph_size, warmup_steps=50, max_steps=None):
    """Run a full batched replay of a GPUReplayTrace.

    Args:
        engine: MoEEngine with cache_size set and graphs captured.
        dm_trace: GPUReplayTrace from policy simulation.
        per_conv_traces: List of ConversationTrace objects.
        full_token_seqs: Dict[int, Tensor] of full token sequences (unused,
            kept for backward compat — Scheduler builds its own).
        max_graph_size: Max total tokens per step.
        warmup_steps: Number of untimed warmup steps.

    Returns:
        Dict with timing results.
    """
    total_steps = len(dm_trace.steps)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    max_seqs = engine.max_seqs
    page_size = getattr(engine, 'page_size', 16)

    # Build conversations list for Scheduler.replay()
    conversations = [
        {'request_idx': i,
         'prompt_token_ids': list(t.prompt_token_ids or []),
         'output_token_ids': list(t.output_token_ids or [])}
        for i, t in enumerate(per_conv_traces)
    ]

    sched = Scheduler(engine, max_seqs, max_graph_size, page_size)

    # --- Warmup pass: replay first N steps with actual trace data ---
    n_warmup = min(warmup_steps, total_steps)
    warmup_ctrl = ReplayController(engine, dm_trace)
    sched.replay(conversations, controller=warmup_ctrl,
                 n_steps=n_warmup, record_tokens=False)

    # --- Timed replay: full run from step 0 with fresh controller ---
    engine.reset()
    controller = ReplayController(engine, dm_trace, track_phases=True)

    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()

    replay_result = sched.replay(
        conversations, controller=controller,
        n_steps=total_steps, record_tokens=False)

    end_evt.record()
    torch.cuda.synchronize()
    total_ms = start_evt.elapsed_time(end_evt)

    if replay_result.skipped_admissions > 0:
        import warnings
        warnings.warn(
            f"Skipped {replay_result.skipped_admissions} admissions due to "
            f"max_seqs={max_seqs} < trace demand. Results may be invalid.")

    # Extract per-phase timing if available
    result = {
        'total_ms': total_ms,
        'ms_per_step': total_ms / total_steps,
        'steps': total_steps,
        'skipped_admissions': replay_result.skipped_admissions,
    }
    phases = controller.get_phase_times_ms()
    if phases:
        result['phases'] = phases
        io_ms = phases.get('io', 0.0)
        result['io_ms'] = io_ms
        result['compute_pct'] = (
            (total_ms - io_ms) / total_ms * 100 if total_ms > 0 else 0)

    return result


def run_batched_replay(model_path, batched_trace_path, per_conv_traces,
                       cache_size, max_graph_size=512, warmup_steps=50,
                       policies=None, max_seqs_override=None,
                       max_steps=None, trace_dir=None):
    """Run GPU replay for all policies on a batched trace.

    Args:
        model_path: Path to model weights.
        batched_trace_path: Path to batched ActivationTrace JSON.
        per_conv_traces: List of ConversationTrace objects.
        cache_size: Number of expert cache slots.
        max_graph_size: Max CUDA graph size.
        warmup_steps: Steps for CUDA warmup.
        policies: List of (name, cache_policy_fn, prefetch_policy_fn) tuples.
            If None, runs all cache x prefetch combinations.
        trace_dir: Directory containing precomputed GPUReplayTrace files
            from run_all_policies.py (e.g. LRU-Oracle.json). If a file
            exists for a policy, it is loaded instead of re-simulating.

    Returns:
        Dict of results keyed by policy name.
    """
    # Load batched activation trace
    at = ActivationTrace.load(batched_trace_path)
    total_steps = len(at.steps)

    # CUDA graph sizes — compact set to save GPU memory
    graph_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224,
                   256, 288, 320, 352, 384, 448, 512]
    graph_sizes = [s for s in graph_sizes if s <= max_graph_size]

    # Read memory budget from trace metadata (set by build_trace.py's
    # simulate_batch). This ensures replay uses exactly the same max_seqs
    # and KV budget as the batch simulator, avoiding silent batch shrinkage.
    trace_sched = at.scheduling_config or {}
    trace_max_seqs = trace_sched.get('max_seqs')
    trace_kv_budget = trace_sched.get('kv_page_budget')
    trace_page_size = trace_sched.get('page_size', 16)

    trace_peak_seqs = max(
        ss.batch_size
        for ss in (at.scheduling or [])
    ) if at.scheduling else 256

    if trace_max_seqs is not None:
        max_seqs = trace_max_seqs
    else:
        max_seqs = trace_peak_seqs
    if max_seqs_override is not None:
        max_seqs = min(max_seqs, max_seqs_override)

    # Compute max_seq_len: actual max from conversations, capped by KV budget.
    # With dynamic pages, one sequence can use all pages (preemption frees others).
    _actual_max = max(t.prompt_tokens + t.output_tokens
                      for t in per_conv_traces)
    if trace_kv_budget and trace_kv_budget > 0:
        max_seq_len = min(_actual_max, trace_kv_budget * trace_page_size)
    else:
        max_seq_len = _actual_max

    # Hard error on batch shrinkage — do not silently degrade
    if max_seqs < trace_peak_seqs:
        raise RuntimeError(
            f"max_seqs={max_seqs} < trace peak batch size {trace_peak_seqs}. "
            f"Cannot faithfully replay. Reduce cache_size or use --max-seqs "
            f"to explicitly accept smaller batches."
        )

    print(f"Memory budget from trace: max_seqs={max_seqs} "
          f"(trace peak: {trace_peak_seqs}), max_seq_len={max_seq_len}, "
          f"kv_budget={trace_kv_budget} pages")

    # Generate policy traces
    if policies is None:
        policies = [
            (f"{cp_name}-{pf_name}", cp_fn, pf_fn)
            for cp_name, cp_fn in CACHE_POLICIES
            for pf_name, pf_fn in PREFETCH_POLICIES
        ]

    print(f"\n=== Loading/generating policy traces (CS={cache_size}) ===")
    dm_traces = {}
    for name, cp_fn, pf_fn in policies:
        precomputed = (os.path.join(trace_dir, f"{name}.json")
                       if trace_dir else None)
        if precomputed and os.path.exists(precomputed):
            dm = GPUReplayTrace.load(precomputed)
            s = dm.summary()
            print(f"  {name:<18s}  demands={s['demand_loads']:6d}  "
                  f"prefetches={s['prefetches']:6d}  total={s['total_transfers']:6d}"
                  f"  [loaded]")
        else:
            dm = simulate(cp_fn(), pf_fn(), at, cache_size=cache_size)
            s = dm.summary()
            print(f"  {name:<18s}  demands={s['demand_loads']:6d}  "
                  f"prefetches={s['prefetches']:6d}  total={s['total_transfers']:6d}"
                  f"  [simulated]")
        dm_traces[name] = dm

    # Load engine — use dynamic KV pages alongside expert cache to avoid
    # over-allocating KV for long sequences (greedy admission + preemption
    # traces have sequences longer than kv_budget // max_seqs).
    kv_page_budget = trace_kv_budget if (trace_kv_budget and trace_kv_budget > 0) else None
    print(f"\n=== Loading model: {model_path} (cache_size={cache_size}, "
          f"max_seqs={max_seqs}, kv_budget={kv_page_budget}) ===")
    engine = MoEEngine(
        model_path,
        cache_size=cache_size,
        max_seqs=max_seqs,
        max_seq_len=max_seq_len,
        kv_page_budget=kv_page_budget,
        use_torch_compile=True,
    )

    # Capture CUDA graphs — one size at a time, stop on OOM
    print(f"Capturing piecewise CUDA graphs for {len(graph_sizes)} sizes "
          f"({graph_sizes})...")
    captured_sizes = []
    for gs in graph_sizes:
        try:
            engine.capture_mixed_cuda_graphs(total_token_sizes=[gs])
            captured_sizes.append(gs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM at N={gs} — stopping graph capture "
                  f"(captured {len(captured_sizes)} sizes: {captured_sizes})")
            break
    if not captured_sizes:
        raise RuntimeError("Failed to capture any CUDA graphs — reduce cache_size or max_seqs")

    graph_overhead_gb = engine.graph_memory_overhead_bytes() / 1024**3
    print(f"Graph memory overhead: {graph_overhead_gb:.2f} GB "
          f"({len(captured_sizes)} sizes captured)")

    # Validate graph coverage: verify captured graphs cover all trace steps
    max_step_tokens = 0
    for ss in (at.scheduling or []):
        n_tokens = sum(
            ar.prefill_chunk_length if ar.is_prefill else 1
            for ar in ss.active_requests
        )
        max_step_tokens = max(max_step_tokens, n_tokens)

    if max_step_tokens > max(captured_sizes):
        raise RuntimeError(
            f"Largest captured graph ({max(captured_sizes)} tokens) < "
            f"max step tokens ({max_step_tokens}). OOM stopped graph "
            f"capture too early. Reduce cache_size, max_seqs, or "
            f"max_graph_size.")

    # Coverage histogram: how many steps use each graph size
    from collections import Counter
    _size_usage = Counter()
    for ss in (at.scheduling or []):
        n_tok = sum(ar.prefill_chunk_length if ar.is_prefill else 1
                    for ar in ss.active_requests)
        covering = min((s for s in captured_sizes if s >= n_tok),
                       default=None)
        if covering is not None:
            _size_usage[covering] += 1
    print(f"Graph coverage: {dict(sorted(_size_usage.items()))}")

    # Replay each policy
    print(f"\n=== Replay ({warmup_steps}-step warmup + {total_steps}-step "
          f"timed run) ===")
    print(f"{'Policy':<20s}  {'Total ms':>10s}  {'ms/step':>8s}  "
          f"{'Steps':>5s}  {'Demands':>7s}  {'Prefetch':>8s}")
    print("-" * 70)

    results = {}
    for name, dm in dm_traces.items():
        s = dm.summary()
        timing = replay_trace(
            engine, dm, per_conv_traces, None,
            max_graph_size, warmup_steps, max_steps,
        )
        print(f"{name:<20s}  {timing['total_ms']:10.1f}  "
              f"{timing['ms_per_step']:8.2f}  {timing['steps']:5d}  "
              f"{s['demand_loads']:7d}  {s['prefetches']:8d}",
              flush=True)
        results[name] = {
            **timing,
            'demands': s['demand_loads'],
            'prefetches': s['prefetches'],
            'total_transfers': s['total_transfers'],
        }

    # Cleanup
    del engine
    torch.cuda.empty_cache()

    return results


def _parse_policy(name):
    """Parse 'CachePolicy-PrefetchPolicy' into (name, cp_fn, pf_fn)."""
    cp_map = {n: fn for n, fn in CACHE_POLICIES}
    pf_map = {n: fn for n, fn in PREFETCH_POLICIES}
    # Split on first '-' so 'Oracle(1)' stays intact
    parts = name.split('-', 1)
    if len(parts) != 2:
        raise ValueError(f"Policy must be CachePolicy-PrefetchPolicy, got: {name}")
    cp_name, pf_name = parts
    if cp_name not in cp_map:
        raise ValueError(f"Unknown cache policy: {cp_name}. "
                         f"Options: {list(cp_map.keys())}")
    if pf_name not in pf_map:
        raise ValueError(f"Unknown prefetch policy: {pf_name}. "
                         f"Options: {list(pf_map.keys())}")
    return (name, cp_map[cp_name], pf_map[pf_name])


def _append_result_to_md(result_dict, results_md_path):
    """Append a timing row to results.md with file locking."""
    import fcntl

    table_header = (
        "### GPU Replay: Wall-Clock Timing (All Policies)\n\n"
        "| Cache% | Policy | ms/step | Compute% | Demands | Prefetches |\n"
        "|--------|--------|---------|----------|---------|------------|\n"
    )

    row = (
        f"| {result_dict['cache_pct']}%    "
        f"| {result_dict['policy']:<20s} "
        f"| {result_dict['ms_per_step']:>7.2f} "
        f"| {result_dict.get('compute_pct', 0):>6.1f}% "
        f"| {result_dict.get('demands', 0):>7d} "
        f"| {result_dict.get('prefetches', 0):>10d} |\n"
    )

    os.makedirs(os.path.dirname(results_md_path), exist_ok=True)

    with open(results_md_path, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            if "### GPU Replay: Wall-Clock Timing" not in content:
                f.write("\n" + table_header)
            f.write(row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_trace_cache_size(trace_path):
    """Read cache_size from a batched trace's scheduling config."""
    with open(trace_path) as f:
        d = json.load(f)
    cs = d.get('scheduling', {}).get('cache_size')
    if cs is None:
        raise ValueError(
            f"cache_size not found in {trace_path} scheduling config. "
            f"Re-run collect_batched_traces.py with --cache-fraction.")
    return cs


def main():
    parser = argparse.ArgumentParser(
        description="Batched GPU replay with multi-sequence scheduling")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--trace-dir", type=str, required=True,
                        help="Directory with batched_trace.json and "
                             "requests/ (a cache%%pct dir)")
    parser.add_argument("--batched-trace", type=str, default=None,
                        help="Path to batched ActivationTrace JSON "
                             "(default: <trace-dir>/batched_trace.json)")
    parser.add_argument("--cache-size", type=int, default=None,
                        help="Override cache size (default: read from trace)")
    parser.add_argument("--max-seqs", type=int, default=256)
    parser.add_argument("--max-graph-size", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max replay steps (default: all)")
    parser.add_argument("--policy", type=str, default=None,
                        help="Single policy (e.g. LRU-Oracle). Default: all.")
    parser.add_argument("--policies", type=str, default=None,
                        help="Comma-separated policy names "
                             "(e.g. LRU-None,Belady-Oracle(1))")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Dir for per-policy result JSONs (default: "
                             "results/MoE/mixtral-8x7B/tmp)")
    parser.add_argument("--results-md", type=str, default=None,
                        help="Path to results.md for incremental appending")
    args = parser.parse_args()

    if args.policy and args.policies:
        parser.error("--policy and --policies are mutually exclusive")

    # Locate batched trace
    if args.batched_trace:
        batched_trace_path = args.batched_trace
    else:
        batched_trace_path = os.path.join(args.trace_dir, "batched_trace.json")
    if not os.path.exists(batched_trace_path):
        parser.error(f"Batched trace not found: {batched_trace_path}")

    # Read cache_size from trace metadata (or use override)
    if args.cache_size is not None:
        cache_size = args.cache_size
    else:
        cache_size = _read_trace_cache_size(batched_trace_path)

    # Infer pct from directory name (e.g. cache60pct)
    trace_dir_name = os.path.basename(os.path.normpath(args.trace_dir))
    pct = None
    if trace_dir_name.startswith("cache") and trace_dir_name.endswith("pct"):
        try:
            pct = int(trace_dir_name[5:-3])
        except ValueError:
            pass

    max_graph_size = args.max_graph_size

    # Load per-conversation traces
    per_conv_traces, manifest = load_traces(args.trace_dir)
    print(f"Loaded {len(per_conv_traces)} per-conversation traces")

    # Parse policies
    policies = None
    if args.policies:
        policies = [_parse_policy(name) for name in args.policies.split(',')]
    elif args.policy:
        policies = [_parse_policy(args.policy)]

    # Determine output directory
    moe_root = Path(__file__).resolve().parent.parent   # src/MoE
    repo_root = moe_root.parent.parent                  # GPU_Offloading
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / "results" / "MoE" / "mixtral-8x7B" / "tmp"

    # Determine results.md path
    if args.results_md:
        results_md_path = args.results_md
    else:
        results_md_path = str(
            repo_root / "results" / "MoE" / "mixtral-8x7B" / "results.md")

    results = run_batched_replay(
        args.model, batched_trace_path, per_conv_traces,
        cache_size, max_graph_size, args.warmup_steps,
        policies, max_seqs_override=args.max_seqs,
        max_steps=args.max_steps,
        trace_dir=args.trace_dir,
    )

    # Save per-policy results
    os.makedirs(str(output_dir), exist_ok=True)
    for policy_name, policy_result in results.items():
        # Enrich with metadata
        policy_result['policy'] = policy_name
        if pct is not None:
            policy_result['cache_pct'] = pct
        policy_result['cache_size'] = cache_size

        # Save per-policy JSON
        if pct is not None:
            out_file = output_dir / f"cache{pct}pct-{policy_name}.json"
        else:
            out_file = output_dir / f"cache{cache_size}-{policy_name}.json"
        with open(str(out_file), 'w') as f:
            json.dump(policy_result, f, indent=2)
        print(f"  Saved: {out_file}")

        # Append to results.md incrementally
        if pct is not None:
            _append_result_to_md(policy_result, results_md_path)

    # Also save combined results if --output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nCombined results saved to {args.output}")


if __name__ == "__main__":
    main()
