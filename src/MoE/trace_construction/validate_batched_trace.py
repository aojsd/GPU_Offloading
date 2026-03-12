"""Validate batched expert traces against real model routing.

Runs the actual model (PP=2) with the exact batch compositions from Phase 1
batched traces, recording expert routing at each (step, layer). Compares the
recorded routing against the trace's expected expert sets to verify that the
batch-union routing recorded by collect_batched_traces.py is faithful.

Since routing is per-token-independent (attention is per-sequence, all other ops
are per-row), validating one cache fraction proves the property for all — but
this script validates all fractions for completeness.

Usage:
    python validate_batched_trace.py \
        --model ../../models/Mixtral-8x7B \
        --trace-dir ../../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b \
        --cache-pct 60 70 80 \
        --pipeline-parallel 2
"""
import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MOE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOE_DIR))

import torch

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import moe_engine as _moe_engine_mod  # noqa: F401 — glibc patches
from moe_engine import MoEEngine
from trace_recorder import TraceRecorder
from gpu_replay_trace import ActivationTrace
from trace_utils import load_traces, ConversationTrace


def load_prompt_tokens(traces: list[ConversationTrace],
                       device: torch.device) -> dict[int, torch.Tensor]:
    """Pre-load all prompt token tensors keyed by trace index."""
    prompts = {}
    for i, t in enumerate(traces):
        if t.prompt_token_ids is not None:
            prompts[i] = torch.tensor(t.prompt_token_ids, dtype=torch.long,
                                      device=device)
        else:
            prompts[i] = torch.ones(t.prompt_tokens, dtype=torch.long,
                                    device=device)
    return prompts


def _load_batched_trace(trace_dir, cache_pct):
    """Load a batched trace and extract its scheduling requirements."""
    batched_path = os.path.join(
        trace_dir, f"cache{cache_pct}pct", "batched_trace.json")
    if not os.path.exists(batched_path):
        raise FileNotFoundError(
            f"Batched trace not found: {batched_path}\n"
            f"Run collect_batched_traces.py first (Phase 1).")
    at = ActivationTrace.load(batched_path)
    if not at.scheduling:
        raise RuntimeError(
            f"cache{cache_pct}pct/batched_trace.json has no scheduling metadata. "
            f"Re-run collect_batched_traces.py to generate step_scheduling.")

    # For validation we need peak concurrent sequences (not the simulator's
    # max_seqs=256) and enough per-slot pages for the longest actual sequence.
    # The cache-fraction budget is irrelevant here — validation runs PP=2
    # with all experts on GPU, so KV capacity depends on PP memory, not the
    # offloading budget.
    peak_seqs = max(ss.batch_size for ss in at.scheduling)
    # max_seq_len=None signals caller to use actual_max_seq from per-conv data
    return at, peak_seqs, None


def _validate_one(engine, recorder, at, per_conv_traces, prompts,
                   max_seqs, max_steps=None):
    """Validate one batched trace against real model routing.

    Returns:
        (total_steps, num_layers, mismatches, elapsed) where mismatches is a
        list of (step, layer, expected_set, actual_set) tuples.
    """
    total_steps = len(at.steps)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    engine.reset()
    recorder.reset_trace()

    free_seq_ids = set(range(max_seqs))
    request_to_slot: dict[int, int] = {}
    active_requests: dict[int, dict] = {}
    mismatches = []
    skipped_graph = 0
    t0 = time.time()

    with torch.inference_mode():
        for step in range(total_steps):
            sched = at.scheduling[step]
            recorder.reset_trace()

            # Process events: completions, admissions
            for evt in sched.events:
                if evt['event'] == 'complete':
                    rid = evt['request_id']
                    if rid in request_to_slot:
                        sid = request_to_slot.pop(rid)
                        engine.free_seq(sid)
                        free_seq_ids.add(sid)
                        active_requests.pop(rid, None)
                elif evt['event'] in ('admit', 'force_admit'):
                    rid = evt['request_id']
                    if not free_seq_ids:
                        continue
                    sid = free_seq_ids.pop()
                    request_to_slot[rid] = sid
                    active_requests[rid] = {
                        'sid': sid,
                        'decode_step': 0,
                    }

            # Build step arguments
            decode_sids = []
            decode_tokens = []
            prefill_sids = []
            prefill_tokens = []
            continuation_sids = []
            continuation_tokens = []
            continuation_offsets = []

            for ar in sched.active_requests:
                rid = ar.request_id
                if rid not in request_to_slot:
                    continue
                sid = request_to_slot[rid]
                state = active_requests[rid]

                if ar.is_prefill:
                    chunk_start = ar.prefill_chunk_start
                    chunk_len = ar.prefill_chunk_length
                    prompt = prompts[rid]
                    chunk_tokens = prompt[chunk_start:chunk_start + chunk_len]
                    if ar.is_continuation:
                        continuation_sids.append(sid)
                        continuation_tokens.append(chunk_tokens)
                        continuation_offsets.append(chunk_start)
                    else:
                        prefill_sids.append(sid)
                        prefill_tokens.append(chunk_tokens)
                else:
                    decode_sids.append(sid)
                    trace = per_conv_traces[rid]
                    decode_idx = state['decode_step']
                    if (trace.output_token_ids is not None
                            and decode_idx < len(trace.output_token_ids)):
                        decode_tokens.append(trace.output_token_ids[decode_idx])
                    else:
                        decode_tokens.append(1)
                    state['decode_step'] = decode_idx + 1

            if decode_tokens:
                decode_tensor = torch.tensor(
                    decode_tokens, dtype=torch.long, device=engine.device)
            else:
                decode_tensor = torch.tensor(
                    [], dtype=torch.long, device=engine.device)

            try:
                engine.step(
                    decode_seq_ids=decode_sids,
                    decode_token_ids=decode_tensor,
                    prefill_seq_ids=prefill_sids,
                    prefill_input_ids=prefill_tokens,
                    continuation_seq_ids=continuation_sids,
                    continuation_input_ids=continuation_tokens,
                    continuation_offsets=continuation_offsets,
                )
            except (RuntimeError, NotImplementedError) as e:
                msg = str(e)
                if ("No piecewise CUDA graph covers" in msg
                        or "requires piecewise CUDA graphs" in msg):
                    skipped_graph += 1
                    continue
                raise

            # Compare recorded routing vs expected
            for entry in recorder.trace:
                layer = entry['layer']
                actual = set(entry['expert_ids'])
                expected = set(at.steps[step][layer])
                if actual != expected:
                    mismatches.append((step, layer, expected, actual))

            if (step + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (step + 1) / elapsed
                print(f"    Step {step+1}/{total_steps} "
                      f"({rate:.0f} steps/s, "
                      f"{len(mismatches)} mismatches, "
                      f"{skipped_graph} skipped)")

    elapsed = time.time() - t0
    return total_steps, at.num_layers, mismatches, skipped_graph, elapsed


def _report_mismatches(cache_pct, total_steps, num_layers, mismatches,
                       skipped_graph, elapsed):
    """Print mismatch report for one cache fraction. Returns True if clean."""
    n_mismatch = len(mismatches)
    validated_steps = total_steps - skipped_graph
    total_checks = validated_steps * num_layers
    print(f"  cache{cache_pct}%: {validated_steps}/{total_steps} steps "
          f"× {num_layers} layers = {total_checks} checks in {elapsed:.1f}s — "
          f"{n_mismatch} mismatches"
          + (f" ({skipped_graph} steps skipped: no graph coverage)"
             if skipped_graph else ""))

    if not mismatches:
        return True

    # First 10 mismatches
    for step, layer, expected, actual in mismatches[:10]:
        extra = actual - expected
        missing = expected - actual
        parts = [f"    Step {step}, layer {layer}:"]
        if missing:
            parts.append(f"missing={sorted(missing)}")
        if extra:
            parts.append(f"extra={sorted(extra)}")
        print(" ".join(parts))
    if n_mismatch > 10:
        print(f"    ... and {n_mismatch - 10} more")

    # Per-layer summary
    layer_counts = Counter(layer for _, layer, _, _ in mismatches)
    print(f"    Per-layer: " + ", ".join(
        f"L{l}={c}" for l, c in sorted(layer_counts.items())))
    print(f"    Overall: {n_mismatch}/{total_checks} "
          f"({n_mismatch/total_checks*100:.2f}%)")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate batched trace routing against real model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--trace-dir", type=str, required=True,
                        help="Trace directory (contains manifest.json + cache*pct/)")
    parser.add_argument("--cache-pct", type=int, nargs='+', required=True,
                        help="Cache percentages to validate (e.g., 50 60 70 80 85)")
    parser.add_argument("--pipeline-parallel", "-pp", type=int, default=2,
                        help="Number of GPUs for pipeline parallelism (default: 2)")
    parser.add_argument("--max-graph-size", type=int, default=512,
                        help="Max CUDA graph token size (default: 512)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Validate only first N steps per trace (default: all)")
    args = parser.parse_args()

    # Load per-conversation traces (shared across all cache fractions)
    per_conv_traces, manifest = load_traces(args.trace_dir)
    print(f"Loaded {len(per_conv_traces)} conversation traces")
    actual_max_seq = max(t.prompt_tokens + t.output_tokens
                         for t in per_conv_traces)

    # Pre-scan all batched traces to find max engine requirements
    print(f"\nScanning {len(args.cache_pct)} batched traces...")
    trace_info = {}  # cache_pct -> (at, max_seqs, max_seq_len)
    global_max_seqs = 0
    global_max_seq_len = 0
    for pct in args.cache_pct:
        at, max_seqs, max_seq_len = _load_batched_trace(args.trace_dir, pct)
        if max_seq_len is None:
            max_seq_len = actual_max_seq
        trace_info[pct] = (at, max_seqs, max_seq_len)
        global_max_seqs = max(global_max_seqs, max_seqs)
        global_max_seq_len = max(global_max_seq_len, max_seq_len)
        print(f"  cache{pct}%: {len(at.steps)} steps, "
              f"max_seqs={max_seqs}, max_seq_len={max_seq_len}")

    print(f"\nEngine requirements: max_seqs={global_max_seqs}, "
          f"max_seq_len={global_max_seq_len}")

    # Create PP engine once (all experts on GPU, no offloading)
    engine = MoEEngine(
        args.model,
        max_seqs=global_max_seqs,
        max_seq_len=global_max_seq_len,
        pipeline_parallel_size=args.pipeline_parallel,
        use_torch_compile=True,
    )

    recorder = TraceRecorder(
        num_layers=per_conv_traces[0].num_layers,
        num_experts=per_conv_traces[0].num_experts)
    engine.trace_recorder = recorder

    # Capture CUDA graphs — same compact set as batched_replay.py
    graph_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224,
                   256, 288, 320, 352, 384, 448, 512]
    graph_sizes = [s for s in graph_sizes if s <= args.max_graph_size]
    print(f"Capturing piecewise CUDA graphs ({len(graph_sizes)} sizes)...")
    for gs in graph_sizes:
        try:
            engine.capture_cuda_graphs(total_token_sizes=[gs])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM at N={gs} — stopped")
            break

    # Pre-load prompt tokens
    prompts = load_prompt_tokens(per_conv_traces, engine.device)

    # Validate each cache fraction
    all_passed = True
    for pct in args.cache_pct:
        at, max_seqs, _ = trace_info[pct]
        print(f"\n--- Validating cache{pct}% ({len(at.steps)} steps) ---")
        total_steps, num_layers, mismatches, skipped_graph, elapsed = \
            _validate_one(engine, recorder, at, per_conv_traces, prompts,
                          global_max_seqs, args.max_steps)
        passed = _report_mismatches(pct, total_steps, num_layers,
                                    mismatches, skipped_graph, elapsed)
        if not passed:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print(f"All {len(args.cache_pct)} batched traces validated. "
              f"Expert routing is faithful.")
        sys.exit(0)
    else:
        print(f"FAILED: routing mismatches detected. "
              f"Batched traces may not faithfully represent model behavior.")
        sys.exit(1)


if __name__ == "__main__":
    main()
