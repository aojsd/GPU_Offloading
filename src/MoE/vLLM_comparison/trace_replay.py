"""Trace-and-replay benchmark: run vLLM, trace batches, replay on custom engine.

Runs a workload on vLLM using step-by-step API, capturing per-step batch
composition (which requests are prefill vs decode, token counts, context lengths).
Then replays those exact batches on the custom MoE engine and compares latency
and correctness.

Usage:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python vLLM_comparison/trace_replay.py
    ... --workload staggered       # specific workload
    ... --skip-custom              # trace vLLM only
    ... --skip-vllm                # replay from saved trace
"""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MOE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOE_DIR))

import torch

# Apply glibc 2.28 monkey patches (must happen before vLLM import)
import moe_engine as _moe_engine_mod  # noqa: F401
from moe_engine import MoEEngine

DEFAULT_MODEL = str(MOE_DIR / "models" / "OLMoE-1B-7B")

# ── Trace data structures ────────────────────────────────────────────

@dataclass
class DecodeInfo:
    request_id: str
    context_length: int     # seq_len BEFORE this step's token
    token_id: int           # token being decoded

@dataclass
class PrefillInfo:
    request_id: str
    prompt_token_ids: list  # full prompt tokens
    num_scheduled_tokens: int  # tokens computed THIS step

@dataclass
class StepTrace:
    step_idx: int
    decode_requests: list       # list of DecodeInfo dicts
    prefill_requests: list      # list of PrefillInfo dicts
    total_tokens: int
    vllm_latency_ms: float
    vllm_output_tokens: dict = field(default_factory=dict)  # req_id -> token_id produced

@dataclass
class WorkloadRequest:
    request_id: str
    arrival_step: int
    prompt_token_ids: list
    max_new_tokens: int


# ── Workload definitions ─────────────────────────────────────────────

def _rand_prompt(length, seed=None):
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        return torch.randint(1, 1000, (length,), generator=g).tolist()
    return torch.randint(1, 1000, (length,)).tolist()


def make_workload(name):
    """Create a list of WorkloadRequest for the named workload."""
    if name == "single":
        return [WorkloadRequest("r0", 0, _rand_prompt(128, seed=42), 50)]

    elif name == "batch_4":
        return [
            WorkloadRequest(f"r{i}", 0, _rand_prompt(128, seed=100+i), 30)
            for i in range(4)
        ]

    elif name == "batch_8":
        return [
            WorkloadRequest(f"r{i}", 0, _rand_prompt(128, seed=200+i), 30)
            for i in range(8)
        ]

    elif name == "staggered":
        # 4 arrive at step 0, 2 at step 10, 2 at step 20
        reqs = []
        for i in range(4):
            reqs.append(WorkloadRequest(
                f"r{i}", 0, _rand_prompt(128, seed=300+i), 40))
        for i in range(2):
            reqs.append(WorkloadRequest(
                f"r{4+i}", 10, _rand_prompt(64, seed=310+i), 30))
        for i in range(2):
            reqs.append(WorkloadRequest(
                f"r{6+i}", 20, _rand_prompt(256, seed=320+i), 20))
        return reqs

    elif name == "continuous":
        # 1 new request every 5 steps, up to 12 requests
        return [
            WorkloadRequest(f"r{i}", i * 5,
                            _rand_prompt(128, seed=400+i), 30)
            for i in range(12)
        ]

    else:
        raise ValueError(f"Unknown workload: {name}")


WORKLOAD_NAMES = ["single", "batch_4", "batch_8", "staggered", "continuous"]


# ── vLLM Tracing ─────────────────────────────────────────────────────

def trace_vllm(model_path, workload, n_warmup_requests=3):
    """Run workload on vLLM, trace every step's batch composition.

    Returns list of StepTrace.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.50,
        disable_log_stats=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )

    # Warmup
    for i in range(n_warmup_requests):
        sp = SamplingParams(max_tokens=1, temperature=0)
        warm_ids = torch.randint(1, 1000, (128,)).tolist()
        llm.llm_engine.add_request(
            request_id=f"warmup_{i}",
            prompt={"prompt_token_ids": warm_ids},
            params=sp,
        )
        while llm.llm_engine.has_unfinished_requests():
            llm.llm_engine.step()
    torch.cuda.synchronize()
    print("  vLLM warmup complete")

    # Build request lookup
    req_lookup = {r.request_id: r for r in workload}
    pending = list(workload)  # requests not yet submitted
    active_reqs = {}  # req_id -> {prompt_token_ids, num_computed, output_tokens}
    finished_reqs = set()

    traces = []
    step_idx = 0
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    while pending or llm.llm_engine.has_unfinished_requests():
        # Submit requests arriving at this step
        arriving = [r for r in pending if r.arrival_step <= step_idx]
        for r in arriving:
            sp = SamplingParams(max_tokens=r.max_new_tokens, temperature=0)
            llm.llm_engine.add_request(
                request_id=r.request_id,
                prompt={"prompt_token_ids": r.prompt_token_ids},
                params=sp,
            )
            active_reqs[r.request_id] = {
                'prompt_token_ids': r.prompt_token_ids,
                'num_computed': 0,
                'output_tokens': [],
                'prompt_len': len(r.prompt_token_ids),
            }
        pending = [r for r in pending if r.arrival_step > step_idx]

        if not llm.llm_engine.has_unfinished_requests():
            step_idx += 1
            continue

        # Execute one step and time it
        start_evt.record()
        step_outputs = llm.llm_engine.step()
        end_evt.record()
        torch.cuda.synchronize()
        latency_ms = start_evt.elapsed_time(end_evt)

        # Classify requests in this step
        decode_infos = []
        prefill_infos = []
        output_tokens = {}

        # Determine which requests were active this step by checking
        # what tokens they produced
        for out in step_outputs:
            rid = out.request_id
            if rid.startswith("warmup_") or rid not in active_reqs:
                continue

            info = active_reqs[rid]
            new_token_ids = list(out.outputs[0].token_ids)

            # How many output tokens total now
            prev_output_len = len(info['output_tokens'])
            curr_output_len = len(new_token_ids)

            if prev_output_len == 0 and info['num_computed'] == 0:
                # This was a prefill step — prompt was just computed
                prefill_infos.append(PrefillInfo(
                    request_id=rid,
                    prompt_token_ids=info['prompt_token_ids'],
                    num_scheduled_tokens=info['prompt_len'],
                ))
                info['num_computed'] = info['prompt_len']

                # First output token (from prefill)
                if curr_output_len > 0:
                    info['output_tokens'] = list(new_token_ids)
                    output_tokens[rid] = new_token_ids[-1]

            elif curr_output_len > prev_output_len:
                # Decode step — produced new token(s)
                new_tok = new_token_ids[prev_output_len]
                context_len = info['prompt_len'] + prev_output_len

                # The token being decoded is the PREVIOUS output token
                # (or prompt's implied next token for the first decode)
                if prev_output_len > 0:
                    decode_token = info['output_tokens'][-1]
                else:
                    # First decode after prefill — token is the one
                    # generated by prefill (already recorded)
                    decode_token = new_token_ids[0] if new_token_ids else 0

                decode_infos.append(DecodeInfo(
                    request_id=rid,
                    context_length=context_len,
                    token_id=decode_token,
                ))
                info['output_tokens'] = list(new_token_ids)
                info['num_computed'] = info['prompt_len'] + curr_output_len
                output_tokens[rid] = new_tok

            # Check if finished
            if out.finished:
                finished_reqs.add(rid)

        total_tokens = (len(decode_infos)
                        + sum(p.num_scheduled_tokens for p in prefill_infos))

        traces.append(StepTrace(
            step_idx=step_idx,
            decode_requests=[asdict(d) for d in decode_infos],
            prefill_requests=[asdict(p) for p in prefill_infos],
            total_tokens=total_tokens,
            vllm_latency_ms=latency_ms,
            vllm_output_tokens=output_tokens,
        ))

        # Clean up finished
        for rid in finished_reqs:
            if rid in active_reqs:
                del active_reqs[rid]
        finished_reqs.clear()

        step_idx += 1

    # Clean up vLLM to free GPU memory
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # Destroy NCCL process group if it exists
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    return traces


# ── Custom Engine Replay ─────────────────────────────────────────────

def replay_on_custom(engine, traces, n_warmup=3, n_trials=5,
                     use_compile=True):
    """Replay a vLLM trace on the custom engine.

    Uses CUDA-graphed decode_step/prefill for pure steps, mixed_step for
    actual mixed batches. This ensures fair comparison with vLLM which also
    uses CUDA graphs.

    Returns (per_step_latencies, per_step_correctness).
    per_step_latencies[i] = median ms for step i
    per_step_correctness[i] = {req_id: (custom_token, vllm_token, match)}
    """
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # Determine total sequence slots needed
    all_req_ids = set()
    for step in traces:
        for d in step.decode_requests:
            all_req_ids.add(d['request_id'])
        for p in step.prefill_requests:
            all_req_ids.add(p['request_id'])
    req_to_slot = {rid: i for i, rid in enumerate(sorted(all_req_ids))}
    n_slots = len(all_req_ids)

    # Capture CUDA graphs for decode (all batch sizes seen in trace)
    decode_batch_sizes = set()
    # Collect total token sizes for pure-prefill steps (flat keying)
    prefill_total_sizes = set()
    # Collect total token sizes for mixed steps (piecewise graphs)
    mixed_total_sizes = set()
    for step in traces:
        nd = len(step.decode_requests)
        np_ = len(step.prefill_requests)
        total = step.total_tokens
        if nd > 0 and np_ == 0:
            decode_batch_sizes.add(nd)
        elif nd == 0 and np_ > 0:
            t = sum(p['num_scheduled_tokens']
                    for p in step.prefill_requests)
            prefill_total_sizes.add(t)
        else:
            # Mixed step — needs piecewise graph keyed by total tokens
            mixed_total_sizes.add(total)

    # Also capture piecewise graphs for scattered-decode steps
    # (decode_step requires contiguous [:B]; scattered slots fall through
    # to mixed_step which will use piecewise if available)
    for step in traces:
        nd = len(step.decode_requests)
        np_ = len(step.prefill_requests)
        if nd > 0 and np_ == 0:
            # Check if slots are contiguous — if not, mixed_step handles it
            # We'll just ensure piecewise covers all decode-only totals too
            mixed_total_sizes.add(nd)

    compile_str = " + torch.compile" if use_compile else ""
    # Capture CUDA graphs — use inference_mode to handle tensors from
    # vLLM's cleanup (may be left in inference-mode state)
    with torch.inference_mode():
        # Capture prefill graphs first (decode capture uses mixed_step for warmup)
        if prefill_total_sizes:
            engine.reset()
            engine.capture_prefill_cuda_graph(
                total_token_sizes=sorted(prefill_total_sizes),
                use_torch_compile=use_compile)
            print(f"  Captured prefill CUDA graphs{compile_str} for "
                  f"total_tokens: {sorted(prefill_total_sizes)}")

        for bs in sorted(decode_batch_sizes):
            engine.reset()
            engine.capture_decode_cuda_graph(
                batch_size=bs, warmup_seq_len=128,
                max_decode_tokens=engine.max_seq_len - 128,
                use_torch_compile=use_compile)
        print(f"  Captured decode CUDA graphs{compile_str} for batch sizes: "
              f"{sorted(decode_batch_sizes)}")

        # Capture piecewise graphs for mixed steps
        if mixed_total_sizes:
            engine.reset()
            engine.capture_mixed_cuda_graphs(
                total_token_sizes=sorted(mixed_total_sizes),
                use_torch_compile=use_compile)
            print(f"  Captured piecewise CUDA graphs{compile_str} for "
                  f"total_tokens: {sorted(mixed_total_sizes)}")

    # We'll run the trace multiple times for reliable timing
    all_latencies = [[] for _ in traces]
    correctness = [None] * len(traces)

    # Use inference_mode for entire replay — CUDA graph static buffers
    # created during capture are inference tensors, must be updated
    # inside inference_mode
    inference_ctx = torch.inference_mode()
    inference_ctx.__enter__()

    for trial in range(n_warmup + n_trials):
        engine.reset()
        is_timed = trial >= n_warmup

        for step_i, step in enumerate(traces):
            nd = len(step.decode_requests)
            np_ = len(step.prefill_requests)

            # Build common data
            decode_seq_ids = [req_to_slot[d['request_id']]
                              for d in step.decode_requests]
            prefill_seq_ids = [req_to_slot[p['request_id']]
                               for p in step.prefill_requests]

            if is_timed:
                start_evt.record()

            if nd > 0 and np_ == 0:
                # Pure decode — use existing CUDA-graphed decode_step
                # Need contiguous batch indices 0..D-1 for decode_step
                # Remap to slot positions
                tokens = torch.tensor(
                    [d['token_id'] for d in step.decode_requests],
                    device="cuda")
                positions = torch.tensor(
                    [engine._seq_lens_cpu[sid].item() for sid in decode_seq_ids],
                    dtype=torch.int32, device="cuda")

                # decode_step expects contiguous slots [:B], but our slots
                # may be scattered. Use mixed_step for scattered slots.
                if decode_seq_ids == list(range(nd)):
                    logits = engine.decode_step(tokens, positions)
                    # logits is [D, vocab]
                else:
                    decode_tokens = tokens
                    logits = engine.mixed_step(
                        decode_seq_ids, decode_tokens, [], [])

            elif nd == 0 and np_ == 1:
                # Pure single prefill — use prefill_to_slot
                p = step.prefill_requests[0]
                sid = prefill_seq_ids[0]
                toks = p['prompt_token_ids'][:p['num_scheduled_tokens']]
                input_ids = torch.tensor(toks, device="cuda")
                logits = engine.prefill_to_slot(sid, input_ids)

            elif nd == 0 and np_ > 1:
                # Pure multi-prefill — unified path (same or variable length)
                prefill_input_ids = []
                for p in step.prefill_requests:
                    toks = p['prompt_token_ids'][:p['num_scheduled_tokens']]
                    prefill_input_ids.append(
                        torch.tensor(toks, device="cuda"))
                logits = engine.prefill_batch_to_slots(
                    prefill_seq_ids, prefill_input_ids)

            else:
                # Mixed decode+prefill — use mixed_step
                decode_token_list = [d['token_id']
                                     for d in step.decode_requests]
                decode_tokens = torch.tensor(decode_token_list, device="cuda")
                prefill_input_ids = []
                for p in step.prefill_requests:
                    toks = p['prompt_token_ids'][:p['num_scheduled_tokens']]
                    prefill_input_ids.append(
                        torch.tensor(toks, device="cuda"))

                logits = engine.mixed_step(
                    decode_seq_ids, decode_tokens,
                    prefill_seq_ids, prefill_input_ids)

            if is_timed:
                end_evt.record()
                torch.cuda.synchronize()
                all_latencies[step_i].append(start_evt.elapsed_time(end_evt))

            # Correctness check (only on first timed trial)
            if trial == n_warmup:
                step_correctness = {}

                if nd > 0 and np_ == 0:
                    # Pure decode: logits is [D, vocab]
                    for i, d in enumerate(step.decode_requests):
                        custom_top1 = logits[i].argmax(dim=-1).item()
                        vllm_tok = step.vllm_output_tokens.get(
                            d['request_id'])
                        match = (custom_top1 == vllm_tok
                                 if vllm_tok is not None else None)
                        step_correctness[d['request_id']] = (
                            custom_top1, vllm_tok, match)

                elif nd == 0 and np_ == 1:
                    # Pure prefill: logits is [S, vocab]
                    p = step.prefill_requests[0]
                    custom_top1 = logits[-1].argmax(dim=-1).item()
                    vllm_tok = step.vllm_output_tokens.get(p['request_id'])
                    match = (custom_top1 == vllm_tok
                             if vllm_tok is not None else None)
                    step_correctness[p['request_id']] = (
                        custom_top1, vllm_tok, match)

                else:
                    # Mixed: logits is [N_total, vocab]
                    for i, d in enumerate(step.decode_requests):
                        custom_top1 = logits[i].argmax(dim=-1).item()
                        vllm_tok = step.vllm_output_tokens.get(
                            d['request_id'])
                        match = (custom_top1 == vllm_tok
                                 if vllm_tok is not None else None)
                        step_correctness[d['request_id']] = (
                            custom_top1, vllm_tok, match)

                    offset = nd
                    for p in step.prefill_requests:
                        n_tok = p['num_scheduled_tokens']
                        last_logit = logits[offset + n_tok - 1]
                        custom_top1 = last_logit.argmax(dim=-1).item()
                        vllm_tok = step.vllm_output_tokens.get(
                            p['request_id'])
                        match = (custom_top1 == vllm_tok
                                 if vllm_tok is not None else None)
                        step_correctness[p['request_id']] = (
                            custom_top1, vllm_tok, match)
                        offset += n_tok

                correctness[step_i] = step_correctness

    inference_ctx.__exit__(None, None, None)

    # Compute median latencies
    median_latencies = []
    for times in all_latencies:
        if times:
            s = sorted(times)
            median_latencies.append(s[len(s) // 2])
        else:
            median_latencies.append(0.0)

    return median_latencies, correctness


# ── Output Formatting ────────────────────────────────────────────────

def print_comparison(traces, custom_latencies, correctness):
    """Print per-step comparison table."""
    print(f"\n{'Step':>5s}  {'D':>3s}  {'P':>3s}  {'Tok':>5s}  "
          f"{'vLLM(ms)':>9s}  {'Custom(ms)':>11s}  {'Speedup':>8s}  {'Match':>6s}")
    print("-" * 65)

    total_vllm = 0
    total_custom = 0
    total_match = 0
    total_compare = 0

    for i, step in enumerate(traces):
        nd = len(step.decode_requests)
        np_ = len(step.prefill_requests)
        nt = step.total_tokens
        v_ms = step.vllm_latency_ms
        c_ms = custom_latencies[i] if i < len(custom_latencies) else 0

        total_vllm += v_ms
        total_custom += c_ms

        # Correctness
        if correctness[i]:
            matches = sum(1 for _, _, m in correctness[i].values() if m)
            total_c = sum(1 for _, _, m in correctness[i].values()
                          if m is not None)
            total_match += matches
            total_compare += total_c
            match_str = f"{matches}/{total_c}"
        else:
            match_str = "N/A"

        speedup = v_ms / c_ms if c_ms > 0 else float('inf')
        print(f"{step.step_idx:>5d}  {nd:>3d}  {np_:>3d}  {nt:>5d}  "
              f"{v_ms:>9.2f}  {c_ms:>11.2f}  {speedup:>7.2f}x  {match_str:>6s}")

    print("-" * 65)
    speedup = total_vllm / total_custom if total_custom > 0 else float('inf')
    match_pct = total_match / total_compare * 100 if total_compare > 0 else 0
    print(f"{'TOTAL':>5s}  {'':>3s}  {'':>3s}  {'':>5s}  "
          f"{total_vllm:>9.2f}  {total_custom:>11.2f}  {speedup:>7.2f}x  "
          f"{total_match}/{total_compare} ({match_pct:.0f}%)")


def print_summary(traces, custom_latencies, correctness):
    """Print aggregate summary by step type."""
    # Classify steps
    pure_decode = []
    pure_prefill = []
    mixed = []

    for i, step in enumerate(traces):
        nd = len(step.decode_requests)
        np_ = len(step.prefill_requests)
        c_ms = custom_latencies[i] if i < len(custom_latencies) else 0
        entry = (step, c_ms)

        if nd > 0 and np_ == 0:
            pure_decode.append(entry)
        elif nd == 0 and np_ > 0:
            pure_prefill.append(entry)
        else:
            mixed.append(entry)

    print(f"\n{'Category':>20s}  {'Steps':>5s}  {'vLLM avg':>10s}  "
          f"{'Custom avg':>11s}  {'Speedup':>8s}")
    print("-" * 60)

    for name, entries in [("Pure Decode", pure_decode),
                          ("Pure Prefill", pure_prefill),
                          ("Mixed", mixed)]:
        if not entries:
            continue
        n = len(entries)
        v_avg = sum(s.vllm_latency_ms for s, _ in entries) / n
        c_avg = sum(c for _, c in entries) / n
        sp = v_avg / c_avg if c_avg > 0 else float('inf')
        print(f"{name:>20s}  {n:>5d}  {v_avg:>9.2f}ms  {c_avg:>10.2f}ms  "
              f"{sp:>7.2f}x")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trace-and-replay mixed batch benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workload", default="staggered",
                        choices=WORKLOAD_NAMES,
                        help="Workload to run")
    parser.add_argument("--skip-vllm", action="store_true",
                        help="Skip vLLM tracing (load trace from file)")
    parser.add_argument("--skip-custom", action="store_true",
                        help="Only run vLLM tracing")
    parser.add_argument("--save-trace", type=str, default=None,
                        help="Save trace to JSON file")
    parser.add_argument("--load-trace", type=str, default=None,
                        help="Load trace from JSON file")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Trials per step for custom engine timing")
    parser.add_argument("--n-warmup", type=int, default=3,
                        help="Warmup trials for custom engine")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile for CUDA graphs")
    args = parser.parse_args()
    args.use_compile = not args.no_compile

    workload = make_workload(args.workload)
    compile_str = " [torch.compile]" if args.use_compile else " [no compile]"
    print("=" * 70)
    print(f"  MIXED BATCH BENCHMARK: {args.workload}{compile_str}")
    print(f"  {len(workload)} requests, arrivals at steps "
          f"{sorted(set(r.arrival_step for r in workload))}")
    print("=" * 70)

    # ── Phase 1: vLLM tracing ──
    if args.load_trace:
        print(f"\nLoading trace from {args.load_trace}...")
        with open(args.load_trace) as f:
            raw = json.load(f)
        traces = [StepTrace(**s) for s in raw]
        print(f"  Loaded {len(traces)} steps")
    elif not args.skip_vllm:
        print("\n── Phase 1: vLLM Tracing ──")
        traces = trace_vllm(args.model, workload)
        print(f"  Traced {len(traces)} steps")

        # Print trace summary
        for step in traces:
            nd = len(step.decode_requests)
            np_ = len(step.prefill_requests)
            print(f"    Step {step.step_idx:>3d}: "
                  f"{nd} decode + {np_} prefill = "
                  f"{step.total_tokens} tokens, "
                  f"{step.vllm_latency_ms:.2f} ms")

        if args.save_trace:
            with open(args.save_trace, 'w') as f:
                json.dump([asdict(t) for t in traces], f, indent=2)
            print(f"  Saved trace to {args.save_trace}")
    else:
        print("Skipping vLLM (--skip-vllm). Need --load-trace.")
        return

    if args.skip_custom:
        print("\nSkipping custom engine replay (--skip-custom)")
        return

    # ── Phase 2: Custom engine replay ──
    print("\n── Phase 2: Custom Engine Replay ──")

    # Determine engine capacity
    all_req_ids = set()
    max_prompt_len = 0
    for step in traces:
        for d in step.decode_requests:
            all_req_ids.add(d['request_id'])
        for p in step.prefill_requests:
            all_req_ids.add(p['request_id'])
            max_prompt_len = max(max_prompt_len, len(p['prompt_token_ids']))

    max_batch = len(all_req_ids) + 1
    # Max seq len: longest prompt + max decode tokens
    max_gen = max(r.max_new_tokens for r in workload)
    max_seq = max_prompt_len + max_gen + 64

    engine = MoEEngine(args.model, max_batch_size=max_batch,
                       max_seq_len=max_seq,
                       use_torch_compile=args.use_compile)

    custom_latencies, correctness = replay_on_custom(
        engine, traces, n_warmup=args.n_warmup, n_trials=args.n_trials,
        use_compile=args.use_compile)

    # ── Phase 3: Comparison ──
    print("\n" + "=" * 70)
    print(f"  RESULTS: {args.workload}")
    print("=" * 70)

    print_comparison(traces, custom_latencies, correctness)
    print_summary(traces, custom_latencies, correctness)

    del engine
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
