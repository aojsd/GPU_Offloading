"""Benchmark and validate custom MoE engine vs vLLM.

Phase 1: Correctness — greedy generation on same prompts, compare tokens.
Phase 2: Performance — decode latency sweep across sequence lengths.

Runs custom engine first (correctness + perf), frees GPU, then loads vLLM
(correctness + perf), and prints a combined comparison.

Must run with:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python src/MoE/benchmark.py

Usage:
    python src/MoE/benchmark.py                        # full run
    python src/MoE/benchmark.py --model path/to/model  # custom model
    python src/MoE/benchmark.py --correctness-only     # just correctness
    python src/MoE/benchmark.py --perf-only            # just performance
    python src/MoE/benchmark.py --max-seq 2048         # cap seq_len sweep
    python src/MoE/benchmark.py --custom-only          # skip vLLM
    python src/MoE/benchmark.py --vllm-only            # skip custom engine
"""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

import argparse
import statistics
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

# Apply glibc 2.28 monkey patches (must happen before vLLM import)
import moe_engine  # noqa: F401
from moe_engine import MoEEngine

DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent / "models" / "OLMoE-1B-7B")

# ── Test prompts for correctness ─────────────────────────────────────
CORRECTNESS_PROMPTS = [
    [50256, 510, 5765, 273, 6181, 310],          # "The capital of France is"
    [1, 4093, 2501, 247, 673, 627, 369, 247],    # misc tokens
    [100, 200, 300, 400, 500, 600, 700, 800],    # sequential IDs
]
MAX_NEW_TOKENS = 50

# ── Performance sweep config ─────────────────────────────────────────
# vLLM limited to 4096 by compiled RoPE max_position_embeddings
PERF_SEQ_LENS = [128, 256, 512, 1024, 2048]
N_DECODE_STEPS = 200   # decode steps per measurement
N_WARMUP = 20
N_TRIALS = 3           # trials per measurement (report median)


# =====================================================================
#  Correctness helpers
# =====================================================================

def custom_greedy_generate(engine, prompt_ids, max_new_tokens):
    """Run greedy generation with custom engine.

    Returns (tokens, logits_list) where logits_list[i] is the top-k logit
    info dict for step i: {token_id: logit_value} for the top 5 tokens.
    """
    engine.reset()
    input_ids = torch.tensor([prompt_ids], device="cuda")

    logits = engine.prefill(input_ids)
    step_logits_raw = logits[0, -1, :]  # [vocab]
    next_token = step_logits_raw.argmax().item()

    tokens = [next_token]
    logits_per_step = [_extract_topk_logits(step_logits_raw, k=10)]

    for _ in range(max_new_tokens - 1):
        positions = engine.seq_lens[:1].clone()
        token_t = torch.tensor([next_token], device="cuda")
        step_logits_raw = engine.decode_step(token_t, positions)[0]  # [vocab]
        next_token = step_logits_raw.argmax().item()
        tokens.append(next_token)
        logits_per_step.append(_extract_topk_logits(step_logits_raw, k=10))
        if next_token == engine.eos_token_id:
            break

    return tokens, logits_per_step


def _extract_topk_logits(logits_1d, k=10):
    """Extract top-k (token_id, logit_value) from a [vocab] logits tensor."""
    topk = torch.topk(logits_1d.float(), k)
    return {tid.item(): val.item() for tid, val in zip(topk.indices, topk.values)}


def vllm_greedy_generate(llm, prompt_ids, max_new_tokens):
    """Run greedy generation with vLLM.

    Returns (tokens, logprobs_list) where logprobs_list[i] is a dict
    {token_id: logprob} for the top tokens at step i.
    """
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0, logprobs=10)
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        sampling_params=sp,
    )
    out = outputs[0].outputs[0]
    tokens = list(out.token_ids)

    logprobs_per_step = []
    if out.logprobs is not None:
        for step_lp in out.logprobs:
            # step_lp: dict[int, Logprob]
            logprobs_per_step.append(
                {tid: lp.logprob for tid, lp in step_lp.items()}
            )

    return tokens, logprobs_per_step


# =====================================================================
#  Performance helpers
# =====================================================================

def _time_custom_decode_once(engine, target_seq_len, n_steps, n_warmup):
    """Single trial: time n_steps CUDA-graph decode steps. Returns ms/step."""
    engine.reset()
    engine.k_cache.normal_(0, 0.01)
    engine.v_cache.normal_(0, 0.01)
    engine.seq_lens[0] = target_seq_len
    engine._seq_lens_cpu[0] = target_seq_len

    token = torch.tensor([100], device="cuda")
    pos = torch.tensor([target_seq_len], dtype=torch.int32, device="cuda")

    use_graph = 1 in engine._cuda_graphs

    for _ in range(n_warmup):
        engine.seq_lens[0] = target_seq_len
        engine._seq_lens_cpu[0] = target_seq_len
        if use_graph:
            engine._decode_step_graphed(token, pos)
        else:
            engine.decode_step(token, pos)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        engine.seq_lens[0] = target_seq_len
        engine._seq_lens_cpu[0] = target_seq_len
        if use_graph:
            engine._decode_step_graphed(token, pos)
        else:
            engine.decode_step(token, pos)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_steps


def time_custom_decode(engine, target_seq_len, n_steps, n_warmup, n_trials):
    """Time decode steps over multiple trials. Returns (median, all_trials)."""
    trials = [_time_custom_decode_once(engine, target_seq_len, n_steps, n_warmup)
              for _ in range(n_trials)]
    return statistics.median(trials), trials


def _time_vllm_decode_once(llm, target_seq_len, n_steps, n_warmup, request_id):
    """Single trial: time n_steps vLLM decode steps. Returns ms/step."""
    from vllm import SamplingParams

    total_tokens = n_warmup + n_steps
    sp = SamplingParams(max_tokens=total_tokens, temperature=0)
    prompt_ids = list(range(100, 100 + target_seq_len))

    llm.llm_engine.add_request(
        request_id=request_id,
        prompt={"prompt_token_ids": prompt_ids},
        params=sp,
    )

    # Step 0 = prefill
    llm.llm_engine.step()

    # Warmup decode steps
    for _ in range(n_warmup):
        llm.llm_engine.step()
    torch.cuda.synchronize()

    # Timed decode steps
    t0 = time.perf_counter()
    for _ in range(n_steps):
        llm.llm_engine.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Drain remaining if any
    while llm.llm_engine.has_unfinished_requests():
        llm.llm_engine.step()

    return (t1 - t0) / n_steps * 1000  # ms


def time_vllm_decode(llm, target_seq_len, n_steps, n_warmup, n_trials):
    """Time decode steps over multiple trials. Returns (median, all_trials)."""
    trials = [_time_vllm_decode_once(llm, target_seq_len, n_steps, n_warmup,
                                     request_id=f"bench_{target_seq_len}_{t}")
              for t in range(n_trials)]
    return statistics.median(trials), trials


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark custom MoE engine vs vLLM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--correctness-only", action="store_true",
                        help="Only run correctness validation")
    parser.add_argument("--perf-only", action="store_true",
                        help="Only run performance benchmark")
    parser.add_argument("--custom-only", action="store_true",
                        help="Only run custom engine (skip vLLM)")
    parser.add_argument("--vllm-only", action="store_true",
                        help="Only run vLLM (skip custom engine)")
    parser.add_argument("--max-seq", type=int, default=2048,
                        help="Maximum sequence length for perf sweep")
    parser.add_argument("--n-steps", type=int, default=N_DECODE_STEPS,
                        help="Decode steps per perf measurement")
    parser.add_argument("--no-graph", action="store_true",
                        help="Disable CUDA graphs for custom engine")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS,
                        help="Trials per measurement (report median)")
    args = parser.parse_args()
    MODEL_DIR = args.model

    run_correctness = not args.perf_only
    run_perf = not args.correctness_only
    run_custom = not args.vllm_only
    run_vllm = not args.custom_only

    seq_lens = [s for s in PERF_SEQ_LENS if s <= args.max_seq]

    # Storage for cross-engine comparison
    custom_tokens = {}   # prompt_idx -> list[int]
    custom_logits = {}   # prompt_idx -> list[dict{token_id: logit}]
    custom_gen_ms = {}   # prompt_idx -> end-to-end generation time (ms)
    vllm_tokens = {}
    vllm_logprobs = {}   # prompt_idx -> list[dict{token_id: logprob}]
    vllm_gen_ms = {}     # prompt_idx -> end-to-end generation time (ms)
    custom_perf = {}     # seq_len -> ms/step
    vllm_perf = {}

    # ── Custom Engine ────────────────────────────────────────────────
    if run_custom:
        print("=" * 70)
        print("CUSTOM ENGINE")
        print("=" * 70)

        max_needed = max(seq_lens) + 256 if run_perf else 1024
        # use_torch_compile=False for correctness (exact match with HF/vLLM)
        # CUDA graph capture defaults to no compile as well
        engine = MoEEngine(MODEL_DIR, max_batch_size=1, max_seq_len=max_needed,
                           use_torch_compile=False)
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory: {mem_gb:.1f} GB")

        # Capture CUDA graph
        if not args.no_graph:
            engine.capture_decode_cuda_graph(
                batch_size=1, warmup_seq_len=128,
                max_decode_tokens=max_needed - 128)
            print("CUDA graph captured")

        # Warmup: one throwaway generation to trigger any lazy init
        engine.reset()
        warmup_ids = torch.tensor([[100, 200, 300, 400]], device="cuda")
        engine.generate(warmup_ids, max_new_tokens=5)
        torch.cuda.synchronize()
        print("Warmup complete")
        print()

        # Correctness (run once for token comparison, then time over trials)
        if run_correctness:
            print("── Correctness (greedy generation) ──")
            for i, prompt in enumerate(CORRECTNESS_PROMPTS):
                # Single run for correctness (captures logits)
                tokens, logits = custom_greedy_generate(engine, prompt, MAX_NEW_TOKENS)
                custom_tokens[i] = tokens
                custom_logits[i] = logits

                # Multi-trial for timing
                trial_times = []
                for _ in range(args.n_trials):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    custom_greedy_generate(engine, prompt, MAX_NEW_TOKENS)
                    torch.cuda.synchronize()
                    trial_times.append((time.perf_counter() - t0) * 1000)
                med_ms = statistics.median(trial_times)
                custom_gen_ms[i] = med_ms
                n_tok = len(tokens)
                print(f"  Prompt {i}: {n_tok} tokens, "
                      f"median {med_ms:.0f} ms over {args.n_trials} trials "
                      f"({n_tok / med_ms * 1000:.0f} tok/s)")
            print()

        # Performance
        if run_perf:
            print(f"── Performance (batch=1, {args.n_steps} steps/trial, "
                  f"{args.n_trials} trials, {N_WARMUP} warmup) ──")
            print(f"{'seq_len':>10s}  {'median':>8s}  {'stdev':>8s}  {'tok/s':>8s}")
            print("-" * 40)
            for sl in seq_lens:
                med, trials = time_custom_decode(
                    engine, sl, args.n_steps, N_WARMUP, args.n_trials)
                custom_perf[sl] = med
                std = statistics.stdev(trials) if len(trials) > 1 else 0
                print(f"{sl:>10,d}  {med:>8.2f}  {std:>7.3f}  {1000/med:>8.0f}")
            print()

        del engine
        torch.cuda.empty_cache()

    # ── vLLM ─────────────────────────────────────────────────────────
    if run_vllm:
        print("=" * 70)
        print("vLLM")
        print("=" * 70)

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=MODEL_DIR,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
        )

        # Warmup vLLM (trigger CUDA graph capture, JIT, etc.)
        sp_warmup = SamplingParams(max_tokens=10, temperature=0)
        for j in range(3):
            warmup_ids = torch.randint(1, 1000, (256,)).tolist()
            llm.generate([{"prompt_token_ids": warmup_ids}],
                         sampling_params=sp_warmup)
        torch.cuda.synchronize()
        print("Warmup complete")
        print()

        # Correctness (run once for token comparison, then time over trials)
        if run_correctness:
            print("── Correctness (greedy generation) ──")
            for i, prompt in enumerate(CORRECTNESS_PROMPTS):
                # Single run for correctness (captures logprobs)
                tokens, logprobs = vllm_greedy_generate(llm, prompt, MAX_NEW_TOKENS)
                vllm_tokens[i] = tokens
                vllm_logprobs[i] = logprobs

                # Multi-trial for timing (no logprobs — avoids overhead)
                sp_fast = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0)
                trial_times = []
                for _ in range(args.n_trials):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    llm.generate([{"prompt_token_ids": prompt}],
                                 sampling_params=sp_fast)
                    torch.cuda.synchronize()
                    trial_times.append((time.perf_counter() - t0) * 1000)
                med_ms = statistics.median(trial_times)
                vllm_gen_ms[i] = med_ms
                n_tok = len(tokens)
                print(f"  Prompt {i}: {n_tok} tokens, "
                      f"median {med_ms:.0f} ms over {args.n_trials} trials "
                      f"({n_tok / med_ms * 1000:.0f} tok/s)")
            print()

        # Performance
        if run_perf:
            print(f"── Performance (batch=1, {args.n_steps} steps/trial, "
                  f"{args.n_trials} trials, {N_WARMUP} warmup) ──")
            print(f"{'seq_len':>10s}  {'median':>8s}  {'stdev':>8s}  {'tok/s':>8s}")
            print("-" * 40)
            for sl in seq_lens:
                med, trials = time_vllm_decode(
                    llm, sl, args.n_steps, N_WARMUP, args.n_trials)
                vllm_perf[sl] = med
                std = statistics.stdev(trials) if len(trials) > 1 else 0
                print(f"{sl:>10,d}  {med:>8.2f}  {std:>7.3f}  {1000/med:>8.0f}")
            print()

        del llm
        torch.cuda.empty_cache()

    # ── Combined results ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # Correctness comparison
    if run_correctness and run_custom and run_vllm:
        print()
        print("── Correctness ──")
        total_prompts = 0
        total_exact = 0
        for i in sorted(set(custom_tokens) & set(vllm_tokens)):
            ct = custom_tokens[i]
            vt = vllm_tokens[i]
            cl = custom_logits.get(i, [])
            vl = vllm_logprobs.get(i, [])
            min_len = min(len(ct), len(vt))
            total_prompts += 1

            # Find first divergence
            match_len = 0
            for j in range(min_len):
                if ct[j] != vt[j]:
                    break
                match_len += 1
            else:
                match_len = min_len

            if match_len == min_len:
                total_exact += 1
                print(f"  Prompt {i}: EXACT MATCH — all {min_len} tokens identical")
            else:
                print(f"  Prompt {i}: {match_len}/{min_len} tokens match, "
                      f"diverges at token {match_len}")
                custom_tok = ct[match_len]
                vllm_tok = vt[match_len]
                print(f"    custom chose {custom_tok}, vllm chose {vllm_tok}")

                # Diagnose using custom engine logits at the divergence step
                if match_len < len(cl):
                    step_logits = cl[match_len]
                    c_logit = step_logits.get(custom_tok)
                    v_logit = step_logits.get(vllm_tok)
                    top_tid = max(step_logits, key=step_logits.get)
                    top_val = step_logits[top_tid]
                    second_tid = sorted(step_logits, key=step_logits.get, reverse=True)[1]
                    second_val = step_logits[second_tid]
                    print(f"    Custom engine logits at step {match_len}:")
                    print(f"      top-1: token {top_tid} = {top_val:.4f}")
                    print(f"      top-2: token {second_tid} = {second_val:.4f}")
                    print(f"      gap (top1 - top2): {top_val - second_val:.4f}")
                    if v_logit is not None:
                        print(f"      vllm's token {vllm_tok}: logit = {v_logit:.4f} "
                              f"(rank in custom top-10: "
                              f"{'not in top-10' if vllm_tok not in step_logits else sorted(step_logits, key=step_logits.get, reverse=True).index(vllm_tok) + 1})")
                    else:
                        print(f"      vllm's token {vllm_tok}: not in custom top-10")

                # Also show vLLM's logprobs at the divergence step
                if match_len < len(vl):
                    step_lp = vl[match_len]
                    print(f"    vLLM logprobs at step {match_len}:")
                    for tid in sorted(step_lp, key=step_lp.get, reverse=True)[:3]:
                        marker = " <-- chosen" if tid == vllm_tok else ""
                        print(f"      token {tid}: logprob = {step_lp[tid]:.4f}{marker}")

        print(f"\n  {total_exact}/{total_prompts} prompts exact match")

    # End-to-end generation time comparison
    if run_correctness and custom_gen_ms and vllm_gen_ms:
        print()
        print("── End-to-End Generation Time ──")
        print(f"  {'Prompt':>8s}  {'Tokens':>6s}  {'Custom ms':>10s}  {'vLLM ms':>10s}  {'Speedup':>10s}")
        print("  " + "-" * 50)
        for i in sorted(set(custom_gen_ms) & set(vllm_gen_ms)):
            n_tok = len(custom_tokens[i])
            c_ms = custom_gen_ms[i]
            v_ms = vllm_gen_ms[i]
            print(f"  {i:>8d}  {n_tok:>6d}  {c_ms:>10.0f}  {v_ms:>10.0f}  {v_ms/c_ms:>9.2f}x")

    # Performance comparison
    if run_perf and custom_perf and vllm_perf:
        print()
        print("── Performance (ms/step, batch=1) ──")
        print(f"{'seq_len':>10s}  {'Custom':>10s}  {'vLLM':>10s}  {'Speedup':>10s}")
        print("-" * 45)
        for sl in seq_lens:
            c = custom_perf.get(sl)
            v = vllm_perf.get(sl)
            if c is not None and v is not None:
                speedup = v / c
                print(f"{sl:>10,d}  {c:>10.2f}  {v:>10.2f}  {speedup:>9.2f}x")
            elif c is not None:
                print(f"{sl:>10,d}  {c:>10.2f}  {'—':>10s}  {'—':>10s}")
            elif v is not None:
                print(f"{sl:>10,d}  {'—':>10s}  {v:>10.2f}  {'—':>10s}")
    elif run_perf:
        # Single engine results already printed above
        pass

    print()


if __name__ == "__main__":
    main()
