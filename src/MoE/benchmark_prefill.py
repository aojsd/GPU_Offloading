"""Benchmark prefill performance: Custom FlashInfer FA3 vs vLLM FA3.

Measures prefill wall-clock time (CUDA events) at various sequence lengths.
Tests eager, CUDA graph, and CUDA graph + torch.compile modes.
Also tests combined prefill+decode to validate holistic performance.

Usage:
    python src/MoE/benchmark_prefill.py                     # full benchmark
    python src/MoE/benchmark_prefill.py --seq-lens 128 512  # specific seq lens
    python src/MoE/benchmark_prefill.py --skip-vllm          # custom only
    python src/MoE/benchmark_prefill.py --skip-eager         # skip eager baseline
"""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import sys
import time
import torch
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from moe_engine import MoEEngine

DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "models", "OLMoE-1B-7B")
DEFAULT_SEQ_LENS = [128, 256, 512, 1024, 2048]
N_WARMUP = 3
N_TRIALS = 5


def benchmark_prefill_custom(engine, seq_lens, batch_size=1):
    """Benchmark custom engine prefill at each seq_len."""
    results = {}
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for seq_len in seq_lens:
        input_ids = torch.randint(1, 1000, (batch_size, seq_len), device="cuda")

        # Warmup
        for _ in range(N_WARMUP):
            engine.reset()
            engine.prefill(input_ids)
        torch.cuda.synchronize()

        # Timed trials
        times = []
        for _ in range(N_TRIALS):
            engine.reset()
            start_evt.record()
            engine.prefill(input_ids)
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))

        median_ms = sorted(times)[N_TRIALS // 2]
        results[seq_len] = median_ms
        print(f"  seq_len={seq_len:>5d}  batch={batch_size}  "
              f"median={median_ms:.2f}ms  (all: {', '.join(f'{t:.2f}' for t in times)})")

    return results


def benchmark_prefill_custom_graph(engine, seq_lens, batch_size=1,
                                   use_torch_compile=False):
    """Benchmark custom engine prefill with CUDA graphs at each seq_len.

    Captures CUDA graphs at the requested seq_lens, then benchmarks replay.
    """
    # Capture CUDA graphs at the requested sizes
    engine.reset()
    engine.capture_prefill_cuda_graph(
        batch_size=batch_size, seq_lengths=seq_lens,
        use_torch_compile=use_torch_compile)

    results = {}
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for seq_len in seq_lens:
        input_ids = torch.randint(1, 1000, (batch_size, seq_len), device="cuda")

        # Warmup (graph replay)
        for _ in range(N_WARMUP):
            engine.reset()
            engine.prefill(input_ids)
        torch.cuda.synchronize()

        # Timed trials
        times = []
        for _ in range(N_TRIALS):
            engine.reset()
            start_evt.record()
            engine.prefill(input_ids)
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))

        median_ms = sorted(times)[N_TRIALS // 2]
        results[seq_len] = median_ms
        print(f"  seq_len={seq_len:>5d}  batch={batch_size}  "
              f"median={median_ms:.2f}ms  (all: {', '.join(f'{t:.2f}' for t in times)})")

    return results


def benchmark_prefill_vllm(model_path, seq_lens):
    """Benchmark vLLM prefill at each seq_len using step-by-step API."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        disable_log_stats=True,
        enable_prefix_caching=False,  # Disable prefix caching for fair comparison
    )
    sp = SamplingParams(max_tokens=1, temperature=0)

    # Warmup (triggers CUDA graph capture etc.)
    for i in range(3):
        tokens = torch.randint(1, 1000, (256,)).tolist()
        llm.llm_engine.add_request(
            request_id=f"warmup_{i}",
            prompt={"prompt_token_ids": tokens},
            params=sp,
        )
        while llm.llm_engine.has_unfinished_requests():
            llm.llm_engine.step()
    torch.cuda.synchronize()

    results = {}
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for seq_len in seq_lens:
        # Warmup at this seq_len — unique random prompts each time
        for i in range(N_WARMUP):
            warm_ids = torch.randint(1, 1000, (seq_len,)).tolist()
            llm.llm_engine.add_request(
                request_id=f"pre_warm_{seq_len}_{i}",
                prompt={"prompt_token_ids": warm_ids},
                params=sp,
            )
            while llm.llm_engine.has_unfinished_requests():
                llm.llm_engine.step()
        torch.cuda.synchronize()

        # Timed trials — measure just the prefill step
        # Unique random prompt per trial ensures full prefill (no prefix cache hits)
        times = []
        for trial in range(N_TRIALS):
            trial_ids = torch.randint(1, 1000, (seq_len,)).tolist()
            llm.llm_engine.add_request(
                request_id=f"bench_{seq_len}_{trial}",
                prompt={"prompt_token_ids": trial_ids},
                params=sp,
            )
            # First step is prefill
            start_evt.record()
            llm.llm_engine.step()
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))
            # Drain remaining steps (1 decode to produce the max_tokens=1 output)
            while llm.llm_engine.has_unfinished_requests():
                llm.llm_engine.step()

        median_ms = sorted(times)[N_TRIALS // 2]
        results[seq_len] = median_ms
        print(f"  seq_len={seq_len:>5d}  batch=1  "
              f"median={median_ms:.2f}ms  (all: {', '.join(f'{t:.2f}' for t in times)})")

    return results


def benchmark_prefill_decode_custom(engine, prefill_len, decode_steps, batch_size=1):
    """Benchmark combined prefill + decode sequence."""
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    input_ids = torch.randint(1, 1000, (batch_size, prefill_len), device="cuda")

    # Capture CUDA graph for decode
    engine.reset()
    engine.capture_decode_cuda_graph(
        batch_size=batch_size, warmup_seq_len=128,
        max_decode_tokens=prefill_len + decode_steps)

    # Warmup
    for _ in range(N_WARMUP):
        engine.reset()
        engine.prefill(input_ids)
        for step in range(decode_steps):
            positions = engine.seq_lens[:batch_size].clone()
            token = torch.zeros(batch_size, dtype=torch.long, device="cuda")
            engine.decode_step(token, positions)
    torch.cuda.synchronize()

    # Timed trials
    times_prefill = []
    times_decode = []
    times_total = []
    for _ in range(N_TRIALS):
        engine.reset()

        # Prefill
        start_evt.record()
        logits = engine.prefill(input_ids)
        end_evt.record()
        torch.cuda.synchronize()
        t_prefill = start_evt.elapsed_time(end_evt)

        # Decode
        next_token = logits[:, -1, :].argmax(dim=-1)
        start_evt.record()
        for step in range(decode_steps):
            positions = engine.seq_lens[:batch_size].clone()
            logits = engine.decode_step(next_token, positions)
            next_token = logits.argmax(dim=-1)
        end_evt.record()
        torch.cuda.synchronize()
        t_decode = start_evt.elapsed_time(end_evt)

        times_prefill.append(t_prefill)
        times_decode.append(t_decode)
        times_total.append(t_prefill + t_decode)

    med_p = sorted(times_prefill)[N_TRIALS // 2]
    med_d = sorted(times_decode)[N_TRIALS // 2]
    med_t = sorted(times_total)[N_TRIALS // 2]
    return med_p, med_d, med_t


def benchmark_prefill_decode_vllm(model_path, prefill_len, decode_steps):
    """Benchmark combined prefill + decode for vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        disable_log_stats=True,
        enable_prefix_caching=False,  # Disable prefix caching for fair comparison
    )
    sp_gen = SamplingParams(max_tokens=decode_steps, temperature=0)

    # Warmup
    for i in range(3):
        tokens = torch.randint(1, 1000, (256,)).tolist()
        sp_w = SamplingParams(max_tokens=1, temperature=0)
        llm.llm_engine.add_request(
            request_id=f"warmup_{i}",
            prompt={"prompt_token_ids": tokens},
            params=sp_w,
        )
        while llm.llm_engine.has_unfinished_requests():
            llm.llm_engine.step()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    times_total = []
    for trial in range(N_TRIALS):
        # Unique random prompt per trial to avoid prefix cache hits
        trial_ids = torch.randint(1, 1000, (prefill_len,)).tolist()
        llm.llm_engine.add_request(
            request_id=f"bench_{trial}",
            prompt={"prompt_token_ids": trial_ids},
            params=sp_gen,
        )
        start_evt.record()
        step = 0
        while llm.llm_engine.has_unfinished_requests():
            llm.llm_engine.step()
            step += 1
        end_evt.record()
        torch.cuda.synchronize()
        times_total.append(start_evt.elapsed_time(end_evt))

    med_t = sorted(times_total)[N_TRIALS // 2]
    return med_t


def main():
    parser = argparse.ArgumentParser(description="Prefill benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-eager", action="store_true")
    parser.add_argument("--skip-combined", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("  PREFILL BENCHMARK: Custom (eager / CUDA graph / compile+graph) vs vLLM")
    print("=" * 80)

    # ── Custom Engine: Eager baseline ──
    custom_eager = {}
    if not args.skip_eager:
        print("\n── Custom Engine Prefill (eager) ──")
        engine = MoEEngine(args.model, max_batch_size=4, max_seq_len=4096,
                           use_torch_compile=False)
        custom_eager = benchmark_prefill_custom(engine, args.seq_lens)
        del engine
        torch.cuda.empty_cache()

    # ── Custom Engine: CUDA graph only (no torch.compile) ──
    print("\n── Custom Engine Prefill (CUDA graph, no compile) ──")
    engine = MoEEngine(args.model, max_batch_size=4, max_seq_len=4096,
                       use_torch_compile=False)
    custom_graph = benchmark_prefill_custom_graph(
        engine, args.seq_lens, use_torch_compile=False)
    del engine
    torch.cuda.empty_cache()

    # ── Custom Engine: torch.compile + CUDA graph ──
    print("\n── Custom Engine Prefill (torch.compile + CUDA graph) ──")
    engine = MoEEngine(args.model, max_batch_size=4, max_seq_len=4096,
                       use_torch_compile=True)
    custom_compiled = benchmark_prefill_custom_graph(
        engine, args.seq_lens, use_torch_compile=True)

    # ── Combined prefill (compiled CUDA graph) + decode (CUDA graph) ──
    combos = [
        (128, 50),   # short prefill + decode
        (512, 50),   # medium prefill + decode
        (2048, 50),  # long prefill + decode
    ]
    custom_combined = {}
    if not args.skip_combined:
        print("\n── Custom Engine Combined (compiled prefill graph + decode graph) ──")
        for prefill_len, decode_steps in combos:
            med_p, med_d, med_t = benchmark_prefill_decode_custom(
                engine, prefill_len, decode_steps)
            custom_combined[(prefill_len, decode_steps)] = (med_p, med_d, med_t)
            print(f"    prefill={prefill_len:>5d} + decode={decode_steps:>3d}:  "
                  f"prefill={med_p:.2f}ms  decode={med_d:.2f}ms  total={med_t:.2f}ms")

    del engine
    torch.cuda.empty_cache()

    # ── vLLM ──
    vllm_results = {}
    if not args.skip_vllm:
        print("\n── vLLM Prefill ──")
        vllm_results = benchmark_prefill_vllm(args.model, args.seq_lens)

    # ── Comparison table ──
    print("\n" + "=" * 80)
    print("  PREFILL COMPARISON (median ms, batch=1)")
    print("=" * 80)

    has_eager = bool(custom_eager)
    header = f"  {'seq_len':>8s}"
    if has_eager:
        header += f"  {'Eager':>10s}"
    header += f"  {'Graph':>10s}  {'Compile+G':>10s}  {'vLLM':>10s}  {'C+G vs vLLM':>12s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for sl in args.seq_lens:
        e = custom_eager.get(sl, float('nan'))
        g = custom_graph.get(sl, float('nan'))
        c = custom_compiled.get(sl, float('nan'))
        v = vllm_results.get(sl, float('nan'))
        delta = c - v if (v == v and c == c) else float('nan')
        pct = delta / v * 100 if (v == v and v > 0 and c == c) else float('nan')

        line = f"  {sl:>8d}"
        if has_eager:
            line += f"  {e:>9.2f}ms" if e == e else f"  {'N/A':>10s}"
        line += f"  {g:>9.2f}ms" if g == g else f"  {'N/A':>10s}"
        line += f"  {c:>9.2f}ms" if c == c else f"  {'N/A':>10s}"
        v_str = f"{v:.2f}ms" if v == v else "N/A"
        d_str = f"{delta:+.2f}ms ({pct:+.1f}%)" if (delta == delta) else "N/A"
        line += f"  {v_str:>10s}  {d_str:>12s}"
        print(line)

    if custom_combined:
        print("\n" + "=" * 80)
        print("  COMBINED PREFILL + DECODE (batch=1, compile+graph)")
        print("=" * 80)
        for (prefill_len, decode_steps), (med_p, med_d, med_t) in custom_combined.items():
            per_decode = med_d / decode_steps
            print(f"    prefill={prefill_len:>5d} + {decode_steps}x decode:  "
                  f"prefill={med_p:.2f}ms  decode_total={med_d:.2f}ms "
                  f"({per_decode:.2f}ms/step)  total={med_t:.2f}ms")


if __name__ == "__main__":
    main()
