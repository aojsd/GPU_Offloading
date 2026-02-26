"""Microbenchmarks for custom MoE engine: correctness, decode, prefill, CUDA graphs, mixed batches.

Subcommands:
    decode      Correctness + decode performance vs vLLM
    prefill     Prefill performance (graph, compile+graph) vs vLLM
    cuda-graph  CUDA graph correctness (eager vs graph exact match) + timing
    mixed       mixed_step smoke tests (pure decode, pure prefill, mixed, multi-batch)
    all         Run all benchmarks

Usage:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python vLLM_comparison/microbenchmark.py decode
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python vLLM_comparison/microbenchmark.py prefill --skip-vllm
    python vLLM_comparison/microbenchmark.py cuda-graph
    python vLLM_comparison/microbenchmark.py mixed
    python vLLM_comparison/microbenchmark.py all
"""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

import argparse
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MOE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOE_DIR))

import torch

# Apply glibc 2.28 monkey patches (must happen before vLLM import)
import moe_engine  # noqa: F401
from moe_engine import MoEEngine

DEFAULT_MODEL_DIR = str(MOE_DIR / "models" / "OLMoE-1B-7B")

# =====================================================================
#  Shared constants
# =====================================================================

CORRECTNESS_PROMPTS = [
    [50256, 510, 5765, 273, 6181, 310],          # "The capital of France is"
    [1, 4093, 2501, 247, 673, 627, 369, 247],    # misc tokens
    [100, 200, 300, 400, 500, 600, 700, 800],    # sequential IDs
]
MAX_NEW_TOKENS = 50

PERF_SEQ_LENS = [128, 256, 512, 1024, 2048]
N_DECODE_STEPS = 200
N_WARMUP = 20
N_TRIALS = 3

PREFILL_SEQ_LENS = [128, 256, 512, 1024, 2048]
PREFILL_N_WARMUP = 3
PREFILL_N_TRIALS = 5


# =====================================================================
#  Correctness helpers (decode)
# =====================================================================

def custom_greedy_generate(engine, prompt_ids, max_new_tokens):
    """Run greedy generation with custom engine.

    Returns (tokens, logits_list) where logits_list[i] is the top-k logit
    info dict for step i: {token_id: logit_value} for the top 10 tokens.
    """
    engine.reset()
    input_ids = torch.tensor([prompt_ids], device="cuda")

    logits = engine.prefill(input_ids)
    step_logits_raw = logits[0, -1, :]
    next_token = step_logits_raw.argmax().item()

    tokens = [next_token]
    logits_per_step = [_extract_topk_logits(step_logits_raw, k=10)]

    for _ in range(max_new_tokens - 1):
        positions = engine.seq_lens[:1].clone()
        token_t = torch.tensor([next_token], device="cuda")
        step_logits_raw = engine.decode_step(token_t, positions)[0]
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
            logprobs_per_step.append(
                {tid: lp.logprob for tid, lp in step_lp.items()}
            )

    return tokens, logprobs_per_step


# =====================================================================
#  Decode performance helpers
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
#  Prefill performance helpers
# =====================================================================

def benchmark_prefill_custom_graph(engine, seq_lens, batch_size=1,
                                   use_torch_compile=False):
    """Benchmark custom engine prefill with CUDA graphs at each seq_len."""
    engine.reset()
    engine.capture_prefill_cuda_graph(
        total_token_sizes=[batch_size * s for s in seq_lens],
        use_torch_compile=use_torch_compile)

    results = {}
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for seq_len in seq_lens:
        input_ids = torch.randint(1, 1000, (batch_size, seq_len), device="cuda")

        for _ in range(PREFILL_N_WARMUP):
            engine.reset()
            engine.prefill(input_ids)
        torch.cuda.synchronize()

        times = []
        for _ in range(PREFILL_N_TRIALS):
            engine.reset()
            start_evt.record()
            engine.prefill(input_ids)
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))

        median_ms = sorted(times)[PREFILL_N_TRIALS // 2]
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
        enable_prefix_caching=False,
    )
    sp = SamplingParams(max_tokens=1, temperature=0)

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
        for i in range(PREFILL_N_WARMUP):
            warm_ids = torch.randint(1, 1000, (seq_len,)).tolist()
            llm.llm_engine.add_request(
                request_id=f"pre_warm_{seq_len}_{i}",
                prompt={"prompt_token_ids": warm_ids},
                params=sp,
            )
            while llm.llm_engine.has_unfinished_requests():
                llm.llm_engine.step()
        torch.cuda.synchronize()

        times = []
        for trial in range(PREFILL_N_TRIALS):
            trial_ids = torch.randint(1, 1000, (seq_len,)).tolist()
            llm.llm_engine.add_request(
                request_id=f"bench_{seq_len}_{trial}",
                prompt={"prompt_token_ids": trial_ids},
                params=sp,
            )
            start_evt.record()
            llm.llm_engine.step()
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))
            while llm.llm_engine.has_unfinished_requests():
                llm.llm_engine.step()

        median_ms = sorted(times)[PREFILL_N_TRIALS // 2]
        results[seq_len] = median_ms
        print(f"  seq_len={seq_len:>5d}  batch=1  "
              f"median={median_ms:.2f}ms  (all: {', '.join(f'{t:.2f}' for t in times)})")

    return results


def benchmark_prefill_decode_custom(engine, prefill_len, decode_steps, batch_size=1):
    """Benchmark combined prefill + decode sequence."""
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    input_ids = torch.randint(1, 1000, (batch_size, prefill_len), device="cuda")

    engine.reset()
    engine.capture_decode_cuda_graph(
        batch_size=batch_size, warmup_seq_len=128,
        max_decode_tokens=prefill_len + decode_steps)

    for _ in range(PREFILL_N_WARMUP):
        engine.reset()
        engine.prefill(input_ids)
        for step in range(decode_steps):
            positions = engine.seq_lens[:batch_size].clone()
            token = torch.zeros(batch_size, dtype=torch.long, device="cuda")
            engine.decode_step(token, positions)
    torch.cuda.synchronize()

    times_prefill = []
    times_decode = []
    times_total = []
    for _ in range(PREFILL_N_TRIALS):
        engine.reset()

        start_evt.record()
        logits = engine.prefill(input_ids)
        end_evt.record()
        torch.cuda.synchronize()
        t_prefill = start_evt.elapsed_time(end_evt)

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

    med_p = sorted(times_prefill)[PREFILL_N_TRIALS // 2]
    med_d = sorted(times_decode)[PREFILL_N_TRIALS // 2]
    med_t = sorted(times_total)[PREFILL_N_TRIALS // 2]
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
        enable_prefix_caching=False,
    )
    sp_gen = SamplingParams(max_tokens=decode_steps, temperature=0)

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
    for trial in range(PREFILL_N_TRIALS):
        trial_ids = torch.randint(1, 1000, (prefill_len,)).tolist()
        llm.llm_engine.add_request(
            request_id=f"bench_{trial}",
            prompt={"prompt_token_ids": trial_ids},
            params=sp_gen,
        )
        start_evt.record()
        while llm.llm_engine.has_unfinished_requests():
            llm.llm_engine.step()
        end_evt.record()
        torch.cuda.synchronize()
        times_total.append(start_evt.elapsed_time(end_evt))

    med_t = sorted(times_total)[PREFILL_N_TRIALS // 2]
    return med_t


# =====================================================================
#  CUDA graph correctness + timing helpers
# =====================================================================

CUDA_GRAPH_N_STEPS = 10
CUDA_GRAPH_WARMUP_SEQ = 5


def run_cuda_graph_correctness(model_dir):
    """Validate CUDA graph decode matches eager decode (exact token match)."""
    print("=" * 70)
    print("CUDA GRAPH CORRECTNESS: graph vs eager decode")
    print("=" * 70)

    engine = MoEEngine(
        model_dir, max_batch_size=4, max_seq_len=1024, use_torch_compile=False)
    engine.capture_prefill_cuda_graph(
        total_token_sizes=[6], use_torch_compile=False)

    # Eager greedy generation
    engine.reset()
    prompt = torch.tensor([[50256, 510, 5765, 273, 6181, 310]], device="cuda")
    tokens_eager = engine.generate(prompt, max_new_tokens=30)
    print(f"Eager:  {tokens_eager[0].tolist()}")

    # Capture CUDA graph WITHOUT torch.compile (exact match expected)
    engine.capture_decode_cuda_graph(batch_size=1, warmup_seq_len=6,
                                      max_decode_tokens=50,
                                      use_torch_compile=False)

    # Re-prefill with actual prompt, then generate using graph
    engine.reset()
    logits = engine.prefill(prompt)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated_graph = [next_token]

    for i in range(29):
        positions = engine.seq_lens[:1].clone()
        logits = engine._decode_step_graphed(next_token, positions)
        next_token = logits.argmax(dim=-1)
        generated_graph.append(next_token)
        if next_token.item() == engine.eos_token_id:
            break

    tokens_graph = torch.cat([prompt, torch.stack(generated_graph, dim=1)], dim=1)
    print(f"Graph:  {tokens_graph[0].tolist()}")

    match = torch.equal(tokens_eager[:, :tokens_graph.shape[1]], tokens_graph)
    print(f"\nExact match: {match}")
    if not match:
        min_len = min(tokens_eager.shape[1], tokens_graph.shape[1])
        for i in range(min_len):
            if tokens_eager[0, i] != tokens_graph[0, i]:
                print(f"  First divergence at position {i}: "
                      f"eager={tokens_eager[0, i].item()}, "
                      f"graph={tokens_graph[0, i].item()}")
                break

    del engine
    torch.cuda.empty_cache()
    return match


def run_cuda_graph_timing(model_dir):
    """Compare eager vs CUDA graph vs graph+compile decode timing."""
    n_steps = CUDA_GRAPH_N_STEPS
    warmup_seq = CUDA_GRAPH_WARMUP_SEQ

    print(f"\n{'='*70}")
    print(f"CUDA GRAPH TIMING: {n_steps} decode steps from pos {warmup_seq}")
    print(f"{'='*70}")

    token = torch.tensor([100], device="cuda")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def time_steps(eng, n, label, use_graph=False):
        for _ in range(20):
            pos = eng.seq_lens[:1].clone()
            if use_graph:
                eng._decode_step_graphed(token, pos)
            else:
                eng.decode_step(token, pos)
        torch.cuda.synchronize()

        start.record()
        for _ in range(n):
            pos = eng.seq_lens[:1].clone()
            if use_graph:
                eng._decode_step_graphed(token, pos)
            else:
                eng.decode_step(token, pos)
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end) / n
        print(f"  {label:55s}  {t:.2f} ms/step  ({1000/t:.0f} tok/s)")
        return t

    # Eager (FlashInfer, no graph)
    eng_eager = MoEEngine(
        model_dir, max_batch_size=4, max_seq_len=1024, use_torch_compile=False)
    eng_eager.capture_prefill_cuda_graph(
        total_token_sizes=[warmup_seq], use_torch_compile=False)
    eng_eager.reset()
    eng_eager.prefill(torch.randint(1, 1000, (1, warmup_seq), device="cuda"))
    t_eager = time_steps(eng_eager, n_steps, "Eager (FlashInfer, no graph)")
    del eng_eager
    torch.cuda.empty_cache()

    # CUDA Graph (no compile)
    eng_no_compile = MoEEngine(
        model_dir, max_batch_size=4, max_seq_len=warmup_seq + 512 + 256,
        use_torch_compile=False)
    eng_no_compile.capture_prefill_cuda_graph(
        total_token_sizes=[warmup_seq], use_torch_compile=False)
    eng_no_compile.capture_decode_cuda_graph(
        batch_size=1, warmup_seq_len=warmup_seq, max_decode_tokens=512,
        use_torch_compile=False)
    eng_no_compile.reset()
    eng_no_compile.prefill(torch.randint(1, 1000, (1, warmup_seq), device="cuda"))
    t_no_compile = time_steps(eng_no_compile, n_steps,
                               "CUDA Graph (no compile)", use_graph=True)
    del eng_no_compile
    torch.cuda.empty_cache()

    # CUDA Graph + torch.compile
    eng_compile = MoEEngine(
        model_dir, max_batch_size=4, max_seq_len=warmup_seq + 512 + 256,
        use_torch_compile=True)
    eng_compile.capture_prefill_cuda_graph(
        total_token_sizes=[warmup_seq], use_torch_compile=True)
    eng_compile.capture_decode_cuda_graph(
        batch_size=1, warmup_seq_len=warmup_seq, max_decode_tokens=512,
        use_torch_compile=True)
    eng_compile.reset()
    eng_compile.prefill(torch.randint(1, 1000, (1, warmup_seq), device="cuda"))
    t_compile = time_steps(eng_compile, n_steps,
                            "CUDA Graph + torch.compile", use_graph=True)
    del eng_compile
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Eager (FlashInfer):            {t_eager:5.2f} ms/step  ({1000/t_eager:.0f} tok/s)")
    print(f"  CUDA Graph (no compile):       {t_no_compile:5.2f} ms/step  ({1000/t_no_compile:.0f} tok/s)")
    print(f"  CUDA Graph + torch.compile:    {t_compile:5.2f} ms/step  ({1000/t_compile:.0f} tok/s)")
    print(f"  Compile speedup vs no-compile: {t_no_compile/t_compile:.2f}x")


# =====================================================================
#  Mixed step smoke tests
# =====================================================================

def test_pure_decode_via_mixed(model_dir):
    """Pure decode through mixed_step must match decode_step exactly."""
    print("Test 1: pure decode via mixed_step...")
    engine = MoEEngine(model_dir, max_batch_size=4, max_seq_len=512,
                       use_torch_compile=False)
    engine.capture_prefill_cuda_graph(
        total_token_sizes=[8], use_torch_compile=False)

    prompt = torch.tensor([50256, 510, 5765, 273, 6181, 310], device="cuda")

    engine.reset()
    engine.prefill_to_slot(0, prompt)
    token = torch.tensor([100], device="cuda")
    positions = engine.seq_lens[:1].clone()
    ref_logits = engine.decode_step(token, positions)

    engine.reset()
    engine.prefill_to_slot(0, prompt)
    mixed_logits = engine.mixed_step([0], token, [], [])

    diff = (ref_logits - mixed_logits).abs().max().item()
    assert diff == 0, f"Pure decode mismatch: max diff = {diff}"
    print(f"  PASS — max diff = {diff}")

    del engine
    torch.cuda.empty_cache()


def test_pure_prefill_via_mixed(model_dir):
    """Pure prefill through mixed_step must match prefill exactly."""
    print("Test 2: pure prefill via mixed_step...")
    engine = MoEEngine(model_dir, max_batch_size=4, max_seq_len=512,
                       use_torch_compile=False)
    engine.capture_prefill_cuda_graph(
        total_token_sizes=[8], use_torch_compile=False)

    prompt = torch.tensor([50256, 510, 5765, 273, 6181, 310], device="cuda")

    engine.reset()
    ref_logits = engine.prefill_to_slot(0, prompt)

    engine.reset()
    mixed_logits = engine.mixed_step(
        [], torch.empty(0, dtype=torch.long, device="cuda"),
        [0], [prompt])

    diff = (ref_logits - mixed_logits).abs().max().item()
    assert diff == 0, f"Pure prefill mismatch: max diff = {diff}"
    print(f"  PASS — max diff = {diff}")

    del engine
    torch.cuda.empty_cache()


def test_mixed_batch(model_dir):
    """Mixed decode + prefill: verify both parts produce reasonable logits."""
    print("Test 3: mixed decode + prefill batch...")
    engine = MoEEngine(model_dir, max_batch_size=8, max_seq_len=512,
                       use_torch_compile=False)
    engine.capture_prefill_cuda_graph(
        total_token_sizes=[8], use_torch_compile=False)

    prompt_a = torch.tensor([50256, 510, 5765, 273, 6181, 310], device="cuda")
    prompt_b = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800],
                            device="cuda")

    engine.reset()
    engine.prefill_to_slot(0, prompt_a)
    decode_token = torch.tensor([100], device="cuda")
    pos = engine.seq_lens[:1].clone()
    ref_decode_logits = engine.decode_step(decode_token, pos)
    ref_decode_top1 = ref_decode_logits.argmax(dim=-1).item()

    engine.reset()
    ref_prefill_logits = engine.prefill_to_slot(1, prompt_b)
    ref_prefill_top1 = ref_prefill_logits[-1].argmax(dim=-1).item()

    engine.reset()
    engine.prefill_to_slot(0, prompt_a)
    mixed_logits = engine.mixed_step(
        [0], decode_token,
        [1], [prompt_b])

    mixed_decode_top1 = mixed_logits[0].argmax(dim=-1).item()
    mixed_prefill_top1 = mixed_logits[-1].argmax(dim=-1).item()

    decode_match = mixed_decode_top1 == ref_decode_top1
    prefill_match = mixed_prefill_top1 == ref_prefill_top1

    decode_diff = (ref_decode_logits[0] - mixed_logits[0]).abs().max().item()
    prefill_diff = (ref_prefill_logits[-1] - mixed_logits[-1]).abs().max().item()

    print(f"  Decode top-1: ref={ref_decode_top1} mixed={mixed_decode_top1} "
          f"match={decode_match} max_logit_diff={decode_diff:.4f}")
    print(f"  Prefill top-1: ref={ref_prefill_top1} mixed={mixed_prefill_top1} "
          f"match={prefill_match} max_logit_diff={prefill_diff:.4f}")

    assert decode_match, "Decode top-1 mismatch!"
    assert prefill_match, "Prefill top-1 mismatch!"
    print("  PASS")

    del engine
    torch.cuda.empty_cache()


def test_multi_decode_multi_prefill(model_dir):
    """Multiple decode + multiple prefill in same batch."""
    print("Test 4: multi-decode + multi-prefill...")
    engine = MoEEngine(model_dir, max_batch_size=8, max_seq_len=512,
                       use_torch_compile=False)
    engine.capture_prefill_cuda_graph(
        total_token_sizes=[64], use_torch_compile=False)

    prompts = [
        torch.randint(1, 1000, (64,), device="cuda"),
        torch.randint(1, 1000, (64,), device="cuda"),
    ]
    prefill_prompts = [
        torch.randint(1, 1000, (32,), device="cuda"),
        torch.randint(1, 1000, (48,), device="cuda"),
    ]

    engine.reset()
    for i, p in enumerate(prompts):
        engine.prefill_to_slot(i, p)

    decode_tokens = torch.randint(1, 1000, (2,), device="cuda")
    mixed_logits = engine.mixed_step(
        [0, 1], decode_tokens,
        [2, 3], prefill_prompts)

    expected_total = 2 + 32 + 48
    assert mixed_logits.shape == (expected_total, engine.vocab_size), \
        f"Shape mismatch: {mixed_logits.shape} vs expected ({expected_total}, {engine.vocab_size})"

    assert engine.seq_lens[0].item() == 65, f"seq 0: {engine.seq_lens[0].item()}"
    assert engine.seq_lens[1].item() == 65, f"seq 1: {engine.seq_lens[1].item()}"
    assert engine.seq_lens[2].item() == 32, f"seq 2: {engine.seq_lens[2].item()}"
    assert engine.seq_lens[3].item() == 48, f"seq 3: {engine.seq_lens[3].item()}"

    print(f"  Output shape: {mixed_logits.shape} — correct")
    print(f"  seq_lens: {engine.seq_lens[:4].tolist()} — correct")
    print("  PASS")

    del engine
    torch.cuda.empty_cache()


# =====================================================================
#  Piecewise CUDA graph tests
# =====================================================================

def _run_eager_mixed(engine, decode_seq_ids, decode_token_ids,
                     prefill_seq_ids, prefill_input_ids):
    """Run mixed_step in eager mode (bypass piecewise dispatch)."""
    saved = getattr(engine, '_piecewise_graphs', None)
    engine._piecewise_graphs = {}
    logits = engine.mixed_step(decode_seq_ids, decode_token_ids,
                               prefill_seq_ids, prefill_input_ids)
    if saved is not None:
        engine._piecewise_graphs = saved
    else:
        del engine._piecewise_graphs
    return logits


def test_piecewise_graphed_vs_eager(model_dir):
    """Piecewise graphed mixed_step matches eager exactly (no torch.compile)."""
    print("Test 5: piecewise graphed vs eager (exact match)...")
    engine = MoEEngine(model_dir, max_batch_size=8, max_seq_len=512,
                       use_torch_compile=False)

    engine.capture_mixed_cuda_graphs(
        total_token_sizes=[128, 256], use_torch_compile=False)

    prompt = torch.tensor([50256, 510, 5765, 273, 6181, 310], device="cuda")

    engine.reset()
    _run_eager_mixed(engine, [], torch.empty(0, dtype=torch.long, device="cuda"),
                     [0], [prompt])
    _run_eager_mixed(engine, [], torch.empty(0, dtype=torch.long, device="cuda"),
                     [1], [prompt])

    seq_lens_snap = engine._seq_lens_cpu[:2].clone()
    k_snap = engine.k_cache.clone()
    v_snap = engine.v_cache.clone()

    # Eager mixed step
    decode_tok = torch.tensor([engine.lm_head.shape[0] - 1], device="cuda",
                              dtype=torch.long)
    prefill_ids = torch.tensor([50256, 510, 5765, 273], device="cuda")

    eager_logits = _run_eager_mixed(engine, [0], decode_tok, [1], [prefill_ids])

    # Restore state and run graphed
    engine._seq_lens_cpu[:2] = seq_lens_snap
    engine.seq_lens[:2] = seq_lens_snap.to("cuda")
    engine.k_cache.copy_(k_snap)
    engine.v_cache.copy_(v_snap)

    graph_logits = engine.mixed_step([0], decode_tok, [1], [prefill_ids])

    match = torch.equal(eager_logits, graph_logits)
    max_diff = (eager_logits - graph_logits).abs().max().item()
    assert match, f"Piecewise graphed != eager: max diff = {max_diff}"
    print(f"  PASS — exact match, max diff = {max_diff}")

    del engine
    torch.cuda.empty_cache()


def test_piecewise_padding(model_dir):
    """Padding correctness: capture at N=128, run with N_actual < 128."""
    print("Test 6: piecewise padding correctness...")
    engine = MoEEngine(model_dir, max_batch_size=8, max_seq_len=512,
                       use_torch_compile=False)
    engine.capture_mixed_cuda_graphs(
        total_token_sizes=[128], use_torch_compile=False)

    prompt1 = torch.tensor([50256, 510, 5765, 273, 6181, 310], device="cuda")
    prompt2 = torch.tensor([50256, 510, 5765, 273], device="cuda")

    # Eager
    engine.reset()
    eager_logits = _run_eager_mixed(
        engine, [], torch.empty(0, dtype=torch.long, device="cuda"),
        [0, 1], [prompt1, prompt2])

    # Graphed (will pad to 128)
    engine.reset()
    graph_logits = engine.mixed_step(
        [], torch.empty(0, dtype=torch.long, device="cuda"),
        [0, 1], [prompt1, prompt2])

    match = torch.equal(eager_logits, graph_logits)
    max_diff = (eager_logits - graph_logits).abs().max().item()
    graph_N = engine._find_nearest_piecewise_graph(10)
    assert match, f"Padded piecewise != eager: max diff = {max_diff}"
    print(f"  PASS — N_actual=10, graph_N={graph_N}, max diff = {max_diff}")

    del engine
    torch.cuda.empty_cache()


def test_piecewise_multi_step(model_dir):
    """Multi-step generation: piecewise matches eager across 5 steps."""
    print("Test 7: piecewise multi-step generation...")
    engine = MoEEngine(model_dir, max_batch_size=8, max_seq_len=512,
                       use_torch_compile=False)
    engine.capture_mixed_cuda_graphs(
        total_token_sizes=[128], use_torch_compile=False)

    prompt = torch.tensor([50256, 510, 5765, 273, 6181, 310], device="cuda")

    # Prefill 2 sequences eagerly
    engine.reset()
    _run_eager_mixed(engine, [],
                     torch.empty(0, dtype=torch.long, device="cuda"),
                     [0], [prompt])
    _run_eager_mixed(engine, [],
                     torch.empty(0, dtype=torch.long, device="cuda"),
                     [1], [prompt])

    seq_lens_snap = engine._seq_lens_cpu[:2].clone()
    k_snap = engine.k_cache.clone()
    v_snap = engine.v_cache.clone()
    seq_lens_gpu_snap = engine.seq_lens[:2].clone()

    # Eager: 5 decode steps
    eager_generated = []
    next_toks = torch.tensor([100, 200], device="cuda", dtype=torch.long)
    for _ in range(5):
        logits = _run_eager_mixed(engine, [0, 1], next_toks, [], [])
        next_toks = logits.argmax(dim=-1)
        eager_generated.append(next_toks.clone())

    # Restore and run graphed
    engine._seq_lens_cpu[:2] = seq_lens_snap
    engine.seq_lens[:2] = seq_lens_gpu_snap
    engine.k_cache.copy_(k_snap)
    engine.v_cache.copy_(v_snap)

    graph_generated = []
    next_toks = torch.tensor([100, 200], device="cuda", dtype=torch.long)
    for _ in range(5):
        logits = engine.mixed_step([0, 1], next_toks, [], [])
        next_toks = logits.argmax(dim=-1)
        graph_generated.append(next_toks.clone())

    all_match = all(torch.equal(e, g)
                    for e, g in zip(eager_generated, graph_generated))
    assert all_match, "Multi-step piecewise != eager"
    print(f"  PASS — 5 steps, tokens match: "
          f"{[g.tolist() for g in graph_generated]}")

    del engine
    torch.cuda.empty_cache()


# =====================================================================
#  Subcommand: decode
# =====================================================================

def cmd_decode(args):
    run_correctness = not args.perf_only
    run_perf = not args.correctness_only
    run_custom = not args.vllm_only
    run_vllm = not args.custom_only

    seq_lens = [s for s in PERF_SEQ_LENS if s <= args.max_seq]

    custom_tokens = {}
    custom_logits = {}
    custom_gen_ms = {}
    vllm_tokens = {}
    vllm_logprobs = {}
    vllm_gen_ms = {}
    custom_perf = {}
    vllm_perf = {}

    if run_custom:
        print("=" * 70)
        print("CUSTOM ENGINE")
        print("=" * 70)

        max_needed = max(seq_lens) + 256 if run_perf else 1024
        engine = MoEEngine(args.model, max_batch_size=1, max_seq_len=max_needed,
                           use_torch_compile=False)

        if not args.no_graph:
            engine.capture_prefill_cuda_graph(
                total_token_sizes=[6, 8], use_torch_compile=False)
            engine.capture_decode_cuda_graph(
                batch_size=1, warmup_seq_len=128,
                max_decode_tokens=max_needed - 128)
            print("CUDA graph captured")

        engine.reset()
        warmup_ids = torch.tensor([[100, 200, 300, 400]], device="cuda")
        engine.generate(warmup_ids, max_new_tokens=5)
        torch.cuda.synchronize()
        print("Warmup complete\n")

        if run_correctness:
            print("── Correctness (greedy generation) ──")
            for i, prompt in enumerate(CORRECTNESS_PROMPTS):
                tokens, logits = custom_greedy_generate(engine, prompt, MAX_NEW_TOKENS)
                custom_tokens[i] = tokens
                custom_logits[i] = logits

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

    if run_vllm:
        print("=" * 70)
        print("vLLM")
        print("=" * 70)

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
        )

        sp_warmup = SamplingParams(max_tokens=10, temperature=0)
        for j in range(3):
            warmup_ids = torch.randint(1, 1000, (256,)).tolist()
            llm.generate([{"prompt_token_ids": warmup_ids}],
                         sampling_params=sp_warmup)
        torch.cuda.synchronize()
        print("Warmup complete\n")

        if run_correctness:
            print("── Correctness (greedy generation) ──")
            for i, prompt in enumerate(CORRECTNESS_PROMPTS):
                tokens, logprobs = vllm_greedy_generate(llm, prompt, MAX_NEW_TOKENS)
                vllm_tokens[i] = tokens
                vllm_logprobs[i] = logprobs

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

    # Combined results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    if run_correctness and run_custom and run_vllm:
        print("\n── Correctness ──")
        total_prompts = 0
        total_exact = 0
        for i in sorted(set(custom_tokens) & set(vllm_tokens)):
            ct = custom_tokens[i]
            vt = vllm_tokens[i]
            cl = custom_logits.get(i, [])
            vl = vllm_logprobs.get(i, [])
            min_len = min(len(ct), len(vt))
            total_prompts += 1

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

                if match_len < len(cl):
                    step_logits = cl[match_len]
                    top_tid = max(step_logits, key=step_logits.get)
                    top_val = step_logits[top_tid]
                    second_tid = sorted(step_logits, key=step_logits.get, reverse=True)[1]
                    second_val = step_logits[second_tid]
                    print(f"    Custom engine logits at step {match_len}:")
                    print(f"      top-1: token {top_tid} = {top_val:.4f}")
                    print(f"      top-2: token {second_tid} = {second_val:.4f}")
                    print(f"      gap (top1 - top2): {top_val - second_val:.4f}")
                    v_logit = step_logits.get(vllm_tok)
                    if v_logit is not None:
                        rank = sorted(step_logits, key=step_logits.get, reverse=True).index(vllm_tok) + 1
                        print(f"      vllm's token {vllm_tok}: logit = {v_logit:.4f} "
                              f"(rank in custom top-10: {rank})")
                    else:
                        print(f"      vllm's token {vllm_tok}: not in custom top-10")

                if match_len < len(vl):
                    step_lp = vl[match_len]
                    print(f"    vLLM logprobs at step {match_len}:")
                    for tid in sorted(step_lp, key=step_lp.get, reverse=True)[:3]:
                        marker = " <-- chosen" if tid == vllm_tok else ""
                        print(f"      token {tid}: logprob = {step_lp[tid]:.4f}{marker}")

        print(f"\n  {total_exact}/{total_prompts} prompts exact match")

    if run_correctness and custom_gen_ms and vllm_gen_ms:
        print("\n── End-to-End Generation Time ──")
        print(f"  {'Prompt':>8s}  {'Tokens':>6s}  {'Custom ms':>10s}  {'vLLM ms':>10s}  {'Speedup':>10s}")
        print("  " + "-" * 50)
        for i in sorted(set(custom_gen_ms) & set(vllm_gen_ms)):
            n_tok = len(custom_tokens[i])
            c_ms = custom_gen_ms[i]
            v_ms = vllm_gen_ms[i]
            print(f"  {i:>8d}  {n_tok:>6d}  {c_ms:>10.0f}  {v_ms:>10.0f}  {v_ms/c_ms:>9.2f}x")

    if run_perf and custom_perf and vllm_perf:
        print("\n── Performance (ms/step, batch=1) ──")
        print(f"{'seq_len':>10s}  {'Custom':>10s}  {'vLLM':>10s}  {'Speedup':>10s}")
        print("-" * 45)
        for sl in seq_lens:
            c = custom_perf.get(sl)
            v = vllm_perf.get(sl)
            if c is not None and v is not None:
                print(f"{sl:>10,d}  {c:>10.2f}  {v:>10.2f}  {v/c:>9.2f}x")
            elif c is not None:
                print(f"{sl:>10,d}  {c:>10.2f}  {'—':>10s}  {'—':>10s}")
            elif v is not None:
                print(f"{sl:>10,d}  {'—':>10s}  {v:>10.2f}  {'—':>10s}")

    print()


# =====================================================================
#  Subcommand: prefill
# =====================================================================

def cmd_prefill(args):
    print("=" * 80)
    print("  PREFILL BENCHMARK: Custom (CUDA graph / compile+graph) vs vLLM")
    print("=" * 80)

    # CUDA graph only (no torch.compile)
    print("\n── Custom Engine Prefill (CUDA graph, no compile) ──")
    engine = MoEEngine(args.model, max_batch_size=4, max_seq_len=4096,
                       use_torch_compile=False)
    custom_graph = benchmark_prefill_custom_graph(
        engine, args.seq_lens, use_torch_compile=False)
    del engine
    torch.cuda.empty_cache()

    # torch.compile + CUDA graph
    print("\n── Custom Engine Prefill (torch.compile + CUDA graph) ──")
    engine = MoEEngine(args.model, max_batch_size=4, max_seq_len=4096,
                       use_torch_compile=True)
    custom_compiled = benchmark_prefill_custom_graph(
        engine, args.seq_lens, use_torch_compile=True)

    # Combined prefill+decode
    combos = [(128, 50), (512, 50), (2048, 50)]
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

    # vLLM
    vllm_results = {}
    if not args.skip_vllm:
        print("\n── vLLM Prefill ──")
        vllm_results = benchmark_prefill_vllm(args.model, args.seq_lens)

    # Comparison table
    print("\n" + "=" * 80)
    print("  PREFILL COMPARISON (median ms, batch=1)")
    print("=" * 80)

    header = f"  {'seq_len':>8s}  {'Graph':>10s}  {'Compile+G':>10s}  {'vLLM':>10s}  {'C+G vs vLLM':>12s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for sl in args.seq_lens:
        g = custom_graph.get(sl, float('nan'))
        c = custom_compiled.get(sl, float('nan'))
        v = vllm_results.get(sl, float('nan'))
        delta = c - v if (v == v and c == c) else float('nan')
        pct = delta / v * 100 if (v == v and v > 0 and c == c) else float('nan')

        line = f"  {sl:>8d}"
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


# =====================================================================
#  Subcommand: cuda-graph
# =====================================================================

def cmd_cuda_graph(args):
    run_cuda_graph_correctness(args.model)
    run_cuda_graph_timing(args.model)


# =====================================================================
#  Subcommand: mixed
# =====================================================================

def cmd_mixed(args):
    print("=" * 70)
    print("MIXED STEP SMOKE TESTS")
    print("=" * 70)

    tests = {
        "pure-decode": test_pure_decode_via_mixed,
        "pure-prefill": test_pure_prefill_via_mixed,
        "mixed": test_mixed_batch,
        "multi": test_multi_decode_multi_prefill,
        "piecewise": test_piecewise_graphed_vs_eager,
        "padding": test_piecewise_padding,
        "multi-step": test_piecewise_multi_step,
    }

    if args.test == "all":
        for test_fn in tests.values():
            test_fn(args.model)
    else:
        tests[args.test](args.model)

    print("\nAll tests passed!")


# =====================================================================
#  Subcommand: all
# =====================================================================

def cmd_all(args):
    print("Running all benchmarks...\n")

    # Correctness tests first
    print("=" * 70)
    print("PHASE 1: Correctness Tests")
    print("=" * 70)

    run_cuda_graph_correctness(args.model)
    print()
    test_pure_decode_via_mixed(args.model)
    test_pure_prefill_via_mixed(args.model)
    test_mixed_batch(args.model)
    test_multi_decode_multi_prefill(args.model)
    test_piecewise_graphed_vs_eager(args.model)
    test_piecewise_padding(args.model)
    test_piecewise_multi_step(args.model)
    print()

    # Performance benchmarks
    print("=" * 70)
    print("PHASE 2: Performance Benchmarks")
    print("=" * 70)

    run_cuda_graph_timing(args.model)


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Microbenchmarks for custom MoE engine")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path to HuggingFace model directory")
    subparsers = parser.add_subparsers(dest="command")

    # decode
    p_decode = subparsers.add_parser("decode",
        help="Correctness + decode performance vs vLLM")
    p_decode.add_argument("--correctness-only", action="store_true")
    p_decode.add_argument("--perf-only", action="store_true")
    p_decode.add_argument("--custom-only", action="store_true")
    p_decode.add_argument("--vllm-only", action="store_true")
    p_decode.add_argument("--max-seq", type=int, default=2048)
    p_decode.add_argument("--n-steps", type=int, default=N_DECODE_STEPS)
    p_decode.add_argument("--n-trials", type=int, default=N_TRIALS)
    p_decode.add_argument("--no-graph", action="store_true")

    # prefill
    p_prefill = subparsers.add_parser("prefill",
        help="Prefill performance vs vLLM")
    p_prefill.add_argument("--seq-lens", nargs="+", type=int,
                           default=PREFILL_SEQ_LENS)
    p_prefill.add_argument("--skip-vllm", action="store_true")
    p_prefill.add_argument("--skip-combined", action="store_true")

    # cuda-graph
    subparsers.add_parser("cuda-graph",
        help="CUDA graph correctness + timing")

    # mixed
    p_mixed = subparsers.add_parser("mixed",
        help="mixed_step smoke tests")
    p_mixed.add_argument("--test", default="all",
        choices=["all", "pure-decode", "pure-prefill", "mixed", "multi",
                 "piecewise", "padding", "multi-step"])

    # all
    subparsers.add_parser("all", help="Run all benchmarks")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "decode": cmd_decode,
        "prefill": cmd_prefill,
        "cuda-graph": cmd_cuda_graph,
        "mixed": cmd_mixed,
        "all": cmd_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
