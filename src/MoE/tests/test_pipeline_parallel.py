#!/usr/bin/env -S python3 -u
"""Test pipeline parallelism: correctness, performance, and trace collection.

Test 1: Logits match — PP=2 vs single-GPU offloaded (epl=4) on Mixtral-8x7B-32L
        for prefill, decode, and mixed batch.
Test 2: Performance — compare step times PP=2 vs single-GPU offloaded (epl=4).
Test 3: Trace collection — verify TraceRecorder produces correct traces with PP.
Test 4: Memory — print per-GPU memory after model load.

Usage:
    python tests/test_pipeline_parallel.py [--model PATH]
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from moe_engine import MoEEngine

DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "Mixtral-8x7B")


def test_correctness(model_path, epl=4):
    """Compare PP=2 logits vs single-GPU offloaded logits."""
    print("\n" + "=" * 70)
    print("TEST 1: Correctness — PP=2 vs single-GPU offloaded (epl={})".format(epl))
    print("=" * 70)

    seq_len = 128
    prompt = torch.randint(1, 1000, (seq_len,), device="cuda:0")

    # ── PP=2 ──
    print("\nLoading PP=2 engine...")
    engine_pp = MoEEngine(model_path, max_seqs=4,
                          max_seq_len=2048 + 128,
                          pipeline_parallel_size=2,
                          use_torch_compile=False)

    with torch.inference_mode():
        engine_pp.capture_cuda_graphs(
            total_token_sizes=[1, 128, 256, 512], use_torch_compile=False)

        # Prefill
        engine_pp.reset()
        logits_pp_prefill = engine_pp.step(
            decode_seq_ids=[],
            decode_token_ids=torch.empty(0, dtype=torch.long, device=engine_pp.device),
            prefill_seq_ids=[0],
            prefill_input_ids=[prompt.to(engine_pp.device)])

        # Decode
        next_tok = logits_pp_prefill[-1].argmax().unsqueeze(0)
        logits_pp_decode = engine_pp.step(
            decode_seq_ids=[0],
            decode_token_ids=next_tok,
            prefill_seq_ids=[],
            prefill_input_ids=[])

        # Mixed batch (decode slot 0 + prefill slot 1)
        prompt2 = torch.randint(1, 1000, (64,), device=engine_pp.device)
        next_tok2 = logits_pp_decode[0].argmax().unsqueeze(0)
        logits_pp_mixed = engine_pp.step(
            decode_seq_ids=[0],
            decode_token_ids=next_tok2,
            prefill_seq_ids=[1],
            prefill_input_ids=[prompt2])

    # Save results, free GPU memory
    pp_prefill = logits_pp_prefill.cpu().clone()
    pp_decode = logits_pp_decode.cpu().clone()
    pp_mixed = logits_pp_mixed.cpu().clone()
    del engine_pp
    gc.collect()
    torch.cuda.empty_cache()

    # ── Single-GPU offloaded ──
    print(f"\nLoading single-GPU engine (epl={epl})...")
    engine_1g = MoEEngine(model_path, max_seqs=4,
                          max_seq_len=2048 + 128,
                          experts_per_layer=epl,
                          use_torch_compile=False)

    with torch.inference_mode():
        engine_1g.capture_cuda_graphs(
            total_token_sizes=[1, 128, 256, 512], use_torch_compile=False)

        # Prefill
        engine_1g.reset()
        logits_1g_prefill = engine_1g.step(
            decode_seq_ids=[],
            decode_token_ids=torch.empty(0, dtype=torch.long, device=engine_1g.device),
            prefill_seq_ids=[0],
            prefill_input_ids=[prompt.to(engine_1g.device)])

        # Decode
        next_tok = logits_1g_prefill[-1].argmax().unsqueeze(0)
        logits_1g_decode = engine_1g.step(
            decode_seq_ids=[0],
            decode_token_ids=next_tok,
            prefill_seq_ids=[],
            prefill_input_ids=[])

        # Mixed batch
        prompt2_1g = prompt2.to(engine_1g.device)
        next_tok2 = logits_1g_decode[0].argmax().unsqueeze(0)
        logits_1g_mixed = engine_1g.step(
            decode_seq_ids=[0],
            decode_token_ids=next_tok2,
            prefill_seq_ids=[1],
            prefill_input_ids=[prompt2_1g])

    sg_prefill = logits_1g_prefill.cpu()
    sg_decode = logits_1g_decode.cpu()
    sg_mixed = logits_1g_mixed.cpu()

    # ── Compare ──
    all_pass = True
    for name, pp_l, sg_l in [
        ("Prefill", pp_prefill, sg_prefill),
        ("Decode", pp_decode, sg_decode),
        ("Mixed", pp_mixed, sg_mixed),
    ]:
        diff = (pp_l.float() - sg_l.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        top1_match = (pp_l.argmax(dim=-1) == sg_l.argmax(dim=-1)).float().mean().item()

        status = "PASS" if top1_match > 0.95 else "FAIL"
        if top1_match <= 0.95:
            all_pass = False
        print(f"\n  {name}:")
        print(f"    Max diff:     {max_diff:.4f}")
        print(f"    Mean diff:    {mean_diff:.6f}")
        print(f"    Top-1 match:  {top1_match:.1%}  [{status}]")

    del engine_1g
    gc.collect()
    torch.cuda.empty_cache()
    return all_pass


def _sync_all():
    """Synchronize all CUDA devices."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def test_performance(model_path, epl=4, n_warmup=3, n_trials=10):
    """Compare step times: PP=2 vs single-GPU offloaded."""
    print("\n" + "=" * 70)
    print("TEST 2: Performance — PP=2 vs single-GPU offloaded (epl={})".format(epl))
    print("=" * 70)

    seq_len = 128
    prompt = torch.randint(1, 1000, (seq_len,), device="cuda:0")

    # ── PP=2 ──
    print("\nLoading PP=2 engine...")
    engine_pp = MoEEngine(model_path, max_seqs=4,
                          max_seq_len=2048 + 128,
                          pipeline_parallel_size=2,
                          use_torch_compile=False)

    with torch.inference_mode():
        engine_pp.capture_cuda_graphs(
            total_token_sizes=[1, 128, 256, 512], use_torch_compile=False)

        # Prefill timing (wall clock with full sync for multi-GPU)
        pp_prefill_times = []
        for i in range(n_warmup + n_trials):
            engine_pp.reset()
            _sync_all()
            t0 = time.perf_counter()
            engine_pp.step(
                decode_seq_ids=[],
                decode_token_ids=torch.empty(0, dtype=torch.long, device="cuda:0"),
                prefill_seq_ids=[0],
                prefill_input_ids=[prompt])
            _sync_all()
            if i >= n_warmup:
                pp_prefill_times.append((time.perf_counter() - t0) * 1000)

        # Decode timing
        pp_decode_times = []
        for i in range(n_warmup + n_trials):
            engine_pp.reset()
            logits = engine_pp.step(
                decode_seq_ids=[],
                decode_token_ids=torch.empty(0, dtype=torch.long, device="cuda:0"),
                prefill_seq_ids=[0],
                prefill_input_ids=[prompt])
            next_tok = logits[-1].argmax().unsqueeze(0)
            _sync_all()
            t0 = time.perf_counter()
            engine_pp.step(
                decode_seq_ids=[0],
                decode_token_ids=next_tok,
                prefill_seq_ids=[],
                prefill_input_ids=[])
            _sync_all()
            if i >= n_warmup:
                pp_decode_times.append((time.perf_counter() - t0) * 1000)

    del engine_pp
    gc.collect()
    torch.cuda.empty_cache()

    # ── Single-GPU offloaded ──
    print(f"\nLoading single-GPU engine (epl={epl})...")
    engine_1g = MoEEngine(model_path, max_seqs=4,
                          max_seq_len=2048 + 128,
                          experts_per_layer=epl,
                          use_torch_compile=False)

    with torch.inference_mode():
        engine_1g.capture_cuda_graphs(
            total_token_sizes=[1, 128, 256, 512], use_torch_compile=False)

        # Prefill timing
        sg_prefill_times = []
        for i in range(n_warmup + n_trials):
            engine_1g.reset()
            _sync_all()
            t0 = time.perf_counter()
            engine_1g.step(
                decode_seq_ids=[],
                decode_token_ids=torch.empty(0, dtype=torch.long, device="cuda:0"),
                prefill_seq_ids=[0],
                prefill_input_ids=[prompt])
            _sync_all()
            if i >= n_warmup:
                sg_prefill_times.append((time.perf_counter() - t0) * 1000)

        # Decode timing
        sg_decode_times = []
        for i in range(n_warmup + n_trials):
            engine_1g.reset()
            logits = engine_1g.step(
                decode_seq_ids=[],
                decode_token_ids=torch.empty(0, dtype=torch.long, device="cuda:0"),
                prefill_seq_ids=[0],
                prefill_input_ids=[prompt])
            next_tok = logits[-1].argmax().unsqueeze(0)
            _sync_all()
            t0 = time.perf_counter()
            engine_1g.step(
                decode_seq_ids=[0],
                decode_token_ids=next_tok,
                prefill_seq_ids=[],
                prefill_input_ids=[])
            _sync_all()
            if i >= n_warmup:
                sg_decode_times.append((time.perf_counter() - t0) * 1000)

    del engine_1g
    gc.collect()
    torch.cuda.empty_cache()

    # ── Report ──
    pp_prefill_avg = sum(pp_prefill_times) / len(pp_prefill_times)
    sg_prefill_avg = sum(sg_prefill_times) / len(sg_prefill_times)
    pp_decode_avg = sum(pp_decode_times) / len(pp_decode_times)
    sg_decode_avg = sum(sg_decode_times) / len(sg_decode_times)

    print(f"\n  Prefill (seq_len={seq_len}):")
    print(f"    PP=2:         {pp_prefill_avg:.2f} ms")
    print(f"    1-GPU epl={epl}: {sg_prefill_avg:.2f} ms")
    print(f"    Speedup:      {sg_prefill_avg / pp_prefill_avg:.2f}x")

    print(f"\n  Decode (1 token):")
    print(f"    PP=2:         {pp_decode_avg:.2f} ms")
    print(f"    1-GPU epl={epl}: {sg_decode_avg:.2f} ms")
    print(f"    Speedup:      {sg_decode_avg / pp_decode_avg:.2f}x")


def test_trace_collection(model_path):
    """Verify TraceRecorder produces correct traces with PP."""
    print("\n" + "=" * 70)
    print("TEST 3: Trace collection with PP=2 + TraceRecorder")
    print("=" * 70)

    from trace_recorder import TraceRecorder

    print("\nLoading PP=2 engine...")
    engine = MoEEngine(model_path, max_seqs=4,
                       max_seq_len=2048 + 128,
                       pipeline_parallel_size=2,
                       use_torch_compile=False)

    import json
    with open(Path(model_path) / "config.json") as f:
        cfg = json.load(f)
    num_layers = cfg["num_hidden_layers"]
    num_experts = (cfg.get("n_routed_experts") or cfg.get("num_experts")
                   or cfg.get("num_local_experts"))

    recorder = TraceRecorder(num_layers=num_layers, num_experts=num_experts)
    engine.trace_recorder = recorder

    with torch.inference_mode():
        engine.capture_cuda_graphs(
            total_token_sizes=[1, 128, 256, 512], use_torch_compile=False)

        # Run a short conversation
        prompt = torch.randint(1, 1000, (64,), device=engine.device)
        engine.reset()
        recorder.reset_trace()

        logits = engine.prefill_to_slot(0, prompt)
        next_tok = logits[-1].argmax().unsqueeze(0)

        n_decode = 10
        for step in range(n_decode):
            logits = engine.step(
                decode_seq_ids=[0],
                decode_token_ids=next_tok,
                prefill_seq_ids=[],
                prefill_input_ids=[])
            next_tok = logits[0].argmax().unsqueeze(0)

    trace = recorder.trace
    # Prefill = step 0 + decode steps 1..n_decode; dense layers are skipped
    first_moe = getattr(engine, 'first_k_dense_replace', 0)
    num_moe_layers = num_layers - first_moe
    expected_entries = (1 + n_decode) * num_moe_layers
    print(f"\n  Trace entries: {len(trace)} (expected {expected_entries})")
    assert len(trace) == expected_entries, \
        f"Expected {expected_entries} trace entries, got {len(trace)}"

    # Verify structure
    for entry in trace:
        assert 'step' in entry
        assert 'layer' in entry
        assert 'expert_ids' in entry
        assert all(0 <= e < num_experts for e in entry['expert_ids'])

    # Verify steps are monotonically ordered
    steps_seen = [e['step'] for e in trace]
    assert steps_seen == sorted(steps_seen), "Steps not monotonic"

    # Check that step 0 has all MoE layers (dense layers are not recorded)
    step0_layers = [e['layer'] for e in trace if e['step'] == 0]
    expected_layers = list(range(first_moe, num_layers))
    assert step0_layers == expected_layers, \
        f"Step 0 missing layers: {step0_layers}"

    print("  Structure:     OK (step, layer, expert_ids present)")
    print("  Expert range:  OK (all within [0, {})".format(num_experts))
    print("  Monotonicity:  OK")
    print("  Step 0 layers: OK ({0}..{1})".format(first_moe, num_layers - 1))

    # Timing: run 5 conversations, measure rate
    t0 = time.time()
    n_convos = 5
    for c in range(n_convos):
        prompt = torch.randint(1, 1000, (64,), device=engine.device)
        engine.reset()
        recorder.reset_trace()
        with torch.inference_mode():
            logits = engine.prefill_to_slot(0, prompt)
            next_tok = logits[-1].argmax().unsqueeze(0)
            for _ in range(10):
                logits = engine.step(
                    decode_seq_ids=[0],
                    decode_token_ids=next_tok,
                    prefill_seq_ids=[],
                    prefill_input_ids=[])
                next_tok = logits[0].argmax().unsqueeze(0)
    elapsed = time.time() - t0
    print(f"\n  {n_convos} conversations in {elapsed:.2f}s "
          f"({n_convos/elapsed:.1f} conv/s)")

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    print("\n  PASS")


def test_memory(model_path):
    """Print per-GPU memory after PP=2 model load."""
    print("\n" + "=" * 70)
    print("TEST 4: Memory — per-GPU allocation with PP=2")
    print("=" * 70)

    engine = MoEEngine(model_path, max_seqs=4,
                       max_seq_len=2048 + 128,
                       pipeline_parallel_size=2,
                       use_torch_compile=False)

    for i in range(2):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"\n  GPU {i}: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")

    del engine
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--epl", type=int, default=4,
                        help="Experts per layer for single-GPU comparison")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "correctness", "performance",
                                 "trace", "memory"],
                        help="Which test to run")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    if n_gpus < 2:
        print("ERROR: PP test requires at least 2 GPUs")
        sys.exit(1)

    if args.test in ("all", "memory"):
        test_memory(args.model)
        gc.collect()
        torch.cuda.empty_cache()

    if args.test in ("all", "correctness"):
        if not test_correctness(args.model, epl=args.epl):
            print("\nFAILED: correctness test")
            sys.exit(1)
        gc.collect()
        torch.cuda.empty_cache()

    if args.test in ("all", "trace"):
        test_trace_collection(args.model)
        gc.collect()
        torch.cuda.empty_cache()

    if args.test in ("all", "performance"):
        test_performance(args.model, epl=args.epl)
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
