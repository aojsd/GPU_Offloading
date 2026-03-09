#!/usr/bin/env -S python3 -u
"""Test dynamic page allocation correctness and performance.

Test 1: Page pool unit tests — alloc/free/ensure/reset bookkeeping.
Test 2: Correctness + performance — static vs dynamic allocation on a
        synthetic 2-conversation continuous batching workload.

Workload: two conversations arriving simultaneously with 256-token
chunked prefill.
  Conv A: 256 prompt tokens, 1000 decode tokens
  Conv B: 1020 prompt tokens, 997 decode tokens

Usage:
    # Single-GPU OLMoE
    python tests/test_dynamic_pages.py --model models/OLMoE-1B-7B

    # PP=2 Mixtral-8x7B (full 32 layers)
    python tests/test_dynamic_pages.py --model models/Mixtral-8x7B --pp 2
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from moe_engine import MoEEngine


DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "OLMoE-1B-7B")


# ── Test 1: Page pool unit tests ──────────────────────────────────────

def test_page_pool(engine):
    """Unit tests for alloc_pages/free_seq_pages/ensure_pages/reset."""
    budget = engine.total_pages
    ps = engine.page_size
    passed = 0
    failed = 0

    def check(condition, msg):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {msg}")

    print("Test 1: Page pool unit tests")

    # ── alloc_pages basic ──
    engine.reset()
    engine.alloc_pages(0, 4)
    check(len(engine._seq_page_list[0]) == 4,
          "alloc 4 pages: seq_page_list should have 4 entries")
    check(engine.pages_free == budget - 4,
          f"alloc 4 pages: pages_free should be {budget - 4}")
    check(engine.pages_in_use == 4,
          "alloc 4 pages: pages_in_use should be 4")

    # Verify block_table entries are valid (not -1)
    bt = engine.block_table[0] if engine.pp_size > 1 else engine.block_table
    bt_vals = bt[0, :4].cpu().tolist()
    check(all(v >= 0 for v in bt_vals),
          f"block_table[0, :4] should be >= 0, got {bt_vals}")
    check(bt[0, 4].item() == -1,
          f"block_table[0, 4] should be -1 (unallocated)")

    # ── ensure_pages no-op ──
    engine.ensure_pages(0, 4)
    check(engine.pages_free == budget - 4,
          "ensure_pages(0,4) no-op: pages_free unchanged")

    # ── ensure_pages grow ──
    engine.ensure_pages(0, 8)
    check(len(engine._seq_page_list[0]) == 8,
          "ensure_pages(0,8): should have 8 entries")
    check(engine.pages_free == budget - 8,
          f"ensure_pages(0,8): pages_free should be {budget - 8}")

    # ── alloc second seq, no overlap ──
    engine.alloc_pages(1, 10)
    seq0_pages = set(engine._seq_page_list[0])
    seq1_pages = set(engine._seq_page_list[1])
    check(len(seq0_pages & seq1_pages) == 0,
          f"seq 0 and 1 pages should not overlap: {seq0_pages} vs {seq1_pages}")
    check(engine.pages_free == budget - 18,
          f"after alloc seq1: pages_free should be {budget - 18}")

    # ── free_seq returns pages ──
    engine.free_seq(0)
    check(engine.pages_free == budget - 10,
          f"after free_seq(0): pages_free should be {budget - 10}")
    check(len(engine._seq_page_list[0]) == 0,
          "after free_seq(0): seq_page_list[0] should be empty")
    bt_row0 = bt[0, :8].cpu().tolist()
    check(all(v == -1 for v in bt_row0),
          f"after free_seq(0): block_table[0,:8] should all be -1, got {bt_row0}")

    # ── reset returns all pages ──
    engine.reset()
    check(engine.pages_free == budget,
          f"after reset: pages_free should be {budget}")
    check(all(len(p) == 0 for p in engine._seq_page_list),
          "after reset: all seq_page_lists should be empty")

    # ── exhaust pool ──
    engine.alloc_pages(0, budget)
    check(engine.pages_free == 0, "exhausted pool: pages_free should be 0")
    try:
        engine.alloc_pages(1, 1)
        check(False, "should have raised RuntimeError on exhausted pool")
    except RuntimeError:
        check(True, "RuntimeError raised on exhausted pool")

    # ── alloc with n_pages=0 is no-op ──
    engine.reset()
    engine.alloc_pages(0, 0)
    check(engine.pages_free == budget,
          "alloc 0 pages: no-op")

    # ── PP replication check ──
    if engine.pp_size > 1:
        engine.reset()
        engine.alloc_pages(0, 4)
        vals = [bt_g[0, :4].cpu().tolist() for bt_g in engine.block_table]
        check(vals[0] == vals[1],
              f"PP block_table replicas should match: {vals}")

    engine.reset()
    print(f"  {passed} passed, {failed} failed")
    return failed == 0


# ── Test 2: Correctness + performance ─────────────────────────────────

def run_batched_scenario(engine, dynamic):
    """Run 2-conversation continuous batching: A(256,1000) + B(1020,997).

    In dynamic mode, pages are allocated one at a time during decode
    (simulating unknown a-priori sequence length). Prefill chunks use
    bulk allocation since the chunk size is known.

    Args:
        engine: MoEEngine instance
        dynamic: if True, call alloc_pages/ensure_pages before each step

    Returns:
        logit_samples: dict step_idx -> logits tensor (CPU)
        step_times_ms: list of per-step GPU times for decode-only phase
    """
    dev = engine.device
    page_size = engine.page_size
    CHUNK = 256

    torch.manual_seed(42)
    prompt_a = torch.randint(1, 30000, (256,), device=dev)
    prompt_b = torch.randint(1, 30000, (1020,), device=dev)

    sid_a, sid_b = 0, 1
    logit_samples = {}
    step_times = []

    engine.reset()

    with torch.inference_mode():
        # ── Step 0: Both prefill first 256-token chunk ──
        if dynamic:
            # Bulk alloc for known prefill chunk size
            engine.alloc_pages(sid_a, math.ceil(256 / page_size))
            engine.alloc_pages(sid_b, math.ceil(256 / page_size))

        logits = engine.mixed_step(
            decode_seq_ids=[],
            decode_token_ids=torch.empty(0, dtype=torch.long, device=dev),
            prefill_seq_ids=[sid_a, sid_b],
            prefill_input_ids=[prompt_a, prompt_b[:256]])

        # A's first decode token from its last prefill logit
        next_a = logits[255].argmax().unsqueeze(0)
        logit_samples[0] = logits.cpu()

        # ── Steps 1-3: A decodes, B continues ──
        b_offset = 256
        last_b_chunk_len = 0
        for step in range(1, 4):
            b_end = min(b_offset + CHUNK, 1020)
            b_chunk = prompt_b[b_offset:b_end]
            last_b_chunk_len = b_end - b_offset

            if dynamic:
                # A grows by 1 decode token
                new_a_len = engine._seq_lens_cpu[sid_a].item() + 1
                engine.ensure_pages(
                    sid_a, math.ceil(new_a_len / page_size))
                # B grows by a known continuation chunk
                engine.ensure_pages(
                    sid_b, math.ceil(b_end / page_size))

            logits = engine.mixed_step(
                decode_seq_ids=[sid_a],
                decode_token_ids=next_a,
                prefill_seq_ids=[],
                prefill_input_ids=[],
                continuation_seq_ids=[sid_b],
                continuation_input_ids=[b_chunk],
                continuation_offsets=[b_offset])

            next_a = logits[0].argmax().unsqueeze(0)
            b_offset = b_end

        # B's first decode token: last logit of its last continuation chunk
        # Layout: [D=1 | C=last_b_chunk_len], B's last logit at index
        # last_b_chunk_len (= D + C - 1 = 1 + C - 1 = C)
        next_b = logits[last_b_chunk_len].argmax().unsqueeze(0)
        logit_samples[3] = logits.cpu()

        # ── Steps 4-999: Both decode (996 steps) ──
        # Allocate one page at a time as sequences cross page boundaries
        decode_tokens = torch.cat([next_a, next_b])
        for step in range(4, 1000):
            if dynamic:
                for sid in [sid_a, sid_b]:
                    cur_len = engine._seq_lens_cpu[sid].item()
                    # After this step, seq will have cur_len + 1 tokens
                    needed = math.ceil((cur_len + 1) / page_size)
                    have = len(engine._seq_page_list[sid])
                    if needed > have:
                        engine.alloc_pages(sid, 1)

            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            logits = engine.mixed_step(
                decode_seq_ids=[sid_a, sid_b],
                decode_token_ids=decode_tokens,
                prefill_seq_ids=[],
                prefill_input_ids=[])
            t1.record()
            torch.cuda.synchronize()
            step_times.append(t0.elapsed_time(t1))

            decode_tokens = logits[:2].argmax(dim=-1)
            if step in (4, 500, 999):
                logit_samples[step] = logits.cpu()

        # ── Steps 1000-1002: A decodes alone, B freed ──
        engine.free_seq(sid_b)
        next_a = decode_tokens[0:1]
        for step in range(1000, 1003):
            if dynamic:
                cur_len = engine._seq_lens_cpu[sid_a].item()
                needed = math.ceil((cur_len + 1) / page_size)
                have = len(engine._seq_page_list[sid_a])
                if needed > have:
                    engine.alloc_pages(sid_a, 1)
            logits = engine.mixed_step(
                decode_seq_ids=[sid_a],
                decode_token_ids=next_a,
                prefill_seq_ids=[],
                prefill_input_ids=[])
            next_a = logits[0].argmax().unsqueeze(0)
            if step == 1002:
                logit_samples[1002] = logits.cpu()

        engine.free_seq(sid_a)

    return logit_samples, step_times


def test_static_vs_dynamic(model_path, pp_size, use_compile):
    """Compare static vs dynamic page allocation: correctness + performance."""
    max_seqs = 2
    max_seq_len = 2048
    page_size = 16
    max_pages_per_seq = math.ceil(max_seq_len / page_size)  # 128
    static_total_pages = max_seqs * max_pages_per_seq        # 256

    graph_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 288, 320, 384, 512]

    # ── Run 1: Static allocation (baseline) ──
    print("\n── Static allocation ──")
    print(f"Creating engine: PP={pp_size}, compile={use_compile}")
    t0 = time.time()
    engine_s = MoEEngine(
        model_path, max_seqs=max_seqs, max_seq_len=max_seq_len,
        page_size=page_size, use_torch_compile=use_compile,
        pipeline_parallel_size=pp_size)
    print(f"Engine created in {time.time() - t0:.1f}s")

    print("Capturing CUDA graphs...")
    engine_s.capture_mixed_cuda_graphs(graph_sizes)

    print("Running scenario (static)...")
    logits_s, times_s = run_batched_scenario(engine_s, dynamic=False)
    del engine_s
    torch.cuda.empty_cache()

    # ── Run 2: Dynamic allocation ──
    print("\n── Dynamic allocation ──")
    print(f"Creating engine: PP={pp_size}, compile={use_compile}, "
          f"kv_page_budget={static_total_pages}")
    t0 = time.time()
    engine_d = MoEEngine(
        model_path, max_seqs=max_seqs, max_seq_len=max_seq_len,
        page_size=page_size, use_torch_compile=use_compile,
        pipeline_parallel_size=pp_size,
        kv_page_budget=static_total_pages)
    print(f"Engine created in {time.time() - t0:.1f}s")

    print("Capturing CUDA graphs...")
    engine_d.capture_mixed_cuda_graphs(graph_sizes)

    print("Running scenario (dynamic)...")
    logits_d, times_d = run_batched_scenario(engine_d, dynamic=True)

    # Verify page pool fully freed after scenario
    assert engine_d.pages_free == static_total_pages, (
        f"All pages should be free after scenario: "
        f"{engine_d.pages_free} != {static_total_pages}")
    del engine_d
    torch.cuda.empty_cache()

    # ── Compare results ──
    print("\n── Results ──")
    all_pass = True

    # Correctness: compare logits at sampled steps
    sample_steps = sorted(set(logits_s.keys()) & set(logits_d.keys()))
    print(f"\nLogit comparison at steps: {sample_steps}")
    for step in sample_steps:
        ls = logits_s[step]
        ld = logits_d[step]
        # Only compare matching dimensions (step 1002 has different layout
        # but both should have 1 token)
        min_len = min(ls.shape[0], ld.shape[0])
        ls_cmp = ls[:min_len]
        ld_cmp = ld[:min_len]

        max_diff = (ls_cmp - ld_cmp).abs().max().item()
        top1_s = ls_cmp.argmax(dim=-1)
        top1_d = ld_cmp.argmax(dim=-1)
        top1_match = (top1_s == top1_d).all().item()

        status = "PASS" if top1_match else "FAIL"
        if not top1_match:
            all_pass = False
        print(f"  Step {step:4d}: max_diff={max_diff:.6f}  "
              f"top1_match={top1_match}  [{status}]")

    # Performance: compare decode step times (skip first 10 for warmup)
    warmup = 10
    if len(times_s) > warmup and len(times_d) > warmup:
        mean_s = sum(times_s[warmup:]) / len(times_s[warmup:])
        mean_d = sum(times_d[warmup:]) / len(times_d[warmup:])
        ratio = mean_d / mean_s if mean_s > 0 else float('inf')
        pct_diff = (ratio - 1.0) * 100

        print(f"\nDecode step latency (steps {4 + warmup}–999):")
        print(f"  Static:  {mean_s:.3f} ms/step")
        print(f"  Dynamic: {mean_d:.3f} ms/step")
        print(f"  Ratio:   {ratio:.4f}x ({pct_diff:+.2f}%)")

        if abs(pct_diff) > 5.0:
            print(f"  WARNING: >5% difference ({pct_diff:+.2f}%)")
            # Don't fail on timing — it's informational
    else:
        print("\nInsufficient timing data (skipped)")

    if all_pass:
        print("\nTest 2 PASSED: all logit comparisons match")
    else:
        print("\nTest 2 FAILED: logit mismatch detected")
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Test dynamic page allocation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallel size (1 or 2)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"PP size: {args.pp}")
    print(f"torch.compile: {not args.no_compile}")

    # ── Test 1: Page pool unit tests ──
    # Create a small engine just for pool tests
    print("\n" + "=" * 60)
    engine = MoEEngine(
        args.model, max_seqs=8, max_seq_len=1600, page_size=16,
        use_torch_compile=False,
        pipeline_parallel_size=args.pp,
        kv_page_budget=100)
    t1_pass = test_page_pool(engine)
    del engine
    torch.cuda.empty_cache()

    # ── Test 2: Static vs dynamic correctness + performance ──
    print("\n" + "=" * 60)
    print("Test 2: Static vs dynamic allocation — correctness + performance")
    t2_pass = test_static_vs_dynamic(
        args.model, args.pp, use_compile=not args.no_compile)

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"Test 1 (page pool):    {'PASS' if t1_pass else 'FAIL'}")
    print(f"Test 2 (static vs dyn): {'PASS' if t2_pass else 'FAIL'}")
    if t1_pass and t2_pass:
        print("All tests PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
