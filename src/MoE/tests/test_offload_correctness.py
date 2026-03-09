#!/usr/bin/env -S python3 -u
"""Verify expert offloading correctness and measure overhead.

Test 1: Latency overhead of split stage4 (no offload engine) vs flat prefill graph
Test 2: Output correctness when experts are demand-loaded vs all-resident
Test 3: Multi-step decode correctness with demand loading

Usage:
    python tests/test_offload_correctness.py [--model PATH]
"""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from moe_engine import MoEEngine

DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "Mixtral-8x7B-20L")


def benchmark_split_overhead(engine, seq_len=128, n_warmup=5, n_trials=20):
    """Measure latency: flat prefill graph vs piecewise (split stage4, no offload engine)."""
    print(f"\n=== Latency overhead: flat vs piecewise (seq_len={seq_len}) ===")

    prompt = torch.randint(1, 1000, (seq_len,), device=engine.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Flat prefill graph
    flat_times = []
    for i in range(n_warmup + n_trials):
        engine.reset()
        start.record()
        engine.prefill_to_slot(0, prompt)
        end.record()
        end.synchronize()
        if i >= n_warmup:
            flat_times.append(start.elapsed_time(end))

    # Piecewise (split stage4, no offload engine)
    piecewise_times = []
    for i in range(n_warmup + n_trials):
        engine.reset()
        start.record()
        engine.mixed_step(
            decode_seq_ids=[],
            decode_token_ids=torch.empty(0, dtype=torch.long, device=engine.device),
            prefill_seq_ids=[0],
            prefill_input_ids=[prompt])
        end.record()
        end.synchronize()
        if i >= n_warmup:
            piecewise_times.append(start.elapsed_time(end))

    flat_median = sorted(flat_times)[len(flat_times) // 2]
    piece_median = sorted(piecewise_times)[len(piecewise_times) // 2]
    overhead_ms = piece_median - flat_median
    overhead_pct = (overhead_ms / flat_median) * 100

    print(f"  Flat prefill median:     {flat_median:.3f} ms")
    print(f"  Piecewise median:        {piece_median:.3f} ms")
    print(f"  Overhead:                {overhead_ms:+.3f} ms ({overhead_pct:+.1f}%)")

    # Decode: flat graph vs piecewise
    print(f"\n  --- Decode (B=1) ---")
    engine.reset()
    engine.prefill_to_slot(0, prompt)

    flat_decode_times = []
    for i in range(n_warmup + n_trials):
        positions = engine.seq_lens[:1].clone()
        tok = torch.tensor([1], device=engine.device)
        start.record()
        engine.decode_step(tok, positions)
        end.record()
        end.synchronize()
        if i >= n_warmup:
            flat_decode_times.append(start.elapsed_time(end))

    engine.reset()
    engine.prefill_to_slot(0, prompt)

    piece_decode_times = []
    for i in range(n_warmup + n_trials):
        tok = torch.tensor([1], dtype=torch.long, device=engine.device)
        start.record()
        engine.mixed_step(
            decode_seq_ids=[0],
            decode_token_ids=tok,
            prefill_seq_ids=[],
            prefill_input_ids=[])
        end.record()
        end.synchronize()
        if i >= n_warmup:
            piece_decode_times.append(start.elapsed_time(end))

    flat_d_med = sorted(flat_decode_times)[len(flat_decode_times) // 2]
    piece_d_med = sorted(piece_decode_times)[len(piece_decode_times) // 2]
    overhead_d = piece_d_med - flat_d_med
    overhead_d_pct = (overhead_d / flat_d_med) * 100

    print(f"  Flat decode median:      {flat_d_med:.3f} ms")
    print(f"  Piecewise decode median: {piece_d_med:.3f} ms")
    print(f"  Overhead:                {overhead_d:+.3f} ms ({overhead_d_pct:+.1f}%)")

    return overhead_pct


def test_demand_load_correctness(engine, seq_len=128):
    """Compare output when all experts resident vs demand-loaded.

    Uses configure() to switch between all-resident and budget=2.
    """
    print(f"\n=== Demand loading correctness (seq_len={seq_len}) ===")

    oe = engine.offload_engine
    E = engine.num_experts
    prompt = torch.randint(1, 1000, (seq_len,), device=engine.device)

    # Baseline: all experts resident (configure with full budget)
    oe.configure(gpu_budget_per_layer=E)
    engine.reset()
    oe.reset_trace()
    logits_baseline = engine.mixed_step(
        decode_seq_ids=[],
        decode_token_ids=torch.empty(0, dtype=torch.long, device=engine.device),
        prefill_seq_ids=[0],
        prefill_input_ids=[prompt]).clone()

    # With budget=2 (demand loading active)
    oe.configure(gpu_budget_per_layer=2, initial_experts=[0, 1])
    engine.reset()
    oe.reset_trace()
    logits_offloaded = engine.mixed_step(
        decode_seq_ids=[],
        decode_token_ids=torch.empty(0, dtype=torch.long, device=engine.device),
        prefill_seq_ids=[0],
        prefill_input_ids=[prompt]).clone()

    diff = (logits_baseline - logits_offloaded).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    top1_baseline = logits_baseline.argmax(dim=-1)
    top1_offloaded = logits_offloaded.argmax(dim=-1)
    top1_match = (top1_baseline == top1_offloaded).all().item()
    n_mismatches = (top1_baseline != top1_offloaded).sum().item()

    stats = oe.get_transfer_stats()

    print(f"  Transfers: {stats['total_transfers']} "
          f"({stats['total_bytes'] / 1e6:.0f} MB, {stats['total_time_ms']:.1f} ms)")
    print(f"  Max absolute diff:  {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  Top-1 token match:  {top1_match} ({n_mismatches}/{seq_len} mismatches)")

    if max_diff < 0.001:
        print("  PASS --- exact match")
    elif top1_match:
        print("  PASS --- small numerical diff, top-1 tokens match")
    else:
        print(f"  FAIL --- {n_mismatches} top-1 mismatches")
        # Show first few mismatches
        mask = top1_baseline != top1_offloaded
        idxs = mask.nonzero(as_tuple=True)[0][:5]
        for idx in idxs:
            i = idx.item()
            print(f"    pos {i}: baseline={top1_baseline[i].item()}, "
                  f"offloaded={top1_offloaded[i].item()}, "
                  f"diff={diff[i].max().item():.4f}")

    # Restore full budget for next test
    oe.configure(gpu_budget_per_layer=E)

    return max_diff, top1_match


def test_decode_with_offloading(engine, prompt_len=128, decode_steps=10):
    """Multi-step decode: compare all-resident vs demand-loaded tokens."""
    print(f"\n=== Decode with offloading (prompt={prompt_len}, "
          f"steps={decode_steps}) ===")

    oe = engine.offload_engine
    E = engine.num_experts
    prompt = torch.randint(1, 1000, (prompt_len,), device=engine.device)

    # Baseline: all experts resident
    # Use mixed_step for prefill (piecewise path) in both runs for consistency.
    # prefill_to_slot uses the flat graph which doesn't trigger demand loading.
    oe.configure(gpu_budget_per_layer=E)
    engine.reset()
    oe.reset_trace()
    logits = engine.mixed_step(
        decode_seq_ids=[],
        decode_token_ids=torch.empty(0, dtype=torch.long, device=engine.device),
        prefill_seq_ids=[0],
        prefill_input_ids=[prompt])
    next_token = logits[-1].argmax().unsqueeze(0)

    tokens_baseline = []
    for step in range(decode_steps):
        logits = engine.mixed_step(
            decode_seq_ids=[0],
            decode_token_ids=next_token,
            prefill_seq_ids=[],
            prefill_input_ids=[])
        next_token = logits[0].argmax().unsqueeze(0)
        tokens_baseline.append(next_token.item())

    # With budget=2 (demand loading)
    oe.configure(gpu_budget_per_layer=2, initial_experts=[0, 1])
    engine.reset()
    oe.reset_trace()
    logits = engine.mixed_step(
        decode_seq_ids=[],
        decode_token_ids=torch.empty(0, dtype=torch.long, device=engine.device),
        prefill_seq_ids=[0],
        prefill_input_ids=[prompt])
    next_token = logits[-1].argmax().unsqueeze(0)

    tokens_offloaded = []
    for step in range(decode_steps):
        logits = engine.mixed_step(
            decode_seq_ids=[0],
            decode_token_ids=next_token,
            prefill_seq_ids=[],
            prefill_input_ids=[])
        next_token = logits[0].argmax().unsqueeze(0)
        tokens_offloaded.append(next_token.item())

    match = tokens_baseline == tokens_offloaded
    print(f"  Baseline tokens:   {tokens_baseline}")
    print(f"  Offloaded tokens:  {tokens_offloaded}")
    print(f"  Exact match: {match}")
    if not match:
        for i, (b, o) in enumerate(zip(tokens_baseline, tokens_offloaded)):
            if b != o:
                print(f"    Step {i}: baseline={b}, offloaded={o}")
    stats = oe.get_transfer_stats()
    print(f"  Transfers: {stats['total_transfers']} over {decode_steps} steps")

    print(f"  {'PASS' if match else 'FAIL'}")

    # Restore full budget
    oe.configure(gpu_budget_per_layer=E)

    return match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    use_compile = not args.no_compile
    print(f"Model: {args.model}")
    print(f"torch.compile: {use_compile}")

    # Test 1: split overhead (no offload engine needed — experts_per_layer=None)
    engine = MoEEngine(args.model, max_seqs=8, max_seq_len=2048,
                       use_torch_compile=use_compile)

    with torch.inference_mode():
        engine.capture_prefill_cuda_graph(
            total_token_sizes=[128, 256],
            use_torch_compile=use_compile)
        engine.reset()
        engine.capture_decode_cuda_graph(
            batch_size=1, warmup_seq_len=128,
            max_decode_tokens=256,
            use_torch_compile=use_compile)
        engine.reset()
        engine.capture_mixed_cuda_graphs(
            total_token_sizes=[1, 128, 256],
            use_torch_compile=use_compile)

    with torch.inference_mode():
        benchmark_split_overhead(engine, seq_len=128)

    del engine
    gc.collect()
    torch.cuda.empty_cache()

    # Tests 2-3: offloading correctness (experts_per_layer=E, all resident,
    # then configure() to reduce budget for demand loading tests)
    with open(Path(args.model) / "config.json") as f:
        cfg = json.load(f)
    E = cfg.get("num_local_experts") or cfg.get("num_experts")
    engine = MoEEngine(args.model, max_seqs=8, max_seq_len=2048,
                       experts_per_layer=E,
                       use_torch_compile=use_compile)

    with torch.inference_mode():
        engine.capture_prefill_cuda_graph(
            total_token_sizes=[128, 256],
            use_torch_compile=use_compile)
        engine.reset()
        engine.capture_mixed_cuda_graphs(
            total_token_sizes=[1, 128, 256],
            use_torch_compile=use_compile)

    with torch.inference_mode():
        _, top1_match = test_demand_load_correctness(engine, seq_len=128)
        decode_match = test_decode_with_offloading(engine, prompt_len=128,
                                                    decode_steps=10)

    if not (top1_match and decode_match):
        print("\nFAILED")
        sys.exit(1)
    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
