#!/usr/bin/env -S python3 -u
"""Test piecewise prefill correctness and performance.

Verifies that prefill through piecewise CUDA graphs (stage4a → CPU break → stage4b)
produces the same output as flat prefill graphs, and measures the latency overhead.

Test 1: Greedy token match — flat prefill vs piecewise prefill (all experts resident)
Test 2: Partial offloading — no CUDA errors, no NaN (experts_per_layer < E)
Test 3: Latency comparison — flat vs piecewise prefill

Usage:
    python tests/test_piecewise_prefill.py --model models/Mixtral-8x7B-20L
    python tests/test_piecewise_prefill.py --model models/OLMoE-1B-7B-0924
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from moe_engine import MoEEngine

DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "Mixtral-8x7B-20L")


def get_flat_prefill_tokens(model_path, prompt, seq_len, graph_sizes):
    """Load engine without offloading, prefill with flat graph, return greedy tokens."""
    print("\n--- Loading engine (flat prefill, no offloading) ---")
    engine = MoEEngine(model_path, max_seqs=8, max_seq_len=2048,
                       use_torch_compile=False)

    with torch.inference_mode():
        engine.capture_prefill_cuda_graph(
            total_token_sizes=graph_sizes, use_torch_compile=False)
        engine.reset()

        logits = engine.prefill_to_slot(0, prompt)
        tokens = logits.argmax(dim=-1).cpu().clone()

        # Timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for i in range(25):
            engine.reset()
            start.record()
            engine.prefill_to_slot(0, prompt)
            end.record()
            end.synchronize()
            if i >= 5:
                times.append(start.elapsed_time(end))
        flat_median = sorted(times)[len(times) // 2]

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return tokens, flat_median


def get_piecewise_prefill_tokens(model_path, prompt, seq_len, graph_sizes,
                                 experts_per_layer):
    """Load engine with offloading (all resident), prefill via piecewise, return tokens."""
    print(f"\n--- Loading engine (piecewise prefill, experts_per_layer={experts_per_layer}) ---")
    engine = MoEEngine(model_path, max_seqs=8, max_seq_len=2048,
                       experts_per_layer=experts_per_layer,
                       use_torch_compile=False)

    with torch.inference_mode():
        engine.capture_mixed_cuda_graphs(
            total_token_sizes=graph_sizes, use_torch_compile=False)
        engine.reset()

        # prefill_to_slot now routes through mixed_step → piecewise
        logits = engine.prefill_to_slot(0, prompt)
        tokens = logits.argmax(dim=-1).cpu().clone()

        # Timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for i in range(25):
            engine.reset()
            start.record()
            engine.prefill_to_slot(0, prompt)
            end.record()
            end.synchronize()
            if i >= 5:
                times.append(start.elapsed_time(end))
        piecewise_median = sorted(times)[len(times) // 2]

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return tokens, piecewise_median


def test_partial_offloading(model_path, prompt, seq_len, graph_sizes,
                            experts_per_layer):
    """Test prefill with partial offloading — no crash, no NaN."""
    print(f"\n--- Loading engine (partial offloading, experts_per_layer={experts_per_layer}) ---")
    engine = MoEEngine(model_path, max_seqs=8, max_seq_len=2048,
                       experts_per_layer=experts_per_layer,
                       use_torch_compile=False)

    with torch.inference_mode():
        engine.capture_mixed_cuda_graphs(
            total_token_sizes=graph_sizes, use_torch_compile=False)
        engine.reset()

        logits = engine.prefill_to_slot(0, prompt)

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    top1 = logits.argmax(dim=-1)

    stats = engine.offload_engine.get_transfer_stats()
    print(f"  Transfers: {stats['total_transfers']} "
          f"({stats['total_bytes'] / 1e6:.0f} MB, {stats['total_time_ms']:.1f} ms)")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    print(f"  Top-1 tokens (first 10): {top1[:10].tolist()}")

    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return not has_nan and not has_inf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    model_path = args.model
    seq_len = args.seq_len
    print(f"Model: {model_path}")
    print(f"Sequence length: {seq_len}")

    # Detect model type for experts_per_layer values
    import json
    cfg_path = Path(model_path) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    E = cfg.get("num_local_experts", cfg.get("num_experts", 8))
    print(f"Total experts: {E}")

    # Choose partial offloading budget
    if E >= 32:
        partial_epl = 16  # OLMoE: 64 experts, test with 16
    else:
        partial_epl = max(2, E // 2)  # Mixtral: 8 experts, test with 4

    prompt = torch.randint(1, 1000, (seq_len,), device="cuda")
    graph_sizes = [seq_len] if seq_len <= 128 else [128, seq_len]

    results = {}

    # Test 1: Flat prefill (baseline)
    print("\n" + "=" * 60)
    print("TEST 1: Flat prefill (no offloading) — baseline")
    print("=" * 60)
    flat_tokens, flat_time = get_flat_prefill_tokens(
        model_path, prompt, seq_len, graph_sizes)
    print(f"  Flat prefill median: {flat_time:.3f} ms")
    print(f"  Top-1 tokens (first 10): {flat_tokens[:10].tolist()}")

    # Test 2: Piecewise prefill (all experts resident)
    print("\n" + "=" * 60)
    print(f"TEST 2: Piecewise prefill (experts_per_layer={E}, all resident)")
    print("=" * 60)
    piecewise_tokens, piecewise_time = get_piecewise_prefill_tokens(
        model_path, prompt, seq_len, graph_sizes, experts_per_layer=E)
    print(f"  Piecewise prefill median: {piecewise_time:.3f} ms")
    print(f"  Top-1 tokens (first 10): {piecewise_tokens[:10].tolist()}")

    # Compare tokens
    match = (flat_tokens == piecewise_tokens).all().item()
    n_mismatch = (flat_tokens != piecewise_tokens).sum().item()
    overhead_ms = piecewise_time - flat_time
    overhead_pct = (overhead_ms / flat_time) * 100

    print(f"\n  --- Comparison ---")
    print(f"  Token match: {match} ({n_mismatch}/{seq_len} mismatches)")
    print(f"  Latency: flat={flat_time:.3f} ms, piecewise={piecewise_time:.3f} ms, "
          f"overhead={overhead_ms:+.3f} ms ({overhead_pct:+.1f}%)")

    if match:
        print("  PASS — greedy tokens match exactly")
        results['token_match'] = True
    else:
        print(f"  FAIL — {n_mismatch} greedy token mismatches")
        mask = flat_tokens != piecewise_tokens
        idxs = mask.nonzero(as_tuple=True)[0][:5]
        for idx in idxs:
            i = idx.item()
            print(f"    pos {i}: flat={flat_tokens[i].item()}, "
                  f"piecewise={piecewise_tokens[i].item()}")
        results['token_match'] = False

    # Test 3: Partial offloading (no crash, no NaN)
    print("\n" + "=" * 60)
    print(f"TEST 3: Partial offloading (experts_per_layer={partial_epl})")
    print("=" * 60)
    no_nan = test_partial_offloading(
        model_path, prompt, seq_len, graph_sizes, experts_per_layer=partial_epl)
    if no_nan:
        print("  PASS — no NaN or Inf in output")
        results['partial_offload'] = True
    else:
        print("  FAIL — NaN or Inf detected")
        results['partial_offload'] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model:              {model_path}")
    print(f"  Seq len:            {seq_len}")
    print(f"  Flat prefill:       {flat_time:.3f} ms")
    print(f"  Piecewise prefill:  {piecewise_time:.3f} ms")
    print(f"  Overhead:           {overhead_ms:+.3f} ms ({overhead_pct:+.1f}%)")
    print(f"  Token match (E={E}):  {'PASS' if results.get('token_match') else 'FAIL'}")
    print(f"  Partial (epl={partial_epl}):  {'PASS' if results.get('partial_offload') else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
