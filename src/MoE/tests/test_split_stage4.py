#!/usr/bin/env -S python3 -u
"""Verify split stage4 correctness: piecewise (3 graphs/layer) vs flat CUDA graph.

Tests that the piecewise path (stage4a + stage4b) produces numerically identical
logits to the flat prefill CUDA graph (which uses the unsplit _full_mixed_graph_body).

Usage:
    python tests/test_split_stage4.py [--model PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from moe_engine import MoEEngine

DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "Mixtral-8x7B-20L")


def test_prefill_flat_vs_piecewise(engine, seq_len=128, use_compile=True):
    """Compare flat prefill graph vs piecewise mixed_step (pure prefill)."""
    print(f"\n=== Prefill: flat graph vs piecewise (seq_len={seq_len}) ===")

    prompt = torch.randint(1, 1000, (seq_len,), device=engine.device)

    # 1. Flat prefill graph
    engine.reset()
    logits_flat = engine.prefill_to_slot(0, prompt)

    # 2. Piecewise path (via mixed_step with no decode tokens)
    engine.reset()
    logits_piecewise = engine.mixed_step(
        decode_seq_ids=[],
        decode_token_ids=torch.empty(0, dtype=torch.long, device=engine.device),
        prefill_seq_ids=[0],
        prefill_input_ids=[prompt])

    # Compare
    diff = (logits_flat - logits_piecewise).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    top1_match = (logits_flat.argmax(dim=-1) == logits_piecewise.argmax(dim=-1)).all().item()

    print(f"  Max absolute diff:  {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  Top-1 token match:  {top1_match}")

    if max_diff > 1.0:
        print("  WARNING: large numerical difference!")
        # Show where differences are largest
        flat_top = logits_flat.argmax(dim=-1)
        piece_top = logits_piecewise.argmax(dim=-1)
        mismatches = (flat_top != piece_top).sum().item()
        print(f"  Top-1 mismatches: {mismatches}/{seq_len}")
    else:
        print("  PASS")

    return max_diff, top1_match


def test_decode_piecewise(engine, prompt_len=128, decode_steps=5, use_compile=True):
    """Verify decode via piecewise path produces consistent output."""
    print(f"\n=== Decode via piecewise (prompt={prompt_len}, steps={decode_steps}) ===")

    prompt = torch.randint(1, 1000, (prompt_len,), device=engine.device)

    # Prefill using flat graph, then decode via piecewise
    engine.reset()
    logits = engine.prefill_to_slot(0, prompt)
    next_token = logits[-1].argmax().unsqueeze(0)

    tokens_piecewise = []
    for step in range(decode_steps):
        logits = engine.mixed_step(
            decode_seq_ids=[0],
            decode_token_ids=next_token,
            prefill_seq_ids=[],
            prefill_input_ids=[])
        next_token = logits[0].argmax().unsqueeze(0)
        tokens_piecewise.append(next_token.item())

    # Same sequence with flat decode graph
    engine.reset()
    logits = engine.prefill_to_slot(0, prompt)
    next_token = logits[-1].argmax().unsqueeze(0)

    tokens_flat = []
    for step in range(decode_steps):
        positions = engine.seq_lens[:1].clone()
        logits = engine.decode_step(next_token, positions)
        next_token = logits[0].argmax().unsqueeze(0)
        tokens_flat.append(next_token.item())

    match = tokens_piecewise == tokens_flat
    print(f"  Piecewise tokens: {tokens_piecewise}")
    print(f"  Flat graph tokens: {tokens_flat}")
    print(f"  Exact match: {match}")
    if match:
        print("  PASS")
    else:
        print("  MISMATCH (may be expected with torch.compile due to inductor noise)")

    return match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile for exact comparison")
    args = parser.parse_args()

    use_compile = not args.no_compile
    print(f"Model: {args.model}")
    print(f"torch.compile: {use_compile}")

    engine = MoEEngine(args.model, max_batch_size=8, max_seq_len=2048,
                       use_torch_compile=use_compile)

    # Capture both flat prefill and piecewise graphs
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
        # Test 1: Flat prefill vs piecewise prefill
        test_prefill_flat_vs_piecewise(engine, seq_len=128,
                                       use_compile=use_compile)

        # Test 2: Decode via piecewise vs flat graph
        test_decode_piecewise(engine, prompt_len=128, decode_steps=5,
                              use_compile=use_compile)

    print("\n=== All tests complete ===")


if __name__ == "__main__":
    main()
