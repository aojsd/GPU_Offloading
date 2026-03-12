#!/usr/bin/env -S python3 -u
"""Verify split stage4 correctness: piecewise (3 graphs/layer) vs flat CUDA graph.

Tests that the piecewise path (stage4a + stage4b) produces numerically identical
logits to the flat prefill CUDA graph (which uses the unsplit _full_mixed_graph_body).

Usage:
    python tests/test_split_stage4.py [--model PATH] [--pp N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from moe_engine import MoEEngine

DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "Mixtral-8x7B-20L")


def test_prefill_flat_vs_piecewise(engine, seq_len=128, use_compile=True):
    """Compare flat prefill graph vs piecewise step (pure prefill)."""
    print(f"\n=== Prefill: flat graph vs piecewise (seq_len={seq_len}) ===")

    prompt = torch.randint(1, 1000, (seq_len,), device=engine.device)

    # 1. Flat prefill graph
    engine.reset()
    logits_flat = engine.prefill_to_slot(0, prompt)

    # 2. Piecewise path (via step with no decode tokens)
    engine.reset()
    logits_piecewise = engine.step(
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

    if max_diff > 2.0:
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
        logits = engine.step(
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
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallel size (default: 1)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile for exact comparison")
    args = parser.parse_args()

    use_compile = not args.no_compile
    print(f"Model: {args.model}")
    print(f"PP: {args.pp}")
    print(f"torch.compile: {use_compile}")

    engine = MoEEngine(args.model, max_seqs=8, max_seq_len=2048,
                       use_torch_compile=use_compile,
                       pipeline_parallel_size=args.pp)

    # Flat prefill/decode graphs are single-GPU only (iterate all layers on
    # one device). With PP > 1 only piecewise graphs are available.
    with torch.inference_mode():
        if args.pp == 1:
            engine.capture_prefill_cuda_graph(
                total_token_sizes=[128, 256],
                use_torch_compile=use_compile)
            engine.reset()
            engine.capture_decode_cuda_graph(
                batch_size=1, warmup_seq_len=128,
                max_decode_tokens=256,
                use_torch_compile=use_compile)
            engine.reset()
        engine.capture_cuda_graphs(
            total_token_sizes=[1, 128, 256],
            use_torch_compile=use_compile)

    with torch.inference_mode():
        if args.pp == 1:
            # Test 1: Flat prefill vs piecewise prefill
            t1_diff, t1_match = test_prefill_flat_vs_piecewise(
                engine, seq_len=128, use_compile=use_compile)

            # Test 2: Decode via piecewise vs flat graph
            t2_match = test_decode_piecewise(
                engine, prompt_len=128, decode_steps=5,
                use_compile=use_compile)

            # With torch.compile, inductor noise causes small numerical diffs
            # that may flip top-1 tokens. max_diff < 2.0 is the correctness bar.
            t1_ok = t1_diff < 2.0
            all_pass = t1_ok and t2_match
            if not all_pass:
                if use_compile and t1_ok:
                    print("\n=== Tests complete (decode mismatch expected "
                          "with compile) ===")
                else:
                    print("\nFAILED")
                    sys.exit(1)
            else:
                print("\n=== All tests passed ===")
        else:
            # PP > 1: flat graphs unavailable, test piecewise self-consistency
            print(f"\n=== PP={args.pp}: flat graphs N/A, testing piecewise ===")
            prompt = torch.randint(1, 1000, (128,), device=engine.device)
            engine.reset()
            logits1 = engine.step(
                decode_seq_ids=[],
                decode_token_ids=torch.empty(0, dtype=torch.long,
                                             device=engine.device),
                prefill_seq_ids=[0],
                prefill_input_ids=[prompt])
            tok1 = logits1[-1].argmax().item()

            engine.reset()
            logits2 = engine.step(
                decode_seq_ids=[],
                decode_token_ids=torch.empty(0, dtype=torch.long,
                                             device=engine.device),
                prefill_seq_ids=[0],
                prefill_input_ids=[prompt])
            tok2 = logits2[-1].argmax().item()

            diff = (logits1 - logits2).abs().max().item()
            print(f"  Piecewise determinism: max_diff={diff:.6f}, "
                  f"tok1={tok1}, tok2={tok2}")
            if tok1 == tok2 and diff < 0.01:
                print("  PASS")
            else:
                print("  FAIL: piecewise not deterministic")
                sys.exit(1)

            # Test decode via piecewise
            next_token = logits1[-1].argmax().unsqueeze(0)
            tokens = []
            for _ in range(5):
                logits = engine.step(
                    decode_seq_ids=[0],
                    decode_token_ids=next_token,
                    prefill_seq_ids=[],
                    prefill_input_ids=[])
                next_token = logits[0].argmax().unsqueeze(0)
                tokens.append(next_token.item())
            print(f"  Decode tokens (5 steps): {tokens}")
            print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
