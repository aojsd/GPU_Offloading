"""Verification tests for unified expert cache.

1. Smoke test: 32L experts_per_layer=4 — load, capture graphs, prefill + 5 decode steps, no NaN
2. Correctness: 20L experts_per_layer=8 vs 20L no offloading — greedy tokens must match
"""
import ctypes
ctypes.CDLL("/gpfs/radev/apps/avx512/software/GCCcore/13.3.0/lib64/libstdc++.so.6",
            mode=ctypes.RTLD_GLOBAL)

import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import argparse
from moe_engine import MoEEngine


def smoke_test_32L(model_path, use_compile=True):
    """Smoke test: 32L experts_per_layer=4, capture graphs, prefill + 5 decode steps."""
    print("\n" + "=" * 70)
    print("  SMOKE TEST: 32L experts_per_layer=4")
    print("=" * 70)

    engine = MoEEngine(
        model_path, device="cuda:0",
        experts_per_layer=4,
    )

    # Verify unified buffer structure
    experts_per_layer = engine.experts_per_layer
    L = engine.num_layers
    E = engine.num_experts
    total_slots = L * experts_per_layer + engine.scratchpad_slots
    print(f"\n  Buffer: {total_slots} slots "
          f"({L}L x {experts_per_layer} experts_per_layer"
          f" + {engine.scratchpad_slots} scratchpad)")
    print(f"  w1_buf shape: {list(engine.w1_buf.shape)}")
    print(f"  w2_buf shape: {list(engine.w2_buf.shape)}")

    # Verify per-layer views share storage with unified buffer
    for l in range(L):
        base = l * experts_per_layer
        assert engine.w1[l].data_ptr() == engine.w1_buf[base].data_ptr(), \
            f"Layer {l} w1 view not sharing storage with unified buffer"
        assert engine.w2[l].data_ptr() == engine.w2_buf[base].data_ptr(), \
            f"Layer {l} w2 view not sharing storage with unified buffer"
    print("  Views share storage: OK")

    # Verify expert_map values
    for l in range(min(3, L)):
        rel = engine.expert_map[l].cpu().tolist()
        abs_ = engine.expert_map_abs[l].cpu().tolist()
        base = l * experts_per_layer
        for eid in range(E):
            if rel[eid] >= 0:
                assert abs_[eid] == base + rel[eid], \
                    f"Layer {l} expert {eid}: rel={rel[eid]} abs={abs_[eid]} expected={base + rel[eid]}"
    print("  Expert maps consistent: OK")

    with torch.inference_mode():
        engine.capture_prefill_cuda_graph(total_token_sizes=[128],
                                           use_torch_compile=use_compile)
        engine.reset()
        engine.capture_mixed_cuda_graphs(total_token_sizes=[1],
                                          use_torch_compile=use_compile)

    print("\n  Graph capture: OK")

    # Prefill + decode
    prompt = torch.randint(1, 1000, (128,), device="cuda")
    engine.reset()

    with torch.inference_mode():
        logits = engine.prefill_to_slot(0, prompt)
        assert not torch.isnan(logits).any(), "NaN in prefill logits!"
        assert not torch.isinf(logits).any(), "Inf in prefill logits!"
        next_token = logits[-1].argmax().unsqueeze(0)
        print(f"  Prefill: OK (top token: {next_token.item()})")

        tokens = [next_token.item()]
        for step in range(5):
            logits = engine.mixed_step(
                decode_seq_ids=[0], decode_token_ids=next_token,
                prefill_seq_ids=[], prefill_input_ids=[])
            assert not torch.isnan(logits).any(), f"NaN in decode step {step}!"
            assert not torch.isinf(logits).any(), f"Inf in decode step {step}!"
            next_token = logits[0].argmax().unsqueeze(0)
            tokens.append(next_token.item())

        print(f"  5 decode steps: OK (tokens: {tokens})")

    # Memory check
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory: {mem_gb:.1f} GB")

    del engine
    torch.cuda.empty_cache()
    print("\n  SMOKE TEST PASSED")
    return True


def correctness_test_20L(model_path, use_compile=True):
    """Correctness: 20L experts_per_layer=8 vs 20L no offloading, greedy tokens must match."""
    print("\n" + "=" * 70)
    print("  CORRECTNESS TEST: 20L experts_per_layer=8 vs 20L no offloading")
    print("=" * 70)

    prompt = torch.randint(1, 1000, (128,), device="cuda")
    n_decode = 20

    # ── Run WITHOUT experts_per_layer (baseline) ──
    print("\n  Loading baseline (no experts_per_layer)...")
    engine_base = MoEEngine(
        model_path, device="cuda:0",
    )

    with torch.inference_mode():
        engine_base.capture_prefill_cuda_graph(total_token_sizes=[128],
                                                use_torch_compile=use_compile)
        engine_base.reset()
        engine_base.capture_mixed_cuda_graphs(total_token_sizes=[1],
                                               use_torch_compile=use_compile)

    engine_base.reset()
    with torch.inference_mode():
        logits = engine_base.prefill_to_slot(0, prompt)
        next_token = logits[-1].argmax().unsqueeze(0)
        tokens_base = [next_token.item()]

        for _ in range(n_decode - 1):
            logits = engine_base.mixed_step(
                decode_seq_ids=[0], decode_token_ids=next_token,
                prefill_seq_ids=[], prefill_input_ids=[])
            next_token = logits[0].argmax().unsqueeze(0)
            tokens_base.append(next_token.item())

    del engine_base
    torch.cuda.empty_cache()

    # ── Run WITH experts_per_layer=8 (all experts resident) ──
    print("  Loading experts_per_layer=8...")
    engine_unified = MoEEngine(
        model_path, device="cuda:0",
        experts_per_layer=8,
    )

    with torch.inference_mode():
        engine_unified.capture_prefill_cuda_graph(total_token_sizes=[128],
                                                   use_torch_compile=use_compile)
        engine_unified.reset()
        engine_unified.capture_mixed_cuda_graphs(total_token_sizes=[1, 128],
                                                  use_torch_compile=use_compile)

    engine_unified.reset()
    with torch.inference_mode():
        logits = engine_unified.prefill_to_slot(0, prompt)
        next_token = logits[-1].argmax().unsqueeze(0)
        tokens_unified = [next_token.item()]

        for _ in range(n_decode - 1):
            logits = engine_unified.mixed_step(
                decode_seq_ids=[0], decode_token_ids=next_token,
                prefill_seq_ids=[], prefill_input_ids=[])
            next_token = logits[0].argmax().unsqueeze(0)
            tokens_unified.append(next_token.item())

    del engine_unified
    torch.cuda.empty_cache()

    # ── Compare ──
    match = sum(1 for a, b in zip(tokens_base, tokens_unified) if a == b)
    total = len(tokens_base)
    print(f"\n  Baseline tokens:  {tokens_base}")
    print(f"  Unified tokens:   {tokens_unified}")
    print(f"  Match: {match}/{total} ({100 * match / total:.1f}%)")

    if tokens_base == tokens_unified:
        print("\n  CORRECTNESS TEST PASSED (exact match)")
        return True
    else:
        # Find first divergence
        for i, (a, b) in enumerate(zip(tokens_base, tokens_unified)):
            if a != b:
                print(f"\n  First divergence at token {i}: "
                      f"baseline={a}, unified={b}")
                break
        print("\n  CORRECTNESS TEST FAILED (tokens differ)")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Path to a model directory (used as both "
                             "--model-32L and --model-20L)")
    parser.add_argument("--model-32L", default="src/MoE/models/Mixtral-8x7B",
                        help="Path to 32L Mixtral model")
    parser.add_argument("--model-20L", default="src/MoE/models/Mixtral-8x7B-20L",
                        help="Path to 20L Mixtral model")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--skip-32L", action="store_true",
                        help="Skip 32L smoke test")
    parser.add_argument("--skip-20L", action="store_true",
                        help="Skip 20L correctness test")
    args = parser.parse_args()

    # --model overrides both model-32L and model-20L when set
    if args.model is not None:
        args.model_32L = args.model
        args.model_20L = args.model

    use_compile = not args.no_compile
    results = {}

    if not args.skip_32L:
        results['32L_smoke'] = smoke_test_32L(args.model_32L, use_compile)

    if not args.skip_20L:
        results['20L_correctness'] = correctness_test_20L(args.model_20L, use_compile)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
