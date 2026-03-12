#!/usr/bin/env -S python3 -u
"""Test scheduled chunked prefill correctness.

Verifies that prefilling a prompt via continuation chunks in mixed_step()
produces the same output as a single full prefill. This is the foundation
for vLLM-style scheduled chunked prefill where a prefill is interleaved
with decode tokens across scheduler steps.

Test 1: Single sequence — full prefill vs 2-chunk continuation
Test 2: Single sequence — full prefill vs 3-chunk continuation
Test 3: Mixed batch — decode tokens + continuation chunk in same step
Test 4: Multiple ShareGPT prompts — systematic comparison

Usage:
    python tests/test_scheduled_chunked_prefill.py --model ../../models/Mixtral-8x7B-20L
    python tests/test_scheduled_chunked_prefill.py --model ../../models/OLMoE-1B-7B-0924
    python tests/test_scheduled_chunked_prefill.py --model ../../models/Mixtral-8x7B --pp 2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from moe_engine import MoEEngine


DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent.parent.parent / "models" / "Mixtral-8x7B-20L")

SHAREGPT_JSON = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "datasets" / "ShareGPT_Vicuna" / "ShareGPT_V3_unfiltered_cleaned_split.json")


def load_sharegpt_prompts(n=10):
    """Load first n ShareGPT prompts from the raw dataset."""
    if not SHAREGPT_JSON.exists():
        return None
    with open(SHAREGPT_JSON) as f:
        data = json.load(f)

    convos = []
    for entry in data:
        if len(convos) >= n:
            break
        turns = entry.get("conversations", [])
        if not turns or turns[0].get("from") != "human":
            continue
        text = turns[0]["value"].strip()
        if not text:
            continue
        convos.append({
            'id': entry['id'],
            'prompt_text': text,
        })
    return convos


def full_prefill(engine, seq_id, input_ids):
    """Prefill a full prompt in one shot via mixed_step. Returns last-token logits."""
    engine.reset()
    empty_dev = engine.device
    with torch.inference_mode():
        logits = engine.mixed_step(
            decode_seq_ids=[],
            decode_token_ids=torch.empty(0, dtype=torch.long, device=empty_dev),
            prefill_seq_ids=[seq_id],
            prefill_input_ids=[input_ids])
    S = input_ids.shape[0]
    return logits[S - 1]  # [vocab_size]


def chunked_prefill_via_continuation(engine, seq_id, input_ids, chunk_size):
    """Prefill a prompt via multiple continuation chunks using mixed_step().

    First chunk uses new-prefill (FA3 self-attention).
    Subsequent chunks use continuation (FlashInfer paged-KV attention).
    Returns last-token logits.
    """
    engine.reset()
    S = input_ids.shape[0]
    empty_dev = engine.device

    chunks = []
    offset = 0
    while offset < S:
        end = min(offset + chunk_size, S)
        chunks.append((offset, input_ids[offset:end]))
        offset = end

    with torch.inference_mode():
        logits = None
        for i, (off, chunk_ids) in enumerate(chunks):
            if i == 0:
                # First chunk: new prefill
                logits = engine.mixed_step(
                    decode_seq_ids=[],
                    decode_token_ids=torch.empty(
                        0, dtype=torch.long, device=empty_dev),
                    prefill_seq_ids=[seq_id],
                    prefill_input_ids=[chunk_ids])
            else:
                # Continuation chunk
                logits = engine.mixed_step(
                    decode_seq_ids=[],
                    decode_token_ids=torch.empty(
                        0, dtype=torch.long, device=empty_dev),
                    prefill_seq_ids=[],
                    prefill_input_ids=[],
                    continuation_seq_ids=[seq_id],
                    continuation_input_ids=[chunk_ids],
                    continuation_offsets=[off])

    # Return logits of the last real token
    last_chunk_len = chunks[-1][1].shape[0]
    return logits[last_chunk_len - 1]  # [vocab_size]


def mixed_decode_plus_continuation(engine, decode_seq_id, decode_token,
                                   cont_seq_id, cont_ids, cont_offset):
    """Run a mixed step with one decode token + one continuation chunk."""
    with torch.inference_mode():
        logits = engine.mixed_step(
            decode_seq_ids=[decode_seq_id],
            decode_token_ids=decode_token,
            prefill_seq_ids=[],
            prefill_input_ids=[],
            continuation_seq_ids=[cont_seq_id],
            continuation_input_ids=[cont_ids],
            continuation_offsets=[cont_offset])
    return logits


def test_two_chunks(engine, prompt):
    """Test: full prefill vs 2-chunk continuation."""
    S = prompt.shape[0]
    split = S // 2

    ref_logits = full_prefill(engine, 0, prompt)
    test_logits = chunked_prefill_via_continuation(engine, 0, prompt, split)

    ref_token = ref_logits.argmax().item()
    test_token = test_logits.argmax().item()

    max_diff = (ref_logits - test_logits).abs().max().item()
    match = ref_token == test_token

    return match, max_diff, ref_token, test_token


def test_three_chunks(engine, prompt):
    """Test: full prefill vs 3-chunk continuation."""
    S = prompt.shape[0]
    chunk_size = max(1, S // 3)

    ref_logits = full_prefill(engine, 0, prompt)
    test_logits = chunked_prefill_via_continuation(
        engine, 0, prompt, chunk_size)

    ref_token = ref_logits.argmax().item()
    test_token = test_logits.argmax().item()

    max_diff = (ref_logits - test_logits).abs().max().item()
    match = ref_token == test_token

    return match, max_diff, ref_token, test_token


def test_mixed_decode_continuation(engine, prompt):
    """Test: decode + continuation chunk in the same mixed_step.

    1. Prefill seq 0 fully (reference decode sequence).
    2. Prefill first chunk of seq 1 as new prefill.
    3. Run mixed step: decode seq 0 + continuation of seq 1.
    4. Verify decode output matches standalone decode.
    """
    S = prompt.shape[0]
    split = S // 2
    chunk1 = prompt[:split]
    chunk2 = prompt[split:]

    empty_dev = engine.device

    # Reference: full prefill seq 0 + one decode step
    engine.reset()
    with torch.inference_mode():
        logits_pf0 = engine.mixed_step(
            [], torch.empty(0, dtype=torch.long, device=empty_dev),
            [0], [prompt])
        next_token = logits_pf0[S - 1].argmax().unsqueeze(0)
        ref_decode_logits = engine.mixed_step(
            [0], next_token, [], [])
    ref_decode_token = ref_decode_logits[0].argmax().item()

    # Test: prefill seq 0 fully, prefill chunk1 of seq 1, then mixed step
    engine.reset()
    with torch.inference_mode():
        logits_pf0 = engine.mixed_step(
            [], torch.empty(0, dtype=torch.long, device=empty_dev),
            [0], [prompt])
        next_token = logits_pf0[S - 1].argmax().unsqueeze(0)

        # Prefill first chunk of seq 1
        engine.mixed_step(
            decode_seq_ids=[],
            decode_token_ids=torch.empty(0, dtype=torch.long,
                                         device=engine.device),
            prefill_seq_ids=[1],
            prefill_input_ids=[chunk1])

        # Mixed step: decode seq 0 + continuation of seq 1
        test_logits = mixed_decode_plus_continuation(
            engine, 0, next_token, 1, chunk2, split)

    test_decode_token = test_logits[0].argmax().item()

    # Decode output should match — seq 0's decode is independent of seq 1
    decode_match = ref_decode_token == test_decode_token
    decode_diff = (ref_decode_logits[0] - test_logits[0]).abs().max().item()

    return decode_match, decode_diff


def main():
    parser = argparse.ArgumentParser(
        description="Test scheduled chunked prefill correctness")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallel size (default: 1)")
    parser.add_argument("--n-prompts", type=int, default=10,
                        help="Number of ShareGPT prompts to test")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"PP: {args.pp}")

    # Load engine with piecewise graphs
    print("\nLoading engine...")
    engine = MoEEngine(args.model, max_seqs=8, max_seq_len=8192,
                       use_torch_compile=False,
                       pipeline_parallel_size=args.pp)

    # Capture graphs — keep small to avoid OOM on truncated models
    graph_sizes = [1, 128, 256, 512, 1024]
    print(f"Capturing piecewise CUDA graphs for sizes: {graph_sizes}")
    with torch.inference_mode():
        engine.capture_mixed_cuda_graphs(
            total_token_sizes=graph_sizes,
            use_torch_compile=False)

    # Test 1: Fixed prompt, 2 chunks
    print("\n" + "=" * 60)
    print("TEST 1: Full prefill vs 2-chunk continuation (fixed prompt)")
    print("=" * 60)
    prompt = torch.randint(1, 1000, (256,), device=engine.device)
    match, diff, ref_tok, test_tok = test_two_chunks(engine, prompt)
    status = "PASS" if match else "FAIL"
    print(f"  Token match: {status} (ref={ref_tok}, test={test_tok})")
    print(f"  Max logit diff: {diff:.4f}")

    # Test 2: Fixed prompt, 3 chunks
    print("\n" + "=" * 60)
    print("TEST 2: Full prefill vs 3-chunk continuation (fixed prompt)")
    print("=" * 60)
    prompt = torch.randint(1, 1000, (384,), device=engine.device)
    match, diff, ref_tok, test_tok = test_three_chunks(engine, prompt)
    status = "PASS" if match else "FAIL"
    print(f"  Token match: {status} (ref={ref_tok}, test={test_tok})")
    print(f"  Max logit diff: {diff:.4f}")

    # Test 3: Mixed decode + continuation
    print("\n" + "=" * 60)
    print("TEST 3: Decode + continuation chunk in same mixed_step")
    print("=" * 60)
    prompt = torch.randint(1, 1000, (256,), device=engine.device)
    decode_match, decode_diff = test_mixed_decode_continuation(engine, prompt)
    status = "PASS" if decode_match else "FAIL"
    print(f"  Decode token match: {status}")
    print(f"  Max decode logit diff: {decode_diff:.4f}")

    # Test 4: ShareGPT prompts
    print("\n" + "=" * 60)
    print(f"TEST 4: ShareGPT prompts ({args.n_prompts} prompts)")
    print("=" * 60)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    convos = load_sharegpt_prompts(args.n_prompts)
    if convos is None:
        print("  SKIP: no ShareGPT traces found")
    else:
        passes = 0
        fails = 0
        max_graph = max(graph_sizes)

        for conv in convos:
            if not conv['prompt_text']:
                continue
            messages = [{"role": "user", "content": conv['prompt_text']}]
            try:
                tokens = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True)
            except Exception:
                tokens = tokenizer.encode(conv['prompt_text'])
            if len(tokens) > max_graph or len(tokens) < 4:
                continue

            input_ids = torch.tensor(tokens, dtype=torch.long,
                                     device=engine.device)
            S = input_ids.shape[0]

            # Test with chunk_size = S // 2
            chunk_size = max(1, S // 2)
            ref_logits = full_prefill(engine, 0, input_ids)
            test_logits = chunked_prefill_via_continuation(
                engine, 0, input_ids, chunk_size)

            ref_token = ref_logits.argmax().item()
            test_token = test_logits.argmax().item()
            max_diff = (ref_logits - test_logits).abs().max().item()

            status = "PASS" if ref_token == test_token else "FAIL"
            if ref_token == test_token:
                passes += 1
            else:
                fails += 1
            print(f"  {conv['id']}: S={S:4d}, chunk={chunk_size:4d}, "
                  f"{status}, max_diff={max_diff:.4f}, "
                  f"ref={ref_token}, test={test_token}")

        total = passes + fails
        print(f"\n  Results: {passes}/{total} PASS, {fails}/{total} FAIL")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("All tests complete. Check PASS/FAIL above.")


if __name__ == "__main__":
    main()
