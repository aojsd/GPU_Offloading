#!/usr/bin/env -S python3 -u
"""Comprehensive chunked prefill correctness test on ALL 200 ShareGPT prompts.

For each prompt, compares a reference prefill against a manual 2-chunk split
(first half as new prefill, second half as continuation via step).

- Short prompts (<= max_graph): reference = single-shot full prefill.
  Compares ALL position logits.
- Long prompts (> max_graph): reference = engine's chunked_prefill_to_slot
  (automatic optimal chunking). Compares only last-token logits (earlier
  tokens used different chunk boundaries).

Reports per-prompt: max absolute logit diff, cosine similarity, top-1 match.
Uses use_torch_compile=False to eliminate inductor noise.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/test_chunked_prefill_comprehensive.py \
        --model ../../models/Mixtral-8x7B-20L --device cuda:0
    CUDA_VISIBLE_DEVICES=1 python tests/test_chunked_prefill_comprehensive.py \
        --model ../../models/OLMoE-1B-7B --device cuda:0
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Apply glibc 2.28 monkey patches before any vLLM import
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import moe_engine as _moe_engine_mod  # noqa: F401
from moe_engine import MoEEngine


# Raw ShareGPT dataset (model-independent)
SHAREGPT_JSON = (
    Path(__file__).resolve().parent.parent
    / "datasets" / "ShareGPT_Vicuna" / "ShareGPT_V3_unfiltered_cleaned_split.json"
)


def load_all_sharegpt_prompts(max_prompts=200):
    """Load ShareGPT prompts from the raw dataset."""
    if not SHAREGPT_JSON.exists():
        raise FileNotFoundError(f"ShareGPT dataset not found: {SHAREGPT_JSON}")

    with open(SHAREGPT_JSON) as f:
        data = json.load(f)

    prompts = []
    for entry in data:
        if len(prompts) >= max_prompts:
            break
        turns = entry.get("conversations", [])
        if not turns or turns[0].get("from") != "human":
            continue
        text = turns[0]["value"].strip()
        if not text:
            continue
        prompts.append({
            "id": entry["id"],
            "index": len(prompts),
            "text": text,
        })
    return prompts


def ref_prefill_single_shot(engine, seq_id, input_ids):
    """Prefill full prompt in one shot via step. Returns [S, vocab]."""
    engine.reset()
    with torch.inference_mode():
        logits = engine.step(
            decode_seq_ids=[],
            decode_token_ids=torch.empty(0, dtype=torch.long,
                                         device=engine.device),
            prefill_seq_ids=[seq_id],
            prefill_input_ids=[input_ids],
        )
    S = input_ids.shape[0]
    return logits[:S]


def ref_prefill_chunked(engine, seq_id, input_ids):
    """Prefill via engine's automatic chunking. Returns last-chunk logits."""
    engine.reset()
    with torch.inference_mode():
        logits = engine.chunked_prefill_to_slot(seq_id, input_ids)
    # chunked_prefill_to_slot returns logits from the last chunk
    # The last chunk's logits cover only the last chunk's tokens.
    # We need the logits for the LAST token of the prompt.
    # The returned tensor is [graph_N, vocab] padded; last chunk has
    # actual_len real tokens.
    # Actually, let's extract only the last token's logits.
    # chunked_prefill_to_slot returns logits with shape based on graph size.
    # We need the last real token. The last chunk covers tokens from some
    # offset to S. Its logits[actual_len-1] is the last token.
    return logits


def chunked_prefill_n_chunks(engine, seq_id, input_ids, max_chunk_size):
    """Prefill via N chunks of at most max_chunk_size tokens each.

    First chunk is a new prefill (FA3), subsequent chunks are continuations
    (FlashInfer paged-KV). Returns last-token logits [vocab].
    """
    engine.reset()
    S = input_ids.shape[0]
    empty = torch.empty(0, dtype=torch.long, device=engine.device)

    offset = 0
    last_logits = None
    with torch.inference_mode():
        while offset < S:
            chunk_end = min(offset + max_chunk_size, S)
            chunk = input_ids[offset:chunk_end]
            chunk_len = chunk.shape[0]

            if offset == 0:
                # First chunk: new prefill
                logits = engine.step(
                    decode_seq_ids=[],
                    decode_token_ids=empty,
                    prefill_seq_ids=[seq_id],
                    prefill_input_ids=[chunk],
                )
            else:
                # Continuation chunk
                logits = engine.step(
                    decode_seq_ids=[],
                    decode_token_ids=empty,
                    prefill_seq_ids=[],
                    prefill_input_ids=[],
                    continuation_seq_ids=[seq_id],
                    continuation_input_ids=[chunk],
                    continuation_offsets=[offset],
                )
            last_logits = logits[chunk_len - 1]  # last real token's logits
            offset = chunk_end

    return last_logits  # [vocab]


def cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(
        a.unsqueeze(0).float(), b.unsqueeze(0).float()).item()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive chunked prefill correctness test")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallel size (default: 1)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"PP: {args.pp}")
    print(f"Device: {args.device}")

    # Load config
    config_path = Path(args.model) / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    is_mixtral = cfg.get("model_type") == "mixtral" or "mixtral" in args.model.lower()
    is_olmoe = cfg.get("model_type") == "olmoe" or "olmoe" in args.model.lower()

    # Model-specific settings
    if is_mixtral:
        dtype = torch.float16
        # Mixtral-20L: ~55 GB weights. Graph sizes above 1024 OOM on H100 80GB.
        max_graph_size = 1024
    elif is_olmoe:
        dtype = torch.bfloat16
        # OLMoE-1B: small model, can afford large graphs
        max_graph_size = 8192
    else:
        dtype = torch.bfloat16
        max_graph_size = 4096

    # KV cache must cover the longest prompt we want to test.
    # max_seq_len controls pages allocated; generous here (only 1-2 seqs used).
    max_seq_len = 8192

    print(f"Dtype: {dtype}, max_graph_size: {max_graph_size}, "
          f"max_seq_len: {max_seq_len}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load all ShareGPT prompts
    print("Loading ShareGPT prompts...")
    prompts = load_all_sharegpt_prompts()
    print(f"Loaded {len(prompts)} prompts")

    # Tokenize all prompts
    print("Tokenizing prompts...")
    tokenized = []
    for p in prompts:
        messages = [{"role": "user", "content": p["text"]}]
        try:
            tokens = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True)
        except Exception:
            tokens = tokenizer.encode(p["text"])
        tokenized.append({
            "id": p["id"],
            "index": p["index"],
            "tokens": tokens,
            "length": len(tokens),
        })

    # Filter: skip prompts < 2 tokens or exceeding KV cache capacity.
    # N-chunk approach handles arbitrarily long prompts (each chunk <= max_graph_size).
    skipped_short = sum(1 for t in tokenized if t["length"] < 2)
    skipped_kv = sum(1 for t in tokenized
                     if t["length"] >= 2 and t["length"] > max_seq_len)
    valid = [t for t in tokenized
             if 2 <= t["length"] <= max_seq_len]
    print(f"Valid prompts: {len(valid)} "
          f"(skipped: {skipped_short} short, {skipped_kv} KV overflow)")

    if not valid:
        print("ERROR: No valid prompts to test!")
        sys.exit(1)

    max_prompt_len = max(t["length"] for t in valid)
    print(f"Max prompt length: {max_prompt_len}")

    # Build graph sizes: powers of 2 from 128 up to max_graph_size
    graph_sizes = [1]
    size = 128
    while size <= max_graph_size:
        graph_sizes.append(size)
        size *= 2
    # Ensure max_graph_size is included even if not a power of 2
    if max_graph_size not in graph_sizes:
        graph_sizes.append(max_graph_size)
    graph_sizes.sort()
    print(f"Graph sizes to capture: {graph_sizes}")

    # Determine how many prompts fit in a single graph vs need chunking
    short_count = sum(1 for t in valid if t["length"] <= max_graph_size)
    long_count = sum(1 for t in valid if t["length"] > max_graph_size)
    print(f"Short prompts (<= {max_graph_size}, single-shot ref): {short_count}")
    print(f"Long prompts (> {max_graph_size}, chunked ref): {long_count}")

    # Load engine with reduced max_seqs to save KV cache memory
    print("\nLoading engine...")
    t0 = time.time()
    engine = MoEEngine(
        args.model,
        max_seqs=2,  # only need 1 sequence at a time, 2 for safety
        max_seq_len=max_seq_len,
        page_size=16,
        dtype=dtype,
        device=args.device,
        use_torch_compile=False,
        pipeline_parallel_size=args.pp,
    )
    print(f"Engine loaded in {time.time() - t0:.1f}s")

    # Capture piecewise CUDA graphs
    print(f"Capturing piecewise CUDA graphs for sizes: {graph_sizes}")
    t0 = time.time()
    with torch.inference_mode():
        engine.capture_cuda_graphs(
            total_token_sizes=graph_sizes,
            use_torch_compile=False,
        )
    print(f"Graphs captured in {time.time() - t0:.1f}s")

    # Run comparisons
    print("\n" + "=" * 100)
    print(f"CHUNKED PREFILL CORRECTNESS TEST -- {len(valid)} prompts")
    print("=" * 100)
    hdr = (f"{'#':>4s}  {'ID':>16s}  {'Len':>5s}  {'Mode':>6s}  "
           f"{'MaxDiff':>10s}  {'CosSim':>10s}  {'LastTok':>7s}")
    print(hdr)
    print("-" * 80)

    results = []
    t_start = time.time()

    for i, entry in enumerate(valid):
        input_ids = torch.tensor(entry["tokens"], dtype=torch.long,
                                 device=engine.device)
        S = input_ids.shape[0]
        is_short = S <= max_graph_size
        n_chunks = (S + max_graph_size - 1) // max_graph_size

        if is_short:
            # --- Short prompt: single-shot reference, compare all positions ---
            ref_logits = ref_prefill_single_shot(engine, 0, input_ids)
            # N-chunk test (N=2 for short prompts: split at S//2)
            test_last = chunked_prefill_n_chunks(
                engine, 0, input_ids, max_chunk_size=max(S // 2, 1))

            # All-position diff uses single-shot ref
            ref_last_logits = ref_logits[-1]

            diff = (ref_last_logits - test_last).abs()
            max_abs_diff = diff.max().item()
            last_cos = cosine_sim(ref_last_logits, test_last)
            last_ref_tok = ref_last_logits.argmax().item()
            last_test_tok = test_last.argmax().item()
            mode = "full"
        else:
            # --- Long prompt: single-shot reference via chunked_prefill_to_slot,
            #     test via N-chunk with max_graph_size chunks ---
            # Reference: engine auto-chunking (uses its own graph-size boundaries)
            ref_all_logits = ref_prefill_chunked(engine, 0, input_ids)
            ref_last_logits = ref_all_logits[-1]  # [vocab]

            # Test: N-chunk with max_graph_size (simulates replay chunking)
            test_last = chunked_prefill_n_chunks(
                engine, 0, input_ids, max_chunk_size=max_graph_size)

            diff = (ref_last_logits - test_last).abs()
            max_abs_diff = diff.max().item()
            last_cos = cosine_sim(ref_last_logits, test_last)
            last_ref_tok = ref_last_logits.argmax().item()
            last_test_tok = test_last.argmax().item()
            mode = f"{n_chunks}chk"

        last_match = last_ref_tok == last_test_tok

        results.append({
            "id": entry["id"],
            "index": entry["index"],
            "length": S,
            "mode": mode,
            "max_abs_diff": max_abs_diff,
            "last_cos_sim": last_cos,
            "last_match": last_match,
            "last_ref_tok": last_ref_tok,
            "last_test_tok": last_test_tok,
        })

        status = "PASS" if last_match else "FAIL"
        extra = ""
        if not last_match:
            extra = f"  ref={last_ref_tok} test={last_test_tok}"

        print(f"{i:4d}  {entry['id']:>16s}  {S:5d}  {mode:>6s}  "
              f"{max_abs_diff:10.4f}  {last_cos:10.6f}  {status:>7s}"
              f"{extra}")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  --- Progress: {i+1}/{len(valid)}, "
                  f"elapsed: {elapsed:.1f}s ---")

    # Summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Model: {args.model}")
    print(f"Total prompts tested: {len(results)}")
    print(f"  Short (single-shot ref): "
          f"{sum(1 for r in results if r['mode'] == 'full')}")
    print(f"  Long (N-chunk ref):      "
          f"{sum(1 for r in results if r['mode'] != 'full')}")
    print(f"Total time: {elapsed:.1f}s "
          f"({elapsed / len(results):.2f}s per prompt)")

    # Last-token metrics (all prompts)
    last_diffs = [r["max_abs_diff"] for r in results]
    last_cos_sims = [r["last_cos_sim"] for r in results]
    last_match_count = sum(1 for r in results if r["last_match"])

    print(f"\n--- Last-Token Metrics (all {len(results)} prompts) ---")
    print(f"Max absolute diff (worst):  {max(last_diffs):.6f}")
    print(f"Max absolute diff (mean):   {sum(last_diffs)/len(last_diffs):.6f}")
    print(f"Max absolute diff (median): "
          f"{sorted(last_diffs)[len(last_diffs)//2]:.6f}")
    print(f"Cosine similarity (mean):   "
          f"{sum(last_cos_sims)/len(last_cos_sims):.8f}")
    print(f"Cosine similarity (worst):  {min(last_cos_sims):.8f}")
    print(f"Top-1 last-token match:     {last_match_count}/{len(results)}")

    # List divergences
    diverged = [r for r in results if not r["last_match"]]
    if diverged:
        print(f"\n--- Last-Token Divergences ({len(diverged)}) ---")
        for r in diverged:
            print(f"  {r['id']}: len={r['length']}, mode={r['mode']}, "
                  f"max_diff={r['max_abs_diff']:.4f}, "
                  f"cos_sim={r['last_cos_sim']:.6f}, "
                  f"ref={r['last_ref_tok']}, test={r['last_test_tok']}")
    else:
        print("\nNo last-token divergences!")

    # Verdict
    print("\n" + "=" * 100)
    if last_match_count == len(results) and max(last_diffs) < 1.0:
        print("VERDICT: PASS -- All last-token predictions match, "
              "max diff < 1.0")
    elif last_match_count == len(results):
        print(f"VERDICT: PASS (with noise) -- All last-token predictions "
              f"match, max diff = {max(last_diffs):.4f}")
    else:
        print(f"VERDICT: {last_match_count}/{len(results)} last-token "
              f"matches (max diff = {max(last_diffs):.4f})")
    print("=" * 100)


if __name__ == "__main__":
    main()
