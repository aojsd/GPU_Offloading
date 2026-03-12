#!/usr/bin/env -S python3 -u
"""PP chunked prefill correctness test on ShareGPT prompts.

Splits the model across GPUs via pipeline parallelism. For each prompt,
compares a reference prefill against N-chunk manual split (first chunk as
new prefill, rest as continuations).

- Short prompts (<= max_graph): reference = single-shot full prefill.
- Long prompts (> max_graph): reference = engine's chunked_prefill_to_slot.
- All comparisons: last-token logits only (cosine sim + top-1 match).

Uses use_torch_compile=False to eliminate inductor noise.

Usage:
    python tests/test_chunked_prefill_pp.py --model ../../models/Mixtral-8x7B --pp 2
    python tests/test_chunked_prefill_pp.py --model ../../models/OLMoE-1B-7B-0924 --pp 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Apply glibc 2.28 monkey patches before any vLLM import
import moe_engine as _moe_engine_mod  # noqa: F401
from moe_engine import MoEEngine

DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "Mixtral-8x7B")

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
    """Prefill via engine's automatic chunking. Returns logits from last chunk."""
    engine.reset()
    with torch.inference_mode():
        logits = engine.chunked_prefill_to_slot(seq_id, input_ids)
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
                logits = engine.step(
                    decode_seq_ids=[],
                    decode_token_ids=empty,
                    prefill_seq_ids=[seq_id],
                    prefill_input_ids=[chunk],
                )
            else:
                logits = engine.step(
                    decode_seq_ids=[],
                    decode_token_ids=empty,
                    prefill_seq_ids=[],
                    prefill_input_ids=[],
                    continuation_seq_ids=[seq_id],
                    continuation_input_ids=[chunk],
                    continuation_offsets=[offset],
                )
            last_logits = logits[chunk_len - 1]
            offset = chunk_end

    return last_logits  # [vocab]


def cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(
        a.unsqueeze(0).float(), b.unsqueeze(0).float()).item()


def main():
    parser = argparse.ArgumentParser(
        description="PP chunked prefill correctness test")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to model directory")
    parser.add_argument("--pp", type=int, default=2,
                        help="Pipeline parallel size (default: 2)")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Max number of ShareGPT prompts to test")
    args = parser.parse_args()

    model_path = args.model
    pp_size = args.pp

    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    if n_gpus < pp_size:
        print(f"ERROR: PP={pp_size} requires at least {pp_size} GPUs, "
              f"found {n_gpus}")
        sys.exit(1)

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    print(f"\nModel: {model_path}")

    # Load config
    with open(Path(model_path) / "config.json") as f:
        cfg = json.load(f)
    num_layers = cfg["num_hidden_layers"]
    is_mixtral = cfg.get("model_type") == "mixtral" or "mixtral" in model_path.lower()
    print(f"Layers: {num_layers}, PP: {pp_size}")

    # Model-specific settings
    if is_mixtral:
        dtype = torch.float16
        max_graph_size = 512
    else:
        dtype = torch.bfloat16
        max_graph_size = 512

    max_seq_len = 8192
    page_size = 16

    print(f"Dtype: {dtype}, max_graph_size: {max_graph_size}, "
          f"max_seq_len: {max_seq_len}, PP={pp_size}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load all ShareGPT prompts
    print("Loading ShareGPT prompts...")
    prompts = load_all_sharegpt_prompts(max_prompts=args.n_prompts)
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
    print(f"Max prompt length among valid: {max_prompt_len}")

    # Build graph sizes: powers of 2 from 64 up to max_graph_size
    graph_sizes = [1]
    size = 64
    while size <= max_graph_size:
        graph_sizes.append(size)
        size *= 2
    if max_graph_size not in graph_sizes:
        graph_sizes.append(max_graph_size)
    graph_sizes.sort()
    print(f"Graph sizes to capture: {graph_sizes}")

    short_count = sum(1 for t in valid if t["length"] <= max_graph_size)
    long_count = sum(1 for t in valid if t["length"] > max_graph_size)
    print(f"Short prompts (<= {max_graph_size}, single-shot ref): {short_count}")
    print(f"Long prompts (> {max_graph_size}, N-chunk ref): {long_count}")

    # Load engine
    print(f"\nLoading PP={pp_size} engine ({num_layers} layers)...")
    t0 = time.time()
    engine = MoEEngine(
        model_path,
        max_seqs=2,
        max_seq_len=max_seq_len,
        page_size=page_size,
        dtype=dtype,
        device="cuda:0",
        pipeline_parallel_size=pp_size,
        use_torch_compile=False,
    )
    load_time = time.time() - t0
    print(f"Engine loaded in {load_time:.1f}s")

    for i in range(pp_size):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")

    # Capture piecewise CUDA graphs
    print(f"\nCapturing piecewise CUDA graphs for sizes: {graph_sizes}")
    t0 = time.time()
    with torch.inference_mode():
        engine.capture_cuda_graphs(
            total_token_sizes=graph_sizes,
            use_torch_compile=False,
        )
    capture_time = time.time() - t0
    print(f"Graphs captured in {capture_time:.1f}s")

    for i in range(pp_size):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i} after capture: allocated={alloc:.2f} GB, "
              f"reserved={reserved:.2f} GB")

    # Run comparisons
    print("\n" + "=" * 80)
    print(f"PP=2 CHUNKED PREFILL CORRECTNESS TEST -- {len(valid)} prompts")
    print("=" * 80)
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
            # Short prompt: single-shot reference vs N-chunk
            ref_logits = ref_prefill_single_shot(engine, 0, input_ids)
            test_last = chunked_prefill_n_chunks(
                engine, 0, input_ids, max_chunk_size=max(S // 2, 1))
            ref_last_logits = ref_logits[-1]
            mode = "full"
        else:
            # Long prompt: auto-chunked reference vs N-chunk with max_graph_size
            ref_all_logits = ref_prefill_chunked(engine, 0, input_ids)
            ref_last_logits = ref_all_logits[-1]
            test_last = chunked_prefill_n_chunks(
                engine, 0, input_ids, max_chunk_size=max_graph_size)
            mode = f"{n_chunks}chk"

        diff = (ref_last_logits - test_last).abs()
        max_abs_diff = diff.max().item()
        last_cos = cosine_sim(ref_last_logits, test_last)
        last_ref_tok = ref_last_logits.argmax().item()
        last_test_tok = test_last.argmax().item()
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
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    layers_per_gpu = (num_layers + pp_size - 1) // pp_size
    print(f"Model: {model_path} ({num_layers} layers)")
    print(f"Pipeline parallelism: PP={pp_size} ({layers_per_gpu} layers/GPU)")
    print(f"Total prompts tested: {len(results)}")
    print(f"  Short (single-shot ref): "
          f"{sum(1 for r in results if r['mode'] == 'full')}")
    print(f"  Long (N-chunk ref):      "
          f"{sum(1 for r in results if r['mode'] != 'full')}")
    print(f"Total time: {elapsed:.1f}s "
          f"({elapsed / len(results):.2f}s per prompt)")

    last_diffs = [r["max_abs_diff"] for r in results]
    last_cos_sims = [r["last_cos_sim"] for r in results]
    last_match_count = sum(1 for r in results if r["last_match"])

    print(f"\n--- Last-Token Metrics (all {len(results)} prompts) ---")
    print(f"Max absolute diff (worst):  {max(last_diffs):.6f}")
    print(f"Max absolute diff (mean):   "
          f"{sum(last_diffs)/len(last_diffs):.6f}")
    print(f"Max absolute diff (median): "
          f"{sorted(last_diffs)[len(last_diffs)//2]:.6f}")
    print(f"Cosine similarity (mean):   "
          f"{sum(last_cos_sims)/len(last_cos_sims):.8f}")
    print(f"Cosine similarity (worst):  {min(last_cos_sims):.8f}")
    print(f"Top-1 last-token match:     {last_match_count}/{len(results)}")

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

    print("\n" + "=" * 80)
    if last_match_count == len(results) and max(last_diffs) < 1.0:
        print("VERDICT: PASS -- All last-token predictions match, "
              "max diff < 1.0")
    elif last_match_count == len(results):
        print(f"VERDICT: PASS (with noise) -- All last-token predictions "
              f"match, max diff = {max(last_diffs):.4f}")
    else:
        print(f"VERDICT: {last_match_count}/{len(results)} last-token "
              f"matches (max diff = {max(last_diffs):.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
