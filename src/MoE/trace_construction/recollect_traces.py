"""Re-collect specific conversation traces with different parameters.

Targeted re-collection: loads the engine once, then re-runs only the specified
conversations (by index from manifest). Updates the trace files in-place and
patches the manifest.

Usage:
    python recollect_traces.py \
        --trace-dir ../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b \
        --model ../models/Mixtral-8x7B-Instruct-v0.1 \
        --dataset ../datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json \
        --indices 50 146 \
        --max-output-tokens 1000 \
        --pipeline-parallel 2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MOE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOE_DIR))

import torch
from transformers import AutoTokenizer

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import moe_engine as _moe_engine_mod  # noqa: F401
from moe_engine import MoEEngine


def main():
    parser = argparse.ArgumentParser(
        description="Re-collect specific conversation traces")
    parser.add_argument("--trace-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs='+', required=True,
                        help="Manifest indices to re-collect")
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--pipeline-parallel", "-pp", type=int, default=1)
    args = parser.parse_args()

    # Load manifest and index mapping
    with open(os.path.join(args.trace_dir, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(args.trace_dir, "index_mapping.json")) as f:
        index_map = json.load(f)

    # Build index -> manifest entry + trace file mapping
    targets = {}
    for entry in index_map:
        if entry['index'] in args.indices:
            targets[entry['index']] = entry

    if len(targets) != len(args.indices):
        missing = set(args.indices) - set(targets.keys())
        print(f"Error: indices {missing} not found in index_mapping.json")
        sys.exit(1)

    # Load dataset to get conversation text
    with open(args.dataset) as f:
        data = json.load(f)
    convos_by_id = {}
    for item in data:
        turns = item.get("conversations", [])
        human_turns = [t for t in turns if t["from"] == "human"]
        if human_turns:
            convos_by_id[item["id"]] = human_turns[0]["value"]

    # Read model config
    with open(Path(args.model) / "config.json") as f:
        cfg = json.load(f)
    num_layers = cfg['num_hidden_layers']
    num_experts = cfg.get('num_experts') or cfg.get('num_local_experts')
    top_k = cfg.get('num_experts_per_tok') or cfg.get('num_experts_per_topk')
    model_name = Path(args.model).name

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Compute max_seq_len
    max_prompt = max(
        len(tokenizer.apply_chat_template(
            [{"role": "user", "content": convos_by_id[t['conversation_id']]}],
            add_generation_prompt=True))
        for t in targets.values()
    )
    max_seq_len = max_prompt + args.max_output_tokens + 256  # margin

    # Create engine
    print(f"Loading model (pp={args.pipeline_parallel})...")
    engine = MoEEngine(
        args.model,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        pipeline_parallel_size=args.pipeline_parallel,
        use_torch_compile=False,
    )
    from trace_recorder import TraceRecorder
    recorder = TraceRecorder(
        num_layers=num_layers, num_experts=num_experts,
        record_router_inputs=False)
    engine.trace_recorder = recorder

    graph_sizes = [1, 128, 512, 2048, 4096]
    print(f"Capturing CUDA graphs for sizes: {graph_sizes}")
    engine.capture_mixed_cuda_graphs(
        total_token_sizes=graph_sizes,
        use_torch_compile=False,
    )

    # Re-collect each target
    for idx in sorted(targets.keys()):
        entry = targets[idx]
        conv_id = entry['conversation_id']
        trace_file = os.path.join(args.trace_dir, entry['trace_file'])
        text = convos_by_id[conv_id]

        messages = [{"role": "user", "content": text}]
        tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)
        input_ids = torch.tensor(tokens, dtype=torch.long, device=engine.device)

        engine.reset()
        recorder.reset_trace()

        # Prefill
        logits = engine.chunked_prefill_to_slot(0, input_ids)
        next_token = logits[-1].argmax().unsqueeze(0)

        n_decoded = 0
        output_token_ids = []
        t0 = time.time()
        for step in range(args.max_output_tokens):
            logits = engine.mixed_step(
                decode_seq_ids=[0],
                decode_token_ids=next_token,
                prefill_seq_ids=[],
                prefill_input_ids=[],
            )
            next_token = logits[0].argmax().unsqueeze(0)
            output_token_ids.append(next_token.item())
            n_decoded += 1
            if next_token.item() == engine.eos_token_id:
                break
        elapsed = time.time() - t0

        output_text = tokenizer.decode(output_token_ids, skip_special_tokens=True)
        hit_eos = (n_decoded < args.max_output_tokens)

        # Save trace (overwrite)
        trace_data = {
            "conversation_id": conv_id,
            "model": model_name,
            "prompt_text": text,
            "output_text": output_text,
            "prompt_tokens": len(tokens),
            "output_tokens": n_decoded,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "trace": recorder.trace,
            "transfers": [],
        }
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f)

        # Update manifest
        for mc in manifest['conversations']:
            if mc.get('index') == idx or mc.get('conversation_id') == conv_id:
                mc['output_tokens'] = n_decoded
                break

        print(f"  [{idx}] {conv_id}: prompt={len(tokens)}, "
              f"decoded={n_decoded} ({'EOS' if hit_eos else 'CAP'}), "
              f"{elapsed:.1f}s")

    # Save updated manifest
    with open(os.path.join(args.trace_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Updated {len(targets)} traces and manifest.")


if __name__ == "__main__":
    main()
