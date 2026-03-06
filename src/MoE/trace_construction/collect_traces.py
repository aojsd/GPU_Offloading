"""Collect per-conversation expert traces from ShareGPT conversations.

Phase 1 of the trace construction pipeline: run each conversation through an
MoE model one at a time, recording which experts are selected at each layer
at each step. Produces one JSON trace file per conversation.

Usage:
    # Single GPU with offloading:
    python collect_traces.py \
        --model ../models/Mixtral-8x7B-Instruct-v0.1 \
        --dataset ../datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-conversations 200 \
        --experts-per-layer 2 \
        --output-dir ../datasets/ShareGPT_Vicuna/expert_traces/

    # Pipeline parallel (multi-GPU, no offloading):
    python collect_traces.py \
        --model ../models/Mixtral-8x7B-Instruct-v0.1 \
        --dataset ../datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-conversations 200 \
        --pipeline-parallel 2 \
        --output-dir ../datasets/ShareGPT_Vicuna/expert_traces/
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

# Apply glibc 2.28 monkey patches before any vLLM import
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import moe_engine as _moe_engine_mod  # noqa: F401
from moe_engine import MoEEngine


def load_sharegpt(path: str, num_conversations: int = None):
    """Load ShareGPT conversations, extracting the first human turn."""
    with open(path) as f:
        data = json.load(f)
    conversations = []
    for item in data:
        turns = item.get("conversations", [])
        human_turns = [t for t in turns if t["from"] == "human"]
        if not human_turns:
            continue
        conversations.append({
            "id": item["id"],
            "text": human_turns[0]["value"],
        })
        if num_conversations and len(conversations) >= num_conversations:
            break
    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-conversation expert traces from ShareGPT")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to ShareGPT JSON file")
    parser.add_argument("--num-conversations", type=int, default=None,
                        help="Number of conversations to process (default: all)")
    parser.add_argument("--max-output-tokens", type=int, default=-1,
                        help="Max decode tokens per conversation (-1 = no cap, run to EOS)")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="Skip conversations with prompts longer than this "
                             "(default: largest captured graph size)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trace files")
    parser.add_argument("--experts-per-layer", type=int, default=None,
                        help="Experts per layer on GPU (default: all). "
                             "Use for large models that don't fit on GPU.")
    parser.add_argument("--pipeline-parallel", "-pp", type=int, default=1,
                        help="Number of GPUs for pipeline parallelism (default: 1). "
                             "Disables offloading; all experts resident on GPU.")
    parser.add_argument("--record-router-inputs", action="store_true",
                        help="Also record router inputs (larger files)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip conversations whose trace file already exists")
    args = parser.parse_args()

    if args.pipeline_parallel > 1 and args.experts_per_layer is not None:
        parser.error("--pipeline-parallel and --experts-per-layer are mutually exclusive")

    requests_dir = os.path.join(args.output_dir, "requests")
    os.makedirs(requests_dir, exist_ok=True)

    # Read model config to get num_experts before engine creation
    with open(Path(args.model) / "config.json") as f:
        cfg = json.load(f)
    num_experts = cfg.get("num_experts") or cfg.get("num_local_experts")
    num_layers = cfg["num_hidden_layers"]
    top_k = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_topk")
    model_name = Path(args.model).name

    print(f"Model: {model_name} ({num_layers}L, {num_experts}E, top-{top_k})")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load conversations
    print(f"Loading dataset: {args.dataset}")
    conversations = load_sharegpt(args.dataset, args.num_conversations)
    print(f"Loaded {len(conversations)} conversations")

    # KV cache allocation: compute max_seq_len from available GPU memory.
    # KV per token per layer = 2 (K+V) * num_kv_heads * head_dim * dtype_bytes
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["intermediate_size"]
    dtype_bytes = 2  # BF16
    kv_bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes

    # Estimate model weight memory (per GPU for PP)
    # Non-expert: attention (QKV+O), norms, embeddings, lm_head, routers
    per_layer_non_expert = (
        hidden_size * (hidden_size + 2 * num_kv_heads * head_dim + hidden_size)  # QKV+O
        + hidden_size * 2  # 2 RMS norms
        + num_experts * hidden_size  # router
    ) * dtype_bytes
    # Expert weights: 3 matrices per expert (gate, up, down)
    per_expert_bytes = (2 * intermediate_size * hidden_size + hidden_size * intermediate_size) * dtype_bytes
    all_expert_bytes = num_layers * num_experts * per_expert_bytes
    global_non_expert = (
        num_layers * per_layer_non_expert
        + cfg["vocab_size"] * hidden_size * dtype_bytes * 2  # embed + lm_head
    )
    total_weight_bytes = global_non_expert + all_expert_bytes

    pp_size = args.pipeline_parallel
    # With PP, weights are split across GPUs (roughly evenly)
    weight_per_gpu = total_weight_bytes / max(pp_size, 1)
    # Reserve for graph buffers (~200MB per graph size per GPU) and overhead
    graph_overhead = len([1, 128, 512, 2048, 4096]) * 200 * 1024**2
    fixed_overhead = 2 * 1024**3  # 2 GB for CUDA context, activations, etc.

    gpu_total = torch.cuda.get_device_properties(0).total_memory
    available_for_kv = gpu_total - weight_per_gpu - graph_overhead - fixed_overhead
    # KV per token (all layers on this GPU)
    layers_per_gpu = num_layers // max(pp_size, 1)
    kv_bytes_per_token = kv_bytes_per_token_per_layer * layers_per_gpu
    # batch_size=1, so max_seq_len = available / kv_per_token
    max_seq_len_from_mem = int(available_for_kv / kv_bytes_per_token)
    # Clamp to something reasonable
    if args.max_output_tokens > 0 and args.max_prompt_tokens is not None:
        max_seq_len = args.max_prompt_tokens + args.max_output_tokens
    elif args.max_output_tokens > 0:
        max_seq_len = min(max_seq_len_from_mem, 32768)
    else:
        max_seq_len = min(max_seq_len_from_mem, 32768)  # safety cap
    print(f"KV cache: max_seq_len={max_seq_len} "
          f"(memory-derived limit: {max_seq_len_from_mem})")

    # Create engine — use_torch_compile=False for exact routing decisions.
    if args.pipeline_parallel > 1:
        print(f"Loading model (pipeline_parallel={args.pipeline_parallel})...")
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
            record_router_inputs=args.record_router_inputs)
        engine.trace_recorder = recorder
    else:
        epl = args.experts_per_layer if args.experts_per_layer else num_experts
        print(f"Loading model (experts_per_layer={epl})...")
        engine = MoEEngine(
            args.model,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            experts_per_layer=epl,
            use_torch_compile=False,
        )
        recorder = engine.offload_engine
        if args.record_router_inputs:
            recorder.record_router_inputs = True

    # Capture CUDA graphs (piecewise, required for offload engine / PP path)
    # 1 required for decode (single-token steps); rest cover prefill prompts.
    # Fewer sizes = less graph pool memory, enabling larger max prompt.
    # For batch replay later, add intermediate sizes (16, 32, etc).
    graph_sizes = [1, 128, 512, 2048, 4096]
    print(f"Capturing mixed CUDA graphs for sizes: {graph_sizes}")
    engine.capture_mixed_cuda_graphs(
        total_token_sizes=graph_sizes,
        use_torch_compile=False,
    )

    # Auto-set max_prompt_tokens from KV cache capacity (chunked prefill
    # handles prompts exceeding single graph size)
    max_prompt_tokens = args.max_prompt_tokens or max_seq_len
    print(f"Max prompt tokens: {max_prompt_tokens}")

    # Process conversations
    manifest = []
    skipped = 0
    t_start = time.time()

    for i, conv in enumerate(conversations):
        # Tokenize using chat template (instruct format)
        messages = [{"role": "user", "content": conv["text"]}]
        tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)
        if len(tokens) > max_prompt_tokens:
            skipped += 1
            continue
        if len(tokens) == 0:
            skipped += 1
            continue

        # Resume: skip if trace file already exists
        trace_path = os.path.join(requests_dir, f"{conv['id']}.json")
        if args.resume and os.path.exists(trace_path):
            # Load existing trace to populate manifest
            with open(trace_path) as f:
                existing = json.load(f)
            manifest.append({
                "conversation_id": conv["id"],
                "prompt_tokens": existing.get("prompt_tokens", len(tokens)),
                "output_tokens": existing.get("output_tokens", 0),
                "trace_file": f"requests/{conv['id']}.json",
            })
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(conversations)}] {conv['id']}: skipped (exists)")
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long, device=engine.device)

        # Reset engine state
        engine.reset()
        recorder.reset_trace()

        # Prefill — uses chunked prefill for prompts exceeding max graph size
        logits = engine.chunked_prefill_to_slot(0, input_ids)
        next_token = logits[-1].argmax().unsqueeze(0)

        n_decoded = 0
        decode_limit = (max_seq_len - len(tokens)) if args.max_output_tokens < 0 else args.max_output_tokens
        output_token_ids = []
        for step in range(decode_limit):
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

        # Decode output tokens to text
        output_text = tokenizer.decode(output_token_ids, skip_special_tokens=True)

        # Save trace
        trace_data = {
            "conversation_id": conv["id"],
            "model": model_name,
            "prompt_text": conv["text"],
            "output_text": output_text,
            "prompt_tokens": len(tokens),
            "output_tokens": n_decoded,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "trace": recorder.trace,
            "transfers": [],
        }

        trace_path = os.path.join(requests_dir, f"{conv['id']}.json")
        with open(trace_path, 'w') as f:
            json.dump(trace_data, f)

        if args.record_router_inputs and recorder._router_inputs:
            ri_path = os.path.splitext(trace_path)[0] + '_router_inputs.npz'
            recorder.save_router_inputs(ri_path)

        manifest.append({
            "conversation_id": conv["id"],
            "prompt_tokens": len(tokens),
            "output_tokens": n_decoded,
            "trace_file": f"requests/{conv['id']}.json",
        })

        elapsed = time.time() - t_start
        rate = (i + 1 - skipped) / elapsed if elapsed > 0 else 0
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(conversations)}] {conv['id']}: "
                  f"prompt={len(tokens)}, decoded={n_decoded}, "
                  f"rate={rate:.1f} conv/s, elapsed={elapsed:.1f}s")

    # Save manifest
    manifest_data = {
        "model": model_name,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "total_conversations": len(manifest),
        "skipped": skipped,
        "max_output_tokens": args.max_output_tokens,
        "conversations": manifest,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone! Collected {len(manifest)} traces in {elapsed:.1f}s")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {args.output_dir}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
