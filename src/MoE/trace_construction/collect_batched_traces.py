"""
Merged Phase 1+2: GPU-based continuous batching with expert trace collection.

Implements vLLM V1-style scheduling (greedy admission, LIFO preemption with
recompute) and records expert activations + scheduling metadata simultaneously.

This module re-exports all public API from scheduler.py and provides the
collect_batched() backward-compatible wrapper + CLI entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Re-export everything from scheduler.py so existing imports keep working.
# CPU tests import BatchScheduler, ActiveState, MockPageAllocator, etc.
# from this module — the aliases ensure zero test changes.
from scheduler import (                          # noqa: F401
    Scheduler as BatchScheduler,                 # backward-compat alias
    ActiveState,
    ScheduleResult,
    PageAllocator,
    MockPageAllocator,
    extract_next_tokens,
    save_batched_trace,
    save_conversations,
    GRAPH_SIZES,
    CollectionResult,
    ReplayResult,
    load_full_tokens,
)

# Also export Scheduler under its real name
from scheduler import Scheduler                  # noqa: F401

# Re-import for internal use in collect_batched()
from trace_construction.build_trace import pages_needed  # noqa: F401


# ---------------------------------------------------------------------------
# GPU integration — backward-compat wrapper
# ---------------------------------------------------------------------------

def collect_batched(
    engine,
    conversations: list[dict],
    max_seqs: int,
    max_graph_size: int,
    page_size: int = 16,
) -> dict:
    """Run batched trace collection with continuous batching on GPU.

    Backward-compatible wrapper around Scheduler.collect(). Returns a raw
    dict (same format as before) rather than CollectionResult.

    Args:
        engine: MoEEngine with kv_page_budget set and dynamic pages enabled.
        conversations: List of dicts, each with keys:
            conversation_id, prompt_token_ids, max_output_tokens.
        max_seqs: Maximum concurrent sequences (must match engine.max_seqs).
        max_graph_size: Maximum total tokens per step.
        page_size: KV cache page size in tokens.

    Returns:
        Dict with 'all_step_scheduling', 'step_count', 'conversations',
        'trace', 'num_layers', 'num_experts'.
    """
    sched = BatchScheduler(engine, max_seqs, max_graph_size, page_size)
    result = sched.collect(conversations)
    return result.to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Run GPU-based batched trace collection.

    Example:
        python trace_construction/collect_batched_traces.py \\
            --model models/Mixtral-8x7B-Instruct-v0.1 \\
            --dataset datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json \\
            --num-conversations 200 \\
            --max-output-tokens 4096 \\
            --max-seqs 32 \\
            --pp 2 \\
            --output-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b-batched/
    """
    # Path setup for script mode: add src/MoE to path so moe_engine etc. resolve.
    _script_dir = Path(__file__).resolve().parent
    _moe_dir = _script_dir.parent
    if str(_moe_dir) not in sys.path:
        sys.path.insert(0, str(_moe_dir))
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

    parser = argparse.ArgumentParser(
        description="GPU-based batched trace collection with continuous batching")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to ShareGPT JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trace files")
    parser.add_argument("--num-conversations", type=int, default=None,
                        help="Number of conversations (default: all)")
    parser.add_argument("--max-output-tokens", type=int, default=4096,
                        help="Per-request output token limit (default: 4096)")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="Skip prompts longer than this (default: max_seq_len)")
    parser.add_argument("--max-seqs", type=int, default=32,
                        help="Max concurrent sequences (default: 32)")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallel size (default: 1)")
    parser.add_argument("--kv-page-budget", type=int, default=None,
                        help="Explicit KV page count (default: computed from GPU memory)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip if output files already exist")
    args = parser.parse_args()

    # Apply glibc 2.28 patches before any vLLM import
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import moe_engine as _moe_engine_mod  # noqa: F401
    from moe_engine import MoEEngine
    from transformers import AutoTokenizer
    from collect_traces import load_sharegpt

    page_size = 16

    # Read model config
    with open(Path(args.model) / "config.json") as f:
        cfg = json.load(f)
    num_experts = cfg.get("num_experts") or cfg.get("num_local_experts")
    num_layers = cfg["num_hidden_layers"]
    top_k = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_topk")
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["intermediate_size"]
    model_name = Path(args.model).name
    pp_size = args.pp

    print(f"Model: {model_name} ({num_layers}L, {num_experts}E, top-{top_k})")

    # Compute kv_page_budget from available GPU memory if not specified
    if args.kv_page_budget:
        kv_page_budget = args.kv_page_budget
    else:
        dtype_bytes = 2  # BF16
        kv_bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes
        per_layer_non_expert = (
            hidden_size * (hidden_size + 2 * num_kv_heads * head_dim + hidden_size)
            + hidden_size * 2
            + num_experts * hidden_size
        ) * dtype_bytes
        per_expert_bytes = (
            2 * intermediate_size * hidden_size
            + hidden_size * intermediate_size
        ) * dtype_bytes
        all_expert_bytes = num_layers * num_experts * per_expert_bytes
        global_non_expert = (
            num_layers * per_layer_non_expert
            + cfg["vocab_size"] * hidden_size * dtype_bytes * 2
        )
        total_weight_bytes = global_non_expert + all_expert_bytes
        weight_per_gpu = total_weight_bytes / max(pp_size, 1)
        graph_overhead = len(GRAPH_SIZES) * 200 * 1024 ** 2
        fixed_overhead = 3 * 1024 ** 3  # 3 GB for CUDA context + activations
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        available_for_kv = (
            gpu_total - weight_per_gpu - graph_overhead - fixed_overhead)
        if available_for_kv <= 0:
            raise RuntimeError(
                f"Insufficient GPU memory for KV cache. "
                f"Estimated weights: {weight_per_gpu / 1e9:.1f} GB, "
                f"graphs: {graph_overhead / 1e9:.1f} GB, "
                f"total GPU: {gpu_total / 1e9:.1f} GB")
        layers_per_gpu = num_layers // max(pp_size, 1)
        kv_bytes_per_token = kv_bytes_per_token_per_layer * layers_per_gpu
        max_kv_tokens = int(available_for_kv / kv_bytes_per_token)
        kv_page_budget = max(max_kv_tokens // page_size, args.max_seqs)

    max_seq_len = (kv_page_budget // args.max_seqs) * page_size
    print(f"KV budget: {kv_page_budget} pages = {kv_page_budget * page_size} tokens "
          f"({max_seq_len} per seq at max_seqs={args.max_seqs})")

    max_prompt_tokens = args.max_prompt_tokens or max_seq_len

    # Resume: check if output already exists
    batched_trace_path = os.path.join(args.output_dir, 'batched_trace.json')
    if args.resume and os.path.exists(batched_trace_path):
        print(f"Resume: {batched_trace_path} already exists, skipping collection.")
        return

    # Load tokenizer and dataset
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Loading dataset: {args.dataset}")
    raw_convs = load_sharegpt(args.dataset, args.num_conversations)
    print(f"Loaded {len(raw_convs)} raw conversations")

    # Tokenize with chat template
    conversations_input = []
    skipped = 0
    for conv in raw_convs:
        messages = [{"role": "user", "content": conv["text"]}]
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if not tokens or len(tokens) > max_prompt_tokens:
            skipped += 1
            continue
        conversations_input.append({
            'conversation_id': conv['id'],
            'prompt_token_ids': tokens,
            'max_output_tokens': args.max_output_tokens,
        })
    print(f"Tokenized: {len(conversations_input)} conversations "
          f"(skipped {skipped} for length)")

    if not conversations_input:
        print("No conversations to process. Exiting.")
        return

    # Create engine
    print(f"Loading model (pp={pp_size}, max_seqs={args.max_seqs}, "
          f"kv_page_budget={kv_page_budget})...")
    t0 = time.time()
    engine = MoEEngine(
        args.model,
        max_seqs=args.max_seqs,
        max_seq_len=max_seq_len,
        pipeline_parallel_size=pp_size,
        kv_page_budget=kv_page_budget,
        use_torch_compile=True,
    )
    print(f"Engine created in {time.time() - t0:.1f}s")

    # Capture CUDA graphs — stop on OOM
    print(f"Capturing CUDA graphs for {len(GRAPH_SIZES)} sizes...")
    captured = []
    for gs in GRAPH_SIZES:
        try:
            engine.capture_mixed_cuda_graphs(total_token_sizes=[gs])
            captured.append(gs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM at N={gs} — stopping ({len(captured)} sizes captured)")
            break
    if not captured:
        raise RuntimeError("Failed to capture any CUDA graphs")
    print(f"Captured {len(captured)} graph sizes (max={max(captured)})")

    # Run batched collection
    print(f"\nRunning batched collection: "
          f"{len(conversations_input)} conversations, "
          f"max_seqs={args.max_seqs}, max_graph_size={max(captured)}")
    t0 = time.time()
    result = collect_batched(
        engine, conversations_input,
        max_seqs=args.max_seqs,
        max_graph_size=max(captured),
        page_size=page_size,
    )
    elapsed = time.time() - t0
    print(f"Collection done: {result['step_count']} steps in {elapsed:.1f}s "
          f"({result['step_count'] / max(elapsed, 0.001):.1f} steps/s)")

    total_preemptions = sum(c['num_preemptions'] for c in result['conversations'])
    print(f"Total preemptions: {total_preemptions}")

    # Serialize outputs
    scheduling_config = {
        'kv_page_budget': kv_page_budget,
        'page_size': page_size,
        'max_seqs': args.max_seqs,
        'max_graph_size': max(captured),
        'prefill_chunk_size': 256,
        'num_conversations': len(conversations_input),
        'preemption_policy': 'lifo_recompute',
        'page_allocation': 'dynamic',
        'model': model_name,
    }
    save_batched_trace(result, args.output_dir, scheduling_config)
    manifest_entries = save_conversations(result, args.output_dir, top_k)

    manifest_data = {
        'model': model_name,
        'num_layers': result['num_layers'],
        'num_experts': result['num_experts'],
        'top_k': top_k,
        'total_conversations': len(manifest_entries),
        'step_count': result['step_count'],
        'scheduling': scheduling_config,
        'conversations': manifest_entries,
    }
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Manifest: {len(manifest_entries)} conversations → {manifest_path}")


if __name__ == "__main__":
    main()
