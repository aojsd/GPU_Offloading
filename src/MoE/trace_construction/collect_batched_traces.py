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

# Path setup: when run as a script from src/MoE/, Python adds trace_construction/
# to sys.path but not src/MoE/ itself. Add it so scheduler.py resolves.
_this_dir = Path(__file__).resolve().parent
_moe_dir = _this_dir.parent
if str(_moe_dir) not in sys.path:
    sys.path.insert(0, str(_moe_dir))
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

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
    pages_needed,
)

# Also export Scheduler under its real name
from scheduler import Scheduler                  # noqa: F401


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

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
# Memory budget computation
# ---------------------------------------------------------------------------

def compute_replay_kv_budget(
    model_config_path: str,
    cache_fraction: float,
    page_size: int = 16,
    gpu_memory_gb: float = 80.0,
    overhead_gb: float = 2.5,
    graph_sizes: list[int] | None = None,
    dtype_bytes: int = 2,
) -> dict:
    """Compute KV page budget for single-GPU replay with expert offloading.

    Memory model: GPU = non_expert_model + expert_cache + graphs + KV + overhead.
    KV gets whatever remains after the other components.

    Args:
        model_config_path: Path to model's config.json.
        cache_fraction: Fraction of total experts to keep in GPU cache (0-1).
        page_size: KV cache page size in tokens.
        gpu_memory_gb: Total GPU memory (default: 80 for H100).
        overhead_gb: Fixed overhead for CUDA context, activations, etc.
        graph_sizes: CUDA graph sizes for replay (default: GRAPH_SIZES).
            Used to estimate graph memory overhead.
        dtype_bytes: Bytes per parameter (2 for BF16).

    Returns:
        Dict with memory breakdown including 'kv_page_budget' and
        'expert_cache_size'.
    """
    with open(model_config_path) as f:
        cfg = json.load(f)

    num_layers = cfg['num_hidden_layers']
    hidden_size = cfg['hidden_size']
    intermediate_size = cfg['intermediate_size']
    num_heads = cfg['num_attention_heads']
    num_kv_heads = cfg['num_key_value_heads']
    head_dim = hidden_size // num_heads
    vocab_size = cfg['vocab_size']
    num_experts = cfg.get('num_experts') or cfg.get('num_local_experts')

    # Per-expert memory: gate_proj[I,H] + up_proj[I,H] + down_proj[H,I]
    expert_params = 2 * intermediate_size * hidden_size + hidden_size * intermediate_size
    expert_bytes = expert_params * dtype_bytes
    total_experts = num_layers * num_experts

    # Non-expert model: attention (Q/K/V/O proj) + norms + router + embeddings
    per_layer_attn = (
        (hidden_size * hidden_size) +                      # Q proj
        (hidden_size * num_kv_heads * head_dim) +          # K proj
        (hidden_size * num_kv_heads * head_dim) +          # V proj
        (hidden_size * hidden_size) +                      # O proj
        hidden_size * 2 +                                  # 2 RMS norms
        num_experts * hidden_size                           # router
    )
    non_expert_params = (
        per_layer_attn * num_layers +
        vocab_size * hidden_size +          # embed_tokens
        vocab_size * hidden_size +          # lm_head (conservative)
        hidden_size                         # final norm
    )
    non_expert_bytes = non_expert_params * dtype_bytes

    # Expert cache
    cache_size = int(total_experts * cache_fraction)
    cache_bytes = cache_size * expert_bytes

    # Graph overhead estimate
    if graph_sizes is None:
        graph_sizes = list(GRAPH_SIZES)
    graph_overhead_bytes = len(graph_sizes) * 200 * 1024 ** 2  # ~200 MB per size

    # KV per page per layer: page_size tokens * 2 (K+V) * heads * head_dim * dtype
    kv_per_page_per_layer = page_size * 2 * num_kv_heads * head_dim * dtype_bytes

    gpu_bytes = int(gpu_memory_gb * 1024 ** 3)
    overhead_bytes = int(overhead_gb * 1024 ** 3)

    available_for_kv = (gpu_bytes - non_expert_bytes - cache_bytes
                        - graph_overhead_bytes - overhead_bytes)
    kv_page_budget = max(0, int(available_for_kv / (kv_per_page_per_layer * num_layers)))

    return {
        'model': Path(model_config_path).parent.name,
        'gpu_memory_gb': gpu_memory_gb,
        'non_expert_model_gb': round(non_expert_bytes / 1024**3, 2),
        'expert_cache_size': cache_size,
        'total_experts': total_experts,
        'cache_fraction': cache_fraction,
        'expert_cache_gb': round(cache_bytes / 1024**3, 2),
        'per_expert_mb': round(expert_bytes / 1024**2, 2),
        'graph_overhead_gb': round(graph_overhead_bytes / 1024**3, 2),
        'overhead_gb': overhead_gb,
        'available_for_kv_gb': round(available_for_kv / 1024**3, 2),
        'kv_page_budget': kv_page_budget,
        'kv_capacity_tokens': kv_page_budget * page_size,
        'page_size': page_size,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Run GPU-based batched trace collection.

    Collects expert traces under the same KV memory constraints as single-GPU
    replay. Requires --cache-fraction to compute the replay-scenario KV budget.

    Example:
        python trace_construction/collect_batched_traces.py \\
            --model models/Mixtral-8x7B-Instruct-v0.1 \\
            --dataset datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json \\
            --num-conversations 200 \\
            --max-output-tokens 4096 \\
            --max-seqs 32 \\
            --pp 2 \\
            --cache-fraction 0.5 \\
            --output-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b-batched/
    """
    parser = argparse.ArgumentParser(
        description="GPU-based batched trace collection with continuous batching")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to ShareGPT JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trace files")
    parser.add_argument("--cache-fraction", type=float, required=True,
                        help="Expert cache fraction for replay (0-1). "
                             "Determines KV budget: remaining GPU memory "
                             "after non-expert model + expert cache + graphs.")
    parser.add_argument("--num-conversations", type=int, default=None,
                        help="Number of conversations (default: all)")
    parser.add_argument("--max-output-tokens", type=int, default=4096,
                        help="Per-request output token limit (default: 4096)")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="Skip prompts longer than this (default: max_seq_len)")
    parser.add_argument("--max-seqs", type=int, default=32,
                        help="Max concurrent sequences (default: 32)")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallel size for collection (default: 1)")
    parser.add_argument("--gpu-memory-gb", type=float, default=80.0,
                        help="Target GPU memory for replay budget (default: 80)")
    parser.add_argument("--kv-page-budget", type=int, default=None,
                        help="Override computed KV page budget")
    parser.add_argument("--resume", action="store_true",
                        help="Skip if output files already exist")
    args = parser.parse_args()

    # Apply glibc 2.28 patches before any vLLM import
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import moe_engine as _moe_engine_mod  # noqa: F401
    from moe_engine import MoEEngine
    from transformers import AutoTokenizer
    page_size = 16

    # Read model config
    model_config_path = str(Path(args.model) / "config.json")
    with open(model_config_path) as f:
        cfg = json.load(f)
    num_experts = cfg.get("num_experts") or cfg.get("num_local_experts")
    num_layers = cfg["num_hidden_layers"]
    top_k = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_topk")
    model_name = Path(args.model).name
    pp_size = args.pp

    print(f"Model: {model_name} ({num_layers}L, {num_experts}E, top-{top_k})")

    # Compute KV budget for single-GPU replay scenario
    mem = compute_replay_kv_budget(
        model_config_path,
        cache_fraction=args.cache_fraction,
        page_size=page_size,
        gpu_memory_gb=args.gpu_memory_gb,
        graph_sizes=list(GRAPH_SIZES),
    )
    cache_size = mem['expert_cache_size']

    if args.kv_page_budget is not None:
        kv_page_budget = args.kv_page_budget
        print(f"Using explicit KV budget override: {kv_page_budget} pages")
    else:
        kv_page_budget = mem['kv_page_budget']

    if kv_page_budget <= 0:
        raise RuntimeError(
            f"cache_fraction={args.cache_fraction} is infeasible: "
            f"expert cache ({mem['expert_cache_gb']:.1f} GB) + "
            f"non-expert model ({mem['non_expert_model_gb']:.1f} GB) + "
            f"graphs ({mem['graph_overhead_gb']:.1f} GB) + "
            f"overhead ({mem['overhead_gb']:.1f} GB) = "
            f"{mem['gpu_memory_gb'] - mem['available_for_kv_gb']:.1f} GB "
            f"> {mem['gpu_memory_gb']:.0f} GB GPU memory")

    print(f"Replay memory budget (cache_fraction={args.cache_fraction}):")
    print(f"  Non-expert model: {mem['non_expert_model_gb']:.2f} GB")
    print(f"  Expert cache: {cache_size} slots = {mem['expert_cache_gb']:.2f} GB "
          f"({mem['per_expert_mb']:.1f} MB/expert)")
    print(f"  CUDA graphs: {mem['graph_overhead_gb']:.2f} GB "
          f"({len(GRAPH_SIZES)} sizes)")
    print(f"  Overhead: {mem['overhead_gb']:.1f} GB")
    print(f"  Available for KV: {mem['available_for_kv_gb']:.2f} GB")
    print(f"  KV budget: {kv_page_budget} pages = "
          f"{kv_page_budget * page_size} tokens")

    # Prompt filter: reject prompts that can't fit even 1 output token.
    # Actual max_seq_len computed after tokenization from real conversation lengths.
    kv_capacity_tokens = kv_page_budget * page_size
    max_prompt_tokens = args.max_prompt_tokens or (kv_capacity_tokens - 1)

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

    # max_seq_len: actual max from conversations, capped by total KV budget.
    # With dynamic pages, one sequence can use all pages (preemption frees others).
    actual_max = max(len(c['prompt_token_ids']) + args.max_output_tokens
                     for c in conversations_input)
    max_seq_len = min(actual_max, kv_capacity_tokens)
    print(f"  Max seq len: {max_seq_len} tokens "
          f"(actual_max={actual_max}, kv_cap={kv_capacity_tokens})")

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
        'cache_fraction': args.cache_fraction,
        'cache_size': cache_size,
        'gpu_memory_gb': args.gpu_memory_gb,
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
