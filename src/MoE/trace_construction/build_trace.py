"""Build batched ActivationTrace from per-conversation expert traces.

Phase 2 of the trace construction pipeline: simulate continuous batching
offline (CPU-only) to produce a batched trace representing multiple concurrent
requests. Supports block-based KV accounting and LIFO preemption.

Two modes:
  1. Memory-first (recommended): fix expert cache fraction, maximize KV usage.
     python build_trace.py \
         --input-dir traces/mixtral-8x7b \
         --model-config models/Mixtral-8x7B/config.json \
         --cache-fraction 0.5 \
         --output traces/mixtral-8x7b/batched_cache50pct.json

  2. Legacy: target a batch size or explicit KV page budget.
     python build_trace.py \
         --input-dir traces/ \
         --target-batch-size 16 \
         --output batched_bs16.json
"""
import argparse
import json
import math
import os
import statistics
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MOE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOE_DIR))


# ---------------------------------------------------------------------------
# Per-conversation trace loading
# ---------------------------------------------------------------------------

@dataclass
class ConversationTrace:
    """A single conversation's expert trace loaded from JSON."""
    conversation_id: str
    prompt_tokens: int
    output_tokens: int
    num_layers: int
    num_experts: int
    top_k: int
    # steps[step_idx][layer_idx] = list of expert_ids
    steps: list  # list[list[list[int]]]

    @staticmethod
    def load(path: str) -> 'ConversationTrace':
        with open(path) as f:
            data = json.load(f)
        num_layers = data['num_layers']
        num_experts = data['num_experts']
        flat = data['trace']

        if not flat:
            steps = []
        else:
            max_step = max(e['step'] for e in flat)
            steps = [[[] for _ in range(num_layers)]
                     for _ in range(max_step + 1)]
            for entry in flat:
                steps[entry['step']][entry['layer']] = entry['expert_ids']

        return ConversationTrace(
            conversation_id=data.get('conversation_id', ''),
            prompt_tokens=data.get('prompt_tokens', 0),
            output_tokens=data.get('output_tokens', 0),
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=data.get('top_k', 2),
            steps=steps,
        )


def load_traces(input_dir: str) -> tuple[list[ConversationTrace], dict]:
    """Load all conversation traces and the manifest from a directory."""
    manifest_path = os.path.join(input_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        # trace_file paths in manifest are relative to input_dir
        # (e.g. "requests/conv123.json")
        trace_files = [c['trace_file'] for c in manifest['conversations']]
    else:
        # Fallback: scan requests/ subdirectory, or top-level if no subdirectory
        requests_dir = os.path.join(input_dir, "requests")
        if os.path.isdir(requests_dir):
            trace_files = sorted(
                os.path.join("requests", f)
                for f in os.listdir(requests_dir)
                if f.endswith('.json')
            )
        else:
            trace_files = sorted(
                f for f in os.listdir(input_dir)
                if f.endswith('.json') and f != 'manifest.json'
            )
        manifest = {}

    traces = []
    for fname in trace_files:
        path = os.path.join(input_dir, fname)
        traces.append(ConversationTrace.load(path))
    return traces, manifest


# ---------------------------------------------------------------------------
# Continuous batching simulator
# ---------------------------------------------------------------------------

@dataclass
class ActiveRequest:
    """State of an active request in the simulator."""
    trace: ConversationTrace
    trace_idx: int = 0              # index in original traces list
    convo_step: int = 0             # current step in per-conversation trace
    seq_len: int = 0                # current KV cache length in tokens
    needs_prefill: bool = True      # first step after admission
    admission_order: int = 0        # for LIFO victim selection


def pages_needed(seq_len: int, page_size: int) -> int:
    """KV pages for a sequence of given length."""
    if seq_len <= 0:
        return 0
    return math.ceil(seq_len / page_size)


def total_pages(running: list[ActiveRequest], page_size: int) -> int:
    """Total KV pages across all running requests."""
    return sum(pages_needed(r.seq_len, page_size) for r in running)


def simulate_batch(
    traces: list[ConversationTrace],
    kv_page_budget: int,
    page_size: int = 16,
    max_batch_size: int = None,
) -> dict:
    """Simulate continuous batching with LIFO preemption.

    All requests arrive at t=0 in FCFS order (list order).

    Args:
        traces: Per-conversation traces.
        kv_page_budget: Maximum total KV cache pages across all sequences.
        page_size: Tokens per KV page.
        max_batch_size: Hard cap on concurrent requests. None = no cap
            (only KV budget limits concurrency).

    Returns:
        Dict with keys: num_layers, num_experts, trace (flat format),
        scheduling, statistics, batch_sizes.
    """
    if not traces:
        raise ValueError("No traces provided")

    num_layers = traces[0].num_layers
    num_experts = traces[0].num_experts

    waiting = deque(range(len(traces)))  # indices into traces
    running: list[ActiveRequest] = []
    swapped: list[ActiveRequest] = []    # LIFO stack

    admit_counter = 0
    batched_steps = []    # list of list[list[int]]: steps[t][layer] = [expert_ids]
    batch_sizes = []
    total_preemptions = 0
    peak_pages = 0
    scheduling_events = []  # supplementary preemption/readmission log
    step_scheduling = []     # per-step metadata for replay

    while waiting or running or swapped:
        step_idx = len(batched_steps)
        step_events = []

        # 1. Complete: remove requests that have finished all decode steps
        #    A request at convo_step S has processed steps 0..S-1.
        #    Total steps in trace = 1 (prefill) + output_tokens (decode).
        still_running = []
        for req in running:
            total_steps = len(req.trace.steps)
            if req.convo_step >= total_steps:
                evt = {
                    'step': step_idx,
                    'event': 'complete',
                    'request_id': req.trace_idx,
                    'conversation_id': req.trace.conversation_id,
                }
                scheduling_events.append(evt)
                step_events.append(evt)
            else:
                still_running.append(req)
        running = still_running

        # 2. Re-admit from swap stack (LIFO: most recently swapped first)
        while swapped:
            candidate = swapped[-1]
            candidate_pages = pages_needed(candidate.seq_len, page_size)
            if total_pages(running, page_size) + candidate_pages <= kv_page_budget:
                swapped.pop()
                running.append(candidate)
                evt = {
                    'step': step_idx,
                    'event': 'readmit',
                    'request_id': candidate.trace_idx,
                    'conversation_id': candidate.trace.conversation_id,
                }
                scheduling_events.append(evt)
                step_events.append(evt)
            else:
                break

        # 3. Admit new requests from waiting queue (FCFS)
        while waiting:
            if max_batch_size is not None and len(running) >= max_batch_size:
                break
            idx = waiting[0]
            trace = traces[idx]
            prompt_pages = pages_needed(trace.prompt_tokens, page_size)
            if total_pages(running, page_size) + prompt_pages <= kv_page_budget:
                waiting.popleft()
                req = ActiveRequest(
                    trace=trace,
                    trace_idx=idx,
                    convo_step=0,
                    # Set seq_len to prompt_tokens immediately so page
                    # accounting is correct for subsequent admissions
                    seq_len=trace.prompt_tokens,
                    needs_prefill=True,
                    admission_order=admit_counter,
                )
                admit_counter += 1
                running.append(req)
                evt = {
                    'step': step_idx,
                    'event': 'admit',
                    'request_id': idx,
                    'conversation_id': trace.conversation_id,
                }
                scheduling_events.append(evt)
                step_events.append(evt)
            else:
                break

        if not running:
            # Deadlock check: if swapped requests exist but none can fit,
            # we have a problem (single request too large for budget)
            if swapped:
                # Force-admit the top of swap stack anyway
                req = swapped.pop()
                running.append(req)
                evt = {
                    'step': step_idx,
                    'event': 'force_readmit',
                    'request_id': req.trace_idx,
                    'conversation_id': req.trace.conversation_id,
                }
                scheduling_events.append(evt)
                step_events.append(evt)
            elif waiting:
                # Force-admit the next waiting request
                idx = waiting.popleft()
                req = ActiveRequest(
                    trace=traces[idx],
                    trace_idx=idx,
                    convo_step=0,
                    seq_len=0,
                    needs_prefill=True,
                    admission_order=admit_counter,
                )
                admit_counter += 1
                running.append(req)
                evt = {
                    'step': step_idx,
                    'event': 'force_admit',
                    'request_id': idx,
                    'conversation_id': traces[idx].conversation_id,
                }
                scheduling_events.append(evt)
                step_events.append(evt)
            else:
                break

        # 4. Record: union of all active requests' expert selections
        step_experts = [set() for _ in range(num_layers)]
        for req in running:
            if req.convo_step < len(req.trace.steps):
                for layer in range(num_layers):
                    experts = req.trace.steps[req.convo_step][layer]
                    step_experts[layer].update(experts)

        batched_steps.append([sorted(s) for s in step_experts])
        batch_sizes.append(len(running))

        # Record per-step scheduling metadata for replay
        step_scheduling.append({
            'step': step_idx,
            'batch_size': len(running),
            'active_requests': [
                {
                    'request_id': req.trace_idx,
                    'conversation_id': req.trace.conversation_id,
                    'seq_len': req.seq_len,
                    'is_prefill': req.needs_prefill,
                }
                for req in running
            ],
            'events': step_events,
        })

        # 5. Advance: move each request forward one step
        for req in running:
            if req.needs_prefill:
                # Prefill done (seq_len already set to prompt_tokens on admission)
                req.needs_prefill = False
            else:
                # Decode adds one token
                req.seq_len += 1
            req.convo_step += 1

        # Track peak pages
        current_pages = total_pages(running, page_size)
        peak_pages = max(peak_pages, current_pages)

        # 6. Preempt: if over budget, evict most recently admitted (LIFO)
        while total_pages(running, page_size) > kv_page_budget and len(running) > 1:
            # Find most recently admitted
            victim_idx = max(range(len(running)),
                             key=lambda i: running[i].admission_order)
            victim = running.pop(victim_idx)
            swapped.append(victim)
            total_preemptions += 1
            evt = {
                'step': step_idx,
                'event': 'preempt',
                'request_id': victim.trace_idx,
                'conversation_id': victim.trace.conversation_id,
            }
            scheduling_events.append(evt)
            step_events.append(evt)

    # Build flat trace format
    flat_trace = []
    for step_idx, step_layers in enumerate(batched_steps):
        for layer_idx, expert_ids in enumerate(step_layers):
            if expert_ids:
                flat_trace.append({
                    'step': step_idx,
                    'layer': layer_idx,
                    'expert_ids': expert_ids,
                })

    # Batch size distribution statistics
    bs_stats = _batch_size_stats(batch_sizes)

    return {
        'num_layers': num_layers,
        'num_experts': num_experts,
        'trace': flat_trace,
        'transfers': [],
        'batch_sizes': batch_sizes,
        'scheduling': {
            'kv_page_budget': kv_page_budget,
            'page_size': page_size,
            'max_batch_size': max_batch_size,
            'num_conversations': len(traces),
            'preemption_policy': 'lifo',
        },
        'scheduling_events': scheduling_events,
        'step_scheduling': step_scheduling,
        'statistics': {
            'total_steps': len(batched_steps),
            'total_preemptions': total_preemptions,
            'peak_pages_used': peak_pages,
            **bs_stats,
        },
    }


def _batch_size_stats(batch_sizes: list[int]) -> dict:
    """Compute batch size distribution statistics."""
    if not batch_sizes:
        return {
            'avg_batch_size': 0, 'median_batch_size': 0,
            'std_batch_size': 0, 'min_batch_size': 0,
            'peak_batch_size': 0, 'p25_batch_size': 0,
            'p75_batch_size': 0, 'iqr_batch_size': 0,
        }
    n = len(batch_sizes)
    s = sorted(batch_sizes)
    avg = sum(s) / n
    med = statistics.median(s)
    std = statistics.stdev(s) if n > 1 else 0.0
    p25 = s[n // 4]
    p75 = s[3 * n // 4]
    return {
        'avg_batch_size': round(avg, 2),
        'median_batch_size': med,
        'std_batch_size': round(std, 2),
        'min_batch_size': s[0],
        'peak_batch_size': s[-1],
        'p25_batch_size': p25,
        'p75_batch_size': p75,
        'iqr_batch_size': p75 - p25,
    }


# ---------------------------------------------------------------------------
# Memory budget computation
# ---------------------------------------------------------------------------

def _parse_model_config(model_config_path: str, dtype_bytes: int = 2) -> dict:
    """Parse model config and compute memory sizes for all components."""
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

    # Per-expert memory: w1[2*I, H] + w2[H, I]
    expert_params = 2 * intermediate_size * hidden_size + hidden_size * intermediate_size
    expert_bytes = expert_params * dtype_bytes
    total_experts = num_layers * num_experts

    # Non-expert model memory (attention + norms + embeddings + routers)
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

    # KV cache: per page per layer
    kv_per_page_per_layer = lambda page_size: page_size * 2 * num_kv_heads * head_dim * dtype_bytes

    return {
        'num_layers': num_layers,
        'num_experts': num_experts,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'vocab_size': vocab_size,
        'expert_bytes': expert_bytes,
        'total_experts': total_experts,
        'non_expert_bytes': non_expert_bytes,
        'kv_per_page_per_layer': kv_per_page_per_layer,
        'model_name': Path(model_config_path).parent.name,
    }


def compute_kv_budget_from_cache(
    model_config_path: str,
    cache_fraction: float,
    page_size: int = 16,
    gpu_memory_gb: float = 80.0,
    overhead_gb: float = 2.5,
    dtype_bytes: int = 2,
) -> dict:
    """Compute KV page budget given a fixed expert cache fraction.

    Memory allocation: GPU = non_expert_model + expert_cache + KV_cache + overhead.
    KV gets whatever remains after fixing the other components.

    Returns dict with memory breakdown and computed kv_page_budget.
    """
    mc = _parse_model_config(model_config_path, dtype_bytes)
    kv_ppl = mc['kv_per_page_per_layer'](page_size)

    cache_size = int(mc['total_experts'] * cache_fraction)
    cache_bytes = cache_size * mc['expert_bytes']

    gpu_bytes = int(gpu_memory_gb * 1024**3)
    overhead_bytes = int(overhead_gb * 1024**3)

    avail_kv = gpu_bytes - mc['non_expert_bytes'] - cache_bytes - overhead_bytes
    if avail_kv < 0:
        import warnings
        warnings.warn(
            f"Expert cache ({cache_size} slots = {cache_bytes / 1024**3:.1f} GB) + "
            f"non-expert model ({mc['non_expert_bytes'] / 1024**3:.1f} GB) + "
            f"overhead ({overhead_gb} GB) = "
            f"{(mc['non_expert_bytes'] + cache_bytes + overhead_bytes) / 1024**3:.1f} GB "
            f"exceeds GPU memory ({gpu_memory_gb} GB) by "
            f"{-avail_kv / 1024**3:.2f} GB. "
            f"cache_fraction={cache_fraction} is infeasible for replay on this GPU. "
            f"Max feasible cache_size: "
            f"{(gpu_bytes - mc['non_expert_bytes'] - overhead_bytes) // mc['expert_bytes']}"
        )
    kv_page_budget = max(0, int(avail_kv / (kv_ppl * mc['num_layers'])))

    return {
        'model': mc['model_name'],
        'gpu_memory_gb': gpu_memory_gb,
        'non_expert_model_gb': round(mc['non_expert_bytes'] / 1024**3, 2),
        'expert_cache_size': cache_size,
        'total_experts': mc['total_experts'],
        'cache_fraction': cache_fraction,
        'expert_cache_gb': round(cache_bytes / 1024**3, 2),
        'per_expert_mb': round(mc['expert_bytes'] / 1024**2, 2),
        'all_experts_gb': round(mc['total_experts'] * mc['expert_bytes'] / 1024**3, 2),
        'overhead_gb': overhead_gb,
        'available_for_kv_gb': round(avail_kv / 1024**3, 2),
        'kv_page_budget': kv_page_budget,
        'kv_capacity_tokens': kv_page_budget * page_size,
        'page_size': page_size,
    }


def compute_memory_budget(
    model_config_path: str,
    peak_pages: int,
    page_size: int,
    gpu_memory_gb: float = 80.0,
    dtype_bytes: int = 2,  # BF16
) -> dict:
    """Compute expert cache budget from KV cache usage and model config.

    Returns dict with memory breakdown and computed cache_size.
    """
    mc = _parse_model_config(model_config_path, dtype_bytes)
    kv_ppl = mc['kv_per_page_per_layer'](page_size)

    kv_total = peak_pages * kv_ppl * mc['num_layers']

    gpu_bytes = int(gpu_memory_gb * 1024**3)
    overhead_bytes = int(2.5 * 1024**3)

    remaining = gpu_bytes - mc['non_expert_bytes'] - kv_total - overhead_bytes
    cache_slots = max(0, remaining // mc['expert_bytes'])
    cache_size = min(cache_slots, mc['total_experts'])

    return {
        'model': mc['model_name'],
        'gpu_memory_gb': gpu_memory_gb,
        'kv_cache_bytes': kv_total,
        'kv_cache_gb': round(kv_total / 1024**3, 2),
        'non_expert_model_bytes': mc['non_expert_bytes'],
        'non_expert_model_gb': round(mc['non_expert_bytes'] / 1024**3, 2),
        'overhead_gb': round(overhead_bytes / 1024**3, 2),
        'per_expert_bytes': mc['expert_bytes'],
        'per_expert_mb': round(mc['expert_bytes'] / 1024**2, 2),
        'all_experts_gb': round(mc['total_experts'] * mc['expert_bytes'] / 1024**3, 2),
        'expert_cache_size': cache_size,
        'total_experts': mc['total_experts'],
        'offloading_required': cache_size < mc['total_experts'],
        'cache_fraction': round(cache_size / mc['total_experts'], 3)
            if mc['total_experts'] > 0 else 0,
        'peak_pages': peak_pages,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build batched ActivationTrace from per-conversation traces")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with per-conversation trace JSON files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output batched ActivationTrace JSON path "
                             "(auto-generated if not specified)")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model config.json (required for --cache-fraction)")
    parser.add_argument("--page-size", type=int, default=16,
                        help="KV cache page size in tokens")
    parser.add_argument("--gpu-memory-gb", type=float, default=80.0,
                        help="Total GPU memory in GB")

    # Mode 1: memory-first (recommended)
    parser.add_argument("--cache-fraction", type=float, default=None,
                        help="Expert cache as fraction of total experts (e.g. 0.5). "
                             "KV budget = remaining GPU memory. Requires --model-config.")

    # Mode 2: legacy batch-size targeting
    parser.add_argument("--target-batch-size", type=int, default=None,
                        help="Target average concurrent sequences (legacy mode)")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Max sequence length for KV budget computation (legacy)")
    parser.add_argument("--kv-page-budget", type=int, default=None,
                        help="Explicit KV page budget (overrides other modes)")

    # Shared
    parser.add_argument("--max-batch-size", type=int, default=128,
                        help="Max concurrent sequences (default: 128, matches vLLM)")

    parser.add_argument("--indices", type=int, nargs='+', default=None,
                        help="Select specific conversation indices from manifest")
    parser.add_argument("--summary", type=str, default=None,
                        help="Optional path for summary JSON")
    args = parser.parse_args()

    # Validate mode
    if args.cache_fraction is not None and not args.model_config:
        parser.error("--cache-fraction requires --model-config")

    # Load traces
    print(f"Loading traces from {args.input_dir}")
    traces, manifest = load_traces(args.input_dir)
    print(f"Loaded {len(traces)} conversation traces")

    # Filter to selected indices if specified
    if args.indices is not None:
        traces = [traces[i] for i in args.indices]
        print(f"Selected {len(traces)} conversations by index: {args.indices}")

    if not traces:
        print("Error: no traces found")
        sys.exit(1)

    # Compute KV page budget based on mode
    max_batch_size = args.max_batch_size
    memory_info = None

    if args.kv_page_budget is not None:
        # Explicit budget overrides everything
        kv_page_budget = args.kv_page_budget

    elif args.cache_fraction is not None:
        # Memory-first mode: fix expert cache, maximize KV
        memory_info = compute_kv_budget_from_cache(
            args.model_config,
            args.cache_fraction,
            args.page_size,
            args.gpu_memory_gb,
        )
        kv_page_budget = memory_info['kv_page_budget']

        print(f"\nMemory-first mode ({memory_info['model']}, "
              f"{memory_info['gpu_memory_gb']} GB GPU):")
        print(f"  Non-expert model: {memory_info['non_expert_model_gb']} GB")
        print(f"  Expert cache: {memory_info['expert_cache_size']} / "
              f"{memory_info['total_experts']} experts "
              f"({memory_info['cache_fraction']*100:.0f}%) = "
              f"{memory_info['expert_cache_gb']} GB")
        print(f"  Overhead: {memory_info['overhead_gb']} GB")
        print(f"  Available for KV: {memory_info['available_for_kv_gb']} GB")
        if memory_info['available_for_kv_gb'] < 0:
            print(f"  WARNING: Expert cache exceeds GPU memory! "
                  f"cache_fraction={args.cache_fraction} is infeasible for "
                  f"GPU replay. Batch simulation will still run (CPU-only), "
                  f"but the resulting trace cannot be replayed on a "
                  f"{memory_info['gpu_memory_gb']} GB GPU.")
        print(f"  KV page budget: {kv_page_budget} pages "
              f"({memory_info['kv_capacity_tokens']} tokens)")

    else:
        # Legacy mode: target batch size
        target_bs = args.target_batch_size or 16
        max_batch_size = target_bs
        peak_lens = sorted(t.prompt_tokens + t.output_tokens for t in traces)
        max_actual = peak_lens[-1]
        seq_len_for_budget = min(max_actual, args.max_seq_len)
        pages_per_seq = math.ceil(seq_len_for_budget / args.page_size)
        kv_page_budget = target_bs * pages_per_seq
        print(f"Actual peak seq len: {max_actual} tokens "
              f"(median: {peak_lens[len(peak_lens)//2]})")

    print(f"Max batch size: {max_batch_size}")
    print(f"KV page budget: {kv_page_budget} pages "
          f"({kv_page_budget * args.page_size} tokens capacity)")

    # Run simulation
    print("\nRunning continuous batching simulation...")
    result = simulate_batch(
        traces, kv_page_budget, args.page_size,
        max_batch_size=max_batch_size)

    # Attach memory info if available
    if memory_info:
        result['memory'] = memory_info
    elif args.model_config:
        budget = compute_memory_budget(
            args.model_config,
            result['statistics']['peak_pages_used'],
            args.page_size,
            args.gpu_memory_gb,
        )
        result['memory_budget'] = budget

    stats = result['statistics']
    print(f"\nSimulation results:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Batch size:  avg={stats['avg_batch_size']}, "
          f"median={stats['median_batch_size']}, "
          f"std={stats['std_batch_size']}")
    print(f"               min={stats['min_batch_size']}, "
          f"p25={stats['p25_batch_size']}, "
          f"p75={stats['p75_batch_size']}, "
          f"max={stats['peak_batch_size']}")
    print(f"               IQR={stats['iqr_batch_size']}")
    print(f"  Preemptions: {stats['total_preemptions']}")
    print(f"  Peak pages:  {stats['peak_pages_used']}")

    # Auto-generate output path if not specified
    if args.output is None:
        if args.cache_fraction is not None:
            pct = int(args.cache_fraction * 100)
            output = os.path.join(args.input_dir, f"batched_cache{pct}pct.json")
        elif max_batch_size is not None:
            output = os.path.join(args.input_dir, f"batched_bs{max_batch_size}.json")
        else:
            output = os.path.join(args.input_dir, "batched.json")
    else:
        output = args.output

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, 'w') as f:
        json.dump(result, f)
    print(f"\nSaved batched trace to {output}")
    print(f"  ({len(result['trace'])} flat entries, "
          f"{stats['total_steps']} steps)")

    # Optional summary
    if args.summary:
        summary = {
            'scheduling': result['scheduling'],
            'statistics': stats,
        }
        if 'memory' in result:
            summary['memory'] = result['memory']
        elif 'memory_budget' in result:
            summary['memory_budget'] = result['memory_budget']
        with open(args.summary, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()
