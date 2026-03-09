"""Build batched ActivationTrace from per-conversation expert traces.

Phase 2 of the trace construction pipeline: simulate continuous batching
offline (CPU-only) to produce a batched trace representing multiple concurrent
requests. Uses fixed 256-token prefill chunks, max_graph_size=512 as the
single token budget, block-based KV accounting, and no-preemption page
pre-allocation.

Two modes:
  1. Memory-first (recommended): fix expert cache fraction, maximize KV usage.
     python build_trace.py \
         --input-dir traces/mixtral-8x7b \
         --model-config models/Mixtral-8x7B/config.json \
         --cache-fraction 0.5 \
         --output traces/mixtral-8x7b/cache50pct/batched.json

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

# Fixed prefill chunk size — must match collect_traces.py and batched_replay.py.
# All prefill is done in 256-token chunks (last chunk <= 256).
PREFILL_CHUNK_SIZE = 256


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
    prompt_token_ids: list = None   # token IDs for replay (avoids re-tokenization)
    output_token_ids: list = None   # generated token IDs from trace collection

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
            prompt_token_ids=data.get('prompt_token_ids'),
            output_token_ids=data.get('output_token_ids'),
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
# Continuous batching simulator with scheduled chunked prefill
# ---------------------------------------------------------------------------

@dataclass
class ActiveRequest:
    """State of an active request in the simulator."""
    trace: ConversationTrace
    trace_idx: int = 0              # index in original traces list
    convo_step: int = 0             # current step in per-conversation trace
    seq_len: int = 0                # current KV cache length in tokens
    num_computed_tokens: int = 0    # prefill progress (0 = not started)
    needs_prefill: bool = True      # True until all prompt tokens computed
    admission_order: int = 0        # for LIFO victim selection (safety net)
    # Set during scheduling for this step:
    scheduled_chunk: int = 0        # tokens scheduled this step (prefill chunk or 1 for decode)
    prefill_chunk_start: int = 0    # offset into prompt for this chunk
    is_continuation: bool = False   # True if prefill_chunk_start > 0


def pages_needed(seq_len: int, page_size: int) -> int:
    """KV pages for a sequence of given length."""
    if seq_len <= 0:
        return 0
    return math.ceil(seq_len / page_size)


def _total_preallocated(running: list[ActiveRequest], page_size: int) -> int:
    """Total pre-allocated pages for all running requests."""
    return sum(
        pages_needed(r.trace.prompt_tokens + r.trace.output_tokens, page_size)
        for r in running
    )


def simulate_batch(
    traces: list[ConversationTrace],
    kv_page_budget: int,
    page_size: int = 16,
    max_seqs: int = None,
    max_graph_size: int = None,
    prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
    original_indices: list[int] = None,
) -> dict:
    """Simulate continuous batching with fixed-size chunked prefill.

    Uses no-preemption policy: pages for the full sequence (prompt + output)
    are pre-allocated at admission time. Since output_tokens is known from
    the trace, this eliminates preemption entirely.

    Prefill uses fixed chunk sizes (default 256 tokens, last chunk <= 256).
    Each chunk maps to one convo_step in the per-conversation trace, matching
    how collect_traces.py records per-chunk expert routing. The token budget
    is max_graph_size (default 512), which is the single cap on total tokens
    per step (decode + prefill).

    Args:
        traces: Per-conversation traces.
        kv_page_budget: Maximum total KV cache pages across all sequences.
        page_size: Tokens per KV page.
        max_seqs: Hard cap on concurrent requests. None = no cap.
        max_graph_size: Max total tokens that fit in a captured CUDA graph.
            This is the single token budget per step.
        prefill_chunk_size: Fixed prefill chunk size (default 256).

    Returns:
        Dict with keys: num_layers, num_experts, trace (flat format),
        scheduling, statistics, batch_sizes.
    """
    if not traces:
        raise ValueError("No traces provided")

    num_layers = traces[0].num_layers
    num_experts = traces[0].num_experts

    waiting = deque(range(len(traces)))  # indices into traces
    # Map local index -> original manifest index (identity when not filtered)
    _idx_map = original_indices if original_indices is not None else list(range(len(traces)))
    running: list[ActiveRequest] = []

    admit_counter = 0
    batched_steps = []    # list of list[list[int]]: steps[t][layer] = [expert_ids]
    batch_sizes = []
    peak_pages = 0
    budget_warned = False
    scheduling_events = []  # supplementary event log
    step_scheduling = []     # per-step metadata for replay

    # Effective per-step token cap — max_graph_size is the single budget
    token_cap = max_graph_size if max_graph_size is not None else float('inf')

    while waiting or running:
        step_idx = len(batched_steps)
        step_events = []

        # 1. Complete: remove requests that have finished all decode steps.
        #    A request is complete when convo_step >= len(trace.steps).
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

        # 2. Schedule running requests and compute token budget.
        #    Decodes cost 1 token each. Prefill chunks are fixed-size (256).
        token_budget = token_cap if token_cap != float('inf') else float('inf')

        # First pass: count decode tokens (fixed cost, always scheduled)
        num_decode_tokens = sum(1 for r in running if not r.needs_prefill)
        if token_budget != float('inf'):
            token_budget -= num_decode_tokens
            token_budget = max(token_budget, 0)

        # Second pass: schedule continuing prefill chunks with fixed size.
        # Each prefill chunk is exactly prefill_chunk_size (or remainder).
        # Only schedule if the full chunk fits in the remaining budget.
        for req in running:
            if req.needs_prefill:
                remaining_prompt = req.trace.prompt_tokens - req.num_computed_tokens
                chunk = min(prefill_chunk_size, remaining_prompt)
                if token_budget == float('inf') or token_budget >= chunk:
                    if token_budget != float('inf'):
                        token_budget -= chunk
                    req.scheduled_chunk = chunk
                    req.prefill_chunk_start = req.num_computed_tokens
                    req.is_continuation = (req.num_computed_tokens > 0)
                else:
                    # Not enough budget for this prefill chunk — skip this step
                    req.scheduled_chunk = 0
                    req.prefill_chunk_start = req.num_computed_tokens
                    req.is_continuation = (req.num_computed_tokens > 0)
            else:
                req.scheduled_chunk = 1
                req.prefill_chunk_start = 0
                req.is_continuation = False

        # 3. Admit new requests from waiting queue (FIFO).
        #    Pre-allocate pages for full sequence (prompt + output).
        #    First prefill chunk is fixed-size (min(prefill_chunk_size, prompt)).
        while waiting:
            if max_seqs is not None and len(running) >= max_seqs:
                break
            idx = waiting[0]
            trace = traces[idx]
            full_pages = pages_needed(
                trace.prompt_tokens + trace.output_tokens, page_size)
            current_pages = _total_preallocated(running, page_size)
            if current_pages + full_pages > kv_page_budget:
                break

            # First chunk is fixed-size
            first_chunk = min(prefill_chunk_size, trace.prompt_tokens)
            if token_budget != float('inf') and token_budget < first_chunk:
                break

            waiting.popleft()
            if token_budget != float('inf'):
                token_budget -= first_chunk

            req = ActiveRequest(
                trace=trace,
                trace_idx=_idx_map[idx],
                convo_step=0,
                seq_len=0,
                num_computed_tokens=0,
                needs_prefill=True,
                admission_order=admit_counter,
                scheduled_chunk=first_chunk,
                prefill_chunk_start=0,
                is_continuation=False,
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

        if not running:
            if waiting:
                # Force-admit: single request too large for KV budget
                idx = waiting.popleft()
                trace = traces[idx]
                first_chunk = min(prefill_chunk_size, trace.prompt_tokens)
                req = ActiveRequest(
                    trace=trace,
                    trace_idx=_idx_map[idx],
                    convo_step=0,
                    seq_len=0,
                    num_computed_tokens=0,
                    needs_prefill=True,
                    admission_order=admit_counter,
                    scheduled_chunk=first_chunk,
                    prefill_chunk_start=0,
                    is_continuation=False,
                )
                admit_counter += 1
                running.append(req)
                evt = {
                    'step': step_idx,
                    'event': 'force_admit',
                    'request_id': idx,
                    'conversation_id': trace.conversation_id,
                }
                scheduling_events.append(evt)
                step_events.append(evt)
            else:
                break

        # 4. Record: union of all scheduled requests' expert selections.
        #    Each prefill chunk maps to its own convo_step (per-chunk routing
        #    from collect_traces.py). Skip requests with scheduled_chunk=0.
        step_experts = [set() for _ in range(num_layers)]
        for req in running:
            if req.scheduled_chunk > 0 and req.convo_step < len(req.trace.steps):
                for layer in range(num_layers):
                    experts = req.trace.steps[req.convo_step][layer]
                    step_experts[layer].update(experts)

        batched_steps.append([sorted(s) for s in step_experts])

        # Only count scheduled requests (skip prefills with 0 budget)
        scheduled = [r for r in running if r.scheduled_chunk > 0]
        total_tokens_this_step = sum(r.scheduled_chunk for r in scheduled)
        batch_sizes.append(len(scheduled))

        # Record per-step scheduling metadata for replay
        step_scheduling.append({
            'step': step_idx,
            'batch_size': len(scheduled),
            'total_tokens': total_tokens_this_step,
            'active_requests': [
                {
                    'request_id': req.trace_idx,
                    'conversation_id': req.trace.conversation_id,
                    'seq_len': req.seq_len,
                    'is_prefill': req.needs_prefill,
                    'prefill_chunk_start': req.prefill_chunk_start,
                    'prefill_chunk_length': req.scheduled_chunk if req.needs_prefill else 0,
                    'is_continuation': req.is_continuation,
                }
                for req in scheduled
            ],
            'events': step_events,
        })

        # 5. Advance: update state based on scheduled tokens.
        #    Each prefill chunk advances convo_step by 1 (per-chunk trace).
        for req in running:
            if req.needs_prefill:
                if req.scheduled_chunk > 0:
                    req.num_computed_tokens += req.scheduled_chunk
                    req.seq_len = req.num_computed_tokens
                    req.convo_step += 1  # each chunk = one trace step
                    if req.num_computed_tokens >= req.trace.prompt_tokens:
                        req.needs_prefill = False
                # else: skipped this step, no advance
            else:
                req.seq_len += 1
                req.convo_step += 1

        # Track peak pages
        current_pages = _total_preallocated(running, page_size)
        peak_pages = max(peak_pages, current_pages)

        # No preemption: pages are pre-allocated for the full sequence.
        # Safety check: warn if we exceed budget (force_admit can cause
        # this legitimately; otherwise indicates a bug).
        if current_pages > kv_page_budget and not budget_warned:
            import warnings
            warnings.warn(
                f"Step {step_idx}: page usage {current_pages} exceeds "
                f"budget {kv_page_budget}. Expected only with force_admit."
            )
            budget_warned = True

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
            'max_seqs': max_seqs,
            'max_graph_size': max_graph_size,
            'prefill_chunk_size': prefill_chunk_size,
            'num_conversations': len(traces),
            'preemption_policy': 'none',
            'page_allocation': 'full_sequence',
        },
        'scheduling_events': scheduling_events,
        'step_scheduling': step_scheduling,
        'statistics': {
            'total_steps': len(batched_steps),
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
    parser.add_argument("--max-seqs", type=int, default=256,
                        help="Max concurrent sequences (default: 256, matches vLLM)")
    parser.add_argument("--max-graph-size", type=int, default=512,
                        help="Max total tokens per step / CUDA graph "
                             "(default: 512)")
    parser.add_argument("--prefill-chunk-size", type=int, default=PREFILL_CHUNK_SIZE,
                        help=f"Fixed prefill chunk size (default: {PREFILL_CHUNK_SIZE})")

    parser.add_argument("--indices", type=int, nargs='+', default=None,
                        help="Select specific conversation indices from manifest")
    parser.add_argument("--summary", type=str, default=None,
                        help="Optional path for summary JSON")
    args = parser.parse_args()

    # Validate mode
    if args.cache_fraction is not None and not args.model_config:
        parser.error("--cache-fraction requires --model-config")

    max_graph_size = args.max_graph_size

    # Load traces
    print(f"Loading traces from {args.input_dir}")
    traces, manifest = load_traces(args.input_dir)
    print(f"Loaded {len(traces)} conversation traces")

    # Filter to selected indices if specified
    original_indices = None
    if args.indices is not None:
        original_indices = args.indices
        traces = [traces[i] for i in args.indices]
        print(f"Selected {len(traces)} conversations by index: {args.indices}")

    if not traces:
        print("Error: no traces found")
        sys.exit(1)

    # Compute KV page budget based on mode
    max_seqs = args.max_seqs
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
        max_seqs = target_bs
        peak_lens = sorted(t.prompt_tokens + t.output_tokens for t in traces)
        max_actual = peak_lens[-1]
        seq_len_for_budget = min(max_actual, args.max_seq_len)
        pages_per_seq = math.ceil(seq_len_for_budget / args.page_size)
        kv_page_budget = target_bs * pages_per_seq
        print(f"Actual peak seq len: {max_actual} tokens "
              f"(median: {peak_lens[len(peak_lens)//2]})")

    print(f"Max seqs: {max_seqs}")
    print(f"Max graph size: {max_graph_size}")
    print(f"Prefill chunk size: {args.prefill_chunk_size}")
    print(f"KV page budget: {kv_page_budget} pages "
          f"({kv_page_budget * args.page_size} tokens capacity)")

    # Run simulation
    print("\nRunning continuous batching simulation...")
    result = simulate_batch(
        traces, kv_page_budget, args.page_size,
        max_seqs=max_seqs,
        max_graph_size=max_graph_size,
        prefill_chunk_size=args.prefill_chunk_size,
        original_indices=original_indices,
    )

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
    print(f"  Peak pages:  {stats['peak_pages_used']}")

    # Count chunked prefill steps
    chunked_steps = 0
    for ss in result['step_scheduling']:
        for ar in ss['active_requests']:
            if ar['is_continuation']:
                chunked_steps += 1
                break
    print(f"  Steps with continuations: {chunked_steps}")

    # Auto-generate output path if not specified
    if args.output is None:
        if args.cache_fraction is not None:
            pct = int(args.cache_fraction * 100)
            cache_dir = os.path.join(args.input_dir, f"cache{pct}pct")
            os.makedirs(cache_dir, exist_ok=True)
            output = os.path.join(cache_dir, "batched.json")
        elif max_seqs is not None:
            output = os.path.join(args.input_dir, f"batched_bs{max_seqs}.json")
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
