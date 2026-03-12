# Trace Construction Pipeline

Build realistic batched expert traces from ShareGPT conversations for
cache/prefetch policy experiments.

## Overview

```
ShareGPT JSON ──► collect_batched_traces.py ──► batched ActivationTrace + scheduling (GPU, per cache fraction)
                                                     │
                                                     ▼
                            policy_simulator.py ──► GPUReplayTrace (CPU)
                                                     │
                                                     ▼
                            replay_controller.py ──► wall-clock timing (GPU)
```

**Phase 1 — Collect (GPU, once per cache fraction):** Run continuous batching
with real GPU computation. Produces a batched `ActivationTrace` with per-step
scheduling metadata and per-conversation token sequences. KV budget is computed
from the cache fraction to match single-GPU replay constraints. See
[collect_batched_traces.py](collect_batched_traces.py).

**Phase 2 — Policy (CPU, sweep freely):** Run policy simulators (LRU, Oracle,
Static, etc.) on the batched trace to produce a `GPUReplayTrace`.
Scheduling metadata propagates automatically. See
[policy_simulator.py](../policy_simulator.py).

**Phase 3 — Replay (GPU):** Replay the `GPUReplayTrace` on real hardware with
the `ReplayController`. Uses async prefetch streams and demand loading. See
[replay_controller.py](../replay_controller.py) and [replay.md](../replay.md).

**Replay fidelity contract:** The `step_scheduling` metadata produced by Phase 1
is the ground truth for batch composition. GPU replay MUST faithfully recreate
each step's mix of decode sequences, prefill chunks, and continuation chunks at
the exact token counts recorded in `step_scheduling`. This metadata propagates
through Phase 2 (`simulate()` copies it from `ActivationTrace` to
`GPUReplayTrace`) and must be consumed by the replay loop to dispatch correct
batches. Without faithful batch recreation, compute kernel durations and
prefetch overlap windows are wrong, making timing comparisons between policies
meaningless. See [scripts/README.md](../scripts/README.md) for details.

---

## Directory Structure

Each model's per-conversation traces live under `datasets/<dataset>/expert_traces/<model>/`.
Batched traces and policy traces go in per-experiment subdirectories.

```
datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/
├── cache60pct/                      # 60% expert cache fraction
│   ├── batched_trace.json           # batched ActivationTrace (Phase 1)
│   ├── manifest.json                # per-conversation index
│   ├── requests/                    # per-conversation traces
│   │   ├── {conversation_id}.json
│   │   └── ...
│   ├── LRU-None.json                # GPUReplayTrace (Phase 2)
│   ├── LRU-Oracle.json
│   ├── Belady-Oracle.json
│   └── ...
│
├── cache70pct/                      # 70% expert cache fraction
│   ├── batched_trace.json
│   ├── manifest.json
│   ├── requests/
│   └── ...
│
└── cache80pct/                      # 80% expert cache fraction
    ├── batched_trace.json
    ├── manifest.json
    ├── requests/
    └── ...
```

Each cache fraction directory is self-contained: one batched trace from Phase 1,
plus any number of policy traces from Phase 2, all sharing the same batch schedule.

---

## Phase 1: collect_batched_traces.py

### Purpose

GPU-based batched trace collection. Runs continuous batching with real model
computation, recording expert routing and per-conversation token sequences.
One run per cache fraction, with KV budget computed to match single-GPU
replay constraints.

### Input

- ShareGPT JSON (`ShareGPT_V3_unfiltered_cleaned_split.json`)
- Model weights (Mixtral-8x7B)
- `--cache-fraction F`: expert cache fraction (required) — determines KV budget
- `--num-conversations N`: how many to process (default: 200)
- `--output-dir DIR`: where to write traces (e.g., `cache70pct/`)
- `--max-output-tokens N`: safety cap (default: 4096)
- `--pp N`: pipeline parallelism (default: auto-detect GPUs)

### What we record per layer per step

| Field | What | Source in engine | Purpose |
|-------|------|------------------|---------
| `expert_ids` | Selected top-k expert indices (batch union) | `TraceRecorder` via `process_layer()` | Cache simulation, eviction policies |

### How it works

1. Compute KV page budget via `compute_replay_kv_budget()` from cache fraction.
2. Load model into `MoEEngine` with PP=N (all experts on GPU, no offloading).
3. Run continuous batching (`Scheduler`) with real GPU computation:
   - Fixed 256-token prefill chunks, max_graph_size=512
   - Preemption when KV budget is exhausted
   - EOS detection terminates sequences
   - `TraceRecorder` records `{step, layer, expert_ids}` at each layer
4. Save `batched_trace.json` (ActivationTrace format with scheduling metadata),
   per-conversation JSONs with `prompt_token_ids` and `output_token_ids`,
   and `manifest.json`.

### Scheduling model

Implements vLLM V1-style continuous batching with scheduled chunked prefill:

```
time ──────────────────────────────────────────────────────────►

Request A: [==chunk1==][==chunk2==][decode][decode][done]
Request B:    [====chunk1====][decode][decode][decode][done]
Request C:                        [=prefill=][decode][done]
             ▲                    ▲
             A+B admitted         C admitted (budget allows)
```

Large prompts are split into fixed 256-token chunks, interleaved with decode tokens.
The total tokens per step are capped by `max_graph_size` (default: 512), which is the
single token budget per step.

**Chunk alignment invariant:** The fixed 256-token chunk boundaries must be
identical across Phase 1 (collection) and Phase 3 (replay). If chunk boundaries
differ, different FP rounding from different attention kernels (FA3 for new
prefill vs FlashInfer for continuation) can change expert routing, breaking the
trace-driven guarantee that replay expert selections match collection exactly.

Each global step:
1. **Completion:** Remove finished requests.
2. **Schedule:** Assign token budget to running decodes and continuing prefills.
3. **Admission:** Admit new requests from FCFS queue if KV budget,
   `max_seqs`, and token budget allow.
4. **Record:** Union expert selections + per-step scheduling metadata.
5. **Advance:** Update `num_computed_tokens` for prefill chunks,
   `convo_step` for completed prefills and decodes.

### Preemption policy

When KV budget is exhausted, the scheduler preempts the most recently admitted
request (LIFO). Preempted requests are re-queued and readmitted when budget
frees up, with recompute prefill from the beginning of the sequence.

**Departure from `trace_utils.py`:** The older `trace_utils.py:simulate_batch()`
uses full-sequence page pre-allocation (reserves all pages at admission) with no
preemption. The new `collect_batched_traces.py` uses greedy incremental allocation
with LIFO preemption. This produces higher peak concurrency (more requests admitted
before pages exhaust) and different scheduling decisions (preempted requests get
different expert routing on recompute due to different batch composition).
`trace_utils.py` remains useful for fast CPU-only parameter sweeps without GPU.

### Integration pitfalls

Hard-won issues encountered during GPU integration of the scheduler:

| # | Pitfall | Resolution |
|---|---------|------------|
| 5 | **Dual `seq_len` bookkeeping**: scheduler tracks `req.seq_len`, engine tracks `_seq_lens_cpu[slot]`. | Engine is authoritative. Scheduler's `req.seq_len` is only for page accounting. |
| 7 | **Continuation offsets** must equal `engine._seq_lens_cpu[slot]`, not `req.prefill_chunk_start`. | Read `engine._seq_lens_cpu[slot].item()` when building `continuation_offsets`. |
| 9 | **Zero-budget prefill zombies**: when decode tokens fill the token budget, running prefills get `scheduled_chunk=0` and hold resources without progress. | Preempt running prefills with `scheduled_chunk=0`. |
| 11 | **Per-batch expert union**: `TraceRecorder` records the union of all expert IDs across the batch, not per-conversation. | Sufficient for batched trace format. Per-conversation attribution would require extending `TraceRecorder` with a token-to-request mapping. |
| 12 | **`torch.compile` recompute divergence**: after preemption, recomputed predictions may differ from the non-preempted case due to different batch composition and graph sizes. | Expected, not a bug. `advance_state()` appends whatever the compiled model actually predicts, keeping `output_token_ids` self-consistent with the compiled model's output. |
| 19 | **Intermediate prefill tokens discarded**: only append output token on final prefill chunk. | Track `num_computed` per request. See [replay.md](../replay.md) for full derivation. |
| 20 | **`decode_step` desync after recompute**: first decode after recompute feeds stale token. | Sync `decode_step = len(output_tokens) - 1` after prefill completion. See [replay.md](../replay.md). |

### PP=2 collection memory feasibility

| Component | Per GPU (PP=2) |
|-----------|---------------|
| Model weights | ~27.5 GB |
| CUDA graphs (20 sizes, shared pool) | ~1.7 GB |
| KV cache (kv_page_budget=14,848, 50% cache) | ~14.6 GB |
| Overhead | ~2 GB |
| **Total** | **~45.8 GB** |
| **Free on H100** | **~34 GB** |

With dynamic page allocation, `max_seqs=256` is cheap (only block_table index
memory, no KV pre-allocation). The `kv_page_budget` limits total pages in use.
Greedy admission allows higher peak concurrency than `trace_utils.py`'s
conservative pre-allocation; preemption handles overflow.

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path to model directory | required |
| `--dataset` | Path to ShareGPT JSON | required |
| `--output-dir` | Output directory (e.g., `cache70pct/`) | required |
| `--cache-fraction` | Expert cache as fraction of total experts | required |
| `--num-conversations` | Number of conversations to process | 200 |
| `--max-seqs` | Hard cap on concurrent sequences | **32** |
| `--max-output-tokens` | Max output tokens per conversation | **4096** |
| `--max-graph-size` | Max total tokens per step | **512** |
| `--pp` | Pipeline parallelism degree | auto |
| `--gpu-memory-gb` | Total GPU memory for KV budget computation | **80** |
| `--kv-page-budget` | Explicit KV page budget (overrides auto) | auto |
| `--resume` | Skip conversations with existing output files | off |

### Per-step scheduling metadata

The output includes `step_scheduling` — a per-step list consumed by the replay
loop to manage KV cache and batch composition. Each entry contains:

```json
{
  "step": 42,
  "batch_size": 16,
  "total_tokens": 128,
  "active_requests": [
    {"request_id": 5, "seq_len": 128, "is_prefill": false,
     "prefill_chunk_start": 0, "prefill_chunk_length": 0, "is_continuation": false},
    {"request_id": 12, "seq_len": 0, "is_prefill": true,
     "prefill_chunk_start": 64, "prefill_chunk_length": 64, "is_continuation": true}
  ],
  "events": [
    {"step": 42, "event": "complete", "request_id": 3},
    {"step": 42, "event": "admit", "request_id": 12}
  ]
}
```

**Token type dispatch (for replay):**
- `is_prefill=false` → decode (1 token, FlashInfer paged-KV attention)
- `is_prefill=true, is_continuation=false` → new prefill (FA3 self-attention)
- `is_prefill=true, is_continuation=true` → continuation (FlashInfer paged-KV,
  offset = `prefill_chunk_start`)

**Event types:**
| Event | When | Replay action |
|-------|------|---------------|
| `complete` | Before step compute | Free KV slot, return `seq_id` to pool |
| `admit` | Before step compute | Allocate `seq_id` from free pool |
| `force_admit` | Deadlock recovery | Same as `admit` |
| `preempt` | Before step compute | Free KV slot, preserve `active_requests[rid]` with `decode_step` |

**Data flow:** `step_scheduling` is parsed by `ActivationTrace.from_flat_trace()`
into `ActivationTrace.scheduling: list[StepScheduling]`. Policy simulators
propagate it to `StepTrace.scheduling` in the `GPUReplayTrace`. The
`ReplayController` exposes helpers:
- `get_step_scheduling(step)` → `StepScheduling`
- `get_newly_admitted_seq_ids(step)` → `list[int]`

### Output format

```json
{
  "num_layers": 32,
  "num_experts": 8,
  "trace": [
    {"step": 0, "layer": 0, "expert_ids": [0, 1, 2, 5]},
    ...
  ],
  "batch_sizes": [128, 128, 127, ...],
  "step_scheduling": [ ... ],
  "scheduling": {
    "kv_page_budget": 16900,
    "page_size": 16,
    "max_seqs": 256,
    "max_graph_size": 512,
    "prefill_chunk_size": 256,
    "num_conversations": 200,
    "preemption_policy": "lifo_recompute",
    "page_allocation": "incremental"
  },
  "statistics": {
    "total_steps": 2727,
    "avg_batch_size": 31.44,
    "peak_batch_size": 183,
    ...
  }
}
```

Loadable by `ActivationTrace.from_flat_trace(data)` or `ActivationTrace.load(path)`.

### KV cache budget model

Uses **block-based page accounting** matching the engine's paged attention:
- Pages allocated incrementally on demand (greedy, not pre-allocated for full sequence).
- At admission, only checks that the first prefill chunk fits in free pages.
- Total budget in pages, not tokens. Preemption handles overflow.

Memory-first mode inverts the equation: given `cache_fraction` and
`gpu_memory_gb`, computes the KV page budget from remaining memory after
non-expert model weights, expert cache, and overhead.

---

## Design space

The pipeline separates concerns cleanly:

| Dimension | Controlled by |
|-----------|---------------|
| Model architecture (experts, layers, top-k) | `collect_batched_traces.py` (GPU, once per cache fraction) |
| Dataset (ShareGPT, MMLU, etc.) | `collect_batched_traces.py` (GPU, once per cache fraction) |
| Batch size / concurrency / KV budget | `collect_batched_traces.py` (GPU, tied to cache fraction) |
| Expert cache size | `policy_simulator.py` (CPU, sweep freely) |
| Eviction/prefetch policy | `policy_simulator.py` (CPU, sweep freely) |

---

## End-to-end example (Mixtral-8x7B)

```bash
cd /path/to/src/MoE

# Phase 1: GPU batched collection (one run per cache fraction)
python trace_construction/collect_batched_traces.py \
    --model ../../models/Mixtral-8x7B \
    --dataset ../../datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json \
    --output-dir ../../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache70pct \
    --cache-fraction 0.7 \
    --num-conversations 200 --max-seqs 32 --pp 2 --resume

# Phase 2: Policy simulation (CPU-only, seconds)
python scripts/run_all_policies.py --parallel

# Phase 3: GPU replay
python scripts/batched_replay.py \
    --model ../../models/Mixtral-8x7B \
    --trace-dir ../../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache70pct \
    --cache-size 179 --max-graph-size 512
```

---

## Current traces (Mixtral-8x7B, 200 ShareGPT conversations)

Collected with PP=2 on 2x H100, max_seqs=32, max_output_tokens=4096.
Three cache fractions: 60%, 70%, 80%.

KV budgets (computed by `compute_replay_kv_budget()`):

| Cache % | Expert slots | Cache GB | KV pages | KV tokens |
|---------|-------------|----------|----------|-----------|
| 60% | 153 | 50.2 | 10,444 | 167K |
| 70% | 179 | 58.7 | 6,076 | 97K |
| 80% | 204 | 66.9 | 1,876 | 30K |
