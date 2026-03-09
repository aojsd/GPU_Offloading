# Trace Construction Pipeline

Build realistic batched expert traces from ShareGPT conversations without
requiring a GPU for batch scheduling experiments.

## Overview

```
ShareGPT JSON ──► collect_traces.py ──► per-conversation traces (GPU, once)
                                                     │
                                                     ▼
                            build_trace.py ──► batched ActivationTrace + scheduling (CPU)
                                                     │
                                                     ▼
                            policy_simulator.py ──► GPUReplayTrace (CPU)
                                                     │
                                                     ▼
                            replay_controller.py ──► wall-clock timing (GPU)
```

**Phase 1 — Collect (GPU, once):** Run each ShareGPT conversation through the
model individually. Record per-step, per-layer expert selections and optionally
router inputs. See [collect_traces.py](collect_traces.py).

**Phase 2 — Batch (CPU, sweep freely):** Simulate continuous batching offline.
Combine per-conversation traces into a batched `ActivationTrace` with per-step
scheduling metadata (batch composition, preemptions, readmissions). Sweep
scheduling parameters without touching the GPU. See [build_trace.py](build_trace.py).

**Phase 3 — Policy (CPU, sweep freely):** Run policy simulators (LRU, Oracle,
Static, etc.) on the batched trace to produce a `GPUReplayTrace`.
Scheduling metadata propagates automatically. See
[policy_simulator.py](../policy_simulator.py).

**Phase 4 — Replay (GPU):** Replay the `GPUReplayTrace` on real hardware with
the `ReplayController`. Uses async prefetch streams and demand loading. See
[replay_controller.py](../replay_controller.py) and [replay.md](../replay.md).

**Replay fidelity contract:** The `step_scheduling` metadata produced by Phase 2
is the ground truth for batch composition. GPU replay MUST faithfully recreate
each step's mix of decode sequences, prefill chunks, and continuation chunks at
the exact token counts recorded in `step_scheduling`. This metadata propagates
through Phase 3 (`simulate()` copies it from `ActivationTrace` to
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
├── manifest.json                    # index of all per-conversation traces
├── requests/                        # per-conversation trace files (Phase 1)
│   ├── {conversation_id}.json       # e.g. QWJhYvA_0.json
│   └── ...
├── conversations/                   # human-readable conversation text
│   ├── {conversation_id}.txt
│   └── ...
│
├── cache50pct/                      # experiment: 50% expert cache
│   ├── batched.json                 # batched ActivationTrace (Phase 2)
│   ├── static_cs128.json            # Static policy GPUReplayTrace (Phase 3)
│   ├── oracle_cs128.json            # Oracle policy GPUReplayTrace
│   └── ...
│
├── cache90pct_short/                # experiment: 90% cache, few conversations
│   ├── batched.json                 # batched ActivationTrace (Phase 2)
│   └── ...                          # dm traces generated per-experiment
│
└── ...                              # more experiment directories
```

Each experiment directory is self-contained: one batched trace from Phase 2, plus
any number of policy traces from Phase 3, all sharing the same batch schedule.

---

## Phase 1: collect_traces.py

### Purpose

Run ShareGPT conversations through the MoE model one at a time, recording
per-step expert selections and router inputs. Each conversation produces one
trace file.

### Input

- ShareGPT JSON (`ShareGPT_V3_unfiltered_cleaned_split.json`)
- Model weights (OLMoE or Mixtral)
- `--num-conversations N`: how many to process (default: all)
- `--output-dir DIR`: where to write per-conversation traces
- `--max-output-tokens N`: safety cap (default: -1, no cap — run to EOS)

### What we record per layer per step

| Field | What | Source in engine | Purpose |
|-------|------|------------------|---------
| `expert_ids` | Selected top-k expert indices | `TraceRecorder` via `process_layer()` | Cache simulation, eviction policies |
| `router_input` | Hidden state fed to gating network | `TraceRecorder` (opt-in: `record_router_inputs=True`) | Predictive prefetchers |

### CUDA graph sizes

Collection uses a compact graph set `[1, 64, 128, 256]` — only 4 sizes, since
single-sequence collection never exceeds 256 tokens per step (one 256-token
prefill chunk or one decode token). This differs from `MoEEngine.vllm_graph_sizes(512)`
(51 sizes) which is designed for multi-sequence batched replay where total tokens
per step can reach 512. The smaller set saves ~12 GB of graph capture memory and
is sufficient for per-conversation tracing.

### How it works

1. Load the model into `MoEEngine` with all experts on GPU (no offloading).
2. For each conversation:
   a. Tokenize the first user turn as the prompt.
   b. Run prefill → first decode token.
   c. Run decode until EOS or max_output_tokens cap.
   d. `TraceRecorder` records `{step, layer, expert_ids}` at each layer.
   e. Save trace JSON (and optional `*_router_inputs.npz`).
   f. Reset recorder before the next conversation.
3. Save `manifest.json` indexing all conversations.

### Output format (per conversation)

```json
{
  "conversation_id": "abc123",
  "model": "Mixtral-8x7B",
  "prompt_tokens": 47,
  "output_tokens": 384,
  "num_layers": 32,
  "num_experts": 8,
  "top_k": 2,
  "trace": [
    {"step": 0, "layer": 0, "expert_ids": [2, 5]},
    {"step": 0, "layer": 1, "expert_ids": [1, 7]},
    ...
  ]
}
```

Steps 0..K-1 = prefill chunks (expert_ids = union of that chunk's tokens'
selections). Each chunk is up to 256 tokens, so a 600-token prompt produces
K=3 steps (256 + 256 + 88 tokens). Steps K..N = decode (one token each,
expert_ids = that token's top-k per layer).

---

## Phase 2: build_trace.py

### Purpose

Offline continuous batching simulator. Takes per-conversation activation traces
and scheduling parameters, produces a batched `ActivationTrace` with per-step
scheduling metadata.

### Two modes

**1. Memory-first (recommended):** Fix expert cache fraction → compute KV budget
from remaining GPU memory → run simulation with that budget.

```bash
python build_trace.py \
    --input-dir ../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b \
    --model-config /path/to/Mixtral-8x7B/config.json \
    --cache-fraction 0.5 \
    --output traces/cache50pct/batched.json
```

**2. Legacy:** Target a batch size or explicit KV page budget.

```bash
python build_trace.py \
    --input-dir traces/ \
    --target-batch-size 16 \
    --output traces/batched_bs16.json
```

### Scheduling model

Simulates vLLM-style continuous batching with scheduled chunked prefill:

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
identical across Phase 1 (collection), Phase 2 (batch simulation), and Phase 4
(replay). If chunk boundaries differ, different FP rounding from different
attention kernels (FA3 for new prefill vs FlashInfer for continuation) can change
expert routing, breaking the trace-driven guarantee that replay expert selections
match collection exactly.

Each global step:
1. **Completion:** Remove finished requests.
2. **Schedule:** Assign token budget to running decodes and continuing prefills.
3. **Admission:** Admit new requests from FCFS queue if KV budget,
   `max_seqs`, and token budget allow.
4. **Record:** Union expert selections + per-step scheduling metadata.
5. **Advance:** Update `num_computed_tokens` for prefill chunks,
   `convo_step` for completed prefills and decodes.

### No-preemption policy

Since this is a trace-driven replay where `output_tokens` is known in advance,
pages for the **full sequence** (prompt + output) are pre-allocated at admission.
This eliminates preemption entirely: no request is ever evicted once admitted.

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-dir` | Directory of per-conversation trace files | required |
| `--output` | Output path (auto-generated if omitted) | auto |
| `--cache-fraction` | Expert cache as fraction of total experts | None |
| `--model-config` | Path to config.json (required for `--cache-fraction`) | None |
| `--max-seqs` | Hard cap on concurrent sequences | **256** (vLLM default) |
| `--max-graph-size` | Max total tokens per step (single CUDA graph budget) | **512** |
| `--target-batch-size` | Target batch size (legacy mode) | 16 |
| `--kv-page-budget` | Explicit KV page budget (overrides other modes) | auto |
| `--page-size` | KV cache page size in tokens | 16 |
| `--gpu-memory-gb` | Total GPU memory | 80 |

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
    "preemption_policy": "none",
    "page_allocation": "full_sequence"
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
- Each request pre-allocates `ceil((prompt_tokens + output_tokens) / page_size)` pages.
- Total budget in pages, not tokens.
- Admit only if `total_preallocated_pages + full_sequence_pages <= kv_page_budget`.

Memory-first mode inverts the equation: given `cache_fraction` and
`gpu_memory_gb`, computes the KV page budget from remaining memory after
non-expert model weights, expert cache, and overhead.

---

## Design space

The pipeline separates concerns cleanly:

| Dimension | Controlled by |
|-----------|---------------|
| Model architecture (experts, layers, top-k) | `collect_traces.py` (GPU, run once) |
| Dataset (ShareGPT, MMLU, etc.) | `collect_traces.py` (GPU, run once) |
| Router inputs for predictive prefetching | `collect_traces.py` (GPU, run once) |
| Batch size / concurrency | `build_trace.py` (CPU, sweep freely) |
| Request arrival order | `build_trace.py` (CPU, sweep freely) |
| KV memory budget / preemption | `build_trace.py` (CPU, sweep freely) |
| Expert cache size | `policy_simulator.py` (CPU, sweep freely) |
| Eviction/prefetch policy | `policy_simulator.py` (CPU, sweep freely) |

---

## End-to-end example (Mixtral-8x7B)

```bash
# Phase 1: Collect per-conversation traces (GPU, PP=2, ~30 min for 200 convos)
VLLM_ENABLE_V1_MULTIPROCESSING=0 python collect_traces.py \
    --model /path/to/Mixtral-8x7B-Instruct-v0.1 \
    --dataset ../datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-conversations 200 \
    --max-output-tokens 1000 \
    --pipeline-parallel 2 \
    --output-dir ../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b

# Phase 2: Build batched trace with chunked prefill (CPU-only, seconds)
python build_trace.py \
    --input-dir ../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b \
    --model-config /path/to/Mixtral-8x7B-Instruct-v0.1/config.json \
    --cache-fraction 0.5 \
    --max-seqs 256 --max-graph-size 512 \
    --output ../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache50pct/batched.json

# Phases 3+4: Policy simulation + GPU replay (all-in-one)
cd /path/to/src/MoE
python scripts/batched_replay.py \
    --model /path/to/Mixtral-8x7B-Instruct-v0.1 \
    --trace-dir ../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b \
    --batched-trace .../cache50pct/batched.json \
    --cache-size 128 \
    --max-graph-size 512
```

---

## Current traces (Mixtral-8x7B, 200 ShareGPT conversations)

**Phase 1**: 200 conversations traced with PP=2 on 2x H100.
- Output token distribution: min=27, median=384, max=2576, mean=427
- Prompt token distribution: min=9, median=34, max=6368, mean=353

**Phase 2** (cache_fraction=0.5, max_seqs=128):

| Metric | Value |
|--------|-------|
| Total steps | 2847 |
| Avg batch size | 30.05 |
| Median batch size | 3 |
| Peak batch size | 128 |
| IQR | 42 |
| Preemptions | 0 |
| Peak KV pages | 5621 |

Memory budget (Mixtral-8x7B, 80 GB GPU):
- Non-expert model: 2.99 GB
- Expert cache: 128/256 experts (50%) = 42.0 GB
- Overhead: 2.5 GB (workspace buffers, CUDA graphs, allocator fragmentation)
- Available for KV: 32.51 GB = 16,645 pages = 266,320 tokens

With 270K tokens of KV capacity and only ~156K tokens total across all 200
conversations, the KV budget is never exhausted → 0 preemptions. The bottleneck
is `max_seqs=128`, not memory.
