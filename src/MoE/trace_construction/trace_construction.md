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
                            policy_simulator.py ──► DataMovementTrace (CPU)
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
Static, etc.) on the batched trace to produce a `DataMovementTrace`.
Scheduling metadata propagates automatically. See
[policy_simulator.py](../policy_simulator.py).

**Phase 4 — Replay (GPU):** Replay the `DataMovementTrace` on real hardware with
the `ReplayController`. Uses async prefetch streams and demand loading. See
[replay_controller.py](../replay_controller.py) and [replay.md](../replay.md).

---

## Directory Structure

Each model's per-conversation traces live under `datasets/<dataset>/expert_traces/<model>/`.
Batched traces and policy traces go in per-experiment subdirectories.

```
datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/
├── manifest.json                    # index of all per-conversation traces
├── index_mapping.json               # maps manifest indices to dataset IDs
├── requests/                        # per-conversation trace files (Phase 1)
│   ├── 000.json
│   ├── 001.json
│   └── ...
├── conversations/                   # human-readable conversation text
│   ├── 000.txt
│   └── ...
│
├── cache50pct/                      # experiment: 50% expert cache
│   ├── batched.json                 # batched ActivationTrace (Phase 2)
│   ├── static_cs128.json            # Static policy DataMovementTrace (Phase 3)
│   ├── oracle_cs128.json            # Oracle policy DataMovementTrace
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
- `--max-output-tokens N`: safety cap (default: 1000)

### What we record per layer per step

| Field | What | Source in engine | Purpose |
|-------|------|------------------|---------
| `expert_ids` | Selected top-k expert indices | `TraceRecorder` via `process_layer()` | Cache simulation, eviction policies |
| `router_input` | Hidden state fed to gating network | `TraceRecorder` (opt-in: `record_router_inputs=True`) | Predictive prefetchers |

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

Step 0 = prefill (expert_ids = union of all prompt tokens' selections).
Steps 1..N = decode (one token each, expert_ids = that token's top-k per layer).

### Targeted re-collection

Use [recollect_traces.py](recollect_traces.py) to re-collect specific
conversations by manifest index with different parameters (e.g., higher
`--max-output-tokens`), without re-running the full collection.

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

Simulates a simplified vLLM/Orca-style continuous batching scheduler:

```
time ──────────────────────────────────────────────────────────►

Request A: [===prefill===][decode][decode][decode][done]
Request B:        [==prefill==][decode][decode][decode][decode][done]
Request C:                          [=prefill=][decode][done]
                  ▲               ▲           ▲
                  admit B         admit C      admit next...
```

Each global step:
1. **Completion:** Remove finished requests.
2. **Re-admission:** Pop from LIFO swap stack if budget allows.
3. **Admission:** Admit new requests from FCFS queue if KV budget and
   `max_batch_size` allow.
4. **Record:** Union expert selections + per-step scheduling metadata.
5. **Advance:** Increment seq_lens, clear prefill flag.
6. **Preempt:** If over KV budget, evict most recently admitted (LIFO).

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-dir` | Directory of per-conversation trace files | required |
| `--output` | Output path (auto-generated if omitted) | auto |
| `--cache-fraction` | Expert cache as fraction of total experts | None |
| `--model-config` | Path to config.json (required for `--cache-fraction`) | None |
| `--max-batch-size` | Hard cap on concurrent sequences | **128** (vLLM default) |
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
  "active_requests": [
    {"request_id": 5, "conversation_id": "abc", "seq_len": 128, "is_prefill": false},
    {"request_id": 12, "conversation_id": "xyz", "seq_len": 64, "is_prefill": true},
    ...
  ],
  "events": [
    {"step": 42, "event": "complete", "request_id": 3, "conversation_id": "..."},
    {"step": 42, "event": "admit", "request_id": 12, "conversation_id": "xyz"},
    {"step": 42, "event": "preempt", "request_id": 8, "conversation_id": "..."}
  ]
}
```

**Event types:**
| Event | When | Replay action |
|-------|------|---------------|
| `complete` | Before step compute | Free KV pages for finished request |
| `readmit` | Before step compute | Re-prefill (recompute model) or restore KV (swap model) |
| `admit` | Before step compute | Allocate KV slot, run initial prefill |
| `preempt` | After step compute | Free KV pages, save state if using swap model |
| `force_readmit` | Deadlock recovery | Same as `readmit` |
| `force_admit` | Deadlock recovery | Same as `admit` |

**Data flow:** `step_scheduling` is parsed by `ActivationTrace.from_flat_trace()`
into `ActivationTrace.scheduling: list[StepScheduling]`. Policy simulators
propagate it to `StepTrace.scheduling` in the `DataMovementTrace`. The
`ReplayController` exposes helpers:
- `get_step_scheduling(step)` → `StepScheduling`
- `get_preempted_seq_ids(step)` → `list[int]`
- `get_readmitted_seq_ids(step)` → `list[int]`
- `get_newly_admitted_seq_ids(step)` → `list[int]`

The engine provides `engine.free_seq(seq_id)` to reset KV for preempted sequences.

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
    "max_batch_size": 128,
    "num_conversations": 200,
    "preemption_policy": "lifo"
  },
  "statistics": {
    "total_steps": 2847,
    "avg_batch_size": 30.05,
    "median_batch_size": 3,
    "peak_batch_size": 128,
    ...
  }
}
```

Loadable by `ActivationTrace.from_flat_trace(data)` or `ActivationTrace.load(path)`.

### KV cache budget model

Uses **block-based page accounting** matching the engine's paged attention:
- Each active request uses `ceil(current_seq_len / page_size)` pages.
- Total budget in pages, not tokens.
- Admit a new request if `total_pages + prompt_pages <= kv_page_budget`.
- Each decode step increments each request's seq_len by 1.

Memory-first mode inverts the equation: given `cache_fraction` and
`gpu_memory_gb`, computes the KV page budget from remaining memory after
non-expert model weights, expert cache, and overhead.

### LIFO preemption

When a decode step pushes total pages over budget:
1. Select victim: most recently admitted request (LIFO).
2. Pause and push to LIFO swap stack. Pages logically freed.
3. When space frees up, pop from swap stack (LIFO — most recently swapped
   = first back). Resume decode from where it left off.

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
# Phase 1: Collect per-conversation traces (GPU, ~30 min for 200 convos)
VLLM_ENABLE_V1_MULTIPROCESSING=0 python collect_traces.py \
    --model /path/to/Mixtral-8x7B \
    --dataset ../datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-conversations 200 \
    --max-output-tokens 1000 \
    --pipeline-parallel 2 \
    --output-dir ../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b

# Phase 2: Build batched trace (CPU-only, seconds)
mkdir -p ../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache50pct
python build_trace.py \
    --input-dir ../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b \
    --model-config /path/to/Mixtral-8x7B/config.json \
    --cache-fraction 0.5 \
    --output ../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache50pct/batched.json

# Phase 3: Run policy simulation (CPU-only, seconds)
cd /path/to/src/MoE
python -c "
from data_movement_trace import ActivationTrace
from policy_simulator import StaticPolicy, OraclePolicy
at = ActivationTrace.load('datasets/.../cache50pct/batched.json')
cs = 128  # from memory budget

dm = StaticPolicy().simulate(at, cache_size=cs)
dm.save('datasets/.../cache50pct/static_cs128.json')
print('Static:', dm.summary())

dm = OraclePolicy().simulate(at, cache_size=cs)
dm.save('datasets/.../cache50pct/oracle_cs128.json')
print('Oracle:', dm.summary())
"

# Phase 4: Replay on GPU hardware (measures wall-clock timing)
# python replay_experiment.py --trace cache50pct/oracle_cs128.json ...
```

---

## Current traces (Mixtral-8x7B, 200 ShareGPT conversations)

**Phase 1**: 200 conversations traced with PP=2 on 2x H100.
- Output token distribution: min=27, median=384, max=2576, mean=427
- Prompt token distribution: min=9, median=34, max=6368, mean=353

**Phase 2** (cache_fraction=0.5, max_batch_size=128):

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
is `max_batch_size=128`, not memory.
