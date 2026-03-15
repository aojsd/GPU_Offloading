# Scripts

Standalone experiment runners.
Run from `src/MoE/`.

## Replay Fidelity Requirement

GPU replay MUST faithfully recreate the batch compositions produced by the
continuous batching scheduler (step 01). Each replay step must dispatch the
correct mix of decode sequences, prefill chunks, and continuation chunks at
their traced token counts. Batch size directly drives:

1. **Compute kernel duration** — attention and MoE FFN scale with batch size
2. **Prefetch overlap window** — longer compute = more time to hide async
   CPU→GPU expert transfers behind computation
3. **PCIe contention** — concurrent transfers compete for bus bandwidth

**Only `batched_replay.py` performs faithful multi-sequence replay.**

## Scripts

### `batched_replay.py` — Multi-sequence batched GPU replay (use this)

Loads precomputed `GPUReplayTrace` files (from `run_all_policies.py`) or
re-simulates policies on the fly, then replays on GPU with real computation.
Manages multiple concurrent sequences with `request_id -> seq_id` mapping and
mixed decode/prefill/continuation steps.
**This is the only script that faithfully recreates batched compute.**

```bash
# Replay all policies for a cache fraction
python scripts/batched_replay.py \
    --model ../../models/Mixtral-8x7B \
    --trace-dir ../../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache70pct \
    --cache-size 179

# Single policy
python scripts/batched_replay.py \
    --model ../../models/Mixtral-8x7B \
    --trace-dir ../../datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache70pct \
    --cache-size 179 \
    --policies LRU-Oracle
```

**CUDA graph sizes:** Replay uses a 20-size set with max gap of 64 tokens:
`[1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 448, 512]`.
This reduces graph capture memory from ~15 GB (51 vLLM sizes) to ~4.5 GB while
keeping padding overhead small (worst case: 64 wasted tokens).

### `run_all_policies.py` — CPU-only policy simulation (transfer counts)

Simulation-only tool: runs all cache × prefetch policy combinations and saves
`GPUReplayTrace` files consumed by `batched_replay.py` for GPU replay.

```bash
# Simulate all policies for all cache%:
python scripts/run_all_policies.py

# Parallel (one process per cache%):
python scripts/run_all_policies.py --parallel

# Single cache%:
python scripts/run_all_policies.py --cache-pct 85
```

### Shell scripts (end-to-end experiment pipeline)

| Script | Phase | Description |
|--------|-------|-------------|
| `full_pipeline.sh` | 1→3 | Runs all three phases for given `--cache-pct` values |
| `01_collect_traces.sh` | 1 | GPU-based batched trace collection per cache fraction |
| `02_policy_simulate.sh` | 2 | Run all policy simulations via `run_all_policies.py --parallel` (CPU) |
| `03_gpu_replay.sh` | 3 | Dispatch GPU replay jobs across GPUs via `batched_replay.py` |

### Memory model: cache% → KV budget → EPL

The three-phase pipeline simulates expert offloading at different expert cache
fractions.  The cache fraction determines the GPU memory layout during replay:

```
cache_pct  ──►  expert_cache_slots  ──►  KV page budget  ──►  optimal EPL
           │                         │                    │
           │  cache_pct × total      │  GPU memory left   │  Largest EPL such
           │  routed experts =       │  after expert      │  that (L × EPL +
           │  expert slots on GPU    │  cache + non-      │  scratchpad) expert
           │  during Phase 3 replay. │  expert model +    │  buffer + KV budget
           │                         │  graphs + overhead │  fits in GPU memory.
```

**Why this matters:** The KV page budget drives the continuous-batching
scheduler's admission, preemption, and batch composition.  Phase 1 must use the
*same* KV budget as Phase 3 so that traces capture identical batching patterns.

**EPL (experts_per_layer)** is only used during Phase 1 trace collection, where
the model runs with expert offloading.  It determines the physical expert buffer
size (`num_layers × EPL + num_experts` scratchpad slots).  EPL does NOT affect
the KV budget — it is derived FROM the KV budget to maximize cached experts
while leaving enough GPU memory for the KV allocation.

Both `compute_replay_kv_budget()` and `compute_optimal_epl()` live in
`trace_construction/collect_batched_traces.py`.
