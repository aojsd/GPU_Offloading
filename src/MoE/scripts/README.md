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
    --model models/Mixtral-8x7B \
    --trace-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache70pct \
    --cache-size 179

# Single policy
python scripts/batched_replay.py \
    --model models/Mixtral-8x7B \
    --trace-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache70pct \
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
| `01_collect_traces.sh` | 1 | GPU-based batched trace collection per cache fraction (GPU, PP=N) |
| `02_policy_simulate.sh` | 2 | Run all policy simulations via `run_all_policies.py --parallel` (CPU) |
| `03_gpu_replay.sh` | 3 | Dispatch GPU replay jobs across GPUs via `batched_replay.py` |
