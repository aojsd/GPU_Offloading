# Scripts

Standalone experiment runners.
Run from `src/MoE/`.

## Replay Fidelity Requirement

GPU replay MUST faithfully recreate the batch compositions produced by the
continuous batching simulator (Phase 2). Each replay step must dispatch the
correct mix of decode sequences, prefill chunks, and continuation chunks at
their traced token counts. Batch size directly drives:

1. **Compute kernel duration** — attention and MoE FFN scale with batch size
2. **Prefetch overlap window** — longer compute = more time to hide async
   CPU→GPU expert transfers behind computation
3. **PCIe contention** — concurrent transfers compete for bus bandwidth

**Only `batched_replay.py` performs faithful multi-sequence replay.**

## Scripts

### `batched_replay.py` — Multi-sequence batched GPU replay (use this)

Runs the full Phase 4 replay: loads per-conversation traces, generates batched
scheduling with chunked prefill, simulates caching/prefetching policies, and
replays on GPU with real computation. Manages multiple concurrent sequences
with `request_id -> seq_id` mapping and mixed decode/prefill/continuation steps.
**This is the only script that faithfully recreates batched compute.**

```bash
# Full replay with automatic batch simulation
python scripts/batched_replay.py \
    --model models/Mixtral-8x7B-Instruct-v0.1 \
    --trace-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b \
    --cache-fraction 0.5 \
    --max-seqs 256 --max-graph-size 512

# Replay with pre-built batched trace
python scripts/batched_replay.py \
    --model models/Mixtral-8x7B-Instruct-v0.1 \
    --trace-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b \
    --batched-trace path/to/batched.json \
    --cache-size 128

# Single policy
python scripts/batched_replay.py \
    --model models/Mixtral-8x7B-Instruct-v0.1 \
    --trace-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b \
    --cache-fraction 0.5 \
    --policy LRU-Oracle
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
| `01_collect_traces.sh` | 1 | Collect per-conversation expert traces (GPU, PP=2) |
| `02_batch_simulate.sh` | 2 | Run continuous batching simulation for all cache fractions (CPU) |
| `03_policy_simulate.sh` | 3 | Run all policy simulations via `run_all_policies.py --parallel` (CPU) |
| `04_gpu_replay.sh` | 4 | Dispatch GPU replay jobs across GPUs via `batched_replay.py` |
