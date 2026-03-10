# Piecewise CUDA Graph Memory

## Background

Piecewise CUDA graphs capture 3 stages per layer per batch size for expert
offloading (CPU break between router output and MoE compute). With 32 layers
and 20 batch sizes: **1920 graphs** on a single GPU.

Without pool sharing, this consumed **~25.7 GB/GPU** — dominated by redundant
private allocator pools, not command buffers. Fixed 2026-03-09 via shared
`graph_pool_handle()`, reducing to **~1.7 GB/GPU (93.5% reduction)**.

## Root Cause and Fix

### The Problem: Private Allocator Pools

Without a `pool=` argument, `torch.cuda.graph()` creates a private allocator
pool per graph. Temporaries freed during capture stay reserved in that pool.
With 48 graphs/GPU/size (3 stages x 16 layers for PP=2), each holding ~56 MB
of identically-shaped workspace at N=512, the 48x redundancy drove the cost.

Per-graph cost decomposition:
- CUDA driver command buffers: **~3.1 MB/graph** (fixed, irreducible)
- Private pool reservations: **0-56 MB/graph** (scales with N, **eliminated by fix**)

### The Fix: `pool=graph_pool_handle()` (implemented)

All piecewise graphs now share a single pool per device. The pool only needs
`max(peak_of_any_single_graph)` instead of `sum(all_graphs)`.

Safe because: (1) graphs replay strictly sequentially, (2) all outputs use
`.copy_()` to pre-allocated buffers outside the pool — zero live tensors remain
in the pool after each capture. This is the same pattern vLLM uses
(`vllm/compilation/cuda_graph.py`).

**Does NOT apply to `capture_prefill_cuda_graph()`** (full-forward path, line
835). That path captures FA3 attention inside the graph; FA3 workspace
aliasing across graphs causes illegal memory access with shared pools. The
piecewise path never captures attention, so the failure mode doesn't apply.

## Measured Results

Config: PP=2, max_seqs=8, max_seq_len=256, torch.compile=True, Mixtral-8x7B (32L)

### Before Fix (private pools)

| N | Per-GPU delta (MB) | Cumulative total (GB) |
|-----|-------------------:|----------------------:|
| 1 | -374 (first-capture noise) | -0.73 |
| 2 | 148 | -0.44 |
| 4 | 152 | -0.14 |
| 8 | 150 | 0.15 |
| 16 | 470 | 1.07 |
| 32 | 503 | 2.05 |
| 64 | 536 | 4.08 |
| 128 | 1146 | 8.37 |
| 256 | 1844 | 21.68 |
| 512 | 2858 | 51.37 |

**Total: 51.4 GB (both GPUs), ~25.7 GB/GPU. OOM at N=288 in production.**

### After Fix (shared pool)

| N | Per-GPU delta (MB) | Cumulative total (GB) |
|-----|-------------------:|----------------------:|
| 1 | -468 (first-capture noise) | -0.91 |
| 2 | 54 | -0.81 |
| 4 | 58 | -0.69 |
| 8 | 56 | -0.58 |
| 16 | 76 | -0.44 |
| 32 | 80 | -0.28 |
| 64 | 82 | 0.03 |
| 128 | 104 | 0.40 |
| 256 | 106 | 1.28 |
| 512 | 204 | 3.34 |

**Total: 3.34 GB (both GPUs), ~1.67 GB/GPU. All 20 sizes fit easily.**

### Summary

| Metric | Before | After | Change |
|---|--:|--:|---|
| Total (both GPUs, 20 sizes) | 51.37 GB | 3.34 GB | **-93.5%** |
| Per GPU | 25.7 GB | 1.67 GB | |
| N=512 per GPU | 2858 MB | 204 MB | 14x less |
| N=288 | OOM | 154 MB/GPU | Fixed |

### Correctness Validation

- `test_piecewise_prefill.py`: flat vs piecewise token match **PASS** (0/128).
  Partial offloading (epl=4): **PASS**.
- `test_offload_correctness.py`: demand loading **exact match** (0.000 diff).
  Decode with offloading (10 steps): **exact token match**.
- `test_split_stage4.py`: decode tokens **exact match**. Prefill 4/128
  mismatch is pre-existing torch.compile noise (flat vs piecewise are
  different Inductor compilation units).

### Performance (no regression)

- Prefill: flat 25.785 ms, piecewise 25.868 ms (+0.3%)
- Decode: flat 9.586 ms, piecewise 9.243 ms (-3.6%)

## Phase 4 Memory Budget (single GPU, post-fix)

Graph memory is no longer the bottleneck. With 20 sizes at ~3.3 GB (estimated
single-GPU = 2x PP=2 per-GPU):

- 20 sizes: model(3) + cache(50%=42) + graphs(3.3) + KV = **48.3 GB + KV**
- Fits on 80 GB H100 with ~30 GB for KV cache.

The previous constraint (20 sizes needed ~51 GB for graphs alone) is resolved.

## Notes

**torch.compile has no effect on graph memory.** Measured 51.37 GB with compile
vs 51.38 GB without. The `recompile_limit (8)` warnings are harmless: dynamo
falls back to a generic trace after 8 layer specializations; captured graphs
replay recorded operations regardless.

**Measurement method.** `torch.cuda.mem_get_info()` (free/total GPU memory)
is the correct tool — `memory_allocated()` only tracks PyTorch tensors, missing
CUDA driver allocations. The OOM consistency check (N=288 at exactly the
predicted threshold) confirmed the measurements are directionally correct.

## Future Optimizations (if command buffer overhead matters)

The remaining ~1.7 GB/GPU is dominated by command buffers (~3.1 MB x 960
graphs/GPU for PP=2). These are only worth pursuing if the budget is still
tight after pool sharing.

**1. Stage4b dedup in offloading mode** (easy, zero runtime cost): In
offloading mode, all 32 per-layer stage4b graphs are identical (read from same
`w1_buf`/`w2_buf`/`expert_map_buf`). Capture one, replay 32x. Saves
~96 MB/size. Stage1/4a are NOT deduplicable (per-layer weight tensors).

**2. Fewer graph sizes** (easy, small padding cost): Each eliminated size
saves ~300 MB (single GPU). Going from 20 to 10 sizes saves ~3 GB. Less
urgent now that pool sharing freed ~22 GB.

**3. Graph-only-stage4b** (moderate, needs benchmarking): Run stage1/4a
eagerly with torch.compile, graph only stage4b. Saves ~6 GB in command
buffers but adds 2-5 ms dispatch overhead per forward pass.

## Files

- `profiling/measure_graph_memory.py` — measurement script
- `profiling/graph_memory_results.json` — raw per-size data (before fix)
- `profiling/graph_memory_results_pool.json` — raw per-size data (after fix)
- `moe_engine.py:capture_mixed_cuda_graphs()` — piecewise capture with pool sharing
- `moe_engine.py:_create_intermediate_buffers()` — per-size buffer allocation
- `trace_construction/collect_batched_traces.py:compute_replay_kv_budget()` — budget function (overhead_gb can now be reduced)
