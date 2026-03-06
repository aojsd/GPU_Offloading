# Pipeline Parallelism Performance Analysis

## Architecture

Mixtral-8x7B-32L split across 2x H100 80GB GPUs:
- GPU 0: layers 0-15, embedding, block tables, FlashInfer wrapper
- GPU 1: layers 16-31, final RMSNorm, lm_head
- PP boundary at layer 16: `hidden_buf.copy_()` cross-GPU transfer (8 KB for decode BS=1)
- Each GPU has its own: KV cache, cos_sin_cache, block_table, seq_lens, FlashInfer wrapper

Per-layer piecewise CUDA graph stages:
- Stage 1: RMSNorm + QKV projection + RoPE (CUDA graph replay)
- Stage 2: FlashInfer decode attention (eager, needs `plan()` per step)
- Stage 3: FA3 prefill attention (eager, prefill-only)
- Stage 4a: Post-attention norm + router (CUDA graph replay)
- CPU break: trace recording / offload engine / replay controller
- Stage 4b: MoE FFN via fused_experts (CUDA graph replay)

## Decode Performance Summary

All measurements: BS=1 decode, 20 steps, median wall-clock.

| Config | Wall-clock | Per-layer | Setup | Notes |
|--------|-----------|-----------|-------|-------|
| 20L single, no compile | **9.74 ms** | **0.487 ms** | 0.258 ms | Baseline |
| 20L single, compile | **9.14 ms** | **0.457 ms** | 0.256 ms | 6.2% faster |
| 32L PP=2, no compile | **15.56 ms** | **0.486 ms** | 0.496 ms | Matches baseline per-layer |
| 32L PP=2, compile | **15.15 ms** | **0.473 ms** | 0.499 ms | Matches compile baseline |

### Key Finding

**Per-layer compute in PP mode matches single-GPU baseline exactly.**
No Python overhead penalty from the PP code path.

| N (tokens) | 20L single (ms/L) | PP=2 (ms/L) | PP / 20L |
|-----------|-------------------|-------------|---------|
| 1 (decode) | 0.480 | 0.481 | 1.00x |
| 128 (prefill) | 1.304 | 1.303 | 1.00x |
| 256 (prefill) | 1.465 | 1.462 | 1.00x |
| 512 (prefill) | 1.506 | 1.511 | 1.00x |

The +3.5% with compile is because torch.compile only compiles layers 0-7 before
hitting the Dynamo recompile limit (8 specializations for `self.k_cache[layer]`).
In 20L this means 8/20 = 40% of layers get compiled; in 32L PP, only 8/32 = 25%.
GPU 1 layers (16-31) get no compile benefit at all since they fall back entirely.

## Per-Layer Phase Breakdown

### Without torch.compile

| Phase | 20L single | PP GPU 0 | PP GPU 1 |
|-------|-----------|----------|----------|
| Stage 1 (RMSNorm+QKV+RoPE) | 0.082 ms | 0.083 ms | 0.079 ms |
| Stage 2 (FlashInfer decode) | 0.033 ms | 0.034 ms | 0.030 ms |
| Stage 4a (post-attn+router) | 0.061 ms | 0.062 ms | 0.058 ms |
| Stage 4b (MoE FFN) | 0.335 ms | 0.336 ms | 0.333 ms |
| **Layer total** | **0.511 ms** | **0.515 ms** | **0.499 ms** |

GPU 1 is ~3% faster per-layer — likely because it has less memory pressure (no
embedding table, no block table overhead on same device).

### With torch.compile

Compile benefit concentrated in Stage 1 and Stage 4a (Inductor fuses RMSNorm +
residual + linear ops). Stage 4b (MoE) is already a single Triton kernel, no benefit.

| Phase | Compiled (L0-7) | Uncompiled (L8+) | Savings |
|-------|-----------------|------------------|---------|
| Stage 1 | 0.042 ms | 0.084 ms | 50% |
| Stage 4a | 0.050 ms | 0.062 ms | 19% |
| Stage 4b | 0.336 ms | 0.337 ms | 0% |
| **Layer** | **0.464 ms** | **0.519 ms** | 10.6% |

Dynamo recompile limit (8) causes layers 8+ to fall back to eager kernels.
This affects PP disproportionately since 32L has more layers beyond the limit.

## Overhead Analysis

| Component | PP=2 no-compile | 20L single | Delta |
|-----------|----------------|------------|-------|
| Setup (metadata copy + FlashInfer plan) | 0.496 ms | 0.258 ms | +0.238 ms |
| PP boundary transfer (8 KB NVLink) | 0.024 ms | N/A | +0.024 ms |
| Final (norm + lm_head + logits xfer) | 0.166 ms | 0.142 ms | +0.024 ms |
| **Total overhead** | | | **+0.286 ms** |

Setup doubles because we replicate token_ids, positions, slot_mapping to both GPUs
and call `_plan_flashinfer_decode_pp()` twice (once per GPU). The PP transfer and
logits copy are negligible (8 KB + vocab tensor over NVLink).

## Note on Phase-Timed vs Wall-Clock

Phase-timed totals are ~1ms higher than wall-clock because each measurement
inserts a `_sync_all()` barrier. In normal execution, GPU kernels overlap across
the CPU-issued launch calls — there is no synchronization between stages within
a layer. The wall-clock number is the true performance metric.

## Optimization Log

### 2026-03-05: Graph padding bug explains earlier 29ms result

The prior session measured PP=2 decode at 28.93ms. Root cause: `test_pipeline_parallel.py`
captured graphs at `total_token_sizes=[128, 256, 512]` but not `[1]`. A BS=1 decode
dispatched to the N=128 graph, replaying all 96 CUDA graphs with 128-token buffers
instead of 1 — 128x wasted compute per kernel.

| Graph size | PP=2 decode (BS=1) |
|------------|-------------------|
| N=128 (wrong) | **29.20 ms** |
| N=1 (correct) | **15.56 ms** |

Fix: always include `1` in `total_token_sizes` when capturing piecewise graphs for
decode workloads.

### 2026-03-05: Baselines established

Four configurations measured with sync-barrier phase timing:
- PP per-layer compute matches single-GPU baseline (within noise)
- PP boundary transfer is negligible (0.024 ms)
- Setup overhead doubles due to per-GPU metadata replication (+0.238 ms)
- torch.compile gives 6% improvement but Dynamo recompile limit means only
  first 8 layers benefit; PP gets less benefit than 20L (25% vs 40% compiled)

### nsys profiles collected

Profiles saved in `profiles/` directory:
- `pp2_nocompile.nsys-rep`: 32L PP=2, no compile, 5 steps
- `20L_nocompile.nsys-rep`: 20L single-GPU, no compile, 5 steps

**nsys NVTX per-stage times (ns, 5 steps × 32 or 20 layers):**

| NVTX Range | 20L avg | 20L med | PP=2 avg | PP=2 med | PP/20L (avg) |
|------------|---------|---------|----------|----------|-------------|
| stage4b (per-layer) | 71,812 | 68,827 | 70,741 | 67,371 | 0.99x |
| stage1 (per-layer) | 39,961 | 37,315 | 39,957 | 37,826 | 1.00x |
| stage2 (per-layer) | 32,187 | 30,000 | 30,792 | 28,792 | 0.96x |
| stage4a (per-layer) | 25,667 | 24,418 | 25,343 | 23,940 | 0.99x |
| setup (per-step) | 517,435 | 478,359 | 864,299 | 804,113 | 1.67x |
| cpu_break (per-layer) | 458 | 447 | 513 | 486 | 1.12x |
| embed (per-step) | 28,569 | 27,005 | 31,873 | 29,567 | 1.12x |
| final (per-step) | 67,170 | 63,254 | 117,094 | 112,523 | 1.74x |
| pp_xfer (per-step) | N/A | N/A | 41,963 | 41,402 | — |

**Key kernel: `fused_moe_kernel`** accounts for 52.4% of GPU time (med 118.9 us).
FlashInfer decode: 1.5% (med 9.4 us). cuBLAS GEMMs (QKV/O proj): ~10%.
CPU break (Python overhead between stage4a and stage4b): **<0.5 us/layer** — negligible.

Note: avg > med for all stages because layer 0 and step 0 have warm-up effects
(first invocation ~1.5x slower). The median is more representative of steady-state.

### Optimization targets (prioritized)

1. **Fix Dynamo recompile limit**: If all 32 layers could be compiled, PP=2 would
   drop from 15.15 ms to ~14.5 ms (estimate: 32L × 0.464 ms/L + overhead).
   Options: increase `config.recompile_limit`, or make per-layer weights indexable
   without specialization (e.g., pass layer weights as arguments not self.w1[layer]).
2. **Reduce setup overhead**: 0.496 ms → target 0.258 ms. Only copy metadata to
   the GPU that needs it first; replicate lazily or at PP boundary.
3. **Profile shows no significant inter-kernel gaps** — the piecewise graph replay
   achieves near-full GPU utilization within each stage.
