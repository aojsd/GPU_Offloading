# Decode Pipeline — Implementation & Profiling

## Status: Complete (Phase 1)

Custom engine decode matches or beats vLLM at all sequence lengths (batch=1).

---

## Architecture (per layer)

```
Fused QKV → Q/K RMSNorm → rope_pytorch → reshape_and_cache_flash →
FlashInfer BatchDecode → output projection → RMSNorm + residual →
router → fused_experts → residual
```

- `torch.compile(fullgraph=False)` fuses RMSNorm + residual + RoPE into Triton kernels
- Graph breaks at `reshape_and_cache_flash` and FlashInfer `run()` are acceptable — Inductor fuses the regions between breaks
- `fused_experts` has `torch.library` registration with fake impl → no graph break
- `static_slot_mapping` updated before each graph replay via `.copy_()`

## FlashInfer Decode Attention

Two-phase API: `plan()` (CPU, outside graph) → `run()` (GPU, inside graph).

- **`plan()` before every replay**: recomputes `_plan_info` containing split-KV tile counts that depend on actual page count — buffer copy alone is not sufficient
- **`_seq_lens_cpu` incremented BEFORE `plan()`**: FlashInfer must include the current token's K/V, which is written to cache before attention within each layer
- **`_seq_lens_cpu` mirror**: CPU-side copy of GPU `seq_lens` avoids GPU→CPU sync for plan metadata. Must be kept in sync — any code setting `seq_lens` must also set `_seq_lens_cpu`
- KV cache format: flat NHD `[L, total_pages, page_size, num_kv_heads, head_dim]`
- KV write: `reshape_and_cache_flash` (flat NHD, same as vLLM)
- RoPE: `rope_pytorch` for decode (pure PyTorch, compilable)

---

## Performance vs vLLM (H100 80GB, batch=1, CUDA graphs + torch.compile)

Profiled with `nsys profile --cuda-graph-trace=node`. 30 decode steps per run.

```
seq_len  Custom  vLLM   Gap      Gap%
  128     2.87   2.93  -0.06     -1.9%
  256     2.87   2.93  -0.06     -2.0%
  512     2.89   2.93  -0.05     -1.7%
 1024     2.91   2.95  -0.04     -1.4%
 2048     2.96   3.00  -0.04     -1.3%
```

Custom engine beats vLLM at all sequence lengths. Fewer kernel launches (989 vs 1032) due to more aggressive Inductor fusion.

### Kernel Category Breakdown (seq128, per decode step)

| Category | Custom | vLLM | Delta |
|----------|--------|------|-------|
| MoE (fused_experts) | 731us (25.4%) | 698us (23.8%) | +33us |
| Elementwise | 643us (22.4%) | 744us (25.4%) | **-101us** |
| Linear (cuBLAS) | 436us (15.2%) | 420us (14.3%) | +16us |
| Attention (FlashInfer) | 90us (3.1%) | 0us | +90us |
| Attention (FlashInfer graph) | 0us | 199us (6.8%) | **-199us** |
| KV cache store | 90us (3.1%) | 37us (1.3%) | +53us |
| torch.compile fused | 131us (4.6%) | 126us (4.3%) | +5us |

Key differences:
- **Attention: -109us** — Non-graph FlashInfer (90us) vs vLLM's graph FlashInfer (199us). Our direct `BatchDecodeWithPagedKVCacheKernel` call avoids the `device_kernel` graph wrapper overhead.
- **Elementwise: -101us** — More aggressive Inductor fusion in our single-graph approach.
- **KV store: +53us** — `reshape_and_cache_flash` overhead (our Triton fused store vs vLLM's native kernel).

### Profiling Data

Detailed per-seq-len kernel profiles:
- `profiling/OLMoE-1B-7B/decode_seq128.prof`
- `profiling/OLMoE-1B-7B/decode_seq256.prof`
- `profiling/OLMoE-1B-7B/decode_seq512.prof`
- `profiling/OLMoE-1B-7B/decode_seq1024.prof`
- `profiling/OLMoE-1B-7B/decode_seq2048.prof`

Regenerate: `python src/MoE/vLLM_comparison/nsys_profiler.py all`

---

## Key Challenges & Pitfalls (Decode-Specific)

- **`plan()` must be called before EVERY graph replay** — `_plan_info` tile counts depend on page count, not just buffer contents. Tried `.copy_()` on plan buffers; doesn't work.
- **`_seq_lens_cpu` must be incremented BEFORE `plan()`** — FlashInfer needs to include the current token's K/V in its tile calculation. Off-by-one here causes silent wrong attention.
- **`_seq_lens_cpu` mirror must stay in sync** — Any code touching `seq_lens` (GPU) must also update `_seq_lens_cpu` (CPU). Forgetting this causes plan metadata mismatch.
- **`reshape_and_cache_flash` lacks fake impl → graph break** — FlashInfer `run()` and RoPE DO have registrations and are NOT graph-break sources.
- **`slot_mapping` must be computed outside CUDA graph** — Then copied to static buffer before replay.

---

## CUDA Graph Infrastructure

- Single CUDA graph captures entire decode step (all 16 layers)
- `plan()` runs on CPU before each replay (not captured in graph)
- `run()` executes inside graph (GPU-only, static memory addresses)
- Static buffers: `static_token_ids`, `static_positions`, `static_slot_mapping`, `static_output`
- `PersistentVariableLengthMergeStatesKernel` in "Other" category (FlashInfer internal)
