# Prefill Pipeline — Implementation & Profiling

## Status: Complete (Phase 1), Faster Than vLLM

Custom engine prefill uses vLLM FA3 + piecewise CUDA graphs. **2-32% faster than vLLM**
across all sequence lengths (without torch.compile — compile requires GCC 13 on compute nodes).

---

## Architecture

Prefill uses vLLM FA3 (`flash_attn_varlen_func`) for attention, replacing FlashInfer's
`BatchPrefillWithRaggedKVCacheWrapper`. FA3 is stateless (no plan/run two-phase API) —
just `cu_seqlens` tensors constant per padded size. Multiple CUDA graphs work trivially.

```python
from vllm.vllm_flash_attn import flash_attn_varlen_func
out = flash_attn_varlen_func(
    q, k, v,  # [B*S, num_heads/kv_heads, head_dim]
    cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,  # [B+1], static per padded size
    max_seqlen_q=S, max_seqlen_k=S, causal=True, fa_version=3)
```

- FlashInfer RoPE (`apply_rope_with_cos_sin_cache_inplace`) for prefill
- `reshape_and_cache_flash` for KV cache writes
- `fused_experts` for MoE compute
- Piecewise CUDA graphs at [128, 256, 512, 1024, 2048], auto-pad to nearest captured size

---

## Wall-Clock Performance (verify_prefix_caching_fix.py)

**Prefill only, CUDA graph without torch.compile (median ms, batch=1):**
```
seq_len  Custom   vLLM      Delta       Ratio
  128     7.65ms   7.86ms   -0.21ms     0.97x (custom 3% faster)
  256     8.32ms   8.52ms   -0.21ms     0.98x (custom 2% faster)
  512     9.25ms   9.52ms   -0.27ms     0.97x (custom 3% faster)
 1024    11.61ms  17.20ms   -5.59ms     0.68x (custom 32% faster)
 2048    16.65ms  18.57ms   -1.92ms     0.90x (custom 10% faster)
```

vLLM is measured with `enable_prefix_caching=False` and unique random prompts per trial
to ensure full prefill computation (no KV cache reuse from warmup).

CUDA graphs reduced eager prefill by ~40% at short sequences.

**Combined prefill (compile+graph) + decode (graph):**
```
prefill=128 + 50x decode:  prefill=7.13ms  decode=163.22ms (3.26ms/step)  total=170.34ms
prefill=512 + 50x decode:  prefill=8.82ms  decode=164.17ms (3.28ms/step)  total=173.00ms
prefill=2048 + 50x decode: prefill=16.27ms decode=169.25ms (3.39ms/step)  total=185.52ms
```

---

## Prefix Caching Bug in Previous Benchmarks (Resolved)

### Problem

Previous measurements showed vLLM 60-246% faster at prefill. This was **wrong** — caused
by vLLM V1's `enable_prefix_caching=True` (default) silently reusing KV cache from
warmup runs that used the same prompt tokens.

### Root Cause

The benchmark scripts (`microbenchmark.py prefill`, `nsys_profiler.py`) reused
`prompt_ids = list(range(100, 100 + seq_len))` for both warmup and timed trials. With
prefix caching enabled, vLLM V1's scheduler detected that most tokens were already in the
KV cache and only scheduled the remaining ~16 new tokens (out of 128) per profiled run.

**Evidence** (from `diagnose_scheduler.py`):
```
Warmup request:  total_num_scheduled_tokens=128  # first time, full prefill
Test request:    total_num_scheduled_tokens=16   # prefix cache hit, only 16 new tokens!
```

This made vLLM's `fused_moe_kernel` calls process M=16 tokens (with an optimized small-M
config: BSM=16, BSN=32) instead of M=128 (BSM=32, BSN=128) — appearing 3x faster per
kernel launch. In reality, it was just doing 8x less work.

### Fix

1. Set `enable_prefix_caching=False` in vLLM `LLM()` constructor
2. Use unique random prompts (`torch.randint(1, 1000, (seq_len,)).tolist()`) per trial

Both fixes applied to `microbenchmark.py prefill` and `nsys_profiler.py`.

### Previous (buggy) vs Corrected Numbers

| seq_len | Custom | vLLM (buggy) | vLLM (fair) | Old ratio | True ratio |
|---------|--------|--------------|-------------|-----------|------------|
| 128 | 7.37ms | 4.60ms | 7.86ms | 1.60x slower | **0.97x faster** |
| 1024 | 11.31ms | 5.55ms | 17.20ms | 2.04x slower | **0.68x faster** |
| 2048 | 16.24ms | 5.70ms | 18.57ms | 2.85x slower | **0.90x faster** |

vLLM's near-flat 4.6-5.7ms across all seq_lens was the tell — real prefill compute
scales superlinearly with sequence length (attention is O(S²), MoE/projections are O(S)),
but prefix-cached "prefill" only processes a fixed ~16 new tokens regardless of prompt length.

### How Prefix Caching Works (Detailed Mechanism)

vLLM V1's paged KV cache divides stored K/V tensors into fixed-size blocks (16 tokens
per block). With `enable_prefix_caching=True`, each block is **indexed by a hash of its
token content**. When a new request arrives, the scheduler checks whether KV cache blocks
for the prompt prefix already exist.

**The benchmark loop executed:**
```
warmup_1: prompt = [100, 101, ..., 227]  → 128 tokens, full prefill (no cache)
warmup_2: prompt = [100, 101, ..., 227]  → prefix cache hit
warmup_3: prompt = [100, 101, ..., 227]  → prefix cache hit
trial_1:  prompt = [100, 101, ..., 227]  → prefix cache hit  ← TIMED
trial_2:  prompt = [100, 101, ..., 227]  → prefix cache hit  ← TIMED
```

After warmup_1 completes, the KV cache holds 8 blocks (128 / 16 = 8) with pre-computed
K and V tensors. These persist (plenty of KV cache memory — 495K tokens). When warmup_2
arrives with the same tokens, the scheduler hashes the prompt in block-sized chunks, finds
7 of 8 blocks already cached (the last block isn't committed until the sequence finishes),
sets `num_computed_tokens = 112`, and only schedules `128 - 112 = 16` new tokens.

**Impact on each operation per layer:**

| Operation | Full prefill (128 tok) | Prefix-cached (16 tok) | Reduction |
|-----------|------------------------|------------------------|-----------|
| Embedding | 128 lookups | 16 lookups | 8x |
| RMSNorm + RoPE | 128 tokens | 16 tokens | 8x |
| QKV projection | [128, 2048] matmul | [16, 2048] matmul | 8x |
| Attention (Q×K) | Q=128, KV=128 | Q=16, KV=128 | 8x fewer Q rows |
| MoE routing | 128 → 1024 expert slots | 16 → 128 expert slots | 8x |
| `fused_moe_kernel` | M=128, BSM=32, BSN=128 | M=16, BSM=16, BSN=32 | 8x less compute |
| Output projection | [128, 2048] matmul | [16, 2048] matmul | 8x |

The MoE kernel config also changed: M=16 selects a different pre-tuned Triton config
from `E=64,N=1024,device_name=NVIDIA_H100_80GB_HBM3.json` (BSM=16, BSN=32, stages=5)
vs M=128 (BSM=32, BSN=128, stages=3). The smaller config processes fewer thread blocks,
further amplifying the apparent speedup beyond the 8x work reduction.

---

## nsys Kernel Profiling

Prefill kernel profiles need to be regenerated with the corrected vLLM driver
(prefix caching disabled, unique prompts). Previous `.prof` files are invalid.

Regenerate: `python src/MoE/vLLM_comparison/nsys_profiler.py prefill`

---

## Key Challenges & Pitfalls (Prefill-Specific)

### FlashInfer Prefill Incompatible with Multi-CUDA-Graph

FlashInfer's `fmha_varlen_plan()` (cutlass/FA3 backend on H100) allocates 4 fresh GPU
tensors every `plan()` call. These addresses get baked into the CUDA graph. When a new
graph is captured, `plan()` allocates at new addresses and old tensors get GC'd — replaying
earlier graphs reads freed memory. Tried shared wrappers, plan_info preservation, shared
pools — none work because the API fundamentally reallocates on every `plan()`.

**Solution**: Replace with vLLM FA3 (`flash_attn_varlen_func`) which is stateless.

### Critical CUDA Graph Bug: Static Buffer GC

All tensors captured by a CUDA graph (inputs, intermediates, outputs) MUST be kept alive
for the graph's lifetime. In a capture loop, local variables from earlier iterations get
GC'd by Python's reference counting when reassigned. Fix: save ALL static buffers in the
graph dict (not just `static_input_ids` and `static_output`):

```python
self._prefill_cuda_graphs[(batch_size, S)] = {
    'graph': graph,
    'static_input_ids': static_input_ids,
    'static_output': static_output,
    'static_positions': static_positions,      # MUST keep alive
    'static_slot_mapping': static_slot_mapping, # MUST keep alive
    'static_cu_seqlens': static_cu_seqlens,     # MUST keep alive
}
```

Symptom: 1-2 graphs work fine, 3+ graphs crash with "CUDA error: illegal memory access"
on replay of earlier graphs. Immediate replay after each capture works (freed memory not
yet reused), but replay after all captures fails.

### torch.compile Alone Hurts Prefill

torch.compile **without** CUDA graphs made prefill worse (18.2ms vs 12.6ms eager at seq128).
Even 2 graph breaks per layer x 16 layers = 32 enter/exit transitions with per-call
guard-check overhead exceeded the fusion savings. Fix: **compile + CUDA graph together**.

### vLLM V1 Prefix Caching Silently Inflates Benchmarks

When benchmarking vLLM V1's prefill, **always** either:
1. Set `enable_prefix_caching=False` in `LLM()` constructor, OR
2. Use unique random prompts for every request (warmup AND timed)

Otherwise, repeated prompts cause the scheduler to skip computing already-cached tokens,
making prefill appear faster than it is. The scheduler reports `num_scheduled_tokens` in
`scheduler_output` — monkey-patch `GPUModelRunner.execute_model` to verify.

---

## Prefix Caching Bug — Diagnostic Method

The bug was diagnosed by monkey-patching four vLLM internals: (1) `invoke_fused_moe_kernel`
to log kernel M values and Triton configs, (2) `fused_experts` to log hidden_states shapes,
(3) `GPUModelRunner.execute_model` to log `total_num_scheduled_tokens` from the scheduler.
The smoking gun was the scheduler logging 16 scheduled tokens for a 128-token prompt after
warmup with the same prompt had populated the prefix cache. Scripts have been removed;
the fix is in `microbenchmark.py prefill` and `nsys_profiler.py`.
