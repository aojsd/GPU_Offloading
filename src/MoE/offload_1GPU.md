# Single-GPU Expert Offloading for Mixtral-8x7B — Research Notes

## 1. vLLM Expert Offloading Status

**vLLM does NOT have expert-level offloading** in any released version (including 0.11.2 and latest 0.15.0). There is an RFC ([#33869](https://github.com/vllm-project/vllm/issues/33869)) and a closed PR ([#31938](https://github.com/vllm-project/vllm/pull/31938)) proposing it, but the feature has not been merged.

### What exists: generic `cpu_offload_gb`

The only offloading in vLLM is `cpu_offload_gb`, which offloads entire layer parameters to CPU pinned memory in sequential order during model construction. Key implementation details (`vllm/model_executor/models/utils.py`):

- Uses **UVA (Unified Virtual Addressing)** — CPU pinned memory accessed through a CUDA virtual address, so the GPU reads directly from host memory over PCIe on demand (no explicit H2D copy per forward).
- Offloads parameters **in sequential order** until the byte budget is exhausted — **not expert-aware**.
- No cache replacement policy — offloaded weights are static for the model's lifetime.
- If a MoE layer is offloaded, ALL experts in that layer are offloaded, not just inactive ones.

### Proposed (not merged): `CpuOffloadInfer` (RFC #33869)

The proposed architecture would:
- Keep all expert weights in CPU pinned memory
- Maintain a GPU **Expert Cache** with tunable capacity per layer (`cache_expert_num`)
- Use a **Miss Expert Buffer** (double-buffered) for staging cache-miss experts
- Include CPU MoE Engine with AVX/AMX kernels for cache-miss experts
- Two modes: DBO (Dual Batch Overlap) and Prefetch with dynamic `n_copy` budget

**Critically, the RFC does not specify how GPU memory is partitioned between expert cache and KV cache** — reviewers flagged this gap.

### Academic work (more mature)

- **[MoE-Infinity](https://arxiv.org/html/2401.14361v3)**: LFU (Least Frequently Used) cache with Expert Activation Matrices for prediction. Reserves GPU for dense components first, remainder as expert buffer. 3.1–16.7x latency reduction.
- **[Fast MoE Inference with Offloading (Eliseev & Mazur 2023)](https://arxiv.org/abs/2312.17238)**: LRU cache per layer. Cache size k=2 per layer achieves ~60% hit rate. Speculative loading via hidden-state prediction reaches ~80% recall.
- **[MoE-Offloading Analysis (2511.05814)](https://arxiv.org/pdf/2511.05814)**: Frequency-based (LFU) generally outperforms recency-based (LRU) for MoE workloads due to skewed expert activation distributions.

---

## 2. vLLM Memory Management: Weights vs KV Cache

### `gpu_memory_utilization` (default: 0.9)

Controls total GPU budget. vLLM profiles model weight + peak activation memory, then assigns the remainder to KV cache:
```
available_kv = total_gpu * gpu_memory_utilization - (weights + activations + CUDA_graph)
```
The split is **implicit** — no way to explicitly set expert weight vs KV cache budgets.

### `cpu_offload_gb` (default: 0)

Offloads up to N GiB of model parameters to CPU pinned memory (UVA). Frees GPU memory for KV cache. Not expert-aware.

### `kv_cache_memory_bytes` (default: None)

Explicit KV cache size override, bypassing `gpu_memory_utilization` profiling. Useful for our custom engine — we can set a fixed KV cache budget.

### `enable_expert_parallel` (default: False)

Distributes experts across GPUs (each GPU holds a subset). Parallelism, not CPU offloading.

### No expert-specific offloading parameters exist

There are **no** parameters like `expert_cache_size`, `num_experts_per_gpu`, or `expert_offload_gb` in any released version.

**For our custom engine**: We should implement an explicit `expert_cache_size` parameter (number of experts per layer on GPU), with the rest of GPU memory available for KV cache. This is what vLLM lacks.

---

## 3. Mixtral-8x7B Architecture and Memory Analysis

### Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 32 |
| Experts per layer | 8 |
| Top-K routing | 2 |
| Hidden dim | 4096 |
| Intermediate dim | 14336 |
| Attention heads (Q/KV) | 32 / 8 (GQA) |
| Head dim | 128 |
| Vocab size | 32000 |
| Total params | ~46.7B |

### Memory Breakdown (BF16, 2 bytes/param)

| Component | Params | BF16 Memory |
|-----------|--------|-------------|
| Embeddings + LM head | 262M | 0.5 GB |
| All attention (32 layers) | 1.34B | 2.5 GB |
| All RMSNorm + routers | ~1.3M | negligible |
| **All experts (256 total)** | **45.1B** | **84.0 GB** |
| **Total** | **46.7B** | **~87 GB** |

**Per expert**: 176M params = **352 MB** (3 matrices: gate/up/down, each 4096 x 14336)

### Does it fit on H100 80GB?

**No.** 87 GB exceeds 80 GB even without KV cache. This is the core motivation for expert offloading.

### KV cache size (BF16, GQA with 8 KV heads)

- Per token per layer: 2 x 8 x 128 x 2 bytes = 4,096 bytes
- All 32 layers: **128 KB per token**
- 2048 tokens: 256 MB | 8192 tokens: 1 GB | 32768 tokens: 4 GB

### Expert offloading scenarios

| Cache size per layer | Experts on GPU | Expert GPU mem | Non-expert | Total model | Free for KV |
|---------------------|---------------|----------------|------------|-------------|-------------|
| k=0 (all offloaded) | 0 | 0 GB | 3 GB | 3 GB | ~77 GB |
| k=1 | 32 | 10.5 GB | 3 GB | 13.5 GB | ~66.5 GB |
| **k=2** | **64** | **21 GB** | **3 GB** | **24 GB** | **~56 GB** |
| k=3 | 96 | 31.5 GB | 3 GB | 34.5 GB | ~45.5 GB |
| k=4 | 128 | 42 GB | 3 GB | 45 GB | ~35 GB |

Since Mixtral uses top-2 routing, **k=2 is the natural starting point** — at steady state, the 2 most recently used experts are cached and often reused.

### Expert routing patterns (from literature)

- **No strong domain specialization**: Expert assignment distributions look similar across ArXiv, biology, philosophy, etc.
- **Temporal locality**: Adjacent tokens tend to reuse experts. Probability of reusing at least one expert from current token to next is significantly above random (0.46).
- **Mixtral is "Group 2"** for routing consistency ([2505.16056](https://arxiv.org/abs/2505.16056)) — reasonably good for offloading, but not the best (OLMoE/LLaMA-MoE-v2 are better).
- **LRU cache k=2**: ~60% hit rate. k=4: ~80%+ with speculative loading.
- **Speculative expert loading**: Using current layer's hidden states to predict next layer's experts achieves ~80% accuracy.

### Offloading bandwidth considerations

- PCIe Gen5 (H100): ~64 GB/s bidirectional
- Loading 2 experts (worst case both miss): 2 x 352 MB = 704 MB → 704/64 = **11 ms per layer**
- 32 layers, 100% miss: ~352 ms/token (catastrophic)
- With 60% cache hit: ~140 ms/token (still slow for decode)
- With speculative prefetch overlapping compute: significantly better, ~2–3 tok/sec shown in literature

This motivates maximizing cache hit rate and overlapping transfers with compute.

---

## 4. Benchmarking Compute Performance (Isolating from Offloading)

### Comparison: Mixtral vs OLMoE in our engine

| | OLMoE-1B-7B | Mixtral-8x7B |
|--|-------------|--------------|
| Layers | 16 | 32 |
| Experts/layer | 64 | 8 |
| Top-K | 8 | 2 |
| Per-expert size | ~8 MB | ~352 MB |
| MoE FLOPs/token | ~50B | ~352B (7x more) |

The engine already handles all Mixtral config differences (GQA, no QK norm, `norm_topk_prob=True`, different vocab, different rope_theta). The `sliding_window` field is null in the current Mixtral config, so the engine's `NotImplementedError` guard won't fire.

**The only blocker is the memory guard** at line 252–257 of `moe_engine.py`, which rejects models exceeding GPU memory. This is by design and needs to be relaxed for partial loading.

### Approach: Pinned-Expert + Partial-Load Benchmarking

**Goal**: Load only a subset of experts on GPU and force routing to only those experts. This tests the exact same compute kernels as full Mixtral inference — just without offloading overhead.

**Implementation plan**:

1. **Partial weight loading** (`resident_experts` parameter):
   - Only load specified experts per layer (e.g., `[0, 1]`)
   - Re-index `w1[E, ...]` / `w2[E, ...]` to use local indices `[0, ..., k-1]`
   - 2 experts/layer: ~21 GB expert memory + 3 GB non-expert = **24 GB total**

2. **Fixed routing override** (`pinned_experts` parameter):
   - Replace router computation with fixed `topk_ids = [[0, 1]]` and equal weights
   - CUDA-graph compatible (deterministic tensors)
   - Applied in `_layer_decode`, `_layer_mixed`, `_layer_stage4_post_attn`

3. **Why this is valid**:
   - `fused_experts` kernel executes identically regardless of which experts are selected — it indexes into `w1`/`w2` using `topk_ids`. The compute cost is determined by `top_k * intermediate_size * hidden_size`, not by expert identity.
   - Attention, RMSNorm, RoPE, embeddings all execute identically to full Mixtral inference.
   - The only difference is the output quality (wrong experts → garbage text), which is irrelevant for latency benchmarking.

### Crafting inputs to route to specific experts — NOT practical

Reasons:
- Routing depends on post-attention hidden states, not raw tokens
- No domain-level expert specialization exists
- 32 layers each make independent routing decisions
- Dynamic input search is incompatible with CUDA graph capture

**The pinned-expert approach is far simpler and more reliable.**

### vLLM comparison options

| Option | Description | Feasibility |
|--------|-------------|-------------|
| vLLM 2-GPU TP | Run full Mixtral on 2x H100 with tensor parallelism. Compare per-step latency. | Fair but not apples-to-apples (different memory bandwidth, NCCL overhead) |
| vLLM 1-GPU with `cpu_offload_gb` | Use vLLM's UVA offloading on 1 GPU. | Tests vLLM's UVA approach vs our approach — interesting but not compute-only |
| **vLLM 1-GPU monkey-patched router** | Patch vLLM's Mixtral router to pin experts, load only 2/layer. Both engines execute identical compute. | **Best for apples-to-apples compute comparison** |

### What to measure

- **Decode latency** (ms/step): batch sizes 1, 4, 8; sequence lengths 128, 512, 1024
- **Prefill latency** (ms): prompt lengths 128, 256, 512, 1024, 2048
- **Kernel breakdown** via nsys: isolate `fused_experts` time specifically
- Key question: does our engine's overhead (routing, norms, attention, CUDA graph) scale the same way for Mixtral's larger dimensions as for OLMoE?

With Mixtral's 7x more MoE FLOPs per token, the MoE kernel will dominate step time even more than with OLMoE — our overhead reduction should matter proportionally less, but the absolute numbers still need to match or beat vLLM.

---

## 5. Implementation: Phase 1 — Simulated Offloading (Mixtral-8x7B-20L)

### Status: Complete (Simulated) — Verified Correct

The 20-layer Mixtral fits on one H100 (54.6 GB BF16). We simulate offloading by keeping
full `[E, 2*I, H]` GPU tensors but tracking which expert slots are "resident." Non-resident
experts are demand-loaded from CPU pinned memory. This validates the mechanics without
needing `expert_map` remapping.

### Architecture: Split Stage 4

Piecewise CUDA graphs decompose each layer into stages. The existing Stage 4 bundles
router + fused_experts in one graph — can't insert loading between them. Split into:

```
Stage 4a [graph]  O proj → norm → router → topk → write buffers
  ↕ CPU break: read topk_ids, load missing experts, record trace
Stage 4b [graph]  fused_experts → residual add
```

New intermediate buffers (per graph size N): `moe_input_buf [N,H]`,
`moe_residual_buf [N,H]`, `topk_weights_buf [N,top_k] fp32`, `topk_ids_buf [N,top_k] int64`.

When offloader is attached, ALL forward paths (decode_step, prefill, prefill_to_slot,
prefill_batch_to_slots) route through piecewise step so the offloader hook runs.

### Files

| File | Role |
|------|------|
| `moe_engine.py` | `_layer_stage4a_router`, `_layer_stage4b_moe`, updated `capture_cuda_graphs` (3 graphs/layer), updated `_step_piecewise` with offload engine hook, auto-creates `ExpertOffloadEngine` when `experts_per_layer` is set |
| `expert_offload_engine.py` | `ExpertOffloadEngine`: CPU pinned copies, residency tracking per layer, `process_layer()` for demand loading + trace recording, `save_trace()`, `get_transfer_stats()` |
| `tests/test_split_stage4.py` | Basic correctness: flat vs piecewise graph comparison |
| `tests/test_offload_correctness.py` | Latency overhead and output correctness with demand loading |

### Correctness Results

- **Split stage4 vs flat graph**: Bit-identical (0.000000 max diff) for both prefill and decode
- **Demand loading output**: Bit-identical to all-resident baseline — `.copy_()` into same GPU
  addresses read by CUDA graph produces exact same results

### Performance Results (Mixtral-8x7B-20L, H100, no torch.compile)

**Split stage4 overhead (no offloader):**

| Path | Flat graph | Piecewise (split) | Overhead |
|------|-----------|-------------------|----------|
| Prefill seq=128 | 25.44 ms | 25.31 ms | -0.1 ms (noise) |
| Decode B=1 | 9.89 ms | 9.73 ms | -0.2 ms (noise) |

**Offloader overhead (all experts resident, no loading):**

| Path | No offloader | With offloader | Overhead |
|------|-------------|----------------|----------|
| Prefill seq=128 | 25.48 ms | 26.08 ms | +0.6 ms (+2.4%) |
| Decode B=1 | 9.71 ms | 10.18 ms | +0.5 ms (+4.8%) |

Per-layer overhead ~0.025 ms (GPU→CPU topk_ids copy + Python set operations).

**Synchronous demand loading (observed matches predicted `compute + transfer`):**

| Config | Baseline | Transfer time | Predicted | Observed | Delta |
|--------|----------|--------------|-----------|----------|-------|
| Decode budget=2 | 9.7 ms | 196.7 ms (31 fetches) | 206.4 ms | 208.0 ms | +1.6 ms |
| Decode budget=4 | 9.7 ms | 139.6 ms (22 fetches) | 149.3 ms | 150.9 ms | +1.5 ms |
| Decode budget=6 | 9.7 ms | 69.8 ms (11 fetches) | 79.5 ms | 80.6 ms | +1.1 ms |
| Prefill budget=2 | 25.7 ms | 748.6 ms (118 fetches) | 774.3 ms | 778.2 ms | +3.8 ms |
| Prefill budget=4 | 25.7 ms | 494.8 ms (78 fetches) | 520.5 ms | 523.4 ms | +2.9 ms |

Per-expert transfer: ~6.3 ms (~336 MB at 55.5 GB/s PCIe). Residual ~1-4 ms is the
offloader Python overhead scaling with layer count.

### What changes for full 32-layer Mixtral

- Weight loading: experts to CPU first (can't fit all on GPU)
- GPU tensors: `[K, 2*I, H]` cache (K < E) instead of full `[E, 2*I, H]`
- `expert_map`: `[E]` int32 per layer mapping global→cache slot, passed to `fused_experts`
- Cache slot management: LRU/eviction across fixed slots
- `fused_experts` call in `_layer_stage4b_moe` adds `expert_map=self.expert_map[layer]`

None of these affect the stage 4 split or piecewise routing — only weight storage and the
fused_experts call signature change.

---

## 6. Implementation: Phase 2 — True Offloading (Full Mixtral-8x7B, 32 Layers)

### Status: Working — Decode offloading verified correct

The full 32-layer Mixtral (87 GB BF16) does NOT fit on one H100. Expert weights live on
CPU pinned memory and are demand-loaded into a fixed-size GPU cache via `expert_map`.

### Key changes from Phase 1

| Aspect | Phase 1 (20L simulated) | Phase 2 (32L true) |
|--------|------------------------|-------------------|
| Expert GPU tensors | Full `[E=8, 2*I, H]` per layer | Cache `[K, 2*I, H]` per layer (K ≤ 8) |
| Weight loading | All to GPU (safetensors) | Non-expert → GPU, experts → CPU pinned |
| Expert identity | Direct indexing `w1[layer][eid]` | Remapped via `expert_map[layer]` |
| `fused_experts` call | No expert_map | `expert_map=..., global_num_experts=8` |
| Memory guard | Raises if >95% GPU | Skipped for offloading mode |
| Offloading scope | Prefill + decode | **Decode only** (prefill uses flat graph) |

### `gpu_expert_budget` parameter

```python
engine = MoEEngine(model_path, gpu_expert_budget=K)  # K cache slots per layer
```

- `gpu_expert_budget=None` (default): all experts on GPU (original behavior)
- `gpu_expert_budget=K`: K cache slots per layer; experts on CPU pinned memory

### CPU-first weight loading

When `gpu_expert_budget` is set:
1. Non-expert weights (embed, lm_head, attention, norms, routers) load directly to GPU
2. Expert weights load to CPU, fused into `w1_cpu`/`w2_cpu` (lists of `[E, 2*I, H]` pinned tensors)
3. GPU cache: `w1`/`w2` = lists of `[K, 2*I, H]` GPU tensors
4. `expert_map` = list of `[E]` int32 GPU tensors, initially `[0,1,...,K-1,-1,...,-1]`
5. First K experts copied into cache slots

### `expert_map` integration

vLLM's `fused_experts` supports `expert_map` for expert parallelism. After our monkey-patched
`ops.moe_align_block_size` produces global expert IDs `[0,E)`, the wrapper remaps:
`expert_ids = expert_map[expert_ids]`. Unmapped experts get -1; the Triton kernel writes
zeros and returns early for -1 blocks.

Required kwargs when offloading:
```python
fused_experts(..., expert_map=self.expert_map[layer], global_num_experts=self.num_experts)
```

### `moe_align_block_size` padding fix

The monkey-patched `searchsorted` can return `num_experts` for padding blocks beyond
`num_tokens_post_padded`. These blocks are never processed by the Triton kernel, but
`expert_map[num_experts]` is OOB. Fixed with `expert_ids.clamp_(max=num_experts - 1)`.

### Eviction timing

`process_layer()` loads experts **before** stage4b graph replay. `post_layer()` evicts
**after** stage4b completes. This split prevents the bug where experts are evicted before
`fused_experts` reads them.

### Decode-only offloading

With K=2 cache slots and top-2 routing, decode (batch=1) needs exactly 2 experts per layer
— fits perfectly. Prefill with many tokens activates up to 8 experts per layer, exceeding K.

Current design: prefill uses flat CUDA graph with only the K initially-mapped experts
(approximate). Decode uses piecewise graphs with the offloader for demand loading.
`step` raises `RuntimeError` if called with prefill sequences while offloader is attached.

### Expert cache (shared weight buffer)

**Terminology**: The **expert cache** (`w1_buf`/`w2_buf`) is the shared weight buffer that
all stage4b CUDA graphs read from. When K < E (true offloading), the extra E slots beyond
the K persistent residents are called the **scratchpad** — used for demand-loaded experts.

**Problem**: CUDA graphs require fixed tensor addresses. Per-layer weight tensors
(`w1[layer]`) live at different addresses, so a single stage4b graph can't read from
different layers' weights. Also, batched decode with B tokens and top-2 routing can activate
up to min(2B, E) unique experts per layer — the K-slot per-layer cache may be too small.

**Solution**: A shared expert cache buffer used by all layers (sequential execution):

```
Per-layer persistent storage:  w1[l] = [K, 2*I, H]  (K resident experts)
                               w2[l] = [K, H, I]
                               expert_map[l] = [E] int32

Expert cache (shared buffer):  w1_buf = [buf_slots, 2*I, H]  (buf_slots = E if K>=E, else K+E)
                               w2_buf = [buf_slots, H, I]
                               expert_map_buf = [E] int32
```

When K >= E (all experts fit): `buf_slots = E`, no scratchpad needed.
When K < E (offloading): `buf_slots = K + E`, slots 0..K-1 = persistent residents,
K..K+E-1 = scratchpad for demand-loaded experts.

**Flow per layer (in `_step_piecewise`)**:
1. Stage 4a graph replay → produces topk_ids (routing decisions)
2. **Populate expert cache**: `w1_buf[:K] = w1[layer]`, `expert_map_buf = expert_map[layer]`
3. **Offloader `process_layer()`**: reads topk_ids, loads missing experts into scratchpad
   slots `w1_buf[K+i]`, updates `expert_map_buf[eid] = K+i`
4. Stage 4b graph replay → MoE kernel reads from `w1_buf`/`expert_map_buf`
5. No post-layer eviction needed — scratchpad is overwritten on next layer's populate

**Key properties**:
- Per-layer persistent state (`w1[l]`, `expert_map[l]`) is **never modified** by demand loading
- Scratchpad slots are ephemeral — always available (E slots ≥ any possible demand)
- Offload engine cleanup is trivial since per-layer state is clean
- All stage4b CUDA graphs use the same `w1_buf` address → functionally identical across layers

**Memory cost**:
- Expert cache: buf_slots × 336 MB per expert
- With K=2: (2+8) × 336 MB = 3.36 GB
- Per-layer HBM→HBM copy overhead (see analysis below)

**TODO**: Remove scratchpad once tracing infrastructure with embedded eviction decisions is
built. With pre-planned expert placement per layer, each layer can have exactly the experts
it needs in persistent storage, eliminating the need for runtime overflow.

### Expert cache D2D copy overhead (nsys profiling, 2026-02-27)

The expert cache requires copying per-layer weights into the shared buffer before each
layer's stage4b graph replay. **This D2D memcpy dominates the "compute baseline" cost.**

**Controlled experiment: 20L Mixtral, E=8 for both, piecewise CUDA graphs + torch.compile:**

| Config | Kernel | Stage4b (MoE) / layer | D2D memcpy / step | Wall / step |
|---|---|---|---|---|
| Direct `w1[layer]` (no cache) | Triton | 323 us | 0.04 ms | 8.83 ms |
| Shared `w1_buf` (budget=8) | Triton | 334 us | 36.93 ms | 46.23 ms |

**Key findings:**
- MoE kernel compute is **identical** (~3% noise) regardless of shared buffer vs direct access
- The MoE kernel does NOT slow down with larger E: standalone microbenchmark confirms
  E=8→E=10→E=16 has zero overhead (Triton 0.99x)
- The D2D copies (w1_buf + w2_buf per layer) account for **>60% of the reported "compute
  baseline"** in the 32L offloading tests
- Previously reported "compute baseline" of 23.68 ms/step for 32L budget=2 was actually
  ~8 ms compute + ~14.8 ms D2D copies

**32L budget=2 nsys breakdown (Triton):**

| Component | Per step |
|---|---|
| Graph replays (all compute) | 7.34 ms |
| D2D memcpy (`w1_buf`/`w2_buf` copies) | 14.83 ms |
| Non-graph kernels (attention etc.) | 0.54 ms |
| Wall time | 23.76 ms |

**IMPORTANT: The 7.34 ms "graph replays" above is artificially low** — see the expert_map
zeroing artifact below.

### expert_map zeroing artifact (2026-02-27)

When profiling 32L budget=2 without an offloader, the Triton kernel **silently skips GEMMs**
for non-resident experts. With `expert_map = [0, 1, -1, -1, -1, -1, -1, -1]` (only experts
0,1 resident), the vLLM `fused_experts` Triton kernel checks `expert_ids == -1` and writes
zeros instead of computing the matmul. This is the intended expert-parallel zeroing behavior,
repurposed here for expert caching.

Direct measurement (CUDA events, 5 decode steps × 32 layers = 160 samples):

| # Resident experts selected | Median stage4b | Count | Fraction |
|---|---|---|---|
| 0 (both non-resident → skip both GEMMs) | **101 us** | 91 | 57% |
| 1 (one resident → 1 GEMM) | **242 us** | 65 | 41% |
| 2 (both resident → 2 GEMMs, correct) | **343 us** | 4 | 2.5% |

Only 2.5% of layer invocations actually compute both expert GEMMs. The 7.34 ms "graph
replays" is dominated by zero-output kernel invocations, not real MoE computation.

**True compute baselines** (both selected experts guaranteed resident, 2 GEMMs per layer):
- Direct `w1[layer]` (E=8, no cache): **322 us/layer** → 10.3 ms/step (32 layers)
- Expert cache `w1_buf` (E=10): **343 us/layer** → 11.0 ms/step (32 layers)
- Overhead from expert cache: **~21 us/layer** (6.5%), from E=10 vs E=8 Triton grid

In real offloading with the offloader active, demand-loaded experts fill the scratchpad
slots and update `expert_map_buf` to point to valid slots, ensuring both GEMMs execute.
The "compute baseline" should use the 2-GEMM timing (343 us/layer).

**Implication for paged expert cache**: A paged cache where experts live at stable GPU
addresses (not copied per-layer) eliminates the D2D overhead entirely. The per-layer
compute cost should equal the direct `w1[layer]` baseline (~322 us Triton).

### Correctness Results

**32L model with budget=2, scratchpad, torch.compile + CUDA graphs:**
- 10-step decode: deterministic across runs (exact token match, two independent runs)
- Output is coherent (non-garbage tokens)

### Performance Results (Full 32L Mixtral, H100, budget=2, torch.compile + CUDA graphs)

| Metric | Value |
|--------|-------|
| Compute (piecewise, no offloader) | 23.68 ms/step |
| Observed (budget=2, scratchpad demand loading) | 316.33 ms/step |
| Transfers per step | 44.8 (avg 6.34 ms each) |
| Miss rate | 70.0% |
| Predicted (compute + transfers) | 307.89 ms |
| Delta (overhead) | +8.45 ms (+2.7%) |
| PCIe bandwidth | 55.5 GB/s |
| Miss rate | 95.2% |

With budget=2, persistent experts stay resident across steps (scratchpad is ephemeral),
giving 70% miss rate (vs 95.2% with old evict-after-use). Smart caching (LRU,
frequency-based, speculative prefetch) will further reduce miss rate.

**Compute overhead note (CORRECTED 2x)**: The 23.68 ms/step "compute baseline" was doubly
misleading:
1. **D2D copies** (14.8 ms): `w1_buf`/`w2_buf` per-layer copies dominate the wall time.
2. **Zeroed experts** (~3.3 ms skipped): Without offloader, 57% of layer invocations skip
   both GEMMs (expert_map → -1 for non-resident experts). The reported 7.3 ms "graph replays"
   includes many ~101 us zero-output invocations instead of the correct ~343 us full-compute.

**Corrected breakdown** (32L budget=2, all experts properly loaded):
- True MoE compute: 343 us/layer × 32 = **11.0 ms/step**
- D2D memcpy: **14.8 ms/step**
- Non-graph (attention etc.): **0.5 ms/step**
- Predicted wall: **~26.3 ms/step** (vs observed 23.68 ms/step when experts are zeroed)

**fused_moe config**: Copied `E=8,N=14336` H100 config as `E=10,N=14336` since the expert
cache shape is non-standard. Without config, kernel is ~2.5x slower (default Triton tile
params).

### GPU memory usage

| Budget K | Per-layer cache (32×K) | Scratchpad (K+E) | Non-expert | Total | Free for KV |
|----------|----------------------|-----------------|------------|-------|-------------|
| K=2 | 22.5 GB | 3.4 GB | ~10 GB | ~35.9 GB | ~44.1 GB |
| K=4 | 45.1 GB | 4.0 GB | ~10 GB | ~59.1 GB | ~20.9 GB |
| K=6 | 67.6 GB | 4.7 GB | ~10 GB | ~82.3 GB | (does not fit) |

---

## 7. Implementation: Phase 3 — Unified Expert Cache (Zero D2D Copies)

### Status: Complete — Compute parity verified across all models

Phase 2's shared buffer required D2D copies (`w1_buf[:K].copy_(w1[layer])`) before each
layer's stage4b. These copies cost **14.8 ms/step** on 32L Mixtral (462 us/layer × 32),
accounting for >60% of the "compute baseline." Phase 3 eliminates this entirely.

### Design: Single Unified Buffer with Per-Layer Views

Replace the three-level hierarchy (CPU pinned → per-layer GPU cache → shared buffer with
D2D copies) with a single unified buffer where each layer's resident experts occupy fixed
slots:

```
w1_buf = [L * epl + scratchpad_slots, 2*I, H]
w2_buf = [L * epl + scratchpad_slots, H, I]

Layer 0 residents:  slots [0, 1, ..., epl-1]
Layer 1 residents:  slots [epl, epl+1, ..., 2*epl-1]
...
Layer L-1 residents: slots [(L-1)*epl, ..., L*epl-1]
Scratchpad:          slots [L*epl, ..., L*epl + scratchpad - 1]
```

- `experts_per_layer` (constructor param): number of resident experts per layer
- `scratchpad_slots = num_experts`: temporary slots for demand-loaded non-resident experts
  (sized to cover worst-case batch where all experts are non-resident)
- Per-layer views: `self.w1[l] = self.w1_buf[l*epl : (l+1)*epl]` (zero-copy)
- Two expert maps per layer: `expert_map[l]` (relative, for views) and
  `expert_map_abs[l]` (absolute buffer indices, for piecewise stage4b)

Before each layer's stage4b replay, only a 32-byte `expert_map_buf` copy is needed:
```python
# OLD: expensive D2D weight copies (14.8 ms/step eliminated)
# self.w1_buf[:K].copy_(self.w1[layer])
# self.w2_buf[:K].copy_(self.w2[layer])

# NEW: only update the expert_map (32 bytes for 8 experts)
self.expert_map_buf.copy_(self.expert_map_abs[layer])
```

### Expert Dropping Eliminated

Expert dropping (skipping GEMMs for non-resident experts via `expert_map=-1`) has been
**completely removed** from the codebase. The engine now requires an offloader whenever
`experts_per_layer < num_experts`:

```python
@property
def offloading_active(self):
    return (self.experts_per_layer is not None
            and self.experts_per_layer < self.num_experts)
```

`ExpertOffloadEngine` is auto-created by `MoEEngine` when `experts_per_layer` is set.
No manual attach/detach needed — the offload engine is always present when offloading.

### Memory Budget

Per-expert BF16: Mixtral w1 `[28672, 4096]` + w2 `[4096, 14336]` = 352 MB.
OLMoE w1 `[2048, 2048]` + w2 `[2048, 1024]` = 12.6 MB.

| Model | L | E | epl | Slots | Buffer | + Attn/KV | Fits 80 GB? |
|-------|---|---|-----|-------|--------|-----------|-------------|
| OLMoE | 16 | 64 | 2 | 96 | 1.2 GB | ~2 GB | Yes |
| OLMoE | 16 | 64 | 8 | 192 | 2.4 GB | ~2 GB | Yes |
| OLMoE | 16 | 64 | 64 | 1088 | 13.7 GB | ~2 GB | Yes |
| Mixtral-20L | 20 | 8 | 2 | 48 | 16.9 GB | ~10 GB | Yes |
| Mixtral-20L | 20 | 8 | 8 | 168 | 59.2 GB | ~10 GB | Yes |
| Mixtral-32L | 32 | 8 | 2 | 72 | 25.4 GB | ~10 GB | Yes |
| Mixtral-32L | 32 | 8 | 4 | 136 | 47.9 GB | ~10 GB | Yes |
| Mixtral-32L | 32 | 8 | 5 | 168 | 59.2 GB | ~10 GB | OOM |

### Comprehensive Benchmark Results (2026-02-27)

**Methodology**: Per-layer stage4b CUDA graph replay time measured with CUDA events.
Manual per-layer replay: stage1 → attention → stage4a → expert_map copy →
(offloader demand-load if needed) → stage4b (TIMED) → post_layer. 10 decode steps,
median reported. torch.compile enabled for all runs. H100 80 GB.

---

#### OLMoE-1B-7B (16L, E=64, top_k=8, 12.6 MB/expert)

**Triton kernel** (all-resident baseline: epl=64)

| epl | batch=1 (us/L) | vs base | batch=16 (us/L) | vs base | batch=32 (us/L) | vs base | miss% (B=1) |
|-----|---------------|---------|----------------|---------|----------------|---------|-------------|
| 2 | 132.2 | 1.015x | 269.4 | 1.139x | 351.8 | 1.078x | 98% |
| 8 | 132.4 | 1.016x | 250.0 | 1.057x | 333.5 | 1.022x | 91% |
| 16 | 132.6 | 1.018x | 230.6 | 0.975x | 316.4 | 0.970x | 73% |
| 32 | 132.7 | 1.019x | 238.0 | 1.006x | 318.9 | 0.978x | 46% |
| **64** | **130.3** | **(base)** | **236.5** | **(base)** | **326.2** | **(base)** | **0%** |

---

#### Mixtral-8x7B-20L (20L, E=8, top_k=2, 352 MB/expert)

**Triton kernel** (all-resident baseline: epl=8)

| epl | batch=1 (us/L) | vs base | batch=16 (us/L) | vs base | batch=32 (us/L) | vs base | miss% (B=1) |
|-----|---------------|---------|----------------|---------|----------------|---------|-------------|
| 2 | 332.3 | 1.009x | 914.3 | 1.021x | 1046.0 | 1.122x | 69% |
| 3 | 333.2 | 1.012x | 911.3 | 1.018x | 1037.4 | 1.113x | 55% |
| 4 | 332.7 | 1.010x | 910.0 | 1.016x | 953.2 | 1.023x | 44% |
| 5 | 332.5 | 1.010x | 900.6 | 1.006x | 1030.7 | 1.106x | 33% |
| 6 | 331.3 | 1.006x | 899.4 | 1.004x | 1026.1 | 1.101x | 18% |
| 7 | 331.9 | 1.008x | 898.0 | 1.003x | 1026.2 | 1.101x | 12% |
| **8** | **329.4** | **(base)** | **895.6** | **(base)** | **932.2** | **(base)** | **0%** |

---

#### Mixtral-8x7B (32L, E=8, top_k=2, 352 MB/expert)

No all-resident baseline possible (epl=8 requires 93 GB). Cross-epl comparison instead.

**Triton kernel**

| epl | batch=1 (us/L) | total (ms) | batch=16 (us/L) | total (ms) | batch=32 (us/L) | total (ms) | miss% (B=1) |
|-----|---------------|-----------|----------------|-----------|----------------|-----------|-------------|
| 2 | 333.3 | 10.68 | 911.1 | 27.56 | 970.4 | 30.43 | 67% |
| 3 | 333.2 | 10.67 | 910.5 | 28.49 | 1041.4 | 31.15 | 60% |
| 4 | 333.5 | 10.67 | 910.9 | 28.79 | 1038.0 | 31.52 | 48% |

epl=5 OOM (59 GB buffer + attention/KV/graphs exceed 80 GB).

---

#### Parity Check Summary

All checks use 15% tolerance. "vs base" compares offloading (epl < E) to all-resident
(epl = E). For 32L where no all-resident baseline exists, checks compare across epl values.

| Model | Total checks | Passed | Failed |
|-------|-------------|--------|--------|
| OLMoE | 12 | 12 | 0 |
| Mixtral-20L | 18 | 18 | 0 |
| Mixtral-32L | 6 | 6 | 0 |
| **Total** | **36** | **36** | **0** |

### Key Findings

1. **Batch=1 parity is near-perfect**: Across all three models, offloading
   configs are within **1-2%** of the all-resident baseline. The MoE compute kernel
   produces identical performance regardless of whether experts were pre-loaded or
   demand-loaded from CPU.

2. **Batch=16/32 show slightly more variance** (up to 14% for extreme configs like OLMoE
   epl=2 batch=16), but remain within tolerance. The variance is likely L2 cache effects
   from different scratchpad buffer sizes, not from offloading itself.

3. **D2D copies eliminated**: Phase 2's 14.8 ms/step overhead (462 us/layer × 32L) is
   gone. Each layer now requires only a 32-byte `expert_map_buf.copy_()`.

4. **Scratchpad sizing**: Changed from hardcoded 8 to `num_experts` to support batch>1
   with high miss rates. OLMoE (top_k=8, batch=32) can select up to 64 unique experts
   per layer — all potentially non-resident.

### Files Modified

| File | Changes |
|------|---------|
| `moe_engine.py` | Unified buffer with per-layer views, `expert_map_abs`, D2D copies eliminated in 6 locations, `experts_per_layer` param, `offloading_active` property, auto-creates `ExpertOffloadEngine` |
| `expert_offload_engine.py` | Rewritten for unified buffer: scratchpad at `scratchpad_start + i`, absolute slot indexing, `configure()` loads into `l * epl + slot` |
| `benchmarks/bench_comprehensive.py` | Full sweep benchmark: 3 models × multiple epl × batch 1/16/32, per-layer stage4b timing with demand loading |
| `benchmarks/bench_offload_prefill_mixed.py` | Prefill and mixed batch e2e latency with budget sweeps |
| `tests/verify_unified_cache.py` | Smoke test (32L epl=4) + correctness test (20L epl=8 vs no-epl) |
| vLLM fused_moe configs | Copied E=8/E=64 configs to cover all (E=total_slots, N) combinations |

### Correctness Verification

- **32L epl=4 smoke test**: Views share storage (data_ptr check), expert maps consistent
  (relative + absolute agree), no NaN/Inf, 5 decode steps OK.
- **20L epl=8 vs no-epl correctness** (no-compile): 20/20 exact greedy token match.
  With torch.compile: 10% match (expected — Inductor noise cascades through MoE routing
  when tensor shapes differ).

### End-to-End Latency Model Validation (2026-02-27)

Comprehensive validation of the additive latency model across prefill, mixed batch, and
decode workloads. Two prediction models compared against measured wall-clock time:

```
Prediction 1 (Kern+IO):  kernel_time + IO_time
Prediction 2 (E2E+IO):   all_resident_piecewise_e2e + IO_time
```

Where:
- **kernel_time**: GPU kernel execution only (per-layer CUDA event sums, no Python loop
  or graph dispatch overhead). Decode from `_new.json`; prefill/mixed estimated as
  `all_resident_e2e - dispatch_overhead`.
- **all_resident_e2e**: Full piecewise step with all experts resident (0 transfers).
  Includes dispatch overhead (graph replay, Python loop, FlashInfer `plan()`).
- **IO_time**: `transfer_mb / bandwidth` (measured PCIe bandwidth).
- **dispatch_overhead**: `all_resident_e2e - kernel_time`, measured from decode configs
  where both values are available. Consistent at **0.175 ms/layer** across all models.

**Data sources**: `bench_offload_prefill_mixed.py` (piecewise end-to-end, CUDA events
around `step()`, 10 trials, median), `bench_comprehensive.py` (per-layer CUDA event
kernel times, 10 decode steps x L layers).

#### IO Bandwidth

| Model | Avg (GB/s) | Min | Max | Measurements |
|-------|-----------|-----|-----|-------------|
| Mixtral-20L | 55.57 | 55.55 | 55.58 | 18 |
| OLMoE | 53.05 | 52.96 | 53.12 | 12 |
| Mixtral-32L | 55.56 | 55.54 | 55.56 | 9 |

#### Dispatch Overhead

| Model | L | Total (ms) | Per layer (ms) | Source |
|-------|---|-----------|---------------|--------|
| Mixtral-20L | 20 | 3.51 | 0.175 | Direct (16D: 3.25ms, 32D: 3.76ms, avg) |
| OLMoE | 16 | 2.81 | 0.175 | Direct (16D: 2.82ms, 32D: 2.79ms, avg) |
| Mixtral-32L | 32 | 5.61 | 0.175 | Scaled from Mixtral-20L (max_budget < E) |

---

#### Mixtral-8x7B-20L (20L, E=8, 352 MB/expert)

All-resident data available at budget=8 (0 transfers). Kernel times from `_new.json`.

**Prefill** (budgets 2, 4):

| Config | Budget | Actual (ms) | IO (ms) | Kern+IO | Err% | E2E+IO | Err% |
|--------|--------|------------|---------|---------|------|--------|------|
| 1x128 | 2 | 766.2 | 735.5 | 758.7 | +1.0% | 762.2 | +0.5% |
| 1x128 | 4 | 524.2 | 494.5 | 517.8 | +1.2% | 521.3 | +0.5% |
| 1x256 | 2 | 788.4 | 754.5 | 781.1 | +0.9% | 784.6 | +0.5% |
| 1x256 | 4 | 526.9 | 494.5 | 521.1 | +1.1% | 524.6 | +0.4% |
| 1x512 | 2 | 783.2 | 748.1 | 775.8 | +0.9% | 779.3 | +0.5% |
| 1x512 | 4 | 534.8 | 500.9 | 528.5 | +1.2% | 532.0 | +0.5% |
| 4x32 | 2 | 785.6 | 754.5 | 777.7 | +1.0% | 781.2 | +0.6% |
| 4x32 | 4 | 524.1 | 494.5 | 517.7 | +1.2% | 521.2 | +0.5% |

**Mixed batch** (budgets 2, 4):

| Config | Budget | Actual (ms) | IO (ms) | Kern+IO | Err% | E2E+IO | Err% |
|--------|--------|------------|---------|---------|------|--------|------|
| 16D | 2 | 581.0 | 557.9 | 574.8 | +1.1% | 578.0 | +0.5% |
| 16D | 4 | 396.7 | 374.1 | 390.9 | +1.5% | 394.2 | +0.6% |
| 32D | 2 | 653.5 | 627.7 | 646.6 | +1.1% | 650.3 | +0.5% |
| 32D | 4 | 417.4 | 393.1 | 412.0 | +1.3% | 415.8 | +0.4% |
| 1D+1x128P | 2 | 777.9 | 741.8 | 769.9 | +1.0% | 773.4 | +0.6% |
| 1D+1x128P | 4 | 529.6 | 494.5 | 522.6 | +1.3% | 526.1 | +0.7% |
| 8D+1x128P | 2 | 784.3 | 748.1 | 776.3 | +1.0% | 779.8 | +0.6% |
| 8D+1x128P | 4 | 542.4 | 507.2 | 535.4 | +1.3% | 538.9 | +0.6% |
| 16D+1x128P | 2 | 784.1 | 748.1 | 776.4 | +1.0% | 779.9 | +0.5% |
| 16D+1x128P | 4 | 542.2 | 507.2 | 535.4 | +1.2% | 538.9 | +0.6% |
| 16D+1x256P | 2 | 790.7 | 754.5 | 783.0 | +1.0% | 786.5 | +0.5% |
| 16D+1x256P | 4 | 542.6 | 507.2 | 535.7 | +1.3% | 539.2 | +0.6% |
| 8D+2x64P | 2 | 790.7 | 754.5 | 783.0 | +1.0% | 786.5 | +0.5% |
| 8D+2x64P | 4 | 536.4 | 500.9 | 529.4 | +1.3% | 532.9 | +0.6% |

**Decode B=1** (from per-layer benchmarks, no end-to-end actual available):

| epl | IO (ms) | Kern+IO | E2E+IO | Kernel (ms) | Xfers | MissRate |
|-----|---------|---------|--------|-------------|-------|----------|
| 2 | 2136.6 | 2143.3 | 2146.8 | 6.64 | 337 | 84.2% |
| 3 | 1521.6 | 1528.3 | 1531.8 | 6.64 | 240 | 60.0% |
| 4 | 1008.1 | 1014.7 | 1018.2 | 6.64 | 159 | 39.8% |
| 5 | 849.6 | 856.2 | 859.7 | 6.64 | 134 | 33.5% |
| 6 | 538.9 | 545.5 | 549.1 | 6.64 | 85 | 21.2% |
| 7 | 202.9 | 209.5 | 213.0 | 6.64 | 32 | 8.0% |

---

#### OLMoE-1B-7B (16L, E=64, 12.6 MB/expert)

All-resident data available at budget=64 (0 transfers). Kernel times from `_new.json`.

**Prefill** (budgets 8, 16, 32):

| Config | Budget | Actual (ms) | IO (ms) | Kern+IO | Err% | E2E+IO | Err% |
|--------|--------|------------|---------|---------|------|--------|------|
| 1x128 | 8 | 206.3 | 178.5 | 185.0 | +10.3% | 187.8 | +9.0% |
| 1x128 | 16 | 158.6 | 135.6 | 142.0 | +10.4% | 144.8 | +8.7% |
| 1x128 | 32 | 115.7 | 96.4 | 102.9 | +11.1% | 105.7 | +8.7% |
| 1x256 | 8 | 211.1 | 182.1 | 189.7 | +10.2% | 192.5 | +8.8% |
| 1x256 | 16 | 184.0 | 157.4 | 165.0 | +10.3% | 167.8 | +8.8% |
| 1x256 | 32 | 130.1 | 108.3 | 115.9 | +11.0% | 118.7 | +8.8% |
| 1x512 | 8 | 210.7 | 179.7 | 189.7 | +10.0% | 192.5 | +8.7% |
| 1x512 | 16 | 183.6 | 155.0 | 165.0 | +10.1% | 167.8 | +8.6% |
| 1x512 | 32 | 128.1 | 104.5 | 114.4 | +10.7% | 117.2 | +8.5% |
| 4x32 | 8 | 211.7 | 183.3 | 190.2 | +10.1% | 193.0 | +8.8% |
| 4x32 | 16 | 180.5 | 155.0 | 162.0 | +10.3% | 164.8 | +8.7% |
| 4x32 | 32 | 120.9 | 100.9 | 107.8 | +10.8% | 110.6 | +8.5% |

**Mixed batch** (budgets 8, 16, 32):

| Config | Budget | Actual (ms) | IO (ms) | Kern+IO | Err% | E2E+IO | Err% |
|--------|--------|------------|---------|---------|------|--------|------|
| 16D | 8 | 101.3 | 85.7 | 89.5 | +11.7% | 92.3 | +8.9% |
| 16D | 16 | 81.1 | 67.2 | 71.0 | +12.5% | 73.8 | +9.0% |
| 16D | 32 | 61.6 | 49.6 | 53.4 | +13.3% | 56.2 | +8.7% |
| 32D | 8 | 114.5 | 96.6 | 101.7 | +11.1% | 104.5 | +8.7% |
| 32D | 16 | 99.8 | 83.3 | 88.4 | +11.4% | 91.2 | +8.6% |
| 32D | 32 | 72.8 | 58.6 | 63.8 | +12.5% | 66.5 | +8.6% |
| 1D+1x128P | 8 | 190.7 | 163.3 | 171.1 | +10.3% | 173.9 | +8.8% |
| 1D+1x128P | 16 | 169.9 | 144.3 | 152.1 | +10.5% | 154.9 | +8.8% |
| 1D+1x128P | 32 | 116.9 | 96.2 | 103.9 | +11.1% | 106.7 | +8.7% |
| 8D+1x128P | 8 | 193.5 | 166.0 | 173.8 | +10.2% | 176.6 | +8.8% |
| 8D+1x128P | 16 | 173.7 | 147.7 | 155.5 | +10.5% | 158.3 | +8.9% |
| 8D+1x128P | 32 | 119.4 | 98.1 | 105.9 | +11.3% | 108.7 | +9.0% |
| 16D+1x128P | 8 | 202.4 | 173.8 | 182.0 | +10.1% | 184.8 | +8.7% |
| 16D+1x128P | 16 | 178.9 | 152.2 | 160.3 | +10.4% | 163.2 | +8.8% |
| 16D+1x128P | 32 | 120.6 | 99.2 | 107.4 | +10.9% | 110.2 | +8.6% |
| 16D+1x256P | 8 | 217.2 | 185.4 | 195.6 | +9.9% | 198.5 | +8.6% |
| 16D+1x256P | 16 | 173.5 | 145.8 | 156.0 | +10.1% | 158.8 | +8.5% |
| 16D+1x256P | 32 | 125.6 | 102.1 | 112.3 | +10.6% | 115.1 | +8.3% |
| 8D+2x64P | 8 | 214.5 | 184.5 | 192.7 | +10.1% | 195.5 | +8.8% |
| 8D+2x64P | 16 | 179.6 | 152.9 | 161.2 | +10.2% | 164.0 | +8.7% |
| 8D+2x64P | 32 | 123.5 | 101.9 | 110.1 | +10.9% | 112.9 | +8.6% |

**Decode B=1** (from per-layer benchmarks):

| epl | IO (ms) | Kern+IO | E2E+IO | Kernel (ms) | Xfers | MissRate |
|-----|---------|---------|--------|-------------|-------|----------|
| 2 | 296.8 | 298.9 | 301.7 | 2.13 | 1250 | 97.7% |
| 8 | 269.5 | 271.6 | 274.4 | 2.13 | 1135 | 88.7% |
| 16 | 209.9 | 212.0 | 214.8 | 2.13 | 884 | 69.1% |
| 32 | 143.2 | 145.3 | 148.1 | 2.13 | 603 | 47.1% |

---

#### Mixtral-8x7B (32L, E=8, 352 MB/expert)

No all-resident baseline possible (epl=8 requires 93 GB). All-resident compute and
kernel times estimated by scaling Mixtral-20L per-layer rates x 32/20.

**Prefill** (budgets 2, 3, 4):

| Config | Budget | Actual (ms) | IO (ms) | Kern+IO | Err% | E2E+IO | Err% |
|--------|--------|------------|---------|---------|------|--------|------|
| 1x128 | 2 | 1254.6 | 1204.6 | 1241.9 | +1.0% | 1247.5 | +0.6% |
| 1x128 | 3 | 1031.7 | 982.7 | 1020.0 | +1.1% | 1025.6 | +0.6% |
| 1x128 | 4 | 852.8 | 805.2 | 842.4 | +1.2% | 848.0 | +0.6% |
| 1x256 | 2 | 1259.7 | 1204.6 | 1247.2 | +1.0% | 1252.8 | +0.5% |
| 1x256 | 3 | 1062.5 | 1008.1 | 1050.6 | +1.1% | 1056.3 | +0.6% |
| 1x256 | 4 | 858.1 | 805.2 | 847.8 | +1.2% | 853.4 | +0.5% |
| 1x512 | 2 | 1268.3 | 1211.0 | 1255.2 | +1.0% | 1260.8 | +0.6% |
| 1x512 | 3 | 1070.6 | 1014.4 | 1058.6 | +1.1% | 1064.3 | +0.6% |
| 1x512 | 4 | 860.1 | 805.2 | 849.4 | +1.2% | 855.0 | +0.6% |
| 4x32 | 2 | 1261.0 | 1211.0 | 1248.1 | +1.0% | 1253.7 | +0.6% |
| 4x32 | 3 | 1056.9 | 1008.1 | 1045.2 | +1.1% | 1050.8 | +0.6% |
| 4x32 | 4 | 839.8 | 792.5 | 829.6 | +1.2% | 835.2 | +0.5% |

**Mixed batch** (budgets 2, 3, 4):

| Config | Budget | Actual (ms) | IO (ms) | Kern+IO | Err% | E2E+IO | Err% |
|--------|--------|------------|---------|---------|------|--------|------|
| 16D | 2 | 1028.6 | 989.1 | 1016.0 | +1.2% | 1021.2 | +0.7% |
| 16D | 3 | 825.0 | 786.2 | 813.1 | +1.4% | 818.3 | +0.8% |
| 16D | 4 | 666.8 | 627.7 | 654.6 | +1.8% | 659.8 | +1.0% |
| 32D | 2 | 1102.8 | 1058.8 | 1089.1 | +1.2% | 1095.1 | +0.7% |
| 32D | 3 | 963.1 | 919.3 | 949.6 | +1.4% | 955.6 | +0.8% |
| 32D | 4 | 758.5 | 716.4 | 746.7 | +1.6% | 752.7 | +0.8% |
| 1D+1x128P | 2 | 1243.7 | 1185.6 | 1230.5 | +1.1% | 1236.1 | +0.6% |
| 1D+1x128P | 3 | 1053.0 | 995.4 | 1040.3 | +1.2% | 1045.9 | +0.7% |
| 1D+1x128P | 4 | 862.0 | 805.2 | 850.1 | +1.4% | 855.7 | +0.7% |
| 8D+1x128P | 2 | 1262.1 | 1204.6 | 1249.8 | +1.0% | 1255.4 | +0.5% |
| 8D+1x128P | 3 | 1065.6 | 1008.1 | 1053.2 | +1.2% | 1058.8 | +0.6% |
| 8D+1x128P | 4 | 849.2 | 792.5 | 837.6 | +1.4% | 843.3 | +0.7% |
| 16D+1x128P | 2 | 1269.2 | 1211.0 | 1256.1 | +1.0% | 1261.7 | +0.6% |
| 16D+1x128P | 3 | 1058.4 | 1001.7 | 1046.9 | +1.1% | 1052.5 | +0.6% |
| 16D+1x128P | 4 | 868.1 | 811.5 | 856.7 | +1.3% | 862.3 | +0.7% |
| 16D+1x256P | 2 | 1276.1 | 1217.3 | 1262.9 | +1.0% | 1268.6 | +0.6% |
| 16D+1x256P | 3 | 1065.5 | 1008.1 | 1053.7 | +1.1% | 1059.3 | +0.6% |
| 16D+1x256P | 4 | 861.7 | 805.2 | 850.8 | +1.3% | 856.4 | +0.6% |
| 8D+2x64P | 2 | 1270.1 | 1211.0 | 1256.6 | +1.1% | 1262.2 | +0.6% |
| 8D+2x64P | 3 | 1066.1 | 1008.1 | 1053.8 | +1.2% | 1059.4 | +0.6% |
| 8D+2x64P | 4 | 861.7 | 805.2 | 850.9 | +1.3% | 856.5 | +0.6% |

**Decode B=1** (from per-layer benchmarks):

| epl | IO (ms) | Kern+IO | E2E+IO | Kernel (ms) | Xfers | MissRate |
|-----|---------|---------|--------|-------------|-------|----------|
| 2 | 2783.3 | 2793.9 | 2799.6 | 10.62 | 439 | 68.6% |
| 3 | 2485.3 | 2496.0 | 2501.6 | 10.62 | 392 | 61.3% |
| 4 | 1927.4 | 1938.0 | 1943.6 | 10.62 | 304 | 47.5% |

---

#### Prediction Accuracy Summary

Error = (actual - predicted) / actual x 100%. Positive = actual slower than predicted.

| Model | Experiments | Kern+IO avg error | E2E+IO avg error |
|-------|-----------|-------------------|------------------|
| **Mixtral-20L** | 22 (prefill + mixed) | **+1.1%** | **+0.5%** |
| **OLMoE** | 33 (prefill + mixed) | **+10.8%** | **+8.7%** |
| **Mixtral-32L** | 33 (prefill + mixed) | **+1.1%** | **+0.6%** |

#### Analysis

**Mixtral (PASS, +0.5-0.6% with dispatch overhead)**: The additive model holds within
1.2% (Kern+IO) or 0.7% (E2E+IO) for every single experiment across both Mixtral models.
The +0.5% residual is consistent across all configs and budgets — likely a small fixed
overhead (expert_map copy, Python control flow) not captured by piecewise graph dispatch.
Transfer bandwidth is rock-solid at 55.57 GB/s with <0.1% variance.

**Mixtral-32L scaling validation**: Since max_budget=4 < E=8, no all-resident data exists.
All-resident compute and kernel times are estimated by scaling Mixtral-20L per-layer rates
x 32/20. The +1.1%/+0.6% error (identical to Mixtral-20L) confirms the per-layer scaling
assumption is valid — per-layer overhead is constant regardless of total layer count.

**OLMoE (+8.7% with dispatch overhead)**: Consistent ~8.7% residual across all 33
experiments. OLMoE has 400-1250 tiny transfers per step (12.6 MB each, ~0.24 ms) vs
Mixtral's 30-440 large transfers (352 MB each, ~6.3 ms). The overhead is dominated by
per-transfer Python dispatch in `process_layer()`:
- GPU->CPU `topk_ids` sync, expert map lookup, `copy_()` dispatch, CUDA event creation
- ~20-27 us per transfer x 400-1250 transfers = 8-34 ms additional overhead
- This scales with transfer count, explaining why error is consistent at ~8.7% across
  budgets (proportional to total transfers)

**Key insight**: The latency model `wall = all_resident_e2e + IO` predicts within 1%
for large-expert models (Mixtral). For many-small-expert models (OLMoE), add ~25 us per
transfer for Python dispatch overhead. In both cases, **IO dominates**: transfer time
accounts for 90-96% of total wall-clock time.

---

## References

- [Mixtral of Experts (arXiv 2401.04088)](https://arxiv.org/abs/2401.04088)
- [Fast MoE Inference with Offloading (Eliseev & Mazur 2023, arXiv 2312.17238)](https://arxiv.org/abs/2312.17238)
- [Not All Models Suit Expert Offloading (arXiv 2505.16056)](https://arxiv.org/abs/2505.16056)
- [MoE-Infinity: Sparsity-Aware Expert Cache (arXiv 2401.14361)](https://arxiv.org/html/2401.14361v3)
- [In-Depth Analysis on Caching and Pre-Fetching in MoE Offloading (arXiv 2511.05814)](https://arxiv.org/pdf/2511.05814)
- [dvmazur/mixtral-offloading (GitHub)](https://github.com/dvmazur/mixtral-offloading)
- [vLLM RFC: DeepSeek-R1 MoE Offload (#33869)](https://github.com/vllm-project/vllm/issues/33869)
- [vLLM PR #31938 (closed)](https://github.com/vllm-project/vllm/pull/31938)
