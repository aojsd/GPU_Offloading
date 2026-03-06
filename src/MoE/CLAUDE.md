We want to create a sort of sandbox environment for experimenting with expert offloading for MoE LLMs. The sandbox should allow testing for hypothetical expert caching and prefetching policies (where offloaded experts are stored in CPU memory) with the following flow:
1. We collect a trace of the experts accessed by a real MoE model for a sequence of inputs/batches of inputs.
2. We simulate a custom caching + prefetching policy on this expert access trace. The simulation should return a "data movement" trace with the following information:
- The initial state of the cache (which experts are initially present on GPU vs. offloaded to CPU)
- The timing and target of each prefetch for an expert, ignoring prefetches for experts that are already on GPU. Say prefetches can be triggered at the beginning of a forward pass, and during any layer right after the layer's expert selections have been calculated.
- Any demand loads for offloaded experts (should occur between expert selection and computation).
- Experts that are evicted whenever a prefetch or demand load occurs.
3. We then want the capability to "replay" this trace for the same MoE and requests on real GPU hardware to assess how well a hypothetical data movement policy would have performed in practice. Since we don't plan to change the functional behavior of the LLM, let performance always refer to runtime performance (e.g. time to first token, tokens/s).

Note that in order for this sandbox to be useful, we need to accurately capture both the behavior (expert selection) and compute performance of state-of-the-art serving systems. Ideally, we'd implement this using a framework like vLLM or SGLang, but I'm not sure if their APIs are enough to build this trace-based sandbox. If we instead build our implementation from scratch, we need to ensure that step 3 uses existing state-of-the-art kernels and compilation frameworks to maximize potential performance to provide useful insights. We'd also need to make sure our trace collection is accurate.

IMPORTANT Meta-Instructions:
After reasonable checkpoints, or when finding a critical bug, summarize important ideas and changes and append them to the end of this CLAUDE.md file, or the relevant .md file pointed to within this CLAUDE.md file. Alternatively, do this whenever you have been working for a long time and may soon exceed the 200k context limit. When a major progression is finished (e.g., completing a long task given by user prompt), refactor the CLAUDE.md file to remove the intermediate checkpoint information.

Aim to write most, if not all, code in Python/Pytorch or Pytorch-compatible packages.

For PyTorch code, we are utterly uninterested in completely eager execution when it comes to real experiments or any test that is performance related.

---

## Environment

- **GPUs**: 2x NVIDIA H100 80GB HBM3 (SM90), **CUDA** 12.8, **System**: RHEL 8.10 (glibc 2.28)
- **Python** 3.13.5, **PyTorch** 2.9.0+cu128, **Triton** 3.5.0
- **vLLM** 0.11.2 (pip prebuilt wheel), includes **xformers** 0.0.33.post1 + **FlashInfer** 0.5.2
- glibc 2.28 constrains us: vLLM 0.12+ wheels require glibc 2.31; 0.11.2 is the latest compatible.

---

## Sandbox Design

### Target Model: Mixtral-8x7B

| Parameter | Value |
|---|---|
| Layers | 32 |
| Experts per layer | 8 (top-2 selected per token) |
| Hidden dim | 4096 |
| FFN intermediate dim | 14336 |
| Per expert FP16 | ~336 MB |
| All experts (32 layers x 8) | ~86 GB (exceeds 80GB GPU — offloading mandatory) |
| Expert CPU→GPU transfer | ~6.5 ms at ~50 GB/s PCIe |
| MoE layer compute (top-2) | ~0.27 ms |

A single expert cache miss (6.5ms) costs 24x more than the MoE compute for that layer. Data movement policy is everything.

### Data Movement Trace Format

Uses a **unified expert cache** — a single pool of `cache_size` slots shared
across all layers. Any slot can hold any `(layer, expert_id)`.

```
DataMovementTrace:
  cache_size: int                               # total unified GPU cache slots
  initial_cache_state: list[(layer, expert)]     # experts on GPU at start
  steps: list[StepTrace]                         # one per decode token

StepTrace:
  layers: list[LayerTrace]                       # 32 layers
  scheduling: StepScheduling | None              # batch composition + events

LayerTrace:
  topk_ids: list[list[int]]                      # expert selections per token
  topk_weights: list[list[float]]
  prefetches: list[TransferEvent]                # async transfers for this layer
  demand_loads: list[TransferEvent]              # blocking (cache miss)

TransferEvent:
  target: (layer, expert)
  evict: (layer, expert) | None                  # cross-layer eviction allowed
```

### Replay Loop (Per-Layer)

```
Layer 0:
  1. Issue layers[0].prefetches on transfer stream (async)
  2. Run stages 1-4a on compute stream (overlaps with prefetches)
  3. Sync prefetches, execute demand_loads (blocking)
  4. Issue layers[1].prefetches on transfer stream (async)
  5. Run stage 4b (MoE compute)

Layer L (L > 0):
  1. Run stages 1-4a (prefetches from L-1 running in background)
  2. Sync prefetches, execute demand_loads (blocking)
  3. Issue layers[L+1].prefetches on transfer stream (async)
  4. Run stage 4b (MoE compute)
```

---

## Implementation Phases

1. **Core Compute Engine** (complete): Custom MoE engine with CUDA graph + torch.compile, validated against vLLM on OLMoE-1B-7B and Mixtral-8x7B. See [decode.md](vLLM_comparison/decode.md) and [prefill.md](vLLM_comparison/prefill.md).
2. **Mixed Batch Support** (complete): Mixed prefill+decode via `mixed_step()`, piecewise CUDA graphs (per-layer 4-stage decomposition). 80.4% token match vs vLLM without compile (expected BF16 divergence). See [mixed_batch.md](vLLM_comparison/mixed_batch.md).
3. **Simulated Expert Offloading** (complete): Split stage 4 into 4a (router) + 4b (MoE) with CPU break. ExpertOffloadEngine demand-loads missing experts, records traces. Verified bit-identical + predicted latency = compute + IO. See [offload_1GPU.md](offload_1GPU.md).
4. **Unified Expert Cache** (complete): Single `w1_buf[L*epl + scratchpad, 2*I, H]` buffer with per-layer views. Eliminated D2D copies (was 14.8 ms/step on 32L). Compute parity 36/36 checks passed. Latency model `wall = compute + IO` validated across 88 experiments (prefill, mixed batch, decode): Mixtral-20L +0.5%, Mixtral-32L +0.6%, OLMoE +8.7% (Python dispatch overhead with many small transfers). IO accounts for 90-96% of wall-clock time. See [offload_1GPU.md](offload_1GPU.md).
5. **Async Transfer & Prefetch** (complete): Two-stream architecture — transfer stream overlaps CPU→GPU copies with compute stream. Single CUDA event synchronization. See [replay.md](replay.md).
6. **Trace Format & Replay Controller** (complete): `DataMovementTrace` with unified cache (`cache_size` slots shared across all layers), `ReplayController` with optimized prefetch timing (layer 0 before stage1, subsequent layers before stage4b of previous layer). See [replay.md](replay.md).
7. **Policy Simulation** (complete): Orthogonal `CachePolicy` × `PrefetchPolicy` decomposition. Cache policies: LRU, Belady (MIN), LFU (windowed), StaticFreq (global oracle). Prefetch policies: NoPrefetch, OraclePrefetch (layer-0 + 1-layer lookahead). All 8 combinations validated. See [replay.md](replay.md).

---

## Custom MoE Engine (Phase 1 — Complete)

Model-agnostic engine supporting any HuggingFace MoE model with gate/up/down expert
projections. Architecture params read from `config.json`. Validated on OLMoE-1B-7B
and Mixtral-8x7B (truncated to 20 layers, 54.6 GB BF16, fits single H100).

Guards raise `NotImplementedError` for unsupported features: sliding window attention
and RoPE scaling.

### OLMoE-1B-7B Architecture

| Parameter | Value |
|---|---|
| Layers | 16 |
| Experts per layer | 64 (top-8 selected) |
| Hidden / Intermediate | 2048 / 1024 |
| Attention | 16 heads, 16 KV heads (MHA), head_dim=128 |
| Special | Q/K norm on flat [B, 2048] **before** head reshape and RoPE; norm_topk_prob=False |
| Total size | ~13.8 GB BF16 (fits on one H100) |

### Mixtral-8x7B Architecture (Truncated)

| Parameter | Value |
|---|---|
| Layers | 20 (truncated from 32 via `models/truncate_model.py`) |
| Experts per layer | 8 (top-2 selected) |
| Hidden / Intermediate | 4096 / 14336 |
| Attention | 32 heads, 8 KV heads (GQA), head_dim=128 |
| Special | norm_topk_prob=True; weight naming: w1/w2/w3 (not gate_proj/up_proj/down_proj) |
| Total size | ~54.6 GB BF16 (fits on one H100 with ~17 GB for KV cache) |

**Mixtral-specific adaptations**:
- Weight loading detects `w1`/`w2`/`w3` naming (w1=gate, w3=up, w2=down)
- GQA support: `rope_pytorch` takes separate `num_kv_heads` param for k reshape
- w1/w2 kept as Python lists (not stacked) to avoid 2x peak memory for large FFNs
- fused_moe config: copied from H200 to H100 (same SM90); without it, 10% slower on decode

### Engine API

```python
engine = MoEEngine("src/MoE/models/OLMoE-1B-7B")

# CUDA graph capture — REQUIRED before any prefill/decode
engine.capture_prefill_cuda_graph(total_token_sizes=[128, 256, 512, 1024, 2048])
engine.capture_decode_cuda_graph(batch_size=1, warmup_seq_len=128, max_decode_tokens=256)
engine.capture_mixed_cuda_graphs(total_token_sizes=[128, 256, 512, 1024, 2048])

# Prefill (graph-only, no eager fallback)
logits = engine.prefill(input_ids)            # [B, S, vocab]
logits = engine.prefill_to_slot(seq_id, ids)  # [S, vocab]
logits = engine.prefill_batch_to_slots(       # [N_total, vocab] (flat output)
    seq_ids, input_ids)                       # input_ids: list[Tensor] or Tensor[B,S]

# Decode (graph auto-dispatch)
logits = engine.decode_step(token_ids, pos)   # [B, vocab]

# Mixed batch (auto-dispatches to piecewise CUDA graphs if captured, else eager)
logits = engine.mixed_step(                   # [N_total, vocab]
    decode_seq_ids, decode_tokens, prefill_seq_ids, prefill_input_ids)

tokens = engine.generate(input_ids, max_new_tokens=128)
```

Prefill CUDA graphs are keyed by **total token count** (not `(batch_size, seq_len)` pairs).
One graph serves any sequence combination fitting within that total — e.g., a graph
captured at `total_token_sizes=[256]` handles `1×256`, `4×64`, `2×100+1×56`, etc.
Padding tokens use `slot_mapping = -1` (vLLM sentinel — `reshape_and_cache_flash` skips
KV writes for negative slots). No eager prefill fallback; `_find_nearest_prefill_total`
raises `RuntimeError` if no captured graph covers the requested total.

### Correctness

- Greedy generation: exact match vs HuggingFace (56/56 tokens on test prompt)
- CUDA graph vs eager: exact match (30 tokens)
- vs vLLM: 2/3 prompts exact match (1 diverges at BF16 tie-break, gap=0.0625)

### Performance Summary

**OLMoE-1B-7B standalone benchmarks** (CUDA graph + torch.compile):

| Phase | Custom | vLLM | Status |
|-------|--------|------|--------|
| **Decode** (seq128) | 2.87ms/step | 2.93ms/step | **1-2% faster** — see [decode.md](vLLM_comparison/decode.md) |
| **Prefill** (seq128) | 7.65ms | 7.86ms | **2-3% faster** — see [prefill.md](vLLM_comparison/prefill.md) |
| **Prefill** (seq1024) | 11.61ms | 17.20ms | **32% faster** — see [prefill.md](vLLM_comparison/prefill.md) |

**Mixtral-8x7B-20L standalone benchmarks** (CUDA graph + torch.compile):

| Phase | Custom | vLLM | Status |
|-------|--------|------|--------|
| **Decode** (seq128) | 9.13ms/step | 9.12ms/step | **1.00x** (exact match) |
| **Decode** (seq2048) | 9.31ms/step | 9.15ms/step | **0.98x** |
| **Prefill** (seq128) | 31.57ms | 31.66ms | **0.3% faster** |
| **Prefill** (seq1024) | 86.34ms | 90.02ms | **4.1% faster** |
| **Prefill** (seq2048) | 159.73ms | 162.93ms | **2.0% faster** |

**Trace-and-replay** (piecewise CUDA graphs + torch.compile, vLLM batches replayed):

**OLMoE-1B-7B:**

| Workload | Custom total | vLLM total | Speedup |
|----------|-------------|-----------|---------|
| Single (1 req, 50 decode) | 178.43ms | 206.16ms | **1.16x** |
| Staggered (8 req, mixed) | 204.32ms | 243.21ms | **1.19x** |

**Mixtral-8x7B-20L** (vLLM with chunked prefill enabled — default/best settings):

| Workload | Custom total | vLLM total | Speedup |
|----------|-------------|-----------|---------|
| Single (1 req, 50 decode) | 473.47ms | 489.65ms | **1.03x** |
| Staggered (8 req, mixed) | 709.19ms | 833.70ms | **1.18x** |

| Metric (staggered) | Custom | vLLM | Speedup |
|---------------------|--------|------|---------|
| Avg TTFT | 28.98ms | 63.38ms | **2.19x** |
| Throughput | 366.6 tok/s | 311.9 tok/s | **1.18x** |

Custom engine beats vLLM on all workloads at vLLM's best default settings.
Mixed steps 3.39x faster; TTFT 2.19x faster on staggered workload.
Cross-engine correctness: piecewise is bit-identical to eager (260/260 tokens). Without
torch.compile, custom matches vLLM at **80.4%** (expected BF16 divergence). The 4% match
with torch.compile is confirmed Inductor numerical noise — CUDA graphs add zero error.
See [mixed_batch.md](vLLM_comparison/mixed_batch.md) for full results.

### glibc 2.28 Workaround

vLLM's `_moe_C.abi3.so` requires glibc 2.29. Three ops monkey-patched in `moe_engine.py`:
`moe_align_block_size` (vectorized, no `.item()`), `moe_sum`, `topk_softmax`.
All CUDA-graph-capturable. Key detail: sorted_ids are **flat indices** into `topk_ids.flatten()`;
expert_ids via `searchsorted(..., right=True)`.

---

## Key Pitfalls (General)

- `VLLM_ENABLE_V1_MULTIPROCESSING=0` must be set **before** any vLLM import
- Q/K norm on flat `[B, 2048]` **before** head reshape, not per-head
- w1 gate/up ordering: gate first (rows 0:1024), up second (rows 1024:2048)
- norm_topk_prob: model-specific (OLMoE=False, Mixtral=True) — read from config.json
- `moe_align_block_size`: sorted_ids = FLAT indices; Triton kernel does `flat_idx // top_k`
- torch.compile Inductor produces numerically different greedy tokens from eager (expected)
- V tensor after QKV split is non-contiguous → needs `.contiguous()` or `.reshape()`
- FlashInfer RoPE JIT needs `ninja` in PATH (prefill only)

See [decode.md](vLLM_comparison/decode.md) and [prefill.md](vLLM_comparison/prefill.md) for phase-specific pitfalls.

---

## Files

| File | Purpose |
|------|---------|
| `moe_engine.py` | `MoEEngine` — model-agnostic engine with mixed batch, piecewise CUDA graphs, torch.compile, unified expert cache |
| `expert_offload_engine.py` | `ExpertOffloadEngine` — scratchpad demand loading into unified buffer, residency tracking, trace recording. Auto-created by MoEEngine when `experts_per_layer` is set. |
| `offload_1GPU.md` | Expert offloading research notes, Phases 1-3 implementation docs + performance data |
| `models/download.sh` | Download model weights (OLMoE, Mixtral) |
| `models/truncate_model.py` | Truncate model to N layers (for fitting large models on single GPU) |
| `vLLM_comparison/microbenchmark.py` | Consolidated benchmarks: decode, prefill, CUDA graph correctness, mixed smoke tests |
| `vLLM_comparison/batch_replay.py` | Batch-replay benchmark: run vLLM, trace batches, replay on custom engine |
| `vLLM_comparison/nsys_profiler.py` | Nsight Systems kernel profiling: Custom vs vLLM (decode + prefill) |
| `vLLM_comparison/decode.md` | Decode pipeline details, profiling data, challenges |
| `vLLM_comparison/prefill.md` | Prefill pipeline details, profiling data, gap analysis, challenges |
| `vLLM_comparison/mixed_batch.md` | Mixed prefill+decode batch support: design, piecewise CUDA graph plan |
| `datasets/download.sh` | Download datasets (ShareGPT) to shared storage with symlinks |
| `trace_construction/trace_construction.md` | Design doc: collect per-conversation activations (GPU), build batched traces via continuous batching sim (CPU) |
| `trace_construction/collect_traces.py` | Phase 1: collect per-conversation expert traces from ShareGPT conversations (GPU) |
| `trace_construction/build_trace.py` | Phase 2: continuous batching simulator with LIFO preemption → batched ActivationTrace with per-step scheduling metadata (CPU-only) |
| `trace_construction/recollect_traces.py` | Targeted re-collection of specific conversations with different parameters |
| `data_movement_trace.py` | `DataMovementTrace`, `ActivationTrace`, `TransferEvent`, `StepScheduling`, `RequestScheduling` — trace formats with scheduling metadata, JSON serialization and validation |
| `policy_simulator.py` | `CachePolicy` (LRU, Belady, LFU, StaticFreq) × `PrefetchPolicy` (NoPrefetch, OraclePrefetch) + `simulate()` — orthogonal policy simulators |
| `replay_controller.py` | `ReplayController` — replays `DataMovementTrace` on GPU with async prefetch streams and demand loading |
| `replay.md` | Cache simulation & replay documentation: trace formats, policy simulators, replay controller architecture, prefetch/eviction timing |
| `tests/test_replay_policy.py` | 43 unit tests covering trace serialization, validation, and all policy combos (unified cache) |
| `tests/test_trace_construction.py` | 23 unit tests for continuous batching simulator (page accounting, LIFO preemption, memory budget) |
| `tests/README.md` | Index of all test/benchmark scripts with descriptions and quick start |

Regenerate decode profiles: `python src/MoE/vLLM_comparison/nsys_profiler.py all`
Regenerate prefill profiles: `python src/MoE/vLLM_comparison/nsys_profiler.py prefill`

---

## Expert Offloading Infrastructure

### Architecture: Expert Cache (Phase 3 — Complete)

Two offloading modes, both using piecewise CUDA graphs (stage4a/4b split) for
CPU-side intervention between router and MoE compute:

**1. `experts_per_layer` mode** (ExpertOffloadEngine — trace recording):
```
w1_buf = [L * epl + scratchpad_slots, 2*I, H]   # per-layer partitioned + scratchpad
w2_buf = [L * epl + scratchpad_slots, H, I]
```
Per-layer views, two expert maps (`expert_map[l]` relative, `expert_map_abs[l]` absolute).
Scratchpad for demand-loaded non-residents. Used for recording activation traces.

**2. `cache_size` mode** (ReplayController — trace replay):
```
w1_buf = [cache_size, 2*I, H]                    # flat unified buffer
w2_buf = [cache_size, H, I]
```
Any slot holds any `(layer, expert_id)`. No per-layer partitioning, no scratchpad.
Only `expert_map_abs[l]` is used (absolute slot or -1). Cross-layer eviction supported.

Cannot set both `experts_per_layer` and `cache_size` simultaneously.

**ExpertOffloadEngine** (`expert_offload_engine.py`): Auto-created by MoEEngine when
`experts_per_layer` is set. Reads routing decisions after stage4a, loads missing experts
from CPU pinned memory into scratchpad, updates `expert_map_buf`, records trace of
activations and transfer times. Step counting is automatic (`begin_step()` called by
engine at the start of each piecewise decode step).

**MoE kernel**: vLLM Triton `fused_experts` only. CUTLASS support was removed (Triton
showed better consistency: 36/36 compute parity checks passed across 3 models).

**IMPORTANT**: Eager mode is NOT used in real experiments. It exists only for sanity
checks in test files. All benchmarking and offloading experiments MUST use CUDA graphs
(piecewise for offloading, flat for non-offloading baselines).

See [offload_1GPU.md](offload_1GPU.md) for full design, memory budget, and benchmarks.

---
