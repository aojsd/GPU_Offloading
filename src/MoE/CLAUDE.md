We want to create a sort of sandbox environment for experimenting with expert offloading for MoE LLMs. The sandbox should allow testing for hypothetical expert caching and prefetching policies (where offloaded experts are stored in CPU memory) with the following flow:
1. We collect a trace of the experts accessed by a real MoE model for a sequence of inputs/batches of inputs.
2. We simulate a custom caching + prefetching policy on this expert access trace. The simulation should return a "data movement" trace with the following information:
- The initial state of the cache (which experts are initially present on GPU vs. offloaded to CPU)
- The timing and target of each prefetch for an expert, ignoring prefetches for experts that are already on GPU. Say prefetches can be triggered at the beginning of a forward pass, and during any layer right after the layer's expert selections have been calculated.
- Any demand loads for offloaded experts (should occur between expert selection and computation).
- Experts that are evicted whenever a prefetch or demand load occurs.
3. We then want the capability to "replay" this trace for the same MoE and requests on real GPU hardware to assess how well a hypothetical data movement policy would have performed in practice. Since we don't plan to change the functional behavior of the LLM, let performance always refer to runtime performance (e.g. time to first token, tokens/s).

Note that in order for this sandbox to be useful, we need to accurately capture both the behavior (expert selection) and compute performance of state-of-the-art serving systems. Ideally, we'd implement this using a framework like vLLM or SGLang, but I'm not sure if their APIs are enough to build this trace-based sandbox. If we instead build our implementation from scratch, we need to ensure that step 3 uses existing state-of-the-art kernels and compilation frameworks to maximize potential performance to provide useful insights. We'd also need to make sure our trace collection is accurate.

Meta-Instructions:
After reasonable checkpoints, or when finding a critical bug, summarize important ideas and changes and append them to the end of this CLAUDE.md file, or the relevant .md file pointed to within this CLAUDE.md file. Alternatively, do this whenever you have been working for a long time and may soon exceed the 200k context limit. When a major progression is finished (e.g., completing a long task given by user prompt), refactor the CLAUDE.md file to remove the intermediate checkpoint information.

Aim to write most, if not all, code in Python/Pytorch or Pytorch-compatible packages.

DO NOT commit or push using Git unless explicitly asked to do so.

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

```
DataMovementTrace:
  initial_cache_state: list[(layer, expert)]   # experts on GPU at start
  steps: list[StepTrace]                        # one per decode token

StepTrace:
  layers: list[LayerTrace]                      # 32 layers

LayerTrace:
  topk_ids: list[list[int]]                     # expert selections per token
  topk_weights: list[list[float]]
  prefetches: list[TransferEvent]               # start of layer (overlap with attn)
  post_routing_prefetches: list[TransferEvent]  # after routing decided
  demand_loads: list[TransferEvent]             # blocking (cache miss)

TransferEvent:
  target: (layer, expert)
  evict: (layer, expert) | None
```

### Replay Loop (Per-Layer)

```
for each layer:
  1. Start prefetches (async, prefetch stream)
  2. Run attention on compute stream (overlaps with prefetches)
  3. Read expert selections from trace
  4. Start post-routing prefetches
  5. Execute demand loads (blocking wait for cache misses)
  6. Run MoE compute (all needed experts now on GPU)
  7. Trigger evictions
```

---

## Implementation Phases

1. **Core Compute Engine** (complete): Custom MoE engine with CUDA graph + torch.compile, validated against vLLM SOTA on OLMoE-1B-7B. See [decode.md](decode.md) and [prefill.md](prefill.md) for detailed performance data.
2. **Expert Cache & Async Transfer** (not started): Double-buffered GPU expert slots, CPU pinned memory, three CUDA streams.
3. **Trace Format & Replay Controller** (not started): Replay loop with data movement orchestration + timing.
4. **Trace Collection** (not started): Hook router to record expert selections per layer per token.
5. **Policy Simulation** (not started): LRU, Oracle/Belady, frequency-based, pre-gated policies.

---

## Custom MoE Engine (Phase 1 — Complete)

Model-agnostic engine supporting any HuggingFace MoE model with gate/up/down expert
projections. Architecture params read from `config.json`. Validated on OLMoE-1B-7B;
Mixtral support pending expert offloading (model too large for single GPU).

Guards raise `NotImplementedError` for unsupported features: sliding window attention,
RoPE scaling, and models exceeding GPU memory (offloading not yet implemented).

### OLMoE-1B-7B Architecture

| Parameter | Value |
|---|---|
| Layers | 16 |
| Experts per layer | 64 (top-8 selected) |
| Hidden / Intermediate | 2048 / 1024 |
| Attention | 16 heads, 16 KV heads (MHA), head_dim=128 |
| Special | Q/K norm on flat [B, 2048] **before** head reshape and RoPE; norm_topk_prob=False |
| Total size | ~13.8 GB BF16 (fits on one H100) |

### Engine API

```python
engine = MoEEngine("src/MoE/models/OLMoE-1B-7B")
logits = engine.prefill(input_ids)          # [B, S, vocab]  (CUDA graph auto-dispatch)
logits = engine.decode_step(token_ids, pos) # [B, vocab]     (CUDA graph auto-dispatch)
tokens = engine.generate(input_ids, max_new_tokens=128)
engine.capture_prefill_cuda_graph(batch_size=1, seq_lengths=[128, 256, 512, 1024, 2048])
engine.capture_decode_cuda_graph(batch_size=1, warmup_seq_len=128, max_decode_tokens=256)
```

### Correctness

- Greedy generation: exact match vs HuggingFace (56/56 tokens on test prompt)
- CUDA graph vs eager: exact match (30 tokens)
- vs vLLM: 2/3 prompts exact match (1 diverges at BF16 tie-break, gap=0.0625)

### Performance Summary

| Phase | Custom (CUDA graph) | vLLM (CUDA graph + compile) | Status |
|-------|--------|------|--------|
| **Decode** (seq128) | 2.87ms/step | 2.93ms/step | **1-2% faster** — see [decode.md](decode.md) |
| **Prefill** (seq128) | 7.65ms | 7.86ms | **2-3% faster** — see [prefill.md](prefill.md) |
| **Prefill** (seq1024) | 11.61ms | 17.20ms | **32% faster** — see [prefill.md](prefill.md) |

Custom engine beats vLLM at both prefill and decode across all sequence lengths.
Previous measurements showing vLLM 3x faster at prefill were caused by vLLM V1's
prefix caching silently reusing KV cache from warmup runs with the same prompt.
See [prefill.md](prefill.md) for the full investigation.

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
- norm_topk_prob=False: do NOT renormalize routing weights
- `moe_align_block_size`: sorted_ids = FLAT indices; Triton kernel does `flat_idx // top_k`
- torch.compile Inductor produces numerically different greedy tokens from eager (expected)
- V tensor after QKV split is non-contiguous → needs `.contiguous()` or `.reshape()`
- FlashInfer RoPE JIT needs `ninja` in PATH (prefill only)

See [decode.md](decode.md) and [prefill.md](prefill.md) for phase-specific pitfalls.

---

## Files

| File | Purpose |
|------|---------|
| `moe_engine.py` | `MoEEngine` — model-agnostic engine with piecewise CUDA graphs for both prefill and decode |
| `test_cuda_graph.py` | CUDA graph correctness (eager vs graph exact match) + timing |
| `benchmark.py` | Correctness validation (custom vs HuggingFace vs vLLM) + wall-clock timing |
| `benchmark_prefill.py` | Prefill performance: Custom FA3 vs vLLM FA3, combined prefill+decode |
| `nsys_seq_len_comparison.py` | Nsight Systems kernel profiling: Custom vs vLLM (decode + prefill) |
| `profiling/<model>/` | Per-seq-len kernel profiles (`decode_seq{N}.prof`, `prefill_seq{N}.prof`) |
| `models/OLMoE-1B-7B/` | Model weights + config |
| `decode.md` | Decode pipeline details, profiling data, challenges |
| `prefill.md` | Prefill pipeline details, profiling data, gap analysis, challenges |

Regenerate decode profiles: `python src/MoE/nsys_seq_len_comparison.py all`
Regenerate prefill profiles: `python src/MoE/nsys_seq_len_comparison.py prefill`

---

## Next: Batch Trace Execution — Compilation Constraints (Phase 2 Prep)

Analysis needed: what parts of the algorithm are "baked in" to torch.compile / CUDA graphs?
- FlashInfer decode `plan()` must run outside CUDA graph (CPU metadata) — already handled
- Block tables for paged attention — how static are they w.r.t. compilation?
- What changes between batches in a trace replay? Expert selections, KV lengths, positions
- Which of these can be updated via `.copy_()` on static buffers vs requiring re-capture?

---
