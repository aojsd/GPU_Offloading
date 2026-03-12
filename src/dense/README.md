# Dense Model Offloading Sandbox

A sandbox for experimenting with KV cache offloading for dense (non-MoE) LLMs.
The goal is a custom inference engine (`dense_engine.py`) analogous to the MoE
engine (`../MoE/moe_engine.py`) that:

1. **Parses real HuggingFace dense models** (Llama, Mistral, etc.) from
   `config.json` + safetensors — no synthetic weights.
2. **Runs them accurately** using the same production-grade attention kernels
   as the MoE engine (FlashInfer paged decode, FlashAttention varlen prefill,
   `reshape_and_cache_flash` for KV writes).
3. **Provides a foundation for KV cache offloading** experiments, where the
   offloading target is paged KV blocks (not expert weights).

---

## What `dense_engine.py` Should Do (vs `moe_engine.py`)

### Kept from `moe_engine.py`

| Component | Details |
|-----------|---------|
| Config parsing | `config.json` → `num_layers`, `hidden_size`, `num_attention_heads`, `num_kv_heads`, `head_dim`, `intermediate_size`, `rope_theta`, `vocab_size`, `rms_norm_eps`, etc. |
| Weight loading | Safetensors shards: embed_tokens, per-layer attention (q/k/v/o_proj), per-layer MLP (gate_proj/up_proj/down_proj), final RMSNorm, lm_head |
| GQA support | `num_kv_heads` != `num_heads` (Llama-2 70B, Mistral, etc.) |
| RoPE | `rope_pytorch` (NeoX-style half-split, fully compilable) |
| Q/K norm | Conditional on presence in model weights |
| KV cache | Paged block table with `reshape_and_cache_flash` writes |
| Decode attention | FlashInfer `BatchDecodeWithPagedKVCacheWrapper` |
| Prefill attention | `flash_attn_varlen_func` (vLLM FA3) |
| Mixed batches | Unified prefill+decode via `step()` |
| Compilation | `torch.compile` for fused RMSNorm + residual + RoPE |
| CUDA graphs | Piecewise capture for graph-mode inference |
| Generate loop | Prefill → autoregressive decode with greedy sampling |

### Dropped from `moe_engine.py`

| Component | Reason |
|-----------|--------|
| Expert routing | No experts in dense models |
| `fused_experts` / MoE kernels | Replaced by standard dense MLP |
| `moe_align_block_size`, `moe_sum`, `topk_softmax` | MoE-only ops |
| Expert offloading (`ExpertOffloadEngine`) | Not applicable |
| `experts_per_layer` / `cache_size` modes | Expert cache concepts |
| MLA (Multi-head Latent Attention) | DeepSeek-V2-specific |
| Pipeline parallelism | Simplify to single-GPU initially |
| glibc `_moe_C` monkey-patches | Only needed for MoE C ops |

### Changed

| Component | MoE Engine | Dense Engine |
|-----------|------------|--------------|
| MLP | `fused_experts(hidden, w1, w2, topk_w, topk_ids)` | Standard dense: `gate_up = linear(x)` → SiLU gate * up → `down = linear(x)` |
| Per-layer weights | Router + expert w1/w2/w3 | gate_proj, up_proj, down_proj |
| Offloading target | Expert weights (CPU → GPU) | KV cache blocks (CPU ↔ GPU) |

---

## KV Cache Offloading Design

### Overview

The offloading strategy (prototyped in `include/paged_offload_transformer.py`)
pipelines PCIe transfers with GPU compute using a **3-stream, event-driven
architecture**. The goal: hide the latency of moving KV cache blocks between
CPU and GPU by overlapping transfers with attention and MLP computation.

### Architecture

```
Layer 0:    Fully resident (all KV blocks on GPU, no offloading)
Layers 1..N:  Each sequence's KV blocks split by kv_offload_ratio:
                - Resident blocks → permanent GPU heap
                - Offloaded blocks → CPU pinned memory (transferred on demand)
```

### Dual Scratchpad Design

Two GPU scratchpad regions alternate by layer parity to prevent read/write
conflicts between adjacent layers:

- **Even layers** use scratchpad 0
- **Odd layers** use scratchpad 1

While compute reads from one scratchpad, H2D can safely write to the other.

### Three CUDA Streams

| Stream | Direction | Purpose |
|--------|-----------|---------|
| **Compute** (default) | — | QKV projection, attention, MLP |
| **H2D** | CPU → GPU | Load decode history blocks into scratchpad before decode attention |
| **D2H** | GPU → CPU | Save newly-written prefill blocks from scratchpad to CPU |

### Per-Layer Pipeline (Layers 1..N)

```
Compute stream:                    H2D stream:           D2H stream:
                                   ─────────────         ─────────────
  1. QKV write (all tokens)        copy decode           (waiting)
     (overlaps with I/O)           history → GPU
                                   scratchpad
  2a. Prefill attention
      (local data only)
      ──signal prefill_ready──►                          ◄── wait for
                                                             prefill_ready
  ──wait for h2d_ready──►◄──      ──signal h2d_ready──►  copy prefill
                                                         blocks → CPU
  2b. Decode attention
      (uses scratchpad +
       resident blocks)
      ──signal compute_done──►
  3. MLP
```

### Event Dependencies

| Event | Producer | Consumer | Purpose |
|-------|----------|----------|---------|
| `h2d_ready[i]` | H2D stream | Compute stream | Decode history for layer `i` is in scratchpad; compute can run decode attention |
| `prefill_ready[i]` | Compute stream | D2H stream | Prefill QKV write is done; D2H can copy blocks to CPU |
| `compute_done[i]` | Compute stream | H2D stream | Decode attention done, scratchpad free; H2D for layer `i+2` (same parity) can overwrite. H2D waits on `compute_done[i-3]` |

### Why This Works

1. **QKV projection** is compute-bound — H2D streams decode history in parallel
2. **Prefill attention** uses only local (just-written) data — no H2D dependency
3. **Dual scratchpads** ensure no conflicts: compute reads scratchpad A while H2D writes scratchpad B
4. **D2H runs after prefill** signals readiness — newly written KV blocks go to CPU for future decode steps
5. The `i-3` offset on `compute_done` gives enough pipeline depth to avoid stalls

---

## Existing Files

### Core (kept)

| File | Purpose |
|------|---------|
| `include/paged_transformer.py` | `PagedTransformer` — fully resident paged KV cache transformer with mixed prefill/decode. Uses fused QKV, `ops.reshape_and_cache`, `flash_attn_varlen_func` (prefill), `ops.paged_attention_v2` (decode). **Base for dense_engine.py compute path.** |
| `include/paged_offload_transformer.py` | `PagedOffloadTransformer` — extends PagedTransformer with 3-stream CPU offloading. Dual scratchpads, event-driven H2D/D2H pipeline. **Reference implementation for future offloading in dense_engine.py.** |
| `include/transformer_common.py` | `TransformerArgs` dataclass, `PositionalEncoding`, `compile_if_needed` wrapper |
| `include/misc.py` | `GPUProfiler` (pynvml-based monitoring), `suppress_all_output`, compilation utilities |
| `benchmark_paged_transformer.py` | Benchmark driver for paged transformer variants. Mixed prefill/decode batches, JSON config support, profiling, bandwidth calculations |

### Legacy / Reference

| File | Purpose |
|------|---------|
| `include/resident_transfomer.py` | Baseline fully-resident transformer (no paging, uses `flex_attention`) |
| `include/kv_offload_transformer.py` | Alternative KV offloading with `flex_attention` block masks (non-paged) |
| `include/split_offloaded_transformer.py` | Row-wise weight + KV splitting with `merge_attention_states` |
| `transformer.py` | Legacy benchmark driver for split/KV offload variants |
| `offload_kv_attn.py` | Standalone KV offloading attention benchmark with online softmax merge |
| `offload_mm.py` | Row-wise concurrent GEMM offloading benchmark |
| `offload_ffn.py` | Pipelined MLP offloading with main/secondary offload regions |

### CUDA Kernels

| File | Purpose |
|------|---------|
| `cuda/gemm_row_split.cu` | Row-wise split GEMM benchmarks with cuBLAS |
| `cuda/memory_offload.cu` | Memory movement strategies (explicit overlap, UVM, zero-copy) |
| `cuda/mlp.cu` | Custom MLP kernel implementations |
| `cuda/test.cu` | Basic kernel tests |

### Scratch / Exploration

| File | Purpose |
|------|---------|
| `scratch_examples/merged_kv_attn.py` | Attention merging with LogSumExp |
| `scratch_examples/offload_paged_attn.py` | Paged attention exploration |
| `scratch_examples/test_paged_io.py` | I/O testing harness |
| `scratch_examples/vllm_paged_attn.py` | Reference vLLM paged attention implementation |
