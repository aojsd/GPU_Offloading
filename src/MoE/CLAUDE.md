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
After every major progression, summarize important ideas and changes to be appended to the end of this CLAUDE.md file. Alternatively, do this whenever you have been working for a long time and may soon exceed the 200k context limit.

Aim to write most, if not all, code in Python/Pytorch or Pytorch-compatible packages.

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

### Implementation Phases

1. **Core Compute Engine** (complete): OLMoE-1B-7B custom engine with CUDA graph decode, validated against vLLM SOTA.
2. **Expert Cache & Async Transfer**: Double-buffered GPU expert slots, CPU pinned memory, three CUDA streams.
3. **Trace Format & Replay Controller**: Replay loop with data movement orchestration + timing.
4. **Trace Collection**: Hook router to record expert selections per layer per token.
5. **Policy Simulation**: LRU, Oracle/Belady, frequency-based, pre-gated policies.

---

## Custom MoE Engine

Model-agnostic engine supporting any HuggingFace MoE model with gate/up/down expert
projections. Architecture params read from `config.json`. Currently validated on OLMoE-1B-7B;
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
logits = engine.prefill(input_ids)          # [B, S, vocab]
logits = engine.decode_step(token_ids, pos) # [B, vocab]  (CUDA graph auto-dispatch)
tokens = engine.generate(input_ids, max_new_tokens=128)
engine.capture_decode_cuda_graph(batch_size=1, warmup_seq_len=128, max_decode_tokens=256)
```

### Correctness

- Greedy generation: exact match vs HuggingFace (37/37 tokens, logits cosine=0.9999)
- CUDA graph vs eager: exact match (30 tokens)

### CUDA Graph Decode Pipeline

Per layer: fused QKV projection → Q/K RMSNorm → rope_pytorch → reshape_and_cache_flash →
flash_attn_varlen_func → output projection → RMSNorm + residual → router → fused_experts → residual.

With `torch.compile(fullgraph=False)`, Inductor fuses RMSNorm + residual + RoPE into Triton
kernels. Graph breaks at `reshape_and_cache_flash` and `flash_attn_varlen_func` (no fake impls)
are acceptable — Inductor fuses the regions between breaks. `fused_experts` has proper
`torch.library` registration with fake impl → no graph break.

`static_seq_lens` and `static_slot_mapping` tensors are updated before each graph replay via `.copy_()`.

Key implementation details:
- KV cache format: flat NHD `[L, blocks, block_size, num_kv_heads, head_dim]`
- Decode attention: `flash_attn_varlen_func` from `vllm.vllm_flash_attn` (FA2/FA3, same as vLLM v1)
- KV write: `reshape_and_cache_flash` (flat NHD, simpler than x-packed `reshape_and_cache`)
- RoPE: `rope_pytorch` for decode (pure PyTorch, compilable), FlashInfer for prefill
- torch.compile Inductor produces numerically different tokens from eager (expected, vLLM does too)
- `use_torch_compile=False` for exact correctness validation, `True` for performance

### glibc 2.28 Workaround

vLLM's `_moe_C.abi3.so` requires glibc 2.29. Three ops monkey-patched in `moe_engine.py`:
`moe_align_block_size` (vectorized, no `.item()`), `moe_sum`, `topk_softmax`.
All CUDA-graph-capturable. Key detail: sorted_ids are **flat indices** into `topk_ids.flatten()`;
expert_ids via `searchsorted(..., right=True)`.

### vLLM Baseline Setup

- `VLLM_ENABLE_V1_MULTIPROCESSING=0` required (subprocess won't have patches)
- Import `moe_engine` **before** creating `LLM(...)` to apply patches
- Step-by-step timing: `llm.llm_engine.add_request()` + `llm.llm_engine.step()` loop

---

## Performance vs vLLM (H100 80GB, batch=1, CUDA graphs + torch.compile)

### GPU Kernel Time — Nsight Systems (ms/step)

Profiled with `nsys profile --cuda-graph-trace=node`. 30 decode steps per run.
Custom engine uses torch.compile + CUDA graphs (same as vLLM).

```
seq_len  Custom  vLLM   Gap      Gap%   Custom launches  vLLM launches
  128     2.87   2.92  -0.05     -1.6%          974           1032
  256     2.92   2.93  -0.02     -0.5%          974           1032
  512     2.99   2.94  +0.04     +1.5%          974           1032
 1024     3.15   2.95  +0.20     +6.6%          974           1032
 2048     3.47   3.01  +0.46    +15.1%          974           1032
```

At short sequences (128-512), custom engine matches or beats vLLM.
At longer sequences, FA2 splitKV attention scales worse than vLLM's FlashInfer decode kernel.
Custom uses fewer kernel launches (974 vs 1032) due to Inductor fusion.

### Kernel Category Breakdown at seq_len=128

```
Category                        Custom     vLLM     Delta    Notes
MoE (fused_experts)              730us     696us    +34us    noise
Elementwise                      643us     741us    -98us    custom has more Inductor fusion
Linear (cuBLAS)                  438us     420us    +19us    near parity
KV cache indexing                181us     171us    +10us    near parity
MoE align (patched)              180us     170us    +10us    noise
Router (sort)                    139us     133us     +6us    noise
Attention (FlashAttention)       137us       —         —     custom uses FA2 splitKV
Attention (FlashInfer graph)       —       198us       —     vLLM uses FlashInfer
torch.compile fused              135us     126us     +9us    both use Inductor
KV cache store                    91us      37us    +54us    reshape_and_cache_flash vs vLLM
Router (topk)                     84us      79us     +5us    noise
Reduce / sum                      82us      90us     -8us    noise
MoE (SiLU gate)                   29us      31us     -2us    noise
                                 -----    -----    -----
TOTAL                           2.87ms   2.92ms   -0.05ms
```

Custom is 50us faster overall at seq128. Main differences:
- Attention: FA2 (137us) vs FlashInfer (198us) — FA2 is 31% faster at short sequences
- KV store: custom 91us vs vLLM 37us (+54us) — reshape_and_cache_flash overhead
- Elementwise: custom 643us vs vLLM 741us (-98us) — more aggressive Inductor fusion

### Attention Scaling Behavior

```
seq_len   Custom FA2    vLLM FlashInfer    Delta
  128       137us          198us           -61us  (FA2 faster)
  256       178us          201us           -23us
  512       257us          212us           +45us  (FlashInfer faster)
 1024       414us          230us          +184us
 2048       732us          274us          +458us
```

FA2 splitKV decode scales O(seq_len) while FlashInfer has better constant factors at long
sequences. This is the dominant source of the gap at seq_len >= 512. Could be addressed by
switching to FlashInfer BatchDecode (requires plan() outside CUDA graph) or FA3 (full
CUDA graph support with scheduler_metadata).

### Bandwidth Analysis (batch=1, top-8 of 64 experts)

```
Non-attention data read per step: ~2.29 GB (constant)
  Attention (QKV+O): 512 MB, MoE w1: 1024 MB, MoE w2: 512 MB,
  lm_head: 206 MB, Router: 4 MB

KV cache read per step: 128 KB x seq_len (all 16 layers x K+V)
  seq_len=  128:     0.02 GB      seq_len= 32,768:    4.00 GB
  seq_len= 4,096:    0.50 GB      seq_len=131,072:   16.00 GB

At short seq_lens: MoE dominates (8/64 experts -> small data, many kernels)
At long seq_lens: attention KV reads dominate
```

---

## Key Pitfalls

- Q/K norm on flat `[B, 2048]` **before** head reshape, not per-head
- w1 gate/up ordering: gate first (rows 0:1024), up second (rows 1024:2048)
- norm_topk_prob=False: do NOT renormalize routing weights
- `moe_align_block_size`: sorted_ids = FLAT indices; Triton kernel does `flat_idx // top_k`
- `flash_attn_varlen_func`: use `cu_seqlens_q=[0,1,...,B]`, `max_seqlen_q=1`, `seqused_k=seq_lens`
- `reshape_and_cache_flash` + `flash_attn_varlen_func` lack fake impls → graph breaks with `fullgraph=False`
- `fused_experts` has proper `torch.library` registration with fake impl → no graph break
- torch.compile Inductor produces numerically different greedy tokens from eager (expected behavior)
- V tensor after QKV split is non-contiguous -> needs `.contiguous()` or `.reshape()`
- FlashInfer RoPE JIT needs `ninja` in PATH (used for prefill only)
- `slot_mapping` for reshape_and_cache must be computed outside CUDA graph and copied to static buffer

---

## Files

| File | Purpose |
|------|---------|
| `moe_engine.py` | `MoEEngine` — model-agnostic engine with CUDA graph support |
| `test_cuda_graph.py` | CUDA graph correctness (greedy gen match) + timing |
| `benchmark.py` | Merged correctness validation + performance benchmark (custom vs vLLM) |
| `nsys_seq_len_comparison.py` | Nsight Systems per-seq-len profiling: Custom vs vLLM |
| `profiling/<model>/` | Per-seq-len kernel profiles (`seq{N}.prof`) |
| `models/OLMoE-1B-7B/` | Model weights + config |

Profiling output: `profiling/<model_name>/seq{N}.prof` (side-by-side kernel comparison, custom vs vLLM)
Regenerate: `python src/MoE/nsys_seq_len_comparison.py all [--model path/to/model]`

---
