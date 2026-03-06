## Meta-Instructions

After reasonable checkpoints, or when finding a critical bug, summarize important
ideas and changes and append them to the relevant .md file. Alternatively, do
this whenever you have been working for a long time and may soon exceed the 200k
context limit. When a major progression is finished, refactor the relevant .md
file to remove intermediate checkpoint information.

Aim to write most, if not all, code in Python/PyTorch or PyTorch-compatible packages.

For PyTorch code, we are utterly uninterested in completely eager execution when
it comes to real experiments or any test that is performance related.

---

## Environment

- **GPUs**: 2x NVIDIA H100 80GB HBM3 (SM90), **CUDA** 12.8, **System**: RHEL 8.10 (glibc 2.28)
- **Python** 3.13.5, **PyTorch** 2.9.0+cu128, **Triton** 3.5.0
- **vLLM** 0.11.2 (pip prebuilt wheel; 0.12+ requires glibc 2.31)

---

## Directory Map

See [README.md](README.md) for full project overview, sandbox design, engine API,
and end-to-end usage.

| Path | What it covers |
|------|----------------|
| [README.md](README.md) | Project overview, sandbox design, models, engine API, performance summary, file index |
| [offload_1GPU.md](offload_1GPU.md) | Expert offloading research notes, Phases 3-4, memory analysis, benchmarks |
| [pipeline_parallelism.md](pipeline_parallelism.md) | PP=2 performance analysis, per-layer breakdown, optimization targets |
| [replay.md](replay.md) | Cache simulation & replay: trace formats, policy simulators, replay controller, prefetch/eviction timing |
| [vLLM_comparison/README.md](vLLM_comparison/README.md) | Decode, prefill, mixed batch benchmarks vs vLLM; profiling data |
| [tests/README.md](tests/README.md) | Index of all test/benchmark scripts |
| [trace_construction/README.md](trace_construction/README.md) | Trace collection pipeline: per-conversation GPU traces, batched CPU simulator |

### For Subagents

- **Engine internals or project overview**: Read [README.md](README.md)
- **Benchmarking vs vLLM**: Read [vLLM_comparison/README.md](vLLM_comparison/README.md)
- **Offloading design & memory analysis**: Read [offload_1GPU.md](offload_1GPU.md)
- **Replay/policy simulation**: Read [replay.md](replay.md)
- **Trace collection & batching**: Read [trace_construction/README.md](trace_construction/README.md)
- **Tests & benchmarks**: Read [tests/README.md](tests/README.md)
- **Pipeline parallelism**: Read [pipeline_parallelism.md](pipeline_parallelism.md)

---

## Key Pitfalls

- `VLLM_ENABLE_V1_MULTIPROCESSING=0` must be set **before** any vLLM import
- glibc 2.28: three ops monkey-patched in `moe_engine.py` (`moe_align_block_size`, `moe_sum`, `topk_softmax`); `sorted_ids` are FLAT indices into `topk_ids.flatten()`
- Q/K norm on flat `[B, 2048]` **before** head reshape, not per-head
- norm_topk_prob: model-specific (OLMoE=False, Mixtral=True) — read from config.json
- V tensor after QKV split is non-contiguous → needs `.contiguous()` or `.reshape()`
- FlashInfer RoPE JIT needs `ninja` in PATH (prefill only)
- torch.compile Inductor produces numerically different greedy tokens from eager (expected); no-compile matches vLLM at 80.4%, with compile only 3.8%
- **CUDA graph static buffer GC**: ALL tensors captured by a graph must be kept alive; save in the graph info dict
- **FlashInfer `plan()` before every decode replay**: `_plan_info` tile counts depend on actual page count, not just buffer contents
- **`_seq_lens_cpu += 1` BEFORE `plan()`**: FlashInfer must include the current token's K/V
- **FlashInfer prefill incompatible with multi-graph**: `fmha_varlen_plan()` allocates fresh GPU tensors per call; use vLLM FA3 instead
- Eager mode is NOT for real experiments — only for sanity checks in test files
