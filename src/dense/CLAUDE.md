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

- **GPU**: 1x NVIDIA GH200 480GB (SM 9.0), aarch64 (Grace CPU), Ubuntu 22.04, glibc 2.35, CUDA 12.8
- **Container**: All PyTorch/vLLM scripts must run via `GH_GB200_env/pytorch_ngc.sh`
  - No args = interactive shell; with args = runs command and exits
  - Example: `bash GH_GB200_env/pytorch_ngc.sh "cd /workspace/GPU_Offloading/src/dense && python script.py"`
- **vLLM** 0.17.1rc1 in container (native `moe_align_block_size` CUDA kernel — no glibc patches needed)

---

## Directory Map

See [README.md](README.md) for project overview, offloading design, and file index.

| Path | What it covers |
|------|----------------|
| [README.md](README.md) | Project overview, dense_engine.py plan, KV offloading design, file index |

### For Subagents

- **Project overview or offloading design**: Read [README.md](README.md)
- **MoE engine (reference for dense_engine.py)**: Read `../MoE/moe_engine.py`
- **Paged transformer (base compute path)**: Read `include/paged_transformer.py`
- **KV offloading prototype**: Read `include/paged_offload_transformer.py`
- **Benchmarking**: Read `benchmark_paged_transformer.py`

---

## Key Pitfalls

- `VLLM_ENABLE_V1_MULTIPROCESSING=0` must be set **before** any vLLM import
- V tensor after QKV split is non-contiguous → needs `.contiguous()` or `.reshape()`
- Q/K norm behavior is **model-dependent**: OLMoE uses flat `[B, hidden_dim]` with weight `[H]` before head reshape; **Qwen3** uses per-head `[B, num_heads, head_dim]` with weight `[head_dim]` after reshaping. See `DENSE_ENGINE_PLAN.md` §1.7a for the Qwen3 pattern.
- FlashInfer RoPE JIT needs `ninja` in PATH (prefill only)
- torch.compile Inductor produces numerically different greedy tokens from eager (expected)
- **CUDA graph static buffer GC**: ALL tensors captured by a graph must be kept alive; save in the graph info dict
- **FlashInfer `plan()` before every decode replay**: `_plan_info` tile counts depend on actual page count, not just buffer contents
- **`_seq_lens_cpu += 1` BEFORE `plan()`**: FlashInfer must include the current token's K/V
- **FlashInfer prefill incompatible with multi-graph**: `fmha_varlen_plan()` allocates fresh GPU tensors per call; use vLLM FA3 instead
- Eager mode is NOT for real experiments — only for sanity checks in test files

---

## Architecture Notes

### dense_engine.py (planned)

Analogous to `../MoE/moe_engine.py` but for standard dense transformer models:
- Parses HuggingFace dense models (Llama, Mistral, etc.) from config.json + safetensors
- Standard dense MLP (fused gate_up + SiLU + down) instead of routed MoE experts
- GQA support (num_kv_heads != num_heads)
- Same kernel suite: FlashInfer decode, FA3 prefill, reshape_and_cache_flash
- Same compilation strategy: torch.compile + CUDA graphs

### KV cache offloading

Prototype in `include/paged_offload_transformer.py`. Three-stream pipeline:
- **Compute stream**: QKV, attention, MLP
- **H2D stream**: CPU → GPU decode history blocks
- **D2H stream**: GPU → CPU newly-written prefill blocks

Dual scratchpads alternate by layer parity. Event-driven synchronization
(`h2d_ready`, `prefill_ready`, `compute_done`). Layer 0 fully resident.
See [README.md](README.md) for full pipeline diagram.
