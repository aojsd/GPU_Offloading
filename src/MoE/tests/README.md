# Tests

Unit and correctness tests for the MoE engine. All `test_*.py` files support
`pytest` and can also be run standalone. GPU tests require H100(s) with models
downloaded to `models/`.

Run from `src/MoE/`:

```bash
# CPU-only tests (no GPU needed)
python -m pytest tests/test_replay_policy.py tests/test_scheduling.py tests/test_trace_construction.py -v

# GPU tests — OLMoE on 1 GPU
CUDA_VISIBLE_DEVICES=0 python tests/test_split_stage4.py --model models/OLMoE-1B-7B
CUDA_VISIBLE_DEVICES=0 python tests/test_piecewise_prefill.py --model models/OLMoE-1B-7B
CUDA_VISIBLE_DEVICES=0 python tests/test_offload_correctness.py --model models/OLMoE-1B-7B
CUDA_VISIBLE_DEVICES=0 python tests/test_scheduled_chunked_prefill.py --model models/OLMoE-1B-7B
CUDA_VISIBLE_DEVICES=0 python tests/test_chunked_prefill_comprehensive.py --model models/OLMoE-1B-7B
CUDA_VISIBLE_DEVICES=0 python tests/test_chunked_prefill_pp.py --model models/OLMoE-1B-7B --pp 1
CUDA_VISIBLE_DEVICES=0 python tests/test_dynamic_pages.py --model models/OLMoE-1B-7B
CUDA_VISIBLE_DEVICES=0 python tests/test_gpu_integration.py --model models/OLMoE-1B-7B

# GPU tests — Mixtral-8x7B PP=2 (2x H100)
python tests/test_split_stage4.py --model models/Mixtral-8x7B --pp 2
python tests/test_scheduled_chunked_prefill.py --model models/Mixtral-8x7B --pp 2
python tests/test_chunked_prefill_comprehensive.py --model models/Mixtral-8x7B --pp 2
python tests/test_chunked_prefill_pp.py --model models/Mixtral-8x7B --pp 2
python tests/test_dynamic_pages.py --model models/Mixtral-8x7B --pp 2
python tests/test_gpu_integration.py --model models/Mixtral-8x7B --pp 2
```

## Quick Reference: Which Tests Run Where

| Test | OLMoE 1-GPU | Mixtral PP=2 | Notes |
|------|:-----------:|:------------:|-------|
| `test_replay_policy.py` | CPU | CPU | No GPU needed |
| `test_scheduling.py` | CPU | CPU | No GPU needed |
| `test_trace_construction.py` | CPU | CPU | No GPU needed |
| `test_split_stage4.py` | `--pp 1` | `--pp 2` | PP>1: flat graphs N/A, tests piecewise determinism only |
| `test_scheduled_chunked_prefill.py` | `--pp 1` | `--pp 2` | |
| `test_chunked_prefill_comprehensive.py` | `--pp 1` | `--pp 2` | 200 ShareGPT prompts |
| `test_chunked_prefill_pp.py` | `--pp 1` | `--pp 2` | 200 ShareGPT prompts; originally PP-only, now general |
| `test_dynamic_pages.py` | `--pp 1` | `--pp 2` | |
| `test_gpu_integration.py` | `--pp 1` | `--pp 2` | Tests 9-10 skipped with PP (need offloading) |
| `test_piecewise_prefill.py` | yes | **no** | Tests 2-3 use `experts_per_layer` (offloading) |
| `test_offload_correctness.py` | yes | **no** | Tests 2-3 use `experts_per_layer` (offloading) |
| `test_pipeline_parallel.py` | **no** | 2x H100 | Compares PP=2 vs single-GPU offloaded; needs full Mixtral |
| `test_pcie_contention.py` | **no** | **no** | Hardcoded Mixtral-8x7B single-GPU offloading benchmark |
| `verify_expert_residency.py` | **no** | **no** | Hardcoded Mixtral-8x7B offloading validation |
| `verify_unified_cache.py` | **no** | **no** | Hardcoded Mixtral-8x7B offloading validation |

**Why some tests don't support PP:** The engine raises `ValueError` if both
`pipeline_parallel_size > 1` and `experts_per_layer`/`cache_size` are set — PP
and expert offloading are mutually exclusive (both solve fitting a large model,
by different means).

## Test Files

### CPU-only (no GPU)

#### `test_replay_policy.py` — Trace format & policy simulation (43 tests)

Covers trace serialization, validation, and all cache/prefetch policy
combinations for the unified expert cache.

| Category | Tests | What's verified |
|----------|-------|-----------------|
| Serialization | 4 | TransferEvent round-trip, cross-layer evict, GPUReplayTrace save/load |
| Validation | 5 | Valid trace passes, missing expert, overcapacity, free slot addition |
| ActivationTrace | 7 | from_flat_trace, save/load round-trip, empty trace, router inputs |
| LRU | 5 | No misses when all fit, eviction order, validate, cross-layer eviction |
| Belady-Oracle | 4 | Fewer misses than LRU, generates prefetches, prefetches in target layer |
| LFU | 3 | Evicts least frequent, windowed reset, validate passes |
| StaticFreq | 5 | Initial cache has most frequent, high-freq pinning, validate |
| Mix-and-Match | 4 | All 8 policy combos validate |
| Summary | 2 | Counts correct, Oracle prefetches in summary |
| Unified Cache | 3 | Variable experts per layer, zero-experts layer, cache pressure |

#### `test_scheduling.py` — Continuous batching scheduler (37+ tests)

CPU-only tests for the `Scheduler` class: slot allocation, preemption, EOS,
serialization, and trace construction.

#### `test_trace_construction.py` — Trace construction pipeline (23 tests)

CPU-only tests using synthetic per-conversation traces. Covers page
accounting, LIFO preemption, memory budget, concurrent execution,
ActivationTrace conversion, and scheduling metadata.

### GPU correctness

#### `test_split_stage4.py` — Stage4 split validation

Verifies that splitting stage4 into 4a (router) + 4b (MoE compute) with a
CPU break produces numerically identical logits to the unsplit flat CUDA
graph. With PP > 1, flat graphs are unavailable so the test verifies
piecewise determinism and decode consistency instead.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--pp` | 1 | Pipeline parallel size |
| `--no-compile` | off | Disable torch.compile (for exact comparison) |

```bash
# OLMoE 1 GPU
CUDA_VISIBLE_DEVICES=0 python tests/test_split_stage4.py --model models/OLMoE-1B-7B
# Mixtral PP=2
python tests/test_split_stage4.py --model models/Mixtral-8x7B --pp 2
```

#### `test_scheduled_chunked_prefill.py` — Chunked prefill correctness (4 tests)

Tests that prefilling a prompt via continuation chunks in `mixed_step()`
produces the same output as a single full prefill. Test 4 uses ShareGPT
prompts (loaded from the raw dataset, model-independent).

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--pp` | 1 | Pipeline parallel size |
| `--n-prompts` | 10 | Number of ShareGPT prompts for Test 4 |

```bash
# OLMoE 1 GPU
CUDA_VISIBLE_DEVICES=0 python tests/test_scheduled_chunked_prefill.py --model models/OLMoE-1B-7B
# Mixtral PP=2
python tests/test_scheduled_chunked_prefill.py --model models/Mixtral-8x7B --pp 2
```

#### `test_chunked_prefill_comprehensive.py` — 200-prompt chunked prefill sweep

For each ShareGPT prompt, compares a reference prefill (single-shot) against
a manual N-chunk split (first chunk via FA3, rest via FlashInfer paged-KV
continuation). The two attention kernels use different FP accumulation order,
so ~1/200 prompts may have a top-1 argmax flip when two tokens have nearly
identical logit values. **Expected: 196-200/200 top-1 matches, worst cosine
similarity > 0.99.** The test exits 0 regardless; the verdict line is
informational.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | (required) | Path to model directory |
| `--pp` | 1 | Pipeline parallel size |
| `--device` | `cuda:0` | Device for PP=1 runs |

```bash
# OLMoE 1 GPU
CUDA_VISIBLE_DEVICES=0 python tests/test_chunked_prefill_comprehensive.py --model models/OLMoE-1B-7B
# Mixtral PP=2
python tests/test_chunked_prefill_comprehensive.py --model models/Mixtral-8x7B --pp 2
```

#### `test_chunked_prefill_pp.py` — PP chunked prefill sweep

Same methodology and expected results as the comprehensive test (see note
on ~1/200 expected argmax flips above). Defaults to PP=2 and 200 prompts.
Supports any model/PP combination.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B` | Path to model directory |
| `--pp` | 2 | Pipeline parallel size |
| `--n-prompts` | 200 | Number of ShareGPT prompts |

```bash
# OLMoE 1 GPU
CUDA_VISIBLE_DEVICES=0 python tests/test_chunked_prefill_pp.py --model models/OLMoE-1B-7B --pp 1
# Mixtral PP=2
python tests/test_chunked_prefill_pp.py --model models/Mixtral-8x7B --pp 2
```

#### `test_dynamic_pages.py` — Dynamic page allocation correctness

Tests that dynamic page allocation (`_dynamic_pages=True`) produces
identical results to static allocation across prefill, decode, and
mixed batches.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/OLMoE-1B-7B` | Path to model directory |
| `--pp` | 1 | Pipeline parallel size |
| `--no-compile` | off | Disable torch.compile |

```bash
# OLMoE 1 GPU
CUDA_VISIBLE_DEVICES=0 python tests/test_dynamic_pages.py --model models/OLMoE-1B-7B
# Mixtral PP=2
python tests/test_dynamic_pages.py --model models/Mixtral-8x7B --pp 2
```

#### `test_gpu_integration.py` — End-to-end batched collection & replay (10 tests)

Full-pipeline GPU tests for `collect_batched()` continuous batching,
`Scheduler.collect()` trace collection, policy simulation, and offloaded
replay with real PCIe expert swapping. Tests 1-8 work with any model/PP.
Tests 9-10 require offloading (`--experts-per-layer`) and are skipped
when PP > 1.

| Test | What's verified |
|------|-----------------|
| 1. Single conversation | Chunked prefill + decode produces tokens |
| 2. Multi no-preempt | Multiple conversations complete without preemption |
| 3. Preemption | LIFO preemption + recompute prefill under memory pressure |
| 4. Page invariant | `pages_used == sum(ceil(seq_len/page_size))` every step |
| 5. Seq-len consistency | `active_state.seq_len` matches token count exactly |
| 6. EOS termination | Conversation stops early when model emits EOS |
| 7. Replay faithfulness (no preempt) | Collect → serialize → reload → replay: routing & tokens match |
| 8. Replay faithfulness (preempt) | Same as 7 with 12 convs, 15 preemptions |
| 9. Full pipeline (no preempt) | Collect → LRU-Oracle simulation → offloaded replay with ReplayController |
| 10. Full pipeline (preempt) | Same as 9 with preemptions and real PCIe expert swapping |

```bash
# OLMoE single GPU (~2 min)
CUDA_VISIBLE_DEVICES=0 python tests/test_gpu_integration.py --model models/OLMoE-1B-7B

# Mixtral PP=2 (2x H100, ~5 min)
python tests/test_gpu_integration.py --model models/Mixtral-8x7B --pp 2

# Mixtral single GPU with expert offloading (tests 9-10)
python tests/test_gpu_integration.py --model models/Mixtral-8x7B --experts-per-layer 4
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/OLMoE-1B-7B` | Path to model directory |
| `--pp` | 1 | Pipeline parallel size (1 or 2) |
| `--experts-per-layer` | None | Keep K experts/layer on GPU (mutually exclusive with `--pp > 1`) |
| `--no-compile` | off | Disable torch.compile |

#### `test_piecewise_prefill.py` — Piecewise prefill correctness (single-GPU only)

Three tests: (1) flat vs piecewise prefill greedy tokens match when all
experts are resident, (2) partial offloading prefill smoke test (no crash,
no NaN), (3) partial offloading latency overhead measurement. Tests 2-3
use `experts_per_layer` which is incompatible with PP.

**Not applicable to Mixtral PP=2.** Use `test_split_stage4.py --pp 2` for
piecewise validation with PP.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--seq-len` | 128 | Sequence length to test |

```bash
# OLMoE 1 GPU
CUDA_VISIBLE_DEVICES=0 python tests/test_piecewise_prefill.py --model models/OLMoE-1B-7B
```

#### `test_offload_correctness.py` — Offloaded decode correctness (single-GPU only)

Three tests: (1) split stage4 latency overhead vs flat, (2) demand-loaded
experts produce same logits/tokens as all-resident, (3) multi-step decode
with offloading matches non-offloaded across multiple random seeds.
All tests use `experts_per_layer` which is incompatible with PP.

**Not applicable to Mixtral PP=2.** Expert offloading and PP are mutually
exclusive.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--no-compile` | off | Disable torch.compile |

```bash
# OLMoE 1 GPU
CUDA_VISIBLE_DEVICES=0 python tests/test_offload_correctness.py --model models/OLMoE-1B-7B
```

#### `test_pipeline_parallel.py` — PP=2 vs single-GPU offloaded (Mixtral only)

Four tests: (1) PP=2 logits match single-GPU offloaded (epl=4), (2) performance
vs single-GPU baseline, (3) trace collection produces valid structure,
(4) memory allocation across GPUs. Requires the full Mixtral-8x7B (32L)
and 2x H100.

**Not applicable to OLMoE.** This test specifically compares PP=2 against
single-GPU offloaded, which requires a model too large for one GPU.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B` | Path to model directory |
| `--epl` | 4 | `experts_per_layer` for single-GPU reference |

```bash
# Mixtral 2x H100
python tests/test_pipeline_parallel.py --model models/Mixtral-8x7B
```

#### `test_pcie_contention.py` — PCIe bandwidth contention (Mixtral single-GPU only)

Hardcoded Mixtral-8x7B single-GPU offloading benchmark measuring PCIe
transfer overlap with MoE compute. No CLI args.

**Not applicable to OLMoE or PP=2.**

### GPU verification (standalone scripts)

#### `verify_expert_residency.py` — Demand loading validation

Hardcoded to Mixtral-8x7B with `experts_per_layer=2`. Logs per-layer routing
decisions across 5 decode steps, validates uniform timing.

**Not applicable to OLMoE or PP=2.**

#### `verify_unified_cache.py` — Unified cache smoke + correctness

Test 1 (32L): Loads with `experts_per_layer=4`, validates buffer structure.
Test 2 (20L): Compares all-resident unified cache vs no offloading.

**Not applicable to OLMoE or PP=2.**

## Also see

- [benchmarks/](../benchmarks/) — Performance benchmarks (kernel timing, e2e latency)
- [profiling/](../profiling/) — Nsight Systems and per-phase kernel profiling
- [scripts/](../scripts/) — Experiment runners
- [vLLM_comparison/](vLLM_comparison/) — Custom engine vs vLLM benchmarks

## Models

Scripts expect models in `src/MoE/models/`:

| Model | Path | Notes |
|-------|------|-------|
| OLMoE-1B-7B | `models/OLMoE-1B-7B` | 16L, 64E (top-8), ~13.8 GB |
| Mixtral-8x7B | `models/Mixtral-8x7B` | 32L, 8E (top-2), ~86 GB. Requires PP=2 or offloading. |

Download: `bash models/download.sh`.
