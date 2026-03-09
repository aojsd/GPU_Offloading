# Tests

Unit and correctness tests for the MoE engine. All `test_*.py` files support
`pytest` and can also be run standalone. GPU tests require an H100 with models
downloaded to `models/`.

Run from `src/MoE/`:

```bash
# CPU-only tests (no GPU needed)
python -m pytest tests/test_replay_policy.py -v
python -m pytest tests/test_trace_construction.py -v

# GPU correctness tests
python tests/test_split_stage4.py --model models/Mixtral-8x7B-20L
python tests/test_piecewise_prefill.py --model models/Mixtral-8x7B-20L
python tests/test_offload_correctness.py --model models/Mixtral-8x7B-20L
python tests/test_pipeline_parallel.py
```

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

#### `test_trace_construction.py` — Continuous batching simulator (23 tests)

CPU-only tests using synthetic per-conversation traces. Covers page
accounting, LIFO preemption, memory budget, concurrent execution,
ActivationTrace conversion, and scheduling metadata.

### GPU correctness

#### `test_split_stage4.py` — Stage4 split validation

Verifies that splitting stage4 into 4a (router) + 4b (MoE compute) with a
CPU break produces numerically identical logits to the unsplit flat CUDA
graph. Tests both prefill and decode paths.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--no-compile` | off | Disable torch.compile (for exact comparison) |

#### `test_piecewise_prefill.py` — Piecewise prefill correctness

Three tests: (1) flat vs piecewise prefill greedy tokens match when all
experts are resident, (2) partial offloading prefill smoke test (no crash,
no NaN), (3) partial offloading latency overhead measurement.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--seq-len` | 128 | Sequence length to test |

#### `test_offload_correctness.py` — Offloaded decode correctness

Three tests: (1) split stage4 latency overhead vs flat, (2) demand-loaded
experts produce same logits/tokens as all-resident, (3) multi-step decode
with offloading matches non-offloaded across multiple random seeds.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--no-compile` | off | Disable torch.compile |

#### `test_pipeline_parallel.py` — Pipeline parallelism correctness

Four tests: (1) PP=2 logits match single-GPU offloaded, (2) performance
vs single-GPU baseline, (3) trace collection produces valid structure,
(4) memory allocation across GPUs.

### GPU verification (standalone scripts)

#### `verify_expert_residency.py` — Demand loading validation

Hardcoded to Mixtral-8x7B with `experts_per_layer=2`. Logs per-layer routing
decisions across 5 decode steps, times each stage4b replay, and validates that
timing is uniform (~329 us/layer) regardless of which experts the router
selects — confirming all non-resident experts are demand-loaded before compute.

```bash
python tests/verify_expert_residency.py
```

#### `verify_unified_cache.py` — Unified cache smoke + correctness

Test 1 (32L): Loads with `experts_per_layer=4`, validates buffer structure
(per-layer views share storage, expert maps consistent), runs prefill + 5
decode steps, checks no NaN/Inf. Test 2 (20L): Compares
`experts_per_layer=8` (all resident via unified cache path) vs no
offloading — greedy tokens must match exactly.

```bash
python tests/verify_unified_cache.py
python tests/verify_unified_cache.py --skip-32L
```

## Also see

- [benchmarks/](../benchmarks/) — Performance benchmarks (kernel timing, e2e latency)
- [profiling/](../profiling/) — Nsight Systems and per-phase kernel profiling
- [scripts/](../scripts/) — Experiment runners
- [vLLM_comparison/](../vLLM_comparison/) — Custom engine vs vLLM benchmarks

## Models

Scripts expect models in `src/MoE/models/`:

| Model | Path | Notes |
|-------|------|-------|
| OLMoE-1B-7B | `models/OLMoE-1B-7B` | 16L, 64E (top-8), ~13.8 GB |
| Mixtral-8x7B-20L | `models/Mixtral-8x7B-20L` | 20L, 8E (top-2), ~54.6 GB. Truncated. |
| Mixtral-8x7B | `models/Mixtral-8x7B` | 32L, 8E (top-2), ~86 GB. Requires offloading. |

Download: `bash models/download.sh`. Truncate: `python models/truncate_model.py --layers 20`.
