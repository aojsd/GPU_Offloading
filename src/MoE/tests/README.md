# MoE Engine Tests & Benchmarks

All scripts run from `src/MoE/` and require an H100 GPU with models downloaded to `models/`.

## Correctness Tests

### `test_split_stage4.py` — Stage4 split validation

Verifies that splitting stage4 into 4a (router) + 4b (MoE compute) with a CPU break
produces numerically identical logits to the unsplit flat CUDA graph. Tests both prefill
and decode paths.

```bash
python tests/test_split_stage4.py --model models/Mixtral-8x7B-20L
python tests/test_split_stage4.py --model models/Mixtral-8x7B-20L --no-compile  # exact comparison
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--no-compile` | off | Disable torch.compile (eliminates Inductor noise for exact match) |

**Output**: Max/mean logit diffs, top-1 token match (PASS/MISMATCH).

### `test_piecewise_prefill.py` — Piecewise prefill correctness

Three tests: (1) flat vs piecewise prefill greedy tokens match when all experts are
resident, (2) partial offloading prefill smoke test (no crash, no NaN), (3) partial
offloading latency overhead measurement.

```bash
python tests/test_piecewise_prefill.py --model models/Mixtral-8x7B-20L
python tests/test_piecewise_prefill.py --model models/OLMoE-1B-7B --seq-len 256
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--seq-len` | 128 | Sequence length to test |

**Output**: Token match (first 10 printed), latency comparison (ms), transfer stats for partial offloading.

### `test_offload_correctness.py` — Offloaded decode correctness

Three tests: (1) split stage4 latency overhead vs flat, (2) demand-loaded experts
produce same logits/tokens as all-resident, (3) multi-step decode with offloading
matches non-offloaded across multiple random seeds.

```bash
python tests/test_offload_correctness.py --model models/Mixtral-8x7B-20L
python tests/test_offload_correctness.py --model models/Mixtral-8x7B-20L --no-compile
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--no-compile` | off | Disable torch.compile |

**Output**: Overhead %, logit diffs, token match PASS/FAIL per test.

### `verify_unified_cache.py` — Unified cache smoke + correctness

Test 1 (32L): Loads with `experts_per_layer=4`, validates buffer structure (per-layer
views share storage, expert maps consistent), runs prefill + 5 decode steps, checks
no NaN/Inf. Test 2 (20L): Compares `experts_per_layer=8` (all resident via unified
cache path) vs no offloading — greedy tokens must match exactly.

```bash
python tests/verify_unified_cache.py
python tests/verify_unified_cache.py --skip-32L                         # only 20L correctness
python tests/verify_unified_cache.py --model-20L models/Mixtral-8x7B-20L --skip-32L
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model-32L` | `src/MoE/models/Mixtral-8x7B` | Path to 32L Mixtral model |
| `--model-20L` | `src/MoE/models/Mixtral-8x7B-20L` | Path to 20L Mixtral model |
| `--no-compile` | off | Disable torch.compile |
| `--skip-32L` | off | Skip 32L smoke test |
| `--skip-20L` | off | Skip 20L correctness test |

**Output**: Buffer structure checks, token sequences, SMOKE TEST PASSED / CORRECTNESS PASSED.

### `verify_expert_residency.py` — Demand loading validation

Hardcoded to Mixtral-8x7B with `experts_per_layer=2`. Logs per-layer routing decisions
across 5 decode steps, times each stage4b replay, and validates that timing is uniform
(~329 us/layer) regardless of which experts the router selects — confirming all
non-resident experts are demand-loaded before compute.

```bash
python tests/verify_expert_residency.py
```

No arguments — uses hardcoded Mixtral-8x7B 32L, `experts_per_layer=2`, `torch.compile=False`.

**Output**: Expert maps, per-layer routing log with timing, uniformity check (PASS if max deviation < 50 us).

## Benchmarks

### `bench_comprehensive.py` — Main per-layer kernel benchmark

Full sweep across 3 models (OLMoE, Mixtral-20L, Mixtral-32L) x multiple `experts_per_layer`
values x batch sizes (1, 16, 32). Measures per-layer stage4b CUDA graph replay time with
demand loading active. Proves that compute kernel performance is unchanged between
all-resident and offloading configurations (parity checks). This is the primary
benchmark used to generate data for `offload_1GPU.md`.

```bash
# Single config
python tests/bench_comprehensive.py --model models/Mixtral-8x7B-20L --experts-per-layer 4 --batch 1

# Full sweep for one model
python tests/bench_comprehensive.py --model models/Mixtral-8x7B-20L --sweep

# All 3 models
python tests/bench_comprehensive.py --sweep-all

# Save results
python tests/bench_comprehensive.py --sweep-all --output results.json
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | (required for single) | Path to model directory |
| `--experts-per-layer` | (auto) | Experts per layer for single config |
| `--batch` | 1 | Decode batch size |
| `--n-steps` | 10 | Decode steps to measure |
| `--no-compile` | off | Disable torch.compile |
| `--sweep` | off | Full sweep for one model |
| `--sweep-all` | off | Full sweep for all models |
| `--output` | (none) | Save results to JSON |

**Output**: Per-layer timing table (us), median/min/max, cache miss count, transfer stats, parity checks.

### `bench_offload_prefill_mixed.py` — Prefill & mixed batch e2e latency

Sweeps expert budgets via `configure()` and measures end-to-end wall-clock time for
multiple prefill configurations (1x128, 1x256, 1x512, 4x32) and mixed decode+prefill
configurations (16D, 32D, 1D+1x128P, 16D+1x256P, etc.). Reports latency and transfer
statistics per configuration. This is the primary benchmark for prefill and mixed batch
offloading data in `offload_1GPU.md`.

```bash
python tests/bench_offload_prefill_mixed.py --model models/Mixtral-8x7B-20L
python tests/bench_offload_prefill_mixed.py --model models/OLMoE-1B-7B
python tests/bench_offload_prefill_mixed.py --model models/Mixtral-8x7B
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--n-warmup` | 3 | Warmup trials |
| `--n-trials` | 10 | Timing trials per config |
| `--output` | (none) | Save results to JSON |

**Output**: Per-config latency table (ms) grouped by budget, transfer stats (count, MB, ms), summary matrix.

## Profiling

### `nsys_piecewise_decode.py` — Nsight Systems profiling

Lightweight harness that emits NVTX ranges around each decode step for capture by
`nsys profile`. Useful for visualizing per-layer and per-stage timing breakdowns
in Nsight Systems.

```bash
nsys profile -c cudaProfilerApi -o decode_profile \
    python tests/nsys_piecewise_decode.py --model models/Mixtral-8x7B-20L --experts-per-layer 4
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | (required) | Path to model directory |
| `--experts-per-layer` | None | Experts per layer (None = all resident) |
| `--n-steps` | 10 | Decode steps to profile |
| `--n-warmup` | 5 | Warmup steps before profiling |

**Output**: NVTX ranges in nsys trace. Console prints step count.

## Models

Scripts expect models in `src/MoE/models/`:

| Model | Path | Notes |
|-------|------|-------|
| OLMoE-1B-7B | `models/OLMoE-1B-7B` | 16L, 64E (top-8), ~13.8 GB. Fits on one H100. |
| Mixtral-8x7B-20L | `models/Mixtral-8x7B-20L` | 20L, 8E (top-2), ~54.6 GB. Truncated via `models/truncate_model.py`. |
| Mixtral-8x7B | `models/Mixtral-8x7B` | 32L, 8E (top-2), ~86 GB. Requires offloading (`experts_per_layer < 8`). |

Download: `bash models/download.sh`. Truncate: `python models/truncate_model.py --layers 20`.

## Quick Start

```bash
cd src/MoE

# Correctness: all tests
python tests/test_split_stage4.py
python tests/test_piecewise_prefill.py
python tests/test_offload_correctness.py
python tests/verify_unified_cache.py

# Benchmark: per-layer kernel timing sweep
python tests/bench_comprehensive.py --model models/Mixtral-8x7B-20L --sweep

# Benchmark: prefill + mixed batch e2e
python tests/bench_offload_prefill_mixed.py --model models/Mixtral-8x7B-20L

# Profile with nsys
nsys profile -c cudaProfilerApi python tests/nsys_piecewise_decode.py --model models/Mixtral-8x7B-20L
```
