# Benchmarks

Performance benchmarks for the MoE engine's offloading infrastructure.
All scripts require an H100 GPU with models downloaded to `models/`.
Run from `src/MoE/`.

## Scripts

### `bench_comprehensive.py` — Per-layer kernel timing sweep

Full sweep across 3 models (OLMoE, Mixtral-20L, Mixtral-32L) x multiple
`experts_per_layer` values x batch sizes (1, 16, 32). Measures per-layer
stage4b CUDA graph replay time with demand loading active. Proves that
compute kernel performance is unchanged between all-resident and offloading
configurations (parity checks). Primary benchmark for data in
[offload_1GPU.md](../offload_1GPU.md).

```bash
# Single config
python benchmarks/bench_comprehensive.py --model models/Mixtral-8x7B-20L --experts-per-layer 4 --batch 1

# Full sweep for one model
python benchmarks/bench_comprehensive.py --model models/Mixtral-8x7B-20L --sweep

# All 3 models
python benchmarks/bench_comprehensive.py --sweep-all

# Save results
python benchmarks/bench_comprehensive.py --sweep-all --output results.json
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

Sweeps expert budgets via `configure()` and measures end-to-end wall-clock
time for multiple prefill configurations (1x128, 1x256, 1x512, 4x32) and
mixed decode+prefill configurations (16D, 32D, 1D+1x128P, 16D+1x256P, etc.).
Reports latency and transfer statistics per configuration. Primary benchmark
for prefill and mixed batch offloading data in [offload_1GPU.md](../offload_1GPU.md).

```bash
python benchmarks/bench_offload_prefill_mixed.py --model models/Mixtral-8x7B-20L
python benchmarks/bench_offload_prefill_mixed.py --model models/OLMoE-1B-7B
python benchmarks/bench_offload_prefill_mixed.py --model models/Mixtral-8x7B
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--n-warmup` | 3 | Warmup trials |
| `--n-trials` | 10 | Timing trials per config |
| `--output` | (none) | Save results to JSON |

**Output**: Per-config latency table (ms) grouped by budget, transfer stats (count, MB, ms), summary matrix.

## Also see

- [vLLM_comparison/](../tests/vLLM_comparison/) — vLLM vs custom engine benchmarks (decode, prefill, mixed batch)
- [profiling/](../profiling/) — Nsight Systems and per-phase kernel profiling
