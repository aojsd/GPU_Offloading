# Profiling

Nsight Systems and CUDA event profiling harnesses for the MoE engine.
All scripts require an H100 GPU with models downloaded to `models/`.
Run from `src/MoE/`.

## Scripts

### `nsys_pp_decode.py` — Pipeline-parallel decode profiling

Three modes for comparing single-GPU vs PP=2 decode performance:
- `20L-single`: Mixtral-8x7B-20L on 1 GPU (baseline per-layer cost)
- `32L-single`: Mixtral-8x7B-32L on 1 GPU with offloading
- `32L-pp2`: Mixtral-8x7B-32L across 2 GPUs (pipeline parallel)

Provides both wall-clock timing and detailed per-layer/per-stage CUDA event
breakdown. Data feeds into [pipeline_parallelism.md](../pipeline_parallelism.md).

```bash
# Wall-clock comparison across modes
python profiling/nsys_pp_decode.py --mode all

# Single mode with nsys profiling
nsys profile -c cudaProfilerApi -o pp2_decode \
    python profiling/nsys_pp_decode.py --mode 32L-pp2 --n-steps 5
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--mode` | `all` | Mode: `20L-single`, `32L-single`, `32L-pp2`, or `all` |
| `--n-steps` | 20 | Decode steps to measure |
| `--n-warmup` | 5 | Warmup steps before profiling |
| `--compile` | off | Enable torch.compile |
| `--phase-timing` | off | Per-layer/stage CUDA event breakdown |

**Output**: Per-mode wall-clock timing, per-layer phase breakdown, comparison tables.

### `profile_phases.py` — Comprehensive per-phase kernel profiling

Four modes:
1. **nsys single-config**: Run under Nsight Systems with NVTX, parse sqlite
2. **CUDA-event sweep**: Sweep decode positions (128 to 262144) and prefill sequence lengths
3. **Analyze existing sqlite**: Parse pre-collected nsys data
4. **PCIe transfer measurement**: Expert CPU->GPU transfer bandwidth

Generates `profiling.md` with formatted kernel breakdown tables.

```bash
# CUDA event sweep (no nsys needed)
python profiling/profile_phases.py sweep --model models/Mixtral-8x7B-20L

# nsys capture + analysis
nsys profile -c cudaProfilerApi --capture-range=cudaProfilerApi -o profile \
    python profiling/profile_phases.py nsys --model models/Mixtral-8x7B-20L
python profiling/profile_phases.py analyze --sqlite profile.sqlite

# PCIe transfer measurement
python profiling/profile_phases.py transfer --model models/Mixtral-8x7B-20L
```

| Arg | Default | Description |
|-----|---------|-------------|
| subcommand | (required) | `nsys`, `sweep`, `analyze`, or `transfer` |
| `--model` | `models/Mixtral-8x7B-20L` | Path to model directory |
| `--experts-per-layer` | None | Experts per layer (None = all resident) |
| `--output` | (auto) | Output path for profiling.md |

**Output**: Per-phase timing tables, kernel category breakdowns, PCIe bandwidth measurements.

## Also see

- [vLLM_comparison/](../vLLM_comparison/) — `nsys_profiler.py` for custom vs vLLM kernel profiling
- [benchmarks/](../benchmarks/) — Performance benchmarks (latency, throughput)
