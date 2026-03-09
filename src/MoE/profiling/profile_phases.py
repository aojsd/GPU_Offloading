#!/usr/bin/env python3
"""Profile per-phase kernel timing for Mixtral-8x7B.

Modes:
  1. nsys single-config: Run under Nsight Systems with NVTX, parse sqlite
  2. CUDA-event sweep:   Sweep decode/prefill sizes, generate profiling.md
  3. Analyze existing sqlite file
  4. Measure expert PCIe transfer time

Usage:
  # nsys single-config profiling
  python tests/profile_phases.py --decode 128
  python tests/profile_phases.py --decode 4:2048 --prefill 2:128

  # CUDA-event sweep (generates results/MoE/mixtral-8x7B/profiling.md)
  python tests/profile_phases.py --sweep
  python tests/profile_phases.py --sweep --compile

  # Analyze existing sqlite
  python tests/profile_phases.py --analyze /path/to/file.sqlite --layers 20

  # Measure expert PCIe transfer time
  python tests/profile_phases.py --measure-transfer
"""

import argparse
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MOE_DIR = SCRIPT_DIR.parent
NSYS = os.path.expanduser("~/software/cuda-12.8/bin/nsys")
DEFAULT_MODEL = str(MOE_DIR / "models" / "Mixtral-8x7B-20L")
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(MOE_DIR)),
    "results", "MoE", "mixtral-8x7B")

STAGES = ['stage1', 'stage2', 'stage3', 'stage4a', 'stage4b']
NON_LAYER_PHASES = ['setup', 'embed', 'final']

# Sweep defaults
DECODE_POSITIONS = [2**i for i in range(7, 19)]  # 128 to 262144
PREFILL_SIZES = [128, 256, 512, 1024, 2048]
SWEEP_WARMUP = 5
SWEEP_STEPS = 20


# ═══════════════════════════════════════════════════════════════════
#  Batch parsing (same format as dense/benchmark_paged_transformer.py)
# ═══════════════════════════════════════════════════════════════════

def parse_batch_distribution(arg_list):
    """Parse batch spec: '4:2048' -> [2048]*4, '128' -> [128], '0' -> []."""
    if not arg_list:
        return []
    if len(arg_list) == 1 and arg_list[0].lower() in ('0', 'none'):
        return []
    result = []
    for item in arg_list:
        if ':' in item:
            count_str, len_str = item.split(':')
            result.extend([int(len_str)] * int(count_str))
        else:
            val = int(item)
            if val > 0:
                result.append(val)
    return result


def batch_description(decode_lengths, prefill_lengths):
    """Human-readable batch description."""
    parts = []
    if decode_lengths:
        if len(set(decode_lengths)) == 1:
            parts.append(f"{len(decode_lengths)} decode@{decode_lengths[0]}")
        else:
            parts.append(f"decode({','.join(str(l) for l in decode_lengths)})")
    if prefill_lengths:
        if len(set(prefill_lengths)) == 1:
            parts.append(f"{len(prefill_lengths)} prefill({prefill_lengths[0]})")
        else:
            parts.append(f"prefill({','.join(str(l) for l in prefill_lengths)})")
    return ' + '.join(parts) or 'empty'


def batch_filename(decode_lengths, prefill_lengths):
    """Short filename-safe batch description."""
    parts = []
    if decode_lengths:
        parts.append(f"d{len(decode_lengths)}x{decode_lengths[0]}")
    if prefill_lengths:
        parts.append(f"p{len(prefill_lengths)}x{prefill_lengths[0]}")
    return "_".join(parts) or "empty"


# ═══════════════════════════════════════════════════════════════════
#  Worker (runs under nsys)
# ═══════════════════════════════════════════════════════════════════

def run_worker(args):
    """Load engine, run profiled steps with NVTX. Called under nsys."""
    import ctypes
    ctypes.CDLL(
        "/gpfs/radev/apps/avx512/software/GCCcore/13.3.0/lib64/libstdc++.so.6",
        mode=ctypes.RTLD_GLOBAL)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(MOE_DIR))

    import torch
    from moe_engine import MoEEngine

    decode_lengths = parse_batch_distribution(args.decode)
    prefill_lengths = parse_batch_distribution(args.prefill)
    D = len(decode_lengths)
    P = len(prefill_lengths)
    total_steps = args.warmup + args.steps

    # Each profiled step with prefills needs fresh seq_ids
    total_prefill_slots = P * total_steps
    max_batch = D + total_prefill_slots
    max_seq = max(
        max(decode_lengths, default=0) + total_steps + 10,
        max(prefill_lengths, default=0) + 10,
        256)

    desc = batch_description(decode_lengths, prefill_lengths)
    tokens_per_step = D + sum(prefill_lengths)
    print(f"Batch: {desc}")
    print(f"Total tokens/step: {tokens_per_step}")
    print(f"max_seqs={max_batch}, max_seq_len={max_seq}")

    engine = MoEEngine(args.model, max_seqs=max_batch,
                       max_seq_len=max_seq)

    with torch.inference_mode():
        # Capture prefill graphs for setting up decode sequences
        unique_dec_lens = sorted(set(decode_lengths))
        if unique_dec_lens:
            engine.capture_prefill_cuda_graph(
                total_token_sizes=unique_dec_lens,
                use_torch_compile=args.compile)
            engine.reset()

        # Capture piecewise graphs for mixed steps
        if tokens_per_step > 0:
            engine.capture_mixed_cuda_graphs(
                total_token_sizes=[tokens_per_step],
                use_torch_compile=args.compile)

        # Prefill decode sequences to target lengths
        for i, seq_len in enumerate(decode_lengths):
            prompt = torch.randint(1, 1000, (seq_len,), device=engine.device)
            engine.prefill_to_slot(i, prompt)
        torch.cuda.synchronize()

        def run_one_step(step_idx):
            decode_seq_ids = list(range(D))
            if D > 0:
                dec_tokens = torch.randint(1, 1000, (D,),
                                           device=engine.device)
            else:
                dec_tokens = torch.empty(0, dtype=torch.long,
                                         device=engine.device)

            pf_seq_ids = [D + step_idx * P + j for j in range(P)]
            pf_inputs = [torch.randint(1, 1000, (L,), device=engine.device)
                         for L in prefill_lengths]

            return engine.mixed_step(
                decode_seq_ids=decode_seq_ids,
                decode_token_ids=dec_tokens,
                prefill_seq_ids=pf_seq_ids,
                prefill_input_ids=pf_inputs)

        # Warmup
        for step in range(args.warmup):
            run_one_step(step)
        torch.cuda.synchronize()

        # Profiled steps
        engine._nvtx_enabled = True
        torch.cuda.cudart().cudaProfilerStart()
        for step in range(args.steps):
            torch.cuda.nvtx.range_push(f"step_{step}")
            run_one_step(args.warmup + step)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    print(f"Profiled {args.steps} steps.")


# ═══════════════════════════════════════════════════════════════════
#  nsys Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_sqlite(sqlite_path, num_steps, num_layers):
    """Parse nsys sqlite, map kernels to (step, layer, stage).

    Returns:
      layer_phase_times: dict[(step, layer, stage)] -> total_kernel_ns
      nonlayer_phase_times: dict[(step, phase)] -> total_kernel_ns
      kernel_details: dict[(layer, stage)] -> list of (name, dur_ns)
    """
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    # ── 1. Load NVTX ranges ──
    cur.execute("SELECT text, start, end FROM NVTX_EVENTS WHERE text IS NOT NULL")
    nvtx_rows = cur.fetchall()

    # Step ranges (handle both 'step_N' and 'decode_step_N' formats)
    step_ranges = []
    for t, s, e in nvtx_rows:
        if t.startswith('step_') and t[5:].isdigit():
            step_ranges.append((int(t[5:]), s, e))
        elif t.startswith('decode_step_') and t[12:].isdigit():
            step_ranges.append((int(t[12:]), s, e))
    step_ranges.sort(key=lambda x: x[1])

    if len(step_ranges) < num_steps:
        print(f"Warning: found {len(step_ranges)} step ranges, expected {num_steps}")
        num_steps = len(step_ranges)

    # Layer ranges nested within steps
    layer_nvtx = sorted(
        [(t, s, e) for t, s, e in nvtx_rows if t.startswith('layer_')],
        key=lambda x: x[1])

    layer_ranges = []  # (step_idx, layer_idx, start, end)
    for step_idx, s_start, s_end in step_ranges:
        for text, start, end in layer_nvtx:
            if start >= s_start and end <= s_end:
                layer_ranges.append((step_idx, int(text[6:]), start, end))
    layer_ranges.sort(key=lambda x: (x[0], x[1]))

    # Stage ranges nested within layers
    stage_nvtx = sorted(
        [(t, s, e) for t, s, e in nvtx_rows if t in STAGES],
        key=lambda x: x[1])

    stage_ranges = []  # (step_idx, layer_idx, stage, start, end)
    for step_idx, layer_idx, l_start, l_end in layer_ranges:
        for text, start, end in stage_nvtx:
            if start >= l_start and end <= l_end:
                stage_ranges.append((step_idx, layer_idx, text, start, end))

    # Non-layer phases (setup, embed, final) within steps
    nl_nvtx = sorted(
        [(t, s, e) for t, s, e in nvtx_rows if t in NON_LAYER_PHASES],
        key=lambda x: x[1])

    nl_ranges = []  # (step_idx, phase, start, end)
    for step_idx, s_start, s_end in step_ranges:
        for text, start, end in nl_nvtx:
            if start >= s_start and end <= s_end:
                nl_ranges.append((step_idx, text, start, end))

    # ── 2. Load runtime calls (sorted by start) ──
    cur.execute("""
        SELECT correlationId, start
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        ORDER BY start
    """)
    runtime_calls = cur.fetchall()  # (corrId, start)
    rt_starts = [r[1] for r in runtime_calls]
    rt_corr_ids = [r[0] for r in runtime_calls]

    # ── 3. Load kernel data ──
    cur.execute("""
        SELECT k.correlationId, s.value as name, (k.end - k.start) as dur_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
    """)
    kernel_rows = cur.fetchall()

    corr_to_kernels = defaultdict(list)
    for corr_id, name, dur_ns in kernel_rows:
        corr_to_kernels[corr_id].append((name, dur_ns))

    conn.close()

    # ── 4. Map NVTX ranges -> runtime calls -> kernels ──
    layer_phase_times = defaultdict(int)
    nonlayer_phase_times = defaultdict(int)
    kernel_details = defaultdict(list)

    def _map_range(range_start, range_end):
        """Find all kernels launched within [range_start, range_end]."""
        lo = bisect_left(rt_starts, range_start)
        hi = bisect_right(rt_starts, range_end)
        total_ns = 0
        details = []
        for i in range(lo, hi):
            for name, dur_ns in corr_to_kernels.get(rt_corr_ids[i], []):
                total_ns += dur_ns
                details.append((name, dur_ns))
        return total_ns, details

    for step_idx, layer_idx, stage, s_start, s_end in stage_ranges:
        total_ns, details = _map_range(s_start, s_end)
        layer_phase_times[(step_idx, layer_idx, stage)] = total_ns
        kernel_details[(layer_idx, stage)].extend(details)

    for step_idx, phase, s_start, s_end in nl_ranges:
        total_ns, _ = _map_range(s_start, s_end)
        nonlayer_phase_times[(step_idx, phase)] = total_ns

    return layer_phase_times, nonlayer_phase_times, kernel_details, num_steps


def _distrib(values):
    """Compute distribution statistics."""
    if not values:
        return {'avg': 0, 'med': 0, 'std': 0, 'iqr': 0, 'min': 0, 'max': 0,
                'p25': 0, 'p75': 0}
    n = len(values)
    avg = statistics.mean(values)
    med = statistics.median(values)
    std = statistics.stdev(values) if n > 1 else 0
    if n >= 4:
        q = statistics.quantiles(values, n=4)
        p25, p75, iqr = q[0], q[2], q[2] - q[0]
    else:
        p25 = p75 = med
        iqr = 0
    return {
        'avg': avg, 'med': med, 'std': std, 'iqr': iqr,
        'min': min(values), 'max': max(values), 'p25': p25, 'p75': p75,
    }


def print_analysis(layer_phase_times, nonlayer_phase_times, kernel_details,
                   num_steps, num_layers, label=""):
    """Print formatted analysis tables."""

    # Determine which stages have data
    active_stages = [s for s in STAGES
                     if any(layer_phase_times.get((step, 0, s), 0) > 0
                            for step in range(num_steps))]

    print(f"\n{'='*80}")
    print(f"Phase Kernel Timing: {label}")
    print(f"{num_steps} steps, {num_layers} layers")
    print(f"All times in microseconds (us) unless noted otherwise")
    print(f"{'='*80}")

    # ── 1. Per-phase total (summed across layers, per step) ──
    print(f"\n--- Per-Phase Total (kernel time summed over {num_layers} "
          f"layers, us/step) ---")
    hdr = (f"{'Phase':<10} {'Avg':>8} {'Med':>8} {'Std':>7} {'IQR':>7} "
           f"{'Min':>8} {'Max':>8}")
    print(hdr)

    phase_medians = {}
    for stage in active_stages:
        vals = []
        for step in range(num_steps):
            total = sum(layer_phase_times.get((step, l, stage), 0)
                        for l in range(num_layers))
            vals.append(total / 1000)  # ns -> us
        d = _distrib(vals)
        phase_medians[stage] = d['med']
        print(f"{stage:<10} {d['avg']:>8.1f} {d['med']:>8.1f} {d['std']:>7.1f} "
              f"{d['iqr']:>7.1f} {d['min']:>8.1f} {d['max']:>8.1f}")

    # Grand total across all layer phases
    all_layer_vals = []
    for step in range(num_steps):
        total = sum(layer_phase_times.get((step, l, s), 0)
                    for l in range(num_layers) for s in active_stages)
        all_layer_vals.append(total / 1000)
    d = _distrib(all_layer_vals)
    print(f"{'TOTAL':<10} {d['avg']:>8.1f} {d['med']:>8.1f} {d['std']:>7.1f} "
          f"{d['iqr']:>7.1f} {d['min']:>8.1f} {d['max']:>8.1f}")
    grand_median = d['med']

    # Percentage breakdown
    print(f"\n  Phase share of layer compute (by median):")
    for stage in active_stages:
        pct = phase_medians[stage] / grand_median * 100 if grand_median > 0 else 0
        print(f"    {stage:<8} {pct:5.1f}%  ({phase_medians[stage]:.1f} us)")

    # Non-layer phases
    active_nl = []
    for phase in NON_LAYER_PHASES:
        vals = [nonlayer_phase_times.get((step, phase), 0) / 1000
                for step in range(num_steps)]
        if any(v > 0 for v in vals):
            active_nl.append((phase, vals))

    if active_nl:
        print(f"\n--- Non-Layer Phases (us/step) ---")
        print(hdr)
        for phase, vals in active_nl:
            d = _distrib(vals)
            print(f"{phase:<10} {d['avg']:>8.1f} {d['med']:>8.1f} {d['std']:>7.1f} "
                  f"{d['iqr']:>7.1f} {d['min']:>8.1f} {d['max']:>8.1f}")

    # ── 2. Per-layer breakdown ──
    print(f"\n--- Per-Layer Breakdown (median kernel time, us) ---")
    header = f"{'Layer':>5}"
    for s in active_stages:
        header += f" {s:>8}"
    header += f" {'Total':>8}"
    print(header)

    for layer in range(num_layers):
        row = f"{layer:>5}"
        layer_total = 0
        for stage in active_stages:
            vals = [layer_phase_times.get((step, layer, stage), 0) / 1000
                    for step in range(num_steps)]
            med = statistics.median(vals)
            row += f" {med:>8.1f}"
            layer_total += med
        row += f" {layer_total:>8.1f}"
        print(row)

    # ── 3. Per-layer distribution (show variance across layers) ──
    print(f"\n--- Per-Layer Variation (distribution of layer medians, us) ---")
    print(f"{'Phase':<10} {'Avg':>8} {'Med':>8} {'Std':>7} {'Min':>8} {'Max':>8}")
    for stage in active_stages:
        layer_medians = []
        for layer in range(num_layers):
            vals = [layer_phase_times.get((step, layer, stage), 0) / 1000
                    for step in range(num_steps)]
            layer_medians.append(statistics.median(vals))
        d = _distrib(layer_medians)
        print(f"{stage:<10} {d['avg']:>8.1f} {d['med']:>8.1f} {d['std']:>7.1f} "
              f"{d['min']:>8.1f} {d['max']:>8.1f}")

    # ── 4. Kernel breakdown per phase ──
    print(f"\n--- Top Kernels per Phase (avg per invocation, us) ---")
    for stage in active_stages:
        all_kernels = defaultdict(list)
        for layer in range(num_layers):
            for name, dur_ns in kernel_details.get((layer, stage), []):
                all_kernels[name].append(dur_ns / 1000)

        if not all_kernels:
            continue

        sorted_kernels = sorted(all_kernels.items(),
                                key=lambda x: -sum(x[1]))
        total_us = sum(sum(v) for v in all_kernels.values())
        total_count = sum(len(v) for v in all_kernels.values())
        per_layer_step = total_count / (num_steps * num_layers) if num_steps * num_layers > 0 else 0

        print(f"\n  {stage} ({len(sorted_kernels)} unique kernels, "
              f"~{per_layer_step:.0f} launches/layer/step):")
        print(f"  {'#':>3} {'Kernel':<55} {'Avg':>7} {'Count':>6} {'%':>5}")
        for i, (name, durations) in enumerate(sorted_kernels[:12]):
            avg_us = statistics.mean(durations)
            count = len(durations)
            pct = sum(durations) / total_us * 100 if total_us > 0 else 0
            trunc_name = name[:55]
            print(f"  {i+1:>3} {trunc_name:<55} {avg_us:>7.2f} {count:>6} {pct:>5.1f}")

    # ── 5. Grand summary ──
    print(f"\n{'='*80}")
    print(f"SUMMARY (median per step)")
    print(f"{'='*80}")
    total_layer_ms = grand_median / 1000
    print(f"  Layer compute:    {total_layer_ms:.3f} ms/step "
          f"({total_layer_ms/num_layers:.3f} ms/layer)")
    for stage in active_stages:
        ms = phase_medians[stage] / 1000
        pct = phase_medians[stage] / grand_median * 100 if grand_median > 0 else 0
        print(f"    {stage:<8}  {ms:.3f} ms  ({pct:.1f}%)")

    nl_total = 0
    for phase, vals in active_nl:
        med = statistics.median(vals)
        nl_total += med
    if nl_total > 0:
        print(f"  Non-layer:        {nl_total/1000:.3f} ms/step")
    print(f"  Total kernel:     {(grand_median + nl_total)/1000:.3f} ms/step")


# ═══════════════════════════════════════════════════════════════════
#  Expert PCIe Transfer Measurement
# ═══════════════════════════════════════════════════════════════════

def measure_expert_transfer():
    """Measure H2D transfer time for one Mixtral expert."""
    import ctypes
    ctypes.CDLL(
        "/gpfs/radev/apps/avx512/software/GCCcore/13.3.0/lib64/libstdc++.so.6",
        mode=ctypes.RTLD_GLOBAL)

    import torch

    # Mixtral expert BF16:
    #   w1 (gate+up stacked): [2*14336, 4096] = 224 MB
    #   w2 (down):            [4096, 14336]    = 112 MB
    #   Total: 336 MB
    w1_shape = (2 * 14336, 4096)
    w2_shape = (4096, 14336)

    w1_cpu = torch.randn(w1_shape, dtype=torch.bfloat16).pin_memory()
    w2_cpu = torch.randn(w2_shape, dtype=torch.bfloat16).pin_memory()
    w1_gpu = torch.empty(w1_shape, dtype=torch.bfloat16, device='cuda')
    w2_gpu = torch.empty(w2_shape, dtype=torch.bfloat16, device='cuda')

    total_bytes = (w1_cpu.nelement() + w2_cpu.nelement()) * 2
    total_mb = total_bytes / 1024 / 1024

    # Warmup
    for _ in range(10):
        w1_gpu.copy_(w1_cpu, non_blocking=True)
        w2_gpu.copy_(w2_cpu, non_blocking=True)
    torch.cuda.synchronize()

    # Timed transfers (combined w1+w2)
    n_trials = 100
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    combined_times = []
    for _ in range(n_trials):
        start_ev.record()
        w1_gpu.copy_(w1_cpu, non_blocking=True)
        w2_gpu.copy_(w2_cpu, non_blocking=True)
        end_ev.record()
        torch.cuda.synchronize()
        combined_times.append(start_ev.elapsed_time(end_ev))

    # Timed w1 only
    w1_times = []
    for _ in range(n_trials):
        start_ev.record()
        w1_gpu.copy_(w1_cpu, non_blocking=True)
        end_ev.record()
        torch.cuda.synchronize()
        w1_times.append(start_ev.elapsed_time(end_ev))

    # Timed w2 only
    w2_times = []
    for _ in range(n_trials):
        start_ev.record()
        w2_gpu.copy_(w2_cpu, non_blocking=True)
        end_ev.record()
        torch.cuda.synchronize()
        w2_times.append(start_ev.elapsed_time(end_ev))

    def _print_stats(label, times, size_mb):
        d = _distrib(times)
        bw = size_mb / (d['med'] / 1000)  # MB/s
        print(f"  {label}")
        print(f"    Size:      {size_mb:.1f} MB")
        print(f"    Median:    {d['med']:.3f} ms")
        print(f"    Mean:      {d['avg']:.3f} ms")
        print(f"    Std:       {d['std']:.3f} ms")
        print(f"    Min/Max:   {d['min']:.3f} / {d['max']:.3f} ms")
        print(f"    Bandwidth: {bw/1024:.1f} GB/s")

    w1_mb = w1_cpu.nelement() * 2 / 1024 / 1024
    w2_mb = w2_cpu.nelement() * 2 / 1024 / 1024

    print(f"\nMixtral Expert PCIe Transfer (H2D, pinned -> GPU)")
    print(f"  {n_trials} trials each\n")
    _print_stats(f"w1 (gate+up) {w1_shape}", w1_times, w1_mb)
    print()
    _print_stats(f"w2 (down)    {w2_shape}", w2_times, w2_mb)
    print()
    _print_stats(f"Combined (w1+w2)", combined_times, total_mb)


# ═══════════════════════════════════════════════════════════════════
#  nsys Orchestrator
# ═══════════════════════════════════════════════════════════════════

def _export_sqlite(rep_path):
    """Export .nsys-rep to .sqlite in /tmp."""
    sqlite_path = os.path.join(tempfile.gettempdir(),
                               Path(rep_path).stem + ".sqlite")
    cmd = [NSYS, "export", "--type", "sqlite",
           "--output", sqlite_path, rep_path, "--force-overwrite=true"]
    subprocess.run(cmd, check=True, capture_output=True)
    return sqlite_path


def run_orchestrator(args):
    """Launch nsys, export sqlite to /tmp, analyze, clean up."""
    decode_lengths = parse_batch_distribution(args.decode)
    prefill_lengths = parse_batch_distribution(args.prefill)

    if not decode_lengths and not prefill_lengths:
        print("Error: specify at least --decode or --prefill")
        sys.exit(1)

    desc = batch_filename(decode_lengths, prefill_lengths)
    compile_tag = "_compile" if args.compile else ""

    # All intermediate files go to /tmp
    rep_path = os.path.join(tempfile.gettempdir(),
                            f"phases_{desc}{compile_tag}.nsys-rep")

    # Build worker command (self-invocation)
    worker_cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--model", args.model,
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
    ]
    if args.decode:
        worker_cmd.extend(["--decode"] + args.decode)
    if args.prefill:
        worker_cmd.extend(["--prefill"] + args.prefill)
    if args.compile:
        worker_cmd.append("--compile")

    nsys_cmd = [
        NSYS, "profile",
        "--trace=cuda,nvtx",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "-o", rep_path,
        "-f", "true",
    ] + worker_cmd

    print(f"Output (temp): {rep_path}")
    print(f"Running: {' '.join(nsys_cmd[:6])} ... {' '.join(nsys_cmd[-4:])}")
    result = subprocess.run(nsys_cmd)
    if result.returncode != 0:
        print(f"nsys failed (rc={result.returncode})")
        sys.exit(1)

    # Export sqlite to /tmp
    print("Exporting to sqlite...")
    sqlite_path = _export_sqlite(rep_path)

    # Get num_layers from model config
    config_path = os.path.join(args.model, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    num_layers = config.get("num_hidden_layers", 20)

    # Analyze and print
    desc_str = batch_description(decode_lengths, prefill_lengths)
    label = f"Mixtral-8x7B-{num_layers}L, {desc_str}"
    print(f"\nAnalyzing {sqlite_path}...")
    lpt, nlpt, kd, actual_steps = analyze_sqlite(
        sqlite_path, args.steps, num_layers)
    print_analysis(lpt, nlpt, kd, actual_steps, num_layers, label)

    # Clean up intermediate files
    for f in [rep_path, sqlite_path]:
        try:
            os.unlink(f)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════
#  CUDA-Event Sweep (generates profiling.md)
# ═══════════════════════════════════════════════════════════════════

def _sweep_imports():
    """Lazy imports for sweep mode (need CUDA)."""
    import ctypes
    ctypes.CDLL(
        "/gpfs/radev/apps/avx512/software/GCCcore/13.3.0/lib64/libstdc++.so.6",
        mode=ctypes.RTLD_GLOBAL)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(MOE_DIR))

    import torch
    import torch.nn.functional as F
    from moe_engine import MoEEngine
    return torch, F, MoEEngine


def measure_decode_phases(engine, torch, F, target_pos,
                          n_warmup=SWEEP_WARMUP, n_steps=SWEEP_STEPS):
    """Measure per-stage GPU time for decode at target_pos using CUDA events."""
    info = engine._piecewise_graphs[1]
    H = engine.hidden_size
    L = engine.num_layers
    wrapper = engine._decode_wrapper

    layer_events = [
        [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        for _ in range(L)
    ]

    phase_records = []

    for step in range(n_warmup + n_steps):
        engine._seq_lens_cpu[0] = target_pos
        engine.seq_lens[0] = target_pos

        engine._seq_lens_cpu[0] += 1
        engine._plan_flashinfer_decode_for_subset([0])

        info['static_token_ids'][0] = 100
        info['static_positions'][0] = target_pos
        pg = target_pos // engine.page_size
        off = target_pos % engine.page_size
        info['static_slot_mapping'][0] = (
            engine.block_table[0, pg].long() * engine.page_size + off)

        info['hidden_buf'].copy_(
            F.embedding(info['static_token_ids'], engine.embed_tokens))

        for layer in range(L):
            evts = layer_events[layer]

            evts[0].record()
            info['stage1_graphs'][layer].replay()
            evts[1].record()

            q_decode = info['q_buf'][:1]
            decode_out = wrapper.run(
                q_decode, (engine.k_cache[layer], engine.v_cache[layer]))
            info['attn_out_buf'][:1].copy_(decode_out.reshape(1, H))
            evts[2].record()

            info['stage4a_graphs'][layer].replay()
            evts[3].record()

            info['stage4b_graphs'][layer].replay()
            evts[4].record()

        torch.cuda.synchronize()

        engine.seq_lens[0] = target_pos + 1

        if step >= n_warmup:
            step_phases = {'stage1': 0, 'stage2': 0, 'stage4a': 0, 'stage4b': 0}
            for layer in range(L):
                evts = layer_events[layer]
                step_phases['stage1'] += evts[0].elapsed_time(evts[1]) * 1000
                step_phases['stage2'] += evts[1].elapsed_time(evts[2]) * 1000
                step_phases['stage4a'] += evts[2].elapsed_time(evts[3]) * 1000
                step_phases['stage4b'] += evts[3].elapsed_time(evts[4]) * 1000
            phase_records.append(step_phases)

    result = {}
    for stage in ['stage1', 'stage2', 'stage4a', 'stage4b']:
        vals = [r[stage] / L for r in phase_records]
        result[stage] = statistics.mean(vals)
    result['total'] = sum(result[s] for s in ['stage1', 'stage2', 'stage4a', 'stage4b'])
    return result


def measure_prefill_phases(engine, torch, F, seq_len,
                           n_warmup=SWEEP_WARMUP, n_steps=SWEEP_STEPS):
    """Measure per-stage GPU time for prefill of seq_len using CUDA events."""
    info = engine._piecewise_graphs[seq_len]
    H = engine.hidden_size
    L = engine.num_layers
    graph_N = info['static_token_ids'].shape[0]

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    layer_events = [
        [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        for _ in range(L)
    ]

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32,
                              device=engine.device)

    phase_records = []

    for step in range(n_warmup + n_steps):
        seq_id = step

        input_ids = torch.randint(1, 1000, (seq_len,), device=engine.device)
        info['static_token_ids'][:seq_len].copy_(input_ids)
        if seq_len < graph_N:
            info['static_token_ids'][seq_len:].zero_()

        positions = torch.arange(seq_len, dtype=torch.int32,
                                 device=engine.device)
        info['static_positions'][:seq_len].copy_(positions)
        if seq_len < graph_N:
            info['static_positions'][seq_len:].zero_()

        pg = (positions // engine.page_size).long()
        off = (positions % engine.page_size).long()
        slots = engine.block_table[seq_id, pg].long() * engine.page_size + off
        info['static_slot_mapping'][:seq_len].copy_(slots)
        if seq_len < graph_N:
            info['static_slot_mapping'][seq_len:].fill_(-1)

        info['hidden_buf'].copy_(
            F.embedding(info['static_token_ids'], engine.embed_tokens))

        for layer in range(L):
            evts = layer_events[layer]

            evts[0].record()
            info['stage1_graphs'][layer].replay()
            evts[1].record()

            q_pf = info['q_buf'][:seq_len]
            k_pf = info['k_buf'][:seq_len]
            v_pf = info['v_buf'][:seq_len]
            prefill_out = flash_attn_varlen_func(
                q_pf, k_pf, v_pf,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                causal=True, fa_version=3)
            info['attn_out_buf'][:seq_len].copy_(
                prefill_out.reshape(seq_len, H))
            if seq_len < graph_N:
                info['attn_out_buf'][seq_len:].zero_()
            evts[2].record()

            info['stage4a_graphs'][layer].replay()
            evts[3].record()

            info['stage4b_graphs'][layer].replay()
            evts[4].record()

        torch.cuda.synchronize()

        engine._seq_lens_cpu[seq_id] = seq_len
        engine.seq_lens[seq_id] = seq_len

        if step >= n_warmup:
            step_phases = {'stage1': 0, 'stage3': 0, 'stage4a': 0, 'stage4b': 0}
            for layer in range(L):
                evts = layer_events[layer]
                step_phases['stage1'] += evts[0].elapsed_time(evts[1]) * 1000
                step_phases['stage3'] += evts[1].elapsed_time(evts[2]) * 1000
                step_phases['stage4a'] += evts[2].elapsed_time(evts[3]) * 1000
                step_phases['stage4b'] += evts[3].elapsed_time(evts[4]) * 1000
            phase_records.append(step_phases)

    result = {}
    for stage in ['stage1', 'stage3', 'stage4a', 'stage4b']:
        vals = [r[stage] / L for r in phase_records]
        result[stage] = statistics.mean(vals)
    result['total'] = sum(result[s] for s in ['stage1', 'stage3', 'stage4a', 'stage4b'])
    return result


def _fmt_us(val):
    """Format a microsecond value for the markdown table."""
    if val < 1000:
        return f"{val:.1f}"
    else:
        return f"{val:,.0f}"


def generate_markdown(decode_results, prefill_results, num_layers, compile_flag):
    """Generate profiling.md content."""
    compile_str = ("torch.compile + CUDA graph" if compile_flag
                   else "CUDA graph (no compile)")

    lines = [
        f"# Mixtral-8x7B-{num_layers}L Per-Phase Kernel Timing",
        "",
        "## Environment",
        "",
        f"- Model: Mixtral-8x7B-{num_layers}L, single H100 80GB",
        f"- Execution: {compile_str}, piecewise per-layer graphs",
        f"- Timing: CUDA events (GPU stream elapsed time, zero observer overhead)",
        f"- {SWEEP_STEPS} profiled steps, {SWEEP_WARMUP} warmup, "
        f"averaged across steps and layers",
        f"- All values are **per-layer averages** in microseconds (us)",
        "",
        "## Computation Phases",
        "",
        "| Phase | Decode | Prefill | Contents |",
        "|-------|--------|---------|----------|",
        "| stage1 | yes | yes | RMSNorm + QKV GEMM + RoPE + KV cache write |",
        "| stage2 | yes | - | FlashInfer paged decode attention |",
        "| stage3 | - | yes | Flash Attention v3 causal prefill |",
        "| stage4a | yes | yes | O proj + residual + RMSNorm + "
        "router GEMM + softmax + top-k |",
        "| stage4b | yes | yes | fused_moe (Triton) + SiLU + residual add |",
        "",
    ]

    # Decode table
    lines.append("## Decode (per-layer avg, us)")
    lines.append("")
    lines.append("1 decode token at the given sequence position.")
    lines.append("")
    lines.append("| Seq Pos | stage1 | stage2 | stage4a | stage4b | Total |")
    lines.append("|--------:|-------:|-------:|--------:|--------:|------:|")

    for pos in sorted(decode_results.keys()):
        r = decode_results[pos]
        lines.append(
            f"| {pos:,} | {_fmt_us(r['stage1'])} | {_fmt_us(r['stage2'])} | "
            f"{_fmt_us(r['stage4a'])} | {_fmt_us(r['stage4b'])} | "
            f"{_fmt_us(r['total'])} |")

    lines.append("")

    # Prefill table
    lines.append("## Prefill (per-layer avg, us)")
    lines.append("")
    lines.append("1 prefill sequence of the given length, no concurrent decodes.")
    lines.append("")
    lines.append("| Seq Len | stage1 | stage3 | stage4a | stage4b | Total |")
    lines.append("|--------:|-------:|-------:|--------:|--------:|------:|")

    for sz in sorted(prefill_results.keys()):
        r = prefill_results[sz]
        lines.append(
            f"| {sz:,} | {_fmt_us(r['stage1'])} | {_fmt_us(r['stage3'])} | "
            f"{_fmt_us(r['stage4a'])} | {_fmt_us(r['stage4b'])} | "
            f"{_fmt_us(r['total'])} |")

    lines.append("")

    # Expert transfer reference
    lines.append("## Reference: Expert PCIe Transfer")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append("| Expert size (BF16) | 336 MB |")
    lines.append("| H2D transfer time | 6,314 us |")
    lines.append("| PCIe bandwidth | 52.0 GB/s |")
    lines.append("")

    return "\n".join(lines)


def run_sweep(args):
    """Run CUDA-event sweep over decode positions and prefill sizes."""
    import gc
    import time

    torch, F, MoEEngine = _sweep_imports()

    n_warmup = args.warmup
    n_steps = args.steps

    # ── Decode sweep ──
    max_pos = max(DECODE_POSITIONS)
    max_seq = max_pos + n_warmup + n_steps + 16
    print(f"\n--- Decode sweep ---")
    print(f"  max_seqs=1, max_seq_len={max_seq}")

    engine = MoEEngine(args.model, max_seqs=1, max_seq_len=max_seq)
    L = engine.num_layers

    with torch.inference_mode():
        engine.capture_mixed_cuda_graphs(
            total_token_sizes=[1], use_torch_compile=args.compile)

        for layer in range(L):
            engine.k_cache[layer].normal_(0, 0.01)
            engine.v_cache[layer].normal_(0, 0.01)
        torch.cuda.synchronize()

        decode_results = {}
        for pos in DECODE_POSITIONS:
            t0 = time.time()
            r = measure_decode_phases(engine, torch, F, pos, n_warmup, n_steps)
            elapsed = time.time() - t0
            decode_results[pos] = r
            print(f"  decode@{pos:>7,}: "
                  f"s1={r['stage1']:.1f}  s2={r['stage2']:.1f}  "
                  f"s4a={r['stage4a']:.1f}  s4b={r['stage4b']:.1f}  "
                  f"total={r['total']:.1f} us/layer  ({elapsed:.1f}s)")

    del engine
    gc.collect()
    torch.cuda.empty_cache()

    # ── Prefill sweep ──
    max_batch = n_warmup + n_steps
    max_seq = max(PREFILL_SIZES) + 16
    print(f"\n--- Prefill sweep ---")
    print(f"  max_seqs={max_batch}, max_seq_len={max_seq}")

    engine = MoEEngine(args.model, max_seqs=max_batch,
                       max_seq_len=max_seq)

    prefill_results = {}
    with torch.inference_mode():
        for sz in PREFILL_SIZES:
            engine._piecewise_graphs = {}
            torch.cuda.empty_cache()
            engine.capture_mixed_cuda_graphs(
                total_token_sizes=[sz], use_torch_compile=args.compile)

            engine.reset()
            t0 = time.time()
            r = measure_prefill_phases(engine, torch, F, sz, n_warmup, n_steps)
            elapsed = time.time() - t0
            prefill_results[sz] = r
            print(f"  prefill({sz:>5}): "
                  f"s1={r['stage1']:.1f}  s3={r['stage3']:.1f}  "
                  f"s4a={r['stage4a']:.1f}  s4b={r['stage4b']:.1f}  "
                  f"total={r['total']:.1f} us/layer  ({elapsed:.1f}s)")

    del engine
    gc.collect()
    torch.cuda.empty_cache()

    # ── Generate markdown ──
    md = generate_markdown(decode_results, prefill_results, L, args.compile)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "profiling.md")
    with open(out_path, 'w') as f:
        f.write(md)
    print(f"\nWritten to {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Profile per-phase kernel timing for Mixtral-8x7B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/profile_phases.py --decode 128
  python tests/profile_phases.py --decode 4:2048 --prefill 2:128
  python tests/profile_phases.py --sweep
  python tests/profile_phases.py --sweep --compile
  python tests/profile_phases.py --analyze /path/to/file.sqlite --layers 20
  python tests/profile_phases.py --measure-transfer
""")

    parser.add_argument("--decode", nargs='+', type=str, default=[],
                        help="Decode sequences: '128' or '4:2048'")
    parser.add_argument("--prefill", nargs='+', type=str, default=[],
                        help="Prefill sequences: '1024' or '2:128'")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model path (default: Mixtral-8x7B-20L)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of profiled steps")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for CUDA graph capture")

    parser.add_argument("--worker", action="store_true",
                        help="Internal: run as worker under nsys")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Analyze existing sqlite file (skip profiling)")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of layers (for --analyze mode)")

    parser.add_argument("--sweep", action="store_true",
                        help="Run CUDA-event sweep, generate profiling.md")
    parser.add_argument("--measure-transfer", action="store_true",
                        help="Measure expert PCIe transfer time and exit")

    args = parser.parse_args()

    if args.measure_transfer:
        measure_expert_transfer()
    elif args.sweep:
        run_sweep(args)
    elif args.analyze:
        num_layers = args.layers or 20
        decode_lengths = parse_batch_distribution(args.decode)
        prefill_lengths = parse_batch_distribution(args.prefill)
        desc = batch_description(decode_lengths, prefill_lengths)
        label = f"Mixtral-8x7B-{num_layers}L, {desc}"
        print(f"\nAnalyzing {args.analyze}...")
        lpt, nlpt, kd, actual_steps = analyze_sqlite(
            args.analyze, args.steps, num_layers)
        print_analysis(lpt, nlpt, kd, actual_steps, num_layers, label)
    elif args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
