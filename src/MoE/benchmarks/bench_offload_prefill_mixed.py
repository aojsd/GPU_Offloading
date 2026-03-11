#!/usr/bin/env -S python3 -u
"""Comprehensive offloading benchmark for prefill and mixed batches.

Sweeps cache sizes (via configure()) and measures latency + transfer stats
for multiple prefill and mixed decode+prefill configurations.

Usage:
    python tests/bench_offload_prefill_mixed.py --model models/Mixtral-8x7B-20L
    python tests/bench_offload_prefill_mixed.py --model models/OLMoE-1B-7B
    python tests/bench_offload_prefill_mixed.py --model models/Mixtral-8x7B
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from moe_engine import MoEEngine

# ── Test Configurations ───────────────────────────────────────────────

# (name, list_of_sequence_lengths)
PREFILL_CONFIGS = [
    ("1×128",  [128]),
    ("1×256",  [256]),
    ("1×512",  [512]),
    ("4×32",   [32, 32, 32, 32]),
]

# (name, n_decode, list_of_prefill_lengths)
MIXED_CONFIGS = [
    ("16D",            16, []),
    ("32D",            32, []),
    ("1D+1×128P",       1, [128]),
    ("8D+1×128P",       8, [128]),
    ("16D+1×128P",     16, [128]),
    ("16D+1×256P",     16, [256]),
    ("8D+2×64P",        8, [64, 64]),
]


def detect_model(model_path):
    """Auto-detect model type and return (E, L, max_epl, budget_sweep)."""
    with open(Path(model_path) / "config.json") as f:
        cfg = json.load(f)
    E = (cfg.get("n_routed_experts") or cfg.get("num_local_experts")
         or cfg.get("num_experts", 8))
    L = cfg.get("num_hidden_layers", 32)

    if E == 64:  # OLMoE
        max_epl = 64
        budgets = [8, 16, 32, 64]
    elif L <= 20:  # Mixtral-20L
        max_epl = 8
        budgets = [2, 4, 8]
    else:  # Mixtral-32L
        max_epl = 4
        budgets = [2, 3, 4]

    return E, L, max_epl, budgets


def run_prefill_bench(engine, seq_lengths, n_warmup=3, n_trials=10):
    """Benchmark a single prefill configuration.

    Returns: (median_ms, transfer_stats_dict)
    """
    oe = engine.offload_engine
    device = engine.device
    prompts = [torch.randint(1, 1000, (s,), device=device) for s in seq_lengths]
    seq_ids = list(range(len(seq_lengths)))

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    times = []
    all_stats = []

    for i in range(n_warmup + n_trials):
        engine.reset()
        oe.reset_trace()

        start_ev.record()
        if len(seq_lengths) == 1:
            engine.prefill_to_slot(0, prompts[0])
        else:
            engine.prefill_batch_to_slots(seq_ids, prompts)
        end_ev.record()
        end_ev.synchronize()

        if i >= n_warmup:
            times.append(start_ev.elapsed_time(end_ev))
            all_stats.append(oe.get_transfer_stats())

    median_ms = sorted(times)[len(times) // 2]
    median_idx = times.index(median_ms)
    stats = all_stats[median_idx]

    return median_ms, stats


def run_mixed_bench(engine, n_decode, prefill_lengths,
                    n_warmup=3, n_trials=10):
    """Benchmark a single mixed configuration.

    Assumes decode sequences 0..n_decode-1 are already prefilled.
    Uses slots 32+ for prefill sequences.

    Returns: (median_ms, transfer_stats_dict)
    """
    oe = engine.offload_engine
    device = engine.device

    decode_seq_ids = list(range(n_decode)) if n_decode > 0 else []
    decode_token_ids = (torch.randint(1, 1000, (n_decode,), device=device)
                        if n_decode > 0
                        else torch.empty(0, dtype=torch.long, device=device))

    prefill_seq_ids = [32 + i for i in range(len(prefill_lengths))]
    prefill_inputs = [torch.randint(1, 1000, (s,), device=device)
                      for s in prefill_lengths]

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    times = []
    all_stats = []

    for i in range(n_warmup + n_trials):
        # Reset prefill slots so they're fresh
        for sid in prefill_seq_ids:
            engine.seq_lens[sid] = 0
            engine._seq_lens_cpu[sid] = 0

        oe.reset_trace()

        start_ev.record()
        engine.mixed_step(
            decode_seq_ids=decode_seq_ids,
            decode_token_ids=decode_token_ids,
            prefill_seq_ids=prefill_seq_ids if prefill_lengths else [],
            prefill_input_ids=prefill_inputs if prefill_lengths else [])
        end_ev.record()
        end_ev.synchronize()

        if i >= n_warmup:
            times.append(start_ev.elapsed_time(end_ev))
            all_stats.append(oe.get_transfer_stats())

    median_ms = sorted(times)[len(times) // 2]
    median_idx = times.index(median_ms)
    stats = all_stats[median_idx]

    return median_ms, stats


def setup_decode_sequences(engine, n_sequences, prompt_len=128):
    """Prefill n_sequences with random prompts for decode benchmarking."""
    for sid in range(n_sequences):
        prompt = torch.randint(1, 1000, (prompt_len,), device=engine.device)
        engine.prefill_to_slot(sid, prompt)


def fmt_row(name, median, xfers, xfer_mb, xfer_ms):
    return (f"  {name:<16} {median:>10.2f} {xfers:>7} "
            f"{xfer_mb:>9.0f} {xfer_ms:>9.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(
        Path(__file__).resolve().parent.parent / "models" / "Mixtral-8x7B-20L"))
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    E, L, max_epl, budgets = detect_model(args.model)
    model_name = Path(args.model).name
    print(f"Model: {model_name} ({L}L, {E}E)")
    print(f"Max experts_per_layer: {max_epl}")
    print(f"Budget sweep: {budgets}")
    print(f"Warmup: {args.n_warmup}, Trials: {args.n_trials}")

    # Compute max decode sequences needed
    max_decode = max(cfg[1] for cfg in MIXED_CONFIGS)

    # Graph sizes to capture — must cover all test configs
    graph_sizes = sorted(set([16, 32, 128, 256, 512]))
    print(f"Piecewise graph sizes: {graph_sizes}")

    max_seqs = max_decode + 8  # extra slots for prefill seqs
    print(f"max_seqs: {max_seqs}")

    # Create engine
    print(f"\nLoading engine with experts_per_layer={max_epl}...")
    engine = MoEEngine(args.model, max_seqs=max_seqs,
                       max_seq_len=2048,
                       experts_per_layer=max_epl,
                       use_torch_compile=False)

    with torch.inference_mode():
        print("Capturing piecewise CUDA graphs...")
        engine.capture_mixed_cuda_graphs(
            total_token_sizes=graph_sizes, use_torch_compile=False)

    oe = engine.offload_engine
    all_results = []

    header = (f"  {'Config':<16} {'Median(ms)':>10} {'Xfers':>7} "
              f"{'XferMB':>9} {'XferMs':>9}")

    with torch.inference_mode():
        for budget in budgets:
            tag = "all resident" if budget >= E else f"{budget}/{E}"
            print(f"\n{'='*70}")
            print(f"  Budget: {budget} experts/layer ({tag})")
            print(f"{'='*70}")

            initial = list(range(min(budget, E)))
            oe.configure(gpu_budget_per_layer=budget, initial_experts=initial)

            # ── Prefill Benchmarks ──
            print(f"\n  --- Prefill ---")
            print(header)

            for name, seq_lens in PREFILL_CONFIGS:
                median_ms, stats = run_prefill_bench(
                    engine, seq_lens,
                    n_warmup=args.n_warmup, n_trials=args.n_trials)

                xfers = stats['total_transfers']
                xfer_mb = stats['total_bytes'] / 1e6
                xfer_ms = stats['total_time_ms']

                print(fmt_row(name, median_ms, xfers, xfer_mb, xfer_ms))

                all_results.append({
                    'type': 'prefill', 'config': name,
                    'budget': budget,
                    'total_tokens': sum(seq_lens),
                    'median_ms': round(median_ms, 2),
                    'transfers': xfers,
                    'transfer_mb': round(xfer_mb, 1),
                    'transfer_ms': round(xfer_ms, 1),
                })

            # ── Mixed Benchmarks ──
            print(f"\n  --- Mixed ---")
            print(header)

            # Setup: prefill decode sequences
            engine.reset()
            oe.reset_trace()
            print(f"  (setting up {max_decode} decode sequences...)")
            setup_decode_sequences(engine, max_decode, prompt_len=128)
            print(header)

            for name, n_decode, prefill_lens in MIXED_CONFIGS:
                median_ms, stats = run_mixed_bench(
                    engine, n_decode, prefill_lens,
                    n_warmup=args.n_warmup, n_trials=args.n_trials)

                xfers = stats['total_transfers']
                xfer_mb = stats['total_bytes'] / 1e6
                xfer_ms = stats['total_time_ms']

                print(fmt_row(name, median_ms, xfers, xfer_mb, xfer_ms))

                all_results.append({
                    'type': 'mixed', 'config': name,
                    'budget': budget,
                    'n_decode': n_decode,
                    'prefill_lens': prefill_lens,
                    'total_tokens': n_decode + sum(prefill_lens),
                    'median_ms': round(median_ms, 2),
                    'transfers': xfers,
                    'transfer_mb': round(xfer_mb, 1),
                    'transfer_ms': round(xfer_ms, 1),
                })

    # ── Summary Table ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {model_name} ({L}L, {E}E)")
    print(f"{'='*70}")

    # Group by type
    for bench_type in ['prefill', 'mixed']:
        entries = [r for r in all_results if r['type'] == bench_type]
        if not entries:
            continue
        configs = sorted(set(r['config'] for r in entries),
                         key=lambda c: next(i for i, r in enumerate(entries)
                                            if r['config'] == c))
        print(f"\n  {bench_type.upper()}:")
        # Header with budget columns
        budget_hdr = "".join(f"  B={b:>3}" for b in budgets)
        print(f"  {'Config':<16} {budget_hdr}")
        print(f"  {'':─<16} {'':─<{len(budgets)*7}}")
        for config in configs:
            row = f"  {config:<16}"
            for b in budgets:
                match = [r for r in entries
                         if r['config'] == config and r['budget'] == b]
                if match:
                    row += f"  {match[0]['median_ms']:>5.1f}"
                else:
                    row += "      -"
            print(row + "  ms")

    # Save results
    if args.output:
        output_data = {
            'model': str(args.model),
            'model_name': model_name,
            'layers': L, 'experts': E,
            'max_epl': max_epl, 'budgets': budgets,
            'n_warmup': args.n_warmup, 'n_trials': args.n_trials,
            'results': all_results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
