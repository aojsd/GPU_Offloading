"""Comprehensive benchmark: stage4b compute kernel timing across all configs.

Tests that fused_experts compute kernel performance is UNCHANGED between
all-resident (experts_per_layer=E) and offloading (experts_per_layer<E with offload engine) configurations.

Measures per-layer stage4b CUDA graph replay time — the compute-only cost
of the MoE kernel after expert weights are already in the unified buffer.

Usage:
  # Single config
  python tests/bench_comprehensive.py --model ../../models/Mixtral-8x7B --experts-per-layer 4 --batch 1

  # Full sweep for one model (groups by model+experts_per_layer to minimize reloads)
  python tests/bench_comprehensive.py --model ../../models/Mixtral-8x7B --sweep

  # Full sweep for all models
  python tests/bench_comprehensive.py --sweep-all
"""
import ctypes
ctypes.CDLL("/gpfs/radev/apps/avx512/software/GCCcore/13.3.0/lib64/libstdc++.so.6",
            mode=ctypes.RTLD_GLOBAL)

import os, sys, gc, json, argparse, traceback
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from moe_engine import MoEEngine


def measure_stage4b(engine, batch_size, n_steps=10):
    """Measure per-layer stage4b replay times with precise CUDA events.

    Handles both all-resident (offload_engine=None) and offloading cases.
    Returns dict[layer] -> list of durations (microseconds).
    """
    oe = engine.offload_engine
    info = engine._piecewise_graphs[batch_size]
    num_layers = engine.num_layers
    H = engine.hidden_size

    # Prefill batch_size sequences (use separate slots, each with 128-token prompt)
    engine.reset()
    next_tokens = []
    for seq_id in range(batch_size):
        prompt = torch.randint(1, 1000, (128,), device="cuda")
        logits = engine.prefill_to_slot(seq_id, prompt)
        next_tokens.append(logits[-1].argmax().unsqueeze(0))
    next_token_batch = torch.stack([t.squeeze() for t in next_tokens])

    # Warmup decode steps (use high-level API to ensure everything is set up)
    decode_seq_ids = list(range(batch_size))
    for i in range(5):
        # begin_step auto-called inside mixed_step
        logits = engine.mixed_step(
            decode_seq_ids=decode_seq_ids,
            decode_token_ids=next_token_batch,
            prefill_seq_ids=[], prefill_input_ids=[])
        next_token_batch = logits[:batch_size].argmax(dim=-1)

    if oe:
        oe.reset_trace()

    # Measure: manual per-layer replay with precise stage4b timing
    layer_times = {l: [] for l in range(num_layers)}

    for step in range(n_steps):
        if oe:
            oe.begin_step()  # manual replay — not auto-called

        # Copy tokens into static buffer
        info['static_token_ids'][:batch_size].copy_(next_token_batch)
        if batch_size < info['static_token_ids'].shape[0]:
            info['static_token_ids'][batch_size:].zero_()

        # Compute positions
        decode_positions = engine._seq_lens_cpu[decode_seq_ids].to(
            torch.int32).to(engine.device)
        info['static_positions'][:batch_size].copy_(decode_positions)
        if batch_size < info['static_positions'].shape[0]:
            info['static_positions'][batch_size:].zero_()

        # Compute slot mapping
        d_idx = torch.tensor(decode_seq_ids, device=engine.device,
                             dtype=torch.long)
        d_page = (decode_positions // engine.page_size).long()
        d_offset = (decode_positions % engine.page_size).long()
        slot = (engine.block_table[d_idx, d_page].long()
                * engine.page_size + d_offset)
        info['static_slot_mapping'][:batch_size].copy_(slot)
        if batch_size < info['static_slot_mapping'].shape[0]:
            info['static_slot_mapping'][batch_size:].fill_(-1)

        # Increment seq lens and plan FlashInfer
        for sid in decode_seq_ids:
            engine._seq_lens_cpu[sid] += 1
        engine._plan_flashinfer_decode_for_subset(decode_seq_ids)

        # Embed
        info['hidden_buf'].copy_(
            F.embedding(info['static_token_ids'], engine.embed_tokens))

        q_buf = info['q_buf']
        attn_out_buf = info['attn_out_buf']

        for layer in range(num_layers):
            # Stage 1
            info['stage1_graphs'][layer].replay()

            # Stage 2: decode attention
            q_decode = q_buf[:batch_size]
            if engine.is_mla:
                q_nope_h = q_decode.view(batch_size, engine.num_heads,
                                         engine.qk_nope_head_dim)
                q_pe_h = info.get('q_pe_buf', q_buf)[:batch_size]
                q_pe_h = q_pe_h.view(batch_size, engine.num_heads,
                                     engine.qk_rope_head_dim)
                q_absorbed = torch.einsum('bhp,hpc->bhc', q_nope_h,
                                          engine.W_UK_T[layer])
                out = engine._mla_decode_wrapper.run(
                    q_absorbed, q_pe_h,
                    engine.ckv_cache[layer], engine.kpe_cache[layer])
                attn_v = torch.einsum('bhc,hcv->bhv', out, engine.W_UV[layer])
                decode_out = attn_v.reshape(batch_size,
                                            engine.num_heads * engine.v_head_dim)
            else:
                decode_out = engine._decode_wrapper.run(
                    q_decode, (engine.k_cache[layer], engine.v_cache[layer]))
            attn_out_buf[:batch_size].copy_(decode_out.reshape(batch_size, H))

            # Zero padding
            if batch_size < q_buf.shape[0]:
                attn_out_buf[batch_size:].zero_()

            # Stage 4a: router
            info['stage4a_graphs'][layer].replay()

            # Demand-load missing experts (also updates expert_map_buf)
            if oe:
                oe.process_layer(
                    layer, info['topk_ids_buf'], batch_size)

            # Stage 4b: MoE compute — TIMED
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            info['stage4b_graphs'][layer].replay()
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_us = start_evt.elapsed_time(end_evt) * 1000
            layer_times[layer].append(elapsed_us)

            if oe:
                oe.post_layer(layer)

        # Update GPU seq_lens
        for sid in decode_seq_ids:
            engine.seq_lens[sid] += 1

        # Get next token
        hidden = info['hidden_buf']
        hidden = F.rms_norm(hidden, (engine.hidden_size,), engine.final_norm,
                            engine.rms_norm_eps)
        logits = F.linear(hidden, engine.lm_head)
        next_token_batch = logits[:batch_size].argmax(dim=-1)

    return layer_times


def run_grouped_configs(model_path, experts_per_layer, batch_sizes,
                        n_steps=10, use_compile=True):
    """Run benchmark for one (model, experts_per_layer), testing all batch_sizes.

    Loads the engine ONCE, captures graphs for all batch sizes, then measures
    each batch size sequentially. Returns list of result dicts.
    """
    print(f"\n{'─'*80}")
    print(f"  Loading: experts_per_layer={experts_per_layer}, kernel=Triton")

    engine = MoEEngine(
        model_path, device="cuda:0",
        experts_per_layer=experts_per_layer,
    )
    num_layers = engine.num_layers
    num_experts = engine.num_experts
    is_offloading = experts_per_layer < num_experts

    print(f"  L={num_layers}, E={num_experts}, "
          f"experts_per_layer={experts_per_layer}, "
          f"offloading={'yes' if is_offloading else 'no (all-resident)'}")

    # Capture graphs for all batch sizes at once.
    # When offloading is active, prefill_to_slot routes through mixed_step,
    # so piecewise graphs must also cover the prefill size (128 tokens).
    piecewise_sizes = sorted(set(batch_sizes) | {128}) if is_offloading else batch_sizes
    with torch.inference_mode():
        engine.capture_prefill_cuda_graph(
            total_token_sizes=[128],
            use_torch_compile=use_compile)
        engine.reset()
        engine.capture_mixed_cuda_graphs(
            total_token_sizes=piecewise_sizes,
            use_torch_compile=use_compile)

    oe = engine.offload_engine

    results = []
    for batch_size in batch_sizes:
        print(f"  Measuring batch={batch_size}...", end=" ", flush=True)

        if oe:
            oe.reset_trace()

        with torch.inference_mode():
            layer_times = measure_stage4b(
                engine, batch_size, n_steps=n_steps)

        # Compute stats
        all_medians = []
        for layer in range(num_layers):
            times = sorted(layer_times[layer])
            median = times[len(times) // 2]
            all_medians.append(median)

        overall_median = sorted(all_medians)[len(all_medians) // 2]
        total_ms = sum(all_medians) / 1000

        transfer_stats = None
        if oe:
            transfer_stats = oe.get_transfer_stats()

        result = {
            'model': model_path,
            'num_layers': num_layers,
            'num_experts': num_experts,
            'experts_per_layer': experts_per_layer,
            'batch_size': batch_size,
            'kernel': 'Triton',
            'offloading': is_offloading,
            'median_per_layer_us': overall_median,
            'total_ms': total_ms,
            'layer_medians': all_medians,
            'transfer_stats': transfer_stats,
        }
        results.append(result)

        miss_info = ""
        if transfer_stats and transfer_stats['total_transfers'] > 0:
            miss_info = f", miss_rate={transfer_stats['miss_rate']:.0%}"
        print(f"{overall_median:.1f} us/layer, {total_ms:.2f} ms total"
              f"{miss_info}")

    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return results


def get_model_info(model_path):
    """Read model config and compute memory info."""
    config = json.load(open(os.path.join(model_path, "config.json")))
    num_layers = config.get('num_hidden_layers', 32)
    num_experts = (config.get('n_routed_experts')
                   or config.get('num_experts')
                   or config.get('num_local_experts', 8))
    hidden_size = config.get('hidden_size', 4096)
    intermediate_size = config.get('moe_intermediate_size',
                                   config.get('intermediate_size', 14336))
    top_k = config.get('num_experts_per_tok', 2)

    # Per-expert BF16 memory
    w1_bytes = 2 * intermediate_size * hidden_size * 2
    w2_bytes = hidden_size * intermediate_size * 2
    expert_bytes = w1_bytes + w2_bytes

    return {
        'num_layers': num_layers,
        'num_experts': num_experts,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'top_k': top_k,
        'expert_bytes': expert_bytes,
    }


def get_experts_per_layer_values(model_info):
    """Determine viable experts_per_layer values for a model."""
    L = model_info['num_layers']
    E = model_info['num_experts']
    expert_bytes = model_info['expert_bytes']
    top_k = model_info['top_k']

    # GPU budget: 80 GB H100, reserve ~15 GB for non-expert state
    gpu_budget = 65e9

    viable = []
    for n in range(2, E + 1):
        total_slots = L * n + 8
        mem = total_slots * expert_bytes
        if mem <= gpu_budget:
            viable.append(n)

    if not viable:
        return [2]  # at least try experts_per_layer=2

    # For many experts (OLMoE: E=64), sample key values
    if E > 16:
        sampled = set()
        sampled.add(viable[0])      # smallest viable
        sampled.add(top_k)          # covers one token's selections
        sampled.add(min(top_k * 2, E))  # 2x top_k
        sampled.add(min(top_k * 4, E))  # 4x top_k
        if E in viable:
            sampled.add(E)          # all-resident baseline
        return sorted(v for v in sampled if v in viable or v <= viable[-1])
    else:
        # For small E (Mixtral: E=8), test all viable values
        return viable


def print_summary_table(results, model_info):
    """Print formatted comparison table grouped by (batch, kernel)."""
    L = model_info['num_layers']
    E = model_info['num_experts']

    # Group by (batch_size, kernel)
    groups = {}
    for r in results:
        key = (r['batch_size'], r['kernel'])
        groups.setdefault(key, []).append(r)

    model_name = results[0]['model'].split('/')[-1]
    print(f"\n{'='*95}")
    print(f"  SUMMARY: {model_name} ({L}L, E={E}, top_k={model_info['top_k']})")
    print(f"{'='*95}")

    for (batch_size, kernel), group in sorted(groups.items()):
        group.sort(key=lambda r: r['experts_per_layer'])

        # Find baseline (all-resident = experts_per_layer == E)
        baseline = next((r for r in group
                         if r['experts_per_layer'] == E), None)

        print(f"\n  Batch={batch_size}, Kernel={kernel}")
        print(f"  {'EPL':>5} {'Mode':>14} {'us/layer':>10} "
              f"{'total(ms)':>10} {'vs base':>9} {'misses':>8}")
        print(f"  {'-'*62}")

        for r in group:
            mode = "all-res" if not r['offloading'] else f"offload"
            us = f"{r['median_per_layer_us']:.1f}"
            ms = f"{r['total_ms']:.2f}"

            if baseline and r != baseline:
                ratio = r['median_per_layer_us'] / baseline['median_per_layer_us']
                vs = f"{ratio:.3f}x"
            elif r == baseline:
                vs = "(base)"
            else:
                vs = ""

            miss = ""
            ts = r.get('transfer_stats')
            if ts and ts['total_transfers'] > 0:
                miss = f"{ts['miss_rate']:.0%}"

            print(f"  {r['experts_per_layer']:>5} {mode:>14} {us:>10} "
                  f"{ms:>10} {vs:>9} {miss:>8}")

    # Parity check
    print(f"\n  {'─'*62}")
    print(f"  PARITY CHECK:")
    all_ok = True
    for (batch_size, kernel), group in sorted(groups.items()):
        baseline = next((r for r in group
                         if r['experts_per_layer'] == E), None)
        if not baseline:
            # No all-resident baseline; check consistency across values
            offloading_runs = [r for r in group if r['offloading']]
            if len(offloading_runs) >= 2:
                ref = offloading_runs[0]
                for r in offloading_runs[1:]:
                    ratio = r['median_per_layer_us'] / ref['median_per_layer_us']
                    ok = 0.85 <= ratio <= 1.15
                    status = "OK" if ok else "DIFF"
                    if not ok:
                        all_ok = False
                    print(f"    B={batch_size} {kernel}: "
                          f"experts_per_layer={r['experts_per_layer']} vs "
                          f"experts_per_layer={ref['experts_per_layer']}: "
                          f"{ratio:.3f}x {status}")
            continue

        for r in group:
            if r['offloading']:
                ratio = r['median_per_layer_us'] / baseline['median_per_layer_us']
                ok = 0.85 <= ratio <= 1.15
                status = "OK" if ok else "DIFF"
                if not ok:
                    all_ok = False
                print(f"    B={batch_size} {kernel}: "
                      f"experts_per_layer={r['experts_per_layer']} vs "
                      f"all-res: {ratio:.3f}x {status}")

    verdict = "ALL CHECKS PASSED" if all_ok else "SOME CHECKS DIFFER"
    print(f"\n  {verdict} (15% tolerance)")


def run_sweep(model_path, n_steps=10, use_compile=True):
    """Run full sweep for one model, returning all results."""
    model_info = get_model_info(model_path)
    epl_values = get_experts_per_layer_values(model_info)
    batch_sizes = [1, 16, 32]

    model_name = model_path.split('/')[-1]
    print(f"\n{'#'*95}")
    print(f"  MODEL: {model_name}")
    print(f"  L={model_info['num_layers']}, E={model_info['num_experts']}, "
          f"top_k={model_info['top_k']}, "
          f"{model_info['expert_bytes']/1e6:.1f} MB/expert")
    print(f"  experts_per_layer values: {epl_values}")
    print(f"  batch sizes: {batch_sizes}")
    print(f"  kernel: Triton")
    total = len(epl_values)
    print(f"  Total engine loads: {total}")
    print(f"{'#'*95}")

    all_results = []
    for experts_per_layer in epl_values:
        try:
            results = run_grouped_configs(
                model_path, experts_per_layer, batch_sizes,
                n_steps=n_steps, use_compile=use_compile)
            all_results.extend(results)
        except Exception as e:
            print(f"  FAILED experts_per_layer={experts_per_layer} "
                  f"Triton: {e}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    print_summary_table(all_results, model_info)
    return all_results, model_info


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive MoE stage4b kernel benchmark")
    parser.add_argument("--model", type=str,
                        help="Path to model directory")
    parser.add_argument("--experts-per-layer", type=int,
                        help="Experts per layer")
    parser.add_argument("--batch", type=int, default=1,
                        help="Decode batch size")
    parser.add_argument("--n-steps", type=int, default=10,
                        help="Decode steps to measure")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep for --model")
    parser.add_argument("--sweep-all", action="store_true",
                        help="Run full sweep for all models")
    parser.add_argument("--output", type=str,
                        help="Save results to JSON file")
    args = parser.parse_args()

    use_compile = not args.no_compile

    if args.sweep_all:
        models = [
            "../../models/OLMoE-1B-7B",
            "../../models/Mixtral-8x7B-20L",
            "../../models/Mixtral-8x7B",
        ]
        all_results = {}
        for model in models:
            results, info = run_sweep(
                model, n_steps=args.n_steps, use_compile=use_compile)
            all_results[model] = {
                'model_info': info,
                'results': results,
            }

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")

    elif args.sweep:
        assert args.model, "--model required for --sweep"
        results, info = run_sweep(
            args.model, n_steps=args.n_steps, use_compile=use_compile)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump({'model_info': info, 'results': results},
                          f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")

    else:
        assert args.model and args.experts_per_layer, \
            "--model and --experts-per-layer required"
        results = run_grouped_configs(
            args.model, args.experts_per_layer, [args.batch],
            n_steps=args.n_steps, use_compile=use_compile)
        r = results[0]
        print(f"\n  Result: {r['median_per_layer_us']:.1f} us/layer, "
              f"{r['total_ms']:.2f} ms total")


if __name__ == "__main__":
    main()
