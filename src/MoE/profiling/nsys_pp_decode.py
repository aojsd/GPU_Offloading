"""Profile pipeline-parallel decode to understand per-phase cost breakdown.

Three modes:
  20L-single  : Mixtral-8x7B-20L on 1 GPU (baseline per-layer cost)
  32L-single  : Mixtral-8x7B-32L on 1 GPU, no offloading (full model reference)
  32L-pp2     : Mixtral-8x7B-32L on 2 GPUs, PP=2 (what we're optimizing)

Usage:
  # Pure timing (no nsys):
  python tests/nsys_pp_decode.py --model-20L ../../models/Mixtral-8x7B-20L \
      --model-32L /path/to/Mixtral-8x7B --mode all

  # nsys profiling (run under nsys):
  ~/software/cuda-12.8/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
      -o /tmp/pp2_decode \
      python tests/nsys_pp_decode.py --model-32L /path/to/Mixtral-8x7B \
          --mode 32L-pp2 --nsys

  # Event-level timing per phase (no nsys needed):
  python tests/nsys_pp_decode.py --model-20L ../../models/Mixtral-8x7B-20L \
      --model-32L /path/to/Mixtral-8x7B --mode all --phase-timing
"""
import ctypes
ctypes.CDLL("/gpfs/radev/apps/avx512/software/GCCcore/13.3.0/lib64/libstdc++.so.6",
            mode=ctypes.RTLD_GLOBAL)

import os, sys, time, gc, json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from moe_engine import MoEEngine


def _sync_all():
    """Synchronize all CUDA devices."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def measure_decode(engine, n_warmup=5, n_steps=20, nsys=False,
                   phase_timing=False):
    """Run decode steps and return per-step wall-clock times (ms).

    If phase_timing=True, also returns per-phase breakdown using CUDA events
    on the primary device (only valid for single-GPU configs).
    """
    pp = engine.pp_size > 1

    with torch.inference_mode():
        # Prefill a single sequence
        prompt = torch.randint(1, 1000, (128,), device=engine.device)
        engine.reset()
        logits = engine.prefill_to_slot(0, prompt)
        next_token = logits[-1].argmax().unsqueeze(0)
        _sync_all()

        # Warmup
        for _ in range(n_warmup):
            logits = engine.step(
                decode_seq_ids=[0], decode_token_ids=next_token,
                prefill_seq_ids=[], prefill_input_ids=[])
            next_token = logits[0].argmax().unsqueeze(0)
        _sync_all()

        # Profiled steps
        if nsys:
            torch.cuda.cudart().cudaProfilerStart()

        step_times = []
        for step in range(n_steps):
            if nsys:
                torch.cuda.nvtx.range_push(f"decode_step_{step}")

            _sync_all()
            t0 = time.perf_counter()
            logits = engine.step(
                decode_seq_ids=[0], decode_token_ids=next_token,
                prefill_seq_ids=[], prefill_input_ids=[])
            _sync_all()
            step_times.append((time.perf_counter() - t0) * 1000)

            next_token = logits[0].argmax().unsqueeze(0)
            if nsys:
                torch.cuda.nvtx.range_pop()

        if nsys:
            torch.cuda.cudart().cudaProfilerStop()

    return step_times


def measure_decode_phase_timing(engine, n_warmup=5, n_steps=20):
    """Measure per-phase timing using torch.cuda.Event on primary device.

    Returns dict with per-phase median times. For PP configs, events are
    placed on GPU 0 for GPU 0 layers and GPU 1 for GPU 1 layers.
    """
    pp = engine.pp_size > 1
    dev = engine.device

    with torch.inference_mode():
        # Prefill
        prompt = torch.randint(1, 1000, (128,), device=dev)
        engine.reset()
        logits = engine.prefill_to_slot(0, prompt)
        next_token = logits[-1].argmax().unsqueeze(0)
        _sync_all()

        # Warmup
        for _ in range(n_warmup):
            logits = engine.step(
                decode_seq_ids=[0], decode_token_ids=next_token,
                prefill_seq_ids=[], prefill_input_ids=[])
            next_token = logits[0].argmax().unsqueeze(0)
        _sync_all()

        # We'll manually time phases by inserting sync points around
        # the piecewise graph replay. We do this by calling the internal
        # method step-by-step with timing.

        # For simplicity: enable NVTX, use wall-clock with sync barriers
        # between sections. This is invasive but gives us the breakdown.

        info = engine._piecewise_graphs[1]  # graph for N=1
        H = engine.hidden_size

        phase_records = []
        for step in range(n_warmup + n_steps):
            record = {}

            # Setup: token/position/slot_mapping copy + FlashInfer plan
            _sync_all()
            t0 = time.perf_counter()

            # Increment seq_lens and plan FlashInfer
            engine._seq_lens_cpu[0] += 1
            if pp:
                for gpu_idx in range(engine.pp_size):
                    engine._plan_flashinfer_decode_pp([0], gpu_idx)
            else:
                engine._plan_flashinfer_decode_for_subset([0])

            # Copy token into static buffers
            if pp:
                for gpu_idx in range(engine.pp_size):
                    b = info['pp_bufs'][gpu_idx]
                    b['static_token_ids'][0] = next_token[0]
                    b['static_positions'][0] = engine._seq_lens_cpu[0] - 1
                    # Slot mapping
                    bt = engine.block_table[gpu_idx]
                    pos = engine._seq_lens_cpu[0] - 1
                    pg = pos // engine.page_size
                    off = pos % engine.page_size
                    b['static_slot_mapping'][0] = (
                        bt[0, pg].long() * engine.page_size + off)
            else:
                info['static_token_ids'][0] = next_token[0]
                info['static_positions'][0] = engine._seq_lens_cpu[0] - 1
                pos = engine._seq_lens_cpu[0] - 1
                pg = pos // engine.page_size
                off = pos % engine.page_size
                info['static_slot_mapping'][0] = (
                    engine.block_table[0, pg].long() * engine.page_size + off)

            _sync_all()
            record['setup'] = (time.perf_counter() - t0) * 1000

            # Embed
            def _buf(layer):
                if pp:
                    return info['pp_bufs'][engine.pp_layer_gpu[layer]]
                return info

            _sync_all()
            t0 = time.perf_counter()
            buf0 = _buf(0)
            buf0['hidden_buf'].copy_(
                torch.nn.functional.embedding(
                    buf0['static_token_ids'], engine.embed_tokens))
            _sync_all()
            record['embed'] = (time.perf_counter() - t0) * 1000

            # Per-layer timing
            layer_times = []
            for layer in range(engine.num_layers):
                lt = {}
                buf = _buf(layer)

                # PP transfer
                if pp and layer in engine.pp_boundaries:
                    _sync_all()
                    t0 = time.perf_counter()
                    prev_gpu = engine.pp_layer_gpu[layer - 1]
                    buf['hidden_buf'].copy_(
                        info['pp_bufs'][prev_gpu]['hidden_buf'])
                    _sync_all()
                    lt['pp_xfer'] = (time.perf_counter() - t0) * 1000

                # Stage 1
                _sync_all()
                t0 = time.perf_counter()
                info['stage1_graphs'][layer].replay()
                _sync_all()
                lt['stage1'] = (time.perf_counter() - t0) * 1000

                # Stage 2 (FlashInfer decode)
                _sync_all()
                t0 = time.perf_counter()
                if pp:
                    gpu_idx = engine.pp_layer_gpu[layer]
                    wrapper = engine._decode_wrappers[gpu_idx]
                else:
                    wrapper = engine._decode_wrapper
                q_decode = buf['q_buf'][:1]
                decode_out = wrapper.run(
                    q_decode,
                    (engine.k_cache[layer], engine.v_cache[layer]))
                buf['attn_out_buf'][:1].copy_(decode_out.reshape(1, H))
                _sync_all()
                lt['stage2'] = (time.perf_counter() - t0) * 1000

                # Stage 4a
                _sync_all()
                t0 = time.perf_counter()
                info['stage4a_graphs'][layer].replay()
                _sync_all()
                lt['stage4a'] = (time.perf_counter() - t0) * 1000

                # Stage 4b
                _sync_all()
                t0 = time.perf_counter()
                info['stage4b_graphs'][layer].replay()
                _sync_all()
                lt['stage4b'] = (time.perf_counter() - t0) * 1000

                layer_times.append(lt)
            record['layers'] = layer_times

            # Final
            _sync_all()
            t0 = time.perf_counter()
            last_buf = _buf(engine.num_layers - 1)
            hidden = last_buf['hidden_buf']
            hidden = torch.nn.functional.rms_norm(
                hidden, (engine.hidden_size,), engine.final_norm,
                engine.rms_norm_eps)
            logits = torch.nn.functional.linear(hidden, engine.lm_head)
            if pp:
                logits = logits.to(engine.pp_devices[0])
            _sync_all()
            record['final'] = (time.perf_counter() - t0) * 1000

            # Update seq_lens on GPU
            if pp:
                for sl in engine.seq_lens:
                    sl[0] += 1
            else:
                engine.seq_lens[0] += 1

            next_token = logits[0].argmax().unsqueeze(0)

            if step >= n_warmup:
                phase_records.append(record)

    return phase_records


def print_phase_summary(records, label, num_layers):
    """Print summary table from phase timing records."""
    import statistics

    n = len(records)
    print(f"\n{'='*70}")
    print(f"Phase Timing: {label} ({n} steps)")
    print(f"{'='*70}")

    # Aggregate per-step totals
    setup_times = [r['setup'] for r in records]
    embed_times = [r['embed'] for r in records]
    final_times = [r['final'] for r in records]

    # Per-layer aggregation
    stage_totals = {'stage1': [], 'stage2': [], 'stage4a': [], 'stage4b': []}
    pp_xfer_times = []
    per_layer_totals = []

    for r in records:
        layer_sum = 0
        s1_sum = s2_sum = s4a_sum = s4b_sum = 0
        pp_sum = 0
        for lt in r['layers']:
            s1_sum += lt['stage1']
            s2_sum += lt['stage2']
            s4a_sum += lt['stage4a']
            s4b_sum += lt['stage4b']
            pp = lt.get('pp_xfer', 0)
            pp_sum += pp
            layer_sum += lt['stage1'] + lt['stage2'] + lt['stage4a'] + lt['stage4b'] + pp
        stage_totals['stage1'].append(s1_sum)
        stage_totals['stage2'].append(s2_sum)
        stage_totals['stage4a'].append(s4a_sum)
        stage_totals['stage4b'].append(s4b_sum)
        pp_xfer_times.append(pp_sum)
        per_layer_totals.append(layer_sum)

    total_step = [r['setup'] + r['embed'] + sum(
        lt['stage1'] + lt['stage2'] + lt['stage4a'] + lt['stage4b'] +
        lt.get('pp_xfer', 0) for lt in r['layers']) + r['final']
        for r in records]

    def _stats(vals):
        return f"{statistics.median(vals):.3f} (mean {statistics.mean(vals):.3f})"

    print(f"  Total step:       {_stats(total_step)} ms")
    print(f"  Setup:            {_stats(setup_times)} ms")
    print(f"  Embed:            {_stats(embed_times)} ms")
    print(f"  Stage1 (all L):   {_stats(stage_totals['stage1'])} ms")
    print(f"  Stage2 (all L):   {_stats(stage_totals['stage2'])} ms")
    print(f"  Stage4a (all L):  {_stats(stage_totals['stage4a'])} ms")
    print(f"  Stage4b (all L):  {_stats(stage_totals['stage4b'])} ms")
    if any(t > 0 for t in pp_xfer_times):
        print(f"  PP transfers:     {_stats(pp_xfer_times)} ms")
    print(f"  Final:            {_stats(final_times)} ms")
    print(f"  Layer compute:    {_stats(per_layer_totals)} ms")
    print(f"  Per-layer avg:    {statistics.median(per_layer_totals)/num_layers:.3f} ms")

    # Per-layer breakdown (median across steps)
    print(f"\n  Per-layer breakdown (median across {n} steps):")
    print(f"  {'Layer':>5} {'Stage1':>8} {'Stage2':>8} {'Stage4a':>8} "
          f"{'Stage4b':>8} {'PP_xfer':>8} {'Total':>8}")
    for l in range(num_layers):
        s1 = statistics.median([r['layers'][l]['stage1'] for r in records])
        s2 = statistics.median([r['layers'][l]['stage2'] for r in records])
        s4a = statistics.median([r['layers'][l]['stage4a'] for r in records])
        s4b = statistics.median([r['layers'][l]['stage4b'] for r in records])
        pp = statistics.median([r['layers'][l].get('pp_xfer', 0)
                                for r in records])
        tot = s1 + s2 + s4a + s4b + pp
        print(f"  {l:5d} {s1:8.3f} {s2:8.3f} {s4a:8.3f} "
              f"{s4b:8.3f} {pp:8.3f} {tot:8.3f}")

    return {
        'label': label,
        'num_layers': num_layers,
        'median_step_ms': statistics.median(total_step),
        'median_per_layer_ms': statistics.median(per_layer_totals) / num_layers,
        'median_setup_ms': statistics.median(setup_times),
        'median_embed_ms': statistics.median(embed_times),
        'median_final_ms': statistics.median(final_times),
        'median_stage1_ms': statistics.median(stage_totals['stage1']),
        'median_stage2_ms': statistics.median(stage_totals['stage2']),
        'median_stage4a_ms': statistics.median(stage_totals['stage4a']),
        'median_stage4b_ms': statistics.median(stage_totals['stage4b']),
        'median_pp_xfer_ms': statistics.median(pp_xfer_times),
    }


def run_config(model_path, mode, n_warmup, n_steps, nsys, phase_timing,
               use_torch_compile):
    """Create engine, capture graphs, measure decode."""
    print(f"\n{'#'*70}")
    print(f"# Mode: {mode}")
    print(f"# Model: {model_path}")
    print(f"# torch.compile: {use_torch_compile}")
    print(f"{'#'*70}")

    pp_size = 2 if mode == "32L-pp2" else 1

    engine = MoEEngine(
        model_path, device="cuda:0",
        pipeline_parallel_size=pp_size,
    )

    with torch.inference_mode():
        if pp_size == 1:
            engine.capture_prefill_cuda_graph(
                total_token_sizes=[128],
                use_torch_compile=use_torch_compile)
            engine.reset()
        engine.capture_cuda_graphs(
            total_token_sizes=[1, 128],
            use_torch_compile=use_torch_compile)

    if nsys:
        engine._nvtx_enabled = True

    # Wall-clock timing
    import statistics
    step_times = measure_decode(engine, n_warmup, n_steps, nsys=nsys)
    med = statistics.median(step_times)
    mean = statistics.mean(step_times)
    per_layer = med / engine.num_layers
    print(f"\n  Wall-clock: median={med:.2f}ms  mean={mean:.2f}ms  "
          f"per_layer={per_layer:.3f}ms  ({engine.num_layers}L)")

    result = {
        'mode': mode,
        'num_layers': engine.num_layers,
        'pp_size': pp_size,
        'compile': use_torch_compile,
        'median_ms': med,
        'mean_ms': mean,
        'per_layer_ms': per_layer,
        'step_times': step_times,
    }

    # Phase timing (detailed breakdown)
    phase_result = None
    if phase_timing:
        engine.reset()
        records = measure_decode_phase_timing(engine, n_warmup, n_steps)
        phase_result = print_phase_summary(records, mode, engine.num_layers)

    # Cleanup
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return result, phase_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-20L", default=None,
                        help="Path to Mixtral-8x7B-20L")
    parser.add_argument("--model-32L", default=None,
                        help="Path to Mixtral-8x7B (32L)")
    parser.add_argument("--mode", default="all",
                        choices=["20L-single", "32L-single", "32L-pp2", "all"])
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--nsys", action="store_true",
                        help="Enable NVTX + cudaProfiler for nsys")
    parser.add_argument("--phase-timing", action="store_true",
                        help="Detailed per-phase timing with sync barriers")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile")
    args = parser.parse_args()

    modes = ([args.mode] if args.mode != "all"
             else ["20L-single", "32L-single", "32L-pp2"])

    results = {}
    for mode in modes:
        if mode == "20L-single":
            if not args.model_20L:
                print(f"Skipping {mode}: --model-20L not provided")
                continue
            model = args.model_20L
        else:
            if not args.model_32L:
                print(f"Skipping {mode}: --model-32L not provided")
                continue
            model = args.model_32L

        r, pr = run_config(model, mode, args.n_warmup, args.n_steps,
                           args.nsys, args.phase_timing, args.compile)
        results[mode] = {'wall_clock': r, 'phases': pr}

    # Comparison summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Mode':<15} {'Layers':>6} {'Median':>10} {'Per-Layer':>10}")
        for mode, r in results.items():
            wc = r['wall_clock']
            print(f"  {mode:<15} {wc['num_layers']:>6} "
                  f"{wc['median_ms']:>9.2f}ms {wc['per_layer_ms']:>9.3f}ms")

        # Compare PP per-layer cost vs 20L baseline
        if '20L-single' in results and '32L-pp2' in results:
            base = results['20L-single']['wall_clock']['per_layer_ms']
            pp = results['32L-pp2']['wall_clock']['per_layer_ms']
            overhead = (pp / base - 1) * 100
            print(f"\n  PP overhead vs 20L baseline: "
                  f"{pp:.3f} vs {base:.3f} ms/layer = {overhead:+.1f}%")


if __name__ == "__main__":
    main()
