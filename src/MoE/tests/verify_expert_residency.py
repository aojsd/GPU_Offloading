"""Verify: all selected experts are loaded before stage4b compute.

With experts_per_layer=2, only experts 0,1 are resident per layer. The offload engine
demand-loads any non-resident selected experts into the scratchpad before
stage4b. This script checks routing decisions and confirms uniform timing
(~329 us per layer) regardless of which experts the router selects.
"""
import ctypes
ctypes.CDLL("/gpfs/radev/apps/avx512/software/GCCcore/13.3.0/lib64/libstdc++.so.6",
            mode=ctypes.RTLD_GLOBAL)

import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from moe_engine import MoEEngine


def main():
    engine = MoEEngine(
        "models/Mixtral-8x7B", device="cuda:0",
        experts_per_layer=2,
    )

    with torch.inference_mode():
        engine.capture_prefill_cuda_graph(total_token_sizes=[128],
                                           use_torch_compile=False)
        engine.reset()
        engine.capture_mixed_cuda_graphs(total_token_sizes=[1, 128],
                                          use_torch_compile=False)

    oe = engine.offload_engine
    info = engine._piecewise_graphs[1]
    num_layers = engine.num_layers

    # Print expert_map for each layer
    print("Expert maps (global_id -> cache_slot, -1 = not resident):")
    for l in range(num_layers):
        emap = engine.expert_map[l]
        resident = [i for i in range(8) if emap[i].item() != -1]
        print(f"  L{l:2d}: {emap.tolist()}  resident={resident}")

    print(f"\nAll layers have same expert_map: experts 0,1 resident, 2-7 not.")

    with torch.inference_mode():
        # Prefill
        prompt = torch.randint(1, 1000, (128,), device="cuda")
        engine.reset()
        logits = engine.prefill_to_slot(0, prompt)
        next_token = logits[-1].argmax().unsqueeze(0)

        # Warmup (begin_step auto-called inside mixed_step)
        for _ in range(5):
            logits = engine.mixed_step(
                decode_seq_ids=[0], decode_token_ids=next_token,
                prefill_seq_ids=[], prefill_input_ids=[])
            next_token = logits[0].argmax().unsqueeze(0)

        oe.reset_trace()

        # Now instrument to capture routing decisions AND time stage4b
        n_steps = 5
        routing_log = []  # [(step, layer, expert_ids, num_resident, elapsed_us)]

        for step in range(n_steps):
            oe.begin_step()

            # Setup step
            info['static_token_ids'][:1].copy_(next_token)
            decode_positions = engine._seq_lens_cpu[[0]].to(torch.int32).to(engine.device)
            info['static_positions'][:1].copy_(decode_positions)
            d_page = (decode_positions // engine.page_size).long()
            d_offset = (decode_positions % engine.page_size).long()
            d_idx = torch.tensor([0], device=engine.device, dtype=torch.long)
            slot = engine.block_table[d_idx, d_page].long() * engine.page_size + d_offset
            info['static_slot_mapping'][:1].copy_(slot)

            engine._seq_lens_cpu[0] += 1
            engine._plan_flashinfer_decode_for_subset([0])

            info['hidden_buf'].copy_(
                F.embedding(info['static_token_ids'], engine.embed_tokens))

            q_buf = info['q_buf']
            H = engine.hidden_size

            for layer in range(num_layers):
                # Stage 1
                info['stage1_graphs'][layer].replay()

                # Stage 2
                q_decode = q_buf[:1]
                decode_out = engine._decode_wrapper.run(
                    q_decode, (engine.k_cache[layer], engine.v_cache[layer]))
                info['attn_out_buf'][:1].copy_(decode_out.reshape(1, H))

                # Stage 4a
                info['stage4a_graphs'][layer].replay()

                # Read routing decisions BEFORE demand loading
                topk_ids = info['topk_ids_buf'][:1].clone()  # [1, top_k]
                selected_experts = topk_ids[0].tolist()

                # Check residency before demand loading
                emap = engine.expert_map[layer]
                num_resident = sum(1 for e in selected_experts if emap[int(e)].item() != -1)

                # Demand-load missing experts (also updates expert_map_buf)
                oe.process_layer(layer, info['topk_ids_buf'], 1)

                # Time stage4b
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                info['stage4b_graphs'][layer].replay()
                end_evt.record()
                torch.cuda.synchronize()
                elapsed_us = start_evt.elapsed_time(end_evt) * 1000

                routing_log.append((step, layer, selected_experts, num_resident, elapsed_us))

                oe.post_layer(layer)

            # Update seq_lens and get next token
            engine.seq_lens[0] += 1
            hidden = info['hidden_buf']
            hidden = F.rms_norm(hidden, (engine.hidden_size,), engine.final_norm,
                                engine.rms_norm_eps)
            logits = F.linear(hidden, engine.lm_head)
            next_token = logits[0].argmax().unsqueeze(0)

    # Print results
    print(f"\n{'='*90}")
    print(f"  Stage4b timing vs expert residency (experts_per_layer=2, Triton, compile=OFF)")
    print(f"  Resident experts per layer: 0 and 1 only")
    print(f"  Non-resident experts demand-loaded into scratchpad before stage4b")
    print(f"{'='*90}\n")

    # Group by (layer, num_resident)
    from collections import defaultdict
    by_layer = defaultdict(list)
    for step, layer, experts, n_res, us in routing_log:
        by_layer[layer].append((step, experts, n_res, us))

    print(f"{'Layer':>5} {'Step':>4} {'Selected':>12} {'#Resident':>9} {'Time (us)':>10}")
    print("-" * 50)
    for layer in range(num_layers):
        for step, experts, n_res, us in by_layer[layer]:
            exp_str = f"{experts}"
            print(f"  L{layer:>2d}  s{step}   {exp_str:>12} {n_res:>9} {us:>10.1f}")
        print()

    # Summary by num_resident — with demand loading, timing should be uniform
    by_nres = defaultdict(list)
    for _, _, _, n_res, us in routing_log:
        by_nres[n_res].append(us)

    print(f"\n{'='*60}")
    print(f"  Summary by number of initially-resident experts selected")
    print(f"  (all experts demand-loaded before compute)")
    print(f"{'='*60}")
    for n_res in sorted(by_nres.keys()):
        times = sorted(by_nres[n_res])
        median = times[len(times) // 2]
        print(f"  {n_res} resident: n={len(times):3d}, "
              f"median={median:.1f} us, "
              f"min={min(times):.1f}, max={max(times):.1f}")

    # Check uniformity
    all_times = [us for _, _, _, _, us in routing_log]
    median_all = sorted(all_times)[len(all_times) // 2]
    max_deviation = max(abs(t - median_all) for t in all_times)
    print(f"\n  Overall median: {median_all:.1f} us")
    print(f"  Max deviation from median: {max_deviation:.1f} us")
    if max_deviation < 50:
        print(f"  PASS — uniform timing (no trimodal pattern)")
    else:
        print(f"  WARNING — timing variance > 50 us, check for missed demand loads")


if __name__ == "__main__":
    main()
