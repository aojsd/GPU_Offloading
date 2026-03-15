"""DP+EP launcher for MoE inference.

Usage:
    torchrun --nproc_per_node=2 launch_ep.py \
        --model /path/to/mixtral --ep_size 2 \
        --prompts prompts.jsonl
"""
import argparse, os, json, sys
import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep_size", type=int, default=2)
    parser.add_argument("--strategy", default="linear",
                        choices=["linear", "round_robin"])
    parser.add_argument("--max_seqs", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--prompts", type=str, required=True,
                        help="JSONL file: one {text, max_tokens} per line")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # ── Load and distribute prompts ──
    # Round-robin assignment: rank i gets prompts i, i+world_size, ...
    all_prompts = []
    with open(args.prompts) as f:
        for line in f:
            all_prompts.append(json.loads(line))
    my_prompts = [p for i, p in enumerate(all_prompts)
                  if i % world_size == rank]
    print(f"Rank {rank}: {len(my_prompts)} prompts "
          f"(of {len(all_prompts)} total)")

    # ── Create engine (each rank creates its own) ──
    from moe_engine import MoEEngine
    engine = MoEEngine(
        model_path=args.model,
        max_seqs=args.max_seqs,
        max_seq_len=args.max_seq_len,
        expert_parallel_size=args.ep_size,
        expert_placement_strategy=args.strategy,
    )

    # ── Create scheduler (each rank has its own) ──
    from scheduler import Scheduler
    sched = Scheduler(engine, args.max_seqs, max_graph_size=512,
                      page_size=engine.page_size)

    # Capture CUDA graphs (all EP ranks MUST capture identical sizes)
    engine.capture_cuda_graphs(
        total_token_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512])

    # ── Build conversations for this rank ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    conversations = []
    for i, p in enumerate(my_prompts):
        ids = tokenizer.encode(p['text'])
        conversations.append({
            'conversation_id': f"rank{rank}_conv{i}",
            'prompt_token_ids': ids,
            'max_output_tokens': p.get('max_tokens', 128),
        })

    # ── Run collection ──
    # NOTE: Requires Step 8 (dummy-step logic) to avoid NCCL deadlock
    # when ranks finish at different times.
    result = sched.collect(conversations)
    if rank == 0:
        print(f"Rank 0: {result.step_count} steps, "
              f"{len(result.conversations)} conversations")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
