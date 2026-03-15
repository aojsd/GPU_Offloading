"""EP consistency tests: sync, dummy steps, and EP=N vs EP=1 equivalence.

Run with torchrun --nproc_per_node=N where N = ep_size.

Examples:
    H100_env/vllm_apptainer.sh torchrun --nproc_per_node=2 \
        src/MoE/tests/test_ep_consistency.py --model /path/to/model
    H100_env/vllm_apptainer.sh torchrun --nproc_per_node=4 \
        src/MoE/tests/test_ep_consistency.py --model /path/to/model
    H100_env/vllm_apptainer.sh torchrun --nproc_per_node=2 \
        src/MoE/tests/test_ep_consistency.py --model /path/to/mixtral --skip-test5
"""
import argparse, os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.distributed as dist


# ── Helpers ──────────────────────────────────────────────────────

def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank


def make_engine(model_path, ep_size, max_seqs=2, max_seq_len=512,
                compile=True):
    from moe_engine import MoEEngine
    # kv_page_budget enables dynamic page allocation (required for alloc_pages)
    kv_budget = max_seqs * (max_seq_len // 16)
    return MoEEngine(
        model_path=model_path,
        max_seqs=max_seqs,
        max_seq_len=max_seq_len,
        use_torch_compile=compile,
        expert_parallel_size=ep_size,
        kv_page_budget=kv_budget,
    )


def dummy_step(engine):
    """Zero-token forward pass — keeps NCCL collectives in sync."""
    engine.step(
        decode_seq_ids=[],
        decode_token_ids=torch.tensor([], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[], prefill_input_ids=[])


def greedy_generate(engine, rank, prompt_ids, max_new, page_size):
    """Single-sequence chunked-prefill + decode, EP-safe.

    All ranks iterate the same loops.  Rank 0 does real work, others
    call dummy_step.  This eliminates fragile separate step counting.

    Returns output tokens on rank 0, None on others.
    """
    engine.reset()
    chunk = 256
    sid = 0

    if rank == 0:
        total_pages = (len(prompt_ids) + max_new + page_size - 1) // page_size
        engine.alloc_pages(sid, total_pages)

    logits = None

    # Chunked prefill — all ranks iterate the same offsets
    for offset in range(0, len(prompt_ids), chunk):
        if rank == 0:
            c = prompt_ids[offset:offset + chunk]
            if offset == 0:
                logits = engine.step(
                    decode_seq_ids=[],
                    decode_token_ids=torch.tensor([], dtype=torch.long,
                                                  device=engine.device),
                    prefill_seq_ids=[sid],
                    prefill_input_ids=[torch.tensor(c, dtype=torch.long,
                                                    device=engine.device)])
            else:
                logits = engine.step(
                    decode_seq_ids=[],
                    decode_token_ids=torch.tensor([], dtype=torch.long,
                                                  device=engine.device),
                    prefill_seq_ids=[], prefill_input_ids=[],
                    continuation_seq_ids=[sid],
                    continuation_input_ids=[torch.tensor(c, dtype=torch.long,
                                                         device=engine.device)],
                    continuation_offsets=[offset])
        else:
            dummy_step(engine)

    if rank == 0:
        tok = logits[0].argmax().item()
        out = [tok]
    else:
        tok = None
        out = None

    # Decode — all ranks iterate the same count
    for _ in range(max_new - 1):
        if rank == 0:
            logits = engine.step(
                decode_seq_ids=[sid],
                decode_token_ids=torch.tensor([tok], dtype=torch.long,
                                              device=engine.device),
                prefill_seq_ids=[], prefill_input_ids=[])
            tok = logits[0].argmax().item()
            out.append(tok)
        else:
            dummy_step(engine)

    return out


# ── Tests ────────────────────────────────────────────────────────

def test1_basic_sync(engine, rank):
    """All ranks run symmetric prefill+decode — graph_N sync must agree."""
    engine.reset()
    sid = 0
    engine.alloc_pages(sid, 1)
    logits = engine.step(
        decode_seq_ids=[],
        decode_token_ids=torch.tensor([], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[sid],
        prefill_input_ids=[torch.randint(100, 5000, (4,),
                                         device=engine.device)])
    tok = logits[0].argmax().item()
    logits = engine.step(
        decode_seq_ids=[sid],
        decode_token_ids=torch.tensor([tok], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[], prefill_input_ids=[])
    if rank == 0:
        print("  PASS: test1_basic_sync")


def test2_dummy_step_no_deadlock(engine, rank):
    """Rank 0 does real work, all other ranks do dummy steps."""
    engine.reset()
    sid = 0
    if rank == 0:
        engine.alloc_pages(sid, 1)
        logits = engine.step(
            decode_seq_ids=[],
            decode_token_ids=torch.tensor([], dtype=torch.long,
                                          device=engine.device),
            prefill_seq_ids=[sid],
            prefill_input_ids=[torch.randint(100, 5000, (8,),
                                             device=engine.device)])
        tok = logits[0].argmax().item()
        logits = engine.step(
            decode_seq_ids=[sid],
            decode_token_ids=torch.tensor([tok], dtype=torch.long,
                                          device=engine.device),
            prefill_seq_ids=[], prefill_input_ids=[])
    else:
        dummy_step(engine)
        dummy_step(engine)
    if rank == 0:
        print("  PASS: test2_dummy_step_no_deadlock")


def test3_asymmetric_completion(engine, rank):
    """Rank 0 finishes early (8 steps), others run 16. Tests dummy padding."""
    engine.reset()
    sid = 0
    n_steps_self = 8 if rank == 0 else 16
    n_steps_max = 16

    total_pages = (4 + n_steps_max + engine.page_size - 1) // engine.page_size
    engine.alloc_pages(sid, total_pages)

    logits = engine.step(
        decode_seq_ids=[],
        decode_token_ids=torch.tensor([], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[sid],
        prefill_input_ids=[torch.randint(100, 5000, (4,),
                                         device=engine.device)])
    tok = logits[0].argmax().item()

    for step in range(n_steps_max):
        if step < n_steps_self:
            logits = engine.step(
                decode_seq_ids=[sid],
                decode_token_ids=torch.tensor([tok], dtype=torch.long,
                                              device=engine.device),
                prefill_seq_ids=[], prefill_input_ids=[])
            tok = logits[0].argmax().item()
        else:
            dummy_step(engine)
    if rank == 0:
        print("  PASS: test3_asymmetric_completion")


def test4_asymmetric_prefill_lengths(engine, rank):
    """Ranks prefill different-length prompts. graph_N sync pads to max."""
    engine.reset()
    sid = 0
    # Spread prompt lengths across ranks to stress the sync
    prompt_len = 32 * (rank + 1)  # rank0=32, rank1=64, rank2=96, rank3=128

    n_pages = (prompt_len + engine.page_size - 1) // engine.page_size
    engine.alloc_pages(sid, n_pages)

    prompt = torch.randint(100, 5000, (prompt_len,), device=engine.device)
    logits = engine.step(
        decode_seq_ids=[],
        decode_token_ids=torch.tensor([], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[sid],
        prefill_input_ids=[prompt])
    assert logits.shape[0] == 1, f"Expected 1 logit row, got {logits.shape[0]}"
    if rank == 0:
        print("  PASS: test4_asymmetric_prefill_lengths")


def test5_ep_vs_ep1_equivalence(rank, model_path, ep_size):
    """EP=N generation must match EP=1 (single-GPU) for the same prompt.

    Mathematically, AllGather gives each rank the full token set, each
    rank computes complementary expert partials (via expert_map), and
    ReduceScatter sums them — identical to single-GPU computing all
    experts.

    With compile=False (eager), FP accumulation order is deterministic,
    so EP=N and EP=1 should produce identical greedy tokens.  A minor
    caveat: ReduceScatter SUM vs in-kernel FP32 accumulation could
    theoretically differ by 1 ULP in BF16, but in practice 16 tokens
    of greedy generation never diverges.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = "The capital of France is"
    prompt_ids = tokenizer.encode(prompt)
    max_new = 16

    # ── EP=N run (all ranks participate) ──
    engine_ep = make_engine(model_path, ep_size=ep_size, max_seqs=1,
                            max_seq_len=512, compile=False)
    graph_sizes = [1, 2, 4, 8, 16, 32]
    engine_ep.capture_cuda_graphs(total_token_sizes=graph_sizes)
    page_size = engine_ep.page_size

    tokens_ep = greedy_generate(engine_ep, rank, prompt_ids, max_new,
                                page_size)

    del engine_ep
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        # ── EP=1 run (single GPU, all experts) ──
        engine_ep1 = make_engine(model_path, ep_size=1, max_seqs=1,
                                 max_seq_len=512, compile=False)
        engine_ep1.capture_cuda_graphs(total_token_sizes=graph_sizes)
        tokens_ep1 = greedy_generate(engine_ep1, 0, prompt_ids, max_new,
                                     page_size)
        del engine_ep1
        torch.cuda.empty_cache()

        n_match = sum(a == b for a, b in zip(tokens_ep, tokens_ep1))
        pct = 100.0 * n_match / max_new
        print(f"  EP={ep_size} vs EP=1: {n_match}/{max_new} tokens match "
              f"({pct:.1f}%)")
        if tokens_ep != tokens_ep1:
            print(f"    EP={ep_size}: {tokens_ep}")
            print(f"    EP=1:  {tokens_ep1}")

    # Broadcast n_match to all ranks so final pass/fail is consistent
    match_buf = torch.tensor(
        [n_match if rank == 0 else 0],
        dtype=torch.int32, device=f"cuda:{rank}")
    dist.broadcast(match_buf, src=0)
    return int(match_buf.item()), max_new


def test6_multi_sequence(engine, rank, ep_size):
    """Multiple sequences per rank with different batch sizes across ranks.

    Rank 0: 2 sequences, Rank 1: 3 sequences.  This stresses the graph_N
    sync with different decode batch sizes across EP ranks.
    """
    engine.reset()
    max_seqs = engine.max_seqs
    n_seqs = min(rank + 2, max_seqs)  # rank 0: 2, rank 1: 3, etc. (capped)
    max_n_seqs = min(ep_size + 1, max_seqs)  # max across all ranks

    prompt_len = 8

    # Prefill all sequences (one per step, with dummy steps for sync)
    first_tokens = []
    for i in range(max_n_seqs):
        if i < n_seqs:
            engine.alloc_pages(i, 2)
            logits = engine.step(
                decode_seq_ids=[],
                decode_token_ids=torch.tensor([], dtype=torch.long,
                                              device=engine.device),
                prefill_seq_ids=[i],
                prefill_input_ids=[torch.randint(100, 5000, (prompt_len,),
                                                 device=engine.device)])
            first_tokens.append(logits[0].argmax().item())
        else:
            dummy_step(engine)

    # Decode all sequences simultaneously for 4 steps.
    # Different batch sizes across ranks (2 vs 3) — graph_N sync pads.
    tokens = list(first_tokens)
    for step in range(4):
        sids = list(range(n_seqs))
        logits = engine.step(
            decode_seq_ids=sids,
            decode_token_ids=torch.tensor(tokens, dtype=torch.long,
                                          device=engine.device),
            prefill_seq_ids=[], prefill_input_ids=[])
        assert logits.shape[0] == n_seqs, (
            f"rank {rank}: expected {n_seqs} logit rows, "
            f"got {logits.shape[0]}")
        tokens = [logits[i].argmax().item() for i in range(n_seqs)]

    if rank == 0:
        print("  PASS: test6_multi_sequence")


def test7_scheduler_ep_collect(rank, model_path, ep_size):
    """Scheduler collect() with asymmetric workloads across EP ranks.

    Rank 0: 1 short conversation.
    Rank 1: 2 longer conversations.

    This stresses the EP termination sync in collect() — rank 0 finishes
    many steps before rank 1 and must issue dummy engine.step() calls to
    keep NCCL collectives matched.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    engine = make_engine(model_path, ep_size=ep_size, max_seqs=4,
                         max_seq_len=512, compile=False)
    graph_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256]
    engine.capture_cuda_graphs(total_token_sizes=graph_sizes)

    from scheduler import Scheduler
    sched = Scheduler(engine, max_seqs=4, max_graph_size=256,
                      page_size=engine.page_size)

    if rank == 0:
        # 1 short conversation
        ids = tokenizer.encode("Hello")
        convs = [{
            'conversation_id': 'r0_c0',
            'prompt_token_ids': ids,
            'max_output_tokens': 4,
        }]
    else:
        # 2 longer conversations
        convs = []
        for i in range(2):
            ids = tokenizer.encode("The capital of France is" * (i + 1))
            convs.append({
                'conversation_id': f'r{rank}_c{i}',
                'prompt_token_ids': ids,
                'max_output_tokens': 16,
            })

    result = sched.collect(convs)

    # Verify: all conversations completed with output tokens
    expected_n = len(convs)
    assert len(result.conversations) == expected_n, (
        f"rank {rank}: expected {expected_n} conversations, "
        f"got {len(result.conversations)}")
    for conv in result.conversations:
        assert len(conv['output_token_ids']) > 0, (
            f"rank {rank}: conversation {conv['conversation_id']} "
            f"has no output tokens")

    del engine
    torch.cuda.empty_cache()

    if rank == 0:
        print("  PASS: test7_scheduler_ep_collect")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--skip-test5", action="store_true",
                        help="Skip EP=N vs EP=1 equivalence (for models too "
                             "large to fit on a single GPU)")
    args = parser.parse_args()

    rank = setup()
    ep_size = dist.get_world_size()
    if rank == 0:
        print("=" * 60)
        print(f"EP Consistency Tests (EP={ep_size}, {ep_size} GPUs)")
        print("=" * 60)

    # Tests 1-4, 6: use a shared EP=N engine
    if rank == 0:
        print(f"\nCreating EP={ep_size} engine...")
    engine = make_engine(args.model, ep_size=ep_size, max_seqs=4,
                         max_seq_len=512, compile=False)
    graph_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256]
    engine.capture_cuda_graphs(total_token_sizes=graph_sizes)

    test1_basic_sync(engine, rank)
    dist.barrier()

    test2_dummy_step_no_deadlock(engine, rank)
    dist.barrier()

    test3_asymmetric_completion(engine, rank)
    dist.barrier()

    test4_asymmetric_prefill_lengths(engine, rank)
    dist.barrier()

    test6_multi_sequence(engine, rank, ep_size)
    dist.barrier()

    del engine
    torch.cuda.empty_cache()
    dist.barrier()

    # Test 5: EP=N vs EP=1 equivalence
    if args.skip_test5:
        if rank == 0:
            print(f"\n  SKIP: test5 EP={ep_size} vs EP=1 (--skip-test5)")
        n_match, max_new = 16, 16  # dummy pass
    else:
        if rank == 0:
            print()
        n_match, max_new = test5_ep_vs_ep1_equivalence(rank, args.model,
                                                        ep_size)
    dist.barrier()

    # Test 7: scheduler collect() EP sync
    if rank == 0:
        print()
    test7_scheduler_ep_collect(rank, args.model, ep_size)
    dist.barrier()

    if rank == 0:
        pct = 100.0 * n_match / max_new
        ok = (n_match == max_new) if not args.skip_test5 else True
        print("\n" + "=" * 60)
        status = "ALL PASSED" if ok else "SOME FAILED"
        print(f"EP Consistency Tests (EP={ep_size}): {status}")
        if not ok:
            print(f"  test5: {n_match}/{max_new} tokens matched "
                  f"(expected 100%)")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
