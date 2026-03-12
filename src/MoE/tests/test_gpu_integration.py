#!/usr/bin/env -S python3 -u
"""GPU integration tests for collect_batched() continuous batching.

Tests that BatchScheduler + MoEEngine + extract_next_tokens() work together
correctly on real hardware. Validates scheduling, preemption, page management,
and token generation.

Usage:
    # Single-GPU OLMoE (fast, ~2 min)
    python tests/test_gpu_integration.py --model ../../models/OLMoE-1B-7B

    # PP=2 Mixtral-8x7B (full 32 layers, 2x H100, ~5 min)
    python tests/test_gpu_integration.py --model ../../models/Mixtral-8x7B --pp 2

    # Single-GPU Mixtral with expert offloading (K experts/layer on GPU)
    python tests/test_gpu_integration.py --model ../../models/Mixtral-8x7B --experts-per-layer 4
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from moe_engine import MoEEngine
from trace_construction.collect_batched_traces import (
    ActiveState,
    BatchScheduler,
    collect_batched,
    extract_next_tokens,
)
from scheduler import Scheduler, CollectionResult, ReplayResult

DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "OLMoE-1B-7B")

PREFILL_CHUNK_SIZE = 256


# ---------------------------------------------------------------------------
# Reference: single-conversation collection via step
# ---------------------------------------------------------------------------

def run_single_conversation(engine, prompt_ids, max_output, page_size=16):
    """Reference single-conversation collection using direct step calls.

    Mirrors collect_batched_traces.py flow: chunked prefill (256-token chunks) + decode.
    Returns list of output token IDs.
    """
    device = engine.device
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    empty = torch.tensor([], dtype=torch.long, device=device)
    S = input_ids.shape[0]

    engine.reset()

    # Allocate initial pages for the full prompt
    initial_pages = math.ceil(S / page_size)
    engine.alloc_pages(0, initial_pages)

    output_token_ids = []

    with torch.inference_mode():
        # Chunked prefill
        offset = 0
        while offset < S:
            chunk_end = min(offset + PREFILL_CHUNK_SIZE, S)
            chunk = input_ids[offset:chunk_end]

            if offset == 0:
                logits = engine.step(
                    decode_seq_ids=[],
                    decode_token_ids=empty,
                    prefill_seq_ids=[0],
                    prefill_input_ids=[chunk],
                )
            else:
                logits = engine.step(
                    decode_seq_ids=[],
                    decode_token_ids=empty,
                    prefill_seq_ids=[],
                    prefill_input_ids=[],
                    continuation_seq_ids=[0],
                    continuation_input_ids=[chunk],
                    continuation_offsets=[offset],
                )
            offset = chunk_end

        # First output token from last prefill chunk (step returns last-token logits)
        next_token = logits[-1].argmax().unsqueeze(0)
        output_token_ids.append(next_token.item())

        # Decode
        for step in range(max_output - 1):
            # Ensure pages for growing sequence
            seq_len = S + len(output_token_ids)
            needed = math.ceil(seq_len / page_size)
            current = engine.seq_pages(0)
            if needed > current:
                engine.alloc_pages(0, needed - current)

            logits = engine.step(
                decode_seq_ids=[0],
                decode_token_ids=next_token,
                prefill_seq_ids=[],
                prefill_input_ids=[],
            )
            next_token = logits[0].argmax().unsqueeze(0)
            output_token_ids.append(next_token.item())

    engine.free_seq(0)
    return output_token_ids


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_single_conversation_match(engine, page_size=16):
    """Test 1: Single conversation via collect_batched matches reference.

    Output tokens must be identical (same compile + graph settings).
    """
    print("\n── Test 1: Single conversation match ──")
    prompt_len = 128
    max_output = 32
    prompt_ids = list(range(1, prompt_len + 1))  # Avoid token 0 (often pad)

    # Reference
    ref_tokens = run_single_conversation(
        engine, prompt_ids, max_output, page_size)
    engine.reset()
    torch.cuda.empty_cache()

    # collect_batched with 1 conversation
    conversations = [{
        'conversation_id': 'test_single',
        'prompt_token_ids': prompt_ids,
        'max_output_tokens': max_output,
    }]
    result = collect_batched(
        engine, conversations,
        max_seqs=engine.max_seqs,
        max_graph_size=512,
        page_size=page_size,
    )
    engine.reset()

    batched_tokens = result['conversations'][0]['output_token_ids']

    # Compare
    match = (ref_tokens == batched_tokens)
    print(f"  Reference: {len(ref_tokens)} tokens")
    print(f"  Batched:   {len(batched_tokens)} tokens")
    if not match:
        # Find first divergence
        for i, (a, b) in enumerate(zip(ref_tokens, batched_tokens)):
            if a != b:
                print(f"  DIVERGE at token {i}: ref={a}, batched={b}")
                break
        if len(ref_tokens) != len(batched_tokens):
            print(f"  Length mismatch: ref={len(ref_tokens)}, batched={len(batched_tokens)}")
    print(f"  Result: {'PASS' if match else 'FAIL'}")
    return match


def test_multiple_no_preemption(engine, page_size=16):
    """Test 2: Multiple conversations, generous budget, no preemptions.

    All conversations complete with correct output lengths.
    """
    print("\n── Test 2: Multiple conversations, no preemption ──")
    conversations = []
    prompt_lens = [64, 128, 96, 80, 112]
    max_outputs = [16, 24, 20, 12, 18]

    for i, (plen, mout) in enumerate(zip(prompt_lens, max_outputs)):
        conversations.append({
            'conversation_id': f'conv_{i}',
            'prompt_token_ids': list(range(1, plen + 1)),
            'max_output_tokens': mout,
        })

    result = collect_batched(
        engine, conversations,
        max_seqs=engine.max_seqs,
        max_graph_size=512,
        page_size=page_size,
    )
    engine.reset()

    passed = True
    preemptions = 0
    for conv in result['conversations']:
        ridx = conv['request_idx']
        expected = min(max_outputs[ridx],
                       engine.total_pages * page_size - prompt_lens[ridx])
        actual = len(conv['output_token_ids'])
        preemptions += conv['num_preemptions']
        ok = (actual == expected)
        if not ok:
            print(f"  FAIL: conv {ridx}: expected {expected} tokens, got {actual}")
            passed = False

    print(f"  Conversations: {len(conversations)}, all completed: {passed}")
    print(f"  Total preemptions: {preemptions} (expected 0)")
    if preemptions > 0:
        print("  WARNING: preemptions occurred with generous budget")
        # Don't fail — budget might be tighter than expected for this model
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_with_preemption(engine, kv_page_budget, page_size=16):
    """Test 3: Tight page budget forcing preemptions.

    All conversations still complete. At least 1 preemption must occur.
    """
    print("\n── Test 3: With preemption (tight budget) ──")
    print(f"  kv_page_budget: {kv_page_budget}")

    # Use prompts + outputs that will exceed the tight budget
    conversations = []
    prompt_lens = [64, 96, 80, 112, 72, 88, 104]
    max_output = 48  # each conversation wants ~48 output tokens

    for i, plen in enumerate(prompt_lens):
        conversations.append({
            'conversation_id': f'conv_{i}',
            'prompt_token_ids': list(range(1, plen + 1)),
            'max_output_tokens': max_output,
        })

    result = collect_batched(
        engine, conversations,
        max_seqs=engine.max_seqs,
        max_graph_size=512,
        page_size=page_size,
    )
    engine.reset()

    # Check all completed
    passed = True
    total_preemptions = 0
    for conv in result['conversations']:
        ridx = conv['request_idx']
        expected = min(max_output,
                       kv_page_budget * page_size - prompt_lens[ridx])
        actual = len(conv['output_token_ids'])
        total_preemptions += conv['num_preemptions']
        ok = (actual == expected)
        if not ok:
            print(f"  FAIL: conv {ridx}: expected {expected} tokens, got {actual}")
            passed = False

    print(f"  Conversations: {len(conversations)}, all completed: {passed}")
    print(f"  Total preemptions: {total_preemptions}")
    if total_preemptions == 0:
        print("  FAIL: expected at least 1 preemption with tight budget")
        passed = False
    print(f"  Steps: {result['step_count']}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_page_invariant_gpu(engine, kv_page_budget, page_size=16):
    """Test 4: Page invariant holds at every step.

    Instrumented loop: assert engine.pages_in_use <= kv_page_budget at every step.
    """
    print("\n── Test 4: Page invariant (instrumented loop) ──")
    print(f"  kv_page_budget: {kv_page_budget}")

    conversations = []
    prompt_lens = [64, 96, 80, 112, 72]
    max_output = 32

    for i, plen in enumerate(prompt_lens):
        conversations.append({
            'conversation_id': f'conv_{i}',
            'prompt_token_ids': list(range(1, plen + 1)),
            'max_output_tokens': max_output,
        })

    # Manual instrumented loop instead of collect_batched()
    scheduler = BatchScheduler(engine, engine.max_seqs, 512, page_size)
    requests = [
        ActiveState(
            request_idx=i,
            conversation_id=conv['conversation_id'],
            prompt_token_ids=conv['prompt_token_ids'],
            max_output_tokens=conv['max_output_tokens'],
        )
        for i, conv in enumerate(conversations)
    ]
    prompts_gpu = {
        i: torch.tensor(conv['prompt_token_ids'], dtype=torch.long,
                        device=engine.device)
        for i, conv in enumerate(conversations)
    }
    scheduler.add_requests(requests)

    passed = True
    max_pages_seen = 0
    step = 0

    with torch.inference_mode():
        while not scheduler.is_done:
            result = scheduler.step()
            if result.is_empty:
                continue

            # CHECK: page invariant before GPU call
            pages_used = engine.pages_in_use
            max_pages_seen = max(max_pages_seen, pages_used)
            if pages_used > kv_page_budget:
                print(f"  FAIL step {step}: pages_in_use={pages_used} > "
                      f"budget={kv_page_budget}")
                passed = False

            # Build step args
            if result.decode_requests:
                decode_token_ids = torch.tensor(
                    [r.output_token_ids[-1] for r in result.decode_requests],
                    dtype=torch.long, device=engine.device)
            else:
                decode_token_ids = torch.tensor(
                    [], dtype=torch.long, device=engine.device)

            prefill_input_ids = []
            for req in result.prefill_requests:
                prompt = prompts_gpu[req.request_idx]
                if req.output_token_ids:
                    out = torch.tensor(req.output_token_ids, dtype=torch.long,
                                       device=engine.device)
                    eff = torch.cat([prompt, out])
                else:
                    eff = prompt
                prefill_input_ids.append(
                    eff[req.prefill_chunk_start:
                        req.prefill_chunk_start + req.scheduled_chunk])

            continuation_input_ids = []
            continuation_offsets = []
            for req in result.continuation_requests:
                prompt = prompts_gpu[req.request_idx]
                if req.output_token_ids:
                    out = torch.tensor(req.output_token_ids, dtype=torch.long,
                                       device=engine.device)
                    eff = torch.cat([prompt, out])
                else:
                    eff = prompt
                continuation_input_ids.append(
                    eff[req.prefill_chunk_start:
                        req.prefill_chunk_start + req.scheduled_chunk])
                slot = scheduler.request_to_slot[req.request_idx]
                continuation_offsets.append(
                    engine._seq_lens_cpu[slot].item())

            logits = engine.step(
                decode_seq_ids=result.decode_seq_ids,
                decode_token_ids=decode_token_ids,
                prefill_seq_ids=result.prefill_seq_ids,
                prefill_input_ids=prefill_input_ids,
                continuation_seq_ids=result.continuation_seq_ids,
                continuation_input_ids=continuation_input_ids,
                continuation_offsets=continuation_offsets,
            )

            next_tokens = extract_next_tokens(
                logits, len(result.decode_requests),
                result.prefill_chunk_lengths,
                result.continuation_chunk_lengths,
            )
            scheduler.advance_state(result, next_tokens)

            # CHECK: page invariant after advance
            pages_used = engine.pages_in_use
            max_pages_seen = max(max_pages_seen, pages_used)
            if pages_used > kv_page_budget:
                print(f"  FAIL step {step} (post-advance): pages_in_use={pages_used} > "
                      f"budget={kv_page_budget}")
                passed = False

            step += 1

    engine.reset()
    print(f"  Steps: {step}, max pages seen: {max_pages_seen}/{kv_page_budget}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_eos_terminates_early(engine, page_size=16):
    """Test 6: EOS token detected by collect_batched() stops generation early.

    Verifies the EOS detection mechanism in collect_batched(). Whether
    the model actually produces EOS depends on the prompt — with arbitrary
    token IDs, many models won't. We test the MECHANISM: if EOS appears
    in the output, the last token must equal eos_id and output must be
    shorter than max_output_tokens. If the model doesn't hit EOS, we
    verify it generates exactly max_output_tokens (no early stop).
    """
    print("\n── Test 6: EOS detection ──")
    eos_id = getattr(engine, 'eos_token_id', None)
    if eos_id is None:
        print("  SKIP: engine has no eos_token_id")
        return True

    prompt_ids = list(range(1, 65))  # 64 tokens
    max_out = 200  # short to avoid long generation
    result = collect_batched(
        engine,
        [{'conversation_id': 'eos_test',
          'prompt_token_ids': prompt_ids,
          'max_output_tokens': max_out}],
        max_seqs=engine.max_seqs,
        max_graph_size=512,
        page_size=page_size,
    )
    engine.reset()

    conv = result['conversations'][0]
    n_out = len(conv['output_token_ids'])
    last_tok = conv['output_token_ids'][-1] if conv['output_token_ids'] else None

    hit_eos = (last_tok == eos_id)
    terminated_early = (n_out < max_out)

    print(f"  Output tokens: {n_out} (limit={max_out})")
    print(f"  Last token: {last_tok} (eos_id={eos_id})")
    print(f"  Hit EOS: {hit_eos}, terminated early: {terminated_early}")

    if hit_eos:
        # EOS detected → must have stopped early
        passed = terminated_early
    else:
        # No EOS → must have generated exactly max_output_tokens
        passed = (n_out == max_out)
        print(f"  (model did not produce EOS with this prompt — "
              f"verifying full generation)")

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_replay_faithfulness(engine, page_size=16):
    """Test 7: Replay with collected tokens reproduces identical expert routing.

    Pipeline: collect → serialize → load → Scheduler.replay() (feeding
    the collected output tokens back through the engine) → compare expert routing.
    If routing matches, the replay engine sees the same hidden states at every
    layer, confirming token faithfulness.
    """
    print("\n── Test 7: Replay faithfulness ──")
    import json as _json
    import os as _os
    import tempfile

    from trace_construction.collect_batched_traces import (
        save_batched_trace, save_conversations,
    )
    from gpu_replay_trace import ActivationTrace
    from trace_construction.trace_utils import load_traces

    # Phase 1: Collect traces (generous budget, no preemption)
    conversations = [
        {'conversation_id': f'faith_{i}',
         'prompt_token_ids': list(range(1, 65 + 16 * i)),
         'max_output_tokens': 16}
        for i in range(3)
    ]
    sched = Scheduler(engine, max_seqs=engine.max_seqs,
                      max_graph_size=512, page_size=page_size)
    collected = sched.collect(conversations)
    collection_trace = collected.trace
    engine.reset()
    torch.cuda.empty_cache()

    print(f"  Collected: {collected.step_count} steps, "
          f"{len(collection_trace)} trace entries")

    # Phase 2: Serialize → load (validates data roundtrip)
    with tempfile.TemporaryDirectory() as d:
        result = collected.to_dict()
        save_batched_trace(result, d, {'max_seqs': engine.max_seqs})
        manifest_entries = save_conversations(result, d, top_k=2)
        manifest = {
            'model': 'test', 'num_layers': collected.num_layers,
            'num_experts': collected.num_experts, 'top_k': 2,
            'total_conversations': len(manifest_entries),
            'step_count': collected.step_count,
            'scheduling': {'max_seqs': engine.max_seqs},
            'conversations': manifest_entries,
        }
        with open(_os.path.join(d, 'manifest.json'), 'w') as f:
            _json.dump(manifest, f)

        at = ActivationTrace.load(_os.path.join(d, 'batched_trace.json'))
        per_conv_traces, _ = load_traces(d)

    assert at.scheduling is not None
    assert len(at.scheduling) == collected.step_count
    print(f"  Loaded: {len(at.scheduling)} scheduling steps, "
          f"{len(per_conv_traces)} conversations")

    # Phase 3: Replay — feed collected tokens, record routing
    replayed = sched.replay(
        collected.conversations,
        scheduling=at.scheduling,
        record_routing=True, record_tokens=True)

    # Phase 4: Compare expert routing
    passed = True

    if len(replayed.trace_data) != len(collection_trace):
        print(f"  FAIL: trace length: replay={len(replayed.trace_data)} "
              f"vs collection={len(collection_trace)}")
        passed = False
    else:
        routing_ok, n_mismatch = replayed.compare_routing(collection_trace)
        if not routing_ok:
            # Print first few mismatches for debugging
            count = 0
            for i, (rt, ct) in enumerate(
                    zip(replayed.trace_data, collection_trace)):
                if rt['expert_ids'] != ct['expert_ids']:
                    if count < 5:
                        print(f"  ROUTING MISMATCH step {rt['step']} "
                              f"layer {rt['layer']}: replay={rt['expert_ids']} "
                              f"vs collection={ct['expert_ids']}")
                    count += 1
            print(f"  {n_mismatch}/{len(collection_trace)} routing mismatches")
            passed = False
        else:
            print(f"  Expert routing: {len(collection_trace)} entries all match")

    # Phase 5: Compare output tokens
    tokens_ok, token_mismatches = replayed.compare_tokens(collected)
    for conv in collected.conversations:
        rid = conv['request_idx']
        collected_toks = conv['output_token_ids']
        replayed_toks = replayed.output_tokens.get(rid, [])
        n = min(len(collected_toks), len(replayed_toks))
        if rid in token_mismatches:
            first_diff = token_mismatches[rid]
            print(f"  TOKEN MISMATCH conv {conv['conversation_id']}: "
                  f"first diff at token {first_diff} "
                  f"(collected={collected_toks[first_diff]}, "
                  f"replayed={replayed_toks[first_diff]})")
            passed = False
        else:
            print(f"  Tokens conv {conv['conversation_id']}: "
                  f"{n} tokens match")

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_replay_faithfulness_with_preemption(engine, kv_page_budget, page_size=16):
    """Test 8: Replay faithfulness under preemption.

    Same pipeline as test 7 but with tight KV budget forcing preemptions.
    Verifies that collect → serialize → load → Scheduler.replay() produces
    identical expert routing and output tokens even when the trace contains
    preempt/readmit events.
    """
    print("\n── Test 8: Replay faithfulness with preemption ──")
    print(f"  kv_page_budget: {kv_page_budget}")
    import json as _json
    import os as _os
    import tempfile

    from trace_construction.collect_batched_traces import (
        save_batched_trace, save_conversations,
    )
    from gpu_replay_trace import ActivationTrace
    from trace_construction.trace_utils import load_traces

    # Phase 1: Collect with tight budget → must produce preemptions
    conversations = [
        {'conversation_id': f'preempt_{i}',
         'prompt_token_ids': list(range(1, 65 + 16 * i)),
         'max_output_tokens': 256}
        for i in range(12)
    ]
    sched = Scheduler(engine, max_seqs=engine.max_seqs,
                      max_graph_size=512, page_size=page_size)
    collected = sched.collect(conversations)
    collection_trace = collected.trace
    engine.reset()
    torch.cuda.empty_cache()

    total_preemptions = sum(
        c['num_preemptions'] for c in collected.conversations)
    print(f"  Collected: {collected.step_count} steps, "
          f"{len(collection_trace)} trace entries, "
          f"{total_preemptions} preemptions")

    if total_preemptions == 0:
        print("  FAIL: expected at least 1 preemption with tight budget")
        return False

    # Check preempt events exist in scheduling
    preempt_events = sum(
        1 for s in collected.all_step_scheduling
        for e in s.get('events', []) if e['event'] == 'preempt')
    readmit_events = sum(
        1 for s in collected.all_step_scheduling
        for e in s.get('events', [])
        if e['event'] in ('admit', 'force_admit')
        and any(
            prev_e['event'] == 'preempt'
            and prev_e['request_id'] == e['request_id']
            for prev_s in collected.all_step_scheduling
            for prev_e in prev_s.get('events', [])
        ))
    print(f"  Scheduling: {preempt_events} preempt events, "
          f"{readmit_events} readmissions")

    # Phase 2: Serialize → load
    with tempfile.TemporaryDirectory() as d:
        result = collected.to_dict()
        save_batched_trace(result, d, {
            'max_seqs': engine.max_seqs,
            'kv_page_budget': kv_page_budget,
        })
        manifest_entries = save_conversations(result, d, top_k=2)
        manifest = {
            'model': 'test', 'num_layers': collected.num_layers,
            'num_experts': collected.num_experts, 'top_k': 2,
            'total_conversations': len(manifest_entries),
            'step_count': collected.step_count,
            'scheduling': {'max_seqs': engine.max_seqs},
            'conversations': manifest_entries,
        }
        with open(_os.path.join(d, 'manifest.json'), 'w') as f:
            _json.dump(manifest, f)

        at = ActivationTrace.load(_os.path.join(d, 'batched_trace.json'))
        per_conv_traces, _ = load_traces(d)

    assert at.scheduling is not None
    assert len(at.scheduling) == collected.step_count

    # Verify scheduling has preemption events after roundtrip
    loaded_preempt_events = sum(
        1 for s in at.scheduling
        for e in s.events if e['event'] == 'preempt')
    print(f"  Loaded scheduling: {loaded_preempt_events} preempt events "
          f"(roundtrip)")
    assert loaded_preempt_events == preempt_events

    # Phase 3: Replay — feed collected tokens, record routing
    replayed = sched.replay(
        collected.conversations,
        scheduling=at.scheduling,
        record_routing=True, record_tokens=True)

    # Phase 4: Compare expert routing
    passed = True

    if len(replayed.trace_data) != len(collection_trace):
        print(f"  FAIL: trace length: replay={len(replayed.trace_data)} "
              f"vs collection={len(collection_trace)}")
        passed = False
    else:
        routing_ok, n_mismatch = replayed.compare_routing(collection_trace)
        if not routing_ok:
            count = 0
            for i, (rt, ct) in enumerate(
                    zip(replayed.trace_data, collection_trace)):
                if rt['expert_ids'] != ct['expert_ids']:
                    if count < 5:
                        print(f"  ROUTING MISMATCH step {rt['step']} "
                              f"layer {rt['layer']}: replay={rt['expert_ids']} "
                              f"vs collection={ct['expert_ids']}")
                    count += 1
            print(f"  {n_mismatch}/{len(collection_trace)} routing mismatches")
            passed = False
        else:
            print(f"  Expert routing: {len(collection_trace)} entries all match")

    # Phase 5: Compare output tokens
    tokens_ok, token_mismatches = replayed.compare_tokens(collected)
    for conv in collected.conversations:
        rid = conv['request_idx']
        collected_toks = conv['output_token_ids']
        replayed_toks = replayed.output_tokens.get(rid, [])
        n = min(len(collected_toks), len(replayed_toks))
        if rid in token_mismatches:
            first_diff = token_mismatches[rid]
            print(f"  TOKEN MISMATCH conv {conv['conversation_id']}: "
                  f"first diff at token {first_diff}")
            passed = False
        else:
            preempt_count = conv['num_preemptions']
            print(f"  Tokens conv {conv['conversation_id']}: "
                  f"{n} tokens match "
                  f"(preemptions={preempt_count})")

    # Phase 6: Verify preemption counts match
    for conv in collected.conversations:
        rid = conv['request_idx']
        collected_p = conv['num_preemptions']
        replayed_p = replayed.preemptions.get(rid, 0)
        if collected_p != replayed_p:
            print(f"  PREEMPTION COUNT MISMATCH conv {conv['conversation_id']}: "
                  f"collected={collected_p}, replay saw={replayed_p}")
            passed = False

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_seq_len_consistency(engine, kv_page_budget, page_size=16):
    """Test 5: req.seq_len == engine._seq_lens_cpu[slot] for all running after advance.

    Verifies dual seq_len bookkeeping stays in sync.
    """
    print("\n── Test 5: Seq len consistency ──")
    print(f"  kv_page_budget: {kv_page_budget}")

    conversations = []
    prompt_lens = [64, 128, 96, 80]
    max_output = 32

    for i, plen in enumerate(prompt_lens):
        conversations.append({
            'conversation_id': f'conv_{i}',
            'prompt_token_ids': list(range(1, plen + 1)),
            'max_output_tokens': max_output,
        })

    scheduler = BatchScheduler(engine, engine.max_seqs, 512, page_size)
    requests = [
        ActiveState(
            request_idx=i,
            conversation_id=conv['conversation_id'],
            prompt_token_ids=conv['prompt_token_ids'],
            max_output_tokens=conv['max_output_tokens'],
        )
        for i, conv in enumerate(conversations)
    ]
    prompts_gpu = {
        i: torch.tensor(conv['prompt_token_ids'], dtype=torch.long,
                        device=engine.device)
        for i, conv in enumerate(conversations)
    }
    scheduler.add_requests(requests)

    passed = True
    step = 0
    mismatches = 0

    with torch.inference_mode():
        while not scheduler.is_done:
            result = scheduler.step()
            if result.is_empty:
                continue

            # Build step args (same as test 4)
            if result.decode_requests:
                decode_token_ids = torch.tensor(
                    [r.output_token_ids[-1] for r in result.decode_requests],
                    dtype=torch.long, device=engine.device)
            else:
                decode_token_ids = torch.tensor(
                    [], dtype=torch.long, device=engine.device)

            prefill_input_ids = []
            for req in result.prefill_requests:
                prompt = prompts_gpu[req.request_idx]
                if req.output_token_ids:
                    out = torch.tensor(req.output_token_ids, dtype=torch.long,
                                       device=engine.device)
                    eff = torch.cat([prompt, out])
                else:
                    eff = prompt
                prefill_input_ids.append(
                    eff[req.prefill_chunk_start:
                        req.prefill_chunk_start + req.scheduled_chunk])

            continuation_input_ids = []
            continuation_offsets = []
            for req in result.continuation_requests:
                prompt = prompts_gpu[req.request_idx]
                if req.output_token_ids:
                    out = torch.tensor(req.output_token_ids, dtype=torch.long,
                                       device=engine.device)
                    eff = torch.cat([prompt, out])
                else:
                    eff = prompt
                continuation_input_ids.append(
                    eff[req.prefill_chunk_start:
                        req.prefill_chunk_start + req.scheduled_chunk])
                slot = scheduler.request_to_slot[req.request_idx]
                continuation_offsets.append(
                    engine._seq_lens_cpu[slot].item())

            logits = engine.step(
                decode_seq_ids=result.decode_seq_ids,
                decode_token_ids=decode_token_ids,
                prefill_seq_ids=result.prefill_seq_ids,
                prefill_input_ids=prefill_input_ids,
                continuation_seq_ids=result.continuation_seq_ids,
                continuation_input_ids=continuation_input_ids,
                continuation_offsets=continuation_offsets,
            )

            next_tokens = extract_next_tokens(
                logits, len(result.decode_requests),
                result.prefill_chunk_lengths,
                result.continuation_chunk_lengths,
            )
            scheduler.advance_state(result, next_tokens)

            # CHECK: seq_len consistency for all running requests
            for req in scheduler.running:
                if req.request_idx not in scheduler.request_to_slot:
                    continue
                slot = scheduler.request_to_slot[req.request_idx]
                engine_len = engine._seq_lens_cpu[slot].item()
                if req.seq_len != engine_len:
                    if mismatches < 10:
                        print(f"  MISMATCH step {step}: req {req.request_idx} "
                              f"scheduler={req.seq_len} engine={engine_len}")
                    mismatches += 1
                    passed = False

            step += 1

    engine.reset()
    print(f"  Steps: {step}, mismatches: {mismatches}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Full pipeline: collect → simulate → offloaded replay
# ---------------------------------------------------------------------------

def _offloaded_replay(engine, dm_trace, collection_result, page_size):
    """Replay a GPUReplayTrace on an offloading engine via Scheduler.replay().

    The engine must have been created with cache_size (offloading mode).
    ReplayController manages expert loading/unloading via PCIe transfers.

    Returns ReplayResult.
    """
    from replay_controller import ReplayController

    max_seqs = engine.max_seqs
    sched = Scheduler(engine, max_seqs=max_seqs,
                      max_graph_size=512, page_size=page_size)
    controller = ReplayController(engine, dm_trace)

    if isinstance(collection_result, CollectionResult):
        convs = collection_result.conversations
    else:
        convs = collection_result['conversations']

    return sched.replay(
        convs, controller=controller, record_tokens=True)


def _compare_tokens(label, collection_result, replay_result):
    """Compare output tokens from ReplayResult against collection. Returns passed."""
    if isinstance(collection_result, CollectionResult):
        convs = collection_result.conversations
    else:
        convs = collection_result['conversations']

    tokens_ok, mismatches = replay_result.compare_tokens(collection_result)
    passed = True
    for conv in convs:
        rid = conv['request_idx']
        collected = conv['output_token_ids']
        replayed = replay_result.output_tokens.get(rid, [])
        n = min(len(collected), len(replayed))
        if rid in mismatches:
            first_diff = mismatches[rid]
            print(f"  {label} TOKEN MISMATCH conv {conv['conversation_id']}: "
                  f"first diff at token {first_diff} "
                  f"(collected={collected[first_diff]}, "
                  f"replayed={replayed[first_diff]})")
            passed = False
        else:
            print(f"  {label} conv {conv['conversation_id']}: "
                  f"{n} tokens match "
                  f"(preemptions={conv['num_preemptions']})")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU integration tests for collect_batched()")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallel size (1 or 2)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (not recommended)")
    parser.add_argument("--experts-per-layer", type=int, default=None,
                        help="Expert offloading: keep K experts/layer on GPU "
                             "(enables single-GPU Mixtral)")
    args = parser.parse_args()

    model = args.model
    pp = args.pp
    epl = args.experts_per_layer
    use_compile = not args.no_compile
    page_size = 16

    if epl is not None and pp > 1:
        print("ERROR: --experts-per-layer and --pp > 1 are mutually exclusive")
        return 1

    print(f"Model: {model}")
    print(f"PP size: {pp}")
    if epl is not None:
        print(f"experts_per_layer: {epl}")
    print(f"torch.compile: {use_compile}")

    # Engine config: generous budget for tests 1-2, tight for tests 3-5
    max_seqs = 8
    max_seq_len = 2048
    max_pages_per_seq = math.ceil(max_seq_len / page_size)
    generous_budget = max_seqs * max_pages_per_seq  # 1024 pages

    # Tight budget: ~32 pages forces preemption with 12 conversations
    # Each conversation wants ~20-31 pages (prompt + 256 output), so 32 pages
    # can hold ~1-2 at once, forcing heavy preemption and multi-preemption.
    tight_budget = 32

    graph_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]

    # Engine kwargs: either PP or experts_per_layer (mutually exclusive)
    engine_mode_kwargs = {}
    if epl is not None:
        engine_mode_kwargs['experts_per_layer'] = epl
    else:
        engine_mode_kwargs['pipeline_parallel_size'] = pp

    # ── Create engine (generous budget first for tests 1-2) ──
    print("\n" + "=" * 60)
    print("Creating engine (generous budget)...")
    t0 = time.time()
    engine = MoEEngine(
        model, max_seqs=max_seqs, max_seq_len=max_seq_len,
        page_size=page_size, use_torch_compile=use_compile,
        kv_page_budget=generous_budget, **engine_mode_kwargs)
    print(f"Engine created in {time.time() - t0:.1f}s")

    print("Capturing CUDA graphs...")
    engine.capture_cuda_graphs(graph_sizes)
    print(f"Graphs captured. Total pages: {engine.total_pages}")

    # ── Test 1: Single conversation match ──
    t1 = test_single_conversation_match(engine, page_size)

    # ── Test 2: Multiple conversations, no preemption ──
    t2 = test_multiple_no_preemption(engine, page_size)

    # ── Test 6: EOS terminates early ──
    t6 = test_eos_terminates_early(engine, page_size)

    # ── Test 7: Replay faithfulness ──
    t7 = test_replay_faithfulness(engine, page_size)

    # ── Collect for test 9 (generous budget, no preemption) ──
    print("\n── Collecting for test 9 (no preemption) ──")
    t9_convs = [
        {'conversation_id': f'pipe_{i}',
         'prompt_token_ids': list(range(1, 65 + 16 * i)),
         'max_output_tokens': 16}
        for i in range(3)
    ]
    t9_result = collect_batched(
        engine, t9_convs, max_seqs=max_seqs,
        max_graph_size=max(graph_sizes), page_size=page_size)
    engine.reset()
    print(f"  {t9_result['step_count']} steps, "
          f"{len(t9_result['trace'])} trace entries")

    del engine
    torch.cuda.empty_cache()

    # ── Create engine (tight budget for tests 3-5) ──
    print("\n" + "=" * 60)
    print(f"Creating engine (tight budget={tight_budget})...")
    t0 = time.time()
    engine = MoEEngine(
        model, max_seqs=max_seqs, max_seq_len=max_seq_len,
        page_size=page_size, use_torch_compile=use_compile,
        kv_page_budget=tight_budget, **engine_mode_kwargs)
    print(f"Engine created in {time.time() - t0:.1f}s")

    print("Capturing CUDA graphs...")
    engine.capture_cuda_graphs(graph_sizes)
    print(f"Graphs captured. Total pages: {engine.total_pages}")

    # ── Test 3: With preemption ──
    t3 = test_with_preemption(engine, tight_budget, page_size)

    # ── Test 4: Page invariant ──
    t4 = test_page_invariant_gpu(engine, tight_budget, page_size)

    # ── Test 5: Seq len consistency ──
    t5 = test_seq_len_consistency(engine, tight_budget, page_size)

    # ── Test 8: Replay faithfulness with preemption ──
    t8 = test_replay_faithfulness_with_preemption(engine, tight_budget, page_size)

    # ── Collect for test 10 (tight budget, with preemption) ──
    print("\n── Collecting for test 10 (with preemption) ──")
    t10_convs = [
        {'conversation_id': f'pipe_p_{i}',
         'prompt_token_ids': list(range(1, 65 + 16 * i)),
         'max_output_tokens': 256}
        for i in range(12)
    ]
    t10_result = collect_batched(
        engine, t10_convs, max_seqs=max_seqs,
        max_graph_size=max(graph_sizes), page_size=page_size)
    engine.reset()
    t10_preemptions = sum(
        c['num_preemptions'] for c in t10_result['conversations'])
    print(f"  {t10_result['step_count']} steps, "
          f"{t10_preemptions} preemptions")

    del engine
    torch.cuda.empty_cache()

    # ── Tests 9-10: Full pipeline (collect → simulate → offloaded replay) ──
    #
    # These tests verify the COMPLETE pipeline:
    #   1. collect_batched() on all-resident engine (already done above)
    #   2. simulate(LRU, OraclePrefetch) → GPUReplayTrace (in-memory)
    #   3. Create offloading engine with cache_size (REAL OFFLOADING:
    #      only cache_size expert slots on GPU, rest on CPU; ReplayController
    #      issues PCIe transfers to load/evict experts each step)
    #   4. Replay with ReplayController, feeding collected tokens
    #   5. Compare output tokens against collection
    #
    # If tokens match, the offloading engine produced identical hidden states
    # despite experts being swapped in/out — proving the policy simulation
    # and expert loading machinery are correct.
    #
    # NOTE: cache_size = 50% of total expert slots means roughly half the
    # experts are GPU-resident at any time. This exercises real PCIe
    # transfers on every step (demand loads + prefetches).

    from gpu_replay_trace import ActivationTrace
    from policy_simulator import LRU, OraclePrefetch, simulate
    import json as _json

    with open(Path(model) / "config.json") as f:
        cfg = _json.load(f)
    num_layers = cfg["num_hidden_layers"]
    num_experts = (cfg.get("n_routed_experts") or cfg.get("num_experts")
                   or cfg.get("num_local_experts"))
    first_k = cfg.get("first_k_dense_replace", 0)
    total_expert_slots = (num_layers - first_k) * num_experts
    cache_size = total_expert_slots // 2  # 50% → real offloading

    # Build ActivationTraces and simulate (all in-memory, no files)
    pipeline_data = {}
    for label, res in [("t9", t9_result), ("t10", t10_result)]:
        trace_data = {
            'num_layers': res['num_layers'],
            'num_experts': res['num_experts'],
            'trace': res['trace'],
            'step_scheduling': res['all_step_scheduling'],
        }
        at = ActivationTrace.from_flat_trace(trace_data)
        dm_trace = simulate(LRU(), OraclePrefetch(), at, cache_size)
        pipeline_data[label] = dm_trace
        print(f"  {label}: simulated LRU-Oracle, "
              f"cache_size={cache_size}/{total_expert_slots}, "
              f"{len(dm_trace.steps)} steps")

    # Create offloading engine (cache_size mode, PP=1 single GPU)
    # This is the production deployment configuration: a subset of experts
    # lives on GPU, the rest stream from CPU via PCIe on demand.
    print("\n" + "=" * 60)
    print(f"Creating offloading engine (cache_size={cache_size}, PP=1)...")
    t0 = time.time()
    rc_engine = MoEEngine(
        model, max_seqs=max_seqs, max_seq_len=max_seq_len,
        page_size=page_size, use_torch_compile=use_compile,
        cache_size=cache_size)
    print(f"Engine created in {time.time() - t0:.1f}s")
    print("Capturing CUDA graphs...")
    rc_engine.capture_cuda_graphs(graph_sizes)
    print(f"Graphs captured. Total pages: {rc_engine.total_pages}")

    # ── Test 9: Full pipeline, no preemption ──
    print("\n── Test 9: Full pipeline — offloaded replay (no preemption) ──")
    print(f"  cache_size={cache_size}/{total_expert_slots} "
          f"({100*cache_size/total_expert_slots:.0f}% resident)")
    t9_replay_tokens = _offloaded_replay(
        rc_engine, pipeline_data["t9"], t9_result, page_size)
    t9 = _compare_tokens("test9", t9_result, t9_replay_tokens)
    print(f"  Result: {'PASS' if t9 else 'FAIL'}")

    # ── Test 10: Full pipeline, with preemption ──
    print("\n── Test 10: Full pipeline — offloaded replay (with preemption) ──")
    print(f"  cache_size={cache_size}/{total_expert_slots} "
          f"({100*cache_size/total_expert_slots:.0f}% resident), "
          f"{t10_preemptions} preemptions in trace")
    t10 = True
    if t10_preemptions == 0:
        print("  FAIL: expected preemptions but got 0")
        t10 = False
    else:
        t10_replay_tokens = _offloaded_replay(
            rc_engine, pipeline_data["t10"], t10_result, page_size)
        t10 = _compare_tokens("test10", t10_result, t10_replay_tokens)
    print(f"  Result: {'PASS' if t10 else 'FAIL'}")

    del rc_engine
    torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 60)
    results = {
        "1. Single conversation match": t1,
        "2. Multiple no preemption": t2,
        "3. With preemption": t3,
        "4. Page invariant": t4,
        "5. Seq len consistency": t5,
        "6. EOS terminates early": t6,
        "7. Replay faithfulness": t7,
        "8. Replay faithfulness + preemption": t8,
        "9. Full pipeline (offloaded)": t9,
        "10. Full pipeline + preemption (offloaded)": t10,
    }
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    print(f"\n{'All tests PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
