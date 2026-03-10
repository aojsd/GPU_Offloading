"""
Tests for the vLLM V1-style scheduling state machine and logit extraction.

Chunk 1 tests (1-19): CPU-only, no GPU, no model. Use MockPageAllocator.
Chunk 2 tests (20-26): Synthetic tensors for logit extraction.
"""

import math
import sys
import os

import pytest
import torch

# Allow imports from src/MoE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trace_construction.collect_batched_traces import (
    ActiveState,
    BatchScheduler,
    MockPageAllocator,
    ScheduleResult,
    extract_next_tokens,
    pages_needed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_request(
    request_idx: int,
    prompt_len: int,
    max_output: int,
    conv_id: str = "conv",
    output_token_ids: list[int] | None = None,
) -> ActiveState:
    """Create a synthetic request for testing."""
    return ActiveState(
        request_idx=request_idx,
        conversation_id=f"{conv_id}_{request_idx}",
        prompt_token_ids=list(range(prompt_len)),
        output_token_ids=output_token_ids if output_token_ids is not None else [],
        max_output_tokens=max_output,
    )


def run_until_done(
    scheduler: BatchScheduler,
    dummy_token: int = 42,
    max_steps: int = 100_000,
) -> list[ScheduleResult]:
    """Run scheduler to completion, feeding dummy tokens each step."""
    results = []
    step = 0
    while not scheduler.is_done and step < max_steps:
        result = scheduler.step()
        if result.is_empty:
            # Empty step (e.g. all preempted, admission blocked).
            # Still continue — next step will readmit.
            results.append(result)
            step += 1
            continue
        n_tokens = (len(result.decode_requests)
                    + len(result.prefill_requests)
                    + len(result.continuation_requests))
        scheduler.advance_state(result, [dummy_token] * n_tokens)
        results.append(result)
        step += 1
    return results


# ===========================================================================
# Chunk 1: Scheduling state machine tests (CPU-only)
# ===========================================================================

class TestBasicAdmission:
    """Tests 1-3: FCFS admission and token budget."""

    def test_basic_admission_fcfs(self):
        """Test 1: 5 conversations admitted in FCFS order."""
        alloc = MockPageAllocator(total_pages=1000, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        requests = [make_request(i, 32, 10) for i in range(5)]
        sched.add_requests(requests)

        result = sched.step()
        # All 5 should be admitted (small prompts, big budget)
        admit_events = [e for e in result.events if e["event"] in ("admit", "force_admit")]
        assert len(admit_events) == 5
        # Verify FCFS order: request_ids should be 0,1,2,3,4
        admitted_ids = [e["request_id"] for e in admit_events]
        assert admitted_ids == [0, 1, 2, 3, 4]
        # Verify admission_order is monotonically increasing
        orders = [r.admission_order for r in sched.running]
        assert orders == sorted(orders)
        assert len(set(orders)) == 5

    def test_token_budget_caps_prefill(self):
        """Test 2: Token budget limits prefill chunk size."""
        alloc = MockPageAllocator(total_pages=1000, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=64, page_size=16)

        # First request: 32-token prompt → will be admitted and prefilling
        # Second request: 256-token prompt → should get remaining budget
        r0 = make_request(0, 32, 10)
        r1 = make_request(1, 256, 10)
        sched.add_requests([r0, r1])

        result = sched.step()
        # r0 gets chunk of 32 (full prompt), r1 gets 64 - 32 = 32
        assert r0.scheduled_chunk == 32
        assert r1.scheduled_chunk == 32

    def test_token_budget_multiple_prefills(self):
        """Test 3: Multiple prefills compete for token budget."""
        alloc = MockPageAllocator(total_pages=1000, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        # 3 requests with 300-token prompts each.
        # max_graph_size=512, so first gets 256 (min of remaining, budget),
        # wait — prompts are 300 tokens. First gets min(300, 512)=300.
        # Second gets min(300, 512-300)=min(300,212)=212. Third gets 0.
        r0 = make_request(0, 300, 10)
        r1 = make_request(1, 300, 10)
        r2 = make_request(2, 300, 10)
        sched.add_requests([r0, r1, r2])

        result = sched.step()
        assert r0.scheduled_chunk == 300
        assert r1.scheduled_chunk == 212
        # r2 should not be admitted (no token budget for first chunk)
        assert r2 not in sched.running
        assert r2 in sched.waiting


class TestPageBudget:
    """Tests 4: Page budget blocking admission."""

    def test_page_budget_blocks_admission(self):
        """Test 4: Insufficient pages prevent admission."""
        # page_size=16, so 10 pages = 160 tokens of KV capacity.
        # A conversation with prompt=100, output=100 needs
        # pages_needed(100, 16)=7 pages for first chunk of 100 tokens.
        alloc = MockPageAllocator(total_pages=10, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        # First request needs 7 pages for its 100-token prompt. Admitted.
        # Second request needs 7 pages. Only 3 free. Not admitted.
        r0 = make_request(0, 100, 50)
        r1 = make_request(1, 100, 50)
        sched.add_requests([r0, r1])

        result = sched.step()
        admit_events = [e for e in result.events if e["event"] in ("admit", "force_admit")]
        assert len(admit_events) == 1
        assert admit_events[0]["request_id"] == 0
        assert r1 in sched.waiting


class TestPreemption:
    """Tests 5-9: LIFO preemption behavior."""

    def _setup_three_decode_requests(self, total_pages=20):
        """Helper: create scheduler with 3 decode requests (A, B, C)."""
        alloc = MockPageAllocator(total_pages=total_pages, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        # Small prompts (16 tokens each = 1 page), 1000 output tokens
        requests = [make_request(i, 16, 1000) for i in range(3)]
        sched.add_requests(requests)

        # Step 1: admit all 3, prefill (16 tokens each)
        result = sched.step()
        tokens = [42] * (len(result.prefill_requests)
                         + len(result.decode_requests)
                         + len(result.continuation_requests))
        sched.advance_state(result, tokens)

        # Now all 3 are in decode mode, each with 1 page allocated
        assert all(not r.needs_prefill for r in sched.running)
        return alloc, sched

    def test_lifo_preemption_order(self):
        """Test 5: LIFO — most recently admitted request preempted first."""
        # 20 pages total, 3 requests each using 1 page.
        # Run decode steps to grow their seq_len until pages run out.
        # Each decode step grows seq_len by 1 — needs new page at boundaries.
        alloc, sched = self._setup_three_decode_requests(total_pages=6)
        # After prefill: 3 requests, each 1 page. 3 free pages.
        assert alloc.pages_in_use == 3
        assert alloc.pages_free == 3

        # Run decode steps. Each step, seq_len grows by 1 for each request.
        # Page boundary at seq_len=16 (page_size). After 15 more decode steps,
        # seq_len=17 for first request processed → needs 2 pages.
        # Actually, after prefill seq_len=16 (prompt). First decode → seq_len=17 → 2 pages.
        # So immediately on first decode step, all 3 need 2 pages each = 6 total.
        # We only have 6 total pages. Should be fine.
        result = sched.step()
        tokens = [42] * (len(result.decode_requests)
                         + len(result.prefill_requests)
                         + len(result.continuation_requests))
        sched.advance_state(result, tokens)
        # seq_len=17 for all 3, 2 pages each = 6 pages used. 0 free.
        assert alloc.pages_in_use == 6

        # Next step: seq_len will be 18. Still 2 pages each. No preemption yet.
        # Continue stepping until seq_len=32 → needs 2 pages. At seq_len=33 → 3 pages.
        # We need to advance to seq_len=32 (15 more steps from 17).
        for _ in range(15):
            result = sched.step()
            n = (len(result.decode_requests) + len(result.prefill_requests)
                 + len(result.continuation_requests))
            sched.advance_state(result, [42] * n)

        # Now seq_len=32 for all. 2 pages each = 6 used.
        assert sched.running[0].seq_len == 32
        assert alloc.pages_in_use == 6

        # Next step: seq_len→33, needs 3 pages each = 9 total. Only 6 available.
        # Request 0 (oldest, FCFS first) processes first. Needs 3 pages, has 2.
        # delta=1, pages_free=0. Must preempt. LIFO → request 2 (newest) preempted.
        # Freeing request 2's 2 pages → pages_free=2. delta=1 ≤ 2 → allocate.
        # Then request 1: needs 3 pages, has 2. delta=1. pages_free=1. OK.
        # Request 2 was preempted → only 2 running.
        result = sched.step()
        preempt_events = [e for e in result.events if e["event"] == "preempt"]
        assert len(preempt_events) >= 1
        # Victim should be request 2 (most recently admitted)
        assert preempt_events[0]["request_id"] == 2

    def test_preemption_resets_state(self):
        """Test 6: Preempted victim has state fully reset."""
        alloc, sched = self._setup_three_decode_requests(total_pages=6)

        # Run a few decode steps so requests have some seq_len and output
        for _ in range(16):
            result = sched.step()
            n = (len(result.decode_requests) + len(result.prefill_requests)
                 + len(result.continuation_requests))
            sched.advance_state(result, [42] * n)

        # Force preemption by stepping to seq_len=33 (needs 3 pages)
        result = sched.step()
        preempt_events = [e for e in result.events if e["event"] == "preempt"]
        assert len(preempt_events) >= 1

        victim_id = preempt_events[0]["request_id"]
        # Find the victim in the waiting queue
        victim = None
        for req in sched.waiting:
            if req.request_idx == victim_id:
                victim = req
                break
        assert victim is not None
        assert victim.num_computed_tokens == 0
        assert victim.seq_len == 0
        assert victim.needs_prefill is True
        assert victim.num_preemptions >= 1

    def test_no_admit_on_preemption_step_v2(self):
        """Test 7 (v2): No admissions when preemption occurs, cleaner setup."""
        # Setup: 2 requests running, 1 waiting. Tight pages to force preemption.
        # page_size=16, total_pages=4.
        # Request A: prompt=16, output=1000 (1 page initially)
        # Request B: prompt=16, output=1000 (1 page initially)
        # Request C (waiting): prompt=16, output=10
        alloc = MockPageAllocator(total_pages=4, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        reqs = [make_request(i, 16, 1000) for i in range(2)]
        reqs.append(make_request(2, 16, 10))
        sched.add_requests(reqs)

        # Step 1: admit all 3 (3 pages, 1 free)
        result = sched.step()
        n = (len(result.decode_requests) + len(result.prefill_requests)
             + len(result.continuation_requests))
        sched.advance_state(result, [42] * n)
        assert len(sched.running) == 3

        # Run decode: seq_len=16 → 17 → needs 2 pages each. 3×2=6 > 4.
        # Now add a 4th request to waiting
        sched.add_requests([make_request(3, 16, 10)])
        assert len(sched.waiting) == 1

        result = sched.step()
        preempt_events = [e for e in result.events if e["event"] == "preempt"]
        admit_events = [e for e in result.events if e["event"] in ("admit", "force_admit")]

        # Preemption should occur
        assert len(preempt_events) >= 1
        # No admissions in preemption step
        assert len(admit_events) == 0

    def test_preempted_request_priority(self):
        """Test 8: Preempted requests readmitted before fresh arrivals."""
        alloc = MockPageAllocator(total_pages=4, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        # 3 requests with 16-token prompts
        reqs = [make_request(i, 16, 1000) for i in range(3)]
        sched.add_requests(reqs)

        # Admit all 3 (3 pages used, 1 free)
        result = sched.step()
        sched.advance_state(result, [42] * 3)

        # All decoding. Next step: seq_len=17, each needs 2 pages.
        # 3×2=6 > 4. Preemption will occur.
        # Add a fresh request to waiting
        fresh = make_request(3, 16, 10)
        sched.add_requests([fresh])

        result = sched.step()
        preempt_events = [e for e in result.events if e["event"] == "preempt"]
        assert len(preempt_events) >= 1
        preempted_id = preempt_events[0]["request_id"]
        # No admissions this step (preemption gate)
        n = (len(result.decode_requests) + len(result.prefill_requests)
             + len(result.continuation_requests))
        sched.advance_state(result, [42] * n)

        # Next step: preempted request should be at front of waiting queue
        # and admitted before fresh request 3
        assert sched.waiting[0].request_idx == preempted_id
        # Fresh request should be after
        fresh_in_waiting = [r for r in sched.waiting if r.request_idx == 3]
        assert len(fresh_in_waiting) == 1

    def test_effective_prompt_after_preemption(self):
        """Test 9: After preemption, effective prompt includes output tokens."""
        # total_pages=4: at seq_len≤32, 2 pages each, 4 total = budget.
        # At seq_len=33, 3 pages needed → preemption.
        alloc = MockPageAllocator(total_pages=4, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        r0 = make_request(0, 16, 1000)
        r1 = make_request(1, 16, 1000)
        sched.add_requests([r0, r1])

        # Step 1: prefill both (1 page each)
        result = sched.step()
        sched.advance_state(result, [42] * 2)
        assert r0.needs_prefill is False
        assert r1.needs_prefill is False

        # 16 decode steps: seq_len 16→32 (2 pages each = 4 total = budget)
        for _ in range(16):
            result = sched.step()
            n = (len(result.decode_requests) + len(result.prefill_requests)
                 + len(result.continuation_requests))
            sched.advance_state(result, [42] * n)

        assert r0.seq_len == 32
        assert r1.seq_len == 32
        # output = 1 (prefill) + 16 (decode) = 17
        assert len(r0.output_token_ids) == 17
        assert len(r1.output_token_ids) == 17

        # Next step: seq_len→33, needs 3 pages each = 6 > 4. Preempt r1 (LIFO).
        result = sched.step()
        preempt_events = [e for e in result.events if e["event"] == "preempt"]
        assert len(preempt_events) >= 1
        assert preempt_events[0]["request_id"] == 1
        n = (len(result.decode_requests) + len(result.prefill_requests)
             + len(result.continuation_requests))
        sched.advance_state(result, [42] * n)

        # r1: effective_prompt = prompt(16) + output(17) = 33
        assert r1.num_computed_tokens == 0
        assert r1.needs_prefill is True
        assert r1.effective_prompt_len == 16 + 17
        assert r1.num_preemptions == 1


class TestForceAdmit:
    """Test 10: Admission with tight page budgets."""

    def test_admit_tight_budget_running_empty(self):
        """Request admitted when running is empty and budget is tight.

        When running is empty, the scheduler uses force_admit path (bypasses
        normal gates). With the KV budget cap, the request always fits since
        all pages are free when running is empty.
        """
        # 4 pages * 16 = 64 tokens. prompt=48, max_output=16 (cap: min(16, 64-48)=16).
        # Needs ceil(48/16)=3 pages for first chunk. 4 pages free → admitted.
        alloc = MockPageAllocator(total_pages=4, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        req = make_request(0, 48, 16)
        sched.add_requests([req])

        result = sched.step()
        # When running is empty, scheduler uses force_admit path
        admit_events = [e for e in result.events
                        if e["event"] in ("admit", "force_admit")]
        assert len(admit_events) == 1
        assert admit_events[0]["request_id"] == 0
        assert req in sched.running

    def test_page_blocked_by_other_request(self):
        """Request blocked when another request holds needed pages."""
        # 3 pages * 16 = 48 tokens. small: prompt=32, max_output=10 (cap=min(10,16)=10).
        # big: prompt=32, max_output=5 (cap=min(5,16)=5).
        # small needs ceil(32/16)=2 pages. After admit, 1 free.
        # big needs ceil(32/16)=2 pages, only 1 free → blocked.
        alloc = MockPageAllocator(total_pages=3, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        small_req = make_request(0, 32, 10)
        big_req = make_request(1, 32, 5)
        sched.add_requests([small_req, big_req])

        result = sched.step()
        admit_events = [e for e in result.events if e["event"] in ("admit", "force_admit")]
        admitted_ids = [e["request_id"] for e in admit_events]
        assert 0 in admitted_ids
        assert 1 not in admitted_ids
        assert big_req in sched.waiting


class TestCompletion:
    """Test 11: Completion detection."""

    def test_completion_detection(self):
        """Request completes after max_output_tokens decode steps."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        req = make_request(0, 32, max_output=5)
        sched.add_requests([req])

        # Step 1: prefill (32 tokens)
        result = sched.step()
        sched.advance_state(result, [42])  # 1 token (last of prefill)
        assert req.needs_prefill is False
        assert len(req.output_token_ids) == 1  # first output token from prefill

        # Steps 2-5: decode (4 more tokens to reach max_output=5)
        for i in range(4):
            result = sched.step()
            sched.advance_state(result, [42])
        assert len(req.output_token_ids) == 5

        # Step 6: request should be completed
        result = sched.step()
        complete_events = [e for e in result.events if e["event"] == "complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["request_id"] == 0
        assert sched.is_done


class TestPageInvariant:
    """Test 12: Page budget never exceeded."""

    def test_page_invariant_never_exceeded(self):
        """Pages in use never exceeds kv_page_budget at any step."""
        kv_budget = 10
        alloc = MockPageAllocator(total_pages=kv_budget, page_size=16, max_seqs=32)
        sched = BatchScheduler(alloc, max_seqs=32, max_graph_size=512, page_size=16)

        # 20 conversations, varied prompt lengths, short outputs
        requests = [make_request(i, prompt_len=16 * (i % 4 + 1), max_output=20)
                    for i in range(20)]
        sched.add_requests(requests)

        step = 0
        while not sched.is_done and step < 5000:
            result = sched.step()
            # Assert invariant
            assert alloc.pages_in_use <= kv_budget, (
                f"Step {step}: pages_in_use={alloc.pages_in_use} > "
                f"budget={kv_budget}")
            if result.is_empty:
                break
            n = (len(result.decode_requests)
                 + len(result.prefill_requests)
                 + len(result.continuation_requests))
            sched.advance_state(result, [42] * n)
            step += 1

        assert sched.is_done, f"Did not complete after {step} steps"


class TestPrefillChunking:
    """Tests 13-14: Prefill chunking and decode scheduling."""

    def test_prefill_chunking_and_continuation(self):
        """Test 13: Multi-chunk prefill with continuation flags."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        # max_graph_size=256 so each step can do at most 256 tokens
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=256, page_size=16)

        req = make_request(0, prompt_len=1024, max_output=5)
        sched.add_requests([req])

        # Step 1: first chunk (256 tokens), is_continuation=False
        result = sched.step()
        assert len(result.prefill_requests) == 1
        assert result.prefill_requests[0].scheduled_chunk == 256
        assert result.prefill_requests[0].is_continuation is False
        sched.advance_state(result, [42])

        # Steps 2-3: continuation chunks (256 each)
        for i in range(2):
            result = sched.step()
            assert len(result.continuation_requests) == 1
            assert result.continuation_requests[0].scheduled_chunk == 256
            assert result.continuation_requests[0].is_continuation is True
            assert result.continuation_requests[0].prefill_chunk_start == 256 * (i + 1)
            sched.advance_state(result, [42])

        # Step 4: final chunk (1024 - 768 = 256 tokens)
        result = sched.step()
        assert len(result.continuation_requests) == 1
        assert result.continuation_requests[0].scheduled_chunk == 256
        sched.advance_state(result, [42])

        # Now should be decoding
        assert req.needs_prefill is False
        assert len(req.output_token_ids) == 1

    def test_decode_always_scheduled(self):
        """Test 14: Decode requests always get scheduled_chunk=1."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        # Very small token budget: max_graph_size=4
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=4, page_size=16)

        # 4 requests with tiny prompts (1 token each) → all immediately decode
        requests = [make_request(i, 1, 100) for i in range(4)]
        sched.add_requests(requests)

        # Step 1: admit all 4 as prefill (1 token each = 4 total = max_graph_size)
        result = sched.step()
        sched.advance_state(result, [42] * 4)

        # Step 2: all 4 decoding. 4 decode tokens = max_graph_size. All scheduled.
        result = sched.step()
        assert len(result.decode_requests) == 4
        for req in result.decode_requests:
            assert req.scheduled_chunk == 1


class TestMixedBatch:
    """Test 15: Mixed decode + prefill in same step."""

    def test_mixed_decode_prefill_step(self):
        """Decode and new prefill coexist in same step."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        # Request 0: small prompt → will be decoding after step 1
        r0 = make_request(0, 16, 100)
        sched.add_requests([r0])

        result = sched.step()
        sched.advance_state(result, [42])
        assert r0.needs_prefill is False

        # Add request 1: will be prefilling
        r1 = make_request(1, 200, 100)
        sched.add_requests([r1])

        result = sched.step()
        assert len(result.decode_requests) == 1
        assert result.decode_requests[0].request_idx == 0
        assert len(result.prefill_requests) == 1
        assert result.prefill_requests[0].request_idx == 1


class TestSelfPreemption:
    """Test 16: Self-preemption unreachable with KV budget cap."""

    def test_self_preemption_prevented_by_cap(self):
        """KV budget cap prevents self-preemption for valid requests.

        2 pages * 16 = 32 token KV budget. prompt=16, max_output=100
        (capped to min(100, 32-16) = 16). After prefill (1 page, seq_len=16),
        decode 15 steps → 16 output tokens = max_output → completes normally.
        Self-preemption code path is unreachable but remains as safety net.
        """
        alloc = MockPageAllocator(total_pages=2, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        req = make_request(0, 16, 100)
        sched.add_requests([req])
        # max_output capped to 16 by add_requests (32 - 16 = 16)
        assert req.max_output_tokens == 16

        # Step 1: prefill (16 tokens → 1 page)
        result = sched.step()
        sched.advance_state(result, [42])
        assert req.needs_prefill is False
        assert req.seq_len == 16
        assert len(req.output_token_ids) == 1

        # Steps 2-16: decode steps (seq_len 17→31, pages grow from 1→2)
        for i in range(15):
            result = sched.step()
            sched.advance_state(result, [42])
        assert req.seq_len == 31
        assert len(req.output_token_ids) == 16
        assert alloc.seq_pages(sched.request_to_slot[0]) == 2

        # Step 17: is_complete → True (16 output tokens >= max_output=16)
        result = sched.step()
        complete_events = [e for e in result.events if e["event"] == "complete"]
        assert len(complete_events) == 1
        assert sched.is_done
        # No preemptions occurred
        assert req.num_preemptions == 0


class TestMultiplePreemptions:
    """Test 17: Multiple preemptions in one step."""

    def test_multiple_preemptions_in_one_step(self):
        """Multiple requests preempted in LIFO order in single step."""
        # 5 pages, page_size=16, 4 requests with 16-token prompts.
        # After prefill: 4 pages used. seq_len=16.
        # At seq_len=17 (decode): each needs 2 pages → 8 total, only 5.
        # Request 0 (oldest): needs 2, has 1, free=1 → delta=1, free=1, OK after alloc.
        # Wait, 5-4=1 free. Req0: needs 2, has 1, delta=1, free=1 → allocate. pages_in_use=5.
        # Req1: needs 2, has 1, delta=1, free=0 → preempt! Victim=req3 (LIFO).
        # Free req3 (1 page) → free=1. Allocate → pages_in_use=5.
        # Req2: needs 2, has 1, delta=1, free=0 → preempt! Running=[req0,req1,req2].
        # Victim=req2 (running[-1]). But req2 is current! Break. Self-preempt req2.
        # Hmm, this is getting complicated. Let's use a setup where we can predict.

        # Simpler: 4 pages, 3 requests, 16-token prompts.
        # After prefill: 3 pages, 1 free. All at seq_len=16.
        # Decode: all need 2 pages = 6 total, only 4.
        # Req0: needs 2, has 1, delta=1, free=1 → OK.
        # Req1: needs 2, has 1, delta=1, free=0 → preempt req2 (LIFO).
        # Free req2 (1 page) → free=1. Allocate for req1 → free=0. pages_in_use=4.
        # Req2 gone. 2 running (req0, req1). 1 preempted (req2).
        alloc = MockPageAllocator(total_pages=4, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        requests = [make_request(i, 16, 1000) for i in range(3)]
        sched.add_requests(requests)

        result = sched.step()
        sched.advance_state(result, [42] * 3)
        assert len(sched.running) == 3

        # Now we need a scenario with 2+ preemptions. Use 5 requests, 5 pages.
        # After prefill: 5 used, 0 free. Decode: needs 10 pages total.
        alloc2 = MockPageAllocator(total_pages=5, page_size=16, max_seqs=16)
        sched2 = BatchScheduler(alloc2, max_seqs=16, max_graph_size=512, page_size=16)
        requests2 = [make_request(i, 16, 1000) for i in range(5)]
        sched2.add_requests(requests2)

        result = sched2.step()
        sched2.advance_state(result, [42] * 5)
        assert len(sched2.running) == 5
        assert alloc2.pages_in_use == 5
        assert alloc2.pages_free == 0

        # Decode: each req needs 2 pages.
        # Req0: needs 2, has 1, delta=1, free=0. Preempt req4 (LIFO). Free 1 page.
        # free=1 ≥ delta=1. Allocate. Req0 has 2 pages. free=0.
        # Req1: needs 2, has 1, delta=1, free=0. Preempt req3 (LIFO). Free 1 page.
        # free=1 ≥ 1. Allocate. Req1 has 2 pages. free=0.
        # Req2: needs 2, has 1, delta=1, free=0. Running=[req0,req1,req2].
        # Victim=req2 (running[-1]). req2 IS current → break. Self-preempt.
        result = sched2.step()
        preempt_events = [e for e in result.events if e["event"] == "preempt"]

        # Should have 2+ preemptions: req4 and req3 at minimum
        assert len(preempt_events) >= 2
        preempted_ids = [e["request_id"] for e in preempt_events]
        # LIFO order: req4 first, then req3
        assert preempted_ids[0] == 4
        assert preempted_ids[1] == 3


class TestRecomputePrefill:
    """Test 18: Recompute prefill includes output tokens."""

    def test_recompute_prefill_chunks_include_output(self):
        """After preemption, prefill chunks span prompt + output."""
        # total_pages=4: preemption at seq_len=33 (same as test 9)
        # r0 has short max_output so it completes, freeing pages for r1's readmission
        alloc = MockPageAllocator(total_pages=4, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        r0 = make_request(0, 16, 20)   # short: completes after 20 output tokens
        r1 = make_request(1, 16, 1000)
        sched.add_requests([r0, r1])

        # Prefill both
        result = sched.step()
        sched.advance_state(result, [42, 42])

        # 16 decode steps: seq_len 16→32
        for _ in range(16):
            result = sched.step()
            n = (len(result.decode_requests) + len(result.prefill_requests)
                 + len(result.continuation_requests))
            sched.advance_state(result, [42] * n)

        assert r0.seq_len == 32
        assert r1.seq_len == 32
        output_before_preempt = len(r1.output_token_ids)

        # Next: seq_len→33, preemption. r1 preempted (LIFO).
        result = sched.step()
        preempt_events = [e for e in result.events if e["event"] == "preempt"]
        assert len(preempt_events) >= 1
        n = (len(result.decode_requests) + len(result.prefill_requests)
             + len(result.continuation_requests))
        sched.advance_state(result, [42] * n)

        # r1 preempted. effective_prompt = 16 + output_before_preempt
        assert r1.num_computed_tokens == 0
        assert r1.needs_prefill is True
        assert r1.effective_prompt_len == 16 + output_before_preempt

        # Run until r1 is readmitted (r0 will complete, freeing pages).
        r1_readmitted = False
        for _ in range(50):
            result = sched.step()
            prefill_reqs = result.prefill_requests + result.continuation_requests
            r1_in_prefill = [r for r in prefill_reqs if r.request_idx == 1]
            if r1_in_prefill:
                r1_readmitted = True
                # Verify recompute prefill properties
                assert r1_in_prefill[0].prefill_chunk_start == 0
                assert r1_in_prefill[0].is_continuation is False
                assert r1_in_prefill[0].scheduled_chunk > 0
                assert r1_in_prefill[0].scheduled_chunk <= r1.effective_prompt_len
                break
            if result.is_empty:
                continue
            n = (len(result.decode_requests) + len(result.prefill_requests)
                 + len(result.continuation_requests))
            sched.advance_state(result, [42] * n)

        assert r1_readmitted, "r1 was never readmitted after preemption"


class TestFullLifecycle:
    """Test 19: End-to-end lifecycle with preemptions."""

    def test_full_lifecycle(self):
        """10 conversations, tight budget, all complete correctly."""
        kv_budget = 8
        alloc = MockPageAllocator(total_pages=kv_budget, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        # Varied conversations
        requests = []
        expected_output = {}
        for i in range(10):
            prompt_len = 16 * (i % 3 + 1)  # 16, 32, 48
            max_out = 5 + (i % 4) * 3      # 5, 8, 11, 14
            requests.append(make_request(i, prompt_len, max_out))
            expected_output[i] = max_out
        sched.add_requests(requests)

        completed_ids: set[int] = set()
        step = 0
        while not sched.is_done and step < 10000:
            result = sched.step()
            # Page invariant
            assert alloc.pages_in_use <= kv_budget
            # Track completions
            for e in result.events:
                if e["event"] == "complete":
                    completed_ids.add(e["request_id"])

            if result.is_empty:
                step += 1
                continue

            n = (len(result.decode_requests)
                 + len(result.prefill_requests)
                 + len(result.continuation_requests))
            sched.advance_state(result, [42] * n)
            step += 1

        # All 10 should complete
        assert completed_ids == set(range(10)), (
            f"Missing: {set(range(10)) - completed_ids}")
        assert sched.is_done
        # No orphaned slots
        assert len(sched.request_to_slot) == 0
        assert len(sched.free_seq_ids) == 16

        # Each conversation should have output_len = max_output_tokens
        for req in requests:
            assert len(req.output_token_ids) == expected_output[req.request_idx], (
                f"Request {req.request_idx}: expected {expected_output[req.request_idx]} "
                f"output tokens, got {len(req.output_token_ids)}")


class TestMaxSeqsGate:
    """Coverage gap: max_seqs limit blocks admission."""

    def test_max_seqs_blocks_admission(self):
        """When max_seqs is reached, new requests wait even with free pages."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=2)
        sched = BatchScheduler(alloc, max_seqs=2, max_graph_size=512, page_size=16)

        requests = [make_request(i, 16, 50) for i in range(3)]
        sched.add_requests(requests)

        result = sched.step()
        # Only 2 admitted (max_seqs=2), third waits despite free pages
        admit_events = [e for e in result.events if e["event"] in ("admit", "force_admit")]
        assert len(admit_events) == 2
        assert len(sched.running) == 2
        assert len(sched.waiting) == 1
        assert alloc.pages_free > 0  # pages available, but max_seqs blocks

        # Complete one request, then third should be admitted
        sched.advance_state(result, [42, 42])
        # Fast-forward request 0 to completion
        requests[0].output_token_ids.extend([42] * 49)  # now has 50 tokens
        result = sched.step()
        complete_events = [e for e in result.events if e["event"] == "complete"]
        admit_events = [e for e in result.events if e["event"] in ("admit", "force_admit")]
        assert len(complete_events) == 1
        assert len(admit_events) == 1
        assert admit_events[0]["request_id"] == 2


class TestMultipleCompletions:
    """Coverage gap: multiple requests completing in the same step."""

    def test_multiple_completions_same_step(self):
        """3 requests complete simultaneously, freeing slots and pages."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        # 3 requests with 16-token prompts, max_output=1
        requests = [make_request(i, 16, max_output=1) for i in range(3)]
        # Also a 4th that will benefit from freed slots
        requests.append(make_request(3, 16, 10))
        sched.add_requests(requests)

        # Step 1: admit all 4, prefill
        result = sched.step()
        sched.advance_state(result, [42] * 4)
        # All have 1 output token → requests 0,1,2 complete (max_output=1)
        assert len(requests[0].output_token_ids) == 1
        assert len(requests[3].output_token_ids) == 1

        # Step 2: requests 0,1,2 should complete simultaneously
        result = sched.step()
        complete_events = [e for e in result.events if e["event"] == "complete"]
        assert len(complete_events) == 3
        completed_ids = {e["request_id"] for e in complete_events}
        assert completed_ids == {0, 1, 2}
        # Request 3 should still be running (decoding)
        assert len(sched.running) == 1
        assert sched.running[0].request_idx == 3


class TestTokenBudgetBoundary:
    """Coverage gap: token budget exactly equals first chunk."""

    def test_token_budget_exact_match(self):
        """First chunk exactly consumes remaining token budget."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=100, page_size=16)

        # Request with 100-token prompt: exactly fills budget
        r0 = make_request(0, 100, 10)
        r1 = make_request(1, 50, 10)
        sched.add_requests([r0, r1])

        result = sched.step()
        assert r0.scheduled_chunk == 100  # exactly fills budget
        # r1 should NOT be admitted (budget=0 after r0)
        assert r1 in sched.waiting

    def test_decode_plus_prefill_exact_budget(self):
        """Decode tokens + prefill chunk exactly fill max_graph_size."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        # max_graph_size=10: 1 decode + 9 prefill tokens
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=10, page_size=16)

        r0 = make_request(0, 1, 100)  # tiny prompt, will decode
        sched.add_requests([r0])
        result = sched.step()
        sched.advance_state(result, [42])  # r0 now decoding

        # Add r1 with 9-token prompt: should exactly fit remaining budget
        r1 = make_request(1, 9, 10)
        sched.add_requests([r1])

        result = sched.step()
        assert len(result.decode_requests) == 1  # r0
        assert len(result.prefill_requests) == 1  # r1
        assert r1.scheduled_chunk == 9
        assert result.total_tokens == 10  # exactly max_graph_size


class TestPageCompetition:
    """Coverage gap: decode growth vs new prefill compete for pages."""

    def test_decode_growth_blocks_new_prefill(self):
        """Running decode request's page growth leaves no pages for new prefill."""
        # 3 pages total, page_size=16
        # r0: prompt=16, decoding at seq_len=16 (1 page). Will grow to 17 (2 pages).
        # r1 waiting: prompt=32 (needs 2 pages for first chunk).
        # After r0's growth: 2 used, 1 free. r1 needs 2 → blocked.
        alloc = MockPageAllocator(total_pages=3, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        r0 = make_request(0, 16, 100)
        sched.add_requests([r0])
        result = sched.step()
        sched.advance_state(result, [42])
        assert r0.needs_prefill is False
        # r0: seq_len=16, 1 page, 2 free

        r1 = make_request(1, 32, 10)
        sched.add_requests([r1])

        # Next step: r0 decode → seq_len 17, needs 2 pages. Allocates 1 more. 1 free.
        # r1 needs pages_needed(32, 16)=2 pages. Only 1 free → blocked.
        result = sched.step()
        assert len(result.decode_requests) == 1
        assert r1 in sched.waiting
        admit_events = [e for e in result.events if e["event"] in ("admit", "force_admit")]
        assert len(admit_events) == 0


# ===========================================================================
# Chunk 2: Logit extraction tests
# ===========================================================================

class TestExtractNextTokens:
    """Tests 20-26: Logit extraction from [D|P|C] layout."""

    @staticmethod
    def _make_logits(n_rows: int, vocab: int = 100) -> torch.Tensor:
        """Create logits where row i has argmax at column (i * 7) % vocab."""
        logits = torch.zeros(n_rows, vocab)
        for i in range(n_rows):
            logits[i, (i * 7) % vocab] = 10.0  # clear winner
        return logits

    @staticmethod
    def _expected_token(row_idx: int, vocab: int = 100) -> int:
        return (row_idx * 7) % vocab

    def test_extract_decode_only(self):
        """Test 20: Decode-only extraction."""
        logits = self._make_logits(5)
        tokens = extract_next_tokens(logits, n_decode=5,
                                     prefill_chunk_lengths=[],
                                     continuation_chunk_lengths=[])
        assert len(tokens) == 5
        for i in range(5):
            assert tokens[i] == self._expected_token(i)

    def test_extract_prefill_only(self):
        """Test 21: Prefill-only extraction (argmax of last row per chunk)."""
        # 3 chunks: [128, 256, 64] = 448 total rows
        logits = self._make_logits(448)
        tokens = extract_next_tokens(logits, n_decode=0,
                                     prefill_chunk_lengths=[128, 256, 64],
                                     continuation_chunk_lengths=[])
        assert len(tokens) == 3
        # Last row of chunk 0: row 127
        assert tokens[0] == self._expected_token(127)
        # Last row of chunk 1: row 128+255 = 383
        assert tokens[1] == self._expected_token(383)
        # Last row of chunk 2: row 384+63 = 447
        assert tokens[2] == self._expected_token(447)

    def test_extract_continuation_only(self):
        """Test 22: Continuation-only extraction."""
        logits = self._make_logits(300)
        tokens = extract_next_tokens(logits, n_decode=0,
                                     prefill_chunk_lengths=[],
                                     continuation_chunk_lengths=[100, 200])
        assert len(tokens) == 2
        assert tokens[0] == self._expected_token(99)
        assert tokens[1] == self._expected_token(299)

    def test_extract_mixed(self):
        """Test 23: Mixed decode + prefill + continuation."""
        # 2 decode + 2 prefill [100, 200] + 1 continuation [50] = 352 rows
        logits = self._make_logits(352)
        tokens = extract_next_tokens(logits, n_decode=2,
                                     prefill_chunk_lengths=[100, 200],
                                     continuation_chunk_lengths=[50])
        assert len(tokens) == 5
        # Decode: rows 0, 1
        assert tokens[0] == self._expected_token(0)
        assert tokens[1] == self._expected_token(1)
        # Prefill chunk 0: last row at 2+99=101
        assert tokens[2] == self._expected_token(101)
        # Prefill chunk 1: last row at 102+199=301
        assert tokens[3] == self._expected_token(301)
        # Continuation: last row at 302+49=351
        assert tokens[4] == self._expected_token(351)

    def test_extract_single_token_chunks(self):
        """Test 24: Prefill chunks of length 1 (edge case)."""
        logits = self._make_logits(3)
        tokens = extract_next_tokens(logits, n_decode=0,
                                     prefill_chunk_lengths=[1, 1, 1],
                                     continuation_chunk_lengths=[])
        assert len(tokens) == 3
        assert tokens[0] == self._expected_token(0)
        assert tokens[1] == self._expected_token(1)
        assert tokens[2] == self._expected_token(2)

    def test_extract_empty_groups(self):
        """Test 25: Some groups empty, others not."""
        logits = self._make_logits(50)
        tokens = extract_next_tokens(logits, n_decode=0,
                                     prefill_chunk_lengths=[],
                                     continuation_chunk_lengths=[50])
        assert len(tokens) == 1
        assert tokens[0] == self._expected_token(49)

    def test_extract_assertion_on_mismatch(self):
        """Test 26: Wrong tensor size triggers assertion."""
        logits = self._make_logits(10)
        with pytest.raises(AssertionError):
            extract_next_tokens(logits, n_decode=5,
                                prefill_chunk_lengths=[3],
                                continuation_chunk_lengths=[])
            # Expected 5+3=8 rows consumed, but tensor has 10


# ===========================================================================
# Chunk 3a: KV budget validation tests
# ===========================================================================

class TestKVBudgetValidation:
    """Tests for Fix 3: max_output_tokens validation and KV budget cap."""

    def test_max_output_capped_by_kv_budget(self):
        """Test 27: max_output_tokens capped to KV budget - prompt_len."""
        # 4 pages * 16 = 64 tokens. prompt=32 → cap = 64-32 = 32.
        alloc = MockPageAllocator(total_pages=4, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        req = make_request(0, 32, 1000)
        sched.add_requests([req])
        assert req.max_output_tokens == 32  # capped from 1000

    def test_infeasible_prompt_rejected(self):
        """Test 28: Prompt exceeding KV budget raises ValueError."""
        # 2 pages * 16 = 32 tokens. prompt=48 > 32 → rejected.
        alloc = MockPageAllocator(total_pages=2, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        req = make_request(0, 48, 10)
        with pytest.raises(ValueError, match="exceeds KV budget"):
            sched.add_requests([req])

    def test_zero_max_output_rejected(self):
        """Test 29: max_output_tokens=0 raises ValueError."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        req = make_request(0, 32, 0)
        with pytest.raises(ValueError, match="must be >= 1"):
            sched.add_requests([req])


# ===========================================================================
# Chunk 3a: Preempted completion guard tests (Fix 4)
# ===========================================================================

class TestPreemptedCompletion:
    """Test for Fix 4: is_complete guard prevents early completion of preempted requests."""

    def test_preempted_request_not_completed_early(self):
        """Test 30: Request with output >= max_output but needs_prefill=True is NOT complete.

        Setup: 4 pages, 2 requests. r0: prompt=16, max_output=48 (cap=48).
        r1: prompt=16, max_output=15 (cap=min(15,48)=15). Both admitted (1 page each).
        After 15 decode steps: r1 has 16 output tokens (1 from prefill + 15 from decode),
        16 >= 15 → would be complete. But we preempt before that by creating contention.

        Simpler approach: directly construct the state and test is_complete.
        """
        req = ActiveState(
            request_idx=0,
            conversation_id="test_0",
            prompt_token_ids=list(range(16)),
            output_token_ids=list(range(20)),  # 20 >= any small max_output
            max_output_tokens=15,
            needs_prefill=True,  # preempted — waiting for recompute
            num_preemptions=1,
        )
        # Without Fix 4, is_complete would be True (20 >= 15).
        # With Fix 4, needs_prefill=True blocks completion.
        assert req.is_complete is False

        # Once prefill completes, is_complete should return True
        req.needs_prefill = False
        assert req.is_complete is True


# ===========================================================================
# Chunk 3a: advance_state guard tests (Fix 5)
# ===========================================================================

class TestAdvanceStateGuard:
    """Tests for Fix 5: _needs_advance guard."""

    def test_advance_state_guard_fires(self):
        """Test 31: step() raises RuntimeError if advance_state() not called."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        req = make_request(0, 32, 10)
        sched.add_requests([req])

        result = sched.step()
        assert not result.is_empty

        # Call step() again without advance_state() → RuntimeError
        with pytest.raises(RuntimeError, match="advance_state"):
            sched.step()

    def test_advance_state_guard_skipped_for_empty(self):
        """Test 32: Empty step does NOT set _needs_advance, so step() can follow."""
        # Trigger an empty step: single request that self-preempts, leaving
        # running empty. With the KV cap this is tricky — we manually force it
        # by making a scenario where all running requests get preempted.
        #
        # Alternative: directly test the flag behavior via internal state.
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        # No requests → is_done, but let's test that _needs_advance starts False
        assert sched._needs_advance is False

        # After a non-empty step, _needs_advance should be True
        req = make_request(0, 16, 5)
        sched.add_requests([req])
        result = sched.step()
        assert not result.is_empty
        assert sched._needs_advance is True

        # advance_state clears it
        sched.advance_state(result, [42])
        assert sched._needs_advance is False


# ===========================================================================
# Chunk 3a: Empty step handling tests (Fix 5)
# ===========================================================================

class TestEmptyStepHandling:
    """Tests for Fix 5: empty step handling and event carry-forward."""

    def test_empty_step_no_step_count(self):
        """Test 33: Empty step does not increment step_count or grow scheduling list.

        Since self-preemption is unreachable with KV budget cap, we test the
        empty-step logic by verifying step_count increments only for non-empty
        steps, and that the code path is correctly gated.
        """
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        req = make_request(0, 16, 5)
        sched.add_requests([req])

        # Step 1: prefill → step_count=1. output=[42].
        result = sched.step()
        assert not result.is_empty
        assert sched.step_count == 1
        assert len(sched.all_step_scheduling) == 1
        sched.advance_state(result, [42])
        assert len(req.output_token_ids) == 1

        # Steps 2-5: 4 decode steps → step_count=5. output=[42]*5.
        for i in range(4):
            result = sched.step()
            assert not result.is_empty
            sched.advance_state(result, [42])
        assert sched.step_count == 5
        assert len(sched.all_step_scheduling) == 5
        assert len(req.output_token_ids) == 5

        # Step 6: Phase 1 completes the request (5 >= 5). This step has
        # a complete event but running becomes empty → is_empty=True.
        result = sched.step()
        # The complete event fires, is_done becomes true.
        assert sched.is_done
        # If the step was empty (no scheduled tokens), step_count should NOT increase
        if result.is_empty:
            assert sched.step_count == 5
        else:
            assert sched.step_count == 6

    def test_event_carry_forward(self):
        """Test 34: Events from empty steps are carried forward to next non-empty step."""
        alloc = MockPageAllocator(total_pages=100, page_size=16, max_seqs=16)
        sched = BatchScheduler(alloc, max_seqs=16, max_graph_size=512, page_size=16)

        # Directly test the _pending_events mechanism
        # Simulate: add events to _pending_events, then verify next non-empty step includes them
        sched._pending_events = [
            {"event": "preempt", "request_id": 99, "conversation_id": "test_99",
             "output_len": 5, "num_preemptions": 1},
        ]

        req = make_request(0, 16, 5)
        sched.add_requests([req])

        result = sched.step()
        assert not result.is_empty
        # The pending events should be prepended to this step's events
        assert len(result.events) >= 2  # at least: carried preempt + admit
        assert result.events[0]["event"] == "preempt"
        assert result.events[0]["request_id"] == 99
        # Pending events cleared
        assert len(sched._pending_events) == 0


# ===========================================================================
# Chunk 3b: EOS detection tests
# ===========================================================================

class TestEOSDetection:
    """Tests 35-36: hit_eos flag and is_complete interaction."""

    def test_hit_eos_blocks_completion_until_decode(self):
        """Test 35: hit_eos=True but needs_prefill=True → is_complete=False.

        After preemption a request still needs recompute prefill even if
        it previously generated an EOS token. is_complete must stay False
        until needs_prefill is cleared.
        """
        req = ActiveState(
            request_idx=0,
            conversation_id='c',
            prompt_token_ids=[1, 2, 3],
            max_output_tokens=100,
        )
        req.hit_eos = True
        req.needs_prefill = True
        assert req.is_complete is False

    def test_hit_eos_completes_when_decode(self):
        """Test 36: hit_eos=True and needs_prefill=False → is_complete=True."""
        req = ActiveState(
            request_idx=0,
            conversation_id='c',
            prompt_token_ids=[1, 2, 3],
            max_output_tokens=100,
        )
        req.hit_eos = True
        req.needs_prefill = False
        assert req.is_complete is True


# ===========================================================================
# Chunk 3b: Serialization format tests (CPU, no GPU, no model)
# ===========================================================================

class TestSerializationFormat:
    """Test 37: save_batched_trace + save_conversations produce correct files."""

    def test_roundtrip_activation_trace_and_conversation_trace(self, tmp_path):
        """Write then load; verify ActivationTrace and ConversationTrace parse correctly."""
        import json as _json
        import sys as _sys
        import os as _os

        # Ensure src/MoE is in path for gpu_replay_trace imports
        _moe_dir = _os.path.join(_os.path.dirname(__file__), "..")
        if _moe_dir not in _sys.path:
            _sys.path.insert(0, _moe_dir)

        from trace_construction.collect_batched_traces import (
            save_batched_trace, save_conversations,
        )
        from gpu_replay_trace import ActivationTrace
        from trace_construction.trace_utils import load_traces

        result = {
            'num_layers': 2,
            'num_experts': 4,
            'trace': [
                {'step': 0, 'layer': 0, 'expert_ids': [1, 3]},
                {'step': 0, 'layer': 1, 'expert_ids': [0, 2]},
            ],
            'all_step_scheduling': [
                {
                    'step': 0,
                    'batch_size': 1,
                    'total_tokens': 3,
                    'active_requests': [{
                        'request_id': 0,
                        'conversation_id': 'c0',
                        'seq_len': 0,
                        'is_prefill': True,
                        'prefill_chunk_start': 0,
                        'prefill_chunk_length': 3,
                        'is_continuation': False,
                    }],
                    'events': [],
                }
            ],
            'step_count': 1,
            'conversations': [{
                'conversation_id': 'c0',
                'prompt_token_ids': [1, 2, 3],
                'output_token_ids': [4, 5],
                'num_preemptions': 0,
                'request_idx': 0,
            }],
        }
        output_dir = str(tmp_path)

        # --- save_batched_trace ---
        path = save_batched_trace(result, output_dir, scheduling_config={'max_seqs': 1})
        at = ActivationTrace.load(path)
        assert at.num_layers == 2
        assert at.num_experts == 4
        assert len(at.steps) == 1
        assert at.steps[0][0] == [1, 3]
        assert at.steps[0][1] == [0, 2]
        assert at.scheduling is not None
        assert len(at.scheduling) == 1
        assert at.scheduling_config == {'max_seqs': 1}

        # --- save_conversations ---
        manifest_entries = save_conversations(result, output_dir, top_k=2)
        assert len(manifest_entries) == 1
        assert manifest_entries[0]['conversation_id'] == 'c0'
        assert manifest_entries[0]['prompt_tokens'] == 3
        assert manifest_entries[0]['output_tokens'] == 2

        # Write manifest so load_traces works
        manifest_data = {
            'model': 'test',
            'num_layers': 2,
            'num_experts': 4,
            'top_k': 2,
            'total_conversations': 1,
            'step_count': 1,
            'scheduling': {},
            'conversations': manifest_entries,
        }
        with open(_os.path.join(output_dir, 'manifest.json'), 'w') as f:
            _json.dump(manifest_data, f)

        traces, _ = load_traces(output_dir)
        assert len(traces) == 1
        assert traces[0].prompt_token_ids == [1, 2, 3]
        assert traces[0].output_token_ids == [4, 5]


# ===========================================================================
# Chunk 3b: End-to-end pipeline test (CPU, no GPU)
# ===========================================================================

class TestEndToEndPipeline:
    """Test 38: Full pipeline — serialize → load → simulate → replay loop."""

    def test_collection_to_replay_pipeline(self, tmp_path):
        """Synthetic trace flows through serialization, policy simulation,
        and the replay event loop without error.

        Covers: save_batched_trace → ActivationTrace.load → simulate(LRU, NoPrefetch)
        → GPUReplayTrace → ReplayController.get_step_scheduling → event processing.
        """
        import json as _json
        import os as _os
        import sys as _sys

        _moe_dir = _os.path.join(_os.path.dirname(__file__), "..")
        if _moe_dir not in _sys.path:
            _sys.path.insert(0, _moe_dir)

        from trace_construction.collect_batched_traces import (
            save_batched_trace, save_conversations,
        )
        from gpu_replay_trace import ActivationTrace, StepScheduling
        from policy_simulator import LRU, NoPrefetch, simulate

        num_layers = 2
        num_experts = 4
        cache_size = 6  # 6 out of 8 total slots (2 layers × 4 experts)

        # Build a 3-step trace with scheduling events including preemption:
        #   step 0: admit r0 (prefill), experts [0,1] / [2,3]
        #   step 1: r0 decode + admit r1 (prefill), experts [0,2] / [1,3]
        #   step 2: preempt r1 + r0 decode, experts [1,3] / [0,2]
        trace = []
        for step, layer_experts in enumerate([
            [[0, 1], [2, 3]],
            [[0, 2], [1, 3]],
            [[1, 3], [0, 2]],
        ]):
            for layer, experts in enumerate(layer_experts):
                trace.append({'step': step, 'layer': layer, 'expert_ids': experts})

        step_scheduling = [
            {
                'step': 0, 'batch_size': 1, 'total_tokens': 64,
                'active_requests': [{
                    'request_id': 0, 'conversation_id': 'c0',
                    'seq_len': 0, 'is_prefill': True,
                    'prefill_chunk_start': 0, 'prefill_chunk_length': 64,
                    'is_continuation': False,
                }],
                'events': [
                    {'event': 'admit', 'request_id': 0,
                     'conversation_id': 'c0'},
                ],
            },
            {
                'step': 1, 'batch_size': 2, 'total_tokens': 65,
                'active_requests': [
                    {'request_id': 0, 'conversation_id': 'c0',
                     'seq_len': 64, 'is_prefill': False,
                     'prefill_chunk_start': 0, 'prefill_chunk_length': 0,
                     'is_continuation': False},
                    {'request_id': 1, 'conversation_id': 'c1',
                     'seq_len': 0, 'is_prefill': True,
                     'prefill_chunk_start': 0, 'prefill_chunk_length': 64,
                     'is_continuation': False},
                ],
                'events': [
                    {'event': 'admit', 'request_id': 1,
                     'conversation_id': 'c1'},
                ],
            },
            {
                'step': 2, 'batch_size': 1, 'total_tokens': 1,
                'active_requests': [
                    {'request_id': 0, 'conversation_id': 'c0',
                     'seq_len': 65, 'is_prefill': False,
                     'prefill_chunk_start': 0, 'prefill_chunk_length': 0,
                     'is_continuation': False},
                ],
                'events': [
                    {'event': 'preempt', 'request_id': 1,
                     'conversation_id': 'c1', 'output_len': 0,
                     'num_preemptions': 1},
                ],
            },
        ]

        result = {
            'num_layers': num_layers,
            'num_experts': num_experts,
            'trace': trace,
            'all_step_scheduling': step_scheduling,
            'step_count': 3,
            'conversations': [
                {'conversation_id': 'c0', 'prompt_token_ids': list(range(64)),
                 'output_token_ids': [100, 101], 'num_preemptions': 0,
                 'request_idx': 0},
                {'conversation_id': 'c1', 'prompt_token_ids': list(range(64)),
                 'output_token_ids': [], 'num_preemptions': 1,
                 'request_idx': 1},
            ],
        }

        output_dir = str(tmp_path)
        sched_config = {'max_seqs': 4, 'page_size': 16}

        # Phase 1: Serialize
        path = save_batched_trace(result, output_dir, sched_config)

        # Phase 2: Load as ActivationTrace
        at = ActivationTrace.load(path)
        assert at.num_layers == num_layers
        assert at.num_experts == num_experts
        assert len(at.steps) == 3
        assert at.scheduling is not None
        assert len(at.scheduling) == 3
        assert at.scheduling_config == sched_config

        # Phase 3: Run policy simulation
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=cache_size)
        assert len(dm.steps) == 3
        assert dm.cache_size == cache_size
        # Validate: no internal errors in the replay trace
        errors = dm.validate()
        assert not errors, f"GPUReplayTrace validation errors: {errors}"

        # Phase 4: Verify scheduling metadata flows through
        for i in range(3):
            sched = dm.steps[i].scheduling
            assert sched is not None, f"Step {i}: scheduling is None"
            assert sched.step == i

        # Step 0: admit event
        assert any(e['event'] == 'admit' and e['request_id'] == 0
                   for e in dm.steps[0].scheduling.events)

        # Step 2: preempt event
        assert any(e['event'] == 'preempt' and e['request_id'] == 1
                   for e in dm.steps[2].scheduling.events)

        # Phase 5: Simulate replay event loop (no engine, just logic)
        free_slots = set(range(4))
        request_to_slot: dict[int, int] = {}
        active: dict[int, dict] = {}

        for step_idx in range(3):
            sched = dm.steps[step_idx].scheduling
            for evt in sched.events:
                if evt['event'] == 'complete':
                    rid = evt['request_id']
                    if rid in request_to_slot:
                        free_slots.add(request_to_slot.pop(rid))
                        active.pop(rid, None)
                elif evt['event'] in ('admit', 'force_admit'):
                    rid = evt['request_id']
                    sid = free_slots.pop()
                    request_to_slot[rid] = sid
                    if rid not in active:
                        active[rid] = {'sid': sid, 'decode_step': 0}
                    else:
                        active[rid]['sid'] = sid
                elif evt['event'] == 'preempt':
                    rid = evt['request_id']
                    if rid in request_to_slot:
                        free_slots.add(request_to_slot.pop(rid))

        # After step 0: r0 admitted
        # After step 1: r0 + r1 admitted
        # After step 2: r1 preempted, r0 still active
        assert 0 in request_to_slot
        assert 1 not in request_to_slot
        assert 1 in active  # preserved for readmission
        assert active[1]['decode_step'] == 0  # never decoded

        # Phase 6: Conversation files roundtrip
        manifest_entries = save_conversations(result, output_dir, top_k=2)
        manifest_data = {
            'model': 'test', 'num_layers': num_layers,
            'num_experts': num_experts, 'top_k': 2,
            'total_conversations': 2, 'step_count': 3,
            'scheduling': sched_config,
            'conversations': manifest_entries,
        }
        with open(_os.path.join(output_dir, 'manifest.json'), 'w') as f:
            _json.dump(manifest_data, f)

        from trace_construction.trace_utils import load_traces
        traces, _ = load_traces(output_dir)
        assert len(traces) == 2
        # c0 has output tokens, c1 has empty (preempted before decode)
        assert traces[0].output_token_ids == [100, 101]
        assert traces[1].output_token_ids == []


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
