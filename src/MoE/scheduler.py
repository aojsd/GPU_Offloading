"""Unified Scheduler: continuous batching state machine + GPU orchestration.

Combines the vLLM V1-style scheduling state machine with GPU collection and
replay into a single class. Works with MockPageAllocator for CPU-only tests
and with MoEEngine for GPU workloads.

This module contains:
- ActiveState: per-request mutable state
- ScheduleResult: typed return from scheduler
- PageAllocator: protocol for page management
- MockPageAllocator: CPU-side page pool for testing
- Scheduler: unified state machine + GPU orchestration
- extract_next_tokens: logit extraction from step [D|P|C] layout
- CollectionResult / ReplayResult: typed results for GPU methods
- Serialization helpers: save_batched_trace, save_conversations
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch


def pages_needed(seq_len: int, page_size: int) -> int:
    """KV pages for a sequence of given length."""
    if seq_len <= 0:
        return 0
    return math.ceil(seq_len / page_size)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ActiveState:
    """Per-request mutable state for continuous batching.

    Effective prompt = prompt_token_ids + output_token_ids.
    For fresh requests this equals the original prompt.
    After preemption with N generated tokens, the model recomputes KV for
    the full prompt + output[:N] sequence from scratch.
    """
    request_idx: int                          # index in original conversation list
    conversation_id: str
    prompt_token_ids: list[int]               # original prompt (immutable)
    output_token_ids: list[int] = field(default_factory=list)
    max_output_tokens: int = 0
    num_computed_tokens: int = 0              # prefill progress into effective_prompt
    seq_len: int = 0                          # current KV cache length
    needs_prefill: bool = True
    admission_order: int = 0                  # monotonically increasing, for LIFO
    num_preemptions: int = 0
    hit_eos: bool = False
    # Per-step (set by scheduler):
    scheduled_chunk: int = 0
    prefill_chunk_start: int = 0
    is_continuation: bool = False

    @property
    def effective_prompt_len(self) -> int:
        """Length of the full sequence to (re)compute: prompt + output so far."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_complete(self) -> bool:
        return ((len(self.output_token_ids) >= self.max_output_tokens
                 or self.hit_eos)
                and not self.needs_prefill)


@dataclass
class ScheduleResult:
    """What the scheduler decided to execute this step."""
    decode_seq_ids: list[int] = field(default_factory=list)
    decode_requests: list[ActiveState] = field(default_factory=list)
    prefill_seq_ids: list[int] = field(default_factory=list)
    prefill_requests: list[ActiveState] = field(default_factory=list)
    prefill_chunk_lengths: list[int] = field(default_factory=list)
    continuation_seq_ids: list[int] = field(default_factory=list)
    continuation_requests: list[ActiveState] = field(default_factory=list)
    continuation_chunk_lengths: list[int] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    step_scheduling: dict = field(default_factory=dict)
    preempted_this_step: bool = False
    total_tokens: int = 0
    is_empty: bool = True


# ---------------------------------------------------------------------------
# Page allocator protocol + mock
# ---------------------------------------------------------------------------

@runtime_checkable
class PageAllocator(Protocol):
    total_pages: int
    @property
    def pages_free(self) -> int: ...
    @property
    def pages_in_use(self) -> int: ...
    def alloc_pages(self, seq_id: int, n_pages: int) -> None: ...
    def free_seq(self, seq_id: int) -> None: ...
    def ensure_pages(self, seq_id: int, needed: int) -> None: ...
    def seq_pages(self, seq_id: int) -> int: ...


class MockPageAllocator:
    """CPU-side page pool matching MoEEngine's dynamic page interface.

    Used for CPU-only scheduler testing without any GPU dependency.
    """

    def __init__(self, total_pages: int, page_size: int = 16, max_seqs: int = 256):
        self.total_pages = total_pages
        self.page_size = page_size
        self._free_pages: deque[int] = deque(range(total_pages))
        self._seq_page_list: list[list[int]] = [[] for _ in range(max_seqs)]

    @property
    def pages_free(self) -> int:
        return len(self._free_pages)

    @property
    def pages_in_use(self) -> int:
        return self.total_pages - len(self._free_pages)

    def alloc_pages(self, seq_id: int, n_pages: int) -> None:
        if n_pages <= 0:
            return
        if len(self._free_pages) < n_pages:
            raise RuntimeError(
                f"Cannot allocate {n_pages} pages for seq {seq_id}: "
                f"only {len(self._free_pages)} free pages remain "
                f"(budget={self.total_pages})")
        pages = [self._free_pages.popleft() for _ in range(n_pages)]
        self._seq_page_list[seq_id].extend(pages)

    def free_seq(self, seq_id: int) -> None:
        pages = self._seq_page_list[seq_id]
        if pages:
            self._free_pages.extend(pages)
            self._seq_page_list[seq_id] = []

    def ensure_pages(self, seq_id: int, needed: int) -> None:
        current = len(self._seq_page_list[seq_id])
        if needed > current:
            self.alloc_pages(seq_id, needed - current)

    def seq_pages(self, seq_id: int) -> int:
        return len(self._seq_page_list[seq_id])


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Unified vLLM V1-style continuous batching scheduler.

    Implements greedy admission, LIFO preemption with recompute, and
    no-admission-on-preemption gating. The scheduler operates on a
    PageAllocator (MockPageAllocator for tests, MoEEngine for production).

    CPU mode: pass MockPageAllocator. Use step(), advance_state(), add_requests().
    GPU mode: pass MoEEngine. Additionally use collect() and replay().

    Usage:
        scheduler = Scheduler(allocator, max_seqs=256, max_graph_size=512)
        scheduler.add_requests([...])
        while not scheduler.is_done:
            result = scheduler.step()
            # caller runs engine.step() using result
            # caller extracts tokens from logits
            scheduler.advance_state(result, next_tokens)
    """

    def __init__(
        self,
        allocator: PageAllocator | MockPageAllocator,
        max_seqs: int,
        max_graph_size: int,
        page_size: int = 16,
    ):
        self.allocator = allocator
        self.max_seqs = max_seqs
        self.max_graph_size = max_graph_size
        self.page_size = page_size

        self.running: list[ActiveState] = []         # maintained in FCFS order
        self.waiting: deque[ActiveState] = deque()   # FCFS, preempted at front
        self.free_seq_ids: set[int] = set(range(max_seqs))
        self.request_to_slot: dict[int, int] = {}    # request_idx -> seq_id

        self._admission_counter: int = 0
        self.step_count: int = 0
        self.all_step_scheduling: list[dict] = []
        self._needs_advance: bool = False
        self._pending_events: list[dict] = []

    @property
    def engine(self):
        """Return self.allocator as engine. For GPU methods only."""
        return self.allocator

    @property
    def is_done(self) -> bool:
        return len(self.running) == 0 and len(self.waiting) == 0

    def _reset(self):
        """Reset internal state for a fresh collection/replay run."""
        self.running.clear()
        self.waiting.clear()
        self.free_seq_ids = set(range(self.max_seqs))
        self.request_to_slot.clear()
        self._admission_counter = 0
        self.step_count = 0
        self.all_step_scheduling.clear()
        self._needs_advance = False
        self._pending_events.clear()

    def add_requests(self, requests: list[ActiveState]) -> None:
        """Enqueue new requests to the waiting queue.

        Validates max_output_tokens >= 1 and caps by KV budget to prevent
        single-request self-preemption livelock.
        """
        max_seq_tokens = self.allocator.total_pages * self.page_size
        for req in requests:
            if req.max_output_tokens < 1:
                raise ValueError(
                    f"Request {req.request_idx}: max_output_tokens must be >= 1, "
                    f"got {req.max_output_tokens}")
            cap = max_seq_tokens - len(req.prompt_token_ids)
            if cap < 1:
                raise ValueError(
                    f"Request {req.request_idx}: prompt ({len(req.prompt_token_ids)} "
                    f"tokens) exceeds KV budget ({max_seq_tokens} tokens)")
            req.max_output_tokens = min(req.max_output_tokens, cap)
        self.waiting.extend(requests)

    def step(self) -> ScheduleResult:
        """Run one scheduling step. Returns execution groups for step.

        Implements the 6-phase vLLM V1 scheduling loop:
        1. Complete finished requests
        2. Schedule running + allocate pages (with preemption)
        3. Admit new requests (gated on no preemptions)
        4. Build execution groups
        5. Record metadata
        6. Return ScheduleResult
        """
        if self._needs_advance:
            raise RuntimeError(
                "step() called without advance_state() after previous non-empty step. "
                "Call advance_state() before calling step() again.")

        result = ScheduleResult()
        events: list[dict] = []
        preempted_this_step = False

        # ── Phase 1: Complete ────────────────────────────────────────────
        still_running: list[ActiveState] = []
        for req in self.running:
            if req.is_complete:
                slot = self.request_to_slot.pop(req.request_idx)
                self.allocator.free_seq(slot)
                self.free_seq_ids.add(slot)
                events.append({
                    "event": "complete",
                    "request_id": req.request_idx,
                    "conversation_id": req.conversation_id,
                    "output_len": len(req.output_token_ids),
                })
            else:
                still_running.append(req)
        self.running = still_running

        # ── Phase 2: Schedule running + page allocation (with preemption)
        # Compute an approximate token budget for page estimation.
        # Subtract decode tokens first (they always consume 1 token each).
        num_decode = sum(1 for r in self.running if not r.needs_prefill)
        approx_prefill_budget = self.max_graph_size - num_decode

        # Iterate running in FCFS order, allocate pages, preempt if needed.
        i = 0
        while i < len(self.running):
            req = self.running[i]
            slot = self.request_to_slot[req.request_idx]

            if req.needs_prefill:
                # Estimate chunk for page allocation. Use the prefill-only
                # portion of the token budget to avoid over-allocating.
                remaining = req.effective_prompt_len - req.num_computed_tokens
                chunk_estimate = min(remaining, max(approx_prefill_budget, 0))
                if chunk_estimate <= 0:
                    i += 1
                    continue
                needed = pages_needed(
                    req.num_computed_tokens + chunk_estimate, self.page_size)
            else:
                # Decode: need pages for seq_len + 1
                needed = pages_needed(req.seq_len + 1, self.page_size)

            current_pages = self.allocator.seq_pages(slot)
            delta = needed - current_pages

            if delta > 0 and self.allocator.pages_free < delta:
                # ── Preemption loop ──────────────────────────────────
                preempted_this_step = True
                while (self.allocator.pages_free < delta
                       and len(self.running) > 1):
                    # LIFO: evict from tail of running
                    victim = self.running[-1]
                    if victim is req:
                        # Can't preempt ourselves from the tail if we're
                        # the last one. Try second-to-last.
                        if len(self.running) < 2:
                            break
                        # We are at the tail — this shouldn't happen if
                        # i < len(running)-1, but handle gracefully.
                        break

                    victim_slot = self.request_to_slot.pop(victim.request_idx)
                    self.allocator.free_seq(victim_slot)
                    self.free_seq_ids.add(victim_slot)

                    # Reset victim for full recompute
                    victim.num_computed_tokens = 0
                    victim.seq_len = 0
                    victim.needs_prefill = True
                    victim.num_preemptions += 1
                    victim.scheduled_chunk = 0
                    victim.prefill_chunk_start = 0
                    victim.is_continuation = False

                    # Prepend to waiting (priority over fresh arrivals)
                    self.waiting.appendleft(victim)
                    self.running.pop()  # remove from tail
                    events.append({
                        "event": "preempt",
                        "request_id": victim.request_idx,
                        "conversation_id": victim.conversation_id,
                        "output_len": len(victim.output_token_ids),
                        "num_preemptions": victim.num_preemptions,
                    })

                # After preemption loop, check if we can now allocate
                if delta > 0 and self.allocator.pages_free < delta:
                    # Still can't fit — self-preempt
                    self_slot = self.request_to_slot.pop(req.request_idx)
                    self.allocator.free_seq(self_slot)
                    self.free_seq_ids.add(self_slot)
                    req.num_computed_tokens = 0
                    req.seq_len = 0
                    req.needs_prefill = True
                    req.num_preemptions += 1
                    req.scheduled_chunk = 0
                    req.prefill_chunk_start = 0
                    req.is_continuation = False
                    self.waiting.appendleft(req)
                    self.running.pop(i)
                    events.append({
                        "event": "preempt",
                        "request_id": req.request_idx,
                        "conversation_id": req.conversation_id,
                        "output_len": len(req.output_token_ids),
                        "num_preemptions": req.num_preemptions,
                    })
                    # Don't increment i — next element shifted into position
                    continue
                else:
                    # Allocate the pages we need
                    self.allocator.ensure_pages(slot, needed)
            elif delta > 0:
                self.allocator.ensure_pages(slot, needed)

            i += 1

        # Second pass: compute token budget and assign chunks.
        # Decode tokens are always scheduled (1 each).
        num_decode = sum(1 for r in self.running if not r.needs_prefill)
        token_budget = self.max_graph_size - num_decode

        for req in self.running:
            slot = self.request_to_slot[req.request_idx]
            if req.needs_prefill:
                remaining = req.effective_prompt_len - req.num_computed_tokens
                chunk = min(remaining, token_budget)
                if chunk <= 0:
                    req.scheduled_chunk = 0
                    req.prefill_chunk_start = req.num_computed_tokens
                    req.is_continuation = (req.num_computed_tokens > 0)
                    continue
                token_budget -= chunk
                req.scheduled_chunk = chunk
                req.prefill_chunk_start = req.num_computed_tokens
                req.is_continuation = (req.num_computed_tokens > 0)
                # Ensure pages for the chunk we're actually scheduling
                needed = pages_needed(
                    req.num_computed_tokens + chunk, self.page_size)
                self.allocator.ensure_pages(slot, needed)
            else:
                req.scheduled_chunk = 1
                req.prefill_chunk_start = 0
                req.is_continuation = False

        # ── Phase 3: Admit new requests (gated on no preemptions) ────────
        if not preempted_this_step:
            while self.waiting:
                # Force-admit safety net: if nothing running, admit one
                force = (len(self.running) == 0)

                req = self.waiting[0]  # peek
                remaining_prompt = req.effective_prompt_len - req.num_computed_tokens
                first_chunk = min(remaining_prompt, max(token_budget, 0))
                if first_chunk <= 0 and not force:
                    break
                if force:
                    # Cap by available pages (all should be free since running is empty)
                    max_by_pages = self.allocator.pages_free * self.page_size
                    first_chunk = min(
                        remaining_prompt,
                        max(max_by_pages, 1),
                        max(self.max_graph_size - num_decode, 1))

                first_chunk_pages = pages_needed(first_chunk, self.page_size)

                # Three admission gates (bypassed if force)
                if not force:
                    if len(self.running) >= self.max_seqs:
                        break
                    if self.allocator.pages_free < first_chunk_pages:
                        break
                    if token_budget < first_chunk:
                        break

                # Admit
                req = self.waiting.popleft()
                if not self.free_seq_ids:
                    # No free slots — shouldn't happen if max_seqs is correct
                    self.waiting.appendleft(req)
                    break

                slot = min(self.free_seq_ids)  # deterministic slot selection
                self.free_seq_ids.remove(slot)
                self.request_to_slot[req.request_idx] = slot

                req.admission_order = self._admission_counter
                self._admission_counter += 1

                # Allocate pages for first chunk (guard against zero free pages)
                actual_pages = pages_needed(first_chunk, self.page_size)
                if actual_pages > self.allocator.pages_free:
                    # Can't allocate — return slot and break
                    self.free_seq_ids.add(slot)
                    del self.request_to_slot[req.request_idx]
                    self.waiting.appendleft(req)
                    break
                self.allocator.alloc_pages(slot, actual_pages)

                req.scheduled_chunk = first_chunk
                req.prefill_chunk_start = req.num_computed_tokens
                req.is_continuation = (req.num_computed_tokens > 0)
                token_budget -= first_chunk

                self.running.append(req)
                event_type = "force_admit" if force else "admit"
                events.append({
                    "event": event_type,
                    "request_id": req.request_idx,
                    "conversation_id": req.conversation_id,
                    "is_recompute": req.num_preemptions > 0,
                    "effective_prompt_len": req.effective_prompt_len,
                })

        # ── Phase 4: Build execution groups ──────────────────────────────
        total_tokens = 0
        for req in self.running:
            if req.scheduled_chunk <= 0:
                continue
            slot = self.request_to_slot[req.request_idx]
            if not req.needs_prefill:
                # Decode
                result.decode_seq_ids.append(slot)
                result.decode_requests.append(req)
                total_tokens += 1
            elif req.is_continuation:
                # Continuing prefill
                result.continuation_seq_ids.append(slot)
                result.continuation_requests.append(req)
                result.continuation_chunk_lengths.append(req.scheduled_chunk)
                total_tokens += req.scheduled_chunk
            else:
                # New prefill (first chunk)
                result.prefill_seq_ids.append(slot)
                result.prefill_requests.append(req)
                result.prefill_chunk_lengths.append(req.scheduled_chunk)
                total_tokens += req.scheduled_chunk

        result.total_tokens = total_tokens
        result.is_empty = (total_tokens == 0)
        result.preempted_this_step = preempted_this_step
        result.events = events

        # ── Phase 5: Record metadata (only for non-empty steps) ─────────
        if not result.is_empty:
            # Prepend any pending events from previous empty steps
            if self._pending_events:
                result.events = self._pending_events + result.events
                self._pending_events = []

            active_requests_meta = []
            for req in self.running:
                if req.scheduled_chunk <= 0:
                    continue
                active_requests_meta.append({
                    "request_id": req.request_idx,
                    "conversation_id": req.conversation_id,
                    "seq_len": req.seq_len,
                    "is_prefill": req.needs_prefill,
                    "prefill_chunk_start": req.prefill_chunk_start,
                    "prefill_chunk_length": (req.scheduled_chunk
                                             if req.needs_prefill else 0),
                    "is_continuation": req.is_continuation,
                })

            step_meta = {
                "step": self.step_count,
                "batch_size": len(active_requests_meta),
                "total_tokens": total_tokens,
                "active_requests": active_requests_meta,
                "events": result.events,
            }
            result.step_scheduling = step_meta
            self.all_step_scheduling.append(step_meta)
            self.step_count += 1
            self._needs_advance = True
        else:
            # Empty step: buffer events for carry-forward
            self._pending_events.extend(events)

        return result

    def advance_state(self, result: ScheduleResult, next_tokens: list[int]) -> None:
        """Advance request states after GPU execution + token extraction.

        Args:
            result: The ScheduleResult from the most recent step().
            next_tokens: Token IDs in order:
                [decode_0, ..., decode_{D-1},
                 prefill_0, ..., prefill_{P-1},
                 continuation_0, ..., continuation_{C-1}]
        """
        self._needs_advance = False

        expected = (len(result.decode_requests)
                    + len(result.prefill_requests)
                    + len(result.continuation_requests))
        if len(next_tokens) != expected:
            raise ValueError(
                f"Expected {expected} tokens, got {len(next_tokens)}")

        idx = 0

        # Decode requests: append token, advance seq_len
        for req in result.decode_requests:
            req.output_token_ids.append(next_tokens[idx])
            req.seq_len += 1
            idx += 1

        # Prefill + continuation: advance num_computed, maybe transition to decode
        for req in result.prefill_requests + result.continuation_requests:
            token = next_tokens[idx]
            idx += 1
            req.num_computed_tokens += req.scheduled_chunk
            req.seq_len = req.num_computed_tokens
            if req.num_computed_tokens >= req.effective_prompt_len:
                # Prefill complete — transition to decode
                req.needs_prefill = False
                req.output_token_ids.append(token)

    # ===================================================================
    # GPU orchestration: collect()
    # ===================================================================

    def collect(self, conversations: list[dict]) -> 'CollectionResult':
        """Run continuous batching on GPU. Returns CollectionResult.

        Uses self as the state machine (resets internal state first).
        Wires TraceRecorder automatically. Handles EOS detection.

        Args:
            conversations: list of dicts with keys:
                conversation_id, prompt_token_ids, max_output_tokens.
        """
        self._reset()
        engine = self.engine

        assert self.max_seqs == engine.max_seqs, (
            f"Scheduler/engine max_seqs mismatch: "
            f"{self.max_seqs} vs {engine.max_seqs}")

        # Lazy GPU imports
        from trace_recorder import TraceRecorder

        # Attach trace recorder — save/restore previous
        _prev_recorder = getattr(engine, 'trace_recorder', None)
        recorder = TraceRecorder(engine.num_layers, engine.num_experts)
        engine.trace_recorder = recorder

        # Build requests
        requests = [
            ActiveState(
                request_idx=i,
                conversation_id=conv['conversation_id'],
                prompt_token_ids=conv['prompt_token_ids'],
                max_output_tokens=conv['max_output_tokens'],
            )
            for i, conv in enumerate(conversations)
        ]

        # Pre-load prompts on GPU
        prompts_gpu = {
            i: torch.tensor(conv['prompt_token_ids'], dtype=torch.long,
                            device=engine.device)
            for i, conv in enumerate(conversations)
        }

        self.add_requests(requests)

        with torch.inference_mode():
            while not self.is_done:
                result = self.step()
                if result.is_empty:
                    continue

                # 1. Decode tokens
                if result.decode_requests:
                    decode_token_ids = torch.tensor(
                        [r.output_token_ids[-1] for r in result.decode_requests],
                        dtype=torch.long, device=engine.device)
                else:
                    decode_token_ids = torch.tensor(
                        [], dtype=torch.long, device=engine.device)

                # 2. Prefill tokens
                prefill_input_ids = []
                for req in result.prefill_requests:
                    eff = _get_effective_prompt_gpu(req, prompts_gpu)
                    prefill_input_ids.append(
                        eff[req.prefill_chunk_start:
                            req.prefill_chunk_start + req.scheduled_chunk])

                # 3. Continuation tokens + offsets
                continuation_input_ids = []
                continuation_offsets = []
                for req in result.continuation_requests:
                    eff = _get_effective_prompt_gpu(req, prompts_gpu)
                    continuation_input_ids.append(
                        eff[req.prefill_chunk_start:
                            req.prefill_chunk_start + req.scheduled_chunk])
                    slot = self.request_to_slot[req.request_idx]
                    continuation_offsets.append(
                        engine._seq_lens_cpu[slot].item())

                # 4. GPU forward pass
                logits = engine.step(
                    decode_seq_ids=result.decode_seq_ids,
                    decode_token_ids=decode_token_ids,
                    prefill_seq_ids=result.prefill_seq_ids,
                    prefill_input_ids=prefill_input_ids,
                    continuation_seq_ids=result.continuation_seq_ids,
                    continuation_input_ids=continuation_input_ids,
                    continuation_offsets=continuation_offsets,
                )

                # 5. Extract tokens
                next_tokens = extract_next_tokens(
                    logits, len(result.decode_requests),
                    result.prefill_chunk_lengths,
                    result.continuation_chunk_lengths,
                )

                # 5b. EOS detection — must run BEFORE advance_state so that
                # is_complete=True is visible in Phase 1 of the NEXT step.
                eos_id = getattr(engine, 'eos_token_id', None)
                if eos_id is not None:
                    batch_order = (result.decode_requests
                                   + result.prefill_requests
                                   + result.continuation_requests)
                    for _i, _req in enumerate(batch_order):
                        if _i < len(next_tokens) and next_tokens[_i] == eos_id:
                            _req.hit_eos = True

                # 6. Advance state
                self.advance_state(result, next_tokens)

        engine.trace_recorder = _prev_recorder

        # Sanity: recorder and scheduler step counters must agree.
        assert (recorder._step == self.step_count - 1
                or (recorder._step == -1 and self.step_count == 0)), (
            f"Recorder step {recorder._step} != scheduler step "
            f"{self.step_count - 1}")

        return CollectionResult(
            all_step_scheduling=self.all_step_scheduling,
            step_count=self.step_count,
            conversations=[
                {'conversation_id': r.conversation_id,
                 'request_idx': r.request_idx,
                 'prompt_token_ids': r.prompt_token_ids,
                 'output_token_ids': r.output_token_ids,
                 'num_preemptions': r.num_preemptions}
                for r in requests
            ],
            trace=recorder.trace,
            num_layers=engine.num_layers,
            num_experts=engine.num_experts,
        )

    # ===================================================================
    # GPU orchestration: replay()
    # ===================================================================

    def replay(
        self,
        conversations: list[dict],
        *,
        scheduling: list = None,
        controller=None,
        n_steps: int = None,
        record_routing: bool = False,
        record_tokens: bool = True,
    ) -> 'ReplayResult':
        """Replay pre-recorded scheduling on GPU.

        The unified replay loop handling events, token args, page management,
        and the advance_state token protocol (Pitfalls 19-20).

        Args:
            conversations: list of dicts with prompt_token_ids, output_token_ids,
                and request_idx (or index as implicit request_idx).
            scheduling: list[StepScheduling] — used when controller is None.
            controller: ReplayController — provides scheduling via
                get_step_scheduling(). Mutually exclusive with scheduling.
            n_steps: override step count (default: len(scheduling) or
                len(controller trace steps)).
            record_routing: attach TraceRecorder, return trace in result.
            record_tokens: track output tokens for comparison (default True).

        Returns:
            ReplayResult with output_tokens, trace_data, steps.
        """
        engine = self.engine
        engine.reset()

        # Normalize conversations: ensure request_idx key exists
        for i, conv in enumerate(conversations):
            if 'request_idx' not in conv:
                conv['request_idx'] = i

        # Build full token sequences (prompt + output) for faithful replay
        full_token_seqs = _build_full_token_seqs(conversations, engine.device)

        # Per-conv data for decode token lookup
        per_conv_output_ids = {
            conv['request_idx']: conv.get('output_token_ids', [])
            for conv in conversations
        }
        prompt_lens = {
            conv['request_idx']: len(conv['prompt_token_ids'])
            for conv in conversations
        }

        # Attach TraceRecorder or ReplayController
        _prev_recorder = getattr(engine, 'trace_recorder', None)
        _prev_controller = getattr(engine, 'replay_controller', None)
        recorder = None
        if record_routing:
            from trace_recorder import TraceRecorder
            recorder = TraceRecorder(engine.num_layers, engine.num_experts)
            engine.trace_recorder = recorder
        if controller is not None:
            controller.setup()
            engine.replay_controller = controller

        # Determine step count and scheduling source
        if controller is not None:
            total_steps = n_steps or len(controller.trace.steps)
            def get_sched(s):
                return controller.get_step_scheduling(s)
        else:
            total_steps = n_steps or len(scheduling)
            def get_sched(s):
                return scheduling[s]

        # Per-request state
        free_seq_ids = set(range(self.max_seqs))
        request_to_slot: dict[int, int] = {}
        active_requests: dict[int, dict] = {}
        output_tokens: dict[int, list] = {
            conv['request_idx']: [] for conv in conversations
        }
        replay_preemptions: dict[int, int] = {
            conv['request_idx']: 0 for conv in conversations
        }
        skipped_admissions = 0

        with torch.inference_mode():
            for step in range(total_steps):
                sched = get_sched(step)
                if sched is None:
                    break

                # 1. Process events
                for evt in sched.events:
                    if evt['event'] == 'complete':
                        rid = evt['request_id']
                        if rid in request_to_slot:
                            sid = request_to_slot.pop(rid)
                            engine.free_seq(sid)
                            free_seq_ids.add(sid)
                            active_requests.pop(rid, None)

                    elif evt['event'] in ('admit', 'force_admit'):
                        rid = evt['request_id']
                        if not free_seq_ids:
                            skipped_admissions += 1
                            continue
                        sid = free_seq_ids.pop()
                        request_to_slot[rid] = sid
                        if rid not in active_requests:
                            active_requests[rid] = {
                                'sid': sid, 'decode_step': 0,
                                'num_computed': 0, 'needs_prefill': True,
                            }
                        else:
                            active_requests[rid]['sid'] = sid
                            # Readmission: reset prefill progress for recompute
                            active_requests[rid]['num_computed'] = 0
                            active_requests[rid]['needs_prefill'] = True

                    elif evt['event'] == 'preempt':
                        rid = evt['request_id']
                        replay_preemptions[rid] += 1
                        if rid in request_to_slot:
                            sid = request_to_slot.pop(rid)
                            engine.free_seq(sid)
                            free_seq_ids.add(sid)
                            # active_requests[rid] is kept intentionally:
                            # preserves decode_step for readmission.

                # 2. Build step args from scheduling metadata
                decode_sids = []
                decode_tokens_list = []
                prefill_sids = []
                prefill_input_ids = []
                prefill_chunk_lengths = []
                cont_sids = []
                cont_input_ids = []
                cont_offsets = []
                cont_chunk_lengths = []
                batch_ar_info = []  # (rid, is_prefill, chunk_length)

                for ar in sched.active_requests:
                    rid = ar.request_id
                    if rid not in request_to_slot:
                        continue
                    sid = request_to_slot[rid]
                    state = active_requests[rid]

                    if ar.is_prefill:
                        tokens = full_token_seqs[rid]
                        chunk = tokens[ar.prefill_chunk_start:
                                       ar.prefill_chunk_start
                                       + ar.prefill_chunk_length]
                        batch_ar_info.append(
                            (rid, True, ar.prefill_chunk_length))
                        if ar.is_continuation:
                            cont_sids.append(sid)
                            cont_input_ids.append(chunk)
                            cont_offsets.append(ar.prefill_chunk_start)
                            cont_chunk_lengths.append(ar.prefill_chunk_length)
                        else:
                            prefill_sids.append(sid)
                            prefill_input_ids.append(chunk)
                            prefill_chunk_lengths.append(ar.prefill_chunk_length)
                    else:
                        batch_ar_info.append((rid, False, 0))
                        decode_sids.append(sid)
                        out_ids = per_conv_output_ids[rid]
                        di = state['decode_step']
                        if out_ids and di < len(out_ids):
                            decode_tokens_list.append(out_ids[di])
                        else:
                            decode_tokens_list.append(1)  # dummy fallback
                        state['decode_step'] = di + 1

                if not decode_sids and not prefill_sids and not cont_sids:
                    continue

                decode_tensor = torch.tensor(
                    decode_tokens_list if decode_tokens_list else [],
                    dtype=torch.long, device=engine.device)

                # 3. Ensure pages (dynamic page mode only)
                if getattr(engine, '_dynamic_pages', False):
                    for sid in decode_sids:
                        sl = int(engine._seq_lens_cpu[sid].item())
                        engine.ensure_pages(
                            sid, (sl + self.page_size) // self.page_size)
                    for sid, ids in zip(prefill_sids, prefill_input_ids):
                        engine.ensure_pages(
                            sid, math.ceil(len(ids) / self.page_size))
                    for sid, off, ids in zip(
                            cont_sids, cont_offsets, cont_input_ids):
                        engine.ensure_pages(
                            sid, math.ceil((off + len(ids)) / self.page_size))

                # 4. GPU forward
                logits = engine.step(
                    decode_seq_ids=decode_sids,
                    decode_token_ids=decode_tensor,
                    prefill_seq_ids=prefill_sids,
                    prefill_input_ids=prefill_input_ids,
                    continuation_seq_ids=cont_sids,
                    continuation_input_ids=cont_input_ids,
                    continuation_offsets=cont_offsets,
                )

                # 5. Token protocol (Pitfalls 19-20)
                if record_tokens:
                    replay_next = extract_next_tokens(
                        logits, len(decode_sids),
                        prefill_chunk_lengths, cont_chunk_lengths)

                    for tok_idx, (rid, is_pf, chunk_len) in enumerate(
                            batch_ar_info):
                        if tok_idx >= len(replay_next):
                            break
                        state = active_requests[rid]
                        if is_pf:
                            state['num_computed'] += chunk_len
                            eff_prompt = (prompt_lens[rid]
                                          + len(output_tokens[rid]))
                            if state['num_computed'] >= eff_prompt:
                                state['needs_prefill'] = False
                                output_tokens[rid].append(
                                    replay_next[tok_idx])
                                # Sync decode_step (Pitfall 20)
                                state['decode_step'] = len(
                                    output_tokens[rid]) - 1
                        else:
                            output_tokens[rid].append(
                                replay_next[tok_idx])

        # Cleanup
        engine.trace_recorder = _prev_recorder
        engine.replay_controller = _prev_controller
        engine.reset()

        return ReplayResult(
            output_tokens=output_tokens,
            trace_data=recorder.trace if recorder else None,
            steps=total_steps,
            preemptions=replay_preemptions,
            skipped_admissions=skipped_admissions,
        )


# ---------------------------------------------------------------------------
# Logit extraction
# ---------------------------------------------------------------------------

def extract_next_tokens(
    logits: torch.Tensor,
    n_decode: int,
    prefill_chunk_lengths: list[int],
    continuation_chunk_lengths: list[int],
) -> list[int]:
    """Extract per-sequence next token IDs from step logits.

    Logit layout from step (selective lm_head):
      [D decode rows | 1 row per prefill seq | 1 row per continuation seq]
    - Decode: argmax of logits[i] for each decode sequence
    - Prefill: one row per sequence (last-token logits)
    - Continuation: one row per sequence (last-token logits)

    Args:
        logits: [D + num_prefill_seqs + num_cont_seqs, vocab_size] tensor
        n_decode: number of decode sequences (rows 0..n_decode-1)
        prefill_chunk_lengths: length of each new-prefill chunk
        continuation_chunk_lengths: length of each continuation chunk

    Returns:
        List of next_token_ids in order:
        [decode_0, ..., prefill_0, ..., continuation_0, ...]
    """
    tokens: list[int] = []

    # Decode: one logit row per sequence
    if n_decode > 0:
        decode_logits = logits[:n_decode]
        tokens.extend(decode_logits.argmax(dim=-1).tolist())

    # Prefill: one row per sequence (already last-token logits)
    offset = n_decode
    for _ in prefill_chunk_lengths:
        tokens.append(logits[offset].argmax(dim=-1).item())
        offset += 1

    # Continuation: one row per sequence (already last-token logits)
    for _ in continuation_chunk_lengths:
        tokens.append(logits[offset].argmax(dim=-1).item())
        offset += 1

    assert offset == logits.shape[0], (
        f"Logit extraction consumed {offset} rows but tensor has "
        f"{logits.shape[0]} rows")

    return tokens


# ---------------------------------------------------------------------------
# GPU integration helpers
# ---------------------------------------------------------------------------

def _get_effective_prompt_gpu(
    req: ActiveState, prompts_gpu: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Effective prompt (prompt + output tokens so far) as a GPU tensor."""
    prompt = prompts_gpu[req.request_idx]
    if req.output_token_ids:
        output = torch.tensor(
            req.output_token_ids, dtype=torch.long, device=prompt.device)
        return torch.cat([prompt, output])
    return prompt


def _build_full_token_seqs(
    conversations: list[dict], device: torch.device,
) -> dict[int, torch.Tensor]:
    """Build prompt+output token tensors keyed by request_idx."""
    seqs = {}
    for conv in conversations:
        rid = conv['request_idx']
        ids = list(conv['prompt_token_ids']) + list(
            conv.get('output_token_ids', []))
        if ids:
            seqs[rid] = torch.tensor(ids, dtype=torch.long, device=device)
        else:
            seqs[rid] = torch.ones(
                len(conv['prompt_token_ids']), dtype=torch.long, device=device)
    return seqs


def load_full_tokens(traces, device: torch.device) -> dict[int, torch.Tensor]:
    """Pre-load full token sequences (prompt + output) keyed by trace index.

    During recompute after preemption, prefill chunks extend past the original
    prompt into previously-generated output tokens. Using just prompt_token_ids
    would produce out-of-bounds slices for recompute prefill steps.
    For non-preempted requests the extra tokens are never accessed.

    Args:
        traces: list of ConversationTrace objects (from trace_utils.load_traces).
        device: torch device to place tensors on.
    """
    full_tokens = {}
    for i, t in enumerate(traces):
        ids = list(t.prompt_token_ids or [])
        ids += list(t.output_token_ids or [])
        if ids:
            full_tokens[i] = torch.tensor(ids, dtype=torch.long, device=device)
        else:
            full_tokens[i] = torch.ones(t.prompt_tokens, dtype=torch.long,
                                        device=device)
    return full_tokens


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CollectionResult:
    """Result of Scheduler.collect()."""
    all_step_scheduling: list[dict]
    step_count: int
    conversations: list[dict]    # [{conversation_id, request_idx, prompt/output_token_ids, num_preemptions}]
    trace: list[dict]            # [{step, layer, expert_ids}]
    num_layers: int
    num_experts: int

    def to_dict(self) -> dict:
        """Raw dict for backward compat with collect_batched() callers."""
        return {
            'all_step_scheduling': self.all_step_scheduling,
            'step_count': self.step_count,
            'conversations': self.conversations,
            'trace': self.trace,
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
        }

    @staticmethod
    def from_dict(d: dict) -> 'CollectionResult':
        """Construct from collect_batched() raw dict."""
        return CollectionResult(
            all_step_scheduling=d['all_step_scheduling'],
            step_count=d['step_count'],
            conversations=d['conversations'],
            trace=d['trace'],
            num_layers=d['num_layers'],
            num_experts=d['num_experts'],
        )

    def to_activation_trace(self):
        """In-memory ActivationTrace (no disk I/O)."""
        from gpu_replay_trace import ActivationTrace
        return ActivationTrace.from_flat_trace({
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
            'trace': self.trace,
            'step_scheduling': self.all_step_scheduling,
        })

    def save(self, output_dir, scheduling_config=None, top_k=2):
        """Write batched_trace.json + per-conv JSONs + manifest."""
        d = self.to_dict()
        save_batched_trace(d, output_dir, scheduling_config or {})
        return save_conversations(d, output_dir, top_k)


@dataclass
class ReplayResult:
    """Result of Scheduler.replay()."""
    output_tokens: dict[int, list[int]]   # {request_idx: [token_ids]}
    trace_data: list[dict] | None         # from TraceRecorder, if record_routing
    steps: int
    preemptions: dict[int, int] = field(default_factory=dict)
    skipped_admissions: int = 0

    def compare_tokens(self, collected: CollectionResult | dict) -> tuple[bool, dict]:
        """Compare vs collection. Returns (all_match, {rid: first_diff_idx})."""
        if isinstance(collected, CollectionResult):
            convs = collected.conversations
        else:
            convs = collected['conversations']

        mismatches = {}
        for conv in convs:
            rid = conv['request_idx']
            coll = conv['output_token_ids']
            repl = self.output_tokens.get(rid, [])
            n = min(len(coll), len(repl))
            if n > 0 and coll[:n] != repl[:n]:
                first_diff = next(
                    j for j in range(n) if coll[j] != repl[j])
                mismatches[rid] = first_diff
        return (len(mismatches) == 0, mismatches)

    def compare_routing(self, collection_trace: list[dict]) -> tuple[bool, int]:
        """Compare expert routing. Returns (all_match, mismatch_count)."""
        if self.trace_data is None:
            raise ValueError("No trace_data — replay() was called without "
                             "record_routing=True")
        count = 0
        for rt, ct in zip(self.trace_data, collection_trace):
            if rt['expert_ids'] != ct['expert_ids']:
                count += 1
        return (count == 0, count)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def save_batched_trace(result: dict, output_dir: str,
                       scheduling_config: dict) -> str:
    """Write batched ActivationTrace JSON. Returns path.

    Format is identical to ActivationTrace.save() so ActivationTrace.load()
    and from_flat_trace() work without modification.
    """
    data = {
        'num_layers': result['num_layers'],
        'num_experts': result['num_experts'],
        'trace': result['trace'],
        'step_scheduling': result['all_step_scheduling'],
        'scheduling': scheduling_config,
        'transfers': [],   # expected by old readers, always empty here
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'batched_trace.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Batched trace: {result['step_count']} steps → {path}")
    return path


def save_conversations(result: dict, output_dir: str,
                       top_k: int) -> list:
    """Write per-conversation JSON files. Returns manifest entry list.

    Files are written to <output_dir>/requests/<conversation_id>.json
    matching the ConversationTrace.load() format from trace_utils.py.
    trace: [] because batch-union routing cannot be attributed per-conversation.
    """
    requests_dir = os.path.join(output_dir, 'requests')
    os.makedirs(requests_dir, exist_ok=True)
    manifest_entries = []
    for conv in result['conversations']:
        conv_data = {
            'conversation_id': conv['conversation_id'],
            'prompt_tokens': len(conv['prompt_token_ids']),
            'output_tokens': len(conv['output_token_ids']),
            'num_layers': result['num_layers'],
            'num_experts': result['num_experts'],
            'top_k': top_k,
            'trace': [],                          # no per-conv trace
            'prompt_token_ids': conv['prompt_token_ids'],
            'output_token_ids': conv['output_token_ids'],
            'num_preemptions': conv['num_preemptions'],
        }
        fname = f"requests/{conv['conversation_id']}.json"
        with open(os.path.join(output_dir, fname), 'w') as f:
            json.dump(conv_data, f)
        manifest_entries.append({
            'conversation_id': conv['conversation_id'],
            'prompt_tokens': conv_data['prompt_tokens'],
            'output_tokens': conv_data['output_tokens'],
            'trace_file': fname,
        })
    return manifest_entries


# Graph sizes for batched collection and replay — must match batched_replay.py.
GRAPH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192,
               224, 256, 288, 320, 352, 384, 448, 512]
