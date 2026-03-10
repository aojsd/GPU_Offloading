"""
Merged Phase 1+2: GPU-based continuous batching with expert trace collection.

Implements vLLM V1-style scheduling (greedy admission, LIFO preemption with
recompute) and records expert activations + scheduling metadata simultaneously.

This module contains:
- ActiveState: per-request mutable state
- MockPageAllocator: CPU-side page pool for testing (mirrors MoEEngine interface)
- BatchScheduler: 6-phase vLLM V1 scheduling state machine
- ScheduleResult: typed return from scheduler
- extract_next_tokens: logit extraction from mixed_step [D|P|C] layout
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

# Reuse from existing modules
from trace_construction.build_trace import pages_needed


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

class BatchScheduler:
    """vLLM V1-style continuous batching scheduler.

    Implements greedy admission, LIFO preemption with recompute, and
    no-admission-on-preemption gating. The scheduler operates on a
    PageAllocator (MockPageAllocator for tests, MoEEngine for production).

    Usage:
        scheduler = BatchScheduler(allocator, max_seqs=256, max_graph_size=512)
        scheduler.add_requests([...])
        while not scheduler.is_done:
            result = scheduler.step()
            # caller runs engine.mixed_step() using result
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
    def is_done(self) -> bool:
        return len(self.running) == 0 and len(self.waiting) == 0

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
        """Run one scheduling step. Returns execution groups for mixed_step.

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


# ---------------------------------------------------------------------------
# Logit extraction
# ---------------------------------------------------------------------------

def extract_next_tokens(
    logits: torch.Tensor,
    n_decode: int,
    prefill_chunk_lengths: list[int],
    continuation_chunk_lengths: list[int],
) -> list[int]:
    """Extract per-sequence next token IDs from mixed_step logits.

    Logit layout from mixed_step: [D decode rows | P prefill rows | C continuation rows]
    - Decode: argmax of logits[i] for each decode sequence
    - Prefill: argmax of LAST row in each chunk
    - Continuation: argmax of LAST row in each chunk

    Args:
        logits: [N_total, vocab_size] tensor
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

    # Prefill: last row of each chunk
    offset = n_decode
    for chunk_len in prefill_chunk_lengths:
        last_row = logits[offset + chunk_len - 1]
        tokens.append(last_row.argmax(dim=-1).item())
        offset += chunk_len

    # Continuation: last row of each chunk
    for chunk_len in continuation_chunk_lengths:
        last_row = logits[offset + chunk_len - 1]
        tokens.append(last_row.argmax(dim=-1).item())
        offset += chunk_len

    assert offset == logits.shape[0], (
        f"Logit extraction consumed {offset} rows but tensor has "
        f"{logits.shape[0]} rows")

    return tokens


# ---------------------------------------------------------------------------
# GPU integration
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


def collect_batched(
    engine,
    conversations: list[dict],
    max_seqs: int,
    max_graph_size: int,
    page_size: int = 16,
) -> dict:
    """Run batched trace collection with continuous batching on GPU.

    Args:
        engine: MoEEngine with kv_page_budget set and dynamic pages enabled.
        conversations: List of dicts, each with keys:
            conversation_id, prompt_token_ids, max_output_tokens.
        max_seqs: Maximum concurrent sequences (must match engine.max_seqs).
        max_graph_size: Maximum total tokens per step.
        page_size: KV cache page size in tokens.

    Returns:
        Dict with 'all_step_scheduling', 'step_count', 'conversations'.
    """
    assert max_seqs == engine.max_seqs, (
        f"Scheduler/engine max_seqs mismatch: {max_seqs} vs {engine.max_seqs}")

    scheduler = BatchScheduler(engine, max_seqs, max_graph_size, page_size)

    # Attach trace recorder — save/restore so tests that set engine.trace_recorder
    # externally are not affected by collect_batched().
    from trace_recorder import TraceRecorder  # lazy: avoids top-level path dep
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

    scheduler.add_requests(requests)

    with torch.inference_mode():
        while not scheduler.is_done:
            result = scheduler.step()
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
                slot = scheduler.request_to_slot[req.request_idx]
                continuation_offsets.append(
                    engine._seq_lens_cpu[slot].item())

            # 4. GPU forward pass
            logits = engine.mixed_step(
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
            # is_complete=True is visible in Phase 1 of the NEXT scheduler step.
            eos_id = getattr(engine, 'eos_token_id', None)
            if eos_id is not None:
                batch_order = (result.decode_requests + result.prefill_requests
                               + result.continuation_requests)
                for _i, _req in enumerate(batch_order):
                    if _i < len(next_tokens) and next_tokens[_i] == eos_id:
                        _req.hit_eos = True

            # 6. Advance state
            scheduler.advance_state(result, next_tokens)

    engine.trace_recorder = _prev_recorder

    # Sanity: recorder and scheduler step counters must agree.
    # Both count only non-empty steps (empty steps skip mixed_step).
    assert (recorder._step == scheduler.step_count - 1
            or (recorder._step == -1 and scheduler.step_count == 0)), (
        f"Recorder step {recorder._step} != scheduler step "
        f"{scheduler.step_count - 1}")

    return {
        'all_step_scheduling': scheduler.all_step_scheduling,
        'step_count': scheduler.step_count,
        'conversations': [
            {'conversation_id': r.conversation_id,
             'request_idx': r.request_idx,
             'prompt_token_ids': r.prompt_token_ids,
             'output_token_ids': r.output_token_ids,
             'num_preemptions': r.num_preemptions}
            for r in requests
        ],
        'trace': recorder.trace,
        'num_layers': engine.num_layers,
        'num_experts': engine.num_experts,
    }


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
    matching the ConversationTrace.load() format from build_trace.py.
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Graph sizes for batched collection and replay — must match batched_replay.py.
# Using the same sizes ensures every batch composition seen during collection
# can also be replayed exactly.
GRAPH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192,
               224, 256, 288, 320, 352, 384, 448, 512]


def main():
    """Run GPU-based batched trace collection.

    Example:
        python trace_construction/collect_batched_traces.py \\
            --model models/Mixtral-8x7B-Instruct-v0.1 \\
            --dataset datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json \\
            --num-conversations 200 \\
            --max-output-tokens 4096 \\
            --max-seqs 32 \\
            --pp 2 \\
            --output-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b-batched/
    """
    # Path setup for script mode: add src/MoE to path so moe_engine etc. resolve.
    _script_dir = Path(__file__).resolve().parent
    _moe_dir = _script_dir.parent
    if str(_moe_dir) not in sys.path:
        sys.path.insert(0, str(_moe_dir))
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

    parser = argparse.ArgumentParser(
        description="GPU-based batched trace collection with continuous batching")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to ShareGPT JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trace files")
    parser.add_argument("--num-conversations", type=int, default=None,
                        help="Number of conversations (default: all)")
    parser.add_argument("--max-output-tokens", type=int, default=4096,
                        help="Per-request output token limit (default: 4096)")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="Skip prompts longer than this (default: max_seq_len)")
    parser.add_argument("--max-seqs", type=int, default=32,
                        help="Max concurrent sequences (default: 32)")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline parallel size (default: 1)")
    parser.add_argument("--kv-page-budget", type=int, default=None,
                        help="Explicit KV page count (default: computed from GPU memory)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip if output files already exist")
    args = parser.parse_args()

    # Apply glibc 2.28 patches before any vLLM import
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import moe_engine as _moe_engine_mod  # noqa: F401
    from moe_engine import MoEEngine
    import torch
    from transformers import AutoTokenizer
    from collect_traces import load_sharegpt

    page_size = 16

    # Read model config
    with open(Path(args.model) / "config.json") as f:
        cfg = json.load(f)
    num_experts = cfg.get("num_experts") or cfg.get("num_local_experts")
    num_layers = cfg["num_hidden_layers"]
    top_k = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_topk")
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["intermediate_size"]
    model_name = Path(args.model).name
    pp_size = args.pp

    print(f"Model: {model_name} ({num_layers}L, {num_experts}E, top-{top_k})")

    # Compute kv_page_budget from available GPU memory if not specified
    if args.kv_page_budget:
        kv_page_budget = args.kv_page_budget
    else:
        dtype_bytes = 2  # BF16
        kv_bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes
        per_layer_non_expert = (
            hidden_size * (hidden_size + 2 * num_kv_heads * head_dim + hidden_size)
            + hidden_size * 2
            + num_experts * hidden_size
        ) * dtype_bytes
        per_expert_bytes = (
            2 * intermediate_size * hidden_size
            + hidden_size * intermediate_size
        ) * dtype_bytes
        all_expert_bytes = num_layers * num_experts * per_expert_bytes
        global_non_expert = (
            num_layers * per_layer_non_expert
            + cfg["vocab_size"] * hidden_size * dtype_bytes * 2
        )
        total_weight_bytes = global_non_expert + all_expert_bytes
        weight_per_gpu = total_weight_bytes / max(pp_size, 1)
        graph_overhead = len(GRAPH_SIZES) * 200 * 1024 ** 2
        fixed_overhead = 3 * 1024 ** 3  # 3 GB for CUDA context + activations
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        available_for_kv = (
            gpu_total - weight_per_gpu - graph_overhead - fixed_overhead)
        if available_for_kv <= 0:
            raise RuntimeError(
                f"Insufficient GPU memory for KV cache. "
                f"Estimated weights: {weight_per_gpu / 1e9:.1f} GB, "
                f"graphs: {graph_overhead / 1e9:.1f} GB, "
                f"total GPU: {gpu_total / 1e9:.1f} GB")
        layers_per_gpu = num_layers // max(pp_size, 1)
        kv_bytes_per_token = kv_bytes_per_token_per_layer * layers_per_gpu
        max_kv_tokens = int(available_for_kv / kv_bytes_per_token)
        kv_page_budget = max(max_kv_tokens // page_size, args.max_seqs)

    max_seq_len = (kv_page_budget // args.max_seqs) * page_size
    print(f"KV budget: {kv_page_budget} pages = {kv_page_budget * page_size} tokens "
          f"({max_seq_len} per seq at max_seqs={args.max_seqs})")

    max_prompt_tokens = args.max_prompt_tokens or max_seq_len

    # Resume: check if output already exists
    batched_trace_path = os.path.join(args.output_dir, 'batched_trace.json')
    if args.resume and os.path.exists(batched_trace_path):
        print(f"Resume: {batched_trace_path} already exists, skipping collection.")
        return

    # Load tokenizer and dataset
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Loading dataset: {args.dataset}")
    raw_convs = load_sharegpt(args.dataset, args.num_conversations)
    print(f"Loaded {len(raw_convs)} raw conversations")

    # Tokenize with chat template
    conversations_input = []
    skipped = 0
    for conv in raw_convs:
        messages = [{"role": "user", "content": conv["text"]}]
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if not tokens or len(tokens) > max_prompt_tokens:
            skipped += 1
            continue
        conversations_input.append({
            'conversation_id': conv['id'],
            'prompt_token_ids': tokens,
            'max_output_tokens': args.max_output_tokens,
        })
    print(f"Tokenized: {len(conversations_input)} conversations "
          f"(skipped {skipped} for length)")

    if not conversations_input:
        print("No conversations to process. Exiting.")
        return

    # Create engine
    print(f"Loading model (pp={pp_size}, max_seqs={args.max_seqs}, "
          f"kv_page_budget={kv_page_budget})...")
    t0 = time.time()
    engine = MoEEngine(
        args.model,
        max_seqs=args.max_seqs,
        max_seq_len=max_seq_len,
        pipeline_parallel_size=pp_size,
        kv_page_budget=kv_page_budget,
        use_torch_compile=True,
    )
    print(f"Engine created in {time.time() - t0:.1f}s")

    # Capture CUDA graphs — stop on OOM
    print(f"Capturing CUDA graphs for {len(GRAPH_SIZES)} sizes...")
    captured = []
    for gs in GRAPH_SIZES:
        try:
            engine.capture_mixed_cuda_graphs(total_token_sizes=[gs])
            captured.append(gs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM at N={gs} — stopping ({len(captured)} sizes captured)")
            break
    if not captured:
        raise RuntimeError("Failed to capture any CUDA graphs")
    print(f"Captured {len(captured)} graph sizes (max={max(captured)})")

    # Run batched collection
    print(f"\nRunning batched collection: "
          f"{len(conversations_input)} conversations, "
          f"max_seqs={args.max_seqs}, max_graph_size={max(captured)}")
    t0 = time.time()
    result = collect_batched(
        engine, conversations_input,
        max_seqs=args.max_seqs,
        max_graph_size=max(captured),
        page_size=page_size,
    )
    elapsed = time.time() - t0
    print(f"Collection done: {result['step_count']} steps in {elapsed:.1f}s "
          f"({result['step_count'] / max(elapsed, 0.001):.1f} steps/s)")

    total_preemptions = sum(c['num_preemptions'] for c in result['conversations'])
    print(f"Total preemptions: {total_preemptions}")

    # Serialize outputs
    scheduling_config = {
        'kv_page_budget': kv_page_budget,
        'page_size': page_size,
        'max_seqs': args.max_seqs,
        'max_graph_size': max(captured),
        'prefill_chunk_size': 256,
        'num_conversations': len(conversations_input),
        'preemption_policy': 'lifo_recompute',
        'page_allocation': 'dynamic',
        'model': model_name,
    }
    save_batched_trace(result, args.output_dir, scheduling_config)
    manifest_entries = save_conversations(result, args.output_dir, top_k)

    manifest_data = {
        'model': model_name,
        'num_layers': result['num_layers'],
        'num_experts': result['num_experts'],
        'top_k': top_k,
        'total_conversations': len(manifest_entries),
        'step_count': result['step_count'],
        'scheduling': scheduling_config,
        'conversations': manifest_entries,
    }
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Manifest: {len(manifest_entries)} conversations → {manifest_path}")


if __name__ == "__main__":
    main()
