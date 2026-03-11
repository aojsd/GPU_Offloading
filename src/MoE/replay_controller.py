"""Replay Controller for MoE expert offloading (Phase 6).

Replays a pre-computed GPUReplayTrace on real GPU hardware using a
unified expert cache shared across all layers. Drives the same
_mixed_step_piecewise loop as ExpertOffloadEngine but uses trace-specified
prefetch/eviction/demand-load schedules instead of reactive demand loading.

Prefetch timing:
    Layer 0: prefetches issued before stage1 (begin_layer_prefetch).
    Layer L>0: prefetches issued before stage4b of layer L-1 (inside
        process_layer_replay of L-1), overlapping with stage4b(L-1) +
        stages 1-4a(L).

Data flow:
    1. Record activation trace:   ExpertOffloadEngine.save_trace()
    2. Simulate policy:           simulate(cache, prefetch, trace, cache_size) -> GPUReplayTrace
    3. Replay on hardware:        ReplayController(engine, trace) -> timing

Usage:
    from gpu_replay_trace import GPUReplayTrace
    from replay_controller import ReplayController

    trace = GPUReplayTrace.load("lru_movement.json")
    controller = ReplayController(engine, trace)
    controller.setup()

    engine.replay_controller = controller
    for step in range(len(trace.steps)):
        logits = engine.mixed_step(...)

    timing = controller.get_replay_stats()
"""

import torch

from gpu_replay_trace import GPUReplayTrace, StepScheduling, TransferEvent


class PhaseTimer:
    """CUDA event-based per-phase timing for replay steps.

    Records GPU timestamps at phase boundaries with near-zero overhead
    (each record() enqueues a single stream command). Phase times are
    computed post-run from event pairs after torch.cuda.synchronize().

    Per-layer phases (summed across all layers and steps):
        stage1:    Pre-attention (RMSNorm + RoPE + QKV projection)
        attention: Decode + prefill attention (stages 2+3)
        stage4a:   Post-attention RMSNorm + router
        io:        CPU break (prefetch sync + demand loads + map update
                   + next-layer prefetch dispatch)
        stage4b:   Fused MoE computation

    Per-step phases:
        step_setup:  Token/position/slot setup + FlashInfer plan + embed
        step_finish: Final RMSNorm + lm_head
    """

    PHASES = ('step_setup', 'stage1', 'attention', 'stage4a',
              'io', 'stage4b', 'step_finish')

    def __init__(self, num_steps, num_layers):
        self._num_steps = num_steps
        self._num_layers = num_layers
        self._step = -1
        # Events per step: step_start, after_setup,
        #   per layer: after_stage1, after_attn, after_stage4a,
        #              after_io, after_stage4b,
        #   after_final
        self._eps = 2 + num_layers * 5 + 1
        total = self._eps * num_steps
        self._events = [torch.cuda.Event(enable_timing=True)
                        for _ in range(total)]

    # ── Recording (called by engine during replay) ──

    def step_start(self):
        self._step += 1
        self._events[self._step * self._eps].record()

    def after_setup(self):
        self._events[self._step * self._eps + 1].record()

    def after_stage1(self, layer):
        self._events[self._step * self._eps + 2 + layer * 5].record()

    def after_attn(self, layer):
        self._events[self._step * self._eps + 2 + layer * 5 + 1].record()

    def after_stage4a(self, layer):
        self._events[self._step * self._eps + 2 + layer * 5 + 2].record()

    def after_io(self, layer):
        self._events[self._step * self._eps + 2 + layer * 5 + 3].record()

    def after_stage4b(self, layer):
        self._events[self._step * self._eps + 2 + layer * 5 + 4].record()

    def after_final(self):
        self._events[self._step * self._eps + self._eps - 1].record()

    # ── Aggregation (call after torch.cuda.synchronize()) ──

    def get_phase_times_ms(self) -> dict[str, float]:
        """Compute per-phase aggregate times in milliseconds."""
        totals = {p: 0.0 for p in self.PHASES}
        L = self._num_layers
        n = self._step + 1
        e = self._events
        eps = self._eps
        for s in range(n):
            base = s * eps
            totals['step_setup'] += e[base].elapsed_time(e[base + 1])
            for l in range(L):
                lb = base + 2 + l * 5
                before_s1 = e[base + 1] if l == 0 else e[lb - 1]
                totals['stage1'] += before_s1.elapsed_time(e[lb])
                totals['attention'] += e[lb].elapsed_time(e[lb + 1])
                totals['stage4a'] += e[lb + 1].elapsed_time(e[lb + 2])
                totals['io'] += e[lb + 2].elapsed_time(e[lb + 3])
                totals['stage4b'] += e[lb + 3].elapsed_time(e[lb + 4])
            last_s4b = e[base + 2 + (L - 1) * 5 + 4]
            totals['step_finish'] += last_s4b.elapsed_time(
                e[base + eps - 1])
        return totals


class ReplayController:
    """Replays a GPUReplayTrace with async transfers on real GPU hardware.

    Uses a unified expert cache: any slot can hold any (layer, expert_id).
    No per-layer partitioning or scratchpad.

    Replaces ExpertOffloadEngine during replay mode. The MoEEngine's
    _mixed_step_piecewise dispatches to this controller when
    engine.replay_controller is set.

    Interface:
        begin_step()            — called at start of each decode step
        begin_layer_prefetch()  — called before stage1, ONLY for layer 0
        process_layer_replay()  — called after stage4a (sync + demand loads
                                  + issue next-layer prefetches)
        post_layer()            — called after stage4b (no-op for unified)
    """

    def __init__(self, engine, trace: GPUReplayTrace, track_io=False,
                 track_phases=False):
        """Create replay controller.

        Args:
            engine: MoEEngine instance with cache_size set.
            trace: Pre-computed data movement trace to replay.
            track_io: Record CUDA events around demand loads to measure
                I/O vs compute time. Most meaningful for no-prefetch runs
                where all I/O is blocking on the compute stream.
            track_phases: Record CUDA events at every phase boundary
                (stage1, attention, stage4a, io, stage4b, step_setup,
                step_finish). Superset of track_io. The engine records
                events via self._phase_timer in _mixed_step_piecewise.
        """
        self.trace = trace
        self.device = engine.device

        # GPU buffers (shared with engine)
        self.w1_buf = engine.w1_buf
        self.w2_buf = engine.w2_buf
        self.w1_cpu = engine.w1_cpu
        self.w2_cpu = engine.w2_cpu
        self.expert_map_abs = engine.expert_map_abs
        self.expert_map_buf = engine.expert_map_buf
        self.cache_size = engine.cache_size
        self.num_layers = engine.num_layers
        self.num_experts = engine.num_experts

        # Async transfer infrastructure
        self._transfer_stream = torch.cuda.Stream(device=self.device)
        self._prefetch_done_event = torch.cuda.Event()

        # Unified cache state: {(layer, eid): absolute_slot}
        self._resident = {}

        # Free slot tracking
        self._free_slots = set()

        # Step/layer tracking
        self._current_step = -1
        self._has_prefetches = False

        # I/O timing: CUDA event pairs around demand load blocks
        self._track_io = track_io
        self._demand_event_pairs = []  # [(start_evt, end_evt), ...]

        # Per-phase timing (superset of track_io)
        self._phase_timer = None
        if track_phases:
            self._phase_timer = PhaseTimer(
                len(trace.steps), engine.num_layers)

    def setup(self):
        """Load initial cache state from trace into unified buffer.

        Assigns slots sequentially from 0. Updates expert_map_abs for
        each layer so stage4b CUDA graphs can find the experts.
        """
        # Clear all expert maps
        for l in range(self.num_layers):
            self.expert_map_abs[l].fill_(-1)
        self._resident.clear()
        self._free_slots = set(range(self.cache_size))

        # Load initial state
        for i, (layer, eid) in enumerate(self.trace.initial_cache_state):
            slot = i  # Sequential slot assignment
            self._free_slots.discard(slot)
            self.w1_buf[slot].copy_(self.w1_cpu[layer][eid])
            self.w2_buf[slot].copy_(self.w2_cpu[layer][eid])
            self.expert_map_abs[layer][eid] = slot
            self._resident[(layer, eid)] = slot

        torch.cuda.synchronize()
        total_loaded = len(self.trace.initial_cache_state)
        print(f"ReplayController: loaded {total_loaded} experts into "
              f"unified cache ({self.cache_size} slots, "
              f"{len(self.trace.steps)} steps to replay)")

    def begin_step(self):
        """Called at start of each decode step."""
        self._current_step += 1

    def begin_layer_prefetch(self, layer):
        """Issue pre-attention prefetches. Only called for layer 0.

        These transfers run on the transfer stream and overlap with the
        full compute pipeline (stages 1-4a) of layer 0.
        """
        assert layer == 0, (
            f"begin_layer_prefetch only valid for layer 0, got {layer}")

        step_trace = self.trace.steps[self._current_step]
        layer_trace = step_trace.layers[0]

        self._has_prefetches = bool(layer_trace.prefetches)
        if not layer_trace.prefetches:
            return

        with torch.cuda.stream(self._transfer_stream):
            for event in layer_trace.prefetches:
                self._execute_transfer(event)
            self._prefetch_done_event.record()

    def process_layer_replay(self, layer, topk_ids_buf, n_tokens):
        """Called after stage4a (routing known).

        1. Wait for async prefetches targeting this layer
        2. Copy expert_map_abs[layer] → expert_map_buf
        3. Execute demand_loads (blocking on compute stream)
        4. Re-copy expert_map_abs → expert_map_buf (updated by transfers)
        5. Issue prefetches for layer+1 (async, overlaps with stage4b)
        """
        step_trace = self.trace.steps[self._current_step]
        layer_trace = step_trace.layers[layer]

        # Step 1: sync prefetches targeting this layer (must complete before
        # we read expert_map_abs, which prefetches write to on the transfer
        # stream via _execute_transfer)
        if self._has_prefetches:
            torch.cuda.current_stream().wait_event(
                self._prefetch_done_event)

        # Step 2: set base expert map for this layer (now race-free —
        # all prefetch writes to expert_map_abs are visible after sync)
        self.expert_map_buf.copy_(self.expert_map_abs[layer])

        # Step 3: demand loads (blocking, on compute stream)
        if layer_trace.demand_loads:
            if self._track_io:
                io_start = torch.cuda.Event(enable_timing=True)
                io_end = torch.cuda.Event(enable_timing=True)
                io_start.record()
            for event in layer_trace.demand_loads:
                self._execute_transfer(event)
            if self._track_io:
                io_end.record()
                self._demand_event_pairs.append((io_start, io_end))

        # Step 4: re-copy expert_map (may have been updated by transfers)
        self.expert_map_buf.copy_(self.expert_map_abs[layer])

        # Step 5: issue prefetches for next layer (async, before stage4b)
        # These overlap with stage4b of THIS layer + stages 1-4a of NEXT
        self._has_prefetches = False
        if layer + 1 < self.num_layers:
            next_layer_trace = step_trace.layers[layer + 1]
            if next_layer_trace.prefetches:
                self._has_prefetches = True
                with torch.cuda.stream(self._transfer_stream):
                    for event in next_layer_trace.prefetches:
                        self._execute_transfer(event)
                    self._prefetch_done_event.record()

    def post_layer(self, layer):
        """Called after stage4b. No-op for unified cache (no scratchpad)."""
        pass

    def _execute_transfer(self, event: TransferEvent):
        """Execute a single transfer event on the unified cache.

        If event.evict is set: free the evicted expert's slot, load the
        new expert there.
        If event.evict is None: take a slot from the free pool.
        """
        target_layer, target_eid = event.target

        if event.evict is not None:
            # Evict: free the slot and reuse it
            evict_key = tuple(event.evict)
            slot = self._resident.pop(evict_key)
            evict_layer, evict_eid = evict_key
            self.expert_map_abs[evict_layer][evict_eid] = -1
        else:
            # Free slot available
            slot = self._free_slots.pop()

        # Load new expert into slot
        self.w1_buf[slot].copy_(self.w1_cpu[target_layer][target_eid],
                                 non_blocking=True)
        self.w2_buf[slot].copy_(self.w2_cpu[target_layer][target_eid],
                                 non_blocking=True)

        # Update unified cache state
        self._resident[(target_layer, target_eid)] = slot
        self.expert_map_abs[target_layer][target_eid] = slot

    def get_step_scheduling(self, step: int = None) -> 'StepScheduling | None':
        """Get scheduling metadata for a step.

        Args:
            step: Step index. If None, uses the current step.

        Returns:
            StepScheduling or None if no scheduling data is available.
        """
        if step is None:
            step = self._current_step
        if step < 0 or step >= len(self.trace.steps):
            return None
        return self.trace.steps[step].scheduling

    def get_newly_admitted_seq_ids(self, step: int = None) -> list[int]:
        """Get request_ids newly admitted at this step (need initial prefill).

        Returns:
            List of request_ids admitted, or empty list.
        """
        sched = self.get_step_scheduling(step)
        if sched is None:
            return []
        return [evt['request_id'] for evt in sched.events
                if evt['event'] in ('admit', 'force_admit')]

    def get_demand_load_time_ms(self) -> float:
        """Total demand load time in ms. Call after torch.cuda.synchronize().

        Only meaningful when track_io=True. For no-prefetch runs, this is
        the total blocking I/O time on the compute stream, so:
            compute_time = total_time - demand_load_time
        """
        total = 0.0
        for start, end in self._demand_event_pairs:
            total += start.elapsed_time(end)
        return total

    def get_phase_times_ms(self) -> dict[str, float]:
        """Per-phase aggregate times in ms. Call after synchronize().

        Requires track_phases=True. Returns dict with keys:
            step_setup, stage1, attention, stage4a, io, stage4b, step_finish
        """
        if self._phase_timer is None:
            raise RuntimeError("track_phases=True required")
        return self._phase_timer.get_phase_times_ms()

    def get_replay_stats(self) -> dict:
        """Compute summary statistics from the trace being replayed."""
        return self.trace.summary()
