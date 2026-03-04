"""Replay Controller for MoE expert offloading (Phase 6).

Replays a pre-computed DataMovementTrace on real GPU hardware. Drives the
same _mixed_step_piecewise loop as ExpertOffloadEngine but uses trace-
specified prefetch/eviction/demand-load schedules instead of reactive
demand loading.

Data flow:
    1. Record activation trace:   ExpertOffloadEngine.save_trace()
    2. Simulate policy:           LRUPolicy.simulate() -> DataMovementTrace
    3. Replay on hardware:        ReplayController(engine, trace) -> timing

Usage:
    from data_movement_trace import DataMovementTrace
    from replay_controller import ReplayController

    trace = DataMovementTrace.load("lru_movement.json")
    controller = ReplayController(engine, trace)
    controller.setup()

    # Set engine.replay_controller = controller, then run inference:
    engine.replay_controller = controller
    for step in range(len(trace.steps)):
        logits = engine.mixed_step(...)

    timing = controller.get_timing()
"""

import torch

from data_movement_trace import DataMovementTrace, TransferEvent


class ReplayController:
    """Replays a DataMovementTrace with async transfers on real GPU hardware.

    Replaces ExpertOffloadEngine during replay mode. The MoEEngine's
    _mixed_step_piecewise dispatches to this controller when
    engine.replay_controller is set.

    Interface matches ExpertOffloadEngine's expected calls:
        begin_step()            — called at start of each decode step
        begin_layer_prefetch()  — called before stage1 (pre-attention async)
        process_layer_replay()  — called after stage4a (post-routing + sync)
        post_layer()            — called after stage4b
    """

    def __init__(self, engine, trace: DataMovementTrace):
        """Create replay controller.

        Args:
            engine: MoEEngine instance with experts_per_layer set.
            trace: Pre-computed data movement trace to replay.
        """
        self.trace = trace
        self.device = engine.device

        # GPU buffers (shared with engine)
        self.w1_buf = engine.w1_buf
        self.w2_buf = engine.w2_buf
        self.w1_cpu = engine.w1_cpu
        self.w2_cpu = engine.w2_cpu
        self.expert_map = engine.expert_map
        self.expert_map_abs = engine.expert_map_abs
        self.expert_map_buf = engine.expert_map_buf
        self.scratchpad_start = engine.scratchpad_start
        self.scratchpad_slots = engine.scratchpad_slots
        self.experts_per_layer = engine.experts_per_layer
        self.num_layers = engine.num_layers
        self.num_experts = engine.num_experts

        # Async transfer infrastructure
        self._transfer_stream = torch.cuda.Stream(device=self.device)
        self._prefetch_done_event = torch.cuda.Event()
        self._post_routing_done_event = torch.cuda.Event()

        # CUDA events for transfer timing
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

        # Cache state: which experts are in which slots
        # _resident[layer] = {expert_id: absolute_slot}
        self._resident = [{} for _ in range(self.num_layers)]

        # Step/layer tracking
        self._current_step = -1
        self._next_scratchpad_slot = 0
        self._has_prefetches = False
        self._has_post_routing = False

        # Per-layer scratchpad assignments for expert_map_buf update
        self._scratchpad_assignments = {}  # eid -> slot

        # Timing records
        self.step_timings = []  # list of per-step timing dicts

    def setup(self):
        """Load initial cache state from trace into unified buffer.

        Iterates through trace.initial_cache_state, loads specified experts
        into their layer's persistent cache slots, and updates expert_map
        and expert_map_abs.
        """
        # Clear all expert maps
        for l in range(self.num_layers):
            self.expert_map[l].fill_(-1)
            self.expert_map_abs[l].fill_(-1)
            self._resident[l].clear()

        # Group initial state by layer
        by_layer = [[] for _ in range(self.num_layers)]
        for (layer, expert_id) in self.trace.initial_cache_state:
            by_layer[layer].append(expert_id)

        # Load into persistent cache slots
        for l in range(self.num_layers):
            base = l * self.experts_per_layer
            for slot_offset, eid in enumerate(by_layer[l]):
                if slot_offset >= self.experts_per_layer:
                    raise ValueError(
                        f"Layer {l}: initial state has "
                        f"{len(by_layer[l])} experts but capacity is "
                        f"{self.experts_per_layer}")
                abs_slot = base + slot_offset
                self.w1_buf[abs_slot].copy_(self.w1_cpu[l][eid])
                self.w2_buf[abs_slot].copy_(self.w2_cpu[l][eid])
                self.expert_map[l][eid] = slot_offset
                self.expert_map_abs[l][eid] = abs_slot
                self._resident[l][eid] = abs_slot

        torch.cuda.synchronize()
        total_loaded = sum(len(v) for v in by_layer)
        print(f"ReplayController: loaded {total_loaded} experts "
              f"({len(self.trace.steps)} steps to replay)")

    def begin_step(self):
        """Called at start of each decode step."""
        self._current_step += 1

    def begin_layer_prefetch(self, layer):
        """Issue pre-attention prefetches (async on transfer stream).

        Called BEFORE stages 1–3. These transfers overlap with
        attention compute on the default stream.
        """
        step_trace = self.trace.steps[self._current_step]
        layer_trace = step_trace.layers[layer]

        # Reset per-layer state
        self._next_scratchpad_slot = 0
        self._scratchpad_assignments.clear()
        self._has_prefetches = bool(layer_trace.prefetches)

        if not layer_trace.prefetches:
            return

        with torch.cuda.stream(self._transfer_stream):
            for event in layer_trace.prefetches:
                self._execute_transfer(event)
            self._prefetch_done_event.record()

    def process_layer_replay(self, layer, topk_ids_buf, n_tokens):
        """Called after stage4a (routing known).

        1. Copy expert_map_abs[layer] → expert_map_buf
        2. Issue post_routing_prefetches (async)
        3. Wait for all async transfers (pre-attn + post-routing)
        4. Execute demand_loads (blocking on compute stream)
        5. Update expert_map_buf with all slot assignments
        """
        step_trace = self.trace.steps[self._current_step]
        layer_trace = step_trace.layers[layer]

        # Step 1: set base expert map
        self.expert_map_buf.copy_(self.expert_map_abs[layer])

        # Step 2: post-routing prefetches (async)
        self._has_post_routing = bool(layer_trace.post_routing_prefetches)
        if layer_trace.post_routing_prefetches:
            with torch.cuda.stream(self._transfer_stream):
                for event in layer_trace.post_routing_prefetches:
                    self._execute_transfer(event)
                self._post_routing_done_event.record()

        # Step 3: wait for all async transfers
        if self._has_prefetches:
            torch.cuda.current_stream().wait_event(
                self._prefetch_done_event)
        if self._has_post_routing:
            torch.cuda.current_stream().wait_event(
                self._post_routing_done_event)

        # Step 4: demand loads (blocking, on compute stream)
        for event in layer_trace.demand_loads:
            self._execute_transfer(event)

        # Step 5: re-copy expert_map_abs (may have been updated by
        # persistent replacements in _execute_transfer)
        self.expert_map_buf.copy_(self.expert_map_abs[layer])

        # Overlay scratchpad assignments
        for eid, slot in self._scratchpad_assignments.items():
            self.expert_map_buf[eid] = slot

    def post_layer(self, layer):
        """Called after stage4b. Scratchpad is ephemeral."""
        self._scratchpad_assignments.clear()

    def _execute_transfer(self, event: TransferEvent):
        """Execute a single transfer event.

        If event.evict is set, this is a persistent cache replacement:
        free the evicted expert's slot and load the new expert there.
        If event.evict is None, this is a scratchpad load (ephemeral).
        """
        target_layer, target_eid = event.target

        if event.evict is not None:
            # ── Persistent cache replacement ──
            evict_layer, evict_eid = event.evict

            # Get evicted expert's absolute slot
            slot = self._resident[evict_layer].pop(evict_eid)

            # Clear evicted expert from maps
            self.expert_map[evict_layer][evict_eid] = -1
            self.expert_map_abs[evict_layer][evict_eid] = -1

            # Load new expert into freed slot
            self.w1_buf[slot].copy_(self.w1_cpu[target_layer][target_eid],
                                     non_blocking=True)
            self.w2_buf[slot].copy_(self.w2_cpu[target_layer][target_eid],
                                     non_blocking=True)

            # Update persistent state
            base = target_layer * self.experts_per_layer
            self._resident[target_layer][target_eid] = slot
            self.expert_map[target_layer][target_eid] = slot - base
            self.expert_map_abs[target_layer][target_eid] = slot
        else:
            # ── Scratchpad load (ephemeral) ──
            slot = self.scratchpad_start + self._next_scratchpad_slot
            self._next_scratchpad_slot += 1

            self.w1_buf[slot].copy_(self.w1_cpu[target_layer][target_eid],
                                     non_blocking=True)
            self.w2_buf[slot].copy_(self.w2_cpu[target_layer][target_eid],
                                     non_blocking=True)

            # Track for expert_map_buf update (not persistent)
            self._scratchpad_assignments[target_eid] = slot

    def get_replay_stats(self) -> dict:
        """Compute summary statistics from the trace being replayed."""
        return self.trace.summary()
