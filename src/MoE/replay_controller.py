"""Replay Controller for MoE expert offloading (Phase 6).

Replays a pre-computed DataMovementTrace on real GPU hardware using a
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
    2. Simulate policy:           LRUPolicy.simulate() -> DataMovementTrace
    3. Replay on hardware:        ReplayController(engine, trace) -> timing

Usage:
    from data_movement_trace import DataMovementTrace
    from replay_controller import ReplayController

    trace = DataMovementTrace.load("lru_movement.json")
    controller = ReplayController(engine, trace)
    controller.setup()

    engine.replay_controller = controller
    for step in range(len(trace.steps)):
        logits = engine.mixed_step(...)

    timing = controller.get_replay_stats()
"""

import torch

from data_movement_trace import DataMovementTrace, TransferEvent


class ReplayController:
    """Replays a DataMovementTrace with async transfers on real GPU hardware.

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

    def __init__(self, engine, trace: DataMovementTrace):
        """Create replay controller.

        Args:
            engine: MoEEngine instance with cache_size set.
            trace: Pre-computed data movement trace to replay.
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

        # Timing records
        self.step_timings = []

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

        1. Copy expert_map_abs[layer] → expert_map_buf
        2. Wait for async prefetches targeting this layer
        3. Execute demand_loads (blocking on compute stream)
        4. Re-copy expert_map_abs → expert_map_buf (updated by transfers)
        5. Issue prefetches for layer+1 (async, overlaps with stage4b)
        """
        step_trace = self.trace.steps[self._current_step]
        layer_trace = step_trace.layers[layer]

        # Step 1: set base expert map for this layer
        self.expert_map_buf.copy_(self.expert_map_abs[layer])

        # Step 2: sync prefetches targeting this layer
        if self._has_prefetches:
            torch.cuda.current_stream().wait_event(
                self._prefetch_done_event)

        # Step 3: demand loads (blocking, on compute stream)
        for event in layer_trace.demand_loads:
            self._execute_transfer(event)

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

    def get_replay_stats(self) -> dict:
        """Compute summary statistics from the trace being replayed."""
        return self.trace.summary()
