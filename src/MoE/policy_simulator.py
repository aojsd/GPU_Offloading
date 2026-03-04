"""Cache policy simulators for MoE expert offloading (Phase 7).

Pure-Python simulators that consume an ActivationTrace (recorded expert
access patterns) and produce a DataMovementTrace (prefetch/eviction/
demand-load schedule) for replay on real GPU hardware.

Policies:
    LRUPolicy       — Least Recently Used eviction, demand-load only.
    OraclePolicy    — Belady's optimal (MIN) with lookahead prefetching.
    FrequencyPolicy — Least Frequently Used (static or windowed).
    PreGatedPolicy  — Wraps any base policy, adds 1-layer-ahead prefetch.

Usage:
    trace = ActivationTrace.load("expert_trace.json")
    policy = LRUPolicy()
    dm_trace = policy.simulate(trace, experts_per_layer=2)
    errors = dm_trace.validate()
    assert not errors
    dm_trace.save("lru_movement.json")
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

from data_movement_trace import (
    ActivationTrace,
    DataMovementTrace,
    LayerTrace,
    StepTrace,
    TransferEvent,
)


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""

    @abstractmethod
    def simulate(self, activation_trace: ActivationTrace,
                 experts_per_layer: int,
                 initial_experts: Optional[list[int]] = None
                 ) -> DataMovementTrace:
        """Run the policy simulation on an activation trace.

        Args:
            activation_trace: Expert access pattern from real inference.
            experts_per_layer: GPU cache capacity per layer.
            initial_experts: Which experts to pre-load per layer at start.
                Defaults to [0, 1, ..., experts_per_layer - 1].

        Returns:
            DataMovementTrace ready for replay.
        """
        ...


def _default_initial_experts(experts_per_layer: int,
                             num_experts: int) -> list[int]:
    """Default initial cache: first experts_per_layer experts."""
    return list(range(min(experts_per_layer, num_experts)))


def _build_initial_state(num_layers: int,
                         initial_experts: list[int]
                         ) -> list[tuple[int, int]]:
    """Build initial_cache_state list from per-layer initial experts."""
    return [(l, eid) for l in range(num_layers) for eid in initial_experts]


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction.

    Per-layer LRU cache. On cache miss:
      - If cache not full: load without eviction (scratchpad).
      - If cache full: evict LRU expert, load into freed persistent slot.

    All loads are demand loads (no prefetching).
    """

    def simulate(self, activation_trace: ActivationTrace,
                 experts_per_layer: int,
                 initial_experts: Optional[list[int]] = None
                 ) -> DataMovementTrace:
        num_layers = activation_trace.num_layers
        num_experts = activation_trace.num_experts

        if initial_experts is None:
            initial_experts = _default_initial_experts(
                experts_per_layer, num_experts)

        # Per-layer LRU: OrderedDict preserves insertion/access order
        cache = [OrderedDict() for _ in range(num_layers)]
        for l in range(num_layers):
            for eid in initial_experts:
                cache[l][eid] = True

        initial_state = _build_initial_state(num_layers, initial_experts)
        steps = []

        for step_idx, step_experts in enumerate(activation_trace.steps):
            layer_traces = []
            for layer in range(num_layers):
                needed = step_experts[layer]
                demand_loads = []

                for eid in needed:
                    if eid in cache[layer]:
                        # Cache hit — move to end (most recently used)
                        cache[layer].move_to_end(eid)
                    else:
                        # Cache miss — demand load
                        evict = None
                        if len(cache[layer]) >= experts_per_layer:
                            # Evict LRU (first item in OrderedDict)
                            evict_eid, _ = cache[layer].popitem(last=False)
                            evict = (layer, evict_eid)
                        cache[layer][eid] = True
                        demand_loads.append(TransferEvent(
                            target=(layer, eid), evict=evict))

                layer_traces.append(LayerTrace(
                    topk_ids=[needed],
                    topk_weights=[[1.0 / len(needed)] * len(needed)
                                  if needed else []],
                    prefetches=[],
                    post_routing_prefetches=[],
                    demand_loads=demand_loads,
                ))
            steps.append(StepTrace(layers=layer_traces))

        return DataMovementTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            experts_per_layer=experts_per_layer,
            initial_cache_state=initial_state,
            steps=steps,
        )


class OraclePolicy(EvictionPolicy):
    """Belady's optimal (MIN) algorithm with lookahead prefetching.

    Evicts the resident expert whose next use is farthest in the future.
    Requires full trace knowledge (offline/oracle). This is the theoretical
    lower bound on cache misses.

    Prefetching: at layer L, look ahead to layer L+1 (same step). If
    L+1 needs a non-resident expert, issue it as a prefetch at layer L
    so it overlaps with attention compute.
    """

    def simulate(self, activation_trace: ActivationTrace,
                 experts_per_layer: int,
                 initial_experts: Optional[list[int]] = None
                 ) -> DataMovementTrace:
        num_layers = activation_trace.num_layers
        num_experts = activation_trace.num_experts

        if initial_experts is None:
            initial_experts = _default_initial_experts(
                experts_per_layer, num_experts)

        # ── Pass 1: Build next-use index ──
        # next_use[layer][expert_id] = sorted list of (step_idx, position)
        # where position = step_idx * num_layers + layer (global order)
        next_use = [[[] for _ in range(num_experts)]
                    for _ in range(num_layers)]
        for step_idx, step_experts in enumerate(activation_trace.steps):
            for layer in range(num_layers):
                for eid in step_experts[layer]:
                    pos = step_idx * num_layers + layer
                    next_use[layer][eid].append(pos)

        # For each (layer, expert), build an iterator over future uses
        # We'll track a pointer per (layer, expert)
        use_ptr = [[0] * num_experts for _ in range(num_layers)]

        def get_next_use(layer: int, eid: int, current_pos: int) -> int:
            """Return the next use position after current_pos, or inf."""
            ptr = use_ptr[layer][eid]
            uses = next_use[layer][eid]
            # Advance pointer past current_pos
            while ptr < len(uses) and uses[ptr] <= current_pos:
                ptr += 1
            use_ptr[layer][eid] = ptr
            if ptr < len(uses):
                return uses[ptr]
            return float('inf')

        # ── Pass 2: Simulate forward ──
        cache = [OrderedDict() for _ in range(num_layers)]
        for l in range(num_layers):
            for eid in initial_experts:
                cache[l][eid] = True

        initial_state = _build_initial_state(num_layers, initial_experts)
        steps = []

        for step_idx, step_experts in enumerate(activation_trace.steps):
            layer_traces = []
            for layer in range(num_layers):
                current_pos = step_idx * num_layers + layer
                needed = step_experts[layer]

                # Determine misses for current layer
                demand_loads = []
                for eid in needed:
                    if eid not in cache[layer]:
                        evict = None
                        if len(cache[layer]) >= experts_per_layer:
                            evict_eid = self._belady_victim(
                                cache[layer], layer, current_pos,
                                get_next_use, needed)
                            del cache[layer][evict_eid]
                            evict = (layer, evict_eid)
                        cache[layer][eid] = True
                        demand_loads.append(TransferEvent(
                            target=(layer, eid), evict=evict))

                # Lookahead prefetch: check layer+1 (same step)
                prefetches = []
                if layer + 1 < num_layers:
                    next_layer_needed = step_experts[layer + 1]
                    for eid in next_layer_needed:
                        if eid not in cache[layer + 1]:
                            # Check if cache has room or find victim
                            evict = None
                            if len(cache[layer + 1]) >= experts_per_layer:
                                next_pos = (step_idx * num_layers +
                                            layer + 1)
                                evict_eid = self._belady_victim(
                                    cache[layer + 1], layer + 1, next_pos,
                                    get_next_use, next_layer_needed)
                                del cache[layer + 1][evict_eid]
                                evict = (layer + 1, evict_eid)
                            cache[layer + 1][eid] = True
                            prefetches.append(TransferEvent(
                                target=(layer + 1, eid), evict=evict))

                layer_traces.append(LayerTrace(
                    topk_ids=[needed],
                    topk_weights=[[1.0 / len(needed)] * len(needed)
                                  if needed else []],
                    prefetches=prefetches,
                    post_routing_prefetches=[],
                    demand_loads=demand_loads,
                ))
            steps.append(StepTrace(layers=layer_traces))

        return DataMovementTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            experts_per_layer=experts_per_layer,
            initial_cache_state=initial_state,
            steps=steps,
        )

    @staticmethod
    def _belady_victim(cache: OrderedDict, layer: int,
                       current_pos: int, get_next_use, protected) -> int:
        """Find the cache resident whose next use is farthest away.

        Args:
            cache: Current cache contents (OrderedDict).
            layer: Layer index.
            current_pos: Current global position (step*L + layer).
            get_next_use: Function(layer, eid, pos) -> next use position.
            protected: Set/list of expert IDs needed now (don't evict).

        Returns:
            Expert ID to evict.
        """
        best_eid = None
        best_dist = -1
        for eid in cache:
            if eid in protected:
                continue
            dist = get_next_use(layer, eid, current_pos)
            if dist > best_dist:
                best_dist = dist
                best_eid = eid
        if best_eid is None:
            # All residents are protected — evict the one with farthest use
            for eid in cache:
                dist = get_next_use(layer, eid, current_pos)
                if dist > best_dist:
                    best_dist = dist
                    best_eid = eid
        return best_eid


class FrequencyPolicy(EvictionPolicy):
    """Frequency-based (LFU) eviction.

    Tracks per-layer expert access frequency. On eviction, removes the
    least frequently used expert. Ties broken by expert_id (lowest evicted).

    Attributes:
        window_size: If set, reset frequency counts every window_size steps.
            None means static LFU (counts never reset).
    """

    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size

    def simulate(self, activation_trace: ActivationTrace,
                 experts_per_layer: int,
                 initial_experts: Optional[list[int]] = None
                 ) -> DataMovementTrace:
        num_layers = activation_trace.num_layers
        num_experts = activation_trace.num_experts

        if initial_experts is None:
            initial_experts = _default_initial_experts(
                experts_per_layer, num_experts)

        # Per-layer frequency counts and cache
        freq = [[0] * num_experts for _ in range(num_layers)]
        cache = [set(initial_experts) for _ in range(num_layers)]

        initial_state = _build_initial_state(num_layers, initial_experts)
        steps = []

        for step_idx, step_experts in enumerate(activation_trace.steps):
            # Windowed reset
            if (self.window_size is not None and step_idx > 0 and
                    step_idx % self.window_size == 0):
                freq = [[0] * num_experts for _ in range(num_layers)]

            layer_traces = []
            for layer in range(num_layers):
                needed = step_experts[layer]
                demand_loads = []

                # Update frequency for accessed experts
                for eid in needed:
                    freq[layer][eid] += 1

                for eid in needed:
                    if eid not in cache[layer]:
                        evict = None
                        if len(cache[layer]) >= experts_per_layer:
                            evict_eid = self._lfu_victim(
                                cache[layer], freq[layer], needed)
                            cache[layer].discard(evict_eid)
                            evict = (layer, evict_eid)
                        cache[layer].add(eid)
                        demand_loads.append(TransferEvent(
                            target=(layer, eid), evict=evict))

                layer_traces.append(LayerTrace(
                    topk_ids=[needed],
                    topk_weights=[[1.0 / len(needed)] * len(needed)
                                  if needed else []],
                    prefetches=[],
                    post_routing_prefetches=[],
                    demand_loads=demand_loads,
                ))
            steps.append(StepTrace(layers=layer_traces))

        return DataMovementTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            experts_per_layer=experts_per_layer,
            initial_cache_state=initial_state,
            steps=steps,
        )

    @staticmethod
    def _lfu_victim(cache: set, freq: list[int],
                    protected: list[int]) -> int:
        """Find the least frequently used expert in cache.

        Protected experts (needed now) are avoided if possible.
        Ties broken by lowest expert_id.
        """
        best_eid = None
        best_freq = float('inf')
        for eid in sorted(cache):
            if eid in protected:
                continue
            if freq[eid] < best_freq:
                best_freq = freq[eid]
                best_eid = eid
        if best_eid is None:
            # All residents are protected
            for eid in sorted(cache):
                if freq[eid] < best_freq:
                    best_freq = freq[eid]
                    best_eid = eid
        return best_eid


class PreGatedPolicy(EvictionPolicy):
    """Pre-gated prefetching policy.

    Wraps a base eviction policy and adds 1-layer-ahead prefetching.
    At layer L, looks ahead to layer L+1's needed experts (same step)
    and issues pre-attention prefetches for non-resident ones.

    In a real system, this corresponds to running the router's gate
    network early (before attention) to predict which experts the next
    layer will need. Since we're simulating on recorded traces, we
    simply look ahead in the trace data.

    Eviction and demand-load decisions are delegated to the base policy's
    cache management logic.

    Attributes:
        base_policy_type: 'lru' or 'frequency' — which policy to use for
            cache eviction decisions.
        window_size: Passed to FrequencyPolicy if base is 'frequency'.
    """

    def __init__(self, base_policy_type: str = 'lru',
                 window_size: Optional[int] = None):
        self.base_policy_type = base_policy_type
        self.window_size = window_size

    def simulate(self, activation_trace: ActivationTrace,
                 experts_per_layer: int,
                 initial_experts: Optional[list[int]] = None
                 ) -> DataMovementTrace:
        num_layers = activation_trace.num_layers
        num_experts = activation_trace.num_experts

        if initial_experts is None:
            initial_experts = _default_initial_experts(
                experts_per_layer, num_experts)

        # Set up per-layer cache state based on base policy type
        if self.base_policy_type == 'lru':
            cache = [OrderedDict() for _ in range(num_layers)]
            for l in range(num_layers):
                for eid in initial_experts:
                    cache[l][eid] = True
        else:
            cache = [set(initial_experts) for _ in range(num_layers)]
            freq = [[0] * num_experts for _ in range(num_layers)]

        initial_state = _build_initial_state(num_layers, initial_experts)
        steps = []

        for step_idx, step_experts in enumerate(activation_trace.steps):
            # Windowed frequency reset
            if (self.base_policy_type == 'frequency' and
                    self.window_size is not None and step_idx > 0 and
                    step_idx % self.window_size == 0):
                freq = [[0] * num_experts for _ in range(num_layers)]

            layer_traces = []
            for layer in range(num_layers):
                needed = step_experts[layer]

                # Update frequency counts if applicable
                if self.base_policy_type == 'frequency':
                    for eid in needed:
                        freq[layer][eid] += 1

                # ── Demand loads for current layer ──
                demand_loads = []
                for eid in needed:
                    if self.base_policy_type == 'lru':
                        if eid in cache[layer]:
                            cache[layer].move_to_end(eid)
                        else:
                            evict = None
                            if len(cache[layer]) >= experts_per_layer:
                                evict_eid, _ = cache[layer].popitem(
                                    last=False)
                                evict = (layer, evict_eid)
                            cache[layer][eid] = True
                            demand_loads.append(TransferEvent(
                                target=(layer, eid), evict=evict))
                    else:
                        if eid not in cache[layer]:
                            evict = None
                            if len(cache[layer]) >= experts_per_layer:
                                evict_eid = FrequencyPolicy._lfu_victim(
                                    cache[layer], freq[layer], needed)
                                cache[layer].discard(evict_eid)
                                evict = (layer, evict_eid)
                            cache[layer].add(eid)
                            demand_loads.append(TransferEvent(
                                target=(layer, eid), evict=evict))

                # ── Lookahead prefetch for layer+1 ──
                prefetches = []
                if layer + 1 < num_layers:
                    next_needed = step_experts[layer + 1]
                    next_cache = cache[layer + 1]

                    for eid in next_needed:
                        is_resident = (eid in next_cache)
                        if not is_resident:
                            evict = None
                            if self.base_policy_type == 'lru':
                                if len(next_cache) >= experts_per_layer:
                                    evict_eid, _ = next_cache.popitem(
                                        last=False)
                                    evict = (layer + 1, evict_eid)
                                next_cache[eid] = True
                            else:
                                if len(next_cache) >= experts_per_layer:
                                    evict_eid = (
                                        FrequencyPolicy._lfu_victim(
                                            next_cache, freq[layer + 1],
                                            next_needed))
                                    next_cache.discard(evict_eid)
                                    evict = (layer + 1, evict_eid)
                                next_cache.add(eid)
                            prefetches.append(TransferEvent(
                                target=(layer + 1, eid), evict=evict))

                layer_traces.append(LayerTrace(
                    topk_ids=[needed],
                    topk_weights=[[1.0 / len(needed)] * len(needed)
                                  if needed else []],
                    prefetches=prefetches,
                    post_routing_prefetches=[],
                    demand_loads=demand_loads,
                ))
            steps.append(StepTrace(layers=layer_traces))

        return DataMovementTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            experts_per_layer=experts_per_layer,
            initial_cache_state=initial_state,
            steps=steps,
        )
