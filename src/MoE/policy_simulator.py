"""Cache policy simulators for MoE expert offloading (Phase 7).

Pure-Python simulators that consume an ActivationTrace (recorded expert
access patterns) and produce a DataMovementTrace (prefetch/eviction/
demand-load schedule) for replay on real GPU hardware.

All policies use a unified expert cache shared across all layers, keyed
by (layer, expert_id) pairs. The cache_size parameter is the total number
of expert slots (not per-layer).

Policies:
    LRUPolicy       — Least Recently Used eviction, demand-load only.
    OraclePolicy    — Belady's optimal (MIN) with lookahead prefetching.
    FrequencyPolicy — Least Frequently Used (static or windowed).
    StaticPolicy    — Global frequency ranking (oracle); pins top experts.
    PreGatedPolicy  — Wraps any base policy, adds 1-layer-ahead prefetch.

Usage:
    trace = ActivationTrace.load("expert_trace.json")
    policy = LRUPolicy()
    dm_trace = policy.simulate(trace, cache_size=6)
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
                 cache_size: int,
                 initial_cache: Optional[list[tuple[int, int]]] = None
                 ) -> DataMovementTrace:
        """Run the policy simulation on an activation trace.

        Args:
            activation_trace: Expert access pattern from real inference.
            cache_size: Total unified cache capacity (expert slots shared
                across all layers).
            initial_cache: List of (layer, expert_id) pairs to pre-load.
                Defaults to filling uniformly across layers.

        Returns:
            DataMovementTrace ready for replay.
        """
        ...


def _default_initial_cache(cache_size: int, num_layers: int,
                           num_experts: int) -> list[tuple[int, int]]:
    """Default initial cache: fill uniformly across layers.

    Iterates expert 0 for all layers, then expert 1 for all layers, etc.,
    stopping when cache_size entries have been added.
    """
    initial = []
    for eid in range(num_experts):
        for layer in range(num_layers):
            if len(initial) >= cache_size:
                return initial
            initial.append((layer, eid))
    return initial


def _make_layer_trace(needed: list[int],
                      prefetches: list[TransferEvent],
                      demand_loads: list[TransferEvent]) -> LayerTrace:
    """Build a LayerTrace with uniform weights."""
    n = len(needed)
    weights = [1.0 / n] * n if n > 0 else []
    return LayerTrace(
        topk_ids=[needed],
        topk_weights=[weights],
        prefetches=prefetches,
        demand_loads=demand_loads,
    )


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction with unified cache.

    Single LRU cache keyed by (layer, expert_id). On cache miss:
      - If cache not full: load into free slot (evict=None).
      - If cache full: evict globally-LRU expert.

    All loads are demand loads (no prefetching).
    """

    def simulate(self, activation_trace: ActivationTrace,
                 cache_size: int,
                 initial_cache: Optional[list[tuple[int, int]]] = None
                 ) -> DataMovementTrace:
        num_layers = activation_trace.num_layers
        num_experts = activation_trace.num_experts

        if initial_cache is None:
            initial_cache = _default_initial_cache(
                cache_size, num_layers, num_experts)

        # Unified LRU: OrderedDict keyed by (layer, eid)
        cache = OrderedDict()
        for key in initial_cache:
            cache[tuple(key)] = True

        steps = []
        for step_idx, step_experts in enumerate(activation_trace.steps):
            layer_traces = []
            for layer in range(num_layers):
                needed = step_experts[layer]
                demand_loads = []

                for eid in needed:
                    key = (layer, eid)
                    if key in cache:
                        cache.move_to_end(key)
                    else:
                        evict = None
                        if len(cache) >= cache_size:
                            evict_key, _ = cache.popitem(last=False)
                            evict = evict_key
                        cache[key] = True
                        demand_loads.append(TransferEvent(
                            target=(layer, eid), evict=evict))

                layer_traces.append(_make_layer_trace(
                    needed, prefetches=[], demand_loads=demand_loads))
            steps.append(StepTrace(layers=layer_traces))

        return DataMovementTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            cache_size=cache_size,
            initial_cache_state=list(initial_cache),
            steps=steps,
        )


class OraclePolicy(EvictionPolicy):
    """Belady's optimal (MIN) algorithm with lookahead prefetching.

    Unified cache: evicts the resident (layer, expert_id) whose next use
    is farthest in the future, across all layers. Requires full trace
    knowledge (offline/oracle). This is the theoretical lower bound on
    cache misses.

    Prefetching: at layer L, look ahead to layer L+1 (same step). If
    L+1 needs a non-resident expert, issue it as a prefetch stored in
    layers[L+1].prefetches, so the replay controller can overlap it
    with stage4b(L) + stages 1-4a(L+1).
    """

    def simulate(self, activation_trace: ActivationTrace,
                 cache_size: int,
                 initial_cache: Optional[list[tuple[int, int]]] = None
                 ) -> DataMovementTrace:
        num_layers = activation_trace.num_layers
        num_experts = activation_trace.num_experts

        if initial_cache is None:
            initial_cache = _default_initial_cache(
                cache_size, num_layers, num_experts)

        # ── Pass 1: Build next-use index ──
        # next_use[layer][expert_id] = sorted list of global positions
        # where position = step_idx * num_layers + layer
        next_use = [[[] for _ in range(num_experts)]
                    for _ in range(num_layers)]
        for step_idx, step_experts in enumerate(activation_trace.steps):
            for layer in range(num_layers):
                for eid in step_experts[layer]:
                    pos = step_idx * num_layers + layer
                    next_use[layer][eid].append(pos)

        # Per (layer, expert) pointer into next_use
        use_ptr = [[0] * num_experts for _ in range(num_layers)]

        def get_next_use(layer: int, eid: int, current_pos: int) -> int:
            """Return the next use position after current_pos, or inf."""
            ptr = use_ptr[layer][eid]
            uses = next_use[layer][eid]
            while ptr < len(uses) and uses[ptr] <= current_pos:
                ptr += 1
            use_ptr[layer][eid] = ptr
            if ptr < len(uses):
                return uses[ptr]
            return float('inf')

        # ── Pass 2: Simulate forward ──
        cache = OrderedDict()
        for key in initial_cache:
            cache[tuple(key)] = True

        steps = []
        for step_idx, step_experts in enumerate(activation_trace.steps):
            layer_traces = []
            # Collect prefetches for each layer (filled by previous layer)
            pending_prefetches = [[] for _ in range(num_layers)]

            for layer in range(num_layers):
                current_pos = step_idx * num_layers + layer
                needed = step_experts[layer]

                # Protected: experts this layer needs (don't evict them)
                protected = {(layer, eid) for eid in needed}

                # Demand loads for current layer
                demand_loads = []
                for eid in needed:
                    key = (layer, eid)
                    if key not in cache:
                        evict = None
                        if len(cache) >= cache_size:
                            evict_key = self._belady_victim(
                                cache, current_pos, get_next_use,
                                protected)
                            del cache[evict_key]
                            evict = evict_key
                        cache[key] = True
                        demand_loads.append(TransferEvent(
                            target=(layer, eid), evict=evict))

                # Lookahead prefetch for layer+1
                if layer + 1 < num_layers:
                    next_needed = step_experts[layer + 1]
                    # Protect both current and next layer's needs
                    pf_protected = (protected |
                                    {(layer + 1, e) for e in next_needed})
                    for eid in next_needed:
                        key = (layer + 1, eid)
                        if key not in cache:
                            evict = None
                            if len(cache) >= cache_size:
                                evict_key = self._belady_victim(
                                    cache,
                                    step_idx * num_layers + layer + 1,
                                    get_next_use, pf_protected)
                                del cache[evict_key]
                                evict = evict_key
                            cache[key] = True
                            pending_prefetches[layer + 1].append(
                                TransferEvent(
                                    target=(layer + 1, eid), evict=evict))

                layer_traces.append(_make_layer_trace(
                    needed,
                    prefetches=pending_prefetches[layer],
                    demand_loads=demand_loads))
            steps.append(StepTrace(layers=layer_traces))

        return DataMovementTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            cache_size=cache_size,
            initial_cache_state=list(initial_cache),
            steps=steps,
        )

    @staticmethod
    def _belady_victim(cache: OrderedDict, current_pos: int,
                       get_next_use, protected: set) -> tuple:
        """Find the cache resident whose next use is farthest away.

        Args:
            cache: Current unified cache (OrderedDict keyed by (layer, eid)).
            current_pos: Current global position (step*L + layer).
            get_next_use: Function(layer, eid, pos) -> next use position.
            protected: Set of (layer, eid) keys not to evict.

        Returns:
            (layer, eid) key to evict.
        """
        best_key = None
        best_dist = -1
        for key in cache:
            if key in protected:
                continue
            layer, eid = key
            dist = get_next_use(layer, eid, current_pos)
            if dist > best_dist:
                best_dist = dist
                best_key = key
        if best_key is None:
            # All residents are protected — evict farthest anyway
            for key in cache:
                layer, eid = key
                dist = get_next_use(layer, eid, current_pos)
                if dist > best_dist:
                    best_dist = dist
                    best_key = key
        return best_key


class FrequencyPolicy(EvictionPolicy):
    """Frequency-based (LFU) eviction with unified cache.

    Tracks per-(layer, expert) access frequency. On eviction, removes
    the least frequently used expert globally. Ties broken by key order
    (lowest layer first, then lowest expert_id).

    Attributes:
        window_size: If set, reset frequency counts every window_size steps.
            None means static LFU (counts never reset).
    """

    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size

    def simulate(self, activation_trace: ActivationTrace,
                 cache_size: int,
                 initial_cache: Optional[list[tuple[int, int]]] = None
                 ) -> DataMovementTrace:
        num_layers = activation_trace.num_layers
        num_experts = activation_trace.num_experts

        if initial_cache is None:
            initial_cache = _default_initial_cache(
                cache_size, num_layers, num_experts)

        # Unified frequency counts and cache
        freq = {}  # (layer, eid) -> count
        cache = set()
        for key in initial_cache:
            key = tuple(key)
            cache.add(key)
            freq[key] = 0

        steps = []
        for step_idx, step_experts in enumerate(activation_trace.steps):
            # Windowed reset
            if (self.window_size is not None and step_idx > 0 and
                    step_idx % self.window_size == 0):
                for k in freq:
                    freq[k] = 0

            layer_traces = []
            for layer in range(num_layers):
                needed = step_experts[layer]
                demand_loads = []

                # Update frequency for accessed experts
                for eid in needed:
                    key = (layer, eid)
                    freq[key] = freq.get(key, 0) + 1

                for eid in needed:
                    key = (layer, eid)
                    if key not in cache:
                        evict = None
                        if len(cache) >= cache_size:
                            protected = {(layer, e) for e in needed}
                            evict_key = self._lfu_victim(
                                cache, freq, protected)
                            cache.discard(evict_key)
                            evict = evict_key
                        cache.add(key)
                        if key not in freq:
                            freq[key] = 1
                        demand_loads.append(TransferEvent(
                            target=(layer, eid), evict=evict))

                layer_traces.append(_make_layer_trace(
                    needed, prefetches=[], demand_loads=demand_loads))
            steps.append(StepTrace(layers=layer_traces))

        return DataMovementTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            cache_size=cache_size,
            initial_cache_state=list(initial_cache),
            steps=steps,
        )

    @staticmethod
    def _lfu_victim(cache: set, freq: dict,
                    protected: set) -> tuple:
        """Find the least frequently used expert in the unified cache.

        Protected experts are avoided if possible.
        Ties broken by (layer, eid) sort order.
        """
        best_key = None
        best_freq = float('inf')
        for key in sorted(cache):
            if key in protected:
                continue
            f = freq.get(key, 0)
            if f < best_freq:
                best_freq = f
                best_key = key
        if best_key is None:
            # All residents are protected
            for key in sorted(cache):
                f = freq.get(key, 0)
                if f < best_freq:
                    best_freq = f
                    best_key = key
        return best_key


class StaticPolicy(EvictionPolicy):
    """Static frequency-based policy with unified cache.

    Pre-computes global access frequencies over the ENTIRE trace (including
    future steps), then:
      1. Initializes the cache with the top-`cache_size` experts by frequency.
      2. On cache miss, always evicts the resident with the lowest global
         frequency (ties broken by (layer, eid) order).

    Because frequencies are fixed, the top (cache_size - num_experts) experts
    are effectively pinned and never evicted. Only the bottom num_experts
    slots cycle through demand-loaded experts.

    All loads are demand loads (no prefetching).
    """

    def simulate(self, activation_trace: ActivationTrace,
                 cache_size: int,
                 initial_cache: Optional[list[tuple[int, int]]] = None
                 ) -> DataMovementTrace:
        num_layers = activation_trace.num_layers
        num_experts = activation_trace.num_experts

        # ── Pre-compute global frequencies over entire trace ──
        global_freq = {}  # (layer, eid) -> total access count
        for step_experts in activation_trace.steps:
            for layer in range(num_layers):
                for eid in step_experts[layer]:
                    key = (layer, eid)
                    global_freq[key] = global_freq.get(key, 0) + 1

        # ── Initial cache: top cache_size by frequency ──
        if initial_cache is None:
            # All possible (layer, eid) pairs, sorted by descending frequency
            all_experts = [(layer, eid)
                           for layer in range(num_layers)
                           for eid in range(num_experts)]
            all_experts.sort(key=lambda k: (-global_freq.get(k, 0), k))
            initial_cache = all_experts[:cache_size]

        cache = set()
        for key in initial_cache:
            cache.add(tuple(key))

        # ── Simulate forward ──
        steps = []
        for step_idx, step_experts in enumerate(activation_trace.steps):
            layer_traces = []
            for layer in range(num_layers):
                needed = step_experts[layer]
                demand_loads = []

                for eid in needed:
                    key = (layer, eid)
                    if key not in cache:
                        evict = None
                        if len(cache) >= cache_size:
                            protected = {(layer, e) for e in needed}
                            evict_key = self._min_freq_victim(
                                cache, global_freq, protected)
                            cache.discard(evict_key)
                            evict = evict_key
                        cache.add(key)
                        demand_loads.append(TransferEvent(
                            target=(layer, eid), evict=evict))

                layer_traces.append(_make_layer_trace(
                    needed, prefetches=[], demand_loads=demand_loads))
            steps.append(StepTrace(layers=layer_traces))

        return DataMovementTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            cache_size=cache_size,
            initial_cache_state=list(initial_cache),
            steps=steps,
        )

    @staticmethod
    def _min_freq_victim(cache: set, global_freq: dict,
                         protected: set) -> tuple:
        """Find the resident with the lowest global frequency.

        Protected experts are avoided if possible.
        Ties broken by (layer, eid) sort order.
        """
        best_key = None
        best_freq = float('inf')
        for key in sorted(cache):
            if key in protected:
                continue
            f = global_freq.get(key, 0)
            if f < best_freq:
                best_freq = f
                best_key = key
        if best_key is None:
            for key in sorted(cache):
                f = global_freq.get(key, 0)
                if f < best_freq:
                    best_freq = f
                    best_key = key
        return best_key


class PreGatedPolicy(EvictionPolicy):
    """Pre-gated prefetching policy with unified cache.

    Wraps a base eviction policy and adds 1-layer-ahead prefetching.
    At layer L, looks ahead to layer L+1's needed experts (same step)
    and issues prefetches for non-resident ones. These prefetches are
    stored in layers[L+1].prefetches so the replay controller issues
    them before stage4b of layer L.

    Eviction and demand-load decisions are delegated to the base policy's
    cache management logic.

    Attributes:
        base_policy_type: 'lru' or 'frequency' — which policy to use.
        window_size: Passed to FrequencyPolicy if base is 'frequency'.
    """

    def __init__(self, base_policy_type: str = 'lru',
                 window_size: Optional[int] = None):
        self.base_policy_type = base_policy_type
        self.window_size = window_size

    def simulate(self, activation_trace: ActivationTrace,
                 cache_size: int,
                 initial_cache: Optional[list[tuple[int, int]]] = None
                 ) -> DataMovementTrace:
        num_layers = activation_trace.num_layers
        num_experts = activation_trace.num_experts

        if initial_cache is None:
            initial_cache = _default_initial_cache(
                cache_size, num_layers, num_experts)

        # Set up unified cache based on base policy type
        if self.base_policy_type == 'lru':
            cache = OrderedDict()
            for key in initial_cache:
                cache[tuple(key)] = True
        else:
            cache = set()
            freq = {}
            for key in initial_cache:
                key = tuple(key)
                cache.add(key)
                freq[key] = 0

        steps = []
        for step_idx, step_experts in enumerate(activation_trace.steps):
            # Windowed frequency reset
            if (self.base_policy_type == 'frequency' and
                    self.window_size is not None and step_idx > 0 and
                    step_idx % self.window_size == 0):
                for k in freq:
                    freq[k] = 0

            layer_traces = []
            pending_prefetches = [[] for _ in range(num_layers)]

            for layer in range(num_layers):
                needed = step_experts[layer]

                # Update frequency counts if applicable
                if self.base_policy_type == 'frequency':
                    for eid in needed:
                        key = (layer, eid)
                        freq[key] = freq.get(key, 0) + 1

                # ── Demand loads for current layer ──
                demand_loads = []
                for eid in needed:
                    key = (layer, eid)
                    if self.base_policy_type == 'lru':
                        if key in cache:
                            cache.move_to_end(key)
                        else:
                            evict = None
                            if len(cache) >= cache_size:
                                evict_key, _ = cache.popitem(last=False)
                                evict = evict_key
                            cache[key] = True
                            demand_loads.append(TransferEvent(
                                target=(layer, eid), evict=evict))
                    else:
                        if key not in cache:
                            evict = None
                            if len(cache) >= cache_size:
                                protected = {(layer, e) for e in needed}
                                evict_key = FrequencyPolicy._lfu_victim(
                                    cache, freq, protected)
                                cache.discard(evict_key)
                                evict = evict_key
                            cache.add(key)
                            if key not in freq:
                                freq[key] = 1
                            demand_loads.append(TransferEvent(
                                target=(layer, eid), evict=evict))

                # ── Lookahead prefetch for layer+1 ──
                if layer + 1 < num_layers:
                    next_needed = step_experts[layer + 1]

                    for eid in next_needed:
                        key = (layer + 1, eid)
                        is_resident = (key in cache)
                        if not is_resident:
                            evict = None
                            if self.base_policy_type == 'lru':
                                if len(cache) >= cache_size:
                                    evict_key, _ = cache.popitem(
                                        last=False)
                                    evict = evict_key
                                cache[key] = True
                            else:
                                if len(cache) >= cache_size:
                                    protected = (
                                        {(layer, e) for e in needed} |
                                        {(layer + 1, e)
                                         for e in next_needed})
                                    evict_key = (
                                        FrequencyPolicy._lfu_victim(
                                            cache, freq, protected))
                                    cache.discard(evict_key)
                                    evict = evict_key
                                cache.add(key)
                            pending_prefetches[layer + 1].append(
                                TransferEvent(
                                    target=(layer + 1, eid), evict=evict))

                layer_traces.append(_make_layer_trace(
                    needed,
                    prefetches=pending_prefetches[layer],
                    demand_loads=demand_loads))
            steps.append(StepTrace(layers=layer_traces))

        return DataMovementTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            cache_size=cache_size,
            initial_cache_state=list(initial_cache),
            steps=steps,
        )
