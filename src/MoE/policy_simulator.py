"""Cache policy simulators for MoE expert offloading.

Decomposes expert cache management into two orthogonal policies:
  - CachePolicy:    eviction strategy (which expert to remove on cache full)
  - PrefetchPolicy: prefetch strategy (what to load proactively)

Cache policies:
    LRU        — Least Recently Used eviction.
    Belady     — Optimal (MIN): evicts farthest-future-use (offline/oracle).
    LFU        — Least Frequently Used (static or windowed).
    StaticFreq — Global frequency ranking (oracle); pins top experts.

Prefetch policies:
    NoPrefetch     — Demand loads only, no prefetching.
    OraclePrefetch — 1-layer lookahead + layer-0 prefetch (requires full trace).

Naming convention: <cache>-<prefetch>
    e.g. "LRU-None", "Belady-Oracle", "Static-Oracle"

Usage:
    trace = ActivationTrace.load("expert_trace.json")
    dm = simulate(LRU(), NoPrefetch(), trace, cache_size=6)
    dm = simulate(StaticFreq(), OraclePrefetch(), trace, cache_size=128)
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

from data_movement_trace import (
    ActivationTrace,
    DataMovementTrace,
    LayerTrace,
    StepScheduling,
    StepTrace,
    TransferEvent,
)


# ── Cache Policies (eviction strategy) ─────────────────────────────

class CachePolicy(ABC):
    """Abstract eviction strategy for the unified expert cache."""

    def setup(self, activation_trace: ActivationTrace,
              cache_size: int, num_layers: int, num_experts: int
              ) -> Optional[list[tuple[int, int]]]:
        """Pre-compute trace-level data. Return custom initial_cache or None."""
        return None

    @abstractmethod
    def init_cache(self, initial_cache: list[tuple[int, int]]) -> None: ...

    def begin_step(self, step_idx: int) -> None:
        """Called at start of each step (for windowed policies)."""
        pass

    def record_access(self, key: tuple[int, int]) -> None:
        """Called for every needed expert (hit or miss), before demand loads.

        Override for policies that track access frequency.
        """
        pass

    @abstractmethod
    def contains(self, key: tuple[int, int]) -> bool: ...

    @abstractmethod
    def on_hit(self, key: tuple[int, int]) -> None:
        """Called when a resident expert is accessed (for LRU reordering)."""
        ...

    @abstractmethod
    def insert(self, key: tuple[int, int]) -> None: ...

    @abstractmethod
    def remove(self, key: tuple[int, int]) -> None: ...

    @abstractmethod
    def evict_victim(self, protected: set, current_pos: int
                     ) -> tuple[int, int]: ...

    @abstractmethod
    def occupancy(self) -> int: ...


class LRU(CachePolicy):
    """Least Recently Used eviction."""

    def init_cache(self, initial_cache):
        self._cache = OrderedDict()
        for key in initial_cache:
            self._cache[tuple(key)] = True

    def contains(self, key):
        return key in self._cache

    def on_hit(self, key):
        self._cache.move_to_end(key)

    def insert(self, key):
        self._cache[key] = True

    def remove(self, key):
        del self._cache[key]

    def occupancy(self):
        return len(self._cache)

    def evict_victim(self, protected, current_pos):
        for key in self._cache:
            if key not in protected:
                return key
        return next(iter(self._cache))


class Belady(CachePolicy):
    """Belady's optimal (MIN): evicts the expert with farthest future use.

    Requires full trace knowledge (offline/oracle). This is the theoretical
    lower bound on cache misses.
    """

    def setup(self, activation_trace, cache_size, num_layers, num_experts):
        self._next_use = [[[] for _ in range(num_experts)]
                          for _ in range(num_layers)]
        for step_idx, step_experts in enumerate(activation_trace.steps):
            for layer in range(num_layers):
                for eid in step_experts[layer]:
                    self._next_use[layer][eid].append(
                        step_idx * num_layers + layer)
        self._use_ptr = [[0] * num_experts for _ in range(num_layers)]
        return None

    def init_cache(self, initial_cache):
        self._cache = OrderedDict()
        for key in initial_cache:
            self._cache[tuple(key)] = True

    def contains(self, key):
        return key in self._cache

    def on_hit(self, key):
        pass

    def insert(self, key):
        self._cache[key] = True

    def remove(self, key):
        del self._cache[key]

    def occupancy(self):
        return len(self._cache)

    def _get_next_use(self, layer, eid, current_pos):
        ptr = self._use_ptr[layer][eid]
        uses = self._next_use[layer][eid]
        while ptr < len(uses) and uses[ptr] <= current_pos:
            ptr += 1
        self._use_ptr[layer][eid] = ptr
        return uses[ptr] if ptr < len(uses) else float('inf')

    def evict_victim(self, protected, current_pos):
        best_key, best_dist = None, -1
        for key in self._cache:
            if key in protected:
                continue
            dist = self._get_next_use(key[0], key[1], current_pos)
            if dist > best_dist:
                best_dist, best_key = dist, key
        if best_key is None:
            for key in self._cache:
                dist = self._get_next_use(key[0], key[1], current_pos)
                if dist > best_dist:
                    best_dist, best_key = dist, key
        return best_key


class LFU(CachePolicy):
    """Least Frequently Used eviction with optional windowed reset.

    Args:
        window_size: If set, reset frequency counts every window_size steps.
            None means static LFU (counts never reset).
    """

    def __init__(self, window_size: Optional[int] = None):
        self._window_size = window_size

    def init_cache(self, initial_cache):
        self._cache = set()
        self._freq = {}
        for key in initial_cache:
            key = tuple(key)
            self._cache.add(key)
            self._freq[key] = 0

    def begin_step(self, step_idx):
        if (self._window_size is not None and step_idx > 0 and
                step_idx % self._window_size == 0):
            for k in self._freq:
                self._freq[k] = 0

    def record_access(self, key):
        self._freq[key] = self._freq.get(key, 0) + 1

    def contains(self, key):
        return key in self._cache

    def on_hit(self, key):
        pass

    def insert(self, key):
        self._cache.add(key)
        if key not in self._freq:
            self._freq[key] = 1

    def remove(self, key):
        self._cache.discard(key)

    def occupancy(self):
        return len(self._cache)

    def evict_victim(self, protected, current_pos):
        best_key, best_freq = None, float('inf')
        for key in sorted(self._cache):
            if key in protected:
                continue
            f = self._freq.get(key, 0)
            if f < best_freq:
                best_freq, best_key = f, key
        if best_key is None:
            for key in sorted(self._cache):
                f = self._freq.get(key, 0)
                if f < best_freq:
                    best_freq, best_key = f, key
        return best_key


class StaticFreq(CachePolicy):
    """Global frequency ranking: initial cache = top-k by frequency,
    eviction = lowest global frequency.

    Pre-computes access frequencies over the ENTIRE trace. The top
    (cache_size - num_experts) experts are effectively pinned and never
    evicted. Requires oracle knowledge (full trace).
    """

    def setup(self, activation_trace, cache_size, num_layers, num_experts):
        self._global_freq = {}
        for step_experts in activation_trace.steps:
            for layer in range(num_layers):
                for eid in step_experts[layer]:
                    key = (layer, eid)
                    self._global_freq[key] = self._global_freq.get(key, 0) + 1
        all_experts = [(layer, eid)
                       for layer in range(num_layers)
                       for eid in range(num_experts)]
        all_experts.sort(key=lambda k: (-self._global_freq.get(k, 0), k))
        return all_experts[:cache_size]

    def init_cache(self, initial_cache):
        self._cache = set()
        for key in initial_cache:
            self._cache.add(tuple(key))

    def contains(self, key):
        return key in self._cache

    def on_hit(self, key):
        pass

    def insert(self, key):
        self._cache.add(key)

    def remove(self, key):
        self._cache.discard(key)

    def occupancy(self):
        return len(self._cache)

    def evict_victim(self, protected, current_pos):
        best_key, best_freq = None, float('inf')
        for key in sorted(self._cache):
            if key in protected:
                continue
            f = self._global_freq.get(key, 0)
            if f < best_freq:
                best_freq, best_key = f, key
        if best_key is None:
            for key in sorted(self._cache):
                f = self._global_freq.get(key, 0)
                if f < best_freq:
                    best_freq, best_key = f, key
        return best_key


class StaticScratchpad(CachePolicy):
    """Static pinned experts + FIFO scratchpad for demand/prefetch.

    Like StaticFreq, but reserves `scratchpad_size` empty slots. The top
    `cache_size - scratchpad_size` experts by global frequency are pinned
    and NEVER evicted. All demand loads and prefetches go into the
    scratchpad. Eviction uses FIFO (oldest insertion first) to ensure
    Oracle prefetch cannot increase total IO vs NoPrefetch.

    Args:
        scratchpad_size: Number of slots reserved for dynamic use. Should
            be 2*E (2x experts per layer) so current + next layer's
            non-static experts fit simultaneously.
    """

    def __init__(self, scratchpad_size: int):
        self._scratchpad_size = scratchpad_size

    def setup(self, activation_trace, cache_size, num_layers, num_experts):
        self._global_freq = {}
        for step_experts in activation_trace.steps:
            for layer in range(num_layers):
                for eid in step_experts[layer]:
                    key = (layer, eid)
                    self._global_freq[key] = self._global_freq.get(key, 0) + 1
        all_experts = [(layer, eid)
                       for layer in range(num_layers)
                       for eid in range(num_experts)]
        all_experts.sort(key=lambda k: (-self._global_freq.get(k, 0), k))
        n_pinned = cache_size - self._scratchpad_size
        self._pinned = set(tuple(k) for k in all_experts[:n_pinned])
        return all_experts[:n_pinned]  # scratchpad starts empty

    def init_cache(self, initial_cache):
        self._cache = set()
        self._fifo = OrderedDict()  # FIFO eviction order for scratchpad
        for key in initial_cache:
            self._cache.add(tuple(key))

    def begin_step(self, step_idx):
        # Expire all scratchpad entries — no cross-step reuse
        for key in list(self._fifo):
            self._cache.discard(key)
        self._fifo.clear()

    def contains(self, key):
        return key in self._cache

    def on_hit(self, key):
        pass

    def insert(self, key):
        self._cache.add(key)
        if key not in self._pinned:
            if key in self._fifo:
                self._fifo.move_to_end(key)
            else:
                self._fifo[key] = True

    def remove(self, key):
        self._cache.discard(key)
        self._fifo.pop(key, None)

    def occupancy(self):
        return len(self._cache)

    def evict_victim(self, protected, current_pos):
        # FIFO: evict oldest scratchpad entry that isn't protected
        for key in self._fifo:
            if key not in protected:
                return key
        # Fallback: evict oldest (ignore protected)
        for key in self._fifo:
            return key
        return None


# ── Prefetch Policies ──────────────────────────────────────────────

class PrefetchPolicy:
    """Abstract prefetch strategy. Default: no prefetching."""

    def prefetches_layer0(self) -> bool:
        return False

    def prefetches_next_layer(self) -> bool:
        return False


class NoPrefetch(PrefetchPolicy):
    """No prefetching — demand loads only."""
    pass


class OraclePrefetch(PrefetchPolicy):
    """Oracle: layer-0 prefetch at step start + 1-layer lookahead.

    Requires full trace knowledge. Layer 0 prefetches overlap with
    embed + stages 1-4a. Layer L+1 prefetches overlap with stage4b(L)
    + stages 1-4a(L+1).

    Args:
        max_per_layer: Maximum prefetches to issue per layer. None means
            unlimited (prefetch all misses). When set, only the first
            `max_per_layer` non-resident experts are prefetched; the rest
            become demand loads. This limits eviction pressure from
            prefetching when transfers cannot be fully hidden.
    """

    def __init__(self, max_per_layer: Optional[int] = None):
        self._max_per_layer = max_per_layer

    @property
    def max_per_layer(self) -> Optional[int]:
        return self._max_per_layer

    def prefetches_layer0(self):
        return True

    def prefetches_next_layer(self):
        return True


# ── Simulation ─────────────────────────────────────────────────────

def _default_initial_cache(cache_size: int, num_layers: int,
                           num_experts: int) -> list[tuple[int, int]]:
    """Default initial cache: fill uniformly across layers."""
    initial = []
    for eid in range(num_experts):
        for layer in range(num_layers):
            if len(initial) >= cache_size:
                return initial
            initial.append((layer, eid))
    return initial


def _get_scheduling(activation_trace: ActivationTrace,
                     step_idx: int) -> 'StepScheduling | None':
    if activation_trace.scheduling is not None:
        return activation_trace.scheduling[step_idx]
    return None


def _make_layer_trace(needed: list[int],
                      prefetches: list[TransferEvent],
                      demand_loads: list[TransferEvent]) -> LayerTrace:
    n = len(needed)
    weights = [1.0 / n] * n if n > 0 else []
    return LayerTrace(
        topk_ids=[needed],
        topk_weights=[weights],
        prefetches=prefetches,
        demand_loads=demand_loads,
    )


def _do_evict(cache_policy: CachePolicy, cache_size: int,
              protected: set, current_pos: int):
    """Evict one expert if cache is full. Return evict key or None."""
    if cache_policy.occupancy() >= cache_size:
        victim = cache_policy.evict_victim(protected, current_pos)
        cache_policy.remove(victim)
        return victim
    return None


def simulate(cache_policy: CachePolicy,
             prefetch_policy: PrefetchPolicy,
             activation_trace: ActivationTrace,
             cache_size: int,
             initial_cache: Optional[list[tuple[int, int]]] = None
             ) -> DataMovementTrace:
    """Run a cache+prefetch policy simulation on an activation trace.

    Args:
        cache_policy: Eviction strategy (LRU, Belady, LFU, StaticFreq).
        prefetch_policy: Prefetch strategy (NoPrefetch, OraclePrefetch).
        activation_trace: Expert access pattern.
        cache_size: Total unified cache capacity (expert slots shared
            across all layers).
        initial_cache: Optional explicit initial cache contents.

    Returns:
        DataMovementTrace ready for replay.
    """
    num_layers = activation_trace.num_layers
    num_experts = activation_trace.num_experts

    custom_initial = cache_policy.setup(
        activation_trace, cache_size, num_layers, num_experts)
    if initial_cache is None:
        initial_cache = (custom_initial if custom_initial is not None
                         else _default_initial_cache(
                             cache_size, num_layers, num_experts))

    cache_policy.init_cache(initial_cache)

    do_pf_layer0 = prefetch_policy.prefetches_layer0()
    do_pf_next = prefetch_policy.prefetches_next_layer()
    pf_max = (prefetch_policy.max_per_layer
              if hasattr(prefetch_policy, 'max_per_layer') else None)

    steps = []
    for step_idx, step_experts in enumerate(activation_trace.steps):
        cache_policy.begin_step(step_idx)
        layer_traces = []
        pending_prefetches = [[] for _ in range(num_layers)]

        # Layer 0 prefetch: issued at start of step, overlaps with
        # embed + stages 1-4a(layer 0).
        if do_pf_layer0:
            layer0_needed = step_experts[0]
            layer0_pos = step_idx * num_layers
            pf0_protected = {(0, eid) for eid in layer0_needed}
            pf0_count = 0
            for eid in layer0_needed:
                key = (0, eid)
                if not cache_policy.contains(key):
                    if pf_max is not None and pf0_count >= pf_max:
                        break
                    evict = _do_evict(cache_policy, cache_size,
                                      pf0_protected, layer0_pos)
                    cache_policy.insert(key)
                    pending_prefetches[0].append(
                        TransferEvent(target=(0, eid), evict=evict))
                    pf0_count += 1

        for layer in range(num_layers):
            current_pos = step_idx * num_layers + layer
            needed = step_experts[layer]
            protected = {(layer, eid) for eid in needed}

            # Record accesses for all needed experts (for freq policies)
            for eid in needed:
                cache_policy.record_access((layer, eid))

            # Demand loads for current layer
            demand_loads = []
            for eid in needed:
                key = (layer, eid)
                if cache_policy.contains(key):
                    cache_policy.on_hit(key)
                else:
                    evict = _do_evict(cache_policy, cache_size,
                                      protected, current_pos)
                    cache_policy.insert(key)
                    demand_loads.append(
                        TransferEvent(target=(layer, eid), evict=evict))

            # Lookahead prefetch for layer+1
            if do_pf_next and layer + 1 < num_layers:
                next_needed = step_experts[layer + 1]
                pf_protected = (protected |
                                {(layer + 1, e) for e in next_needed})
                pf_count = 0
                for eid in next_needed:
                    key = (layer + 1, eid)
                    if not cache_policy.contains(key):
                        if pf_max is not None and pf_count >= pf_max:
                            break
                        evict = _do_evict(
                            cache_policy, cache_size, pf_protected,
                            step_idx * num_layers + layer + 1)
                        cache_policy.insert(key)
                        pending_prefetches[layer + 1].append(
                            TransferEvent(target=(layer + 1, eid),
                                          evict=evict))
                        pf_count += 1

            layer_traces.append(_make_layer_trace(
                needed,
                prefetches=pending_prefetches[layer],
                demand_loads=demand_loads))

        steps.append(StepTrace(
            layers=layer_traces,
            scheduling=_get_scheduling(activation_trace, step_idx)))

    return DataMovementTrace(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        initial_cache_state=list(initial_cache),
        steps=steps,
    )
