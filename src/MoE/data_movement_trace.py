"""Data movement trace format for MoE expert offloading replay.

Defines the structured trace format that encodes prefetch, eviction, and
demand-load decisions produced by cache policy simulators (Phase 7) and
consumed by the ReplayController (Phase 6).

Uses a unified expert cache shared across all layers (keyed by (layer, eid)
pairs) with a single cache_size budget. Any expert can evict any other
expert regardless of layer.

Also defines ActivationTrace — a structured view of the flat expert access
trace recorded by ExpertOffloadEngine.save_trace(), used as input to
policy simulators.

Data flow:
    ExpertOffloadEngine.save_trace()  →  ActivationTrace (input to policies)
    EvictionPolicy.simulate()         →  DataMovementTrace (input to replay)
    ReplayController(trace)           →  actual GPU execution with timing
"""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransferEvent:
    """A single CPU→GPU expert transfer.

    Attributes:
        target: (layer, expert_id) — which expert to load onto GPU.
        evict:  (layer, expert_id) — which resident expert to evict to
                free a cache slot, or None when the cache has a free slot
                (no eviction needed). In the unified cache, evict can
                target any (layer, expert_id) regardless of the target's
                layer.
    """
    target: tuple[int, int]
    evict: Optional[tuple[int, int]] = None

    def to_dict(self) -> dict:
        d = {'target': list(self.target)}
        if self.evict is not None:
            d['evict'] = list(self.evict)
        return d

    @staticmethod
    def from_dict(d: dict) -> 'TransferEvent':
        target = tuple(d['target'])
        evict = tuple(d['evict']) if 'evict' in d else None
        return TransferEvent(target=target, evict=evict)


@dataclass
class LayerTrace:
    """Per-layer data movement schedule for one decode step.

    Attributes:
        topk_ids:      Expert selections per token — [n_tokens][top_k].
        topk_weights:  Routing weights per token — [n_tokens][top_k].
        prefetches:    Async transfers that bring this layer's experts onto GPU.
                       For layer 0: issued before stage1 (start of network).
                       For layer L>0: issued before stage4b of layer L-1,
                       overlapping with stage4b(L-1) + stages 1-4a(L).
        demand_loads:  Blocking transfers before stage4b. Cache misses not
                       covered by prefetches.
    """
    topk_ids: list[list[int]]
    topk_weights: list[list[float]]
    prefetches: list[TransferEvent] = field(default_factory=list)
    demand_loads: list[TransferEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'topk_ids': self.topk_ids,
            'topk_weights': self.topk_weights,
            'prefetches': [e.to_dict() for e in self.prefetches],
            'demand_loads': [e.to_dict() for e in self.demand_loads],
        }

    @staticmethod
    def from_dict(d: dict) -> 'LayerTrace':
        return LayerTrace(
            topk_ids=d['topk_ids'],
            topk_weights=d['topk_weights'],
            prefetches=[TransferEvent.from_dict(e) for e in d['prefetches']],
            demand_loads=[TransferEvent.from_dict(e)
                          for e in d['demand_loads']],
        )


@dataclass
class StepTrace:
    """Per-step (per-decode-token) data movement schedule.

    Attributes:
        layers: One LayerTrace per transformer layer (e.g. 32 for Mixtral).
    """
    layers: list[LayerTrace]

    def to_dict(self) -> dict:
        return {'layers': [lt.to_dict() for lt in self.layers]}

    @staticmethod
    def from_dict(d: dict) -> 'StepTrace':
        return StepTrace(
            layers=[LayerTrace.from_dict(lt) for lt in d['layers']])


@dataclass
class DataMovementTrace:
    """Complete data movement trace for an entire decode sequence.

    Produced by policy simulators (Phase 7), consumed by ReplayController
    (Phase 6). Encodes every prefetch, eviction, and demand-load decision
    needed to replay expert offloading on real GPU hardware.

    Uses a unified expert cache shared across all layers. Any cache slot
    can hold any (layer, expert_id) pair.

    Attributes:
        num_layers:          Number of transformer layers.
        num_experts:         Number of experts per MoE layer.
        cache_size:          Total unified GPU cache capacity (number of
                             expert slots shared across all layers).
        initial_cache_state: [(layer, expert_id)] experts on GPU at start.
        steps:               One StepTrace per decode token.
    """
    num_layers: int
    num_experts: int
    cache_size: int
    initial_cache_state: list[tuple[int, int]]
    steps: list[StepTrace]

    def save(self, path: str):
        """Serialize to JSON."""
        data = {
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
            'cache_size': self.cache_size,
            'initial_cache_state': [list(t) for t in self.initial_cache_state],
            'steps': [s.to_dict() for s in self.steps],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"DataMovementTrace saved: {len(self.steps)} steps, "
              f"{path}")

    @staticmethod
    def load(path: str) -> 'DataMovementTrace':
        """Deserialize from JSON."""
        with open(path) as f:
            data = json.load(f)
        trace = DataMovementTrace(
            num_layers=data['num_layers'],
            num_experts=data['num_experts'],
            cache_size=data['cache_size'],
            initial_cache_state=[tuple(t)
                                 for t in data['initial_cache_state']],
            steps=[StepTrace.from_dict(s) for s in data['steps']],
        )
        print(f"DataMovementTrace loaded: {len(trace.steps)} steps, "
              f"{path}")
        return trace

    def validate(self) -> list[str]:
        """Check internal consistency of the unified cache trace.

        Verifies that every expert needed by topk_ids is resident in the
        unified cache (from initial_cache_state + prior transfers) or
        loaded by a prefetch/demand_load in that layer. Also checks that
        total cache occupancy never exceeds cache_size.

        Returns:
            List of error strings (empty = valid).
        """
        errors = []

        # Build initial residency: unified set of (layer, eid)
        resident = set()
        for (layer, eid) in self.initial_cache_state:
            resident.add((layer, eid))

        # Check initial capacity
        if len(resident) > self.cache_size:
            errors.append(
                f"Initial state: {len(resident)} experts but "
                f"cache_size is {self.cache_size}")

        for step_idx, step in enumerate(self.steps):
            if len(step.layers) != self.num_layers:
                errors.append(
                    f"Step {step_idx}: expected {self.num_layers} layers, "
                    f"got {len(step.layers)}")
                continue

            for layer_idx, lt in enumerate(step.layers):
                all_transfers = lt.prefetches + lt.demand_loads

                # Apply transfers to unified cache
                for te in all_transfers:
                    if te.evict is not None:
                        resident.discard(tuple(te.evict))
                    resident.add(tuple(te.target))

                # Check total capacity after transfers
                if len(resident) > self.cache_size:
                    errors.append(
                        f"Step {step_idx}, layer {layer_idx}: "
                        f"cache has {len(resident)} experts, "
                        f"cache_size is {self.cache_size}")

                # Check coverage: every needed expert is available
                needed = set()
                for token_experts in lt.topk_ids:
                    needed.update(token_experts)

                for eid in needed:
                    if (layer_idx, eid) not in resident:
                        errors.append(
                            f"Step {step_idx}, layer {layer_idx}: "
                            f"expert {eid} needed but not resident or "
                            f"loaded")

        return errors

    def summary(self) -> dict:
        """Compute summary statistics."""
        total_prefetches = 0
        total_demand_loads = 0
        total_evictions = 0
        for step in self.steps:
            for lt in step.layers:
                total_prefetches += len(lt.prefetches)
                total_demand_loads += len(lt.demand_loads)
                for te in (lt.prefetches + lt.demand_loads):
                    if te.evict is not None:
                        total_evictions += 1
        total_transfers = total_prefetches + total_demand_loads
        return {
            'steps': len(self.steps),
            'total_transfers': total_transfers,
            'prefetches': total_prefetches,
            'demand_loads': total_demand_loads,
            'evictions': total_evictions,
        }


@dataclass
class ActivationTrace:
    """Structured expert activation trace — input to policy simulators.

    Converted from ExpertOffloadEngine's flat trace format
    ({step, layer, expert_ids} entries).

    Attributes:
        num_layers:  Number of transformer layers.
        num_experts: Number of experts per MoE layer.
        steps:       steps[step_idx][layer_idx] = sorted list of expert_ids
                     accessed at that (step, layer).
    """
    num_layers: int
    num_experts: int
    steps: list[list[list[int]]]

    @staticmethod
    def from_flat_trace(trace_data: dict) -> 'ActivationTrace':
        """Convert ExpertOffloadEngine's flat trace format to structured.

        Args:
            trace_data: Dict with keys 'num_layers', 'num_experts', 'trace'
                where trace is a list of {step, layer, expert_ids} dicts.

        Returns:
            ActivationTrace with steps[step][layer] = [expert_ids].
        """
        num_layers = trace_data['num_layers']
        num_experts = trace_data['num_experts']
        flat = trace_data['trace']

        if not flat:
            return ActivationTrace(num_layers, num_experts, [])

        max_step = max(entry['step'] for entry in flat)
        steps = [[[] for _ in range(num_layers)]
                 for _ in range(max_step + 1)]

        for entry in flat:
            steps[entry['step']][entry['layer']] = entry['expert_ids']

        return ActivationTrace(num_layers, num_experts, steps)

    @staticmethod
    def load(path: str) -> 'ActivationTrace':
        """Load from JSON file saved by ExpertOffloadEngine.save_trace()."""
        with open(path) as f:
            data = json.load(f)
        return ActivationTrace.from_flat_trace(data)

    def save(self, path: str):
        """Save in the flat format compatible with ExpertOffloadEngine."""
        flat_trace = []
        for step_idx, step_layers in enumerate(self.steps):
            for layer_idx, expert_ids in enumerate(step_layers):
                if expert_ids:
                    flat_trace.append({
                        'step': step_idx,
                        'layer': layer_idx,
                        'expert_ids': expert_ids,
                    })
        data = {
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
            'trace': flat_trace,
            'transfers': [],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def num_steps(self) -> int:
        return len(self.steps)
