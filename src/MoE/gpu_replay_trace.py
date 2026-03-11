"""GPU Replay trace format for MoE expert offloading replay.

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
    EvictionPolicy.simulate()         →  GPUReplayTrace (input to replay)
    ReplayController(trace)           →  actual GPU execution with timing
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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
class RequestScheduling:
    """Per-request metadata within a scheduling step."""
    request_id: int
    conversation_id: str
    seq_len: int
    is_prefill: bool
    prefill_chunk_start: int = 0
    prefill_chunk_length: int = 0
    is_continuation: bool = False

    def to_dict(self) -> dict:
        return {
            'request_id': self.request_id,
            'conversation_id': self.conversation_id,
            'seq_len': self.seq_len,
            'is_prefill': self.is_prefill,
            'prefill_chunk_start': self.prefill_chunk_start,
            'prefill_chunk_length': self.prefill_chunk_length,
            'is_continuation': self.is_continuation,
        }

    @staticmethod
    def from_dict(d: dict) -> 'RequestScheduling':
        return RequestScheduling(
            request_id=d['request_id'],
            conversation_id=d['conversation_id'],
            seq_len=d['seq_len'],
            is_prefill=d['is_prefill'],
            prefill_chunk_start=d.get('prefill_chunk_start', 0),
            prefill_chunk_length=d.get('prefill_chunk_length', 0),
            is_continuation=d.get('is_continuation', False),
        )


@dataclass
class StepScheduling:
    """Per-step scheduling metadata from the continuous batching simulator.

    Captures batch composition and scheduling events for each step. Used by the
    replay loop to manage KV cache and batch composition during trace replay.

    Events happen in this order within a step:
      1. 'complete' — requests that finished in the previous step are removed
      2. 'admit' / 'force_admit' — new requests from waiting queue
      3. [computation happens with active_requests]

    With the no-preemption policy (full-sequence page pre-allocation), preempt
    and readmit events never occur.

    Attributes:
        step:             Global step index.
        batch_size:       Number of active requests in this step.
        active_requests:  Per-request metadata (id, seq_len, prefill/continuation status).
        events:           Scheduling events for this step.
    """
    step: int
    batch_size: int
    active_requests: list[RequestScheduling]
    events: list[dict]

    def to_dict(self) -> dict:
        return {
            'step': self.step,
            'batch_size': self.batch_size,
            'active_requests': [r.to_dict() for r in self.active_requests],
            'events': self.events,
        }

    @staticmethod
    def from_dict(d: dict) -> 'StepScheduling':
        return StepScheduling(
            step=d['step'],
            batch_size=d['batch_size'],
            active_requests=[RequestScheduling.from_dict(r)
                             for r in d['active_requests']],
            events=d.get('events', []),
        )


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
        layers:     One LayerTrace per transformer layer (e.g. 32 for Mixtral).
        scheduling: Optional scheduling metadata from batch simulator.
    """
    layers: list[LayerTrace]
    scheduling: Optional[StepScheduling] = None

    def to_dict(self) -> dict:
        d = {'layers': [lt.to_dict() for lt in self.layers]}
        if self.scheduling is not None:
            d['scheduling'] = self.scheduling.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> 'StepTrace':
        scheduling = None
        if 'scheduling' in d:
            scheduling = StepScheduling.from_dict(d['scheduling'])
        return StepTrace(
            layers=[LayerTrace.from_dict(lt) for lt in d['layers']],
            scheduling=scheduling)


@dataclass
class GPUReplayTrace:
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
        print(f"GPUReplayTrace saved: {len(self.steps)} steps, "
              f"{path}")

    @staticmethod
    def load(path: str) -> 'GPUReplayTrace':
        """Deserialize from JSON."""
        with open(path) as f:
            data = json.load(f)
        trace = GPUReplayTrace(
            num_layers=data['num_layers'],
            num_experts=data['num_experts'],
            cache_size=data['cache_size'],
            initial_cache_state=[tuple(t)
                                 for t in data['initial_cache_state']],
            steps=[StepTrace.from_dict(s) for s in data['steps']],
        )
        print(f"GPUReplayTrace loaded: {len(trace.steps)} steps, "
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
        ics_tuples = [tuple(x) for x in self.initial_cache_state]
        if len(ics_tuples) != len(set(ics_tuples)):
            errors.append("initial_cache_state contains duplicates")
        resident = set(ics_tuples)

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
                        resident.remove(tuple(te.evict))
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
        num_layers:      Number of transformer layers.
        num_experts:     Number of experts per MoE layer.
        steps:           steps[step_idx][layer_idx] = sorted list of expert_ids
                         accessed at that (step, layer).
        router_inputs:   Optional path to a companion .npz file containing
                         router inputs (post-layernorm hidden states) keyed
                         as 'step{s}_layer{l}' with shape [n_tokens, hidden_dim]
                         in float16. None when not recorded.
        scheduling:      Optional per-step scheduling metadata from
                         trace_utils.py's continuous batching simulator.
    """
    num_layers: int
    num_experts: int
    steps: list[list[list[int]]]
    router_inputs: Optional[str] = None
    scheduling: Optional[list[StepScheduling]] = None
    scheduling_config: Optional[dict] = None
    first_moe_layer: int = 0  # layers < this are dense (no experts)

    @staticmethod
    def from_flat_trace(trace_data: dict,
                        router_inputs_path: Optional[str] = None
                        ) -> 'ActivationTrace':
        """Convert ExpertOffloadEngine's flat trace format to structured.

        Args:
            trace_data: Dict with keys 'num_layers', 'num_experts', 'trace'
                where trace is a list of {step, layer, expert_ids} dicts.
                Optionally includes 'step_scheduling' from trace_utils.py.
            router_inputs_path: Optional path to companion .npz file with
                router inputs.

        Returns:
            ActivationTrace with steps[step][layer] = [expert_ids].
        """
        num_layers = trace_data['num_layers']
        num_experts = trace_data['num_experts']
        flat = trace_data['trace']

        if not flat:
            return ActivationTrace(num_layers, num_experts, [],
                                   router_inputs=router_inputs_path,
                                   scheduling_config=trace_data.get(
                                       'scheduling', None),
                                   first_moe_layer=trace_data.get(
                                       'first_moe_layer', 0))

        max_step = max(entry['step'] for entry in flat)
        steps = [[[] for _ in range(num_layers)]
                 for _ in range(max_step + 1)]

        for entry in flat:
            steps[entry['step']][entry['layer']] = entry['expert_ids']

        # Parse scheduling metadata if present
        scheduling = None
        if 'step_scheduling' in trace_data:
            scheduling = [StepScheduling.from_dict(s)
                          for s in trace_data['step_scheduling']]

        scheduling_config = trace_data.get('scheduling', None)

        first_moe_layer = trace_data.get('first_moe_layer', 0)

        return ActivationTrace(num_layers, num_experts, steps,
                               router_inputs=router_inputs_path,
                               scheduling=scheduling,
                               scheduling_config=scheduling_config,
                               first_moe_layer=first_moe_layer)

    @staticmethod
    def load(path: str) -> 'ActivationTrace':
        """Load from JSON file saved by ExpertOffloadEngine.save_trace().

        Automatically detects a companion router inputs file
        (<basename>_router_inputs.npz) if present.
        """
        with open(path) as f:
            data = json.load(f)
        # Check for companion router inputs file
        ri_path = os.path.splitext(path)[0] + '_router_inputs.npz'
        ri_path = ri_path if os.path.exists(ri_path) else None
        return ActivationTrace.from_flat_trace(data,
                                               router_inputs_path=ri_path)

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
        if self.first_moe_layer != 0:
            data['first_moe_layer'] = self.first_moe_layer
        if self.scheduling is not None:
            data['step_scheduling'] = [s.to_dict() for s in self.scheduling]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def has_router_inputs(self) -> bool:
        """Check if router inputs are available."""
        return self.router_inputs is not None

    def get_router_input(self, step: int, layer: int) -> Optional[np.ndarray]:
        """Load a single router input array from the companion .npz file.

        Args:
            step: decode step index
            layer: layer index

        Returns:
            float16 numpy array of shape [n_tokens, hidden_dim], or None
            if router inputs are not available.
        """
        if self.router_inputs is None:
            return None
        # Use lazy loading to avoid reading entire file into memory
        key = f"step{step}_layer{layer}"
        with np.load(self.router_inputs) as npz:
            if key in npz:
                return npz[key]
        return None

    def num_steps(self) -> int:
        return len(self.steps)
