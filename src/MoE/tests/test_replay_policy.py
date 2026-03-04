"""Tests for Phases 5-7: data movement trace, policy simulation, replay.

Test categories:
  1. DataMovementTrace serialization round-trip
  2. DataMovementTrace.validate() correctness
  3. ActivationTrace.from_flat_trace() conversion
  4. LRU policy: eviction order, miss counts
  5. Oracle policy: optimality vs LRU
  6. Frequency policy: LFU eviction
  7. PreGated policy: prefetch generation
  8. All policies: validate() passes on output

Run:
    cd GPU_Offloading/src/MoE && python -m pytest tests/test_replay_policy.py -v
"""

import json
import os
import sys
import tempfile

import pytest

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_movement_trace import (
    ActivationTrace,
    DataMovementTrace,
    LayerTrace,
    StepTrace,
    TransferEvent,
)
from policy_simulator import (
    FrequencyPolicy,
    LRUPolicy,
    OraclePolicy,
    PreGatedPolicy,
)


# ── Fixtures ─────────────────────────────────────────────────────────

def make_simple_activation_trace():
    """3 layers, 4 experts, 5 steps. Known access pattern.

    Step 0: all layers use [0, 1]
    Step 1: all layers use [2, 3]
    Step 2: all layers use [0, 2]
    Step 3: all layers use [1, 3]
    Step 4: all layers use [0, 3]
    """
    steps = []
    patterns = [[0, 1], [2, 3], [0, 2], [1, 3], [0, 3]]
    for pattern in patterns:
        steps.append([pattern for _ in range(3)])
    return ActivationTrace(num_layers=3, num_experts=4, steps=steps)


def make_flat_trace_dict():
    """ExpertOffloadEngine-compatible flat trace dict."""
    return {
        'num_layers': 2,
        'num_experts': 4,
        'trace': [
            {'step': 0, 'layer': 0, 'expert_ids': [0, 1]},
            {'step': 0, 'layer': 1, 'expert_ids': [2, 3]},
            {'step': 1, 'layer': 0, 'expert_ids': [1, 2]},
            {'step': 1, 'layer': 1, 'expert_ids': [0, 3]},
        ],
        'transfers': [
            {'step': 0, 'layer': 0, 'expert_id': 0,
             'bytes': 1000, 'time_ms': 1.0},
        ],
    }


# ── 1. Serialization round-trip ──────────────────────────────────────

class TestSerialization:

    def test_transfer_event_round_trip(self):
        te = TransferEvent(target=(2, 5), evict=(2, 1))
        d = te.to_dict()
        te2 = TransferEvent.from_dict(d)
        assert te2.target == (2, 5)
        assert te2.evict == (2, 1)

    def test_transfer_event_no_evict(self):
        te = TransferEvent(target=(0, 3))
        d = te.to_dict()
        assert 'evict' not in d
        te2 = TransferEvent.from_dict(d)
        assert te2.evict is None

    def test_data_movement_trace_save_load(self, tmp_path):
        trace = DataMovementTrace(
            num_layers=2,
            num_experts=4,
            experts_per_layer=2,
            initial_cache_state=[(0, 0), (0, 1), (1, 0), (1, 1)],
            steps=[
                StepTrace(layers=[
                    LayerTrace(
                        topk_ids=[[0, 2]],
                        topk_weights=[[0.6, 0.4]],
                        prefetches=[],
                        post_routing_prefetches=[],
                        demand_loads=[TransferEvent(
                            target=(0, 2), evict=(0, 1))],
                    ),
                    LayerTrace(
                        topk_ids=[[1, 3]],
                        topk_weights=[[0.5, 0.5]],
                        prefetches=[TransferEvent(
                            target=(1, 3), evict=(1, 0))],
                        post_routing_prefetches=[],
                        demand_loads=[],
                    ),
                ]),
            ],
        )

        path = str(tmp_path / "trace.json")
        trace.save(path)
        loaded = DataMovementTrace.load(path)

        assert loaded.num_layers == 2
        assert loaded.num_experts == 4
        assert loaded.experts_per_layer == 2
        assert len(loaded.steps) == 1
        assert loaded.initial_cache_state == [(0, 0), (0, 1), (1, 0), (1, 1)]

        lt0 = loaded.steps[0].layers[0]
        assert lt0.topk_ids == [[0, 2]]
        assert len(lt0.demand_loads) == 1
        assert lt0.demand_loads[0].target == (0, 2)
        assert lt0.demand_loads[0].evict == (0, 1)

        lt1 = loaded.steps[0].layers[1]
        assert len(lt1.prefetches) == 1
        assert lt1.prefetches[0].target == (1, 3)


# ── 2. DataMovementTrace.validate() ─────────────────────────────────

class TestValidation:

    def test_valid_trace_passes(self):
        """A correctly constructed trace should validate with no errors."""
        trace = DataMovementTrace(
            num_layers=1,
            num_experts=4,
            experts_per_layer=2,
            initial_cache_state=[(0, 0), (0, 1)],
            steps=[
                StepTrace(layers=[
                    LayerTrace(
                        topk_ids=[[0, 2]],
                        topk_weights=[[0.5, 0.5]],
                        demand_loads=[TransferEvent(
                            target=(0, 2), evict=(0, 1))],
                    ),
                ]),
                StepTrace(layers=[
                    LayerTrace(
                        topk_ids=[[0, 2]],
                        topk_weights=[[0.5, 0.5]],
                    ),
                ]),
            ],
        )
        errors = trace.validate()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_expert_detected(self):
        """Expert needed but not resident and not loaded → error."""
        trace = DataMovementTrace(
            num_layers=1,
            num_experts=4,
            experts_per_layer=2,
            initial_cache_state=[(0, 0), (0, 1)],
            steps=[
                StepTrace(layers=[
                    LayerTrace(
                        topk_ids=[[0, 3]],  # expert 3 not loaded!
                        topk_weights=[[0.5, 0.5]],
                        demand_loads=[],
                    ),
                ]),
            ],
        )
        errors = trace.validate()
        assert len(errors) == 1
        assert 'expert 3' in errors[0]

    def test_cross_layer_eviction_detected(self):
        """evict.layer != target.layer → error."""
        trace = DataMovementTrace(
            num_layers=2,
            num_experts=4,
            experts_per_layer=2,
            initial_cache_state=[(0, 0), (0, 1), (1, 0), (1, 1)],
            steps=[
                StepTrace(layers=[
                    LayerTrace(
                        topk_ids=[[0, 2]],
                        topk_weights=[[0.5, 0.5]],
                        demand_loads=[TransferEvent(
                            target=(0, 2), evict=(1, 0))],  # cross-layer!
                    ),
                    LayerTrace(
                        topk_ids=[[0, 1]],
                        topk_weights=[[0.5, 0.5]],
                    ),
                ]),
            ],
        )
        errors = trace.validate()
        assert any('evict layer' in e for e in errors)

    def test_initial_overcapacity_detected(self):
        """Initial state exceeds capacity → error."""
        trace = DataMovementTrace(
            num_layers=1,
            num_experts=4,
            experts_per_layer=2,
            initial_cache_state=[(0, 0), (0, 1), (0, 2)],  # 3 > capacity 2
            steps=[],
        )
        errors = trace.validate()
        assert any('layer 0' in e and '3' in e for e in errors)


# ── 3. ActivationTrace conversion ───────────────────────────────────

class TestActivationTrace:

    def test_from_flat_trace(self):
        flat = make_flat_trace_dict()
        at = ActivationTrace.from_flat_trace(flat)
        assert at.num_layers == 2
        assert at.num_experts == 4
        assert len(at.steps) == 2
        assert at.steps[0][0] == [0, 1]
        assert at.steps[0][1] == [2, 3]
        assert at.steps[1][0] == [1, 2]
        assert at.steps[1][1] == [0, 3]

    def test_save_load_round_trip(self, tmp_path):
        at = ActivationTrace(
            num_layers=2, num_experts=4,
            steps=[[[0, 1], [2, 3]], [[1, 2], [0, 3]]])
        path = str(tmp_path / "act_trace.json")
        at.save(path)
        at2 = ActivationTrace.load(path)
        assert at2.num_layers == 2
        assert at2.steps == at.steps

    def test_empty_trace(self):
        flat = {'num_layers': 2, 'num_experts': 4, 'trace': [],
                'transfers': []}
        at = ActivationTrace.from_flat_trace(flat)
        assert len(at.steps) == 0


# ── 4. LRU Policy ───────────────────────────────────────────────────

class TestLRUPolicy:

    def test_no_misses_when_all_resident(self):
        """If cache fits all experts, no demand loads."""
        at = make_simple_activation_trace()
        policy = LRUPolicy()
        dm = policy.simulate(at, experts_per_layer=4)
        for step in dm.steps:
            for lt in step.layers:
                assert lt.demand_loads == []

    def test_eviction_order(self):
        """With capacity=2, verify LRU eviction on known pattern."""
        at = ActivationTrace(
            num_layers=1, num_experts=4,
            steps=[
                [[0, 1]],   # cache: {0, 1}
                [[2, 3]],   # miss 2,3; evict 0 (LRU), then 1 → cache: {2,3}
                [[0, 1]],   # miss 0,1; evict 2 (LRU), then 3 → cache: {0,1}
            ])
        policy = LRUPolicy()
        dm = policy.simulate(at, experts_per_layer=2)

        # Step 0: no misses (0,1 are initial)
        assert dm.steps[0].layers[0].demand_loads == []

        # Step 1: 2 demand loads, evicting 0 then 1
        loads1 = dm.steps[1].layers[0].demand_loads
        assert len(loads1) == 2
        assert loads1[0].target == (0, 2)
        assert loads1[0].evict == (0, 0)  # LRU is 0
        assert loads1[1].target == (0, 3)
        assert loads1[1].evict == (0, 1)  # then 1

        # Step 2: 2 demand loads, evicting 2 then 3
        loads2 = dm.steps[2].layers[0].demand_loads
        assert len(loads2) == 2
        assert loads2[0].target == (0, 0)
        assert loads2[0].evict == (0, 2)
        assert loads2[1].target == (0, 1)
        assert loads2[1].evict == (0, 3)

    def test_validate_passes(self):
        at = make_simple_activation_trace()
        policy = LRUPolicy()
        dm = policy.simulate(at, experts_per_layer=2)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_no_prefetches(self):
        """LRU should never generate prefetches."""
        at = make_simple_activation_trace()
        dm = LRUPolicy().simulate(at, experts_per_layer=2)
        for step in dm.steps:
            for lt in step.layers:
                assert lt.prefetches == []
                assert lt.post_routing_prefetches == []


# ── 5. Oracle Policy ────────────────────────────────────────────────

class TestOraclePolicy:

    def test_fewer_misses_than_lru(self):
        """Oracle should have <= misses than LRU on non-trivial traces."""
        at = make_simple_activation_trace()
        lru_dm = LRUPolicy().simulate(at, experts_per_layer=2)
        oracle_dm = OraclePolicy().simulate(at, experts_per_layer=2)

        lru_misses = sum(
            len(lt.demand_loads)
            for step in lru_dm.steps for lt in step.layers)
        oracle_misses = sum(
            len(lt.demand_loads)
            for step in oracle_dm.steps for lt in step.layers)

        assert oracle_misses <= lru_misses, \
            f"Oracle ({oracle_misses}) should not have more misses " \
            f"than LRU ({lru_misses})"

    def test_generates_prefetches(self):
        """Oracle should generate lookahead prefetches for next layer."""
        at = ActivationTrace(
            num_layers=3, num_experts=4,
            steps=[
                [[0, 1], [2, 3], [0, 2]],  # layer 0 can prefetch for layer 1
            ])
        dm = OraclePolicy().simulate(at, experts_per_layer=2)

        # Layer 0 should have prefetches for layer 1's needs (2, 3)
        # which are not in layer 1's initial cache (0, 1)
        has_prefetches = any(
            len(lt.prefetches) > 0
            for step in dm.steps for lt in step.layers)
        assert has_prefetches, "Oracle should generate prefetches"

    def test_validate_passes(self):
        at = make_simple_activation_trace()
        dm = OraclePolicy().simulate(at, experts_per_layer=2)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"


# ── 6. Frequency Policy ─────────────────────────────────────────────

class TestFrequencyPolicy:

    def test_evicts_least_frequent(self):
        """Expert with fewest accesses should be evicted."""
        # Expert 0 accessed every step, expert 1 only once
        at = ActivationTrace(
            num_layers=1, num_experts=4,
            steps=[
                [[0, 1]],   # both accessed once
                [[0, 2]],   # 0 accessed again, need 2 → evict 1 (freq=1)
            ])
        dm = FrequencyPolicy().simulate(at, experts_per_layer=2)

        loads = dm.steps[1].layers[0].demand_loads
        assert len(loads) == 1
        assert loads[0].target == (0, 2)
        assert loads[0].evict == (0, 1)  # 1 has lower freq than 0

    def test_windowed_reset(self):
        """Windowed mode resets counts periodically."""
        at = ActivationTrace(
            num_layers=1, num_experts=4,
            steps=[
                [[0, 1]],   # freq: 0→1, 1→1
                [[0, 1]],   # freq: 0→2, 1→2
                [[0, 2]],   # window reset at step 2 → freq: 0→1, 2→1; evict 1 (freq=0 after reset)
            ])
        dm = FrequencyPolicy(window_size=2).simulate(
            at, experts_per_layer=2)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_validate_passes(self):
        at = make_simple_activation_trace()
        dm = FrequencyPolicy().simulate(at, experts_per_layer=2)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"


# ── 7. PreGated Policy ──────────────────────────────────────────────

class TestPreGatedPolicy:

    def test_generates_prefetches(self):
        """PreGated should produce prefetches for next layer's needs."""
        at = ActivationTrace(
            num_layers=3, num_experts=4,
            steps=[
                [[0, 1], [2, 3], [0, 2]],
            ])
        dm = PreGatedPolicy(base_policy_type='lru').simulate(
            at, experts_per_layer=2)

        total_prefetches = sum(
            len(lt.prefetches)
            for step in dm.steps for lt in step.layers)
        assert total_prefetches > 0, \
            "PreGated should generate prefetches"

    def test_reduces_demand_loads_vs_lru(self):
        """PreGated(LRU) should have same or fewer demand loads than LRU.

        Prefetches that hit turn into free loads (overlap with attention),
        so demand loads should decrease.
        """
        at = make_simple_activation_trace()
        lru_dm = LRUPolicy().simulate(at, experts_per_layer=2)
        pg_dm = PreGatedPolicy(base_policy_type='lru').simulate(
            at, experts_per_layer=2)

        lru_demands = sum(
            len(lt.demand_loads)
            for step in lru_dm.steps for lt in step.layers)
        pg_demands = sum(
            len(lt.demand_loads)
            for step in pg_dm.steps for lt in step.layers)

        # PreGated moves some demand loads to prefetches
        assert pg_demands <= lru_demands, \
            f"PreGated demands ({pg_demands}) should be <= " \
            f"LRU demands ({lru_demands})"

    def test_validate_passes(self):
        at = make_simple_activation_trace()
        dm = PreGatedPolicy(base_policy_type='lru').simulate(
            at, experts_per_layer=2)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_frequency_base(self):
        """PreGated with frequency base should also validate."""
        at = make_simple_activation_trace()
        dm = PreGatedPolicy(base_policy_type='frequency').simulate(
            at, experts_per_layer=2)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"


# ── 8. Summary statistics ────────────────────────────────────────────

class TestSummary:

    def test_summary_counts(self):
        at = make_simple_activation_trace()
        dm = LRUPolicy().simulate(at, experts_per_layer=2)
        s = dm.summary()
        assert s['steps'] == 5
        assert s['total_transfers'] == s['demand_loads']
        assert s['prefetches'] == 0

    def test_oracle_has_prefetches_in_summary(self):
        at = ActivationTrace(
            num_layers=3, num_experts=4,
            steps=[
                [[0, 1], [2, 3], [0, 2]],
            ])
        dm = OraclePolicy().simulate(at, experts_per_layer=2)
        s = dm.summary()
        # Oracle should have some prefetches (lookahead)
        assert s['total_transfers'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
