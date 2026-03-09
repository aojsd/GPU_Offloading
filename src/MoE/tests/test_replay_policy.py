"""Tests for Phases 5-7: data movement trace, policy simulation, replay.

Tests the unified expert cache (shared across all layers, keyed by
(layer, expert_id) pairs) and the new prefetch timing (layer 0 prefetches
at start, layer L>0 prefetches stored in layers[L].prefetches, issued
before stage4b of layer L-1).

Test categories:
  1. GPUReplayTrace serialization round-trip
  2. GPUReplayTrace.validate() correctness
  3. ActivationTrace.from_flat_trace() conversion
  4. LRU-None: eviction order, miss counts
  5. Belady-Oracle: optimality vs LRU, prefetch generation
  6. LFU-None: frequency eviction
  7. Static-None: frequency-ranked initial cache, eviction
  8. All policies: validate() passes on output

Run:
    cd GPU_Offloading/src/MoE && python -m pytest tests/test_replay_policy.py -v
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gpu_replay_trace import (
    ActivationTrace,
    GPUReplayTrace,
    LayerTrace,
    StepTrace,
    TransferEvent,
)
from policy_simulator import (
    LRU,
    Belady,
    LFU,
    StaticFreq,
    NoPrefetch,
    OraclePrefetch,
    simulate,
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

    def test_transfer_event_cross_layer_evict(self):
        """Unified cache allows cross-layer eviction."""
        te = TransferEvent(target=(0, 2), evict=(1, 3))
        d = te.to_dict()
        te2 = TransferEvent.from_dict(d)
        assert te2.target == (0, 2)
        assert te2.evict == (1, 3)

    def test_gpu_replay_trace_save_load(self, tmp_path):
        trace = GPUReplayTrace(
            num_layers=2,
            num_experts=4,
            cache_size=4,
            initial_cache_state=[(0, 0), (0, 1), (1, 0), (1, 1)],
            steps=[
                StepTrace(layers=[
                    LayerTrace(
                        topk_ids=[[0, 2]],
                        topk_weights=[[0.6, 0.4]],
                        prefetches=[],
                        demand_loads=[TransferEvent(
                            target=(0, 2), evict=(1, 1))],
                    ),
                    LayerTrace(
                        topk_ids=[[1, 3]],
                        topk_weights=[[0.5, 0.5]],
                        prefetches=[TransferEvent(
                            target=(1, 3), evict=(0, 1))],
                        demand_loads=[],
                    ),
                ]),
            ],
        )

        path = str(tmp_path / "trace.json")
        trace.save(path)
        loaded = GPUReplayTrace.load(path)

        assert loaded.num_layers == 2
        assert loaded.num_experts == 4
        assert loaded.cache_size == 4
        assert len(loaded.steps) == 1
        assert loaded.initial_cache_state == [(0, 0), (0, 1), (1, 0), (1, 1)]

        lt0 = loaded.steps[0].layers[0]
        assert lt0.topk_ids == [[0, 2]]
        assert len(lt0.demand_loads) == 1
        assert lt0.demand_loads[0].target == (0, 2)
        assert lt0.demand_loads[0].evict == (1, 1)

        lt1 = loaded.steps[0].layers[1]
        assert len(lt1.prefetches) == 1
        assert lt1.prefetches[0].target == (1, 3)
        assert lt1.prefetches[0].evict == (0, 1)


# ── 2. GPUReplayTrace.validate() ─────────────────────────────────

class TestValidation:

    def test_valid_trace_passes(self):
        """A correctly constructed trace should validate with no errors."""
        trace = GPUReplayTrace(
            num_layers=1,
            num_experts=4,
            cache_size=2,
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
        """Expert needed but not resident and not loaded -> error."""
        trace = GPUReplayTrace(
            num_layers=1,
            num_experts=4,
            cache_size=2,
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

    def test_cross_layer_eviction_allowed(self):
        """Unified cache allows evicting from any layer."""
        trace = GPUReplayTrace(
            num_layers=2,
            num_experts=4,
            cache_size=4,
            initial_cache_state=[(0, 0), (0, 1), (1, 0), (1, 1)],
            steps=[
                StepTrace(layers=[
                    LayerTrace(
                        topk_ids=[[0, 2]],
                        topk_weights=[[0.5, 0.5]],
                        demand_loads=[TransferEvent(
                            target=(0, 2), evict=(1, 0))],
                    ),
                    LayerTrace(
                        topk_ids=[[1, 0]],
                        topk_weights=[[0.5, 0.5]],
                        demand_loads=[TransferEvent(
                            target=(1, 0), evict=(0, 1))],
                    ),
                ]),
            ],
        )
        errors = trace.validate()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_initial_overcapacity_detected(self):
        """Initial state exceeds cache_size -> error."""
        trace = GPUReplayTrace(
            num_layers=1,
            num_experts=4,
            cache_size=2,
            initial_cache_state=[(0, 0), (0, 1), (0, 2)],  # 3 > cache 2
            steps=[],
        )
        errors = trace.validate()
        assert any('3' in e for e in errors)

    def test_free_slot_addition(self):
        """evict=None should work when cache has free slots."""
        trace = GPUReplayTrace(
            num_layers=1,
            num_experts=4,
            cache_size=3,
            initial_cache_state=[(0, 0), (0, 1)],  # 2 of 3 slots used
            steps=[
                StepTrace(layers=[
                    LayerTrace(
                        topk_ids=[[0, 1, 2]],
                        topk_weights=[[0.3, 0.3, 0.4]],
                        demand_loads=[TransferEvent(
                            target=(0, 2), evict=None)],  # use free slot
                    ),
                ]),
            ],
        )
        errors = trace.validate()
        assert errors == [], f"Unexpected errors: {errors}"


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

    def test_no_router_inputs_by_default(self):
        """ActivationTrace without router inputs returns None."""
        at = ActivationTrace(num_layers=2, num_experts=4,
                             steps=[[[0, 1], [2, 3]]])
        assert not at.has_router_inputs()
        assert at.get_router_input(0, 0) is None

    def test_router_inputs_round_trip(self, tmp_path):
        """Save/load with companion .npz router inputs."""
        json_path = str(tmp_path / "trace.json")
        npz_path = str(tmp_path / "trace_router_inputs.npz")

        at = ActivationTrace(
            num_layers=2, num_experts=4,
            steps=[[[0, 1], [2, 3]], [[1, 2], [0, 3]]])
        at.save(json_path)

        # Write companion router inputs
        hidden_dim = 8
        arrays = {
            'step0_layer0': np.ones((1, hidden_dim), dtype=np.float32) * 0.5,
            'step0_layer1': np.ones((1, hidden_dim), dtype=np.float32) * 1.0,
            'step1_layer0': np.ones((1, hidden_dim), dtype=np.float32) * 1.5,
            'step1_layer1': np.ones((1, hidden_dim), dtype=np.float32) * 2.0,
        }
        np.savez_compressed(npz_path, **arrays)

        # Load — should auto-detect .npz
        at2 = ActivationTrace.load(json_path)
        assert at2.has_router_inputs()
        assert at2.steps == at.steps

        # Check router input values
        ri = at2.get_router_input(0, 0)
        assert ri is not None
        assert ri.shape == (1, hidden_dim)
        assert ri.dtype == np.float32
        np.testing.assert_allclose(ri, 0.5)

        ri_last = at2.get_router_input(1, 1)
        np.testing.assert_allclose(ri_last, 2.0)

    def test_router_inputs_missing_key(self, tmp_path):
        """get_router_input returns None for missing step/layer."""
        json_path = str(tmp_path / "trace.json")
        npz_path = str(tmp_path / "trace_router_inputs.npz")

        at = ActivationTrace(num_layers=1, num_experts=2,
                             steps=[[[0, 1]]])
        at.save(json_path)
        np.savez_compressed(npz_path,
                            step0_layer0=np.zeros((1, 4), dtype=np.float32))

        at2 = ActivationTrace.load(json_path)
        assert at2.has_router_inputs()
        assert at2.get_router_input(0, 0) is not None
        assert at2.get_router_input(99, 99) is None

    def test_load_without_npz(self, tmp_path):
        """Load trace without companion .npz — has_router_inputs is False."""
        json_path = str(tmp_path / "trace.json")
        at = ActivationTrace(num_layers=2, num_experts=4,
                             steps=[[[0, 1], [2, 3]]])
        at.save(json_path)

        at2 = ActivationTrace.load(json_path)
        assert not at2.has_router_inputs()

    def test_policies_work_with_router_inputs(self, tmp_path):
        """Policies ignore router_inputs field and simulate normally."""
        json_path = str(tmp_path / "trace.json")
        npz_path = str(tmp_path / "trace_router_inputs.npz")

        at = ActivationTrace(num_layers=1, num_experts=4,
                             steps=[[[0, 1]], [[2, 3]], [[0, 2]]])
        at.save(json_path)
        np.savez_compressed(npz_path,
                            step0_layer0=np.zeros((1, 4), dtype=np.float32))

        at2 = ActivationTrace.load(json_path)
        assert at2.has_router_inputs()

        # All cache+prefetch combos should work fine
        combos = [
            (LRU(), NoPrefetch()),
            (Belady(), OraclePrefetch()),
            (LFU(), NoPrefetch()),
            (StaticFreq(), NoPrefetch()),
        ]
        for cache_pol, pf_pol in combos:
            dm = simulate(cache_pol, pf_pol, at2, cache_size=2)
            errors = dm.validate()
            assert errors == [], \
                f"{cache_pol.__class__.__name__} failed: {errors}"


# ── 4. LRU-None ──────────────────────────────────────────────────────

class TestLRU:

    def test_no_misses_when_all_resident(self):
        """If cache fits all experts for all layers, no demand loads."""
        at = make_simple_activation_trace()
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=12)
        for step in dm.steps:
            for lt in step.layers:
                assert lt.demand_loads == []

    def test_eviction_order(self):
        """With 1 layer, cache_size=2, verify LRU eviction."""
        at = ActivationTrace(
            num_layers=1, num_experts=4,
            steps=[
                [[0, 1]],   # cache: {(0,0), (0,1)}
                [[2, 3]],   # miss both; evict (0,0) LRU, then (0,1)
                [[0, 1]],   # miss both; evict (0,2) LRU, then (0,3)
            ])
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=2)

        # Step 0: no misses (0,1 are initial)
        assert dm.steps[0].layers[0].demand_loads == []

        # Step 1: 2 demand loads
        loads1 = dm.steps[1].layers[0].demand_loads
        assert len(loads1) == 2
        assert loads1[0].target == (0, 2)
        assert loads1[0].evict == (0, 0)  # LRU is (0,0)
        assert loads1[1].target == (0, 3)
        assert loads1[1].evict == (0, 1)  # then (0,1)

        # Step 2: 2 demand loads
        loads2 = dm.steps[2].layers[0].demand_loads
        assert len(loads2) == 2
        assert loads2[0].target == (0, 0)
        assert loads2[0].evict == (0, 2)
        assert loads2[1].target == (0, 1)
        assert loads2[1].evict == (0, 3)

    def test_cross_layer_eviction(self):
        """Unified cache can evict from a different layer."""
        at = ActivationTrace(
            num_layers=2, num_experts=2,
            steps=[
                [[0, 1], [0, 1]],  # all cached initially
                [[0, 1], [0, 1]],  # all cached, touch all
            ])
        # cache_size=3: only 3 of 4 (layer, eid) pairs fit
        # Default initial: (0,0), (1,0), (0,1) — fills 3 slots
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=3)

        # Step 0 layer 1: needs (1,1) which wasn't initially cached
        loads = dm.steps[0].layers[1].demand_loads
        assert len(loads) == 1
        assert loads[0].target == (1, 1)
        assert loads[0].evict is not None

    def test_validate_passes(self):
        at = make_simple_activation_trace()
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=6)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_no_prefetches(self):
        """LRU-None should never generate prefetches."""
        at = make_simple_activation_trace()
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=6)
        for step in dm.steps:
            for lt in step.layers:
                assert lt.prefetches == []


# ── 5. Belady-Oracle ─────────────────────────────────────────────────

class TestBeladyOracle:

    def test_fewer_misses_than_lru(self):
        """Belady should have <= total transfers than LRU."""
        at = make_simple_activation_trace()
        lru_dm = simulate(LRU(), NoPrefetch(), at, cache_size=6)
        oracle_dm = simulate(Belady(), OraclePrefetch(), at, cache_size=6)

        lru_transfers = lru_dm.summary()['total_transfers']
        oracle_transfers = oracle_dm.summary()['total_transfers']

        assert oracle_transfers <= lru_transfers, \
            f"Belady ({oracle_transfers}) should not have more transfers " \
            f"than LRU ({lru_transfers})"

    def test_generates_prefetches(self):
        """Belady-Oracle should generate lookahead prefetches."""
        at = ActivationTrace(
            num_layers=3, num_experts=4,
            steps=[
                [[0, 1], [2, 3], [0, 2]],
            ])
        dm = simulate(Belady(), OraclePrefetch(), at, cache_size=6)

        has_prefetches = any(
            len(lt.prefetches) > 0
            for step in dm.steps for lt in step.layers)
        assert has_prefetches, "Belady-Oracle should generate prefetches"

    def test_prefetches_stored_in_target_layer(self):
        """Prefetches for layer L should be in layers[L].prefetches."""
        at = ActivationTrace(
            num_layers=2, num_experts=4,
            steps=[
                [[0, 1], [2, 3]],
            ])
        dm = simulate(Belady(), OraclePrefetch(), at, cache_size=4)

        lt1 = dm.steps[0].layers[1]
        assert len(lt1.prefetches) > 0, "OraclePrefetch should generate prefetches"
        for pf in lt1.prefetches:
            assert pf.target[0] == 1, \
                f"Prefetch in layers[1] should target layer 1, " \
                f"got {pf.target}"

    def test_validate_passes(self):
        at = make_simple_activation_trace()
        dm = simulate(Belady(), OraclePrefetch(), at, cache_size=6)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"


# ── 6. LFU-None ──────────────────────────────────────────────────────

class TestLFU:

    def test_evicts_least_frequent(self):
        """Expert with fewest accesses should be evicted."""
        at = ActivationTrace(
            num_layers=1, num_experts=4,
            steps=[
                [[0, 1]],   # both accessed once
                [[0, 2]],   # 0 accessed again, need 2 -> evict (0,1) freq=1
            ])
        dm = simulate(LFU(), NoPrefetch(), at, cache_size=2)

        loads = dm.steps[1].layers[0].demand_loads
        assert len(loads) == 1
        assert loads[0].target == (0, 2)
        assert loads[0].evict == (0, 1)  # (0,1) has lower freq than (0,0)

    def test_windowed_reset(self):
        """Windowed mode resets counts periodically."""
        at = ActivationTrace(
            num_layers=1, num_experts=4,
            steps=[
                [[0, 1]],   # freq: (0,0)->1, (0,1)->1
                [[0, 1]],   # freq: (0,0)->2, (0,1)->2
                [[0, 2]],   # window reset; freq: (0,0)->1, (0,2)->1
            ])
        dm = simulate(LFU(window_size=2), NoPrefetch(), at, cache_size=2)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_validate_passes(self):
        at = make_simple_activation_trace()
        dm = simulate(LFU(), NoPrefetch(), at, cache_size=6)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"


# ── 7. Static-None ───────────────────────────────────────────────────

class TestStaticFreq:

    def test_initial_cache_has_most_frequent(self):
        """Initial cache should contain the highest-frequency experts."""
        at = ActivationTrace(
            num_layers=1, num_experts=4,
            steps=[
                [[0, 1]],   # freq: 0->1, 1->1
                [[0, 2]],   # freq: 0->2, 2->1
                [[0, 2]],   # freq: 0->3, 2->2
            ])
        dm = simulate(StaticFreq(), NoPrefetch(), at, cache_size=2)
        initial = set(dm.initial_cache_state)
        assert (0, 0) in initial
        assert (0, 2) in initial

    def test_evicts_least_frequent_globally(self):
        """On miss, should evict the resident with lowest global frequency."""
        at = ActivationTrace(
            num_layers=1, num_experts=4,
            steps=[
                [[0, 1]],   # cache has (0,0) and (0,2) initially
                [[0, 2]],
                [[0, 2]],
                [[0, 3]],   # miss on 3 -> evict lowest freq resident
            ])
        dm = simulate(StaticFreq(), NoPrefetch(), at, cache_size=2)

        loads = dm.steps[3].layers[0].demand_loads
        assert len(loads) == 1
        assert loads[0].target == (0, 3)
        assert loads[0].evict == (0, 2)

    def test_high_freq_experts_never_evicted(self):
        """Top experts by frequency should stay pinned."""
        at = ActivationTrace(
            num_layers=2, num_experts=4,
            steps=[
                [[0], [0]],
                [[0], [0]],
                [[0], [0]],
                [[0], [0]],
                [[0], [0]],   # 5 accesses each for (0,0) and (1,0)
                [[1, 2], [1, 2]],
            ])
        dm = simulate(StaticFreq(), NoPrefetch(), at, cache_size=4)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

        for step in dm.steps:
            for lt in step.layers:
                for dl in lt.demand_loads:
                    if dl.evict is not None:
                        assert dl.evict != (0, 0), \
                            "(0,0) is most frequent, should not be evicted"
                        assert dl.evict != (1, 0), \
                            "(1,0) is most frequent, should not be evicted"

    def test_validate_passes(self):
        at = make_simple_activation_trace()
        dm = simulate(StaticFreq(), NoPrefetch(), at, cache_size=6)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_no_prefetches_with_noprefetch(self):
        """Static-None should never generate prefetches."""
        at = make_simple_activation_trace()
        dm = simulate(StaticFreq(), NoPrefetch(), at, cache_size=6)
        for step in dm.steps:
            for lt in step.layers:
                assert lt.prefetches == []


# ── 8. Mix-and-match combos ──────────────────────────────────────────

class TestMixAndMatch:

    def test_static_oracle_has_prefetches(self):
        """Static-Oracle should produce prefetches."""
        at = ActivationTrace(
            num_layers=3, num_experts=4,
            steps=[
                [[0, 1], [2, 3], [0, 2]],
            ])
        # cache_size=4 so not all 6 unique experts fit → forces misses/prefetches
        dm = simulate(StaticFreq(), OraclePrefetch(), at, cache_size=4)
        has_pf = any(len(lt.prefetches) > 0
                     for step in dm.steps for lt in step.layers)
        assert has_pf, "Static-Oracle should generate prefetches"
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_lru_oracle_has_prefetches(self):
        """LRU-Oracle should produce prefetches."""
        at = ActivationTrace(
            num_layers=3, num_experts=4,
            steps=[
                [[0, 1], [2, 3], [0, 2]],
            ])
        dm = simulate(LRU(), OraclePrefetch(), at, cache_size=6)
        has_pf = any(len(lt.prefetches) > 0
                     for step in dm.steps for lt in step.layers)
        assert has_pf, "LRU-Oracle should generate prefetches"
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_belady_none_no_prefetches(self):
        """Belady-None should not produce prefetches."""
        at = make_simple_activation_trace()
        dm = simulate(Belady(), NoPrefetch(), at, cache_size=6)
        for step in dm.steps:
            for lt in step.layers:
                assert lt.prefetches == []
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_all_combos_validate(self):
        """Every cache x prefetch combo should produce valid traces."""
        at = make_simple_activation_trace()
        caches = [LRU(), Belady(), LFU(), StaticFreq()]
        prefetches = [NoPrefetch(), OraclePrefetch()]
        for cp in caches:
            for pp in prefetches:
                dm = simulate(cp, pp, at, cache_size=6)
                errors = dm.validate()
                assert errors == [], \
                    f"{cp.__class__.__name__}-{pp.__class__.__name__}: {errors}"


# ── 9. Summary statistics ────────────────────────────────────────────

class TestSummary:

    def test_summary_counts(self):
        at = make_simple_activation_trace()
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=6)
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
        dm = simulate(Belady(), OraclePrefetch(), at, cache_size=6)
        s = dm.summary()
        assert s['prefetches'] > 0


# ── 10. Unified cache specific ───────────────────────────────────────

class TestUnifiedCache:

    def test_variable_experts_per_layer(self):
        """Different layers can have different numbers of experts cached."""
        at = ActivationTrace(
            num_layers=2, num_experts=4,
            steps=[
                [[0, 1], [2]],
                [[0, 1], [3]],
            ])
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=3)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_zero_experts_layer_works(self):
        """A layer needing 0 experts should work fine."""
        at = ActivationTrace(
            num_layers=2, num_experts=4,
            steps=[
                [[0, 1, 2, 3], []],
            ])
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=4)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"
        lt1 = dm.steps[0].layers[1]
        assert lt1.demand_loads == []

    def test_cache_pressure_across_layers(self):
        """Cache pressure from one layer affects another."""
        at = ActivationTrace(
            num_layers=2, num_experts=4,
            steps=[
                [[0, 1], [2, 3]],
                [[2, 3], [0, 1]],
            ])
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=4)
        errors = dm.validate()
        assert errors == [], f"Validation errors: {errors}"

        total_loads = sum(
            len(lt.demand_loads) for lt in dm.steps[1].layers)
        assert total_loads > 0, "Should have demand loads on step 1"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
