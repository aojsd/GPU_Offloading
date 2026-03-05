# Cache Simulation & Replay — Design and Implementation

This document covers Phases 5–7 of the MoE expert offloading sandbox: async
transfer infrastructure, the data movement trace format, the policy simulators,
and the replay controller that executes pre-computed data movement schedules on
real GPU hardware.

## Overview

The sandbox separates *what* data movement to perform (policy simulation) from
*how* to perform it (GPU replay). This enables offline evaluation of arbitrary
caching and prefetching strategies against real hardware timing:

```
1. Record expert activations:  ExpertOffloadEngine.save_trace()
                                   ↓
2. Simulate cache policy:      EvictionPolicy.simulate()  →  DataMovementTrace
                                   ↓
3. Replay on GPU hardware:     ReplayController(engine, trace)  →  wall-clock timing
```

Step 2 is pure Python with no GPU dependency. Step 3 uses the same piecewise
CUDA graph infrastructure as the MoEEngine, so replay timing faithfully
reflects production-grade kernel performance.

---

## Data Movement Trace Format

All data structures live in `data_movement_trace.py`.

### ActivationTrace — Policy Input

Structured view of which experts were selected at each (step, layer).
Converted from the flat trace format recorded by `ExpertOffloadEngine.save_trace()`.

```python
ActivationTrace:
    num_layers:  int
    num_experts: int
    steps:       list[list[list[int]]]   # steps[step][layer] = [expert_ids]
```

Load from a saved trace:

```python
trace = ActivationTrace.load("expert_trace.json")
# or convert from ExpertOffloadEngine's flat format:
trace = ActivationTrace.from_flat_trace(trace_data)
```

### DataMovementTrace — Replay Input

Complete schedule of every prefetch, eviction, and demand load for an entire
decode sequence. Produced by policy simulators, consumed by `ReplayController`.

Uses a **unified expert cache** shared across all layers. Any cache slot can
hold any `(layer, expert_id)` pair. The single `cache_size` parameter controls
the total number of expert slots on GPU — there is no per-layer partitioning.
This allows arbitrarily different numbers of experts per layer (e.g., 8 in
layer 0, 0 in layer 5, 3 in layer 10) as long as the total does not exceed
`cache_size`.

```
DataMovementTrace:
    num_layers:          int
    num_experts:         int
    cache_size:          int                         # total unified GPU cache slots
    initial_cache_state: list[(layer, expert_id)]    # experts on GPU at start
    steps:               list[StepTrace]             # one per decode token

StepTrace:
    layers: list[LayerTrace]                         # one per transformer layer

LayerTrace:
    topk_ids:       list[list[int]]      # expert selections per token
    topk_weights:   list[list[float]]
    prefetches:     list[TransferEvent]  # async transfers targeting this layer
    demand_loads:   list[TransferEvent]  # blocking, before stage4b
```

### TransferEvent

A single CPU→GPU expert transfer instruction:

```python
TransferEvent:
    target: (layer, expert_id)              # expert to load
    evict:  (layer, expert_id) | None       # expert to remove from persistent cache
```

The `evict` field determines the transfer's cache semantics:

| `evict` | Semantics | Slot |
|---------|-----------|------|
| `None` | **Free slot load** — cache has unused capacity, no eviction needed | Taken from free pool |
| `(l, e)` | **Replacement** — evict resident `(l, e)`, load target into freed slot | Evicted expert's former slot |

In the unified cache, `evict` can target **any** `(layer, expert_id)` regardless
of the target's layer. There is no same-layer constraint — cross-layer eviction
is fully supported.

### Serialization

```python
trace.save("lru_movement.json")       # JSON with indent=2
trace = DataMovementTrace.load("lru_movement.json")
```

### Validation

`DataMovementTrace.validate()` simulates the residency state machine and returns
a list of error strings (empty = valid). It checks:

1. **Initial capacity**: `len(initial_cache_state) <= cache_size`.
2. **Capacity maintenance**: unified cache never exceeds `cache_size` after transfers.
3. **Expert coverage**: every expert in `topk_ids` is either already resident
   or loaded by a prefetch/demand load in that layer.

Residency tracking uses a single `resident: set[(layer, eid)]`. Each transfer
with `evict != None` removes the evicted expert and adds the target. Transfers
with `evict == None` add the target using a free slot (capacity increases by 1).
State persists across layers and steps.

---

## Policy Simulators

All policies live in `policy_simulator.py` and inherit from `EvictionPolicy`.
Each takes an `ActivationTrace` and `cache_size` budget and returns a validated
`DataMovementTrace`. The cache is **unified across all layers** — a single pool
of slots keyed by `(layer, expert_id)`.

```python
trace = ActivationTrace.load("expert_trace.json")
policy = LRUPolicy()
dm_trace = policy.simulate(trace, cache_size=64)
errors = dm_trace.validate()
assert not errors
dm_trace.save("lru_movement.json")
```

All policies accept an optional `initial_cache: list[(layer, eid)]` parameter.
If not provided, `_default_initial_cache(cache_size, num_layers, num_experts)`
fills uniformly: expert 0 across all layers, then expert 1, etc., until
`cache_size` slots are filled.

### LRUPolicy — Least Recently Used

Unified LRU cache using `OrderedDict` keyed by `(layer, eid)`. On cache miss:
- If cache has free slots: load into free slot (no eviction).
- If cache is full: evict the globally LRU expert, load into freed slot.

All loads are **demand loads only** — no prefetching. This is the simplest
baseline and represents a reactive system with no prediction.

### OraclePolicy — Belady's MIN with Lookahead Prefetch

Two-pass algorithm over the full trace (requires offline/oracle knowledge):

**Pass 1 — Build next-use index**: For every `(layer, expert)`, record all
future access positions as a sorted list. Position is a global ordering:
`pos = step_idx * num_layers + layer`.

**Pass 2 — Simulate forward**: At each `(step, layer)`:

1. **Demand loads for current layer**: For each needed expert not in cache,
   evict the resident whose next use is farthest in the future (Belady's MIN),
   searching **globally across all (layer, eid)** entries. This is the
   theoretical minimum number of cache misses.

2. **Lookahead prefetch for next layer**: At layer `L`, look ahead to layer
   `L+1` (same step). If `L+1` needs a non-resident expert, issue a prefetch.
   The prefetch is stored in `layers[L+1].prefetches` (associated with the
   target layer) so the replay controller issues it at the right time — before
   stage4b of layer `L`, overlapping with stage4b(L) + stages 1-4a(L+1).

   ```
   Layer L+1 prefetches:  [{target: (L+1, eid), evict: (any_layer, victim)}]
   Layer L demand_loads:  [{target: (L, eid), evict: (any_layer, victim)}]
   ```

   Eviction victims can come from **any layer** — the unified cache has no
   same-layer constraint.

Belady victim selection avoids evicting experts that are needed in the current
operation (protected set). Ties are broken by the first candidate found in
iteration order.

### FrequencyPolicy — Least Frequently Used (LFU)

Global frequency counters keyed by `(layer, eid)`. On eviction, removes the
expert with the lowest access count (ties broken by smallest `(layer, eid)`
tuple). All loads are demand loads.

Supports optional **windowed mode**: reset all frequency counts every
`window_size` steps. This prevents stale frequency data from dominating in
non-stationary workloads.

### StaticPolicy — Global Frequency Ranking (Oracle)

Pre-computes access frequencies over the **entire** trace (including future
steps), then:

1. **Initial cache**: the top `cache_size` experts by global frequency.
2. **On miss**: evict the resident with the lowest global frequency.

Since frequencies are fixed and pre-computed, the top `(cache_size - E)`
experts (where `E = num_experts`) are effectively **pinned** and never
evicted. Only the bottom `E` slots cycle through demand-loaded experts.
This policy requires oracle knowledge (full trace) and represents the best
possible static caching strategy — it answers: "how well can you do if you
simply keep the most popular experts?"

All loads are demand loads (no prefetching).

### PreGatedPolicy — Lookahead Prefetch Wrapper

Wraps either LRU or Frequency as a base policy and adds 1-layer-ahead
prefetching (same mechanism as OraclePolicy's lookahead). At layer `L`,
examines layer `L+1`'s needed experts (from the trace) and issues prefetches
for non-resident ones.

Prefetches are stored in `layers[L+1].prefetches` (associated with the target
layer), matching the replay controller's expectation.

In a real system, this corresponds to running the router's gate network early
(before attention) to predict next-layer expert selections. Since we simulate
on recorded traces, we simply look ahead in the trace data.

Cache eviction decisions for both demand loads and prefetches are delegated to
the base policy's logic (LRU ordering or LFU counts). Eviction victims can
come from any layer.

### Policy Comparison

| Policy | Demand Loads | Prefetches | Eviction Strategy | Optimal? |
|--------|-------------|------------|-------------------|----------|
| LRU | Yes | No | Least recently used | No |
| Oracle | Yes | 1-layer lookahead | Belady's MIN (farthest future use) | Yes (eviction) |
| Frequency | Yes | No | Least frequently used (running) | No |
| Static | Yes | No | Least frequently used (global/oracle) | Best static |
| PreGated(LRU) | Yes | 1-layer lookahead | Least recently used | No |
| PreGated(Freq) | Yes | 1-layer lookahead | Least frequently used | No |

---

## Replay Controller

`replay_controller.py` implements `ReplayController`, which replays a
`DataMovementTrace` on real GPU hardware using the MoEEngine's piecewise CUDA
graph infrastructure.

### CUDA Stream Architecture

The controller uses two CUDA streams:

```
Default stream (compute):   CUDA graphs for stages 1, 4a, 4b
                            Eager FlashInfer decode (stage 2)
                            Eager FlashAttention prefill (stage 3)
                            Demand loads (blocking)

Transfer stream:            Async prefetches (overlaps with previous layer's
                            stage4b + current layer's stages 1-4a)
```

One CUDA event serves as the synchronization barrier:

- `_prefetch_done_event`: Recorded on transfer stream after prefetches complete.
  The compute stream waits on this in `process_layer_replay()` before demand loads.

### GPU Buffer Layout

The replay controller shares the MoEEngine's unified expert buffer:

```
w1_buf = [cache_size, 2*I, H]     # flat, no per-layer partitioning
w2_buf = [cache_size, H, I]
```

Any slot can hold any `(layer, expert_id)`. The controller tracks residency
via `_resident: dict[(layer, eid) → slot]` and `_free_slots: set[int]`.

Expert map indexing:
- `expert_map_abs[l][eid]`: Absolute slot in the flat buffer (or -1 if not resident).
- `expert_map_buf[eid]`: Single-layer working copy, updated before each stage4b.

### Per-Layer Execution Timeline

For each layer during `_mixed_step_piecewise`, the MoEEngine calls into the
replay controller at specific hook points. The key design: **only layer 0
prefetches at the start of the network**. For all subsequent layers, prefetches
are issued inside `process_layer_replay(L-1)` right before stage4b of L-1.

```
┌─ Layer 0 ─────────────────────────────────────────────────────────┐
│                                                                   │
│  Transfer:  ① begin_layer_prefetch(0)                             │
│                Issue layers[0].prefetches on transfer stream      │
│                Record _prefetch_done_event                        │
│                     │ (async, overlaps with stages 1-4a)          │
│                     ↓                                             │
│  Compute:   Stage 1 → Stage 2 → Stage 3 → Stage 4a               │
│                     ↓                                             │
│             ② process_layer_replay(0):                            │
│                a. Copy expert_map_abs[0] → expert_map_buf         │
│                b. wait_event(_prefetch_done_event)   ← SYNC ①    │
│                c. Execute demand_loads (blocking)                 │
│                d. Re-copy expert_map_abs → expert_map_buf         │
│                e. Issue layers[1].prefetches on transfer stream   │
│                   Record _prefetch_done_event                     │
│                     ↓                                             │
│             Stage 4b graph (MoE GEMM)                             │
│             post_layer(0) — no-op                                 │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─ Layer L (L > 0) ────────────────────────────────────────────────┐
│                                                                   │
│  Transfer:  (prefetches for layer L already issued in step ②      │
│              of layer L-1, running async during stage4b(L-1)      │
│              + stages 1-4a(L))                                    │
│                                                                   │
│  Compute:   Stage 1 → Stage 2 → Stage 3 → Stage 4a               │
│                     ↓                                             │
│             ② process_layer_replay(L):                            │
│                a. Copy expert_map_abs[L] → expert_map_buf         │
│                b. wait_event(_prefetch_done_event)   ← SYNC      │
│                c. Execute demand_loads (blocking)                 │
│                d. Re-copy expert_map_abs → expert_map_buf         │
│                e. Issue layers[L+1].prefetches (if L+1 < N)      │
│                   Record _prefetch_done_event                     │
│                     ↓                                             │
│             Stage 4b graph (MoE GEMM)                             │
│             post_layer(L) — no-op                                 │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

The overlap window for prefetches targeting layer L is:
- **Layer 0**: stages 1-4a of layer 0 only.
- **Layer L>0**: stage4b of layer L-1 + stages 1-4a of layer L.

### Prefetch and Eviction Timing — Detailed

#### Layer 0 Prefetches (`begin_layer_prefetch`)

Called **before stage 1**, only for layer 0. These transfers run on the transfer
stream and overlap with the full compute pipeline (stages 1–4a) of layer 0:

- **When issued**: Before any compute for layer 0 begins.
- **What they transfer**: Experts specified by `layers[0].prefetches`.
- **Overlap window**: Stages 1–4a provide ~1.1 ms of compute on Mixtral-8x7B
  (0.3 ms stage1 + 0.5 ms FlashInfer decode + 0.2 ms FA3 prefill + 0.1 ms
  stage4a). A single expert transfer at ~50 GB/s PCIe takes ~6.5 ms for a
  336 MB Mixtral expert, so **one expert can be partially overlapped** and
  **small experts (e.g., OLMoE at ~4 MB) can be fully hidden**.
- **Synchronization**: The transfer stream records `_prefetch_done_event`.
  The compute stream waits on this event in `process_layer_replay(0)`.

#### Next-Layer Prefetches (`process_layer_replay`, step e)

Issued inside `process_layer_replay(L)` right before returning (before stage4b
of layer L). These prefetch experts for layer L+1:

- **When issued**: After demand loads for layer L are complete, before stage4b.
- **What they transfer**: Experts specified by `layers[L+1].prefetches`.
- **Overlap window**: stage4b of layer L (~0.27 ms Mixtral) + stages 1-4a of
  layer L+1 (~1.1 ms). Total ~1.4 ms overlap — more than the layer-0-only
  window, enabling better hiding of larger transfers.
- **Synchronization**: The transfer stream records `_prefetch_done_event`.
  The compute stream waits on this in `process_layer_replay(L+1)`.

#### Demand Loads (`process_layer_replay`, step c)

**Blocking** transfers on the default (compute) stream. These represent true
cache misses — experts that no prefetch covered:

- **When issued**: After async prefetches are synchronized.
- **Impact**: Each demand load adds ~6.5 ms (Mixtral) or ~0.08 ms (OLMoE) of
  blocking latency before stage 4b can execute. Multiple demand loads are
  sequential.
- **Slot assignment**: Demand loads with `evict != None` reuse the evicted
  expert's slot. Demand loads with `evict == None` take a slot from the free
  pool.

#### Eviction Timing

Evictions are **immediate and coupled to the transfer that triggers them**.
There is no separate eviction phase:

- When `_execute_transfer(event)` processes an event with `evict != None`:
  1. The evicted expert is removed from `_resident` and `expert_map_abs`.
  2. The freed slot's GPU memory is overwritten by the incoming expert's
     `copy_()` (non-blocking when on transfer stream, blocking on compute stream).
  3. The new expert is added to `_resident` and `expert_map_abs`.

- The eviction is a **logical operation** (map update + slot reuse). There is
  no explicit "free" or "invalidate" — the slot is simply overwritten.
- Cross-layer eviction is fully supported: evicting `(layer=3, eid=5)` to
  load `(layer=7, eid=2)` is valid.

#### Expert Map Update Sequence

Before stage 4b, the expert map buffer must reflect the correct slot for every
expert the MoE kernel will access:

```python
# 1. Start with current cache state
expert_map_buf.copy_(expert_map_abs[layer])

# 2. All transfers (prefetches + demand loads) update expert_map_abs
#    via _execute_transfer(), so re-copy after transfers:
expert_map_buf.copy_(expert_map_abs[layer])
```

No scratchpad overlay is needed — all experts live in the unified cache and
are tracked in `expert_map_abs`.

### Latency Model

The wall-clock time for a layer is:

```
wall = compute_time + max(0, demand_load_time - prefetch_overlap)
```

Where:
- `compute_time` = stages 1 + 2 + 3 + 4a + 4b (~1.4 ms on Mixtral-8x7B)
- `prefetch_overlap` = min(prefetch_transfer_time, stages_1_through_4a_time)
- `demand_load_time` = sum of all demand load transfer times

**Best case** (all needed experts prefetched or resident): `wall ≈ compute_time`.

**Worst case** (all experts are demand loads, no prefetching): For Mixtral with
a small `cache_size` and top-2 selection, up to 2 experts may miss per layer,
adding ~13 ms of blocking IO to the ~1.4 ms of compute. This matches the
validated latency model from Phase 4: IO accounts for 90–96% of wall-clock
time in demand-load-only configurations.

### Comparison: ReplayController vs ExpertOffloadEngine

| Aspect | ReplayController | ExpertOffloadEngine |
|--------|-----------------|---------------------|
| **Decision source** | Pre-computed `DataMovementTrace` | Reactive (reads routing at runtime) |
| **Cache model** | Unified `cache_size` slots across all layers | Per-layer `experts_per_layer` + scratchpad |
| **Prefetching** | Scheduled per trace (2 categories: prefetch + demand) | None (pure demand loading) |
| **Eviction policy** | Encoded in trace (any policy) | Simple demand load (no policy) |
| **Async transfers** | Transfer stream + CUDA event sync | Blocking `synchronize()` per expert |
| **Use case** | Evaluating policy performance | Recording activation traces + baseline timing |
| **Hook points** | `begin_layer_prefetch` (layer 0 only) + `process_layer_replay` + `post_layer` (no-op) | `process_layer` (blocking) |

---

## End-to-End Usage

### Step 1: Record an Activation Trace

```python
from moe_engine import MoEEngine

# Use experts_per_layer for trace recording (ExpertOffloadEngine mode)
engine = MoEEngine("models/Mixtral-8x7B", experts_per_layer=2)
engine.capture_prefill_cuda_graph(total_token_sizes=[128])
engine.capture_mixed_cuda_graphs(total_token_sizes=[128])

# Run inference (prefill + decode steps)
with torch.inference_mode():
    engine.prefill_to_slot(0, input_ids)
    for step in range(50):
        logits = engine.mixed_step([0], [next_token], [], [])
        next_token = logits.argmax(-1)

# Save the trace recorded by ExpertOffloadEngine
engine.offload_engine.save_trace("expert_trace.json")
```

### Step 2: Simulate a Cache Policy

```python
from data_movement_trace import ActivationTrace
from policy_simulator import OraclePolicy

activation_trace = ActivationTrace.load("expert_trace.json")
policy = OraclePolicy()
# cache_size = total unified GPU slots (e.g., 4 experts per layer * 32 layers = 128)
dm_trace = policy.simulate(activation_trace, cache_size=128)

# Validate and inspect
errors = dm_trace.validate()
assert not errors, f"Trace errors: {errors}"

stats = dm_trace.summary()
print(f"Demand loads: {stats['demand_loads']}, "
      f"Prefetches: {stats['prefetches']}, "
      f"Evictions: {stats['evictions']}")

dm_trace.save("oracle_cs128_movement.json")
```

### Step 3: Replay on GPU Hardware

```python
from data_movement_trace import DataMovementTrace
from replay_controller import ReplayController

dm_trace = DataMovementTrace.load("oracle_cs128_movement.json")
# For replay, use cache_size mode (unified cache)
engine = MoEEngine("models/Mixtral-8x7B", cache_size=128)
controller = ReplayController(engine, dm_trace)
controller.setup()  # Load initial cache state into unified GPU buffer

# Attach to engine and run the same inference
engine.replay_controller = controller
with torch.inference_mode():
    engine.prefill_to_slot(0, input_ids)
    for step in range(len(dm_trace.steps)):
        logits = engine.mixed_step([0], [next_token], [], [])
        next_token = logits.argmax(-1)

# Detach
engine.replay_controller = None
replay_stats = controller.get_replay_stats()
```

### Comparing Policies

```python
from policy_simulator import LRUPolicy, OraclePolicy, PreGatedPolicy

policies = {
    'LRU':           LRUPolicy(),
    'Oracle':        OraclePolicy(),
    'PreGated(LRU)': PreGatedPolicy(base_policy_type='lru'),
}

for name, policy in policies.items():
    dm = policy.simulate(activation_trace, cache_size=128)
    errors = dm.validate()
    assert not errors
    s = dm.summary()
    print(f"{name:20s}  demands={s['demand_loads']:4d}  "
          f"prefetches={s['prefetches']:4d}  evictions={s['evictions']:4d}")
```

---

## Tests

`tests/test_replay_policy.py` covers all of the above with 39 tests:

| Category | Tests | What's verified |
|----------|-------|-----------------|
| Serialization | 4 | TransferEvent round-trip, cross-layer evict, DataMovementTrace save/load |
| Validation | 4 | Valid trace passes, missing expert, overcapacity, free slot addition |
| ActivationTrace | 3 | from_flat_trace, save/load round-trip, empty trace |
| LRU | 5 | No misses when all fit, eviction order, validate passes, no prefetches, cross-layer eviction |
| Oracle | 4 | Fewer misses than LRU, generates prefetches, validate passes, prefetches stored in target layer |
| Frequency | 3 | Evicts least frequent, windowed reset, validate passes |
| Static | 5 | Initial cache has most frequent, evicts least frequent globally, high-freq pinning, validate, no prefetches |
| PreGated | 5 | Generates prefetches, reduces demands vs LRU, both base types validate, prefetches in target layer |
| Integration | 4 | Variable experts per layer, zero-experts layer, cache pressure across layers, summary counts |
| Summary | 2 | Counts correct, Oracle prefetches in summary |

Run: `cd GPU_Offloading/src/MoE && python -m pytest tests/test_replay_policy.py -v`

---

## Files

| File | Role |
|------|------|
| `data_movement_trace.py` | `DataMovementTrace`, `ActivationTrace`, `TransferEvent`, `LayerTrace`, `StepTrace` — trace format with serialization and validation |
| `policy_simulator.py` | `LRUPolicy`, `OraclePolicy`, `FrequencyPolicy`, `PreGatedPolicy` — pure-Python policy simulators |
| `replay_controller.py` | `ReplayController` — GPU replay with async CUDA streams |
| `expert_offload_engine.py` | `ExpertOffloadEngine` — reactive demand loading, trace recording |
| `tests/test_replay_policy.py` | 34 unit tests for trace format + all policies (unified cache) |
