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

```
DataMovementTrace:
    num_layers:          int
    num_experts:         int
    experts_per_layer:   int                         # GPU cache capacity per layer
    initial_cache_state: list[(layer, expert_id)]    # experts on GPU at start
    steps:               list[StepTrace]             # one per decode token

StepTrace:
    layers: list[LayerTrace]                         # one per transformer layer

LayerTrace:
    topk_ids:                   list[list[int]]      # expert selections per token
    topk_weights:               list[list[float]]
    prefetches:                 list[TransferEvent]   # async, before attention
    post_routing_prefetches:    list[TransferEvent]   # async, after stage4a
    demand_loads:               list[TransferEvent]   # blocking, before stage4b
```

### TransferEvent

A single CPU→GPU expert transfer instruction:

```python
TransferEvent:
    target: (layer, expert_id)              # expert to load
    evict:  (layer, expert_id) | None       # expert to remove from persistent cache
```

The `evict` field determines the transfer's cache semantics:

| `evict` | Semantics | Persistence | Slot |
|---------|-----------|-------------|------|
| `None` | **Scratchpad load** — expert loaded into ephemeral scratchpad | Discarded after stage4b of this layer | `scratchpad_start + next_slot` |
| `(l, e)` | **Persistent replacement** — evict resident `e` from layer `l`, load target into freed slot | Survives across layers and steps | Evicted expert's former slot |

**Constraint**: `evict.layer` must equal `target.layer`. Each layer's persistent
cache slots occupy a disjoint address range in the unified buffer, so evictions
can only free slots within the same layer.

### Serialization

```python
trace.save("lru_movement.json")       # JSON with indent=2
trace = DataMovementTrace.load("lru_movement.json")
```

### Validation

`DataMovementTrace.validate()` simulates the residency state machine and returns
a list of error strings (empty = valid). It checks:

1. **Initial capacity**: no layer starts with more than `experts_per_layer` experts.
2. **Cross-layer eviction**: `evict.layer == target.layer` for every transfer.
3. **Capacity maintenance**: persistent cache never exceeds capacity after transfers.
4. **Expert coverage**: every expert in `topk_ids` is either already resident
   (persistent cache) or loaded by a transfer in that layer (prefetch,
   post-routing prefetch, or demand load).

Residency tracking rules in validation:

- **Persistent replacements** (`evict != None`): the evicted expert is removed
  from `resident[layer]`, and the target expert is added. This state persists
  across layers and steps.
- **Scratchpad loads** (`evict == None`): tracked in a per-layer `loaded_this_layer`
  set but *not* added to the persistent `resident` state. They are ephemeral and
  only valid for the current layer's stage4b.

---

## Policy Simulators

All policies live in `policy_simulator.py` and inherit from `EvictionPolicy`.
Each takes an `ActivationTrace` and `experts_per_layer` budget and returns a
validated `DataMovementTrace`.

```python
trace = ActivationTrace.load("expert_trace.json")
policy = LRUPolicy()
dm_trace = policy.simulate(trace, experts_per_layer=2)
errors = dm_trace.validate()
assert not errors
dm_trace.save("lru_movement.json")
```

### LRUPolicy — Least Recently Used

Per-layer LRU cache using `OrderedDict`. On cache miss:
- If cache has capacity: scratchpad load (no eviction).
- If cache is full: evict LRU (first item in OrderedDict), load into freed slot.

All loads are **demand loads only** — no prefetching. This is the simplest
baseline and represents a reactive system with no prediction.

### OraclePolicy — Belady's MIN with Lookahead Prefetch

Two-pass algorithm over the full trace (requires offline/oracle knowledge):

**Pass 1 — Build next-use index**: For every `(layer, expert)`, record all
future access positions as a sorted list. Position is a global ordering:
`pos = step_idx * num_layers + layer`.

**Pass 2 — Simulate forward**: At each `(step, layer)`:

1. **Demand loads for current layer**: For each needed expert not in cache,
   evict the resident whose next use is farthest in the future (Belady's MIN).
   This is the theoretical minimum number of cache misses.

2. **Lookahead prefetch for next layer**: At layer `L`, look ahead to layer
   `L+1` (same step). If `L+1` needs a non-resident expert, issue a prefetch
   targeting layer `L+1`'s cache. This prefetch is recorded in layer `L`'s
   `prefetches` list so the replay controller can overlap it with layer `L`'s
   attention compute.

   ```
   Layer L prefetches:    [{target: (L+1, eid), evict: (L+1, victim)}]
   Layer L demand_loads:  [{target: (L, eid), evict: (L, victim)}]
   ```

   The prefetch targets layer `L+1` and therefore evicts from layer `L+1`'s
   cache — maintaining the constraint `evict.layer == target.layer`.

Belady victim selection avoids evicting experts that are needed in the current
operation (protected set). Ties are broken by the first candidate found in
iteration order.

### FrequencyPolicy — Least Frequently Used (LFU)

Per-layer frequency counters. On eviction, removes the expert with the lowest
access count (ties broken by lowest expert ID). All loads are demand loads.

Supports optional **windowed mode**: reset all frequency counts every
`window_size` steps. This prevents stale frequency data from dominating in
non-stationary workloads.

### PreGatedPolicy — Lookahead Prefetch Wrapper

Wraps either LRU or Frequency as a base policy and adds 1-layer-ahead
prefetching (same mechanism as OraclePolicy's lookahead). At layer `L`,
examines layer `L+1`'s needed experts (from the trace) and issues prefetches
for non-resident ones.

In a real system, this corresponds to running the router's gate network early
(before attention) to predict next-layer expert selections. Since we simulate
on recorded traces, we simply look ahead in the trace data.

Cache eviction decisions for both demand loads and prefetches are delegated to
the base policy's logic (LRU ordering or LFU counts).

### Policy Comparison

| Policy | Demand Loads | Prefetches | Eviction Strategy | Optimal? |
|--------|-------------|------------|-------------------|----------|
| LRU | Yes | No | Least recently used | No |
| Oracle | Yes | 1-layer lookahead | Belady's MIN (farthest future use) | Yes (eviction) |
| Frequency | Yes | No | Least frequently used | No |
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

Transfer stream:            Async prefetches (before attention)
                            Async post-routing prefetches (after stage4a)
```

Two CUDA events serve as synchronization barriers between the streams:

- `_prefetch_done_event`: Recorded on transfer stream after pre-attention prefetches complete.
- `_post_routing_done_event`: Recorded on transfer stream after post-routing prefetches complete.

### GPU Buffer Layout

The replay controller shares the MoEEngine's unified expert buffer:

```
w1_buf = [L * experts_per_layer + scratchpad_slots, 2*I, H]
w2_buf = [L * experts_per_layer + scratchpad_slots, H, I]

Persistent cache:  slots [l*epl, ..., (l+1)*epl - 1]    per layer l
Scratchpad:        slots [L*epl, ..., L*epl + E - 1]    shared, ephemeral
```

Two expert maps index into this buffer:
- `expert_map[l][eid]`: Relative slot within layer `l`'s view (for per-layer CUDA graphs).
- `expert_map_abs[l][eid]`: Absolute slot in the full buffer (for stage4b graphs operating on the entire buffer).
- `expert_map_buf[eid]`: Single-layer working copy, updated before each stage4b. 32 bytes per layer (8 experts × 4 bytes).

### Per-Layer Execution Timeline

For each layer during `_mixed_step_piecewise`, the MoEEngine calls into the
replay controller at four hook points:

```
┌─────────────────────── Transfer Stream ──────────────────────────┐
│                                                                  │
│  ① begin_layer_prefetch(layer)                                   │
│     Issue prefetches for experts needed by layer+1               │
│     w1_buf[slot].copy_(w1_cpu[...], non_blocking=True)          │
│     w2_buf[slot].copy_(w2_cpu[...], non_blocking=True)          │
│     Record _prefetch_done_event                                  │
│              │                                                   │
│              │ (async, overlaps with stages 1-3)                 │
│              ↓                                                   │
│  ③ process_layer_replay(layer) — post-routing transfers          │
│     Issue post_routing_prefetches (async)                        │
│     Record _post_routing_done_event                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────── Default Stream (compute) ─────────────────┐
│                                                                  │
│  Stage 1 graph   RMSNorm → QKV proj → Q/K norm → RoPE → KV write│
│       ↓                                                          │
│  Stage 2 eager   FlashInfer BatchDecodeWithPagedKVCache          │
│       ↓                                                          │
│  Stage 3 eager   FlashAttention varlen prefill (if mixed batch)  │
│       ↓                                                          │
│  Stage 4a graph  O_proj → residual → post-attn norm → router    │
│       ↓                                                          │
│  ③ process_layer_replay(layer):                                  │
│     a. Copy expert_map_abs[layer] → expert_map_buf               │
│     b. Issue post_routing_prefetches on transfer stream           │
│     c. wait_event(_prefetch_done_event)       ← SYNC ①          │
│     d. wait_event(_post_routing_done_event)   ← SYNC ③          │
│     e. Execute demand_loads (blocking, default stream)           │
│     f. Re-copy expert_map_abs → expert_map_buf                   │
│        Overlay scratchpad assignments                            │
│       ↓                                                          │
│  Stage 4b graph  fused_experts (Triton GEMM) → residual add     │
│       ↓                                                          │
│  ④ post_layer(layer):                                            │
│     Clear scratchpad assignments (ephemeral)                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Prefetch and Eviction Timing — Detailed

#### Pre-Attention Prefetches (`begin_layer_prefetch`)

Called **before stage 1** of each layer. These transfers run on the transfer
stream and overlap with the full compute pipeline (stages 1–3 + stage 4a):

- **When issued**: Before any compute for this layer begins.
- **What they transfer**: Experts specified by `LayerTrace.prefetches`. These
  typically target the *next* layer's cache (e.g., Oracle/PreGated policies set
  `target = (layer+1, eid)`).
- **Overlap window**: Stages 1–3 provide ~1.1 ms of compute on Mixtral-8x7B
  (0.3 ms stage1 + 0.5 ms FlashInfer decode + 0.2 ms FA3 prefill + 0.1 ms
  stage4a). A single expert transfer at ~50 GB/s PCIe takes ~6.5 ms for a
  336 MB Mixtral expert, so **one expert can be partially overlapped** and
  **small experts (e.g., OLMoE at ~4 MB) can be fully hidden**.
- **Synchronization**: The transfer stream records `_prefetch_done_event` after
  all pre-attention prefetches. The compute stream waits on this event in
  `process_layer_replay()`, after stage 4a but before stage 4b.

For persistent replacements, the evicted expert's slot is immediately reclaimed
and repurposed for the incoming expert. The `expert_map` and `expert_map_abs`
are updated synchronously (CPU-side) during `_execute_transfer`, so they are
consistent before stage 4b's CUDA graph replays.

#### Post-Routing Prefetches (`process_layer_replay`, step b)

Called **after stage 4a** — the router has selected experts, so the exact
demand is known:

- **When issued**: Immediately after stage 4a completes on the compute stream.
- **What they transfer**: Experts specified by `LayerTrace.post_routing_prefetches`.
  These target the current layer's cache (policy knew from the trace which
  experts would be selected but chose to defer transfer until routing confirmed).
- **Overlap window**: Minimal — these run concurrently with the synchronization
  logic in `process_layer_replay()` but must complete before stage 4b.
- **Synchronization**: The transfer stream records `_post_routing_done_event`.
  The compute stream waits on this event immediately after (step d).

#### Demand Loads (`process_layer_replay`, step e)

**Blocking** transfers on the default (compute) stream. These represent true
cache misses — experts that no prefetch covered:

- **When issued**: After all async transfers are synchronized (steps c, d).
- **Impact**: Each demand load adds ~6.5 ms (Mixtral) or ~0.08 ms (OLMoE) of
  blocking latency before stage 4b can execute. Multiple demand loads are
  sequential.
- **Slot assignment**: Demand loads may use persistent replacement (evict a
  resident) or scratchpad (ephemeral).

#### Eviction Timing

Evictions are **immediate and coupled to the transfer that triggers them**.
There is no separate eviction phase:

- When `_execute_transfer(event)` processes an event with `evict != None`:
  1. The evicted expert is removed from `_resident[layer]` and both expert maps.
  2. The freed slot's GPU memory is overwritten by the incoming expert's
     `copy_()` (non-blocking when on transfer stream, blocking on compute stream).
  3. The new expert is added to `_resident[layer]` and both expert maps.

- The eviction is a **logical operation** (map update + slot reuse). There is
  no explicit "free" or "invalidate" — the slot is simply overwritten.

#### Expert Map Update Sequence

Before stage 4b, the expert map buffer must reflect the correct slot for every
expert the MoE kernel will access:

```python
# 1. Start with persistent cache state
expert_map_buf.copy_(expert_map_abs[layer])

# 2. Persistent replacements from prefetches/post-routing/demand loads
#    already updated expert_map_abs during _execute_transfer(), so re-copy:
expert_map_buf.copy_(expert_map_abs[layer])

# 3. Overlay scratchpad assignments (ephemeral, not in expert_map_abs)
for eid, slot in _scratchpad_assignments.items():
    expert_map_buf[eid] = slot
```

This ensures the Triton `fused_experts` kernel in stage 4b reads the correct
weight slice for every selected expert, whether it was persistently cached,
persistently replaced, or loaded into the scratchpad.

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
`experts_per_layer=2` and top-2 selection, up to 2 experts may miss per layer,
adding ~13 ms of blocking IO to the ~1.4 ms of compute. This matches the
validated latency model from Phase 4: IO accounts for 90–96% of wall-clock
time in demand-load-only configurations.

### Comparison: ReplayController vs ExpertOffloadEngine

| Aspect | ReplayController | ExpertOffloadEngine |
|--------|-----------------|---------------------|
| **Decision source** | Pre-computed `DataMovementTrace` | Reactive (reads routing at runtime) |
| **Prefetching** | Scheduled per trace (3 transfer categories) | Optional via `prefetch_experts_async()` |
| **Eviction policy** | Encoded in trace (any policy) | Simple demand load (no policy) |
| **Async transfers** | Full: prefetch stream + sync events | Partial: blocking `synchronize()` per expert |
| **Use case** | Evaluating policy performance | Recording activation traces + baseline timing |
| **Hook points** | `begin_layer_prefetch` + `process_layer_replay` + `post_layer` | `process_layer` (blocking) |

---

## End-to-End Usage

### Step 1: Record an Activation Trace

```python
from moe_engine import MoEEngine

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
dm_trace = policy.simulate(activation_trace, experts_per_layer=4)

# Validate and inspect
errors = dm_trace.validate()
assert not errors, f"Trace errors: {errors}"

stats = dm_trace.summary()
print(f"Demand loads: {stats['demand_loads']}, "
      f"Prefetches: {stats['prefetches']}, "
      f"Evictions: {stats['evictions']}")

dm_trace.save("oracle_epl4_movement.json")
```

### Step 3: Replay on GPU Hardware

```python
from data_movement_trace import DataMovementTrace
from replay_controller import ReplayController

dm_trace = DataMovementTrace.load("oracle_epl4_movement.json")
controller = ReplayController(engine, dm_trace)
controller.setup()  # Load initial cache state into GPU buffer

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
    dm = policy.simulate(activation_trace, experts_per_layer=4)
    errors = dm.validate()
    assert not errors
    s = dm.summary()
    print(f"{name:20s}  demands={s['demand_loads']:4d}  "
          f"prefetches={s['prefetches']:4d}  evictions={s['evictions']:4d}")
```

---

## Tests

`tests/test_replay_policy.py` covers all of the above with 26 tests:

| Category | Tests | What's verified |
|----------|-------|-----------------|
| Serialization | 3 | TransferEvent round-trip, DataMovementTrace save/load |
| Validation | 4 | Valid trace passes, missing expert, cross-layer eviction, overcapacity |
| ActivationTrace | 3 | from_flat_trace, save/load round-trip, empty trace |
| LRU | 4 | No misses when all fit, eviction order, validate passes, no prefetches |
| Oracle | 3 | Fewer misses than LRU, generates prefetches, validate passes |
| Frequency | 3 | Evicts least frequent, windowed reset, validate passes |
| PreGated | 4 | Generates prefetches, reduces demands vs LRU, both base types validate |
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
| `tests/test_replay_policy.py` | 26 unit tests for trace format + all policies |
