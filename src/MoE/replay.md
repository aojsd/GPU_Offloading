# Cache Simulation & Replay — Design and Implementation

This document covers Phases 5–7 of the MoE expert offloading sandbox: async
transfer infrastructure, the data movement trace format, the policy simulators,
and the replay controller that executes pre-computed data movement schedules on
real GPU hardware.

## Overview

The replay is **trace-driven**: each request runs exactly `len(trace.steps)` steps
as determined during trace collection. EOS tokens from the model are ignored.
Generated tokens should be identical to those traced (same model, same kernels,
greedy decoding), so expert routing matches the trace exactly. The purpose is to
measure data movement timing under realistic compute, not to produce correct text.

**Replay fidelity requirement:** For timing results to be meaningful, GPU replay
MUST faithfully recreate the batch compositions from the continuous batching
simulator (Phase 2). Each step must dispatch the exact mix of decode sequences,
prefill chunks, and continuation chunks at their traced token counts. Batch size
directly determines (a) compute kernel duration (attention + MoE), (b) the overlap
window available to hide async prefetches, and (c) PCIe contention from concurrent
transfers. Single-sequence replay produces correct expert cache transfer counts but
**invalid timing data** — all compute/IO ratios are wrong because BS=1 decode
completes in ~9ms while real batches of 30+ sequences take much longer, providing
far more overlap opportunity for prefetches. Only `scripts/batched_replay.py`
performs faithful multi-sequence replay.

The sandbox separates *what* data movement to perform (policy simulation) from
*how* to perform it (GPU replay). This enables offline evaluation of arbitrary
caching and prefetching strategies against real hardware timing:

```
1. Record expert activations:  ExpertOffloadEngine.save_trace()
                                   ↓
2. Simulate cache policy:      simulate(cache, prefetch, trace)  →  GPUReplayTrace
                                   ↓
3. Replay on GPU hardware:     ReplayController(engine, trace)  →  wall-clock timing
```

Step 2 is pure Python with no GPU dependency. Step 3 uses the same piecewise
CUDA graph infrastructure as the MoEEngine, so replay timing faithfully
reflects production-grade kernel performance.

---

## Data Movement Trace Format

All data structures live in `gpu_replay_trace.py`.

### ActivationTrace — Policy Input

Structured view of which experts were selected at each (step, layer).
Converted from the flat trace format recorded by `ExpertOffloadEngine.save_trace()`.

```python
ActivationTrace:
    num_layers:  int
    num_experts: int
    steps:       list[list[list[int]]]               # steps[step][layer] = [expert_ids]
    scheduling:  Optional[list[StepScheduling]]       # per-step batch metadata (from build_trace.py)
```

When loaded from a batched trace (via `ActivationTrace.from_flat_trace()` or
`ActivationTrace.load()`), the `scheduling` field contains per-step metadata
including batch composition (`active_requests` with chunked prefill fields)
and scheduling events (`admit`, `complete`). With the no-preemption policy,
`preempt` and `readmit` events never occur. See
[trace_construction/README.md](trace_construction/README.md) for details.

Load from a saved trace:

```python
trace = ActivationTrace.load("expert_trace.json")
# or convert from ExpertOffloadEngine's flat format:
trace = ActivationTrace.from_flat_trace(trace_data)
```

### GPUReplayTrace — Replay Input

Complete schedule of every prefetch, eviction, and demand load for an entire
decode sequence. Produced by policy simulators, consumed by `ReplayController`.

Uses a **unified expert cache** shared across all layers. Any cache slot can
hold any `(layer, expert_id)` pair. The single `cache_size` parameter controls
the total number of expert slots on GPU — there is no per-layer partitioning.
This allows arbitrarily different numbers of experts per layer (e.g., 8 in
layer 0, 0 in layer 5, 3 in layer 10) as long as the total does not exceed
`cache_size`.

```
GPUReplayTrace:
    num_layers:          int
    num_experts:         int
    cache_size:          int                         # total unified GPU cache slots
    initial_cache_state: list[(layer, expert_id)]    # experts on GPU at start
    steps:               list[StepTrace]             # one per decode token

StepTrace:
    layers: list[LayerTrace]                         # one per transformer layer
    scheduling: StepScheduling | None                # batch metadata (propagated from ActivationTrace)

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
trace = GPUReplayTrace.load("lru_movement.json")
```

### Validation

`GPUReplayTrace.validate()` simulates the residency state machine and returns
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

Policy simulation decomposes into two orthogonal axes:
- **CachePolicy** (eviction strategy): which expert to remove when the cache is full.
- **PrefetchPolicy** (prefetch strategy): what to load proactively.

Any cache policy can be combined with any prefetch policy. All classes live in
`policy_simulator.py`.

```python
from gpu_replay_trace import ActivationTrace
from policy_simulator import LRU, Belady, StaticFreq, OraclePrefetch, NoPrefetch, simulate

trace = ActivationTrace.load("expert_trace.json")
dm_trace = simulate(LRU(), NoPrefetch(), trace, cache_size=64)
errors = dm_trace.validate()
assert not errors
dm_trace.save("lru_none_movement.json")
```

The `simulate()` function accepts an optional `initial_cache: list[(layer, eid)]`.
If not provided, `_default_initial_cache(cache_size, num_layers, num_experts)`
fills uniformly: expert 0 across all layers, then expert 1, etc., until
`cache_size` slots are filled. `StaticFreq` overrides this with the top-k
most frequent experts.

### Cache Policies

#### LRU — Least Recently Used

Unified LRU cache using `OrderedDict` keyed by `(layer, eid)`. On cache miss:
- If cache has free slots: load into free slot (no eviction).
- If cache is full: evict the globally LRU expert, load into freed slot.

#### Belady — Optimal (MIN)

Evicts the expert with the farthest future use (Belady's MIN algorithm).
Requires full trace knowledge (offline/oracle). This is the theoretical
lower bound on cache misses.

Pre-computes a next-use index: for every `(layer, expert)`, a sorted list of
all future access positions (`pos = step_idx * num_layers + layer`). Victim
selection searches **globally across all (layer, eid)** entries, avoiding
experts in the protected set (needed for current operation).

#### LFU — Least Frequently Used

Global frequency counters keyed by `(layer, eid)`. On eviction, removes the
expert with the lowest access count (ties broken by smallest `(layer, eid)`
tuple).

Supports optional **windowed mode**: reset all frequency counts every
`window_size` steps. This prevents stale frequency data from dominating in
non-stationary workloads.

#### StaticFreq — Global Frequency Ranking (Oracle)

Pre-computes access frequencies over the **entire** trace, then:

1. **Initial cache**: the top `cache_size` experts by global frequency.
2. **On miss**: evict the resident with the lowest global frequency.

High-frequency experts are effectively **pinned**: eviction always targets
the lowest-frequency resident. Requires oracle knowledge (full trace).
Answers: "how well can you do if you simply keep the most popular experts?"

### Prefetch Policies

#### NoPrefetch — Demand loads only

No proactive loading. All cache misses result in blocking demand loads.

#### OraclePrefetch — Layer-0 + 1-layer lookahead

Requires full trace knowledge. Two prefetch points:

1. **Layer 0**: At step start, prefetch all layer-0 experts not in cache.
   Overlaps with embed + stages 1-4a of layer 0.

2. **Layer L→L+1 lookahead**: At layer `L`, prefetch non-resident experts
   needed at layer `L+1`. Stored in `layers[L+1].prefetches` so the replay
   controller issues them before stage4b of layer `L`, overlapping with
   stage4b(L) + stages 1-4a(L+1).

   Eviction victims can come from **any layer** — the unified cache has no
   same-layer constraint.

### Naming Convention

Policies are named `<CachePolicy>-<PrefetchPolicy>`:

| Name | Eviction | Prefetch |
|------|----------|----------|
| LRU-None | Least recently used | Demand only |
| LRU-Oracle | Least recently used | Layer-0 + lookahead |
| Belady-None | Farthest future use | Demand only |
| Belady-Oracle | Farthest future use | Layer-0 + lookahead |
| LFU-None | Least frequently used | Demand only |
| LFU-Oracle | Least frequently used | Layer-0 + lookahead |
| StaticFreq-None | Global frequency (oracle) | Demand only |
| StaticFreq-Oracle | Global frequency (oracle) | Layer-0 + lookahead |

---

## Replay Controller

`replay_controller.py` implements `ReplayController`, which replays a
`GPUReplayTrace` on real GPU hardware using the MoEEngine's piecewise CUDA
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

### Scheduling Metadata — Multi-Sequence Replay

When the `GPUReplayTrace` is produced from a batched `ActivationTrace` (via
`build_trace.py` → policy simulator), each `StepTrace` carries optional
`StepScheduling` metadata describing the batch composition and scheduling events.

**This metadata is the ground truth for faithful replay.** The GPU replay loop
MUST consume this metadata to reconstruct the correct batch at each step —
dispatching the right decode sequences, prefill chunks, and continuations at
their traced token counts. Without this, compute kernel durations and prefetch
overlap windows do not reflect realistic serving, making policy timing
comparisons invalid.

**No-preemption policy:** Since replay is trace-driven (output lengths known in
advance), pages for the full sequence (prompt + output) are pre-allocated at
admission. No request is ever preempted or readmitted. Events are limited to
`complete`, `admit`, and `force_admit`.

Each `StepScheduling` contains:
- `active_requests`: list of `RequestScheduling` — per-request metadata including
  `request_id`, `seq_len`, `is_prefill`, `prefill_chunk_start`,
  `prefill_chunk_length`, `is_continuation`.
- `events`: list of scheduling events (`complete`, `admit`).

**Token type dispatch** (from `RequestScheduling`):
- `is_prefill=False` → decode (1 token, FlashInfer paged-KV attention)
- `is_prefill=True, is_continuation=False` → new prefill (FA3 self-attention)
- `is_prefill=True, is_continuation=True` → continuation chunk (FlashInfer
  paged-KV, offset = `prefill_chunk_start`)

The `ReplayController` exposes helpers:

```python
controller.get_step_scheduling(step)       # → StepScheduling or None
controller.get_newly_admitted_seq_ids(step) # → [request_id, ...]
```

**request_id → seq_id mapping:** The trace uses `request_id` (0..N-1, index
into conversations). The engine uses `seq_id` as a KV cache slot in
`[0, max_seqs)`. The replay loop maintains a free pool:

```python
free_seq_ids = set(range(max_seqs))
request_to_slot = {}

# On admit:
sid = free_seq_ids.pop()
request_to_slot[request_id] = sid

# On complete:
engine.free_seq(sid)
free_seq_ids.add(request_to_slot.pop(request_id))
```

The multi-sequence replay loop in `scripts/batched_replay.py` processes each
step's scheduling metadata, dispatches to `engine.mixed_step()` with the
correct decode/prefill/continuation split, and uses actual traced tokens.

### GPU Memory Budget for Replay

The `cache_size` parameter determines how many expert slots fit on GPU. The
memory equation is:

```
GPU_total = non_expert_model + expert_cache + KV_cache + overhead
80 GB     = ~3.0 GB          + cs × 336 MB  + KV       + 2.5 GB
```

Where:
- **non_expert_model**: attention weights, norms, embeddings, router (~3.0 GB for Mixtral)
- **expert_cache**: `cache_size × per_expert_bytes` (336 MB/slot for Mixtral)
- **KV_cache**: depends on batch size and sequence lengths
- **overhead** (2.5 GB): workspace buffers (128 MB), CUDA graphs (~64 MB), small
  tensors, and PyTorch caching allocator fragmentation. The allocator reserves
  memory in large blocks during initialization (up to 4+ GB transiently), which
  is why the overhead needs headroom beyond the ~0.4 GB of steady-state overhead.

Measured on H100 80 GB with Mixtral-8x7B (profiled with `cache_size=1` and
`cache_size=200`):

| Component | Steady-state | Peak (init) |
|---|---|---|
| Non-expert model (alloc) | 3.15 GB | 3.15 GB |
| Workspace + graphs + misc | 0.23 GB | 0.23 GB |
| Allocator fragmentation | 0.21 GB | 4.1 GB |
| **Total fixed overhead** | **3.59 GB** | **7.5 GB** |

The `compute_kv_budget_from_cache()` function in `build_trace.py` uses
`overhead_gb=2.5` on top of the computed `non_expert_bytes` (~3.0 GB), giving a
total fixed budget of ~5.5 GB. This leaves room for both the steady-state
allocator overhead and moderate KV cache.

Max feasible `cache_size` for Mixtral on single H100 (with minimal KV):

| overhead_gb | Max cache_size | % of 256 experts |
|---|---|---|
| 2.0 | 228 | 89.1% |
| 2.5 | 226 | 88.3% |
| 3.0 | 225 | 87.9% |

### Comparison: ReplayController vs ExpertOffloadEngine

| Aspect | ReplayController | ExpertOffloadEngine |
|--------|-----------------|---------------------|
| **Decision source** | Pre-computed `GPUReplayTrace` | Reactive (reads routing at runtime) |
| **Cache model** | Unified `cache_size` slots across all layers | Per-layer `experts_per_layer` + scratchpad |
| **Prefetching** | Scheduled per trace (2 categories: prefetch + demand) | None (pure demand loading) |
| **Eviction policy** | Encoded in trace (any policy) | Simple demand load (no policy) |
| **Async transfers** | Transfer stream + CUDA event sync | Blocking `synchronize()` per expert |
| **Use case** | Evaluating policy performance | Recording activation traces + baseline timing |
| **Hook points** | `begin_layer_prefetch` (layer 0 only) + `process_layer_replay` + `post_layer` (no-op) | `process_layer` (blocking) |

---

## End-to-End Usage

### Single-Sequence Replay (Development Only)

For simple single-sequence development and testing. Transfer counts are correct
but timing is not representative of batched serving — use `batched_replay.py`
for experiments.

```python
from moe_engine import MoEEngine
from gpu_replay_trace import ActivationTrace, GPUReplayTrace
from policy_simulator import Belady, OraclePrefetch, simulate
from replay_controller import ReplayController

# Step 1: Record a trace (ExpertOffloadEngine mode)
engine = MoEEngine("models/Mixtral-8x7B", experts_per_layer=2)
engine.capture_prefill_cuda_graph(total_token_sizes=[128])
engine.capture_mixed_cuda_graphs(total_token_sizes=[128])
with torch.inference_mode():
    engine.prefill_to_slot(0, input_ids)
    for step in range(50):
        logits = engine.mixed_step([0], [next_token], [], [])
        next_token = logits.argmax(-1)
engine.offload_engine.save_trace("expert_trace.json")

# Step 2: Simulate a policy (CPU-only)
trace = ActivationTrace.load("expert_trace.json")
dm_trace = simulate(Belady(), OraclePrefetch(), trace, cache_size=128)
assert not dm_trace.validate()
dm_trace.save("belady_oracle_cs128.json")

# Step 3: Replay on GPU
dm_trace = GPUReplayTrace.load("belady_oracle_cs128.json")
engine = MoEEngine("models/Mixtral-8x7B", cache_size=128)
controller = ReplayController(engine, dm_trace)
controller.setup()
engine.replay_controller = controller
with torch.inference_mode():
    engine.prefill_to_slot(0, input_ids)
    for step in range(len(dm_trace.steps)):
        logits = engine.mixed_step([0], [next_token], [], [])
        next_token = logits.argmax(-1)
engine.replay_controller = None
```

### Multi-Sequence Batched Replay (Experiments — Required for Valid Timing)

For realistic multi-sequence experiments, use `scripts/batched_replay.py`.
**This is the only replay mode that produces valid timing data.** It faithfully
recreates the batch compositions from Phase 1 (GPU batched collection),
handling the full pipeline: loading batched traces, simulating policies,
managing request_id→seq_id mapping, and replaying with mixed decode/prefill/
continuation batches. See [scripts/README.md](scripts/README.md) for usage.

```bash
python scripts/batched_replay.py \
    --model models/Mixtral-8x7B \
    --trace-dir datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b/cache70pct \
    --cache-size 179 --max-graph-size 512
```

### Comparing Policies

```python
from policy_simulator import (
    LRU, Belady, LFU, StaticFreq,
    NoPrefetch, OraclePrefetch, simulate,
)

caches = [LRU(), Belady(), LFU(), StaticFreq()]
prefetches = [NoPrefetch(), OraclePrefetch()]

for cp in caches:
    for pp in prefetches:
        name = f"{cp.__class__.__name__}-{pp.__class__.__name__}"
        dm = simulate(cp, pp, activation_trace, cache_size=128)
        assert not dm.validate()
        s = dm.summary()
        print(f"{name:25s}  demands={s['demand_loads']:4d}  "
              f"prefetches={s['prefetches']:4d}  evictions={s['evictions']:4d}")
```

---

## Tests

`tests/test_replay_policy.py` covers all of the above with 43 tests:

| Category | Tests | What's verified |
|----------|-------|-----------------|
| Serialization | 4 | TransferEvent round-trip, cross-layer evict, GPUReplayTrace save/load |
| Validation | 5 | Valid trace passes, missing expert, overcapacity, free slot addition |
| ActivationTrace | 7 | from_flat_trace, save/load round-trip, empty trace, router inputs round-trip, policies work with router inputs |
| LRU | 5 | No misses when all fit, eviction order, validate passes, no prefetches, cross-layer eviction |
| Belady-Oracle | 4 | Fewer misses than LRU, generates prefetches, validate passes, prefetches stored in target layer |
| LFU | 3 | Evicts least frequent, windowed reset, validate passes |
| StaticFreq | 5 | Initial cache has most frequent, evicts least frequent globally, high-freq pinning, validate, no prefetches with NoPrefetch |
| Mix-and-Match | 4 | Static-Oracle has prefetches, LRU-Oracle has prefetches, Belady-None no prefetches, all 8 combos validate |
| Summary | 2 | Counts correct, Oracle prefetches in summary |
| Unified Cache | 3 | Variable experts per layer, zero-experts layer, cache pressure across layers |

Run: `cd GPU_Offloading/src/MoE && python -m pytest tests/test_replay_policy.py -v`

---

## Files

| File | Role |
|------|------|
| `gpu_replay_trace.py` | `GPUReplayTrace`, `ActivationTrace`, `TransferEvent`, `LayerTrace`, `StepTrace` — trace format with serialization and validation |
| `policy_simulator.py` | `CachePolicy` (LRU, Belady, LFU, StaticFreq) × `PrefetchPolicy` (NoPrefetch, OraclePrefetch) + `simulate()` — orthogonal policy simulators |
| `replay_controller.py` | `ReplayController` — GPU replay with async CUDA streams |
| `expert_offload_engine.py` | `ExpertOffloadEngine` — reactive demand loading, trace recording |
| `tests/test_replay_policy.py` | 43 unit tests for trace format + all policy combos (unified cache) |
