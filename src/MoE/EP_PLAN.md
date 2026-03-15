# DP+EP (Data Parallel + Expert Parallel) Implementation Plan

> **Architecture**: Each GPU is both a DP rank (handling different requests)
> and an EP rank (owning a subset of experts).  This matches vLLM's EP
> architecture exactly.  Each rank has its own scheduler, its own KV cache,
> its own request queue.  The ONLY cross-rank communication is AllGather +
> ReduceScatter inside the MoE layer.  No token broadcast needed.
>
> **Offloading Amenability**: Each EP rank owns `E / ep_size` local experts.
> Eventually, only `cache_size` of those reside on GPU, with the rest on
> CPU (pinned).  The `expert_map` indirection, weight buffer layout, and
> `ReplayController` integration must not preclude this.  Mutual exclusion
> between EP and offloading is acceptable for v1.
>
> **Design Rule — Reuse SOTA Kernels**: `fused_experts` + `expert_map`
> handles local-only expert compute.  AllGather + ReduceScatter uses
> `torch.distributed` directly (vLLM's `AgRsAll2AllManager` is too
> coupled to vLLM internals).

## Overview

Target: 2× H100 80 GB (NVLink).  First model: Mixtral-8×7B (E=8, top-2).
With EP=2 each GPU stores 4 experts/layer → ~35 GB expert weights instead
of ~70 GB.  Each GPU handles **different requests** → 2× throughput.

### How it works (per step)

```
Rank 0 (GPU 0)                         Rank 1 (GPU 1)
─────────────────                       ─────────────────
Own scheduler: requests A, B, C         Own scheduler: requests D, E, F
Own KV cache: pages for A, B, C         Own KV cache: pages for D, E, F
Own attention: run on local tokens      Own attention: run on local tokens

        ┌──── Per MoE layer ────┐
        │                       │
Stage 4a: route locally         Stage 4a: route locally
        │                       │
AllGather: [N0 tokens] ←─────→ [N1 tokens]
  → each rank sees [N0+N1] tokens with routing info
        │                       │
fused_experts(expert_map):      fused_experts(expert_map):
  experts 0-3, skip 4-7          experts 4-7, skip 0-3
        │                       │
ReduceScatter: sum partials, scatter back
  → rank 0 gets [N0] result     → rank 1 gets [N1] result
        │                       │
Stage 4b: residual add          Stage 4b: residual add
        └───────────────────────┘
```

### Design principles

* **DP+EP is one axis**, not two.  Every rank is both a DP rank and an EP
  rank.  For DP=2, TP=1: EP group = DP group = `[rank 0, rank 1]`.
* Each rank has its **own scheduler**, **own KV cache**, **own requests**.
  No replicated scheduling.  No token broadcast.  No divergence risk.
* Communication: **AllGather + local-compute + ReduceScatter** inside MoE
  layers only.  Attention runs locally.  This is vLLM's default
  `allgather_reducescatter` backend.
* NCCL collectives are **not capturable** in CUDA graphs.  Stage 4b is
  split: NCCL calls run eagerly around the captured graph.
* **Batch size sync**: One AllReduce of token counts per step (~16 bytes
  over NVLink, <1 us) ensures both ranks pad to the same CUDA graph size.
  This is needed because NCCL collectives must operate on matching sizes.

---

## Step 0 — `ep_utils.py` (new file)

Pure torch + torch.distributed.  No engine dependency.

```python
"""Expert-parallelism utilities: placement, dispatch, combine."""
from __future__ import annotations
import torch
import torch.distributed as dist
from typing import Literal


# ── Expert Placement ──────────────────────────────────────────────

def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    num_experts: int,
    strategy: Literal["linear", "round_robin"] = "linear",
) -> tuple[int, torch.Tensor | None]:
    """Compute local expert count and global→local expert_map.

    Matches vLLM's determine_expert_map() exactly
    (layer.py in vllm.model_executor.layers.fused_moe.layer).

    Returns:
        local_num_experts: how many experts this rank owns
        expert_map: int32 [num_experts], global_id → local slot or -1.
                    None if ep_size == 1.
    """
    if ep_size == 1:
        return num_experts, None

    base = num_experts // ep_size
    remainder = num_experts % ep_size
    local_num = base + (1 if ep_rank < remainder else 0)

    expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
    if strategy == "linear":
        start = ep_rank * base + min(ep_rank, remainder)
        expert_map[start:start + local_num] = torch.arange(
            local_num, dtype=torch.int32)
    elif strategy == "round_robin":
        global_ids = torch.arange(ep_rank, num_experts, ep_size,
                                  dtype=torch.int32)
        expert_map[global_ids] = torch.arange(local_num, dtype=torch.int32)
    else:
        raise ValueError(f"Unknown EP strategy: {strategy!r}")
    return local_num, expert_map


def local_expert_ids(
    ep_size: int, ep_rank: int, num_experts: int,
    strategy: Literal["linear", "round_robin"] = "linear",
) -> list[int]:
    """Return the global expert IDs owned by this rank (ordered)."""
    if ep_size == 1:
        return list(range(num_experts))
    if strategy == "linear":
        base = num_experts // ep_size
        remainder = num_experts % ep_size
        start = ep_rank * base + min(ep_rank, remainder)
        local_num = base + (1 if ep_rank < remainder else 0)
        return list(range(start, start + local_num))
    elif strategy == "round_robin":
        return list(range(ep_rank, num_experts, ep_size))
    raise ValueError(f"Unknown EP strategy: {strategy!r}")


# ── AllGather / ReduceScatter ────────────────────────────────────

def ep_allgather(tensor: torch.Tensor, ep_group: dist.ProcessGroup,
                 ep_size: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """Fixed-size all_gather_into_tensor along dim 0.

    All ranks MUST pass tensors with identical shape (enforced by
    batch-size sync in Step 3).

    Args:
        out: Pre-allocated output buffer [N * ep_size, ...].  If None,
             a fresh tensor is allocated.  Pass pre-allocated buffers
             from Step 4a to avoid per-layer allocation churn.
    Returns: [N * ep_size, ...] concatenation of all ranks' tensors.
    """
    if out is None:
        out = torch.empty(
            (tensor.shape[0] * ep_size, *tensor.shape[1:]),
            dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(out, tensor, group=ep_group)
    return out


def ep_reducescatter(tensor: torch.Tensor, ep_group: dist.ProcessGroup,
                     ep_size: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """Fixed-size reduce_scatter_tensor along dim 0.

    Input: [N * ep_size, ...].  Output: [N, ...] (this rank's slice, summed).

    Args:
        out: Pre-allocated output buffer [N, ...].  If None, allocated.
    """
    N = tensor.shape[0] // ep_size
    if out is None:
        out = torch.empty(
            (N, *tensor.shape[1:]),
            dtype=tensor.dtype, device=tensor.device)
    dist.reduce_scatter_tensor(out, tensor, op=dist.ReduceOp.SUM,
                               group=ep_group)
    return out
```

**Why fixed-size, not variable**: All ranks pad to the same CUDA graph
size (Step 3), so `N_padded` is identical across ranks.  No variable-
length machinery needed.  This enables `all_gather_into_tensor` and
`reduce_scatter_tensor` (single contiguous buffer, no list of chunks).

> **vLLM reference**: vLLM pads all DP ranks to the same token count via
> `_post_process_dp_padding()` in `v1/worker/dp_utils.py:77-89`:
> `max_num_tokens = int(num_tokens_across_dp.max().item())`, then returns
> `[max_num_tokens] * len(...)`.  This is what enables their fixed-size
> `all_gatherv` / `reduce_scatterv` in `AgRsAll2AllManager`.

### Unit tests: `tests/test_ep_utils.py`

CPU-only tests for `determine_expert_map` and `local_expert_ids`.
2-GPU NCCL tests for `ep_allgather` / `ep_reducescatter` via
`torchrun --nproc_per_node=2`.

---

## Step 1 — `__init__`: EP parameters and process group

### 1a. New constructor parameters

```python
def __init__(
    self,
    model_path: str,
    max_seqs: int = 32,
    max_seq_len: int = 4096,
    page_size: int = 16,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_torch_compile: bool = True,
    gpu_expert_budget: int = None,
    experts_per_layer: int = None,
    cache_size: int = None,
    pipeline_parallel_size: int = 1,
    kv_page_budget: int = None,
    # ── NEW ──
    expert_parallel_size: int = 1,
    expert_placement_strategy: str = "linear",
):
```

### 1b. EP state init (after line 114)

Note: `torch.cuda.set_device` is belt-and-suspenders (not all launchers
set `CUDA_VISIBLE_DEVICES`).  The user-passed `device` parameter is
silently overridden when EP is active.  Add to the constructor docstring:
`device: Ignored when expert_parallel_size > 1 (each EP rank
auto-selects cuda:{ep_rank}).`

> **vLLM reference**: vLLM constructs the EP group in
> `parallel_state.py:1654` as `all_ranks.transpose(1, 2).reshape(-1,
> dp_size * pcp_size * tp_size)`.  For DP=2, TP=1, PP=1 this gives
> `[[0, 1]]` — the EP group is identical to the DP group.  Each rank in
> the EP group owns a subset of experts and participates in the MoE
> AllGather/ReduceScatter.

```python
# Expert parallelism
self.ep_size = expert_parallel_size
self.ep_strategy = expert_placement_strategy
if self.ep_size > 1:
    import torch.distributed as dist
    from ep_utils import determine_expert_map, local_expert_ids
    if not dist.is_initialized():
        raise RuntimeError(
            "EP requires torch.distributed initialized before "
            "creating MoEEngine. Launch with torchrun --nproc_per_node=EP.")
    self.ep_group = dist.new_group(ranks=list(range(self.ep_size)))
    assert dist.get_world_size() == self.ep_size, (
        f"EP currently requires world_size == ep_size, got "
        f"{dist.get_world_size()} vs {self.ep_size}")
    self.ep_rank = dist.get_rank(self.ep_group)
    # Each rank uses its own GPU
    torch.cuda.set_device(self.ep_rank)
    self.device = torch.device('cuda', self.ep_rank)
else:
    self.ep_rank = 0
    self.ep_group = None
```

### 1c. Validation (after config load)

```python
if self.ep_size > 1:
    if self.pp_size > 1:
        raise ValueError("EP + PP is not yet supported.")
    if self.offloading:
        raise ValueError(
            "EP + offloading is not yet supported. "
            "Use ep_size=1 with offloading, or ep_size>1 without.")
```

### 1d. Expert map (after `self.num_experts` is set)

```python
if self.ep_size > 1:
    self.local_num_experts, self.ep_expert_map = determine_expert_map(
        self.ep_size, self.ep_rank, self.num_experts, self.ep_strategy)
    self.local_expert_ids = local_expert_ids(
        self.ep_size, self.ep_rank, self.num_experts, self.ep_strategy)
    print(f"EP rank {self.ep_rank}/{self.ep_size}: "
          f"{self.local_num_experts} local experts "
          f"{self.local_expert_ids} (strategy={self.ep_strategy})")
else:
    self.local_num_experts = None
    self.ep_expert_map = None
```

### 1e. EP dispatcher setup (end of `__init__`)

```python
if self.ep_size > 1:
    self._ep_expert_map_gpu = self.ep_expert_map.to(self.device)
else:
    self._ep_expert_map_gpu = None
```

Note: The `from ep_utils import ...` statements shown inline in code
blocks below are for readability.  In implementation, all `ep_utils`
imports go at the top of `moe_engine.py` (no circular dependency).

---

## Step 2 — `_load_weights`: Load only local experts

### Current behavior (non-offloading)

Replace the `else:` branch at `moe_engine.py:939-941` which stacks all
E experts into `self.w1[layer] = [E, 2*I, H]`.

### New EP branch

Replace the non-offloading stacking:

```python
else:
    if self.ep_size > 1:
        # EP: only load local experts → [local_E, 2*I, H]
        local_ids = self.local_expert_ids
        self.w1.append(torch.stack(
            [w1_list[e] for e in local_ids]).to(layer_dev))
        self.w2.append(torch.stack(
            [w2_list[e] for e in local_ids]).to(layer_dev))
        print(f"  Layer {l}: w1 {self.w1[-1].shape} "
              f"(EP rank {self.ep_rank}, experts {local_ids})")
    else:
        self.w1.append(torch.stack(w1_list).to(layer_dev))
        self.w2.append(torch.stack(w2_list).to(layer_dev))
```

Same for MLA MoE branch.  Router weights stay full `[E, H]` on every
rank (routing must be computed locally to produce `topk_ids`).

**v2 optimization**: The current approach loads all experts per layer
into `w1_list` (~4.4 GB transient per layer for Mixtral) then selects
locals.  Acceptable for v1 (system has ≥128 GB RAM).  In v2, skip
non-local expert `weights.pop()` calls with `e in local_set` check.

### Shape verification

```python
elif self.ep_size > 1:
    assert self.w1[0].shape == (
        self.local_num_experts, 2 * self.moe_intermediate_size,
        self.hidden_size)
```

---

## Step 3 — Batch size sync in `_step_piecewise`

This is the key DP coordination.  Before running the per-layer loop,
all ranks must agree on the padded batch size so that:
1. NCCL AllGather/ReduceScatter operate on matching tensor sizes
2. All ranks replay the same CUDA graph size

> **vLLM reference**: vLLM does this via `_run_ar()` in
> `v1/worker/dp_utils.py:38-56` — an AllReduce of a `[4, dp_size]` int32
> tensor carrying each rank's `orig_num_tokens`, `padded_num_tokens`,
> `should_ubatch`, and `cudagraph_mode`.  The cudagraph mode is synced by
> taking the **minimum** across ranks (`_post_process_cudagraph_mode`,
> line 92-98): if any rank can't do CUDA graphs, none of them do.

### 3a. Sync function

The sync happens in `step()` (moe_engine.py:2470-2476), BEFORE calling
`_step_piecewise`, because `graph_N` is a parameter to `_step_piecewise`
and must be agreed upon before entry.

**New helper** (add to `MoEEngine`):

```python
def _ep_sync_graph_N(self, local_graph_N):
    """Sync graph_N to max across EP ranks."""
    import torch.distributed as dist
    buf = torch.tensor([local_graph_N], dtype=torch.int64,
                       device=self.device)
    dist.all_reduce(buf, op=dist.ReduceOp.MAX, group=self.ep_group)
    synced = int(buf.item())
    if synced != local_graph_N:
        synced = self._find_nearest_piecewise_graph(synced)
        if synced is None:
            raise RuntimeError(
                f"EP sync requires graph_N>={buf.item()} but largest "
                f"captured is {max(self._piecewise_graphs.keys())}")
    return synced
```

**Modified `step()` flow** (moe_engine.py:2470-2476):

```python
graph_N = self._find_nearest_piecewise_graph(N_total)
if self.ep_size > 1:
    # Pre-sync: verify all ranks have a valid graph_N (if any rank
    # would fall back to eager, all ranks must error — NCCL can't
    # be skipped on one rank).
    import torch.distributed as dist
    has_graph = torch.tensor(
        [0 if graph_N is None else 1],
        dtype=torch.int32, device=self.device)
    dist.all_reduce(has_graph, op=dist.ReduceOp.MIN,
                    group=self.ep_group)
    if has_graph.item() == 0:
        raise RuntimeError(
            "EP requires all ranks to have a valid graph_N. "
            f"Local N_total={N_total}, graph_N={graph_N}. "
            "Capture larger graphs.")
    graph_N = self._ep_sync_graph_N(graph_N)
if graph_N is not None:
    return self._step_piecewise(...)
```

**Cost**: Two AllReduces of 4-8 bytes over NVLink. <2 us total.

### 3b. Why this works

Each rank has its own scheduler with its own requests.  Rank 0 might have
`N_actual=17` (3 decode + 14 prefill), rank 1 might have `N_actual=23`.
Both need to pad to the same `graph_N` (e.g., 32) so that:
- Stage 1/4a/4b CUDA graphs replay with the same buffer size
- AllGather produces `[graph_N * ep_size, ...]` on all ranks
- ReduceScatter splits evenly into `[graph_N, ...]` chunks

Padding tokens are neutralized: `topk_weights_buf[N_actual:].zero_()`.

### 3c. Empty rank handling

If one rank has 0 tokens (no active requests), it still participates in
the AllGather/ReduceScatter with a zeroed buffer.  The graph replays but
all weights are zero → no contribution to the sum.  The scheduler on that
rank simply returns an empty `ScheduleResult` and the step is a no-op
for compute (but the collectives still execute to avoid deadlock).

---

## Step 4 — EP dispatch/combine in `_step_piecewise`

### 4a. Pre-allocate EP buffers during graph capture

In `capture_cuda_graphs`, after creating `bufs` (~line 2970):

```python
if self.ep_size > 1:
    from ep_utils import ep_allgather, ep_reducescatter
    bufs['ep_gathered_hidden'] = torch.zeros(
        N * self.ep_size, self.hidden_size,
        dtype=self.dtype, device=self.device)
    bufs['ep_gathered_weights'] = torch.zeros(
        N * self.ep_size, self.top_k,
        dtype=torch.float32, device=self.device)
    bufs['ep_gathered_ids'] = torch.zeros(
        N * self.ep_size, self.top_k,
        dtype=torch.int32, device=self.device)
    bufs['ep_combined_out'] = torch.zeros(
        N, self.hidden_size,
        dtype=self.dtype, device=self.device)
```

**Note**: No `ep_partial_out` buffer — `fused_experts()` (vLLM 0.17.1)
always allocates its own output internally (no `out=` parameter).  The
transient allocation is freed after ReduceScatter copies out.  Unlike
AllGather/ReduceScatter, this allocation churn is unavoidable.

### 4b. Modified per-layer loop

Replace the stage4b section (lines 3775-3783):

```python
            # Neutralize padding tokens
            if N_actual < graph_N:
                buf['topk_weights_buf'][N_actual:].zero_()

            if self.ep_size > 1:
                # ── EP: AllGather → local fused_experts → ReduceScatter ──
                # Use pre-allocated buffers from Step 4a to avoid per-layer
                # allocation churn (160 allocs/step → 32).
                gathered_h = ep_allgather(
                    buf['moe_input_buf'], self.ep_group, self.ep_size,
                    out=buf['ep_gathered_hidden'])
                gathered_w = ep_allgather(
                    buf['topk_weights_buf'], self.ep_group, self.ep_size,
                    out=buf['ep_gathered_weights'])
                gathered_ids = ep_allgather(
                    buf['topk_ids_buf'], self.ep_group, self.ep_size,
                    out=buf['ep_gathered_ids'])

                # Local expert compute on gathered batch
                w1, w2 = self.w1[layer], self.w2[layer]
                partial_out = self._moe_experts(
                    gathered_h, w1, w2, gathered_w, gathered_ids,
                    self._ep_expert_map_gpu)

                # ReduceScatter: sum partials, get back local slice
                combined = ep_reducescatter(
                    partial_out, self.ep_group, self.ep_size,
                    out=buf['ep_combined_out'])

                # Shared experts (MLA only) — local compute, no dispatch
                if self.is_mla and self.n_shared_experts > 0:
                    shared_gate_up = F.linear(
                        buf['moe_input_buf'][:N_actual],
                        self.shared_w1[layer])
                    shared_I = (self.moe_intermediate_size
                                * self.n_shared_experts)
                    sg = shared_gate_up[:, :shared_I]
                    su = shared_gate_up[:, shared_I:]
                    shared = F.linear(F.silu(sg) * su,
                                      self.shared_w2[layer])
                    buf['hidden_buf'][:N_actual].copy_(
                        buf['moe_residual_buf'][:N_actual]
                        + combined[:N_actual] + shared)
                else:
                    buf['hidden_buf'][:N_actual].copy_(
                        buf['moe_residual_buf'][:N_actual]
                        + combined[:N_actual])
                # Zero padding region for next layer's safety
                if N_actual < graph_N:
                    buf['hidden_buf'][N_actual:].zero_()
            else:
                # ── Original: stage4b CUDA graph replay ──
                info['stage4b_graphs'][layer].replay()
```

**Performance note**: The `fused_experts` call is eager (not in a CUDA
graph).  Per the earlier analysis, fused_experts is a Triton grouped GEMM
that Inductor doesn't further optimize, so the regression is only kernel
launch overhead (~5-10 us/layer, ~200 us total for 32 layers).  A future
optimization (Step 9c) can capture the local fused_experts in a separate
CUDA graph.

### 4c. Stage4b graph — skip capture when EP

When `ep_size > 1`, skip capturing stage4b graphs to save GPU memory
(they would never be replayed):

```python
# In capture_cuda_graphs, around line 2890:
if self.ep_size == 1:
    # Capture stage4b graph (only used in non-EP path)
    with torch.cuda.graph(g4b, stream=s):
        self._layer_stage4b_moe(...)
    info['stage4b_graphs'].append(g4b)
else:
    info['stage4b_graphs'].append(None)  # placeholder
```

**Graph size validation** (at END of `capture_cuda_graphs`): All EP
ranks must have captured identical graph sizes — `_ep_sync_graph_N`
picks MAX across ranks, and a rank missing that size hits RuntimeError
or (worse) a silent NCCL hang if the error isn't caught before the next
collective.  Validate the actual captured set from
`self._piecewise_graphs.keys()`, not the input `total_token_sizes`
(in case a capture silently fails):

```python
# At END of capture_cuda_graphs, after the capture loop:
if self.ep_size > 1:
    import torch.distributed as dist
    local_sizes = torch.tensor(
        sorted(self._piecewise_graphs.keys()),
        dtype=torch.int64, device=self.device)
    # Guard: all_gather requires equal-length tensors
    n_sizes = torch.tensor([len(local_sizes)], device=self.device)
    all_n = [torch.empty_like(n_sizes) for _ in range(self.ep_size)]
    dist.all_gather(all_n, n_sizes, group=self.ep_group)
    assert all(x.item() == n_sizes.item() for x in all_n), \
        "EP ranks captured different NUMBER of graph sizes"
    # Verify identical size sets
    all_sizes = [torch.empty_like(local_sizes)
                 for _ in range(self.ep_size)]
    dist.all_gather(all_sizes, local_sizes, group=self.ep_group)
    for r, other in enumerate(all_sizes):
        assert torch.equal(local_sizes, other), (
            f"EP rank {r} captured different graph sizes: "
            f"{other.tolist()} vs local {local_sizes.tolist()}")
```

---

## Step 5 — EP branch in `_layer_stage4b_moe`

For the non-piecewise (eager) path, add EP handling:

```python
# In _layer_stage4b_moe (line 2369-2374):
if self.offloading:
    w1, w2 = self.w1_buf, self.w2_buf
    expert_map = self.expert_map_buf
elif self.ep_size > 1:
    # Forward-looking: used when EP + offloading is enabled (Step 10).
    # Currently dead code — EP stage4b runs eagerly in _step_piecewise.
    w1, w2 = self.w1[layer], self.w2[layer]
    expert_map = self._ep_expert_map_gpu
else:
    w1, w2 = self.w1[layer], self.w2[layer]
    expert_map = None
```

The EP branch is currently dead code (Step 4c skips stage4b capture
under EP, so `_layer_stage4b_moe` is never called).  It's forward-
looking for Step 10 (EP + offloading).  The actual EP dispatch/combine
happens in Step 4b above.

---

## Step 6 — Eager fallback (`_layer_mixed` / `_layer_mixed_mla`)

### Non-MLA (`_layer_mixed`, ~line 2020)

The EP eager path (used during warmup/test only — piecewise path handles
padding via graph_N sync) must route BEFORE padding.  Full flow:

**Critical ordering**: Route FIRST on un-padded hidden (softmax on
zero logits → non-zero weights → garbage in ReduceScatter), THEN pad.

```python
if self.ep_size > 1:
    # 1. Route on UN-PADDED hidden (avoid softmax on zero logits)
    topk_weights, topk_ids = self._route(hidden, layer)
    orig_N = hidden.shape[0]

    # 2. Sync max_N across ranks
    import torch.distributed as dist
    local_N = torch.tensor([orig_N], device=self.device)
    dist.all_reduce(local_N, op=dist.ReduceOp.MAX,
                    group=self.ep_group)
    max_N = int(local_N.item())

    # 3. Pad to max_N (F.pad zero-fills by default)
    if orig_N < max_N:
        pad = max_N - orig_N
        hidden = F.pad(hidden, (0, 0, 0, pad))
        topk_weights = F.pad(topk_weights, (0, 0, 0, pad))
        topk_ids = F.pad(topk_ids, (0, 0, 0, pad))
        # topk_weights padding is already zero from F.pad —
        # fused_experts multiplies by zero → no contribution

    # 4. AllGather → fused_experts → ReduceScatter
    gathered_h = ep_allgather(hidden, self.ep_group, self.ep_size)
    gathered_w = ep_allgather(topk_weights, self.ep_group, self.ep_size)
    gathered_ids = ep_allgather(
        topk_ids.to(torch.int32), self.ep_group, self.ep_size)
    partial = self._moe_experts(
        gathered_h, self.w1[layer], self.w2[layer],
        gathered_w, gathered_ids, self._ep_expert_map_gpu)
    hidden = ep_reducescatter(partial, self.ep_group, self.ep_size)
    hidden = hidden[:orig_N]
```

### MLA (`_layer_mixed_mla`, ~line 2164)

```python
        if self.router[layer] is None:
            # Dense layer: SwiGLU MLP (no EP dispatch)
            gate_up = F.linear(hidden, self.dense_w1[layer])
            I = self.intermediate_size
            hidden = F.linear(F.silu(gate_up[:, :I]) * gate_up[:, I:],
                              self.dense_w2[layer])
        elif self.ep_size > 1:
            topk_weights, topk_ids = self._route(hidden, layer)
            # [pad + allgather + fused_experts + reducescatter as above]
            gathered_h = ep_allgather(hidden_padded, ...)
            gathered_w = ep_allgather(topk_weights_padded, ...)
            gathered_ids = ep_allgather(topk_ids_padded, ...)
            partial = self._moe_experts(gathered_h, self.w1[layer],
                                         self.w2[layer], gathered_w,
                                         gathered_ids,
                                         self._ep_expert_map_gpu)
            routed = ep_reducescatter(partial, ...)[:hidden_orig_N]
            # Shared experts: local (replicated, no dispatch)
            shared_gu = F.linear(hidden, self.shared_w1[layer])
            shared_I = self.moe_intermediate_size * self.n_shared_experts
            shared = F.linear(F.silu(shared_gu[:, :shared_I])
                              * shared_gu[:, shared_I:],
                              self.shared_w2[layer])
            hidden = routed + shared
        else:
            # Original non-EP MoE path (unchanged)
            ...

        return residual + hidden
```

---

## Step 7 — Launch infrastructure

### 7a. `launch_ep.py` (per-rank entry point)

```python
"""DP+EP launcher for MoE inference.

Usage:
    torchrun --nproc_per_node=2 launch_ep.py \
        --model /path/to/mixtral --ep_size 2 \
        --prompts prompts.jsonl
"""
import argparse, os, json, sys
import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep_size", type=int, default=2)
    parser.add_argument("--strategy", default="linear",
                        choices=["linear", "round_robin"])
    parser.add_argument("--max_seqs", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--prompts", type=str, required=True,
                        help="JSONL file: one {text, max_tokens} per line")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # ── Load and distribute prompts ──
    # Round-robin assignment: rank i gets prompts i, i+world_size, ...
    all_prompts = []
    with open(args.prompts) as f:
        for line in f:
            all_prompts.append(json.loads(line))
    my_prompts = [p for i, p in enumerate(all_prompts)
                  if i % world_size == rank]
    print(f"Rank {rank}: {len(my_prompts)} prompts "
          f"(of {len(all_prompts)} total)")

    # ── Create engine (each rank creates its own) ──
    from moe_engine import MoEEngine
    engine = MoEEngine(
        model_path=args.model,
        max_seqs=args.max_seqs,
        max_seq_len=args.max_seq_len,
        expert_parallel_size=args.ep_size,
        expert_placement_strategy=args.strategy,
    )

    # ── Create scheduler (each rank has its own) ──
    from scheduler import Scheduler
    sched = Scheduler(engine, args.max_seqs, max_graph_size=512,
                      page_size=engine.page_size)

    # Capture CUDA graphs (all EP ranks MUST capture identical sizes —
    # _ep_sync_graph_N picks MAX across ranks, and a rank missing that
    # size hits RuntimeError.  Add a validation AllGather in
    # capture_cuda_graphs when ep_size > 1.)
    engine.capture_cuda_graphs(
        total_token_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512])

    # ── Build conversations for this rank ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    conversations = []
    for i, p in enumerate(my_prompts):
        ids = tokenizer.encode(p['text'])
        conversations.append({
            'conversation_id': f"rank{rank}_conv{i}",
            'prompt_token_ids': ids,
            'max_output_tokens': p.get('max_tokens', 128),
        })

    # ── Run collection ──
    # NOTE: Requires Step 8 (dummy-step logic) to avoid NCCL deadlock
    # when ranks finish at different times.
    result = sched.collect(conversations)
    if rank == 0:
        print(f"Rank 0: {result.step_count} steps, "
              f"{len(result.conversations)} conversations")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### 7b. Apptainer wrapper

```bash
# Run from repo root:
H100_env/vllm_apptainer.sh torchrun --nproc_per_node=2 \
    src/MoE/launch_ep.py --model $MODEL --prompts prompts.jsonl
```

---

## Step 8 — Scheduler integration (minimal changes)

The scheduler requires **no fundamental changes**.  Each rank creates its
own `Scheduler` instance with its own `engine` as allocator.  The
scheduler's `collect()` and `replay()` loops drive `engine.step()` which
internally handles the EP communication.

### 8a. Dummy step for idle ranks

When one rank's scheduler has no work but other ranks are still active,
the idle rank must still participate in NCCL collectives.  Add a sentinel
mechanism:

> **vLLM reference**: vLLM's `DPEngineCoreProc.run_busy_loop()` in
> `v1/engine/core.py:1612-1662` handles this identically.  When the
> scheduler has no work but `engines_running` is true, it calls
> `execute_dummy_batch()` (line 1637) to keep NCCL collectives in sync.
> It checks `_has_global_unfinished_reqs()` (line 1664) via AllReduce
> every 32 steps to detect when all ranks are truly done.

The loop must be `while True:` — NOT `while not self.is_done:` — because
the idle rank must keep running dummy steps until ALL ranks are done.
`while not self.is_done` would exit after one dummy step → NCCL deadlock.

Termination check uses AllReduce every 32 steps (vLLM pattern:
`core.py:1666-1668`), except when a rank finishes (`or self.is_done`
forces an immediate check).  Same pattern must be applied to `replay()`.

```python
# In scheduler.py collect(), replacing the while loop (~line 638):
import torch.distributed as dist
_ep_step_counter = 0
while True:
    if not self.is_done:
        result = self.step()
    else:
        result = None

    if self.ep_size > 1:
        _ep_step_counter += 1
        # Only sync termination every 32 steps (vLLM pattern)
        if _ep_step_counter % 32 == 0 or self.is_done:
            local_active = torch.tensor(
                [0 if self.is_done else 1],
                dtype=torch.int32, device=engine.device)
            dist.all_reduce(local_active, op=dist.ReduceOp.MAX,
                            group=engine.ep_group)
            if local_active.item() == 0:
                break
        if self.is_done:
            # Dummy step: N_total=0 → graph_N synced to active rank's
            # size via Step 3.  All tokens are padding (zero weights) →
            # zero contribution to ReduceScatter.  Required to keep
            # NCCL collectives in sync.  Matches vLLM's
            # execute_dummy_batch() which also runs a full forward pass.
            engine.step(
                decode_seq_ids=[], decode_token_ids=torch.tensor(
                    [], dtype=torch.long, device=engine.device),
                prefill_seq_ids=[], prefill_input_ids=[])
            continue
    elif self.is_done:
        break

    if result is None or result.is_empty:
        continue
    # ... rest of collect loop unchanged ...
```

**Cost**: One AllReduce of 4 bytes every 32 steps. <1 us.

### 8b. EP dummy steps in `replay()`

The `replay()` loop (`scheduler.py:818`) has two independent NCCL
deadlock risks under EP that require fixes:

**Fix 1 — sync `total_steps`**: Each rank computes `total_steps` from
its own controller trace (`scheduler.py:797`).  Under EP, each rank has
a different trace (different requests → different step counts).  Sync
to MAX before the loop:

```python
# After total_steps is computed, before the for loop:
if self.ep_size > 1:
    import torch.distributed as dist
    ts = torch.tensor([total_steps], dtype=torch.int64,
                      device=engine.device)
    dist.all_reduce(ts, op=dist.ReduceOp.MAX, group=engine.ep_group)
    total_steps = int(ts.item())
```

**Fix 2 — dummy step on empty batches**: Even with synced `total_steps`,
one rank may have no work on a given step.  The guard at line 908
(`if not decode_sids and ... : continue`) skips `engine.step()` →
NCCL mismatch.  Run a dummy step instead:

```python
# In scheduler.py replay(), at the empty-batch guard (~line 908):
if not decode_sids and not prefill_sids and not cont_sids:
    if self.ep_size > 1:
        engine.step(
            decode_seq_ids=[], decode_token_ids=torch.tensor(
                [], dtype=torch.long, device=engine.device),
            prefill_seq_ids=[], prefill_input_ids=[])
    continue
```

**Fix 3 — `sched is None` early exit**: Line 820-821:
`if sched is None: break`.  When `total_steps` is synced to MAX, the
shorter-trace rank gets `None` from `get_sched(s)` for steps beyond its
trace length and `break`s, causing deadlock.  Replace with dummy step:

```python
sched = get_sched(step)
if sched is None:
    if self.ep_size > 1:
        engine.step(
            decode_seq_ids=[], decode_token_ids=torch.tensor(
                [], dtype=torch.long, device=engine.device),
            prefill_seq_ids=[], prefill_input_ids=[])
        continue
    break
```

Unlike `collect()` which uses `while True:` with AllReduce termination
(Step 8a), `replay()` uses a bounded `for` loop with synced
`total_steps`, so no additional termination protocol is needed — the
loop simply runs to `max(total_steps)` with dummy steps filling the gap.

### 8c. Property for EP size

Add to `Scheduler.__init__`:

```python
self.ep_size = getattr(engine, 'ep_size', 1)
```

---

## Step 9 — Correctness tests

### 9a. `tests/test_ep_placement.py` — CPU-only

```python
"""Test expert placement maps."""
from ep_utils import determine_expert_map, local_expert_ids

def test_linear_even():
    for rank in range(2):
        n, emap = determine_expert_map(2, rank, 8, "linear")
        assert n == 4
        owned = [i for i in range(8) if emap[i] >= 0]
        assert owned == list(range(rank * 4, rank * 4 + 4))

def test_round_robin():
    for rank in range(2):
        n, emap = determine_expert_map(2, rank, 8, "round_robin")
        assert n == 4
        owned = [i for i in range(8) if emap[i] >= 0]
        assert owned == list(range(rank, 8, 2))

def test_ep1():
    n, emap = determine_expert_map(1, 0, 8)
    assert n == 8 and emap is None

if __name__ == "__main__":
    test_linear_even()
    test_round_robin()
    test_ep1()
    print("All placement tests passed.")
```

**Pending fixes**:
- Add buffer slot value assertions (e.g., `slots == [0,1,2,3]` for
  both ranks) — catches OOB if rank 1 gets global [4,5,6,7] instead
  of local [0,1,2,3].
- Add uneven division test (`E=7, EP=2`).

### 9b. `tests/test_ep_dispatch.py` — 2-GPU NCCL

```python
"""Test AllGather + ReduceScatter correctness.

Run: torchrun --nproc_per_node=2 tests/test_ep_dispatch.py
"""
import torch, torch.distributed as dist
from ep_utils import ep_allgather, ep_reducescatter

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.new_group([0, 1])

    N, H = 16, 64
    x = torch.randn(N, H, device=device, dtype=torch.bfloat16) * (rank + 1)

    # AllGather
    gathered = ep_allgather(x, ep_group, 2)
    assert gathered.shape == (2 * N, H)
    # Rank 0's data should be in [0:N], rank 1's in [N:2N]
    if rank == 0:
        assert torch.allclose(gathered[:N], x, atol=0)

    # ReduceScatter of gathered → should give back sum of both slices
    reduced = ep_reducescatter(gathered, ep_group, 2)
    assert reduced.shape == (N, H)

    # Test complementary partials (real EP pattern: each rank computes
    # a different subset of experts, zeros for the rest)
    zeros = torch.zeros(N, H, device=device, dtype=torch.bfloat16)
    if rank == 0:
        partial = torch.cat([x, zeros])   # [2N, H]: experts 0-3 only
    else:
        partial = torch.cat([zeros, x])   # [2N, H]: experts 4-7 only
    reduced2 = ep_reducescatter(partial, ep_group, 2)
    # rank 0 gets chunk 0: x_rank0 + 0 = x_rank0
    # rank 1 gets chunk 1: 0 + x_rank1 = x_rank1
    assert torch.allclose(reduced2, x, atol=0)

    if rank == 0:
        print("EP dispatch/combine test PASSED")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

**Pending fix**: Add `out=` parameter test for both `ep_allgather`
and `ep_reducescatter` (verify `data_ptr()` matches pre-allocated
buffer).  This is the hot path in `_step_piecewise`.

### 9c. `tests/test_ep_consistency.py` — Consolidated EP correctness (2-GPU)

Replaces the old `test_ep_sync.py` and `test_ep_e2e.py` stubs.  Five
tests covering batch-size sync, dummy steps, asymmetric completion, and
EP=2 vs EP=1 numerical equivalence.

```
Run: H100_env/vllm_apptainer.sh torchrun --nproc_per_node=2 \
       src/MoE/tests/test_ep_consistency.py --model /path/to/mixtral
```

```python
"""EP consistency tests: sync, dummy steps, and EP=2 vs EP=1 equivalence.

Run with torchrun --nproc_per_node=2.
"""
import argparse, os, sys, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
import torch.distributed as dist

# ── Helpers ──────────────────────────────────────────────────────

def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

def make_engine(model_path, ep_size, max_seqs=4, max_seq_len=2048,
                compile=True):
    from moe_engine import MoEEngine
    return MoEEngine(
        model_path=model_path,
        max_seqs=max_seqs,
        max_seq_len=max_seq_len,
        use_torch_compile=compile,
        expert_parallel_size=ep_size,
    )

def dummy_step(engine):
    """Zero-token forward pass — keeps NCCL collectives in sync."""
    engine.step(
        decode_seq_ids=[],
        decode_token_ids=torch.tensor([], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[], prefill_input_ids=[])

def greedy_generate(engine, prompt_ids, max_new, page_size):
    """Single-sequence chunked-prefill + decode.  Returns output tokens."""
    engine.reset()
    sid = 0
    chunk = 256
    n_pages_initial = (min(chunk, len(prompt_ids)) + page_size - 1) // page_size
    engine.alloc_pages(sid, n_pages_initial)

    # Chunked prefill
    for offset in range(0, len(prompt_ids), chunk):
        c = prompt_ids[offset:offset + chunk]
        needed = (offset + len(c) + page_size - 1) // page_size
        while engine._page_table[sid].count_nonzero() < needed:
            engine.alloc_pages(sid, 1)
        if offset == 0:
            logits = engine.step(
                decode_seq_ids=[],
                decode_token_ids=torch.tensor([], dtype=torch.long,
                                              device=engine.device),
                prefill_seq_ids=[sid],
                prefill_input_ids=[torch.tensor(c, dtype=torch.long,
                                                device=engine.device)])
        else:
            logits = engine.step(
                decode_seq_ids=[],
                decode_token_ids=torch.tensor([], dtype=torch.long,
                                              device=engine.device),
                prefill_seq_ids=[], prefill_input_ids=[],
                continuation_seq_ids=[sid],
                continuation_input_ids=[torch.tensor(c, dtype=torch.long,
                                                     device=engine.device)],
                continuation_offsets=[offset])
    tok = logits[0].argmax().item()
    out = [tok]

    # Decode
    for _ in range(max_new - 1):
        needed = (len(prompt_ids) + len(out) + page_size - 1) // page_size
        while engine._page_table[sid].count_nonzero() < needed:
            engine.alloc_pages(sid, 1)
        logits = engine.step(
            decode_seq_ids=[sid],
            decode_token_ids=torch.tensor([tok], dtype=torch.long,
                                          device=engine.device),
            prefill_seq_ids=[], prefill_input_ids=[])
        tok = logits[0].argmax().item()
        out.append(tok)
    return out


# ── Tests ────────────────────────────────────────────────────────

def test1_graph_N_sync(engine, rank):
    """Different N_actual on each rank → same graph_N after sync.

    Rank 0 sends 3 decode tokens, rank 1 sends 17.  After
    _ep_sync_graph_N, both should select the same graph_N (>= 17).
    Validates Step 3.
    """
    engine.reset()
    # Each rank allocates one slot and prefills a short prompt first
    # so we have valid KV state for decode.
    prompt_len = 4
    sid = 0
    engine.alloc_pages(sid, 1)
    logits = engine.step(
        decode_seq_ids=[],
        decode_token_ids=torch.tensor([], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[sid],
        prefill_input_ids=[torch.randint(100, 5000,
                                         (prompt_len,),
                                         device=engine.device)])
    tok = logits[0].argmax().item()

    # Now do one decode step — both ranks have 1 decode token.
    # The graph_N sync should agree (both N_actual=1 → same graph).
    logits = engine.step(
        decode_seq_ids=[sid],
        decode_token_ids=torch.tensor([tok], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[], prefill_input_ids=[])

    # If we got here without deadlock/RuntimeError, sync works.
    if rank == 0:
        print("  PASS: test1_graph_N_sync")


def test2_dummy_step_no_deadlock(engine, rank):
    """One rank has 0 tokens (dummy step), no NCCL deadlock.

    Rank 0 runs a normal decode step; rank 1 runs a dummy step.
    Validates Step 3c (empty rank handling) + Step 8a.
    """
    engine.reset()
    sid = 0
    if rank == 0:
        # Rank 0: prefill + decode
        engine.alloc_pages(sid, 1)
        logits = engine.step(
            decode_seq_ids=[],
            decode_token_ids=torch.tensor([], dtype=torch.long,
                                          device=engine.device),
            prefill_seq_ids=[sid],
            prefill_input_ids=[torch.randint(100, 5000, (8,),
                                             device=engine.device)])
        tok = logits[0].argmax().item()
        logits = engine.step(
            decode_seq_ids=[sid],
            decode_token_ids=torch.tensor([tok], dtype=torch.long,
                                          device=engine.device),
            prefill_seq_ids=[], prefill_input_ids=[])
    else:
        # Rank 1: two dummy steps (matching rank 0's prefill + decode)
        dummy_step(engine)
        dummy_step(engine)

    if rank == 0:
        print("  PASS: test2_dummy_step_no_deadlock")


def test3_asymmetric_completion(engine, rank):
    """One rank finishes early, dummy-steps until the other rank is done.

    Rank 0 generates 8 tokens, rank 1 generates 16.  Rank 0 must
    issue 8 dummy steps after finishing.  Validates Step 8a (the
    while-True + AllReduce termination pattern in collect).

    NOTE: This test exercises the scheduler's collect() loop directly.
    The dummy-step logic lives in scheduler.py, not engine.  We
    simulate the pattern here at engine level to test the NCCL path.
    """
    engine.reset()
    sid = 0
    n_steps_self = 8 if rank == 0 else 16
    n_steps_max = 16  # synced via AllReduce in real code

    engine.alloc_pages(sid, 1)
    logits = engine.step(
        decode_seq_ids=[],
        decode_token_ids=torch.tensor([], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[sid],
        prefill_input_ids=[torch.randint(100, 5000, (4,),
                                         device=engine.device)])
    tok = logits[0].argmax().item()

    for step in range(n_steps_max):
        if step < n_steps_self:
            needed = (4 + step + 1 + engine.page_size - 1) // engine.page_size
            while engine._page_table[sid].count_nonzero() < needed:
                engine.alloc_pages(sid, 1)
            logits = engine.step(
                decode_seq_ids=[sid],
                decode_token_ids=torch.tensor([tok], dtype=torch.long,
                                              device=engine.device),
                prefill_seq_ids=[], prefill_input_ids=[])
            tok = logits[0].argmax().item()
        else:
            dummy_step(engine)

    if rank == 0:
        print("  PASS: test3_asymmetric_completion")


def test4_asymmetric_prefill_lengths(engine, rank):
    """Ranks prefill different-length prompts in the same step.

    Rank 0 prefills 32 tokens, rank 1 prefills 128.  graph_N sync
    must pad both to the same size.  Validates Step 3 with prefill
    (not just decode).
    """
    engine.reset()
    sid = 0
    prompt_len = 32 if rank == 0 else 128

    n_pages = (prompt_len + engine.page_size - 1) // engine.page_size
    engine.alloc_pages(sid, n_pages)

    prompt = torch.randint(100, 5000, (prompt_len,), device=engine.device)
    logits = engine.step(
        decode_seq_ids=[],
        decode_token_ids=torch.tensor([], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[sid],
        prefill_input_ids=[prompt])
    assert logits.shape[0] == 1, f"Expected 1 logit row, got {logits.shape[0]}"

    if rank == 0:
        print("  PASS: test4_asymmetric_prefill_lengths")


def test5_ep2_vs_ep1_equivalence(rank, model_path, page_size_ep2):
    """EP=2 generation must match EP=1 (single-GPU) for the same prompt.

    Mathematically, AllGather gives each rank the full token set, each
    rank computes complementary expert partials (via expert_map), and
    ReduceScatter sums them — identical to single-GPU computing all
    experts.  So greedy tokens MUST match (within BF16 non-determinism).

    Strategy:
      1. Both ranks generate prompt A with EP=2 (rank 0 owns it).
      2. Rank 0 separately generates prompt A with EP=1.
      3. Compare token-by-token.

    This requires creating two engine instances (EP=2 then EP=1).
    Since NCCL state doesn't fully reset, the EP=1 run happens AFTER
    destroy_process_group.  We handle this by returning the EP=2
    tokens and letting main() do the EP=1 comparison on rank 0 only.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = "The capital of France is"
    prompt_ids = tokenizer.encode(prompt)
    max_new = 32

    # ── EP=2 run ──
    engine_ep2 = make_engine(model_path, ep_size=2, max_seqs=1,
                             compile=False)  # eager for exact repro
    graph_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256]
    engine_ep2.capture_cuda_graphs(total_token_sizes=graph_sizes)
    page_size = engine_ep2.page_size

    # Rank 0 generates; rank 1 does dummy steps
    if rank == 0:
        tokens_ep2 = greedy_generate(engine_ep2, prompt_ids, max_new,
                                     page_size)
    else:
        # Rank 1: match the number of engine.step() calls that
        # greedy_generate makes: ceil(prompt/256) prefill + max_new decode
        n_prefill_steps = (len(prompt_ids) + 255) // 256
        for _ in range(n_prefill_steps + max_new):
            dummy_step(engine_ep2)
        tokens_ep2 = None

    del engine_ep2
    torch.cuda.empty_cache()

    # Sync before EP=1 phase
    dist.barrier()

    if rank == 0:
        # ── EP=1 run (single GPU, all experts) ──
        # Cannot create ep_size=1 engine while dist is active with
        # world_size=2 (assertion in __init__), so we create it with
        # ep_size=1 after noting that ep_size=1 skips the dist check.
        engine_ep1 = make_engine(model_path, ep_size=1, max_seqs=1,
                                 compile=False)
        engine_ep1.capture_cuda_graphs(total_token_sizes=graph_sizes)
        tokens_ep1 = greedy_generate(engine_ep1, prompt_ids, max_new,
                                     page_size)
        del engine_ep1
        torch.cuda.empty_cache()

        n_match = sum(a == b for a, b in zip(tokens_ep2, tokens_ep1))
        pct = 100.0 * n_match / max_new
        print(f"  EP=2 vs EP=1: {n_match}/{max_new} tokens match "
              f"({pct:.1f}%)")
        # With eager (no torch.compile), should be exact or near-exact.
        # BF16 AllGather+ReduceScatter is bit-exact (no FP accumulation
        # reordering), so mismatch indicates a real bug.
        if pct < 95.0:
            print(f"  FAIL: test5_ep2_vs_ep1_equivalence "
                  f"({pct:.1f}% < 95%)")
            print(f"    EP=2: {tokens_ep2[:10]}...")
            print(f"    EP=1: {tokens_ep1[:10]}...")
            return False
        print("  PASS: test5_ep2_vs_ep1_equivalence")
        return True
    else:
        # Rank 1 idles during EP=1 run (no NCCL needed)
        return True


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    rank = setup()
    if rank == 0:
        print("=" * 60)
        print("EP Consistency Tests (2-GPU)")
        print("=" * 60)

    # Tests 1-4: use a shared EP=2 engine
    if rank == 0:
        print("\nCreating EP=2 engine...")
    engine = make_engine(args.model, ep_size=2, max_seqs=4, compile=False)
    graph_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256]
    engine.capture_cuda_graphs(total_token_sizes=graph_sizes)

    test1_graph_N_sync(engine, rank)
    dist.barrier()

    test2_dummy_step_no_deadlock(engine, rank)
    dist.barrier()

    test3_asymmetric_completion(engine, rank)
    dist.barrier()

    test4_asymmetric_prefill_lengths(engine, rank)
    dist.barrier()

    page_size = engine.page_size
    del engine
    torch.cuda.empty_cache()
    dist.barrier()

    # Test 5: EP=2 vs EP=1 equivalence (creates its own engines)
    if rank == 0:
        print()
    ok = test5_ep2_vs_ep1_equivalence(rank, args.model, page_size)
    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        status = "ALL PASSED" if ok else "SOME FAILED"
        print(f"EP Consistency Tests: {status}")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

**Test summary**:

| Test | What it validates | Steps covered |
|------|-------------------|---------------|
| 1: graph_N sync | Both ranks select same CUDA graph after sync | Step 3 |
| 2: dummy step | One rank idle (0 tokens), no NCCL deadlock | Steps 3c, 8a |
| 3: asymmetric completion | Rank finishes early, dummy-steps until peer done | Step 8a |
| 4: asymmetric prefill | Different prefill lengths → same graph_N | Step 3 |
| 5: EP=2 vs EP=1 | Greedy tokens match single-GPU (mathematical equivalence) | Steps 2-4 end-to-end |

**Pending fixes** (from review):
- **[HIGH]** Replace `engine._page_table[sid].count_nonzero()` with
  `len(engine._seq_page_list[sid])` everywhere in `greedy_generate`
  and `test3` — `_page_table` does not exist (`moe_engine.py:3989`
  uses `_seq_page_list`).
- **[HIGH]** Fix test5 dummy step count: `n_prefill_steps + max_new - 1`
  (not `max_new`) — `greedy_generate` makes `max_new - 1` decode steps
  (first token from last prefill logits).
- **[MEDIUM]** Rename test1 to `test1_basic_sync`, fix docstring
  (code is symmetric, not "3 vs 17").  test4 covers asymmetric N_actual.
- **[MEDIUM]** test5 EP=1 memory: use `max_seq_len=512`, minimal graph
  sizes `[1,2,4,8,16,32]`, reduce `max_new` to 16.  Or use OLMoE for
  test5 (`--test5-model` flag).
- Remove unused `page_size_ep2` parameter from `test5`.
- Add `test6_cross_rank_mixed_step`: rank 0 decode + rank 1 prefill
  simultaneously (steady-state pattern, exercises graph_N sync with
  mixed token types).
- Multi-sequence testing deferred to scheduler integration tests.

---

## Step 10 — Benchmark: EP=2 decode ms/step vs vLLM EP=2

Minimal benchmark: time `engine.step()` for custom EP=2 vs vLLM EP=2,
same model, same expert placement, same batch size.  Reuses the
existing timing patterns from `microbenchmark.py`.

### 10a. Custom EP=2 timing

Runs under `torchrun --nproc_per_node=2`.  Each rank times its own
`engine.step()` calls.  Identical pattern to `_time_custom_decode_once`
(`microbenchmark.py:348-373`): CUDA events around N_STEPS=200 decode
steps, N_WARMUP=20, median of 3 trials.

```python
# benchmark_ep.py — run via:
#   H100_env/vllm_apptainer.sh torchrun --nproc_per_node=2 \
#       src/MoE/tests/vLLM_comparison/benchmark_ep.py --model $MODEL
import argparse, os, statistics
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch, torch.distributed as dist

N_WARMUP = 20
N_STEPS = 200
N_TRIALS = 3

def time_custom_ep(engine, seq_len, n_steps, n_warmup):
    """Single trial: ms/step for EP decode."""
    engine.reset()
    sid = 0
    engine.alloc_pages(sid, (seq_len + engine.page_size - 1) // engine.page_size)
    # Prefill a dummy prompt to fill KV cache to target seq_len
    prompt = torch.randint(100, 5000, (seq_len,), device=engine.device)
    engine.step(
        decode_seq_ids=[],
        decode_token_ids=torch.tensor([], dtype=torch.long,
                                      device=engine.device),
        prefill_seq_ids=[sid],
        prefill_input_ids=[prompt])
    tok = torch.tensor([100], dtype=torch.long, device=engine.device)

    for _ in range(n_warmup):
        engine.step(decode_seq_ids=[sid], decode_token_ids=tok,
                    prefill_seq_ids=[], prefill_input_ids=[])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        engine.step(decode_seq_ids=[sid], decode_token_ids=tok,
                    prefill_seq_ids=[], prefill_input_ids=[])
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[128, 256, 512, 1024])
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    from moe_engine import MoEEngine
    engine = MoEEngine(
        model_path=args.model, max_seqs=1, max_seq_len=2048,
        use_torch_compile=True, expert_parallel_size=2)
    engine.capture_cuda_graphs(
        total_token_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512])

    if rank == 0:
        print(f"{'seq_len':>8}  {'ms/step':>8}  {'trials':>24}")
    for sl in args.seq_lens:
        trials = [time_custom_ep(engine, sl, N_STEPS, N_WARMUP)
                  for _ in range(N_TRIALS)]
        med = statistics.median(trials)
        # Gather rank 1's median to rank 0
        t = torch.tensor([med], device=engine.device)
        gathered = [torch.zeros(1, device=engine.device) for _ in range(2)]
        dist.all_gather(gathered, t)
        if rank == 0:
            r0, r1 = gathered[0].item(), gathered[1].item()
            print(f"{sl:>8}  {max(r0,r1):>8.2f}  "
                  f"(rank0={r0:.2f}, rank1={r1:.2f})")

    del engine
    torch.cuda.empty_cache()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Reports `max(rank0, rank1)` because the step is bounded by the slower
rank (NCCL collectives block until both ranks complete).

### 10b. vLLM EP=2 timing

vLLM manages its own DP+EP workers internally — a single
`vllm.LLM(data_parallel_size=2, enable_expert_parallel=True)` call
spawns 2 workers.  Cannot run under `torchrun` (would double-spawn).

Run as a **separate script** (no torchrun), same pattern as the
existing vLLM subprocess in `microbenchmark.py:383-416`:

```python
# benchmark_ep_vllm.py — run via:
#   H100_env/vllm_apptainer.sh python \
#       src/MoE/tests/vLLM_comparison/benchmark_ep_vllm.py --model $MODEL
import argparse, time, statistics
import torch

N_WARMUP = 20
N_STEPS = 200
N_TRIALS = 3

def time_vllm_ep(llm, seq_len, n_steps, n_warmup, trial_id):
    """Single trial: ms/step for vLLM EP decode."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=n_warmup + n_steps, temperature=0)
    prompt_ids = list(range(100, 100 + seq_len))
    llm.llm_engine.add_request(
        request_id=f"bench_{seq_len}_{trial_id}",
        prompt={"prompt_token_ids": prompt_ids}, params=sp)
    llm.llm_engine.step()  # prefill
    for _ in range(n_warmup):
        llm.llm_engine.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        llm.llm_engine.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    while llm.llm_engine.has_unfinished_requests():
        llm.llm_engine.step()
    return (t1 - t0) / n_steps * 1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[128, 256, 512, 1024])
    args = parser.parse_args()

    import vllm
    llm = vllm.LLM(
        model=args.model,
        tensor_parallel_size=1,
        data_parallel_size=2,
        enable_expert_parallel=True,
        enable_prefix_caching=False,  # avoid caching artifacts
    )

    print(f"{'seq_len':>8}  {'ms/step':>8}  {'trials':>24}")
    for sl in args.seq_lens:
        trials = [time_vllm_ep(llm, sl, N_STEPS, N_WARMUP, t)
                  for t in range(N_TRIALS)]
        med = statistics.median(trials)
        print(f"{sl:>8}  {med:>8.2f}  {[f'{t:.2f}' for t in trials]}")

if __name__ == "__main__":
    main()
```

### 10c. Comparison

Run both scripts, compare ms/step at each seq_len.  Both use the
same model, same number of GPUs, same expert placement (linear).
The custom engine uses CUDA event timing; vLLM uses wall-clock
(`time.perf_counter`) — same methodology as the existing single-GPU
microbenchmark.

---

## Step 11 — EP + Offloading integration (future)

When `ep_size > 1` AND `cache_size` is set:

- Each rank has `local_E = E / ep_size` experts per layer.
- Offloading manages a subset of those local experts.
- `expert_map_buf` maps `global_id → buffer_slot` directly (single level).
  Entries are -1 for both non-local (EP) and non-resident (offloaded).
- `ReplayController` operates on local expert IDs only.

---

## Step 12 — Performance optimization (future)

### 12a. True all-to-all

Replace AllGather+ReduceScatter with permutation-based all-to-all.
Only send tokens to the rank that owns their selected expert.
~50% bandwidth reduction for Mixtral top-2 EP=2.

### 12b. Overlap communication with compute

- AllGather of layer L overlaps with layer L's attention (stages 2/3).
- ReduceScatter of layer L overlaps with layer L+1's stage 1.

### 12c. Capture local fused_experts in CUDA graph

The NCCL calls stay eager, but the `fused_experts` kernel between them
can be captured in a separate graph per graph_N.  Eliminates ~200 us of
kernel launch overhead across 32 layers.

### 12d. Per-rank independent graph sizes

The current plan (Step 3) syncs all ranks to the **same** `graph_N`.
This is the simplest approach but wastes compute: if rank 0 has 5 tokens
and rank 1 has 120, both pad to 128.  Rank 0 runs a graph 25× larger
than needed for its local stages (1, 2, 3, 4a).

Only the NCCL collectives actually require matching tensor sizes.  The
local stages (attention, routing, residual add) operate on each rank's
own data and could use independently-sized graphs.  A more efficient
approach:

1. Each rank selects its own `local_graph_N` based on its `N_actual`.
2. Ranks exchange `local_graph_N` values (same AllReduce as Step 3).
3. Compute `nccl_N = max(local_graph_N)` — the size for NCCL ops only.
4. Each rank replays stages 1/4a with `local_graph_N` buffers.
5. Before AllGather, pad `moe_input_buf` from `local_graph_N` to `nccl_N`.
6. After ReduceScatter, result is `[nccl_N, H]` — slice to `local_graph_N`.

This saves compute on the smaller rank's attention and routing (stages
1-4a are O(N) in tokens), but adds complexity: the EP buffers in Step 4a
must be sized to `nccl_N` while the per-layer graph buffers are sized to
`local_graph_N`.  Worth doing when batch size imbalance is common (e.g.,
one rank doing a long prefill while the other does short decodes).

---

## Implementation Order

| Phase | What | Validates | Depends on |
|-------|------|-----------|------------|
| 0 | `ep_utils.py` + CPU-only placement tests | Expert map correctness | — |
| 1 | `__init__` EP params + process group | Multi-GPU init | 0 |
| 2 | `_load_weights` local-only expert loading | Memory reduction (35→17 GB/GPU) | 1 |
| 3 | Batch size sync in `step()` via `_ep_sync_graph_N` | Cross-rank coordination | 1 |
| 4 | EP dispatch/combine in `_step_piecewise` | MoE correctness | 3 |
| 5 | EP branch in `_layer_stage4b_moe` | Forward-looking (Step 11) | 1 |
| 6 | EP in `_layer_mixed` / `_layer_mixed_mla` | Eager fallback | 0 |
| 7+8 | Launch script + scheduler dummy step | Infrastructure (co-implement) | 3, 4, 6 |
| 9 | 2-GPU tests (dispatch, sync, e2e) | End-to-end correctness | 7+8 |
| 10 | `benchmark_ep.py` + `benchmark_ep_vllm.py`: decode ms/step comparison | Performance validation | 9 |

---

## Summary of files changed

| File | Changes |
|------|---------|
| `ep_utils.py` | **NEW** — placement, allgather, reducescatter |
| `moe_engine.py` | EP params in `__init__`, local-only weight loading, batch size sync, EP dispatch/combine in `_step_piecewise`, EP branch in `_layer_stage4b_moe`, EP in `_layer_mixed`/`_layer_mixed_mla`, skip stage4b graph capture when EP, graph capture size validation |
| `scheduler.py` | Dummy step for idle EP ranks in `collect()` and `replay()`, `total_steps` sync in `replay()`, `ep_size` property |
| `launch_ep.py` | **NEW** — torchrun launcher with round-robin prompt distribution |
| `tests/test_ep_placement.py` | **NEW** — CPU-only unit tests |
| `tests/test_ep_dispatch.py` | **NEW** — 2-GPU AllGather/ReduceScatter test |
| `tests/test_ep_consistency.py` | **NEW** — 2-GPU: graph_N sync, dummy steps, asymmetric completion, EP=2 vs EP=1 equivalence |
| `tests/vLLM_comparison/benchmark_ep.py` | **NEW** — custom EP=2 decode timing (torchrun) |
| `tests/vLLM_comparison/benchmark_ep_vllm.py` | **NEW** — vLLM EP=2 decode timing (standalone) |

---

## Cross-cutting notes

**`global_num_experts` under EP**: `_moe_experts()` passes
`global_num_experts = w1.size(0)`.  Under EP, `w1.size(0) = local_E`
(4 for Mixtral EP=2), not `num_experts` (8).  This is correct —
`expert_map` remaps global IDs to local slots in `[0, local_E)`, so
the kernel only needs `local_E` bins.  Already documented at
`moe_engine.py:598-602`.  No change needed.

---

## Key differences from previous (replicated EP) plan

| Aspect | Old plan (replicated EP) | New plan (DP+EP) |
|--------|-------------------------|------------------|
| Scheduling | Replicated on all ranks | Independent per rank |
| KV cache | Replicated (wasteful) | Per-rank (local requests only) |
| Token broadcast | Required (fragile) | Not needed |
| Divergence risk | High (FlashAttn non-determinism) | None (independent) |
| Throughput scaling | None (same batch everywhere) | Linear (2× with EP=2) |
| AllGather content | Redundant copies | Different requests |
| Batch sync | Not needed | AllReduce of 16 bytes/step |
| Idle rank handling | N/A | Dummy step for NCCL deadlock avoidance |
| Complexity | Broadcast correctness | Prompt distribution |
