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
