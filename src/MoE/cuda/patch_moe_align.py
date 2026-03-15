"""
Build and apply the patched moe_align_block_size kernel.

Usage:
    from cuda.patch_moe_align import patch_moe_align_block_size
    patch_moe_align_block_size()  # call once at startup, before graph capture

This replaces vLLM's ops.moe_align_block_size (which has a 1024-expert limit)
with our patched version (no limit).  The Python wrapper in
vllm.model_executor.layers.fused_moe.moe_align_block_size is monkey-patched
to call our CUDA extension instead of the original.

The extension is JIT-compiled on first use and cached in /tmp/moe_align_ext/.
"""

import os
import torch
import importlib

_ext = None  # cached extension module


def _build_extension():
    """JIT-compile the CUDA extension (cached across runs)."""
    from torch.utils.cpp_extension import load

    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(cuda_dir, "moe_align_block_size_ext.cu")
    build_dir = os.path.join(cuda_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    ext = load(
        name="moe_align_ext",
        sources=[src],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-w"],
        build_directory=build_dir,
        verbose=False,
    )
    return ext


def get_extension():
    """Get (or build) the compiled extension."""
    global _ext
    if _ext is None:
        _ext = _build_extension()
    return _ext


def moe_align_block_size_replacement(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for vLLM's moe_align_block_size.

    Same signature, same output format.  Uses our patched CUDA kernel
    that supports num_experts > 1024.
    """
    import triton  # for cdiv

    ext = get_extension()

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = triton.cdiv(max_num_tokens_padded, block_size) * block_size
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty(
        (1,), dtype=torch.int32, device=topk_ids.device
    )

    ext.moe_align_block_size(
        topk_ids, num_experts, block_size,
        sorted_ids, expert_ids, num_tokens_post_pad,
        expert_map,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad


def patch_moe_align_block_size():
    """Monkey-patch vLLM to use the patched kernel.  Call once at startup."""
    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod

    # Force-build the extension now so JIT compilation happens before graph capture
    get_extension()

    fused_moe_mod.moe_align_block_size = moe_align_block_size_replacement
    print("[patch_moe_align] Replaced moe_align_block_size — "
          "expert count limit removed")
