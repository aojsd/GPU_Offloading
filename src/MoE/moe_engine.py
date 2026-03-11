"""Custom MoE inference engine with vLLM kernel-level performance parity.

Supports any HuggingFace-format MoE model (OLMoE, Mixtral, etc.) that uses
gate/up/down expert projections and standard transformer attention.

Uses the same kernels as vLLM 0.11.2 v1 engine on H100:
  - MoE: vllm.fused_experts (Triton grouped GEMM)
  - Decode attention: FlashInfer BatchDecodeWithPagedKVCache
  - KV write: reshape_and_cache_flash (C++ CUDA kernel)
  - RMSNorm + residual + RoPE: torch.compile fused Triton kernels
  - Projections: torch.nn.functional.linear (cuBLAS)
"""

import json
import math
from collections import deque
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open
from vllm.v1.attention.backends.flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from vllm.vllm_flash_attn import flash_attn_varlen_func


def rope_pytorch(q, k, cos_sin_cache, positions, num_heads, head_dim,
                 num_kv_heads=None):
    """Pure PyTorch NeoX-style RoPE — fully compilable by torch.compile.

    Args:
        q: [N, num_heads * head_dim]
        k: [N, num_kv_heads * head_dim]
        cos_sin_cache: [max_seq, head_dim] (first half cos, second half sin)
        positions: [N] int32/int64
        num_heads: number of Q attention heads
        head_dim: dimension per head
        num_kv_heads: number of KV heads (defaults to num_heads for MHA)
    Returns:
        q_rot, k_rot: same shapes as input
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
    orig_dtype = q.dtype
    N = q.shape[0]
    half = head_dim // 2
    cos = cos_sin_cache[positions.long(), :half].unsqueeze(1)   # [N, 1, half]
    sin = cos_sin_cache[positions.long(), half:].unsqueeze(1)   # [N, 1, half]
    q_h = q.float().view(N, num_heads, head_dim)
    k_h = k.float().view(N, num_kv_heads, head_dim)
    q_rot = torch.cat([q_h[..., :half] * cos - q_h[..., half:] * sin,
                        q_h[..., :half] * sin + q_h[..., half:] * cos], dim=-1)
    k_rot = torch.cat([k_h[..., :half] * cos - k_h[..., half:] * sin,
                        k_h[..., :half] * sin + k_h[..., half:] * cos], dim=-1)
    return q_rot.reshape(N, -1).to(orig_dtype), k_rot.reshape(N, -1).to(orig_dtype)

# ── Conditionally patch vLLM's _moe_C ops (broken on glibc < 2.29) ──
# Load _C.so (works fine — has silu_and_mul, etc.)
import vllm._custom_ops as _vllm_ops
_vllm_so = Path(_vllm_ops.__file__).parent / "_C.abi3.so"
if _vllm_so.exists():
    torch.ops.load_library(str(_vllm_so))

import platform
_glibc_ver = platform.libc_ver()[1]
_needs_moe_patch = _glibc_ver and tuple(int(x) for x in _glibc_ver.split(".")) < (2, 29)


def _moe_align_block_size_torch(topk_ids, num_experts, block_size,
                                sorted_ids, expert_ids, num_tokens_post_pad):
    """Pure PyTorch replacement for _moe_C.moe_align_block_size.

    Fully vectorized — no .item() calls or Python for-loops over experts.
    This makes it compatible with CUDA graph capture and torch.compile.

    sorted_ids must contain FLAT indices into topk_ids.flatten(), not original
    token indices. The Triton kernel recovers the token via flat_idx // top_k.
    """
    N, K = topk_ids.shape
    total_flat = N * K
    flat_expert_ids = topk_ids.flatten()  # [N*K] expert assignments

    # Count per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    expert_counts.scatter_add_(0, flat_expert_ids.long(),
                               torch.ones(total_flat, dtype=torch.int32, device=topk_ids.device))

    # Padded counts (round up to block_size)
    padded_counts = ((expert_counts + block_size - 1) // block_size) * block_size
    cum_padded = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    cum_padded[1:] = torch.cumsum(padded_counts, dim=0)
    cum_orig = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    cum_orig[1:] = torch.cumsum(expert_counts, dim=0)

    # Sort flat positions by expert
    sort_order = flat_expert_ids.argsort(stable=True)
    sorted_flat_pos = torch.arange(total_flat, device=topk_ids.device, dtype=torch.int32)[sort_order]
    expert_of_sorted = flat_expert_ids[sort_order]

    # Vectorized fill of sorted_ids: compute destination for each sorted token
    within_idx = torch.arange(total_flat, device=topk_ids.device, dtype=torch.int32)
    within_idx = within_idx - cum_orig[expert_of_sorted.long()]
    dst = cum_padded[expert_of_sorted.long()] + within_idx

    sorted_ids.fill_(total_flat)  # padding value (out-of-range, ignored by kernel)
    sorted_ids.scatter_(0, dst.long(), sorted_flat_pos)

    # Vectorized fill of expert_ids via searchsorted
    n_blocks = padded_counts // block_size
    cum_blocks = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    cum_blocks[1:] = torch.cumsum(n_blocks, dim=0)
    block_positions = torch.arange(expert_ids.shape[0], device=topk_ids.device, dtype=torch.int32)
    expert_ids[:] = torch.searchsorted(cum_blocks[1:].contiguous(), block_positions, right=True)
    # Clamp padding blocks to valid range. Blocks beyond num_tokens_post_padded
    # are never processed by the Triton kernel, but expert_map[expert_ids] would
    # OOB if searchsorted returns num_experts for these padding positions.
    expert_ids.clamp_(max=num_experts - 1)

    num_tokens_post_pad.fill_(0)
    num_tokens_post_pad[0] = cum_padded[-1]


def _moe_sum_torch(input: torch.Tensor, output: torch.Tensor):
    """Pure PyTorch replacement for _moe_C.moe_sum (sum over top-k dim)."""
    output.copy_(input.sum(dim=1))


def _topk_softmax_torch(topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                         token_expert_indices: torch.Tensor,
                         gating_output: torch.Tensor,
                         renormalize: bool) -> None:
    """Pure PyTorch replacement for _moe_C.topk_softmax.

    Computes softmax(gating_output) -> top-k selection -> fills output tensors.
    topk_weights: [M, topk] float32 output (selected softmax probs)
    topk_ids: [M, topk] int32 output (selected expert indices)
    token_expert_indices: [M, topk] int32 output (flat index = i * E + expert_id)
    gating_output: [M, E] float input (raw router logits)
    """
    M, E = gating_output.shape
    K = topk_weights.shape[1]
    scores = torch.softmax(gating_output.float(), dim=-1)
    tk_w, tk_i = torch.topk(scores, K, dim=-1)
    if renormalize:
        tk_w = tk_w / tk_w.sum(dim=-1, keepdim=True)
    topk_weights.copy_(tk_w)
    topk_ids.copy_(tk_i.to(topk_ids.dtype))
    # token_expert_indices[i, j] = i * E + topk_ids[i, j]
    row_offsets = torch.arange(M, device=gating_output.device, dtype=torch.int32).unsqueeze(1) * E
    token_expert_indices.copy_(row_offsets + tk_i.to(torch.int32))


# Apply monkey-patches only on glibc < 2.29 (needed for old RHEL 8.10 systems)
if _needs_moe_patch:
    _vllm_ops.moe_align_block_size = _moe_align_block_size_torch
    _vllm_ops.moe_sum = _moe_sum_torch
    _vllm_ops.topk_softmax = _topk_softmax_torch
    import vllm.model_executor.layers.fused_moe.moe_align_block_size as _mabs_mod
    _mabs_mod.ops.moe_align_block_size = _moe_align_block_size_torch
    import vllm.model_executor.layers.fused_moe.fused_moe as _fused_moe_mod
    _fused_moe_mod.ops.topk_softmax = _topk_softmax_torch
    _fused_moe_mod.ops.moe_sum = _moe_sum_torch

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


class MoEEngine:
    """Custom MoE inference engine — FlashInfer decode + torch.compile."""

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
    ):
        # Resolve to a concrete device with index (e.g. "cuda" -> "cuda:0")
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.dtype = dtype
        self.max_seqs = max_seqs
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        self.use_torch_compile = use_torch_compile

        # Support both names during transition
        if experts_per_layer is not None:
            self.experts_per_layer = experts_per_layer
        elif gpu_expert_budget is not None:
            self.experts_per_layer = gpu_expert_budget
        else:
            self.experts_per_layer = None
        self.gpu_expert_budget = self.experts_per_layer  # alias for compat

        # Unified cache mode: single shared cache across all layers
        self.cache_size = cache_size
        if self.experts_per_layer is not None and self.cache_size is not None:
            raise ValueError(
                "Cannot set both experts_per_layer and cache_size")
        self.offloading = (self.experts_per_layer is not None or
                           self.cache_size is not None)

        # Pipeline parallelism: validated after config load (needs num_layers)
        self._pipeline_parallel_size = pipeline_parallel_size
        self.trace_recorder = None  # set to TraceRecorder for PP trace collection
        self._nvtx_enabled = False  # set True for NVTX profiling ranges

        # Load config
        with open(Path(model_path) / "config.json") as f:
            cfg = json.load(f)
        self.num_layers = cfg["num_hidden_layers"]
        self.hidden_size = cfg["hidden_size"]
        self.intermediate_size = cfg["intermediate_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        # Expert count: OLMoE uses "num_experts", Mixtral uses "num_local_experts"
        self.num_experts = cfg.get("num_experts") or cfg.get("num_local_experts")
        self.top_k = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_topk")
        self.rope_theta = cfg.get("rope_theta", 10000.0)
        self.vocab_size = cfg["vocab_size"]
        self.eos_token_id = cfg.get("eos_token_id", 2)
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)
        # OLMoE has explicit "norm_topk_prob" in config. For Mixtral and other
        # models that lack this field, vLLM defaults renormalize=True in FusedMoE.
        # Match that behavior: renormalize unless explicitly set to False.
        self.norm_topk_prob = cfg.get("norm_topk_prob", True)

        # Unsupported features — raise early rather than produce wrong results
        if cfg.get("sliding_window") is not None:
            raise NotImplementedError(
                f"Sliding window attention (window={cfg['sliding_window']}) is not "
                f"yet implemented.")
        if cfg.get("rope_scaling") is not None:
            raise NotImplementedError(
                f"RoPE scaling ({cfg['rope_scaling']}) is not yet implemented.")

        # Pipeline parallelism setup (now that num_layers is known)
        self.pp_size = self._pipeline_parallel_size
        if self.pp_size > 1:
            if self.offloading:
                raise ValueError(
                    "Pipeline parallelism requires offloading to be disabled. "
                    "Do not set experts_per_layer or cache_size with PP.")
            n_gpus = torch.cuda.device_count()
            if n_gpus < self.pp_size:
                raise RuntimeError(
                    f"PP={self.pp_size} requires {self.pp_size} GPUs, "
                    f"found {n_gpus}")
            layers_per_gpu = math.ceil(self.num_layers / self.pp_size)
            self.pp_devices = [torch.device(f"cuda:{i}")
                               for i in range(self.pp_size)]
            self.pp_layer_gpu = [
                min(l // layers_per_gpu, self.pp_size - 1)
                for l in range(self.num_layers)
            ]
            self.pp_layer_device = [self.pp_devices[g]
                                    for g in self.pp_layer_gpu]
            # Boundaries: layers where GPU changes (for cross-GPU transfers)
            self.pp_boundaries = set(
                l for l in range(1, self.num_layers)
                if self.pp_layer_gpu[l] != self.pp_layer_gpu[l - 1]
            )
            print(f"Pipeline parallel: {self.pp_size} GPUs, "
                  f"{layers_per_gpu} layers/GPU, "
                  f"boundaries at layers {sorted(self.pp_boundaries)}")
        else:
            self.pp_devices = None
            self.pp_layer_gpu = None
            self.pp_layer_device = None
            self.pp_boundaries = set()

        # Load weights
        print("Loading weights...")
        self._load_weights(model_path)

        # RoPE cos/sin cache, attention scales, KV cache, block table,
        # seq_lens, FlashInfer — per-GPU when PP, single tensors otherwise
        self._attn_scale = 1.0 / math.sqrt(self.head_dim)
        self.max_pages_per_seq = math.ceil(max_seq_len / page_size)
        # Dynamic page allocation: decouple physical page count from
        # per-sequence pre-allocation. Less than 1% overhead vs static.
        self._dynamic_pages = kv_page_budget is not None
        if self._dynamic_pages:
            self.total_pages = kv_page_budget
        else:
            self.total_pages = max_seqs * self.max_pages_per_seq

        if self.pp_size > 1:
            # Per-GPU replicated small tensors
            self.cos_sin_cache = [self._build_rope_cache().to(d)
                                  for d in self.pp_devices]
            self._k_scale = [torch.tensor(1.0, dtype=torch.float32, device=d)
                             for d in self.pp_devices]
            self._v_scale = [torch.tensor(1.0, dtype=torch.float32, device=d)
                             for d in self.pp_devices]

            # KV cache: [total_pages, page_size, num_kv_heads, head_dim] per layer.
            # Layout required by reshape_and_cache_flash.
            self.k_cache = [
                torch.zeros(self.total_pages, page_size,
                            self.num_kv_heads, self.head_dim,
                            dtype=dtype, device=self.pp_layer_device[l])
                for l in range(self.num_layers)
            ]
            self.v_cache = [
                torch.zeros(self.total_pages, page_size,
                            self.num_kv_heads, self.head_dim,
                            dtype=dtype, device=self.pp_layer_device[l])
                for l in range(self.num_layers)
            ]

            # Block table: maps (seq_id, virtual_page_idx) → physical page.
            # Static mode: identity mapping (seq i gets contiguous pages).
            # Dynamic mode: initialized to -1, pages assigned by alloc_pages().
            # Replicated on each GPU for PP.
            if self._dynamic_pages:
                bt = torch.full((max_seqs, self.max_pages_per_seq),
                                -1, dtype=torch.int32)
            else:
                bt = torch.zeros(max_seqs, self.max_pages_per_seq,
                                 dtype=torch.int32)
                for i in range(max_seqs):
                    bt[i] = torch.arange(
                        i * self.max_pages_per_seq,
                        (i + 1) * self.max_pages_per_seq,
                        dtype=torch.int32)
            self.block_table = [bt.to(d) for d in self.pp_devices]

            # Sequence length tracker: replicated on each GPU
            self.seq_lens = [torch.zeros(max_seqs, dtype=torch.int32,
                                         device=d)
                             for d in self.pp_devices]
            self._seq_lens_cpu = torch.zeros(max_seqs, dtype=torch.int32)

            # FlashInfer: one wrapper per GPU
            self._workspace_bufs = [
                torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=d)
                for d in self.pp_devices
            ]
            self._decode_wrappers = [
                BatchDecodeWithPagedKVCacheWrapper(
                    wb, kv_layout="NHD", use_cuda_graph=False)
                for wb in self._workspace_bufs
            ]
            # Alias for prefill graph capture (always uses GPU 0)
            self._decode_wrapper = self._decode_wrappers[0]
        else:
            # Single-GPU path (original)
            self.cos_sin_cache = self._build_rope_cache().to(device)
            self._k_scale = torch.tensor(1.0, dtype=torch.float32,
                                         device=device)
            self._v_scale = torch.tensor(1.0, dtype=torch.float32,
                                         device=device)

            # KV cache: [total_pages, page_size, num_kv_heads, head_dim] per layer.
            # Layout required by reshape_and_cache_flash.
            self.k_cache = [
                torch.zeros(self.total_pages, page_size,
                            self.num_kv_heads, self.head_dim,
                            dtype=dtype, device=device)
                for _ in range(self.num_layers)
            ]
            self.v_cache = [
                torch.zeros(self.total_pages, page_size,
                            self.num_kv_heads, self.head_dim,
                            dtype=dtype, device=device)
                for _ in range(self.num_layers)
            ]

            # Block table: maps (seq_id, virtual_page_idx) → physical page.
            # Static mode: identity mapping (seq i gets contiguous pages).
            # Dynamic mode: initialized to -1, pages assigned by alloc_pages().
            if self._dynamic_pages:
                self.block_table = torch.full(
                    (max_seqs, self.max_pages_per_seq), -1,
                    dtype=torch.int32, device=device)
            else:
                self.block_table = torch.zeros(
                    max_seqs, self.max_pages_per_seq, dtype=torch.int32,
                    device=device)
                for i in range(max_seqs):
                    self.block_table[i] = torch.arange(
                        i * self.max_pages_per_seq,
                        (i + 1) * self.max_pages_per_seq,
                        dtype=torch.int32, device=device)

            # Sequence length tracker
            self.seq_lens = torch.zeros(max_seqs, dtype=torch.int32,
                                        device=device)
            self._seq_lens_cpu = torch.zeros(max_seqs,
                                             dtype=torch.int32)

            # FlashInfer workspace (128 MB)
            self._workspace_buf = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device)
            self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._workspace_buf, kv_layout="NHD", use_cuda_graph=False)

        # Dynamic page allocation pool
        if self._dynamic_pages:
            self._free_pages = deque(range(self.total_pages))
            self._seq_page_list: list[list[int]] = [
                [] for _ in range(max_seqs)]

        # Guard: check model fits on GPU (skip in offloading/PP mode)
        if not self.offloading and self.pp_size == 1:
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if mem_gb > gpu_total_gb * 0.95:
                raise RuntimeError(
                    f"Model requires {mem_gb:.1f} GB but GPU has "
                    f"{gpu_total_gb:.0f} GB. Use experts_per_layer=K or "
                    f"cache_size=N to enable expert offloading.")

        # Expert offload engine — auto-created when experts_per_layer is set.
        # Handles demand loading of non-resident experts between stage4a
        # (routing) and stage4b (MoE compute) in piecewise CUDA graph replay.
        # Not created for cache_size mode (ReplayController handles that).
        if self.experts_per_layer is not None:
            from expert_offload_engine import ExpertOffloadEngine
            self.offload_engine = ExpertOffloadEngine(self)
        else:
            self.offload_engine = None
        self.replay_controller = None  # set to ReplayController for trace replay

        model_type = cfg.get("model_type", "unknown")
        if self.experts_per_layer is not None:
            budget_str = f", experts_per_layer={self.experts_per_layer}"
        elif self.cache_size is not None:
            budget_str = f", cache_size={self.cache_size}"
        elif self.pp_size > 1:
            budget_str = f", PP={self.pp_size}"
        else:
            budget_str = ""
        print(f"MoEEngine ready ({model_type}): {self.num_layers}L, "
              f"{self.num_experts}E (top-{self.top_k}), "
              f"hidden={self.hidden_size}, heads={self.num_heads}, "
              f"qk_norm={self.has_qk_norm}, norm_topk={self.norm_topk_prob}, "
              f"torch_compile={use_torch_compile}{budget_str}")

    # ── MoE Kernel Dispatch ──────────────────────────────────────────

    def _moe_experts(self, hidden_states, w1, w2, topk_weights, topk_ids,
                     expert_map=None):
        """Dispatch MoE computation to vLLM Triton fused_experts.

        Args:
            hidden_states: [N, H] input hidden states
            w1: [E, 2*I, H] gate+up fused weights
            w2: [E, H, I] down projection weights
            topk_weights: [N, top_k] routing weights
            topk_ids: [N, top_k] expert indices (global IDs)
            expert_map: [E] int32 mapping global expert_id -> cache slot (or None)
        """
        kwargs = {}
        if expert_map is not None:
            kwargs['expert_map'] = expert_map
            # Monkey-patched moe_align_block_size (glibc < 2.29) clamps
            # expert_ids to [0, num_experts-1] before expert_map lookup,
            # so num_experts must equal the model's global expert count.
            # Native vLLM 0.17 kernel needs the buffer dimension instead.
            kwargs['global_num_experts'] = (self.num_experts
                                            if _needs_moe_patch
                                            else w1.size(0))
        return fused_experts(
            hidden_states=hidden_states, w1=w1, w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            **kwargs)

    # ── PP Device Helpers ──────────────────────────────────────────

    def _get_cos_sin_cache(self, layer):
        """Get RoPE cos/sin cache for the GPU hosting this layer."""
        if self.pp_size > 1:
            return self.cos_sin_cache[self.pp_layer_gpu[layer]]
        return self.cos_sin_cache

    def _get_block_table(self, layer):
        """Get block table for the GPU hosting this layer."""
        if self.pp_size > 1:
            return self.block_table[self.pp_layer_gpu[layer]]
        return self.block_table

    def _get_seq_lens(self, layer):
        """Get seq_lens tensor for the GPU hosting this layer."""
        if self.pp_size > 1:
            return self.seq_lens[self.pp_layer_gpu[layer]]
        return self.seq_lens

    def _get_kv_scales(self, layer):
        """Get k_scale, v_scale tensors for the GPU hosting this layer."""
        if self.pp_size > 1:
            g = self.pp_layer_gpu[layer]
            return self._k_scale[g], self._v_scale[g]
        return self._k_scale, self._v_scale

    # ── Weight Loading ───────────────────────────────────────────────

    def _load_weights(self, model_path: str):
        model_path = Path(model_path)
        offloading = self.offloading

        # When offloading, load to CPU first to avoid GPU OOM on large models.
        # Non-expert weights (~3 GB) move to GPU; expert weights stay on CPU.
        # When PP, also load to CPU first so each layer goes to correct GPU.
        load_device = ("cpu" if (offloading or self.pp_size > 1)
                       else self.device.type)

        weights = {}
        for shard in sorted(model_path.glob("model-*.safetensors")):
            with safe_open(str(shard), framework="pt", device=load_device) as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key).to(self.dtype)

        # Global weights: embed on first GPU, final_norm + lm_head on last GPU
        if self.pp_size > 1:
            embed_dev = self.pp_devices[0]
            head_dev = self.pp_devices[-1]
        else:
            embed_dev = self.device
            head_dev = self.device
        self.embed_tokens = weights.pop("model.embed_tokens.weight").to(embed_dev)
        self.final_norm = weights.pop("model.norm.weight").to(head_dev)
        self.lm_head = weights.pop("lm_head.weight").to(head_dev)

        # Per-layer weights
        self.input_layernorm = []
        self.post_attn_layernorm = []
        self.q_proj = []
        self.k_proj = []
        self.v_proj = []
        self.o_proj = []
        self.q_norm = []
        self.k_norm = []
        self.router = []

        # Detect Q/K norm from first layer's weights
        self.has_qk_norm = f"model.layers.0.self_attn.q_norm.weight" in weights

        # Expert weight storage depends on mode
        if offloading:
            I = self.intermediate_size
            H = self.hidden_size
            E = self.num_experts

            if self.cache_size is not None:
                # Unified cache mode: flat buffer, no per-layer partitioning
                total_slots = self.cache_size
                scratchpad = 0
            else:
                # Per-layer mode: L partitions + scratchpad
                experts_per_layer = self.experts_per_layer
                scratchpad = E
                total_slots = self.num_layers * experts_per_layer + scratchpad

            # CPU pinned storage for all experts
            self.w1_cpu = []
            self.w2_cpu = []

            # GPU buffer for cached experts
            self.w1_buf = torch.zeros(total_slots, 2 * I, H, dtype=self.dtype,
                                      device=self.device)
            self.w2_buf = torch.zeros(total_slots, H, I, dtype=self.dtype,
                                      device=self.device)

            # Expert maps (absolute: maps expert_id -> buffer slot per layer)
            self.expert_map_abs = []
            self.expert_map_buf = torch.full((E,), -1, dtype=torch.int32,
                                             device=self.device)

            if self.experts_per_layer is not None:
                # Per-layer mode needs relative maps + per-layer views
                self.expert_map = []
                self.w1 = []
                self.w2 = []
                self.scratchpad_start = (self.num_layers *
                                         self.experts_per_layer)
                self.scratchpad_slots = scratchpad
            else:
                self.scratchpad_start = self.cache_size
                self.scratchpad_slots = 0

            buf_bytes = (self.w1_buf.numel() * self.w1_buf.element_size() +
                         self.w2_buf.numel() * self.w2_buf.element_size())
            if self.cache_size is not None:
                print(f"  Unified cache: {total_slots} slots "
                      f"(cache_size={self.cache_size}), "
                      f"w1_buf {list(self.w1_buf.shape)}, "
                      f"w2_buf {list(self.w2_buf.shape)} "
                      f"({buf_bytes / 1e9:.2f} GB)")
            else:
                print(f"  Unified buffer: {total_slots} slots "
                      f"({self.num_layers}L x {experts_per_layer}"
                      f" experts_per_layer + {scratchpad} scratchpad), "
                      f"w1_buf {list(self.w1_buf.shape)}, "
                      f"w2_buf {list(self.w2_buf.shape)} "
                      f"({buf_bytes / 1e9:.2f} GB)")
        else:
            self.w1 = []
            self.w2 = []

        for l in range(self.num_layers):
            p = f"model.layers.{l}"
            # Per-layer device: layer's GPU when PP, else self.device
            layer_dev = (self.pp_layer_device[l]
                         if self.pp_size > 1 else self.device)
            # Attention + norm weights → layer's GPU
            self.input_layernorm.append(
                weights.pop(f"{p}.input_layernorm.weight").to(layer_dev))
            self.post_attn_layernorm.append(
                weights.pop(f"{p}.post_attention_layernorm.weight").to(layer_dev))
            self.q_proj.append(
                weights.pop(f"{p}.self_attn.q_proj.weight").to(layer_dev))
            self.k_proj.append(
                weights.pop(f"{p}.self_attn.k_proj.weight").to(layer_dev))
            self.v_proj.append(
                weights.pop(f"{p}.self_attn.v_proj.weight").to(layer_dev))
            self.o_proj.append(
                weights.pop(f"{p}.self_attn.o_proj.weight").to(layer_dev))
            if self.has_qk_norm:
                self.q_norm.append(
                    weights.pop(f"{p}.self_attn.q_norm.weight").to(layer_dev))
                self.k_norm.append(
                    weights.pop(f"{p}.self_attn.k_norm.weight").to(layer_dev))
            # Router key: OLMoE uses "mlp.gate", Mixtral uses "block_sparse_moe.gate"
            router_key = f"{p}.mlp.gate.weight"
            if router_key not in weights:
                router_key = f"{p}.block_sparse_moe.gate.weight"
            self.router.append(weights.pop(router_key).to(layer_dev))

            # Fuse expert weights: w1 = [E, 2*I, H], w2 = down [E, H, I]
            # Triton fused_experts: w1 = cat(gate, up)
            # Key format: OLMoE: "mlp.experts.{e}", Mixtral: "block_sparse_moe.experts.{e}"
            w1_list, w2_list = [], []
            for e in range(self.num_experts):
                expert_prefix = f"{p}.mlp.experts.{e}"
                if f"{expert_prefix}.gate_proj.weight" not in weights:
                    expert_prefix = f"{p}.block_sparse_moe.experts.{e}"
                # OLMoE: gate_proj/up_proj/down_proj; Mixtral: w1/w3/w2
                if f"{expert_prefix}.gate_proj.weight" in weights:
                    gate = weights.pop(f"{expert_prefix}.gate_proj.weight")
                    up = weights.pop(f"{expert_prefix}.up_proj.weight")
                    down = weights.pop(f"{expert_prefix}.down_proj.weight")
                else:
                    gate = weights.pop(f"{expert_prefix}.w1.weight")
                    up = weights.pop(f"{expert_prefix}.w3.weight")
                    down = weights.pop(f"{expert_prefix}.w2.weight")
                w1_list.append(torch.cat([gate, up], dim=0))
                w2_list.append(down)

            if offloading:
                # Stack on CPU and pin for fast DMA transfers
                w1_full = torch.stack(w1_list).pin_memory()  # [E, 2*I, H]
                w2_full = torch.stack(w2_list).pin_memory()  # [E, H, I]
                self.w1_cpu.append(w1_full)
                self.w2_cpu.append(w2_full)

                if self.experts_per_layer is not None:
                    # Per-layer mode: populate buffer with initial experts
                    base = l * experts_per_layer
                    copy_n = min(experts_per_layer, E)
                    for slot in range(copy_n):
                        self.w1_buf[base + slot].copy_(w1_full[slot])
                        self.w2_buf[base + slot].copy_(w2_full[slot])

                    # Per-layer views (zero-copy into unified buffer)
                    self.w1.append(
                        self.w1_buf[base : base + experts_per_layer])
                    self.w2.append(
                        self.w2_buf[base : base + experts_per_layer])

                    # Relative expert_map: first experts_per_layer → 0..N-1
                    emap_rel = torch.full((E,), -1, dtype=torch.int32,
                                          device=self.device)
                    for slot in range(copy_n):
                        emap_rel[slot] = slot
                    self.expert_map.append(emap_rel)

                    # Absolute expert_map
                    emap_abs = torch.full((E,), -1, dtype=torch.int32,
                                          device=self.device)
                    for slot in range(copy_n):
                        emap_abs[slot] = base + slot
                    self.expert_map_abs.append(emap_abs)

                    print(f"  Layer {l}: w1_cpu {w1_full.shape}, "
                          f"view [{base}:{base+experts_per_layer}] in "
                          f"unified buffer, expert_map "
                          f"[{copy_n} resident / {E} total]")
                else:
                    # Unified cache mode: don't pre-load experts.
                    # ReplayController.setup() populates the buffer.
                    emap_abs = torch.full((E,), -1, dtype=torch.int32,
                                          device=self.device)
                    self.expert_map_abs.append(emap_abs)
                    print(f"  Layer {l}: w1_cpu {w1_full.shape} "
                          f"(unified cache, no pre-load)")
            else:
                self.w1.append(torch.stack(w1_list).to(layer_dev))
                self.w2.append(torch.stack(w2_list).to(layer_dev))
                print(f"  Layer {l}: w1 {self.w1[-1].shape}, "
                      f"w2 {self.w2[-1].shape}"
                      f"{f' ({layer_dev})' if self.pp_size > 1 else ''}")

        del weights
        torch.cuda.empty_cache()

        # Initialize expert_map_buf with layer 0's absolute map for warmup
        if offloading:
            self.expert_map_buf.copy_(self.expert_map_abs[0])

        # Shape verification
        assert self.embed_tokens.shape == (self.vocab_size, self.hidden_size)
        assert self.lm_head.shape == (self.vocab_size, self.hidden_size)
        if offloading:
            if self.experts_per_layer is not None:
                experts_per_layer = self.experts_per_layer
                assert self.w1[0].shape == (experts_per_layer, 2 * self.intermediate_size, self.hidden_size)
                assert self.w2[0].shape == (experts_per_layer, self.hidden_size, self.intermediate_size)
                assert self.w1_buf.shape[0] == self.num_layers * experts_per_layer + self.scratchpad_slots
            else:
                assert self.w1_buf.shape[0] == self.cache_size
            assert self.w1_cpu[0].shape == (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        else:
            assert self.w1[0].shape == (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
            assert self.w2[0].shape == (self.num_experts, self.hidden_size, self.intermediate_size)

        # Stack per-layer weights into single tensors for indexed access.
        # When PP > 1, weights are on different GPUs — keep as lists.
        # self.foo[layer] works identically for both lists and stacked tensors.
        if self.pp_size > 1:
            # Fuse QKV per-layer (keep as list since weights span GPUs)
            self.qkv_proj = [
                torch.cat([q, k, v], dim=0)
                for q, k, v in zip(self.q_proj, self.k_proj, self.v_proj)
            ]
            del self.q_proj, self.k_proj, self.v_proj
        else:
            self.input_layernorm = torch.stack(self.input_layernorm)    # [L, H]
            self.post_attn_layernorm = torch.stack(self.post_attn_layernorm)  # [L, H]
            kv_dim = self.num_kv_heads * self.head_dim
            self.qkv_proj = torch.stack([
                torch.cat([q, k, v], dim=0)
                for q, k, v in zip(self.q_proj, self.k_proj, self.v_proj)
            ])  # [L, H + 2*kv_dim, H]
            self.o_proj = torch.stack(self.o_proj)    # [L, H, H]
            if self.has_qk_norm:
                self.q_norm = torch.stack(self.q_norm)    # [L, H]
                self.k_norm = torch.stack(self.k_norm)    # [L, H]
            self.router = torch.stack(self.router)    # [L, E, H]
            del self.q_proj, self.k_proj, self.v_proj
        # Keep w1/w2 as lists — stacking would require 2x peak memory for large
        # models (Mixtral w1 alone is 35 GB). Indexing self.w1[layer] works the
        # same for both lists and stacked tensors.

    def _build_rope_cache(self) -> torch.Tensor:
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # [max_seq, head_dim]

    # ── Offloading ────────────────────────────────────────────────────

    @property
    def offloading_active(self):
        """True when experts_per_layer < num_experts (some experts on CPU)."""
        return (self.experts_per_layer is not None
                and self.experts_per_layer < self.num_experts)

    # ── Prefill ──────────────────────────────────────────────────────

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill forward pass (CUDA graph only).

        Args: input_ids [B, S]
        Returns: logits [B, S, vocab_size]
        Requires: capture_prefill_cuda_graph() called with sufficient sizes.
        Prefill always uses flat CUDA graph (no offload engine hook — offloading
        targets decode only, as in prefill-decode disaggregated systems).
        """
        B, S = input_ids.shape
        total = B * S
        padded_N = self._find_nearest_prefill_total(total)
        seq_ids = list(range(B))
        seq_lengths = [S] * B
        token_ids_flat = input_ids.reshape(-1)
        logits_flat = self._prefill_graphed_flat(
            token_ids_flat, seq_lengths, seq_ids, padded_N)
        return logits_flat.view(B, S, -1)

    def capture_prefill_cuda_graph(self, total_token_sizes=None,
                                   use_torch_compile=None):
        """Capture flat CUDA graphs for prefill keyed by total token count.

        Each graph serves any combination of sequences whose total tokens
        fit within the captured size (e.g., N=256 serves 1×256, 4×64,
        2×100+1×56, etc.). Uses _full_mixed_graph_body with
        num_decode_tokens=0 (pure prefill via FA3 flash_attn_varlen_func).

        Padding tokens use slot_mapping=-1 (vLLM sentinel — KV writes skipped).

        Args:
            total_token_sizes: List of total token counts to capture
                (default: [128, 256, 512, 1024, 2048])
            use_torch_compile: Override instance default
        """
        if total_token_sizes is None:
            total_token_sizes = [128, 256, 512, 1024, 2048]
        if use_torch_compile is None:
            use_torch_compile = self.use_torch_compile

        if not hasattr(self, '_prefill_cuda_graphs'):
            self._prefill_cuda_graphs = {}

        # fullgraph=False: see comment in capture_mixed_cuda_graphs()
        forward_fn = self._full_mixed_graph_body
        if use_torch_compile:
            forward_fn = torch.compile(forward_fn, fullgraph=False)

        for N in total_token_sizes:
            self.reset()

            # Static input buffers — flat [N] layout
            static_token_ids = torch.randint(
                1, 1000, (N,), device=self.device)
            # Positions: wrap to stay within RoPE cache range when N > max_seq_len
            # (during warmup, exact positions don't matter — just need valid RoPE indices)
            static_positions = (torch.arange(N, dtype=torch.int32, device=self.device)
                                % self.max_seq_len)
            # Slot mapping: cycle through valid KV cache addresses
            # (N may exceed per-sequence capacity; during warmup, exact slots don't matter)
            total_kv_slots = self.total_pages * self.page_size
            static_slot_mapping = torch.arange(
                N, dtype=torch.long, device=self.device) % total_kv_slots
            # cu_seqlens: 1 sequence of length N, rest zero-length
            static_cu_seqlens = torch.full(
                (self.max_seqs + 1,), N,
                dtype=torch.int32, device=self.device)
            static_cu_seqlens[0] = 0

            # Baked-in Python ints: num_decode_tokens=0, prefill_max_seqlen=N
            # torch.compile dead-code-eliminates decode branches
            num_decode_tokens = 0
            prefill_max_seqlen = N

            # Warmup — triggers torch.compile JIT, stabilizes CUDA allocator
            n_warmup = 5 if use_torch_compile else 3
            for _ in range(n_warmup):
                static_output = forward_fn(
                    static_token_ids, static_positions,
                    static_slot_mapping, num_decode_tokens,
                    self._decode_wrapper, static_cu_seqlens,
                    prefill_max_seqlen)
            torch.cuda.synchronize()

            # Capture CUDA graph — each graph gets its own private pool
            # (shared pool causes illegal memory access with 3+ graphs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = forward_fn(
                    static_token_ids, static_positions,
                    static_slot_mapping, num_decode_tokens,
                    self._decode_wrapper, static_cu_seqlens,
                    prefill_max_seqlen)

            self._prefill_cuda_graphs[N] = {
                'graph': graph,
                'static_token_ids': static_token_ids,
                'static_output': static_output,
                # Keep references to ALL static buffers captured by the graph —
                # if these get GC'd, the graph replays with freed addresses.
                'static_positions': static_positions,
                'static_slot_mapping': static_slot_mapping,
                'static_cu_seqlens': static_cu_seqlens,
            }

            compile_str = " + torch.compile" if use_torch_compile else ""
            print(f"  Prefill CUDA graph{compile_str} captured for "
                  f"total_tokens={N}")

    def _prefill_graphed_flat(self, token_ids_flat, seq_lengths, seq_ids, padded_N):
        """Replay flat prefill CUDA graph.

        Unified replay for all prefill paths (single, batch, variable-length).

        Args:
            token_ids_flat: [sum(seq_lengths)] concatenated token IDs
            seq_lengths: list[int] per-sequence lengths
            seq_ids: list[int] KV cache slot indices
            padded_N: captured total_token_size (from _find_nearest_prefill_total)
        Returns: logits [sum(seq_lengths), vocab_size]
        """
        info = self._prefill_cuda_graphs[padded_N]
        n_real = token_ids_flat.shape[0]

        # 1. Copy token_ids into static buffer, zero-pad remainder
        info['static_token_ids'][:n_real].copy_(token_ids_flat)
        if n_real < padded_N:
            info['static_token_ids'][n_real:].zero_()

        # 2. Compute and copy positions
        positions = self._compute_flat_positions(seq_lengths, padded_N)
        info['static_positions'].copy_(positions)

        # 3. Compute and copy slot_mapping (padding → -1)
        slot_mapping = self._compute_flat_slot_mapping(
            seq_ids, seq_lengths, padded_N)
        info['static_slot_mapping'].copy_(slot_mapping)

        # 4. Compute and copy cu_seqlens
        cu_seqlens = self._compute_flat_cu_seqlens(
            seq_lengths, self.max_seqs)
        info['static_cu_seqlens'].copy_(cu_seqlens)

        # 5. Replay graph
        info['graph'].replay()

        # 6. Update seq_lens for all sequences
        if self.pp_size > 1:
            for sl in self.seq_lens:
                for sid, length in zip(seq_ids, seq_lengths):
                    sl[sid] = length
        else:
            for sid, length in zip(seq_ids, seq_lengths):
                self.seq_lens[sid] = length
        for sid, length in zip(seq_ids, seq_lengths):
            self._seq_lens_cpu[sid] = length

        # 7. Return only real token logits
        return info['static_output'][:n_real]

    # ── Decode ───────────────────────────────────────────────────────

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Single decode step.
        Args: token_ids [B], positions [B]
        Returns: logits [B, vocab_size]
        When offload_engine is present, routes through piecewise mixed_step.
        """
        B = token_ids.shape[0]

        if self.offload_engine or self.replay_controller:
            # Route through mixed_step for piecewise graphs + offload/replay hook.
            return self.mixed_step(
                decode_seq_ids=list(range(B)),
                decode_token_ids=token_ids,
                prefill_seq_ids=[],
                prefill_input_ids=[])

        if hasattr(self, '_cuda_graphs') and B in self._cuda_graphs:
            return self._decode_step_graphed(token_ids, positions)

        pos_i32 = positions.to(torch.int32)
        slot_mapping = self._compute_slot_mapping(pos_i32, B)

        # Increment seq_lens BEFORE plan so FlashInfer includes the current
        # token's K/V (written to cache before attention within each layer).
        self._seq_lens_cpu[:B] += 1
        self._plan_flashinfer_decode(self._decode_wrapper, B)

        hidden = F.embedding(token_ids, self.embed_tokens)
        for layer in range(self.num_layers):
            hidden = self._layer_decode(hidden, layer, pos_i32,
                                        slot_mapping, self._decode_wrapper)

        hidden = F.rms_norm(hidden, (self.hidden_size,), self.final_norm, self.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head)
        if self.pp_size > 1:
            for sl in self.seq_lens:
                sl[:B] += 1
        else:
            self.seq_lens[:B] += 1
        return logits

    def _layer_decode(self, hidden, layer, positions, slot_mapping, decode_wrapper):
        """Single decode layer — reshape_and_cache_flash + FlashInfer BatchDecode.

        Uses rope_pytorch (compilable by torch.compile / Inductor).
        """
        B = hidden.shape[0]
        H = self.hidden_size
        kv_dim = self.num_kv_heads * self.head_dim

        residual = hidden
        hidden = F.rms_norm(hidden, (H,), self.input_layernorm[layer], self.rms_norm_eps)

        # Fused QKV projection (single GEMM)
        qkv = F.linear(hidden, self.qkv_proj[layer])
        q, k, v = qkv.split([H, kv_dim, kv_dim], dim=-1)

        if self.has_qk_norm:
            q = F.rms_norm(q, (H,), self.q_norm[layer], self.rms_norm_eps).contiguous()
            k = F.rms_norm(k, (kv_dim,), self.k_norm[layer], self.rms_norm_eps).contiguous()

        # RoPE — always PyTorch (compilable, Inductor fuses with surrounding ops)
        cos_sin = self._get_cos_sin_cache(layer)
        q, k = rope_pytorch(q, k, cos_sin, positions,
                            self.num_heads, self.head_dim, self.num_kv_heads)

        # Write K,V to cache using reshape_and_cache_flash (flat NHD layout)
        k_write = k.view(B, self.num_kv_heads, self.head_dim)
        v_write = v.reshape(B, self.num_kv_heads, self.head_dim)
        k_scale, v_scale = self._get_kv_scales(layer)
        _vllm_ops.reshape_and_cache_flash(
            k_write, v_write,
            self.k_cache[layer], self.v_cache[layer],
            slot_mapping, "auto", k_scale, v_scale)

        # FlashInfer decode attention (plan() already called before layer loop)
        q_attn = q.view(B, self.num_heads, self.head_dim)
        attn_output = decode_wrapper.run(
            q_attn, (self.k_cache[layer], self.v_cache[layer]))
        attn_out = attn_output.reshape(B, H)

        hidden = residual + F.linear(attn_out, self.o_proj[layer])

        residual = hidden
        hidden = F.rms_norm(hidden, (H,), self.post_attn_layernorm[layer], self.rms_norm_eps)

        router_logits = F.linear(hidden, self.router[layer])
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        expert_map = (self.expert_map[layer]
                      if self.experts_per_layer is not None else None)
        hidden = self._moe_experts(hidden, self.w1[layer], self.w2[layer],
                                   topk_weights, topk_ids, expert_map)

        return residual + hidden

    def _compute_slot_mapping(self, positions, B):
        """Compute flat slot indices for reshape_and_cache_flash kernel.
        slot = page_number * page_size + offset_within_page
        """
        batch_idx = torch.arange(B, device=self.device)
        page_idx = (positions // self.page_size).long()
        offset = (positions % self.page_size).long()
        bt = self.block_table[0] if self.pp_size > 1 else self.block_table
        return (bt[batch_idx, page_idx].long() * self.page_size + offset)

    def _compute_flashinfer_metadata(self, B):
        """Compute FlashInfer paged KV metadata from CPU-side seq_lens.

        Returns (indptr_cpu, indices_gpu, last_page_len_cpu) for plan().
        Uses CPU seq_lens mirror to avoid GPU→CPU sync.
        """
        seq_lens = self._seq_lens_cpu[:B]
        pages_per_seq = (seq_lens + self.page_size - 1) // self.page_size

        indptr = torch.zeros(B + 1, dtype=torch.int32)
        indptr[1:] = pages_per_seq.cumsum(0)

        last_page_len = (seq_lens - 1) % self.page_size + 1

        # Build indices: gather page numbers from block_table per sequence
        total_pages = indptr[-1].item()
        max_pages = pages_per_seq.max().item() if B > 0 else 0
        if max_pages == 0:
            return indptr, torch.zeros(0, dtype=torch.int32, device=self.device), last_page_len

        page_range = torch.arange(max_pages, dtype=torch.int32)
        valid = page_range.unsqueeze(0) < pages_per_seq.unsqueeze(1)
        all_pages = self.block_table[:B, :max_pages].cpu()
        indices = all_pages[valid].to(torch.int32).to(self.device)

        return indptr, indices, last_page_len

    def _plan_flashinfer_decode(self, wrapper, B):
        """Call FlashInfer plan() with current batch metadata."""
        indptr, indices, last_page_len = self._compute_flashinfer_metadata(B)
        wrapper.plan(
            indptr, indices, last_page_len,
            num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, page_size=self.page_size,
            pos_encoding_mode="NONE", q_data_type=self.dtype)

    # ── Slot-based Prefill ────────────────────────────────────────────

    @torch.no_grad()
    def prefill_to_slot(self, seq_id: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill a single sequence into a specific KV cache slot (CUDA graph only).

        Args:
            seq_id: slot index in block_table / seq_lens
            input_ids: [S] token IDs (1-D)
        Returns:
            logits: [S, vocab_size]
        Requires: capture_prefill_cuda_graph() (flat) or capture_mixed_cuda_graphs()
        (piecewise, when offloading) called with sufficient sizes.
        When offload_engine is active, uses piecewise graphs via mixed_step()
        to enable expert demand-loading between stage4a and stage4b.
        """
        if self.offload_engine or self.trace_recorder or self.pp_size > 1:
            empty_dev = self.pp_devices[0] if self.pp_size > 1 else self.device
            return self.mixed_step(
                decode_seq_ids=[],
                decode_token_ids=torch.empty(0, dtype=torch.long,
                                             device=empty_dev),
                prefill_seq_ids=[seq_id],
                prefill_input_ids=[input_ids])
        S = input_ids.shape[0]
        padded_N = self._find_nearest_prefill_total(S)
        return self._prefill_graphed_flat(input_ids, [S], [seq_id], padded_N)

    # ── Chunked Prefill ──────────────────────────────────────────────

    @staticmethod
    def _compute_chunk_sizes(total_tokens, graph_sizes):
        """Compute chunk sizes for chunked prefill using greedy largest-first.

        While remaining > max graph size, use max at full capacity.
        When remaining fits in a single graph, use smallest covering graph.
        This minimizes the number of chunks.

        Args:
            total_tokens: total number of prompt tokens
            graph_sizes: list of available graph sizes
        Returns:
            list of (actual_tokens, graph_N) tuples
        """
        sorted_sizes = sorted(graph_sizes)
        largest = sorted_sizes[-1]
        chunks = []
        remaining = total_tokens
        while remaining > 0:
            # Can remaining fit in a single graph?
            covering = [gs for gs in sorted_sizes if gs >= remaining]
            if covering:
                chunks.append((remaining, min(covering)))
                remaining = 0
            else:
                # Use largest graph at full capacity
                chunks.append((largest, largest))
                remaining -= largest
        return chunks

    @torch.no_grad()
    def chunked_prefill_to_slot(self, seq_id: int,
                                input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill a sequence using chunked prefill for long prompts.

        Splits the prompt into chunks using available piecewise graph sizes.
        Chunk 1 uses FA3 self-attention; chunks 2+ use FlashInfer prefill
        with paged KV cache (attending to all previous chunks' KV entries).

        Args:
            seq_id: slot index in block_table / seq_lens
            input_ids: [S] token IDs (1-D)
        Returns:
            logits: [S, vocab_size] (only last chunk's actual logits returned)
        """
        S = input_ids.shape[0]
        graph_sizes = sorted(self._piecewise_graphs.keys())
        max_graph = max(graph_sizes)

        # If fits in a single graph, use regular prefill
        if S <= max_graph:
            return self.prefill_to_slot(seq_id, input_ids)

        chunks = self._compute_chunk_sizes(S, graph_sizes)
        empty_dev = self.pp_devices[0] if self.pp_size > 1 else self.device

        # Manually handle step counting for trace/offload controller
        _controller = (self.trace_recorder or self.replay_controller
                        or self.offload_engine)

        offset = 0
        logits = None
        for chunk_idx, (actual_len, graph_N) in enumerate(chunks):
            chunk_ids = input_ids[offset:offset + actual_len]

            if chunk_idx == 0:
                # First chunk: regular prefill (FA3 self-attention)
                logits = self.mixed_step(
                    decode_seq_ids=[],
                    decode_token_ids=torch.empty(0, dtype=torch.long,
                                                 device=empty_dev),
                    prefill_seq_ids=[seq_id],
                    prefill_input_ids=[chunk_ids])
            else:
                # Subsequent chunks: prefill with KV cache attention
                # begin_step was already called by chunk 0's mixed_step
                logits = self._prefill_chunk_continuation(
                    seq_id, chunk_ids, offset, graph_N)

            offset += actual_len

        return logits

    @torch.no_grad()
    def _prefill_chunk_continuation(self, seq_id, chunk_ids, pos_offset,
                                    graph_N):
        """Run a continuation chunk of chunked prefill.

        Uses FlashInfer prefill-with-paged-KV-cache for attention so that
        Q tokens in this chunk attend to all previous chunks' KV entries.

        Args:
            seq_id: KV cache slot index
            chunk_ids: [C] token IDs for this chunk
            pos_offset: position offset (sum of previous chunks' lengths)
            graph_N: piecewise graph size to use (>= len(chunk_ids))
        """
        info = self._piecewise_graphs[graph_N]
        _nvtx = self._nvtx_enabled
        _controller = (self.trace_recorder or self.replay_controller
                        or self.offload_engine)
        if self.replay_controller:
            self.replay_controller.begin_step()
        C = chunk_ids.shape[0]
        total_seq_len = pos_offset + C  # total tokens after this chunk
        H = self.hidden_size
        pp = self.pp_size > 1

        def _buf(layer):
            if pp:
                return info['pp_bufs'][self.pp_layer_gpu[layer]]
            return info

        # ── 1. Token IDs ──
        primary_dev = self.pp_devices[0] if pp else self.device
        if pp:
            for gpu_idx in range(self.pp_size):
                b = info['pp_bufs'][gpu_idx]
                b['static_token_ids'][:C].copy_(chunk_ids.to(primary_dev))
                if C < graph_N:
                    b['static_token_ids'][C:].zero_()
        else:
            info['static_token_ids'][:C].copy_(chunk_ids)
            if C < graph_N:
                info['static_token_ids'][C:].zero_()

        # ── 2. Positions (starting from pos_offset) ──
        positions = torch.arange(pos_offset, pos_offset + C,
                                 dtype=torch.int32, device=primary_dev)
        if pp:
            for gpu_idx in range(self.pp_size):
                b = info['pp_bufs'][gpu_idx]
                b['static_positions'][:C].copy_(positions)
                if C < graph_N:
                    b['static_positions'][C:].zero_()
        else:
            info['static_positions'][:C].copy_(positions)
            if C < graph_N:
                info['static_positions'][C:].zero_()

        # ── 3. Slot mapping (from pos_offset) ──
        bt = self.block_table[0] if pp else self.block_table
        pos_range = torch.arange(pos_offset, pos_offset + C,
                                 device=primary_dev)
        pg = (pos_range // self.page_size).long()
        off = (pos_range % self.page_size).long()
        slot_mapping = bt[seq_id, pg].long() * self.page_size + off

        if pp:
            for gpu_idx in range(self.pp_size):
                b = info['pp_bufs'][gpu_idx]
                b['static_slot_mapping'][:C].copy_(slot_mapping)
                if C < graph_N:
                    b['static_slot_mapping'][C:].fill_(-1)
        else:
            info['static_slot_mapping'][:C].copy_(slot_mapping)
            if C < graph_N:
                info['static_slot_mapping'][C:].fill_(-1)

        # ── 4. Update _seq_lens_cpu for FlashInfer prefill plan ──
        self._seq_lens_cpu[seq_id] = total_seq_len

        # ── 5. Plan FlashInfer prefill-with-paged-KV for this chunk ──
        num_pages_used = math.ceil(total_seq_len / self.page_size)
        last_page_len = total_seq_len % self.page_size
        if last_page_len == 0:
            last_page_len = self.page_size

        if pp:
            prefill_wrappers = {}
            for gpu_idx in range(self.pp_size):
                dev = self.pp_devices[gpu_idx]
                wb = self._workspace_bufs[gpu_idx]
                pw = BatchPrefillWithPagedKVCacheWrapper(wb, kv_layout="NHD")
                bt_gpu = self.block_table[gpu_idx]
                pw.plan(
                    qo_indptr=torch.tensor([0, C], dtype=torch.int32,
                                           device=dev),
                    paged_kv_indptr=torch.tensor([0, num_pages_used],
                                                 dtype=torch.int32,
                                                 device=dev),
                    paged_kv_indices=bt_gpu[seq_id, :num_pages_used].to(
                        torch.int32).to(dev),
                    paged_kv_last_page_len=torch.tensor(
                        [last_page_len], dtype=torch.int32, device=dev),
                    num_qo_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim_qk=self.head_dim,
                    page_size=self.page_size,
                    causal=True,
                    q_data_type=self.dtype,
                )
                prefill_wrappers[gpu_idx] = pw
        else:
            wb = self._workspace_buf
            pw = BatchPrefillWithPagedKVCacheWrapper(wb, kv_layout="NHD")
            pw.plan(
                qo_indptr=torch.tensor([0, C], dtype=torch.int32,
                                       device=self.device),
                paged_kv_indptr=torch.tensor([0, num_pages_used],
                                             dtype=torch.int32,
                                             device=self.device),
                paged_kv_indices=self.block_table[seq_id, :num_pages_used].to(
                    torch.int32),
                paged_kv_last_page_len=torch.tensor(
                    [last_page_len], dtype=torch.int32, device=self.device),
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                page_size=self.page_size,
                causal=True,
                q_data_type=self.dtype,
            )

        # ── 6. Embed tokens ──
        buf0 = _buf(0)
        buf0['hidden_buf'].copy_(
            F.embedding(buf0['static_token_ids'], self.embed_tokens))

        # ── 7. Per-layer piecewise replay ──
        for layer in range(self.num_layers):
            buf = _buf(layer)
            q_buf = buf['q_buf']
            attn_out_buf = buf['attn_out_buf']

            # Cross-GPU transfer at PP boundaries
            if pp and layer in self.pp_boundaries:
                prev_gpu = self.pp_layer_gpu[layer - 1]
                buf['hidden_buf'].copy_(
                    info['pp_bufs'][prev_gpu]['hidden_buf'])

            # Pre-attention prefetch (replay controller only)
            if layer == 0 and self.replay_controller:
                self.replay_controller.begin_layer_prefetch(0)

            # Stage 1: pre-attention (CUDA graph)
            info['stage1_graphs'][layer].replay()

            # Stage 3: FlashInfer prefill with paged KV cache
            q_pf = q_buf[:C]
            if pp:
                gpu_idx = self.pp_layer_gpu[layer]
                pw = prefill_wrappers[gpu_idx]
            else:
                pw = pw  # already set above
            prefill_out = pw.run(
                q_pf,
                (self.k_cache[layer], self.v_cache[layer]))
            attn_out_buf[:C].copy_(prefill_out.reshape(C, H))

            # Zero padding region
            if C < graph_N:
                attn_out_buf[C:].zero_()

            # Stage 4a: router (CUDA graph)
            info['stage4a_graphs'][layer].replay()

            # CPU break: update expert_map + record trace
            if self.replay_controller:
                self.replay_controller.process_layer_replay(
                    layer, buf['topk_ids_buf'], C)
            elif self.offload_engine:
                self.offload_engine.process_layer(
                    layer, buf['topk_ids_buf'], C,
                    router_input_buf=buf['moe_input_buf'])
                if self.trace_recorder:
                    self.trace_recorder.process_layer(
                        layer, buf['topk_ids_buf'], C,
                        router_input_buf=buf['moe_input_buf'])
            elif self.trace_recorder:
                self.trace_recorder.process_layer(
                    layer, buf['topk_ids_buf'], C,
                    router_input_buf=buf['moe_input_buf'])

            # Stage 4b: MoE (CUDA graph)
            info['stage4b_graphs'][layer].replay()

            if _controller:
                _controller.post_layer(layer)
            if self.offload_engine and _controller is not self.offload_engine:
                self.offload_engine.post_layer(layer)

        # ── 8. Final norm + lm_head ──
        last_buf = _buf(self.num_layers - 1)
        hidden = last_buf['hidden_buf']
        hidden = F.rms_norm(hidden, (self.hidden_size,), self.final_norm,
                            self.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head)
        if pp:
            logits = logits.to(self.pp_devices[0])

        # ── 9. Update seq_lens ──
        if pp:
            for sl in self.seq_lens:
                sl[seq_id] = total_seq_len
        else:
            self.seq_lens[seq_id] = total_seq_len
        # _seq_lens_cpu already set at step 4

        return logits[:C]

    # ── Multi-Batch Prefill ─────────────────────────────────────────

    @torch.no_grad()
    def prefill_batch_to_slots(self, seq_ids, input_ids):
        """Prefill multiple sequences into specific KV cache slots (CUDA graph only).

        Supports both same-length and variable-length sequences.

        Args:
            seq_ids: list[int] — KV cache slot indices for each sequence
            input_ids: list[Tensor] (variable-length 1D) or Tensor [B, S] (same-length)
        Returns:
            logits: Tensor [N_total, vocab_size] (flat)
        Requires: capture_prefill_cuda_graph() (flat) or capture_mixed_cuda_graphs()
        (piecewise, when offloading) called with sufficient sizes.
        When offload_engine is active, uses piecewise graphs via mixed_step()
        to enable expert demand-loading between stage4a and stage4b.
        """
        # Normalize to list of 1D tensors
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2:
            input_ids = [input_ids[i] for i in range(input_ids.shape[0])]

        if self.offload_engine or self.trace_recorder or self.pp_size > 1:
            empty_dev = self.pp_devices[0] if self.pp_size > 1 else self.device
            return self.mixed_step(
                decode_seq_ids=[],
                decode_token_ids=torch.empty(0, dtype=torch.long,
                                             device=empty_dev),
                prefill_seq_ids=list(seq_ids),
                prefill_input_ids=list(input_ids))

        seq_lengths = [ids.shape[0] for ids in input_ids]
        total = sum(seq_lengths)
        padded_N = self._find_nearest_prefill_total(total)
        token_ids_flat = torch.cat(input_ids)
        return self._prefill_graphed_flat(
            token_ids_flat, seq_lengths, seq_ids, padded_N)

    # ── Flat Prefill Helpers ─────────────────────────────────────────

    def _find_nearest_prefill_total(self, total_tokens):
        """Find smallest captured flat prefill graph with N >= total_tokens.
        Raises RuntimeError if none found (no eager fallback).
        """
        candidates = [n for n in self._prefill_cuda_graphs if n >= total_tokens]
        if not candidates:
            captured = sorted(self._prefill_cuda_graphs.keys())
            raise RuntimeError(
                f"No prefill CUDA graph captured for {total_tokens} tokens. "
                f"Captured sizes: {captured}. Call capture_prefill_cuda_graph() "
                f"with a sufficient total_token_sizes list.")
        return min(candidates)

    def _compute_flat_slot_mapping(self, seq_ids, seq_lengths, padded_total):
        """Compute slot_mapping for variable-length sequences at arbitrary KV slots.

        Real tokens get valid slot indices; padding tokens get -1 (vLLM sentinel
        — reshape_and_cache_flash skips writes for slot_idx < 0).

        Args:
            seq_ids: list[int] — KV cache slot indices
            seq_lengths: list[int] — per-sequence token counts
            padded_total: int — total length including padding
        Returns: [padded_total] int64 tensor
        """
        parts = []
        for sid, length in zip(seq_ids, seq_lengths):
            pos = torch.arange(length, device=self.device)
            pg = (pos // self.page_size).long()
            off = (pos % self.page_size).long()
            parts.append(self.block_table[sid, pg].long() * self.page_size + off)
        real = torch.cat(parts) if parts else torch.empty(
            0, dtype=torch.long, device=self.device)
        n_real = real.shape[0]
        if n_real < padded_total:
            padding = torch.full(
                (padded_total - n_real,), -1, dtype=torch.long, device=self.device)
            return torch.cat([real, padding])
        return real

    def _compute_flat_positions(self, seq_lengths, padded_total):
        """Compute positions [0..S1-1, 0..S2-1, ...], pad remainder with 0.

        Args:
            seq_lengths: list[int] — per-sequence token counts
            padded_total: int — total length including padding
        Returns: [padded_total] int32 tensor
        """
        parts = [torch.arange(s, dtype=torch.int32, device=self.device)
                 for s in seq_lengths]
        real = torch.cat(parts) if parts else torch.empty(
            0, dtype=torch.int32, device=self.device)
        n_real = real.shape[0]
        if n_real < padded_total:
            padding = torch.zeros(
                padded_total - n_real, dtype=torch.int32, device=self.device)
            return torch.cat([real, padding])
        return real

    def _compute_flat_cu_seqlens(self, seq_lengths, max_bs):
        """Build cu_seqlens [max_bs + 1] for FA3 varlen attention.

        Unused entries repeat the final cumsum value, creating zero-length
        sequences that FA3 handles as no-ops.

        Args:
            seq_lengths: list[int] — per-sequence token counts
            max_bs: int — max batch size (determines output length)
        Returns: [max_bs + 1] int32 tensor
        """
        cu = torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device)
        cum = 0
        for i, s in enumerate(seq_lengths):
            cum += s
            cu[i + 1] = cum
        # Fill remaining entries with final cumsum (zero-length seqs)
        cu[len(seq_lengths) + 1:] = cum
        return cu

    # ── Mixed Batch Support ──────────────────────────────────────────

    def _plan_flashinfer_decode_for_subset(self, seq_ids):
        """Plan FlashInfer decode for a subset of sequence slots.

        Unlike _plan_flashinfer_decode which assumes contiguous [:B],
        this handles arbitrary slot indices.
        """
        B = len(seq_ids)
        sid_tensor = torch.tensor(seq_ids, dtype=torch.long)
        seq_lens = self._seq_lens_cpu[sid_tensor]
        pages_per_seq = (seq_lens + self.page_size - 1) // self.page_size

        indptr = torch.zeros(B + 1, dtype=torch.int32)
        indptr[1:] = pages_per_seq.cumsum(0)

        last_page_len = (seq_lens - 1) % self.page_size + 1

        max_pages = pages_per_seq.max().item() if B > 0 else 0
        if max_pages == 0:
            indices = torch.zeros(0, dtype=torch.int32, device=self.device)
        else:
            page_range = torch.arange(max_pages, dtype=torch.int32)
            valid = page_range.unsqueeze(0) < pages_per_seq.unsqueeze(1)
            all_pages = self.block_table[sid_tensor, :max_pages].cpu()
            indices = all_pages[valid].to(torch.int32).to(self.device)

        self._decode_wrapper.plan(
            indptr, indices, last_page_len,
            num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, page_size=self.page_size,
            pos_encoding_mode="NONE", q_data_type=self.dtype)

    def _plan_flashinfer_decode_pp(self, seq_ids, gpu_idx):
        """Plan FlashInfer decode for a subset of slots on a specific PP GPU."""
        B = len(seq_ids)
        sid_tensor = torch.tensor(seq_ids, dtype=torch.long)
        seq_lens = self._seq_lens_cpu[sid_tensor]
        pages_per_seq = (seq_lens + self.page_size - 1) // self.page_size

        indptr = torch.zeros(B + 1, dtype=torch.int32)
        indptr[1:] = pages_per_seq.cumsum(0)
        last_page_len = (seq_lens - 1) % self.page_size + 1

        dev = self.pp_devices[gpu_idx]
        bt = self.block_table[gpu_idx]
        max_pages = pages_per_seq.max().item() if B > 0 else 0
        if max_pages == 0:
            indices = torch.zeros(0, dtype=torch.int32, device=dev)
        else:
            page_range = torch.arange(max_pages, dtype=torch.int32)
            valid = page_range.unsqueeze(0) < pages_per_seq.unsqueeze(1)
            all_pages = bt[sid_tensor, :max_pages].cpu()
            indices = all_pages[valid].to(torch.int32).to(dev)

        self._decode_wrappers[gpu_idx].plan(
            indptr, indices, last_page_len,
            num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, page_size=self.page_size,
            pos_encoding_mode="NONE", q_data_type=self.dtype)

    def _layer_mixed(self, hidden, layer, positions, slot_mapping,
                     num_decode_tokens, decode_wrapper,
                     prefill_cu_seqlens, prefill_max_seqlen):
        """Single transformer layer for mixed decode+prefill batch.

        Args:
            hidden: [N_total, H] — concatenated [decode_hidden | prefill_hidden]
            layer: layer index
            positions: [N_total] int32
            slot_mapping: [N_total] int64
            num_decode_tokens: boundary — hidden[:D] are decode, [D:] are prefill
            decode_wrapper: FlashInfer BatchDecode (pre-planned)
            prefill_cu_seqlens: [num_prefill_reqs + 1] int32, for FA3
            prefill_max_seqlen: max prefill seq length
        """
        N = hidden.shape[0]
        H = self.hidden_size
        kv_dim = self.num_kv_heads * self.head_dim
        D = num_decode_tokens

        residual = hidden
        hidden = F.rms_norm(hidden, (H,), self.input_layernorm[layer],
                            self.rms_norm_eps)

        # Fused QKV projection on ALL tokens
        qkv = F.linear(hidden, self.qkv_proj[layer])
        q, k, v = qkv.split([H, kv_dim, kv_dim], dim=-1)
        v = v.contiguous()

        # Q/K norm (if model has it)
        if self.has_qk_norm:
            q = F.rms_norm(q, (H,), self.q_norm[layer],
                           self.rms_norm_eps).contiguous()
            k = F.rms_norm(k, (kv_dim,), self.k_norm[layer],
                           self.rms_norm_eps).contiguous()

        # RoPE on ALL tokens
        cos_sin = self._get_cos_sin_cache(layer)
        q, k = rope_pytorch(q, k, cos_sin, positions,
                            self.num_heads, self.head_dim, self.num_kv_heads)

        # Write K,V to paged cache for ALL tokens
        k_write = k.reshape(N, self.num_kv_heads, self.head_dim)
        v_write = v.reshape(N, self.num_kv_heads, self.head_dim)
        k_scale, v_scale = self._get_kv_scales(layer)
        _vllm_ops.reshape_and_cache_flash(
            k_write, v_write,
            self.k_cache[layer], self.v_cache[layer],
            slot_mapping, "auto", k_scale, v_scale)

        # ── Split attention ──
        # Decode tokens [0:D]: FlashInfer BatchDecode (reads paged KV cache)
        if D > 0 and N > D:
            # Mixed: both decode and prefill
            q_decode = q[:D].view(D, self.num_heads, self.head_dim)
            decode_out = decode_wrapper.run(
                q_decode, (self.k_cache[layer], self.v_cache[layer]))
            decode_out = decode_out.reshape(D, H)

            # Prefill tokens [D:]: FA3 (stateless) instead of FlashInfer —
            # FlashInfer's fmha_varlen_plan() allocates fresh GPU tensors per
            # call, so addresses baked into a CUDA graph get freed by the next
            # plan() call.
            N_pf = N - D
            q_pf = q[D:].reshape(N_pf, self.num_heads, self.head_dim)
            k_pf = k_write[D:]
            v_pf = v_write[D:]
            prefill_out = flash_attn_varlen_func(
                q_pf, k_pf, v_pf,
                cu_seqlens_q=prefill_cu_seqlens,
                cu_seqlens_k=prefill_cu_seqlens,
                max_seqlen_q=prefill_max_seqlen,
                max_seqlen_k=prefill_max_seqlen,
                causal=True, fa_version=3)
            prefill_out = prefill_out.reshape(N_pf, H)

            attn_out = torch.cat([decode_out, prefill_out], dim=0)

        elif D > 0:
            # Pure decode
            q_decode = q[:D].view(D, self.num_heads, self.head_dim)
            attn_out = decode_wrapper.run(
                q_decode, (self.k_cache[layer], self.v_cache[layer]))
            attn_out = attn_out.reshape(D, H)

        else:
            # Pure prefill — FA3 (stateless), same reason as mixed path above
            N_pf = N
            q_pf = q.reshape(N_pf, self.num_heads, self.head_dim)
            k_pf = k_write
            v_pf = v_write
            attn_out = flash_attn_varlen_func(
                q_pf, k_pf, v_pf,
                cu_seqlens_q=prefill_cu_seqlens,
                cu_seqlens_k=prefill_cu_seqlens,
                max_seqlen_q=prefill_max_seqlen,
                max_seqlen_k=prefill_max_seqlen,
                causal=True, fa_version=3)
            attn_out = attn_out.reshape(N_pf, H)

        hidden = residual + F.linear(attn_out, self.o_proj[layer])

        # Post-attention norm + MoE on ALL tokens
        residual = hidden
        hidden = F.rms_norm(hidden, (H,), self.post_attn_layernorm[layer],
                            self.rms_norm_eps)

        router_logits = F.linear(hidden, self.router[layer])
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        expert_map = (self.expert_map[layer]
                      if self.experts_per_layer is not None else None)
        hidden = self._moe_experts(hidden, self.w1[layer], self.w2[layer],
                                   topk_weights, topk_ids, expert_map)

        return residual + hidden

    # ── Piecewise Stage Functions ────────────────────────────────────

    def _layer_stage1_pre_attn(self, hidden, positions, slot_mapping, layer,
                               q_buf, k_buf, v_buf, residual_buf):
        """Stage 1: RMSNorm -> QKV -> Q/K norm -> RoPE -> KV cache write.

        Writes to q_buf, k_buf, v_buf, residual_buf (pre-allocated static
        buffers for CUDA graph capture). Returns nothing — outputs are in buffers.
        """
        N = hidden.shape[0]
        H = self.hidden_size
        kv_dim = self.num_kv_heads * self.head_dim

        normed = F.rms_norm(hidden, (H,), self.input_layernorm[layer],
                            self.rms_norm_eps)

        qkv = F.linear(normed, self.qkv_proj[layer])
        q, k, v = qkv.split([H, kv_dim, kv_dim], dim=-1)
        v = v.contiguous()

        if self.has_qk_norm:
            q = F.rms_norm(q, (H,), self.q_norm[layer],
                           self.rms_norm_eps).contiguous()
            k = F.rms_norm(k, (kv_dim,), self.k_norm[layer],
                           self.rms_norm_eps).contiguous()

        cos_sin = self._get_cos_sin_cache(layer)
        q, k = rope_pytorch(q, k, cos_sin, positions,
                            self.num_heads, self.head_dim, self.num_kv_heads)

        k_3d = k.reshape(N, self.num_kv_heads, self.head_dim)
        v_3d = v.reshape(N, self.num_kv_heads, self.head_dim)
        k_scale, v_scale = self._get_kv_scales(layer)
        _vllm_ops.reshape_and_cache_flash(
            k_3d, v_3d,
            self.k_cache[layer], self.v_cache[layer],
            slot_mapping, "auto", k_scale, v_scale)

        q_buf.copy_(q.view(N, self.num_heads, self.head_dim))
        k_buf.copy_(k_3d)
        v_buf.copy_(v_3d)
        residual_buf.copy_(hidden)

    def _layer_stage4a_router(self, attn_out, residual, layer,
                              moe_input_buf, moe_residual_buf,
                              topk_weights_buf, topk_ids_buf):
        """Stage 4a: O proj -> residual -> norm -> router -> topk.

        Writes routing decisions into buffers for CPU-side inspection between
        stage4a and stage4b. Used for expert offloading (demand loading between
        the two sub-stages).
        """
        H = self.hidden_size
        hidden = residual + F.linear(attn_out, self.o_proj[layer])
        moe_residual_buf.copy_(hidden)
        hidden = F.rms_norm(hidden, (H,), self.post_attn_layernorm[layer],
                            self.rms_norm_eps)
        moe_input_buf.copy_(hidden)

        router_logits = F.linear(hidden, self.router[layer])
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights_buf.copy_(topk_weights)
        topk_ids_buf.copy_(topk_ids)

    def _layer_stage4b_moe(self, moe_input_buf, moe_residual_buf,
                           topk_weights_buf, topk_ids_buf, hidden_out, layer):
        """Stage 4b: MoE compute -> residual add.

        Reads routing decisions and post-norm hidden states from buffers,
        runs MoE, writes output into hidden_out for next layer.

        In offloading mode (experts_per_layer or cache_size set), reads
        from the unified buffer (w1_buf/w2_buf/expert_map_buf). The caller
        must update expert_map_buf with this layer's absolute map before
        calling this.
        """
        if self.offloading:
            w1, w2 = self.w1_buf, self.w2_buf
            expert_map = self.expert_map_buf
        else:
            w1, w2 = self.w1[layer], self.w2[layer]
            expert_map = None
        hidden = self._moe_experts(moe_input_buf, w1, w2,
                                   topk_weights_buf, topk_ids_buf, expert_map)
        hidden_out.copy_(moe_residual_buf + hidden)

    def _full_mixed_graph_body(self, all_token_ids, positions, slot_mapping,
                               num_decode_tokens, decode_wrapper,
                               prefill_cu_seqlens, prefill_max_seqlen):
        """Full mixed forward — embed + layers + final norm + lm_head.

        Designed for torch.compile(dynamic=True) to fuse RMSNorm + residual
        + RoPE between graph-breaking external kernels.
        """
        hidden = F.embedding(all_token_ids, self.embed_tokens)
        for layer in range(self.num_layers):
            hidden = self._layer_mixed(hidden, layer, positions, slot_mapping,
                                       num_decode_tokens, decode_wrapper,
                                       prefill_cu_seqlens, prefill_max_seqlen)
        hidden = F.rms_norm(hidden, (self.hidden_size,), self.final_norm,
                            self.rms_norm_eps)
        return F.linear(hidden, self.lm_head)

    @torch.no_grad()
    def mixed_step(self, decode_seq_ids, decode_token_ids,
                   prefill_seq_ids, prefill_input_ids,
                   continuation_seq_ids=None, continuation_input_ids=None,
                   continuation_offsets=None):
        """Mixed prefill+decode forward pass with optional continuation chunks.

        Concatenates all tokens as [decode | new-prefill | continuation],
        runs shared compute, splits at attention:
          - Stage 2: FlashInfer decode for decode tokens
          - Stage 3a: FA3 self-attention for new prefills (positions from 0)
          - Stage 3b: FlashInfer paged-KV for continuations (positions from offset)

        New prefills start from position 0 (no prior KV); Q attends only to K/V
        within the same call (FA3 causal self-attention).

        Continuations resume a partially-prefilled prompt. Q tokens attend to ALL
        prior KV entries in paged cache (from earlier chunks) plus each other
        (FlashInfer BatchPrefillWithPagedKVCache).

        Caller contract for continuations:
          - Each continuation_offsets[i] MUST equal the current seq_lens for
            that sequence (the number of tokens already in KV cache). An
            assertion verifies this at runtime.
          - Pages for positions [offset, offset + chunk_len) must already be
            allocated in the block table (static allocation handles this).
          - After the call, seq_lens[sid] = offset + chunk_len.

        Output extraction:
          - logits[:D] — decode tokens (one per sequence)
          - logits[D:D+P] — new prefill tokens (concatenated, variable-length)
          - logits[D+P:D+P+C] — continuation tokens (concatenated)
          - To get a sequence's "next token prediction", take the last logit
            row in that sequence's region.

        Args:
            decode_seq_ids: list[int] — KV cache slot indices for decode requests
            decode_token_ids: Tensor [D] — one token per decode request
            prefill_seq_ids: list[int] — KV cache slot indices for new prefills
            prefill_input_ids: list[Tensor] — variable-length new prefill sequences
            continuation_seq_ids: list[int] — KV slots for continuation prefills
            continuation_input_ids: list[Tensor] — continuation chunk tokens
            continuation_offsets: list[int] — position offset for each continuation
                (number of tokens already prefilled for that sequence)
        Returns:
            logits: Tensor [N_total, vocab_size]
                logits[:D] = decode, [D:D+P] = new prefill, [D+P:] = continuation
        """
        if continuation_seq_ids is None:
            continuation_seq_ids = []
        if continuation_input_ids is None:
            continuation_input_ids = []
        if continuation_offsets is None:
            continuation_offsets = []

        D = len(decode_seq_ids)
        prefill_lengths = [ids.shape[0] for ids in prefill_input_ids]
        cont_lengths = [ids.shape[0] for ids in continuation_input_ids]
        N_total = D + sum(prefill_lengths) + sum(cont_lengths)

        # ── Auto-dispatch to piecewise graphs if available ──
        graph_N = self._find_nearest_piecewise_graph(N_total)
        if graph_N is not None:
            return self._mixed_step_piecewise(
                decode_seq_ids, decode_token_ids,
                prefill_seq_ids, prefill_input_ids,
                continuation_seq_ids, continuation_input_ids,
                continuation_offsets, graph_N)

        if continuation_seq_ids:
            raise NotImplementedError(
                "Continuation prefill requires piecewise CUDA graphs. "
                f"Call capture_mixed_cuda_graphs() with sizes >= {N_total}.")

        if self.offload_engine or self.trace_recorder:
            raise RuntimeError(
                f"No piecewise CUDA graph covers {N_total} tokens. "
                f"Offload engine / trace recorder requires piecewise graphs. "
                f"Call capture_mixed_cuda_graphs() with sizes >= {N_total}.")

        if self.pp_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism requires piecewise CUDA graphs. "
                f"Call capture_mixed_cuda_graphs() with sizes >= {N_total}.")

        # ── Eager fallback below ──

        # ── Positions ──
        # Decode: current seq_len (position of next token) — capture BEFORE incr
        decode_positions = (self._seq_lens_cpu[decode_seq_ids].to(torch.int32)
                            .to(self.device)) if D > 0 else torch.empty(
                                0, dtype=torch.int32, device=self.device)

        # Prefill: 0..S-1 per request
        prefill_pos_parts = [
            torch.arange(s, dtype=torch.int32, device=self.device)
            for s in prefill_lengths
        ]
        positions = torch.cat([decode_positions] + prefill_pos_parts)

        # ── Increment _seq_lens_cpu for decode BEFORE FlashInfer plan ──
        if D > 0:
            for sid in decode_seq_ids:
                self._seq_lens_cpu[sid] += 1
            self._plan_flashinfer_decode_for_subset(decode_seq_ids)

        # ── Slot mapping ──
        if D > 0:
            d_idx = torch.tensor(decode_seq_ids, device=self.device, dtype=torch.long)
            d_page = (decode_positions // self.page_size).long()
            d_offset = (decode_positions % self.page_size).long()
            decode_slots = (self.block_table[d_idx, d_page].long() * self.page_size
                            + d_offset)
        else:
            decode_slots = torch.empty(0, dtype=torch.long, device=self.device)

        prefill_slot_parts = []
        for sid, length in zip(prefill_seq_ids, prefill_lengths):
            pos = torch.arange(length, device=self.device)
            pg = (pos // self.page_size).long()
            off = (pos % self.page_size).long()
            prefill_slot_parts.append(
                self.block_table[sid, pg].long() * self.page_size + off)

        slot_mapping = torch.cat([decode_slots] + prefill_slot_parts)

        # ── Concatenate all token IDs ──
        token_parts = []
        if D > 0:
            token_parts.append(decode_token_ids)
        for ids in prefill_input_ids:
            token_parts.append(ids)
        all_token_ids = torch.cat(token_parts)

        # ── Prefill cu_seqlens for FA3 ──
        prefill_cu = torch.zeros(len(prefill_lengths) + 1, dtype=torch.int32,
                                 device=self.device)
        for i, s in enumerate(prefill_lengths):
            prefill_cu[i + 1] = prefill_cu[i] + s
        prefill_max = max(prefill_lengths) if prefill_lengths else 0

        # ── Forward ──
        logits = self._full_mixed_graph_body(
            all_token_ids, positions, slot_mapping,
            D, self._decode_wrapper, prefill_cu, prefill_max)

        # ── Update GPU seq_lens ──
        for sid in decode_seq_ids:
            self.seq_lens[sid] += 1
        for sid, length in zip(prefill_seq_ids, prefill_lengths):
            self.seq_lens[sid] = length
            self._seq_lens_cpu[sid] = length

        return logits

    # ── Piecewise CUDA Graph Capture & Replay ────────────────────────

    def _create_intermediate_buffers(self, N, device):
        """Create intermediate buffers for piecewise CUDA graph capture.

        Data flow between stages (per layer):
          Stage 1:  token_ids → q/k/v_buf (QKV + RoPE + KV cache write)
          Stage 2:  q_buf → attn_out_buf (decode attention, eager)
          Stage 3:  q_buf → attn_out_buf (prefill attention, eager)
          Stage 4a: attn_out_buf + residual_buf → moe_input_buf,
                    moe_residual_buf, topk_weights_buf, topk_ids_buf
                    (O proj + residual + norm + router)
          [CPU break: offload engine inspects routing, loads experts]
          Stage 4b: moe_input_buf + topk → hidden_buf (MoE + residual)
        hidden_buf becomes residual_buf for the next layer.
        """
        total_kv_slots = self.total_pages * self.page_size
        return {
            'q_buf': torch.zeros(N, self.num_heads, self.head_dim,
                                 dtype=self.dtype, device=device),
            'k_buf': torch.zeros(N, self.num_kv_heads, self.head_dim,
                                 dtype=self.dtype, device=device),
            'v_buf': torch.zeros(N, self.num_kv_heads, self.head_dim,
                                 dtype=self.dtype, device=device),
            'attn_out_buf': torch.zeros(N, self.num_heads * self.head_dim,
                                        dtype=self.dtype, device=device),
            'residual_buf': torch.zeros(N, self.hidden_size,
                                        dtype=self.dtype, device=device),
            'hidden_buf': torch.zeros(N, self.hidden_size,
                                      dtype=self.dtype, device=device),
            'moe_input_buf': torch.zeros(N, self.hidden_size,
                                         dtype=self.dtype, device=device),
            'moe_residual_buf': torch.zeros(N, self.hidden_size,
                                            dtype=self.dtype, device=device),
            'topk_weights_buf': torch.zeros(N, self.top_k,
                                            dtype=torch.float32,
                                            device=device),
            'topk_ids_buf': torch.zeros(N, self.top_k, dtype=torch.int64,
                                        device=device),
            'static_positions': (torch.arange(N, dtype=torch.int32,
                                              device=device)
                                 % self.max_seq_len),
            'static_slot_mapping': (torch.arange(N, dtype=torch.long,
                                                 device=device)
                                    % total_kv_slots),
            'static_token_ids': torch.randint(1, 1000, (N,), device=device),
        }

    def capture_mixed_cuda_graphs(self, total_token_sizes=None,
                                  use_torch_compile=None):
        """Capture per-layer piecewise CUDA graphs for mixed batches.

        For each N in total_token_sizes, captures 3 graphs per layer:
          - Stage 1:  RMSNorm -> QKV -> Q/K norm -> RoPE -> KV cache write
          - Stage 4a: O proj -> residual -> norm -> router -> topk
          - Stage 4b: fused_experts -> residual add

        Stages 2 & 3 (attention) run eagerly — single kernel each.
        The split between 4a and 4b creates a CPU break where the offload
        engine can inspect routing decisions and load missing experts.

        When pipeline_parallel_size > 1, creates separate buffer sets per GPU
        and captures each layer's graphs on its assigned GPU.

        Args:
            total_token_sizes: List of total token counts to capture
                (default: [128, 256, 512, 1024, 2048])
            use_torch_compile: Override instance default
        """
        if total_token_sizes is None:
            total_token_sizes = [128, 256, 512, 1024, 2048]
        if use_torch_compile is None:
            use_torch_compile = self.use_torch_compile

        if not hasattr(self, '_piecewise_graphs'):
            self._piecewise_graphs = {}

        # Shared graph memory pool — all piecewise graphs share a single
        # allocator pool per device. Safe because: (1) graphs replay strictly
        # sequentially (layer-by-layer, stage-by-stage), (2) all outputs use
        # .copy_() to pre-allocated buffers outside the pool, leaving zero
        # live tensors in the pool after each capture block. See GRAPH_DEBUG.md.
        if self.pp_size > 1:
            graph_pools = {
                gpu_idx: torch.cuda.graph_pool_handle()
                for gpu_idx in range(self.pp_size)
            }
        else:
            graph_pool = torch.cuda.graph_pool_handle()

        # Compile stage functions. fullgraph=False because FlashInfer and
        # fused_moe contain ops Dynamo can't trace. The resulting graph
        # breaks add overhead (~32 enter/exit per forward), but CUDA graph
        # capture on top eliminates it entirely.
        stage1_fn = self._layer_stage1_pre_attn
        stage4a_fn = self._layer_stage4a_router
        stage4b_fn = self._layer_stage4b_moe
        if use_torch_compile:
            stage1_fn = torch.compile(stage1_fn, fullgraph=False)
            stage4a_fn = torch.compile(stage4a_fn, fullgraph=False)
            stage4b_fn = torch.compile(stage4b_fn, fullgraph=False)

        for N in total_token_sizes:
            self.reset()

            if self.pp_size > 1:
                # Per-GPU buffer sets
                pp_bufs = {
                    gpu_idx: self._create_intermediate_buffers(N, dev)
                    for gpu_idx, dev in enumerate(self.pp_devices)
                }

                # ── Warmup all layers ──
                n_warmup = 5 if use_torch_compile else 3
                for _ in range(n_warmup):
                    pp_bufs[0]['hidden_buf'].copy_(
                        F.embedding(pp_bufs[0]['static_token_ids'],
                                    self.embed_tokens))
                    for layer in range(self.num_layers):
                        gpu_idx = self.pp_layer_gpu[layer]
                        buf = pp_bufs[gpu_idx]
                        dev = self.pp_devices[gpu_idx]
                        if layer in self.pp_boundaries:
                            prev_gpu = self.pp_layer_gpu[layer - 1]
                            buf['hidden_buf'].copy_(
                                pp_bufs[prev_gpu]['hidden_buf'])
                        with torch.cuda.device(dev):
                            stage1_fn(buf['hidden_buf'],
                                      buf['static_positions'],
                                      buf['static_slot_mapping'], layer,
                                      buf['q_buf'], buf['k_buf'],
                                      buf['v_buf'], buf['residual_buf'])
                            buf['attn_out_buf'].copy_(
                                buf['q_buf'].reshape(N, -1))
                            stage4a_fn(buf['attn_out_buf'],
                                       buf['residual_buf'], layer,
                                       buf['moe_input_buf'],
                                       buf['moe_residual_buf'],
                                       buf['topk_weights_buf'],
                                       buf['topk_ids_buf'])
                            stage4b_fn(buf['moe_input_buf'],
                                       buf['moe_residual_buf'],
                                       buf['topk_weights_buf'],
                                       buf['topk_ids_buf'],
                                       buf['hidden_buf'], layer)
                for d in self.pp_devices:
                    torch.cuda.synchronize(d)

                # ── Capture per-layer graphs ──
                # Explicit per-device streams required: the default
                # torch.cuda.graph() fails across devices (PyTorch bug
                # with cross-device graph memory pool sharing).
                pp_streams = {
                    gpu_idx: torch.cuda.Stream(device=dev)
                    for gpu_idx, dev in enumerate(self.pp_devices)
                }
                stage1_graphs = []
                stage4a_graphs = []
                stage4b_graphs = []

                pp_bufs[0]['hidden_buf'].copy_(
                    F.embedding(pp_bufs[0]['static_token_ids'],
                                self.embed_tokens))

                for layer in range(self.num_layers):
                    gpu_idx = self.pp_layer_gpu[layer]
                    buf = pp_bufs[gpu_idx]
                    dev = self.pp_devices[gpu_idx]
                    stream = pp_streams[gpu_idx]
                    if layer in self.pp_boundaries:
                        prev_gpu = self.pp_layer_gpu[layer - 1]
                        buf['hidden_buf'].copy_(
                            pp_bufs[prev_gpu]['hidden_buf'])

                    with torch.cuda.device(dev):
                        g1 = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g1, pool=graph_pools[gpu_idx],
                                              stream=stream):
                            stage1_fn(buf['hidden_buf'],
                                      buf['static_positions'],
                                      buf['static_slot_mapping'], layer,
                                      buf['q_buf'], buf['k_buf'],
                                      buf['v_buf'], buf['residual_buf'])
                        stage1_graphs.append(g1)

                    buf['attn_out_buf'].copy_(buf['q_buf'].reshape(N, -1))

                    with torch.cuda.device(dev):
                        g4a = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g4a, pool=graph_pools[gpu_idx],
                                              stream=stream):
                            stage4a_fn(buf['attn_out_buf'],
                                       buf['residual_buf'], layer,
                                       buf['moe_input_buf'],
                                       buf['moe_residual_buf'],
                                       buf['topk_weights_buf'],
                                       buf['topk_ids_buf'])
                        stage4a_graphs.append(g4a)

                    with torch.cuda.device(dev):
                        g4b = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g4b, pool=graph_pools[gpu_idx],
                                              stream=stream):
                            stage4b_fn(buf['moe_input_buf'],
                                       buf['moe_residual_buf'],
                                       buf['topk_weights_buf'],
                                       buf['topk_ids_buf'],
                                       buf['hidden_buf'], layer)
                        stage4b_graphs.append(g4b)

                self._piecewise_graphs[N] = {
                    'stage1_graphs': stage1_graphs,
                    'stage4a_graphs': stage4a_graphs,
                    'stage4b_graphs': stage4b_graphs,
                    'pp_bufs': pp_bufs,
                }

                compile_str = " + torch.compile" if use_torch_compile else ""
                print(f"  Piecewise CUDA graphs{compile_str} captured for "
                      f"N={N} ({self.num_layers * 3} graphs, "
                      f"PP={self.pp_size})")
            else:
                # Single-GPU path (original)
                bufs = self._create_intermediate_buffers(N, self.device)

                # ── Warmup all layers ──
                n_warmup = 5 if use_torch_compile else 3
                for _ in range(n_warmup):
                    bufs['hidden_buf'].copy_(
                        F.embedding(bufs['static_token_ids'],
                                    self.embed_tokens))
                    for layer in range(self.num_layers):
                        stage1_fn(bufs['hidden_buf'],
                                  bufs['static_positions'],
                                  bufs['static_slot_mapping'], layer,
                                  bufs['q_buf'], bufs['k_buf'],
                                  bufs['v_buf'], bufs['residual_buf'])
                        bufs['attn_out_buf'].copy_(
                            bufs['q_buf'].reshape(N, -1))
                        stage4a_fn(bufs['attn_out_buf'],
                                   bufs['residual_buf'], layer,
                                   bufs['moe_input_buf'],
                                   bufs['moe_residual_buf'],
                                   bufs['topk_weights_buf'],
                                   bufs['topk_ids_buf'])
                        if self.offloading:
                            self.expert_map_buf.copy_(
                                self.expert_map_abs[layer])
                        stage4b_fn(bufs['moe_input_buf'],
                                   bufs['moe_residual_buf'],
                                   bufs['topk_weights_buf'],
                                   bufs['topk_ids_buf'],
                                   bufs['hidden_buf'], layer)
                torch.cuda.synchronize()

                # ── Capture per-layer graphs ──
                # Use explicit stream to avoid cross-device CUDA graph pool issues
                capture_stream = torch.cuda.Stream(device=self.device)
                stage1_graphs = []
                stage4a_graphs = []
                stage4b_graphs = []

                bufs['hidden_buf'].copy_(
                    F.embedding(bufs['static_token_ids'],
                                self.embed_tokens))

                for layer in range(self.num_layers):
                    g1 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g1, pool=graph_pool,
                                          stream=capture_stream):
                        stage1_fn(bufs['hidden_buf'],
                                  bufs['static_positions'],
                                  bufs['static_slot_mapping'], layer,
                                  bufs['q_buf'], bufs['k_buf'],
                                  bufs['v_buf'], bufs['residual_buf'])
                    stage1_graphs.append(g1)

                    bufs['attn_out_buf'].copy_(
                        bufs['q_buf'].reshape(N, -1))

                    g4a = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g4a, pool=graph_pool,
                                          stream=capture_stream):
                        stage4a_fn(bufs['attn_out_buf'],
                                   bufs['residual_buf'], layer,
                                   bufs['moe_input_buf'],
                                   bufs['moe_residual_buf'],
                                   bufs['topk_weights_buf'],
                                   bufs['topk_ids_buf'])
                    stage4a_graphs.append(g4a)

                    if self.offloading:
                        self.expert_map_buf.copy_(
                            self.expert_map_abs[layer])

                    g4b = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g4b, pool=graph_pool,
                                          stream=capture_stream):
                        stage4b_fn(bufs['moe_input_buf'],
                                   bufs['moe_residual_buf'],
                                   bufs['topk_weights_buf'],
                                   bufs['topk_ids_buf'],
                                   bufs['hidden_buf'], layer)
                    stage4b_graphs.append(g4b)

                # Store buffers directly in dict for single-GPU compat
                self._piecewise_graphs[N] = {
                    'stage1_graphs': stage1_graphs,
                    'stage4a_graphs': stage4a_graphs,
                    'stage4b_graphs': stage4b_graphs,
                    **bufs,
                }

                compile_str = " + torch.compile" if use_torch_compile else ""
                print(f"  Piecewise CUDA graphs{compile_str} captured for "
                      f"N={N} ({self.num_layers * 3} graphs)")

    @staticmethod
    def vllm_graph_sizes(max_graph_size: int) -> list[int]:
        """Generate vLLM-style CUDA graph capture sizes.

        Pattern: [1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)
        All sizes <= max_graph_size.
        """
        sizes = [1, 2, 4]
        sizes.extend(range(8, min(256, max_graph_size + 1), 8))
        if max_graph_size >= 256:
            sizes.extend(range(256, max_graph_size + 1, 16))
        return sorted(set(s for s in sizes if s <= max_graph_size))

    def graph_memory_overhead_bytes(self) -> int:
        """Return GPU memory used by captured CUDA graphs.

        Counts static buffer tensors in _piecewise_graphs. Does not include
        CUDA graph command buffers (not directly measurable via tensor API).
        For a more accurate total, compare torch.cuda.memory_allocated()
        before and after capture_mixed_cuda_graphs().
        """
        if not hasattr(self, '_piecewise_graphs') or not self._piecewise_graphs:
            return 0
        total = 0
        for N, info in self._piecewise_graphs.items():
            if 'pp_bufs' in info:
                for gpu_idx, buf in info['pp_bufs'].items():
                    for k, v in buf.items():
                        if isinstance(v, torch.Tensor):
                            total += v.nelement() * v.element_size()
            else:
                for k, v in info.items():
                    if isinstance(v, torch.Tensor):
                        total += v.nelement() * v.element_size()
        return total

    def _find_nearest_piecewise_graph(self, total_tokens):
        """Find smallest piecewise graph with N >= total_tokens, or None."""
        if not hasattr(self, '_piecewise_graphs'):
            return None
        candidates = [n for n in self._piecewise_graphs if n >= total_tokens]
        return min(candidates) if candidates else None

    @torch.no_grad()
    def _mixed_step_piecewise(self, decode_seq_ids, decode_token_ids,
                              prefill_seq_ids, prefill_input_ids,
                              continuation_seq_ids, continuation_input_ids,
                              continuation_offsets, graph_N):
        """Replay mixed step using per-layer piecewise CUDA graphs.

        Stage 1 & 4 are CUDA graphs (keyed by N_total).
        Attention stages run eagerly:
          Stage 2: FlashInfer decode
          Stage 3a: FA3 self-attention (new prefills, positions from 0)
          Stage 3b: FlashInfer paged-KV prefill (continuations, attend to
                    all previous chunks' KV in paged cache)
        Token layout: [D decode | P new-prefill | C continuation]
        Automatically advances the offload/trace engine's step counter.
        Supports pipeline parallelism (pp_size > 1): per-GPU buffers with
        cross-GPU hidden state transfers at PP boundaries.
        """
        info = self._piecewise_graphs[graph_N]

        _nvtx = self._nvtx_enabled
        _controller = (self.trace_recorder or self.replay_controller
                        or self.offload_engine)
        _pt = (self.replay_controller._phase_timer
               if self.replay_controller and self.replay_controller._phase_timer
               else None)
        if _pt:
            _pt.step_start()
        if _controller:
            _controller.begin_step()
        if self.offload_engine and _controller is not self.offload_engine:
            self.offload_engine.begin_step()
        D = len(decode_seq_ids)
        prefill_lengths = [ids.shape[0] for ids in prefill_input_ids]
        cont_lengths = [ids.shape[0] for ids in continuation_input_ids]
        P = sum(prefill_lengths)
        C = sum(cont_lengths)
        N_actual = D + P + C
        H = self.hidden_size
        pp = self.pp_size > 1

        # Helper to get the right buffer dict for a layer
        def _buf(layer):
            if pp:
                return info['pp_bufs'][self.pp_layer_gpu[layer]]
            return info

        # ── 1. Build token_ids ──
        if _nvtx: torch.cuda.nvtx.range_push("setup")
        # Ensure all on primary device (PP: inputs may arrive on any device)
        primary_dev = self.pp_devices[0] if pp else self.device
        token_parts = []
        if D > 0:
            token_parts.append(decode_token_ids.to(primary_dev))
        for ids in prefill_input_ids:
            token_parts.append(ids.to(primary_dev))
        for ids in continuation_input_ids:
            token_parts.append(ids.to(primary_dev))
        all_token_ids = torch.cat(token_parts)

        # Copy into each GPU's static buffer
        if pp:
            for gpu_idx in range(self.pp_size):
                b = info['pp_bufs'][gpu_idx]
                b['static_token_ids'][:N_actual].copy_(all_token_ids)
                if N_actual < graph_N:
                    b['static_token_ids'][N_actual:].zero_()
        else:
            info['static_token_ids'][:N_actual].copy_(all_token_ids)
            if N_actual < graph_N:
                info['static_token_ids'][N_actual:].zero_()

        # ── 2. Compute positions ──
        # Build on CPU, then copy to each GPU
        primary_dev = self.pp_devices[0] if pp else self.device
        decode_positions = (self._seq_lens_cpu[decode_seq_ids].to(torch.int32)
                            .to(primary_dev)) if D > 0 else torch.empty(
                                0, dtype=torch.int32, device=primary_dev)
        prefill_pos_parts = [
            torch.arange(s, dtype=torch.int32, device=primary_dev)
            for s in prefill_lengths
        ]
        cont_pos_parts = [
            torch.arange(off, off + clen, dtype=torch.int32,
                          device=primary_dev)
            for off, clen in zip(continuation_offsets, cont_lengths)
        ]
        positions = torch.cat([decode_positions] + prefill_pos_parts
                              + cont_pos_parts)

        if pp:
            for gpu_idx in range(self.pp_size):
                b = info['pp_bufs'][gpu_idx]
                b['static_positions'][:N_actual].copy_(positions)
                if N_actual < graph_N:
                    b['static_positions'][N_actual:].zero_()
        else:
            info['static_positions'][:N_actual].copy_(positions)
            if N_actual < graph_N:
                info['static_positions'][N_actual:].zero_()

        # ── 3. Compute slot_mapping ──
        # Use GPU 0's block table for computation, then replicate
        bt = self.block_table[0] if pp else self.block_table
        if D > 0:
            d_idx = torch.tensor(decode_seq_ids, device=primary_dev,
                                 dtype=torch.long)
            d_page = (decode_positions // self.page_size).long()
            d_offset = (decode_positions % self.page_size).long()
            decode_slots = (bt[d_idx, d_page].long()
                            * self.page_size + d_offset)
        else:
            decode_slots = torch.empty(0, dtype=torch.long,
                                       device=primary_dev)

        prefill_slot_parts = []
        for sid, length in zip(prefill_seq_ids, prefill_lengths):
            pos = torch.arange(length, device=primary_dev)
            pg = (pos // self.page_size).long()
            off = (pos % self.page_size).long()
            prefill_slot_parts.append(
                bt[sid, pg].long() * self.page_size + off)

        cont_slot_parts = []
        for sid, offset, clen in zip(continuation_seq_ids,
                                     continuation_offsets, cont_lengths):
            pos = torch.arange(offset, offset + clen, device=primary_dev)
            pg = (pos // self.page_size).long()
            off = (pos % self.page_size).long()
            cont_slot_parts.append(
                bt[sid, pg].long() * self.page_size + off)

        slot_mapping = torch.cat([decode_slots] + prefill_slot_parts
                                 + cont_slot_parts)

        if pp:
            for gpu_idx in range(self.pp_size):
                b = info['pp_bufs'][gpu_idx]
                b['static_slot_mapping'][:N_actual].copy_(slot_mapping)
                if N_actual < graph_N:
                    b['static_slot_mapping'][N_actual:].fill_(-1)
        else:
            info['static_slot_mapping'][:N_actual].copy_(slot_mapping)
            if N_actual < graph_N:
                info['static_slot_mapping'][N_actual:].fill_(-1)

        # ── 4. Increment decode seq_lens and plan FlashInfer ──
        if D > 0:
            for sid in decode_seq_ids:
                self._seq_lens_cpu[sid] += 1
            if pp:
                for gpu_idx in range(self.pp_size):
                    self._plan_flashinfer_decode_pp(decode_seq_ids, gpu_idx)
            else:
                self._plan_flashinfer_decode_for_subset(decode_seq_ids)

        # ── 4b. Update _seq_lens_cpu for continuation sequences ──
        # Must happen before FlashInfer paged-KV plan (needs total seq_len).
        # Caller contract: offset must equal the current KV cache length for
        # that sequence (i.e. the number of tokens already prefilled).
        cont_total_seq_lens = []
        for sid, offset, clen in zip(continuation_seq_ids,
                                     continuation_offsets, cont_lengths):
            expected_sl = int(self._seq_lens_cpu[sid].item())
            assert offset == expected_sl, (
                f"Continuation offset mismatch for seq {sid}: "
                f"offset={offset} but seq_lens={expected_sl}. "
                f"Offset must equal the number of tokens already in KV cache."
            )
            total_sl = offset + clen
            self._seq_lens_cpu[sid] = total_sl
            cont_total_seq_lens.append(total_sl)

        # ── 5a. Build prefill cu_seqlens for FA3 (new prefills only) ──
        # Need one per GPU since FA3 tensors must be on correct device
        if pp:
            prefill_cu = {}
            for gpu_idx in range(self.pp_size):
                dev = self.pp_devices[gpu_idx]
                cu = torch.zeros(len(prefill_lengths) + 1, dtype=torch.int32,
                                 device=dev)
                for i, s in enumerate(prefill_lengths):
                    cu[i + 1] = cu[i] + s
                prefill_cu[gpu_idx] = cu
        else:
            prefill_cu_single = torch.zeros(len(prefill_lengths) + 1,
                                            dtype=torch.int32,
                                            device=self.device)
            for i, s in enumerate(prefill_lengths):
                prefill_cu_single[i + 1] = prefill_cu_single[i] + s
        prefill_max = max(prefill_lengths) if prefill_lengths else 0

        # ── 5b. Plan FlashInfer paged-KV for continuation prefills ──
        cont_prefill_wrappers = None
        if C > 0:
            if pp:
                cont_prefill_wrappers = {}
                for gpu_idx in range(self.pp_size):
                    dev = self.pp_devices[gpu_idx]
                    wb = self._workspace_bufs[gpu_idx]
                    pw = BatchPrefillWithPagedKVCacheWrapper(
                        wb, kv_layout="NHD")
                    # Build indptr/indices for all continuation sequences
                    qo_indptr = [0]
                    kv_indptr = [0]
                    kv_indices = []
                    last_page_lens = []
                    bt_gpu = self.block_table[gpu_idx]
                    for sid, total_sl, clen in zip(
                            continuation_seq_ids, cont_total_seq_lens,
                            cont_lengths):
                        qo_indptr.append(qo_indptr[-1] + clen)
                        num_pages = math.ceil(total_sl / self.page_size)
                        kv_indptr.append(kv_indptr[-1] + num_pages)
                        kv_indices.append(
                            bt_gpu[sid, :num_pages].to(torch.int32).to(dev))
                        lpl = total_sl % self.page_size
                        last_page_lens.append(lpl if lpl > 0 else
                                              self.page_size)
                    pw.plan(
                        qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32,
                                               device=dev),
                        paged_kv_indptr=torch.tensor(kv_indptr,
                                                     dtype=torch.int32,
                                                     device=dev),
                        paged_kv_indices=torch.cat(kv_indices),
                        paged_kv_last_page_len=torch.tensor(
                            last_page_lens, dtype=torch.int32, device=dev),
                        num_qo_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim_qk=self.head_dim,
                        page_size=self.page_size,
                        causal=True,
                        q_data_type=self.dtype,
                    )
                    cont_prefill_wrappers[gpu_idx] = pw
            else:
                wb = self._workspace_buf
                pw = BatchPrefillWithPagedKVCacheWrapper(
                    wb, kv_layout="NHD")
                qo_indptr = [0]
                kv_indptr = [0]
                kv_indices = []
                last_page_lens = []
                for sid, total_sl, clen in zip(
                        continuation_seq_ids, cont_total_seq_lens,
                        cont_lengths):
                    qo_indptr.append(qo_indptr[-1] + clen)
                    num_pages = math.ceil(total_sl / self.page_size)
                    kv_indptr.append(kv_indptr[-1] + num_pages)
                    kv_indices.append(
                        self.block_table[sid, :num_pages].to(torch.int32))
                    lpl = total_sl % self.page_size
                    last_page_lens.append(lpl if lpl > 0 else self.page_size)
                pw.plan(
                    qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32,
                                           device=self.device),
                    paged_kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32,
                                                 device=self.device),
                    paged_kv_indices=torch.cat(kv_indices),
                    paged_kv_last_page_len=torch.tensor(
                        last_page_lens, dtype=torch.int32,
                        device=self.device),
                    num_qo_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim_qk=self.head_dim,
                    page_size=self.page_size,
                    causal=True,
                    q_data_type=self.dtype,
                )
                cont_prefill_wrappers = pw  # single wrapper, not dict

        if _nvtx: torch.cuda.nvtx.range_pop()  # setup

        # ── 6. Embed tokens into hidden_buf (GPU 0) ──
        if _nvtx: torch.cuda.nvtx.range_push("embed")
        buf0 = _buf(0)
        buf0['hidden_buf'].copy_(
            F.embedding(buf0['static_token_ids'], self.embed_tokens))
        if _nvtx: torch.cuda.nvtx.range_pop()  # embed
        if _pt: _pt.after_setup()

        # ── 7. Per-layer piecewise replay ──
        for layer in range(self.num_layers):
            if _nvtx: torch.cuda.nvtx.range_push(f"layer_{layer}")
            buf = _buf(layer)
            q_buf = buf['q_buf']
            k_buf = buf['k_buf']
            v_buf = buf['v_buf']
            attn_out_buf = buf['attn_out_buf']

            # Cross-GPU transfer at PP boundaries
            if pp and layer in self.pp_boundaries:
                if _nvtx: torch.cuda.nvtx.range_push("pp_xfer")
                prev_gpu = self.pp_layer_gpu[layer - 1]
                buf['hidden_buf'].copy_(
                    info['pp_bufs'][prev_gpu]['hidden_buf'])
                if _nvtx: torch.cuda.nvtx.range_pop()  # pp_xfer

            # Pre-attention prefetch (replay controller only)
            if layer == 0 and self.replay_controller:
                self.replay_controller.begin_layer_prefetch(0)

            # Stage 1: pre-attention (CUDA graph)
            if _nvtx: torch.cuda.nvtx.range_push("stage1")
            info['stage1_graphs'][layer].replay()
            if _nvtx: torch.cuda.nvtx.range_pop()  # stage1
            if _pt: _pt.after_stage1(layer)

            # Stage 2: FlashInfer decode on q_buf[:D]
            if D > 0:
                if _nvtx: torch.cuda.nvtx.range_push("stage2")
                if pp:
                    gpu_idx = self.pp_layer_gpu[layer]
                    wrapper = self._decode_wrappers[gpu_idx]
                else:
                    wrapper = self._decode_wrapper
                q_decode = q_buf[:D]
                decode_out = wrapper.run(
                    q_decode,
                    (self.k_cache[layer], self.v_cache[layer]))
                attn_out_buf[:D].copy_(decode_out.reshape(D, H))
                if _nvtx: torch.cuda.nvtx.range_pop()  # stage2

            # Stage 3a: FA3 self-attention for new prefills on q_buf[D:D+P]
            if P > 0:
                if _nvtx: torch.cuda.nvtx.range_push("stage3a")
                q_pf = q_buf[D:D + P]
                k_pf = k_buf[D:D + P]
                v_pf = v_buf[D:D + P]
                cu = (prefill_cu[self.pp_layer_gpu[layer]]
                      if pp else prefill_cu_single)
                prefill_out = flash_attn_varlen_func(
                    q_pf, k_pf, v_pf,
                    cu_seqlens_q=cu,
                    cu_seqlens_k=cu,
                    max_seqlen_q=prefill_max,
                    max_seqlen_k=prefill_max,
                    causal=True, fa_version=3)
                attn_out_buf[D:D + P].copy_(prefill_out.reshape(P, H))
                if _nvtx: torch.cuda.nvtx.range_pop()  # stage3a

            # Stage 3b: FlashInfer paged-KV for continuation prefills
            #           on q_buf[D+P:D+P+C]
            if C > 0:
                if _nvtx: torch.cuda.nvtx.range_push("stage3b")
                q_cont = q_buf[D + P:D + P + C]
                if pp:
                    gpu_idx = self.pp_layer_gpu[layer]
                    cpw = cont_prefill_wrappers[gpu_idx]
                else:
                    cpw = cont_prefill_wrappers
                cont_out = cpw.run(
                    q_cont,
                    (self.k_cache[layer], self.v_cache[layer]))
                attn_out_buf[D + P:D + P + C].copy_(
                    cont_out.reshape(C, H))
                if _nvtx: torch.cuda.nvtx.range_pop()  # stage3b

            # Zero padding region of attn_out_buf
            if N_actual < graph_N:
                attn_out_buf[N_actual:].zero_()
            if _pt: _pt.after_attn(layer)

            # Stage 4a: router (CUDA graph)
            if _nvtx: torch.cuda.nvtx.range_push("stage4a")
            info['stage4a_graphs'][layer].replay()
            if _nvtx: torch.cuda.nvtx.range_pop()  # stage4a
            if _pt: _pt.after_stage4a(layer)

            # ── CPU break: update expert_map + load missing experts ──
            if _nvtx: torch.cuda.nvtx.range_push("cpu_break")
            if self.replay_controller:
                self.replay_controller.process_layer_replay(
                    layer, buf['topk_ids_buf'], N_actual)
            elif self.offload_engine:
                self.offload_engine.process_layer(
                    layer, buf['topk_ids_buf'], N_actual,
                    router_input_buf=buf['moe_input_buf'])
                if self.trace_recorder:
                    self.trace_recorder.process_layer(
                        layer, buf['topk_ids_buf'], N_actual,
                        router_input_buf=buf['moe_input_buf'])
            elif self.trace_recorder:
                self.trace_recorder.process_layer(
                    layer, buf['topk_ids_buf'], N_actual,
                    router_input_buf=buf['moe_input_buf'])
            if _nvtx: torch.cuda.nvtx.range_pop()  # cpu_break
            if _pt: _pt.after_io(layer)

            # Neutralize padding tokens: zero weights so fused_moe
            # multiplies by 0 and padding contributes nothing to output.
            # Trace recorder slices topk_ids_buf[:n_tokens], unaffected.
            if N_actual < graph_N:
                buf['topk_weights_buf'][N_actual:].zero_()
                # In offloading mode, also route padding to an uncached
                # expert so expert_map lookup doesn't hit -1.
                if self.offloading:
                    emap = self.expert_map_buf
                    uncached = (emap == -1).nonzero(as_tuple=True)[0]
                    if len(uncached) > 0:
                        buf['topk_ids_buf'][N_actual:].fill_(uncached[0].item())

            # Stage 4b: MoE (CUDA graph)
            if _nvtx: torch.cuda.nvtx.range_push("stage4b")
            info['stage4b_graphs'][layer].replay()
            if _nvtx: torch.cuda.nvtx.range_pop()  # stage4b
            if _pt: _pt.after_stage4b(layer)

            # Clean up after MoE computation
            if _controller:
                _controller.post_layer(layer)
            if self.offload_engine and _controller is not self.offload_engine:
                self.offload_engine.post_layer(layer)
            if _nvtx: torch.cuda.nvtx.range_pop()  # layer_N

        # ── 8. Final norm + lm_head (on last GPU for PP) ──
        if _nvtx: torch.cuda.nvtx.range_push("final")
        last_buf = _buf(self.num_layers - 1)
        hidden = last_buf['hidden_buf']
        hidden = F.rms_norm(hidden, (self.hidden_size,), self.final_norm,
                            self.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head)

        # Transfer logits back to primary device for consistent API
        if pp:
            logits = logits.to(self.pp_devices[0])
        if _nvtx: torch.cuda.nvtx.range_pop()  # final
        if _pt: _pt.after_final()

        # ── 9. Update GPU seq_lens ──
        if pp:
            for sl in self.seq_lens:
                for sid in decode_seq_ids:
                    sl[sid] += 1
                for sid, length in zip(prefill_seq_ids, prefill_lengths):
                    sl[sid] = length
                for sid, total_sl in zip(continuation_seq_ids,
                                         cont_total_seq_lens):
                    sl[sid] = total_sl
        else:
            for sid in decode_seq_ids:
                self.seq_lens[sid] += 1
            for sid, length in zip(prefill_seq_ids, prefill_lengths):
                self.seq_lens[sid] = length
            for sid, total_sl in zip(continuation_seq_ids,
                                     cont_total_seq_lens):
                self.seq_lens[sid] = total_sl
        for sid, length in zip(prefill_seq_ids, prefill_lengths):
            self._seq_lens_cpu[sid] = length
        # _seq_lens_cpu for continuations already set in step 4b

        return logits[:N_actual]

    # ── Generation ───────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128) -> torch.Tensor:
        """Greedy autoregressive generation.
        Args: input_ids [B, S]
        Returns: all_tokens [B, S + generated]
        """
        B, S = input_ids.shape
        if self.pp_size > 1:
            for sl in self.seq_lens:
                sl[:B] = 0
        else:
            self.seq_lens[:B] = 0
        self._seq_lens_cpu[:B] = 0

        logits = self.prefill(input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            seq_lens_ref = self.seq_lens[0] if self.pp_size > 1 else self.seq_lens
            positions = seq_lens_ref[:B].clone()
            logits = self.decode_step(next_token, positions)
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)
            if (next_token == self.eos_token_id).all():
                break

        return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1)

    def reset(self):
        """Reset KV cache and sequence state."""
        if self.pp_size > 1:
            for sl in self.seq_lens:
                sl.zero_()
        else:
            self.seq_lens.zero_()
        self._seq_lens_cpu.zero_()
        for l in range(self.num_layers):
            self.k_cache[l].zero_()
            self.v_cache[l].zero_()
        if self._dynamic_pages:
            self._free_pages = deque(range(self.total_pages))
            self._seq_page_list = [[] for _ in range(self.max_seqs)]
            if self.pp_size > 1:
                for bt in self.block_table:
                    bt.fill_(-1)
            else:
                self.block_table.fill_(-1)

    def free_seq(self, seq_id: int):
        """Free a single sequence's KV cache.

        Resets seq_len to 0. With dynamic page allocation, also returns
        physical pages to the free pool.
        """
        self._seq_lens_cpu[seq_id] = 0
        if self.pp_size > 1:
            for sl in self.seq_lens:
                sl[seq_id] = 0
        else:
            self.seq_lens[seq_id] = 0
        if self._dynamic_pages:
            self.free_seq_pages(seq_id)

    # ── Dynamic Page Allocation ───────────────────────────────────────

    def alloc_pages(self, seq_id: int, n_pages: int):
        """Allocate n_pages from the free pool for seq_id.

        Assigns physical pages to block_table[seq_id] starting at the
        current allocation watermark. Raises RuntimeError if the free
        pool doesn't have enough pages.
        """
        if n_pages <= 0:
            return
        current = len(self._seq_page_list[seq_id])
        if current + n_pages > self.max_pages_per_seq:
            raise RuntimeError(
                f"Cannot allocate {n_pages} pages for seq {seq_id}: "
                f"would exceed max_pages_per_seq={self.max_pages_per_seq} "
                f"(current={current})")
        if len(self._free_pages) < n_pages:
            raise RuntimeError(
                f"Cannot allocate {n_pages} pages for seq {seq_id}: "
                f"only {len(self._free_pages)} free pages remain "
                f"(budget={self.total_pages})")
        pages = [self._free_pages.popleft() for _ in range(n_pages)]
        self._seq_page_list[seq_id].extend(pages)
        page_t = torch.tensor(pages, dtype=torch.int32)
        if self.pp_size > 1:
            for bt in self.block_table:
                bt[seq_id, current:current + n_pages] = page_t.to(bt.device)
        else:
            self.block_table[seq_id, current:current + n_pages] = (
                page_t.to(self.block_table.device))

    def free_seq_pages(self, seq_id: int):
        """Return all pages for seq_id to the free pool."""
        pages = self._seq_page_list[seq_id]
        if not pages:
            return
        n = len(pages)
        self._free_pages.extend(pages)
        self._seq_page_list[seq_id] = []
        if self.pp_size > 1:
            for bt in self.block_table:
                bt[seq_id, :n] = -1
        else:
            self.block_table[seq_id, :n] = -1

    def ensure_pages(self, seq_id: int, needed_pages: int):
        """Ensure seq_id has at least needed_pages allocated.

        Called by the scheduler before mixed_step() for every sequence
        that will grow in this step. No-op when pages are sufficient.
        """
        current = len(self._seq_page_list[seq_id])
        if needed_pages > current:
            self.alloc_pages(seq_id, needed_pages - current)

    @property
    def pages_free(self) -> int:
        """Number of unallocated pages (dynamic mode only)."""
        return len(self._free_pages)

    @property
    def pages_in_use(self) -> int:
        """Number of allocated pages (dynamic mode only)."""
        return self.total_pages - len(self._free_pages)

    def seq_pages(self, seq_id: int) -> int:
        """Number of pages currently allocated for seq_id."""
        return len(self._seq_page_list[seq_id])

    # ── CUDA Graph Support ────────────────────────────────────────────

    def _full_decode_graph_body(self, token_ids, positions, slot_mapping, decode_wrapper):
        """Full decode forward — all layers + final norm + lm_head.

        Designed for torch.compile + CUDA graph capture. Dynamo unrolls the
        layer loop (num_layers is constant) enabling cross-layer fusion of
        residual_add + rms_norm into single Triton kernels.
        """
        hidden = F.embedding(token_ids, self.embed_tokens)
        for layer in range(self.num_layers):
            hidden = self._layer_decode(hidden, layer, positions,
                                        slot_mapping, decode_wrapper)
        hidden = F.rms_norm(hidden, (self.hidden_size,), self.final_norm, self.rms_norm_eps)
        return F.linear(hidden, self.lm_head)

    def capture_decode_cuda_graph(self, batch_size, warmup_seq_len=128,
                                    max_decode_tokens=256, use_torch_compile=None):
        """Capture CUDA graph for decode step with optional torch.compile fusion.

        When use_torch_compile=True, Inductor fuses RMSNorm + residual add + RoPE
        into Triton kernels before CUDA graph capture, matching vLLM v1 performance.

        Args:
            batch_size: Fixed batch size for this graph
            warmup_seq_len: Prefill length for warmup (populates KV cache)
            max_decode_tokens: Maximum additional tokens to generate after prefill.
            use_torch_compile: Override instance default (self.use_torch_compile)
        """
        if use_torch_compile is None:
            use_torch_compile = self.use_torch_compile

        if not hasattr(self, '_cuda_graphs'):
            self._cuda_graphs = {}

        # Dummy prefill to populate KV cache (direct call, no graph needed)
        self.reset()
        dummy = torch.randint(1, 1000, (warmup_seq_len,), device=self.device)
        with torch.no_grad():
            for b in range(batch_size):
                # Use mixed_step for eager prefill (no CUDA graph dependency)
                self.mixed_step(
                    [], torch.empty(0, dtype=torch.long, device=self.device),
                    [b], [dummy])

        # Static input buffers (updated via copy_ before each replay)
        static_token = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        static_pos = self.seq_lens[:batch_size].clone()
        static_slot_mapping = self._compute_slot_mapping(static_pos, batch_size)

        # FlashInfer CUDA graph wrapper with pre-allocated buffers
        max_total_pages = batch_size * self.max_pages_per_seq
        fi_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        fi_indices = torch.zeros(max_total_pages, dtype=torch.int32, device=self.device)
        fi_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=self.device)

        # Pre-fill indices (contiguous page allocation)
        idx = 0
        for b in range(batch_size):
            for p in range(self.max_pages_per_seq):
                fi_indices[idx] = b * self.max_pages_per_seq + p
                idx += 1

        graph_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self._workspace_buf, kv_layout="NHD", use_cuda_graph=True,
            paged_kv_indptr_buffer=fi_indptr,
            paged_kv_indices_buffer=fi_indices,
            paged_kv_last_page_len_buffer=fi_last_page_len)

        # Increment _seq_lens_cpu to account for current token (written to cache
        # before attention within each layer, so FlashInfer must include it).
        self._seq_lens_cpu[:batch_size] += 1

        # Initial plan() — JIT compiles the FlashInfer kernel
        self._plan_flashinfer_decode(graph_wrapper, batch_size)

        # Choose forward function (optionally compiled)
        forward_fn = self._full_decode_graph_body
        if use_torch_compile:
            forward_fn = torch.compile(forward_fn, fullgraph=False)

        # Warmup — triggers torch.compile JIT and stabilizes CUDA allocator
        n_warmup = 5 if use_torch_compile else 3
        for _ in range(n_warmup):
            # Re-plan before each warmup step (seq_lens may change)
            self._plan_flashinfer_decode(graph_wrapper, batch_size)
            static_output = forward_fn(
                static_token, static_pos, static_slot_mapping, graph_wrapper)
        torch.cuda.synchronize()

        # Plan once more before capture
        self._plan_flashinfer_decode(graph_wrapper, batch_size)

        # Capture CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = forward_fn(
                static_token, static_pos, static_slot_mapping, graph_wrapper)

        self._cuda_graphs[batch_size] = {
            'graph': graph,
            'graph_wrapper': graph_wrapper,
            'static_token': static_token,
            'static_pos': static_pos,
            'static_slot_mapping': static_slot_mapping,
            'static_output': static_output,
        }
        compile_str = " + torch.compile" if use_torch_compile else ""
        print(f"  CUDA graph{compile_str} captured for batch_size={batch_size} "
              f"(max_seq={warmup_seq_len + max_decode_tokens})")

    def _decode_step_graphed(self, token_ids, positions):
        """Decode using pre-captured CUDA graph.

        Updates static buffers and re-plans FlashInfer (recomputes _plan_info
        for current seq_lens — needed because split-KV tile counts depend on
        page count), then replays.
        """
        B = token_ids.shape[0]
        info = self._cuda_graphs[B]
        pos_i32 = positions.to(torch.int32)

        # Copy inputs into static buffers
        info['static_token'].copy_(token_ids)
        info['static_pos'].copy_(pos_i32)
        info['static_slot_mapping'].copy_(self._compute_slot_mapping(pos_i32, B))

        # Increment seq_lens BEFORE plan so FlashInfer includes the current
        # token's K/V (written to cache before attention within each layer).
        self._seq_lens_cpu[:B] += 1

        # Re-plan FlashInfer with updated seq_lens (plan_info must match page count)
        wrapper = info['graph_wrapper']
        seq_lens = self._seq_lens_cpu[:B]
        pages_per_seq = (seq_lens + self.page_size - 1) // self.page_size
        indptr = torch.zeros(B + 1, dtype=torch.int32)
        indptr[1:] = pages_per_seq.cumsum(0)
        last_page_len = (seq_lens - 1) % self.page_size + 1
        total_pages = indptr[-1].item()
        indices = wrapper._paged_kv_indices_buf[:total_pages]

        wrapper.plan(
            indptr, indices, last_page_len,
            num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, page_size=self.page_size,
            pos_encoding_mode="NONE", q_data_type=self.dtype)

        # Replay
        info['graph'].replay()

        self.seq_lens[:B] += 1
        return info['static_output']


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "src/MoE/models/OLMoE-1B-7B"
    engine = MoEEngine(model_path, max_seqs=4, max_seq_len=512,
                       use_torch_compile=False)

    # Capture prefill graphs (required before any prefill call)
    engine.capture_prefill_cuda_graph(
        total_token_sizes=[4, 128, 256, 512],
        use_torch_compile=False)

    # Quick smoke test
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    print("\nSmoke test: prefill...")
    logits = engine.prefill(input_ids)
    print(f"  Prefill logits shape: {logits.shape}")
    print(f"  Top token: {logits[0, -1].argmax().item()}")

    print("Smoke test: generate...")
    engine.reset()
    engine.capture_decode_cuda_graph(batch_size=1, warmup_seq_len=4,
                                     max_decode_tokens=20,
                                     use_torch_compile=False)
    engine.reset()
    tokens = engine.generate(input_ids, max_new_tokens=10)
    print(f"  Generated shape: {tokens.shape}")
    print(f"  Tokens: {tokens[0].tolist()}")
