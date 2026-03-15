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
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open
from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens
from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel
from vllm.v1.attention.backends.flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
# Triton kernel that flattens paged block table into a contiguous index array
# entirely on GPU, avoiding the GPU→CPU→GPU roundtrip of the naive approach
from vllm.v1.attention.backends.flashinfer import _copy_page_indices_kernel
from vllm.vllm_flash_attn import flash_attn_varlen_func

# FlashMLA: fused MLA decode kernel (no plan() needed), used for MLA models
try:
    from vllm.third_party.flashmla.flash_mla_interface import (
        flash_mla_with_kvcache, get_mla_metadata, FlashMLASchedMeta
    )
    _HAS_FLASHMLA = True
except ImportError:
    _HAS_FLASHMLA = False


def _yarn_get_mscale(scale: float, mscale_coeff: float) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale_coeff * math.log(scale) + 1.0


def _yarn_find_correction_dim(beta, rope_dim, theta, max_pos):
    return (rope_dim * math.log(max_pos / (beta * 2 * math.pi))) / (
        2 * math.log(theta))



# ── Load vLLM custom ops (_C.so has reshape_and_cache_flash, etc.) ──
import vllm._custom_ops as _vllm_ops

_vllm_so = Path(_vllm_ops.__file__).parent / "_C.abi3.so"
if _vllm_so.exists():
    torch.ops.load_library(str(_vllm_so))

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from ep_utils import determine_expert_map, local_expert_ids as _local_expert_ids
from ep_utils import ep_allgather, ep_reducescatter

# Patch moe_align_block_size to remove the 1024-expert-slot limit.
# This replaces the CUDA kernel with our version that uses a serial scan
# fallback for >1024 experts, enabling unified caches larger than 992 slots.
# The CUDA extension is JIT-compiled on first import (cached in cuda/build/).
from cuda.patch_moe_align import patch_moe_align_block_size
patch_moe_align_block_size()


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
        expert_parallel_size: int = 1,
        expert_placement_strategy: str = "linear",
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

        # Expert parallelism
        self.ep_size = expert_parallel_size
        self.ep_strategy = expert_placement_strategy
        if self.ep_size > 1:
            if not dist.is_initialized():
                raise RuntimeError(
                    "EP requires torch.distributed initialized before "
                    "creating MoEEngine. Launch with torchrun --nproc_per_node=EP.")
            self.ep_group = dist.new_group(ranks=list(range(self.ep_size)))
            assert dist.get_world_size() == self.ep_size, (
                f"EP currently requires world_size == ep_size, got "
                f"{dist.get_world_size()} vs {self.ep_size}")
            self.ep_rank = dist.get_rank(self.ep_group)
            torch.cuda.set_device(self.ep_rank)
            self.device = torch.device('cuda', self.ep_rank)
        else:
            self.ep_rank = 0
            self.ep_group = None

        # Load config
        with open(Path(model_path) / "config.json") as f:
            cfg = json.load(f)
        self.num_layers = cfg["num_hidden_layers"]
        self.hidden_size = cfg["hidden_size"]
        self.intermediate_size = cfg["intermediate_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        # Expert count: OLMoE uses "num_experts", Mixtral uses "num_local_experts",
        # DeepSeek-V2 uses "n_routed_experts"
        self.num_experts = (cfg.get("n_routed_experts")
                            or cfg.get("num_local_experts")
                            or cfg.get("num_experts"))
        self.top_k = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_topk")
        self.rope_theta = cfg.get("rope_theta", 10000.0)
        self.vocab_size = cfg["vocab_size"]
        self.eos_token_id = cfg.get("eos_token_id", 2)
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)
        # OLMoE has explicit "norm_topk_prob" in config. For Mixtral and other
        # models that lack this field, vLLM defaults renormalize=True in FusedMoE.
        # Match that behavior: renormalize unless explicitly set to False.
        self.norm_topk_prob = cfg.get("norm_topk_prob", True)

        # MLA (Multi-head Latent Attention) detection
        self.is_mla = "kv_lora_rank" in cfg

        if self.is_mla:
            self.kv_lora_rank = cfg["kv_lora_rank"]                # 512
            self.qk_nope_head_dim = cfg["qk_nope_head_dim"]        # 128
            self.qk_rope_head_dim = cfg["qk_rope_head_dim"]        # 64
            self.v_head_dim = cfg["v_head_dim"]                     # 128
            self.q_lora_rank = cfg.get("q_lora_rank")               # None for V2-Lite
            self.first_k_dense_replace = cfg.get("first_k_dense_replace", 0)  # 1
            self.n_shared_experts = cfg.get("n_shared_experts", 0)  # 2
            self.moe_intermediate_size = cfg.get("moe_intermediate_size",
                                                  self.intermediate_size)  # 1408
            self.rope_scaling_config = cfg.get("rope_scaling")       # YaRN config dict
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 192
            self.routed_scaling_factor = cfg.get("routed_scaling_factor", 1.0)

            # Grouped top-k routing (DS-V2: n_group=8, topk_group=3)
            self.n_group = cfg.get("n_group", 1)
            self.topk_group = cfg.get("topk_group", 1)

            # Forward-compat guard: reject DS-V3+ features we don't support
            if cfg.get("scoring_func", "softmax") != "softmax":
                raise NotImplementedError(
                    f"scoring_func={cfg['scoring_func']!r} not supported (only 'softmax').")
            if cfg.get("e_score_correction_bias") is not None:
                raise NotImplementedError(
                    "e_score_correction_bias (DS-V3) not supported.")
            if cfg.get("num_nextn_predict_layers", 0) > 0:
                raise NotImplementedError(
                    "Multi-token prediction (num_nextn_predict_layers > 0) not supported.")

            # Pre-compute sm_scale with mscale² correction (vLLM-verified formula)
            mscale_all_dim_coeff = float(self.rope_scaling_config.get("mscale_all_dim", 0))
            factor = self.rope_scaling_config["factor"]
            mscale_val = _yarn_get_mscale(factor, mscale_all_dim_coeff)
            self._mla_sm_scale = (self.qk_head_dim ** -0.5) * mscale_val * mscale_val
        else:
            self.first_k_dense_replace = 0
            self.n_shared_experts = 0
            self.n_group = 1
            self.topk_group = 1
            self.moe_intermediate_size = self.intermediate_size
            self.rope_scaling_config = None

        # FlashMLA kernel requires page_size=64 (its tiling assumes 64-slot pages)
        if self.is_mla:
            self.page_size = 64
            page_size = 64

        # Unsupported features — raise early rather than produce wrong results
        if cfg.get("sliding_window") is not None:
            raise NotImplementedError(
                f"Sliding window attention (window={cfg['sliding_window']}) is not "
                f"yet implemented.")
        if cfg.get("rope_scaling") is not None and not self.is_mla:
            raise NotImplementedError(
                f"RoPE scaling ({cfg['rope_scaling']}) is only implemented for MLA models.")

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

        # EP validation (after PP setup, before weight loading)
        if self.ep_size > 1:
            if self.pp_size > 1:
                raise ValueError("EP + PP is not yet supported.")
            if self.offloading:
                raise ValueError(
                    "EP + offloading is not yet supported. "
                    "Use ep_size=1 with offloading, or ep_size>1 without.")

        # EP expert map (needs self.num_experts from config)
        if self.ep_size > 1:
            self.local_num_experts, self.ep_expert_map = determine_expert_map(
                self.ep_size, self.ep_rank, self.num_experts, self.ep_strategy)
            self.local_expert_ids = _local_expert_ids(
                self.ep_size, self.ep_rank, self.num_experts, self.ep_strategy)
            print(f"EP rank {self.ep_rank}/{self.ep_size}: "
                  f"{self.local_num_experts} local experts "
                  f"{self.local_expert_ids} (strategy={self.ep_strategy})")
        else:
            self.local_num_experts = None
            self.ep_expert_map = None

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
            build_fn = self._build_yarn_rope_cache if self.is_mla else self._build_rope_cache
            self.cos_sin_cache = [build_fn().to(dtype=self.dtype, device=d)
                                  for d in self.pp_devices]
            self._k_scale = [torch.tensor(1.0, dtype=torch.float32, device=d)
                             for d in self.pp_devices]
            self._v_scale = [torch.tensor(1.0, dtype=torch.float32, device=d)
                             for d in self.pp_devices]

            # KV cache: [total_pages, page_size, num_kv_heads, head_dim] per layer.
            # Layout required by reshape_and_cache_flash.
            if self.is_mla:
                # Fused MLA cache: single mla_flat [slots, 576] buffer per layer.
                # mla_cache is a 4D view [pages, 64, 1, 576] for FlashMLA.
                # ckv_flat/kpe_flat/ckv_cache/kpe_cache are sliced views into the
                # same storage so existing KV write paths (reshape_and_cache) and
                # continuation prefill (FlashInfer MLA) still work without copies.
                total_kv_slots = self.total_pages * self.page_size
                self._scratch_slot = total_kv_slots  # index for sentinel writes
                mla_head_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576
                self.mla_flat = [
                    torch.zeros(total_kv_slots + 1, mla_head_dim, dtype=dtype,
                                device=self.pp_layer_device[l])
                    for l in range(self.num_layers)
                ]
                self.mla_cache = [
                    f[:total_kv_slots].view(
                        self.total_pages, self.page_size, 1, mla_head_dim)
                    for f in self.mla_flat
                ]
                self.ckv_flat = [f[:, :self.kv_lora_rank] for f in self.mla_flat]
                self.kpe_flat = [f[:, self.kv_lora_rank:] for f in self.mla_flat]
                self.ckv_cache = [
                    c[:, :, :, :self.kv_lora_rank].squeeze(2)
                    for c in self.mla_cache
                ]
                self.kpe_cache = [
                    c[:, :, :, self.kv_lora_rank:].squeeze(2)
                    for c in self.mla_cache
                ]
                self.k_cache = None
                self.v_cache = None
            else:
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
                self.ckv_flat = None
                self.kpe_flat = None

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

            # GPU mirror of _seq_lens_cpu for Triton kernels (on GPU 0)
            self._num_computed_tokens_gpu = torch.zeros(
                max_seqs, dtype=torch.int32, device=self.pp_devices[0])

            # Block table pointer/stride for Triton slot mapping kernel
            # Only GPU 0's pointer — kernel runs on GPU 0, results replicated
            self._block_table_ptrs = torch.tensor(
                [self.block_table[0].data_ptr()], dtype=torch.uint64,
                device=self.pp_devices[0])
            self._block_table_strides = torch.tensor(
                [self.block_table[0].stride(0)], dtype=torch.int64,
                device=self.pp_devices[0])
            self._block_sizes_tensor = torch.tensor(
                [self.page_size], dtype=torch.int32,
                device=self.pp_devices[0])

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
            if self.is_mla:
                from flashinfer.mla import BatchMLAPagedAttentionWrapper as MLAWrapper
                self._mla_wrappers = [
                    MLAWrapper(wb, use_cuda_graph=False)
                    for wb in self._workspace_bufs
                ]
                self._mla_decode_wrapper = self._mla_wrappers[0]  # alias for prefill graph capture
                self._decode_wrapper = None
                self._decode_wrappers = None
            else:
                self._decode_wrappers = [
                    BatchDecodeWithPagedKVCacheWrapper(
                        wb, kv_layout="NHD", use_cuda_graph=False)
                    for wb in self._workspace_bufs
                ]
                # Alias for prefill graph capture (always uses GPU 0)
                self._decode_wrapper = self._decode_wrappers[0]
                self._mla_decode_wrapper = None
                self._mla_wrappers = None

                # Pre-allocated plan metadata for GQA decode (PP).
                # See single-GPU path below for rationale.
                self._plan_indptr_cpu = torch.zeros(
                    max_seqs + 1, dtype=torch.int32).pin_memory()
                self._plan_indptr_np = self._plan_indptr_cpu.numpy()
                self._plan_lpl_cpu = torch.zeros(
                    max_seqs, dtype=torch.int32).pin_memory()
                self._plan_lpl_np = self._plan_lpl_cpu.numpy()
                self._plan_indptr_gpu_pp = [
                    torch.zeros(max_seqs + 1, dtype=torch.int32, device=d)
                    for d in self.pp_devices]
                self._plan_indices_gpu_pp = [
                    torch.zeros(max_seqs * self.max_pages_per_seq,
                                dtype=torch.int32, device=d)
                    for d in self.pp_devices]
                self._plan_gathered_bt_pp = [
                    torch.zeros(max_seqs, self.max_pages_per_seq,
                                dtype=torch.int32, device=d)
                    for d in self.pp_devices]
                self._plan_idx_long_pp = [
                    torch.zeros(max_seqs, dtype=torch.int64, device=d)
                    for d in self.pp_devices]
        else:
            # Single-GPU path (original)
            if self.is_mla:
                self.cos_sin_cache = self._build_yarn_rope_cache().to(dtype=self.dtype, device=device)
            else:
                self.cos_sin_cache = self._build_rope_cache().to(dtype=self.dtype, device=device)
            self._k_scale = torch.tensor(1.0, dtype=torch.float32,
                                         device=device)
            self._v_scale = torch.tensor(1.0, dtype=torch.float32,
                                         device=device)

            # KV cache: [total_pages, page_size, num_kv_heads, head_dim] per layer.
            # Layout required by reshape_and_cache_flash.
            if self.is_mla:
                # Fused MLA cache layout — see PP path above for rationale
                total_kv_slots = self.total_pages * self.page_size
                self._scratch_slot = total_kv_slots  # index for sentinel writes
                mla_head_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576
                self.mla_flat = [
                    torch.zeros(total_kv_slots + 1, mla_head_dim, dtype=dtype,
                                device=device)
                    for _ in range(self.num_layers)
                ]
                self.mla_cache = [
                    f[:total_kv_slots].view(
                        self.total_pages, self.page_size, 1, mla_head_dim)
                    for f in self.mla_flat
                ]
                self.ckv_flat = [f[:, :self.kv_lora_rank] for f in self.mla_flat]
                self.kpe_flat = [f[:, self.kv_lora_rank:] for f in self.mla_flat]
                self.ckv_cache = [
                    c[:, :, :, :self.kv_lora_rank].squeeze(2)
                    for c in self.mla_cache
                ]
                self.kpe_cache = [
                    c[:, :, :, self.kv_lora_rank:].squeeze(2)
                    for c in self.mla_cache
                ]
                self.k_cache = None
                self.v_cache = None
            else:
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
                self.ckv_flat = None
                self.kpe_flat = None

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

            # GPU mirror of _seq_lens_cpu for Triton kernels
            self._num_computed_tokens_gpu = torch.zeros(
                max_seqs, dtype=torch.int32, device=device)

            # Block table pointer/stride for Triton slot mapping kernel
            self._block_table_ptrs = torch.tensor(
                [self.block_table.data_ptr()], dtype=torch.uint64,
                device=device)
            self._block_table_strides = torch.tensor(
                [self.block_table.stride(0)], dtype=torch.int64,
                device=device)
            self._block_sizes_tensor = torch.tensor(
                [self.page_size], dtype=torch.int32, device=device)

            # Sequence length tracker
            self.seq_lens = torch.zeros(max_seqs, dtype=torch.int32,
                                        device=device)
            self._seq_lens_cpu = torch.zeros(max_seqs,
                                             dtype=torch.int32)

            # FlashInfer workspace (128 MB)
            self._workspace_buf = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device)
            if self.is_mla:
                from flashinfer.mla import BatchMLAPagedAttentionWrapper as MLAWrapper
                self._mla_decode_wrapper = MLAWrapper(
                    self._workspace_buf, use_cuda_graph=False)
                self._mla_wrappers = None
                self._decode_wrapper = None
            else:
                self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    self._workspace_buf, kv_layout="NHD", use_cuda_graph=False)
                self._mla_decode_wrapper = None
                self._mla_wrappers = None

                # Pre-allocated plan metadata for GQA decode.
                # Pinned CPU buffers with numpy views avoid per-step
                # torch.tensor()/torch.zeros() allocation overhead (~30us).
                # GPU buffers avoid the old GPU→CPU→GPU block table roundtrip
                # by using _copy_page_indices_kernel to flatten pages on-device.
                self._plan_indptr_cpu = torch.zeros(
                    max_seqs + 1, dtype=torch.int32).pin_memory()
                self._plan_indptr_np = self._plan_indptr_cpu.numpy()
                self._plan_lpl_cpu = torch.zeros(
                    max_seqs, dtype=torch.int32).pin_memory()
                self._plan_lpl_np = self._plan_lpl_cpu.numpy()
                self._plan_indptr_gpu = torch.zeros(
                    max_seqs + 1, dtype=torch.int32, device=device)
                self._plan_indices_gpu = torch.zeros(
                    max_seqs * self.max_pages_per_seq, dtype=torch.int32,
                    device=device)
                self._plan_gathered_bt = torch.zeros(
                    max_seqs, self.max_pages_per_seq, dtype=torch.int32,
                    device=device)
                self._plan_idx_long = torch.zeros(
                    max_seqs, dtype=torch.int64, device=device)

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

        # EP dispatcher setup
        if self.ep_size > 1:
            self._ep_expert_map_gpu = self.ep_expert_map.to(self.device)
        else:
            self._ep_expert_map_gpu = None

        model_type = cfg.get("model_type", "unknown")
        if self.experts_per_layer is not None:
            budget_str = f", experts_per_layer={self.experts_per_layer}"
        elif self.cache_size is not None:
            budget_str = f", cache_size={self.cache_size}"
        elif self.pp_size > 1:
            budget_str = f", PP={self.pp_size}"
        elif self.ep_size > 1:
            budget_str = f", EP={self.ep_size}"
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
            w1: [E_buf, 2*I, H] gate+up fused weights (E_buf = buffer size)
            w2: [E_buf, H, I] down projection weights
            topk_weights: [N, top_k] routing weights
            topk_ids: [N, top_k] expert indices (global IDs, before remap)
            expert_map: [num_experts] int32 mapping global_expert_id ->
                buffer slot in [0, E_buf) or -1 (skip).  None when all
                experts are in w1/w2 with identity indexing.

        global_num_experts semantics (vLLM 0.17.1):
            fused_experts_impl calls moe_align_block_size which allocates
            internal arrays of size global_num_experts.  expert_map remaps
            global IDs to buffer slot indices.  global_num_experts must be
            >= the largest valid slot index + 1, otherwise the kernel goes
            OOB.  We use max(num_experts, w1.size(0)):
              - Offloading: slots reach cache_size-1, w1.size(0)=cache_size
              - EP: slots are in [0, local_E), but num_experts > local_E
            Using num_experts in EP also preserves moe_align_block_size
            binning structure to match EP=1 accumulation order.
        """
        kwargs = {}
        if expert_map is not None:
            kwargs['expert_map'] = expert_map
            # Must be >= max valid slot index + 1 to avoid kernel OOB.
            # EP: slots are in [0, local_E), but num_experts > local_E.
            # Offloading: slots are in [0, cache_size), cache_size > num_experts.
            # max() covers both.
            kwargs['global_num_experts'] = max(self.num_experts, w1.size(0))
        return fused_experts(
            hidden_states=hidden_states, w1=w1, w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            **kwargs)

    def _route(self, hidden, layer):
        """Compute top-k routing weights and expert IDs for a layer.

        Returns (topk_weights: [N, K] float32, topk_ids: [N, K] int64).
        Handles standard top-k (OLMoE, Mixtral, DS-V2-Lite) and grouped
        top-k (DS-V2) via vLLM's fused grouped_topk CUDA kernel.
        """
        router_logits = F.linear(hidden, self.router[layer])
        if self.n_group > 1:
            # vLLM fused CUDA kernel (ops.grouped_topk): no Inductor
            # involvement, deterministic C++ dispatch. Opaque to Dynamo
            # so both collection and replay engines produce identical results.
            scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_w, topk_ids = ops.grouped_topk(
                scores,
                self.n_group,
                self.topk_group,
                self.top_k,
                self.norm_topk_prob,       # renormalize
                self.routed_scaling_factor,
                self._zero_routing_bias,   # zero bias (no correction)
                0,                         # scoring_func=0: scores pre-computed
            )
            return topk_w, topk_ids.to(torch.int64)
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_w, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        elif self.is_mla:
            topk_w = topk_w * self.routed_scaling_factor
        return topk_w, topk_ids

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
        load_device = ("cpu" if (offloading or self.pp_size > 1
                                  or self.ep_size > 1)
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
        self.o_proj = []
        self.router = []
        if self.is_mla:
            self.kv_a_proj = []        # [576, 2048] per layer
            self.kv_a_layernorm = []   # [512] per layer
            self.kv_b_proj = []        # [4096, 512] per layer
            self.W_UK_T = []           # [num_heads, qk_nope_head_dim, kv_lora_rank] per layer
            self.W_UV = []             # [num_heads, kv_lora_rank, v_head_dim] per layer
            if self.q_lora_rank is not None:
                # DS-V2: Q compression via q_a_proj -> layernorm -> q_b_proj
                self.q_a_proj = []
                self.q_a_layernorm = []
                self.q_b_proj = []
            else:
                # DS-V2-Lite: direct q_proj
                self.q_proj = []
            self.k_proj = None  # signal to skip in QKV fusion
            self.v_proj = None
        else:
            self.q_proj = []
            self.k_proj = []
            self.v_proj = []
            self.q_norm = []
            self.k_norm = []

        # Detect Q/K norm from first layer's weights
        self.has_qk_norm = f"model.layers.0.self_attn.q_norm.weight" in weights

        # Expert weight storage depends on mode
        if offloading:
            I = self.moe_intermediate_size
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

        if self.is_mla:
            self.dense_w1 = []  # [2*dense_I, H] per dense layer (fused gate+up)
            self.dense_w2 = []  # [H, dense_I] per dense layer
            self.shared_w1 = [] # [2*shared_total_I, H] per MoE layer (fused, both shared experts)
            self.shared_w2 = [] # [H, shared_total_I] per MoE layer

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
            self.o_proj.append(
                weights.pop(f"{p}.self_attn.o_proj.weight").to(layer_dev))

            if self.is_mla:
                # MLA attention projections — Q compression branch
                if self.q_lora_rank is not None:
                    self.q_a_proj.append(
                        weights.pop(f"{p}.self_attn.q_a_proj.weight").to(layer_dev))
                    self.q_a_layernorm.append(
                        weights.pop(f"{p}.self_attn.q_a_layernorm.weight").to(layer_dev))
                    self.q_b_proj.append(
                        weights.pop(f"{p}.self_attn.q_b_proj.weight").to(layer_dev))
                else:
                    self.q_proj.append(
                        weights.pop(f"{p}.self_attn.q_proj.weight").to(layer_dev))
                self.kv_a_proj.append(
                    weights.pop(f"{p}.self_attn.kv_a_proj_with_mqa.weight").to(layer_dev))
                self.kv_a_layernorm.append(
                    weights.pop(f"{p}.self_attn.kv_a_layernorm.weight").to(layer_dev))
                kv_b = weights.pop(f"{p}.self_attn.kv_b_proj.weight").to(layer_dev)
                self.kv_b_proj.append(kv_b)

                # Weight absorption: extract W_UK and W_UV from kv_b_proj
                H_mla = self.num_heads
                P = self.qk_nope_head_dim  # 128
                V = self.v_head_dim         # 128
                C = self.kv_lora_rank       # 512
                kv_b_reshaped = kv_b.view(H_mla, P + V, C)  # [16, 256, 512]
                W_UK = kv_b_reshaped[:, :P, :]               # [16, 128, 512]
                W_UV_raw = kv_b_reshaped[:, P:, :]           # [16, 128, 512]
                self.W_UK_T.append(W_UK.contiguous())  # [H, P=128, C=512]
                self.W_UV.append(W_UV_raw.transpose(1, 2).contiguous())  # [H, C=512, V=128]
            else:
                # Standard QKV — UNCHANGED
                self.q_proj.append(
                    weights.pop(f"{p}.self_attn.q_proj.weight").to(layer_dev))
                self.k_proj.append(
                    weights.pop(f"{p}.self_attn.k_proj.weight").to(layer_dev))
                self.v_proj.append(
                    weights.pop(f"{p}.self_attn.v_proj.weight").to(layer_dev))
                if self.has_qk_norm:
                    self.q_norm.append(
                        weights.pop(f"{p}.self_attn.q_norm.weight").to(layer_dev))
                    self.k_norm.append(
                        weights.pop(f"{p}.self_attn.k_norm.weight").to(layer_dev))

            # MLP / expert loading: 3-way branch
            is_dense = self.is_mla and l < self.first_k_dense_replace

            if is_dense:
                # Dense layer: single SwiGLU MLP, no router, no experts
                gate = weights.pop(f"{p}.mlp.gate_proj.weight").to(layer_dev)
                up = weights.pop(f"{p}.mlp.up_proj.weight").to(layer_dev)
                down = weights.pop(f"{p}.mlp.down_proj.weight").to(layer_dev)
                self.dense_w1.append(torch.cat([gate, up], dim=0))  # [2*dense_I, H]
                self.dense_w2.append(down)                           # [H, dense_I]
                self.shared_w1.append(None)
                self.shared_w2.append(None)
                self.router.append(None)
                if offloading:
                    moe_I = self.moe_intermediate_size
                    self.w1_cpu.append(torch.zeros(E, 2 * moe_I, self.hidden_size).pin_memory())
                    self.w2_cpu.append(torch.zeros(E, self.hidden_size, moe_I).pin_memory())
                    self.expert_map_abs.append(torch.full((E,), -1, dtype=torch.int32, device=self.device))
                    if self.experts_per_layer is not None:
                        self.w1.append(None)
                        self.w2.append(None)
                        self.expert_map.append(torch.full((E,), -1, dtype=torch.int32, device=self.device))
                else:
                    self.w1.append(None)
                    self.w2.append(None)
                print(f"  Layer {l}: dense MLP (intermediate={self.intermediate_size})")

            elif self.is_mla:
                # MoE layer with shared + routed experts
                self.dense_w1.append(None)
                self.dense_w2.append(None)

                # Shared experts
                sp = f"{p}.mlp.shared_experts"
                shared_gate = weights.pop(f"{sp}.gate_proj.weight").to(layer_dev)
                shared_up = weights.pop(f"{sp}.up_proj.weight").to(layer_dev)
                shared_down = weights.pop(f"{sp}.down_proj.weight").to(layer_dev)
                self.shared_w1.append(torch.cat([shared_gate, shared_up], dim=0))
                self.shared_w2.append(shared_down)

                # Router
                self.router.append(weights.pop(f"{p}.mlp.gate.weight").to(layer_dev))

                # Routed experts (same as existing, using moe_intermediate_size)
                w1_list, w2_list = [], []
                for e in range(self.num_experts):
                    expert_prefix = f"{p}.mlp.experts.{e}"
                    gate = weights.pop(f"{expert_prefix}.gate_proj.weight")
                    up = weights.pop(f"{expert_prefix}.up_proj.weight")
                    down = weights.pop(f"{expert_prefix}.down_proj.weight")
                    w1_list.append(torch.cat([gate, up], dim=0))
                    w2_list.append(down)

                if offloading:
                    w1_full = torch.stack(w1_list).pin_memory()
                    w2_full = torch.stack(w2_list).pin_memory()
                    self.w1_cpu.append(w1_full)
                    self.w2_cpu.append(w2_full)

                    if self.experts_per_layer is not None:
                        base = l * experts_per_layer
                        copy_n = min(experts_per_layer, E)
                        for slot in range(copy_n):
                            self.w1_buf[base + slot].copy_(w1_full[slot])
                            self.w2_buf[base + slot].copy_(w2_full[slot])
                        self.w1.append(self.w1_buf[base : base + experts_per_layer])
                        self.w2.append(self.w2_buf[base : base + experts_per_layer])
                        emap_rel = torch.full((E,), -1, dtype=torch.int32, device=self.device)
                        for slot in range(copy_n):
                            emap_rel[slot] = slot
                        self.expert_map.append(emap_rel)
                        emap_abs = torch.full((E,), -1, dtype=torch.int32, device=self.device)
                        for slot in range(copy_n):
                            emap_abs[slot] = base + slot
                        self.expert_map_abs.append(emap_abs)
                    else:
                        emap_abs = torch.full((E,), -1, dtype=torch.int32, device=self.device)
                        self.expert_map_abs.append(emap_abs)
                        print(f"  Layer {l}: w1_cpu {w1_full.shape} (unified cache, no pre-load)")
                else:
                    if self.ep_size > 1:
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
                        print(f"  Layer {l}: w1 {self.w1[-1].shape}, "
                              f"w2 {self.w2[-1].shape}"
                              f"{f' ({layer_dev})' if self.pp_size > 1 else ''}")

            else:
                # ORIGINAL code path for OLMoE/Mixtral (UNCHANGED)
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
                    if self.ep_size > 1:
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
        if self.is_mla:
            moe_I = self.moe_intermediate_size
            if not offloading:
                first_moe = self.first_k_dense_replace
                expected_E = self.local_num_experts if self.ep_size > 1 else self.num_experts
                assert self.w1[first_moe].shape == (expected_E, 2 * moe_I, self.hidden_size)
                assert self.w2[first_moe].shape == (expected_E, self.hidden_size, moe_I)
        elif offloading:
            if self.experts_per_layer is not None:
                experts_per_layer = self.experts_per_layer
                assert self.w1[0].shape == (experts_per_layer, 2 * self.moe_intermediate_size, self.hidden_size)
                assert self.w2[0].shape == (experts_per_layer, self.hidden_size, self.moe_intermediate_size)
                assert self.w1_buf.shape[0] == self.num_layers * experts_per_layer + self.scratchpad_slots
            else:
                assert self.w1_buf.shape[0] == self.cache_size
            assert self.w1_cpu[0].shape == (self.num_experts, 2 * self.moe_intermediate_size, self.hidden_size)
        elif self.ep_size > 1:
            assert self.w1[0].shape == (self.local_num_experts, 2 * self.moe_intermediate_size, self.hidden_size)
            assert self.w2[0].shape == (self.local_num_experts, self.hidden_size, self.moe_intermediate_size)
        else:
            assert self.w1[0].shape == (self.num_experts, 2 * self.moe_intermediate_size, self.hidden_size)
            assert self.w2[0].shape == (self.num_experts, self.hidden_size, self.moe_intermediate_size)

        # Stack per-layer weights into single tensors for indexed access.
        # When PP > 1, weights are on different GPUs — keep as lists.
        # self.foo[layer] works identically for both lists and stacked tensors.
        if self.is_mla:
            if self.pp_size == 1:
                self.input_layernorm = torch.stack(self.input_layernorm)
                self.post_attn_layernorm = torch.stack(self.post_attn_layernorm)
                if self.q_lora_rank is not None:
                    self.q_a_proj = torch.stack(self.q_a_proj)
                    self.q_a_layernorm = torch.stack(self.q_a_layernorm)
                    self.q_b_proj = torch.stack(self.q_b_proj)
                else:
                    self.q_proj = torch.stack(self.q_proj)      # [L, 3072, 2048]
                self.kv_a_proj = torch.stack(self.kv_a_proj)    # [L, 576, 2048]
                self.kv_a_layernorm = torch.stack(self.kv_a_layernorm)  # [L, 512]
                self.kv_b_proj = torch.stack(self.kv_b_proj)    # [L, 4096, 512]
                self.o_proj = torch.stack(self.o_proj)           # [L, 2048, 2048]
                self.W_UK_T = torch.stack(self.W_UK_T)          # [L, H, P, C]
                self.W_UV = torch.stack(self.W_UV)              # [L, H, C, V]
            # PP: keep as lists (weights on different GPUs can't be stacked)
            if self.n_group > 1:
                # Zero bias for ops.grouped_topk (fused CUDA kernel requires
                # non-None bias; zero bias = no correction, DS-V2 semantics).
                self._zero_routing_bias = torch.zeros(
                    self.num_experts, device=self.device, dtype=torch.float32)
            self.qkv_proj = None  # Not used for MLA
        elif self.pp_size > 1:
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

    def _build_yarn_rope_cache(self) -> torch.Tensor:
        """YaRN RoPE cache for MLA (applied only to qk_rope_head_dim dims)."""
        cfg = self.rope_scaling_config
        half_dim = self.qk_rope_head_dim // 2  # 32

        # Base frequencies
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, half_dim, dtype=torch.float32) / half_dim))

        # YaRN correction range
        scaling_factor = cfg["factor"]
        beta_fast = cfg.get("beta_fast", 32)
        beta_slow = cfg.get("beta_slow", 1)
        original_max_pos = cfg.get("original_max_position_embeddings", 4096)

        # Find correction range (yarn_find_correction_range from vLLM)
        low = math.floor(_yarn_find_correction_dim(
            beta_fast, self.qk_rope_head_dim, self.rope_theta, original_max_pos))
        high = math.ceil(_yarn_find_correction_dim(
            beta_slow, self.qk_rope_head_dim, self.rope_theta, original_max_pos))
        low = max(low, 0)
        high = min(high, half_dim - 1)

        # Linear ramp mask: 0 at low dims (extrapolate/keep original), 1 at high dims (interpolate)
        if low == high:
            high = high + 0.001  # vLLM convention: avoid div-by-zero
        t = torch.arange(half_dim, dtype=torch.float32)
        mask = ((t - low) / (high - low)).clamp(0, 1)

        # Blend interpolated and original frequencies using YaRN ramp.
        inv_freq_interp = inv_freq / scaling_factor
        inv_freq_yarn = inv_freq_interp * mask + inv_freq * (1 - mask)

        # cos/sin mscale correction
        mscale_coeff = cfg.get("mscale", 1.0)
        mscale_all_dim_coeff = cfg.get("mscale_all_dim", 0.0)
        attn_factor = cfg.get("attn_factor", 1.0)
        if scaling_factor <= 1.0:
            cos_sin_mscale = 1.0
        else:
            cos_sin_mscale = (
                _yarn_get_mscale(scaling_factor, mscale_coeff)
                / _yarn_get_mscale(scaling_factor, mscale_all_dim_coeff)
                * attn_factor
            )

        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq_yarn)
        cos = torch.cos(angles) * cos_sin_mscale
        sin = torch.sin(angles) * cos_sin_mscale
        return torch.cat([cos, sin], dim=-1)  # [max_seq_len, qk_rope_head_dim]

    # ── Offloading ────────────────────────────────────────────────────

    @property
    def offloading_active(self):
        """True when experts_per_layer < num_experts (some experts on CPU)."""
        return (self.experts_per_layer is not None
                and self.experts_per_layer < self.num_experts)

    # ── Prefill ──────────────────────────────────────────────────────

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill forward pass — routes through step (piecewise graphs).

        Args: input_ids [B, S]
        Returns: logits [B, vocab_size] — last-token logits per sequence
        Requires: capture_cuda_graphs() (or capture_prefill_cuda_graph())
        called with sufficient sizes, or falls back to eager.
        """
        B, S = input_ids.shape
        seq_ids = list(range(B))
        input_list = [input_ids[i] for i in range(B)]
        logits_flat = self.step(
            decode_seq_ids=[],
            decode_token_ids=torch.empty(0, dtype=torch.long,
                                         device=self.device),
            prefill_seq_ids=seq_ids,
            prefill_input_ids=input_list)
        # step() returns only last-token logits for prefill: [B, vocab_size]
        return logits_flat.view(B, -1)

    def capture_prefill_cuda_graph(self, total_token_sizes=None,
                                   use_torch_compile=None):
        """Capture CUDA graphs for prefill — delegates to capture_cuda_graphs.

        Retained for backward compatibility. Internally captures piecewise
        per-layer graphs that handle both prefill and decode.

        Args:
            total_token_sizes: List of total token counts to capture
                (default: [128, 256, 512, 1024, 2048])
            use_torch_compile: Override instance default
        """
        if total_token_sizes is None:
            total_token_sizes = [128, 256, 512, 1024, 2048]
        self.capture_cuda_graphs(
            total_token_sizes=total_token_sizes,
            use_torch_compile=use_torch_compile)

    # ── Decode ───────────────────────────────────────────────────────

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """Single decode step — routes through step (piecewise graphs).

        Args: token_ids [B], positions [B] (ignored, kept for API compat)
        Returns: logits [B, vocab_size]
        """
        B = token_ids.shape[0]
        return self.step(
            decode_seq_ids=list(range(B)),
            decode_token_ids=token_ids,
            prefill_seq_ids=[],
            prefill_input_ids=[])

    def _compute_slot_mapping(self, positions, B):
        """Compute flat slot indices for reshape_and_cache_flash kernel.
        slot = page_number * page_size + offset_within_page
        """
        batch_idx = torch.arange(B, device=self.device)
        page_idx = (positions // self.page_size).long()
        offset = (positions % self.page_size).long()
        bt = self.block_table[0] if self.pp_size > 1 else self.block_table
        return (bt[batch_idx, page_idx].long() * self.page_size + offset)

    def _plan_flashinfer_decode(self, wrapper, B):
        """Call FlashInfer plan() with current batch metadata (contiguous [:B]).

        Optimized: uses Triton kernel for page indices, pre-allocated buffers.
        For contiguous [:B], block table rows are already in order — no gather.
        """
        if B == 0:
            return

        # 1. CPU metadata (contiguous [:B])
        seq_lens_np = self._seq_lens_cpu[:B].numpy().astype(np.int32)
        num_blocks_np = (seq_lens_np + self.page_size - 1) // self.page_size

        self._plan_indptr_np[0] = 0
        np.cumsum(num_blocks_np, out=self._plan_indptr_np[1:B + 1])
        num_actual_pages = int(self._plan_indptr_np[B])

        lpl_np = seq_lens_np % self.page_size
        self._plan_lpl_np[:B] = np.where(
            (lpl_np == 0) & (seq_lens_np != 0), self.page_size, lpl_np)

        # 2. indptr to GPU
        self._plan_indptr_gpu[:B + 1].copy_(
            self._plan_indptr_cpu[:B + 1], non_blocking=True)

        # 3. Contiguous path: block table rows 0..B-1 are in order, no gather
        _copy_page_indices_kernel[(B,)](
            self._plan_indices_gpu,
            self.block_table,
            self.block_table.stride(0),
            self._plan_indptr_gpu,
            BLOCK_SIZE=1024)

        # 4. plan()
        wrapper.plan(
            self._plan_indptr_cpu[:B + 1],
            self._plan_indices_gpu[:num_actual_pages],
            self._plan_lpl_cpu[:B],
            num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, page_size=self.page_size,
            pos_encoding_mode="NONE", q_data_type=self.dtype)

    def _plan_mla_decode(self, wrapper, B, gpu_idx=None):
        """Plan FlashInfer MLA decode with current batch metadata.

        CRITICAL: Caller MUST increment self._seq_lens_cpu[:B] += 1 BEFORE
        calling this method, so FlashInfer includes the current token's KV
        in attention.

        Args:
            gpu_idx: For PP mode, which GPU's block_table to use. None for single-GPU.
        """
        seq_lens = self._seq_lens_cpu[:B]
        pages_per_seq = (seq_lens + self.page_size - 1) // self.page_size

        # qo_indptr: [0, 1, 2, ..., B] for decode (1 query token per request)
        qo_indptr = torch.arange(B + 1, dtype=torch.int32)

        # kv_indptr: cumulative page counts
        kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
        kv_indptr[1:] = pages_per_seq.cumsum(0)

        if gpu_idx is not None:
            bt = self.block_table[gpu_idx]
            dev = self.pp_devices[gpu_idx]
        else:
            bt = self.block_table
            dev = self.device

        max_pages = pages_per_seq.max().item() if B > 0 else 0
        if max_pages > 0:
            page_range = torch.arange(max_pages, dtype=torch.int32)
            valid = page_range.unsqueeze(0) < pages_per_seq.unsqueeze(1)
            all_pages = bt[:B, :max_pages].cpu()
            kv_indices = all_pages[valid].to(torch.int32).to(dev)
        else:
            kv_indices = torch.zeros(0, dtype=torch.int32, device=dev)

        kv_len_arr = seq_lens.to(torch.int32)

        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_len_arr,
            num_heads=self.num_heads,
            head_dim_ckv=self.kv_lora_rank,
            head_dim_kpe=self.qk_rope_head_dim,
            page_size=self.page_size,
            causal=False,
            sm_scale=self._mla_sm_scale,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    def _plan_mla_decode_for_subset(self, wrapper, seq_ids, gpu_idx=None):
        """Plan FlashInfer MLA decode for a subset of sequence slots.

        Unlike _plan_mla_decode which assumes contiguous [:B] slots,
        this handles arbitrary slot indices in seq_ids.

        CRITICAL: Caller MUST increment _seq_lens_cpu for each seq_id
        BEFORE calling this method.

        Args:
            seq_ids: list[int] — arbitrary slot indices for decode sequences
            gpu_idx: For PP mode, which GPU's block_table to use. None for single-GPU.
        """
        B = len(seq_ids)
        sid_tensor = torch.tensor(seq_ids, dtype=torch.long)
        seq_lens = self._seq_lens_cpu[sid_tensor]
        pages_per_seq = (seq_lens + self.page_size - 1) // self.page_size

        qo_indptr = torch.arange(B + 1, dtype=torch.int32)
        kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
        kv_indptr[1:] = pages_per_seq.cumsum(0)

        if gpu_idx is not None:
            bt = self.block_table[gpu_idx]
            dev = self.pp_devices[gpu_idx]
        else:
            bt = self.block_table
            dev = self.device

        max_pages = pages_per_seq.max().item() if B > 0 else 0
        if max_pages > 0:
            page_range = torch.arange(max_pages, dtype=torch.int32)
            valid = page_range.unsqueeze(0) < pages_per_seq.unsqueeze(1)
            all_pages = bt[sid_tensor, :max_pages].cpu()
            kv_indices = all_pages[valid].to(torch.int32).to(dev)
        else:
            kv_indices = torch.zeros(0, dtype=torch.int32, device=dev)

        kv_len_arr = seq_lens.to(torch.int32)

        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_len_arr,
            num_heads=self.num_heads,
            head_dim_ckv=self.kv_lora_rank,
            head_dim_kpe=self.qk_rope_head_dim,
            page_size=self.page_size,
            causal=False,
            sm_scale=self._mla_sm_scale,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    def _plan_mla_continuation(self, wrapper, cont_seq_ids, cont_chunk_lens,
                                cont_total_seq_lens, gpu_idx=None):
        """Plan FlashInfer MLA wrapper for multi-token continuation prefill.

        Like _plan_mla_decode_for_subset but builds qo_indptr from per-sequence
        chunk lengths (not all-ones) and uses causal=True for correct masking.

        Caller MUST have already written current chunk KV to cache (stage1)
        and updated _seq_lens_cpu to include current chunk (section 4b).
        """
        B = len(cont_seq_ids)
        if gpu_idx is not None:
            bt = self.block_table[gpu_idx]
            dev = self.pp_devices[gpu_idx]
        else:
            bt = self.block_table
            dev = self.device

        # qo_indptr: cumulative chunk lengths (multi-token queries, not all-ones)
        qo_indptr = torch.zeros(B + 1, dtype=torch.int32)
        qo_indptr[1:] = torch.tensor(cont_chunk_lens, dtype=torch.int32).cumsum(0)

        # kv: based on total sequence lengths (KV cache includes current chunk)
        total_sl = torch.tensor(cont_total_seq_lens, dtype=torch.int32)
        pages_per_seq = (total_sl + self.page_size - 1) // self.page_size
        kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
        kv_indptr[1:] = pages_per_seq.cumsum(0)

        sid_tensor = torch.tensor(cont_seq_ids, dtype=torch.long)
        max_pages = pages_per_seq.max().item() if B > 0 else 0
        if max_pages > 0:
            page_range = torch.arange(max_pages, dtype=torch.int32)
            valid = page_range.unsqueeze(0) < pages_per_seq.unsqueeze(1)
            all_pages = bt[sid_tensor, :max_pages].cpu()
            kv_indices = all_pages[valid].to(torch.int32).to(dev)
        else:
            kv_indices = torch.zeros(0, dtype=torch.int32, device=dev)

        kv_len_arr = total_sl.to(torch.int32)

        wrapper.plan(
            qo_indptr.to(dev), kv_indptr.to(dev), kv_indices, kv_len_arr.to(dev),
            num_heads=self.num_heads,
            head_dim_ckv=self.kv_lora_rank,
            head_dim_kpe=self.qk_rope_head_dim,
            page_size=self.page_size,
            causal=True,
            sm_scale=self._mla_sm_scale,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    # ── Slot-based Prefill ────────────────────────────────────────────

    @torch.no_grad()
    def prefill_to_slot(self, seq_id: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill a single sequence into a specific KV cache slot.

        Routes through step for piecewise CUDA graph execution.

        Args:
            seq_id: slot index in block_table / seq_lens
            input_ids: [S] token IDs (1-D)
        Returns:
            logits: [1, vocab_size] — last-token logits
        """
        empty_dev = self.pp_devices[0] if self.pp_size > 1 else self.device
        return self.step(
            decode_seq_ids=[],
            decode_token_ids=torch.empty(0, dtype=torch.long,
                                         device=empty_dev),
            prefill_seq_ids=[seq_id],
            prefill_input_ids=[input_ids])

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
            logits: [1, vocab_size] — last-token logits
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
                logits = self.step(
                    decode_seq_ids=[],
                    decode_token_ids=torch.empty(0, dtype=torch.long,
                                                 device=empty_dev),
                    prefill_seq_ids=[seq_id],
                    prefill_input_ids=[chunk_ids])
            else:
                # Subsequent chunks: prefill with KV cache attention
                # begin_step was already called by chunk 0's step
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
        self._num_computed_tokens_gpu[seq_id] = total_seq_len

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

            # Stage 3: attention for continuation chunk
            if self.is_mla:
                # MLA continuation: decompress and use FA3 self-attention
                # (same as stage3a in piecewise; full-prefill path)
                q_nope_pf = buf['q_nope_buf'][:C]
                q_pe_pf = buf['q_pe_buf'][:C]
                c_kv_pf = buf['ckv_buf'][:C]
                k_pe_pf = buf['kpe_buf'][:C]
                kv_decompressed = F.linear(c_kv_pf, self.kv_b_proj[layer])
                kv_reshaped = kv_decompressed.view(C, self.num_heads,
                                                    self.qk_nope_head_dim + self.v_head_dim)
                k_nope_pf = kv_reshaped[:, :, :self.qk_nope_head_dim]
                v_pf = kv_reshaped[:, :, self.qk_nope_head_dim:]
                k_pe_exp = k_pe_pf.unsqueeze(1).expand(-1, self.num_heads, -1)
                k_full = torch.cat([k_nope_pf, k_pe_exp], dim=-1)
                q_full_pf = torch.cat([q_nope_pf, q_pe_pf], dim=-1)
                dev = self.pp_devices[self.pp_layer_gpu[layer]] if pp else self.device
                cu_pf = torch.tensor([0, C], dtype=torch.int32, device=dev)
                pf_out = flash_attn_varlen_func(
                    q_full_pf.contiguous(), k_full.contiguous(), v_pf.contiguous(),
                    cu_seqlens_q=cu_pf,
                    cu_seqlens_k=cu_pf,
                    max_seqlen_q=C,
                    max_seqlen_k=C,
                    softmax_scale=self._mla_sm_scale,
                    causal=True, fa_version=3)
                attn_out_buf[:C].copy_(
                    pf_out.reshape(C, self.num_heads * self.v_head_dim))
            else:
                q_buf = buf['q_buf']
                q_pf = q_buf[:C]
                if pp:
                    gpu_idx = self.pp_layer_gpu[layer]
                    pw = prefill_wrappers[gpu_idx]
                # else pw already set above (single-GPU)
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

        # ── 8. Final norm + lm_head (last token only) ──
        last_buf = _buf(self.num_layers - 1)
        hidden = last_buf['hidden_buf'][C - 1:C]  # last actual token
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

        return logits

    # ── Multi-Batch Prefill ─────────────────────────────────────────

    @torch.no_grad()
    def prefill_batch_to_slots(self, seq_ids, input_ids):
        """Prefill multiple sequences into specific KV cache slots.

        Routes through step for piecewise CUDA graph execution.
        Supports both same-length and variable-length sequences.

        Args:
            seq_ids: list[int] — KV cache slot indices for each sequence
            input_ids: list[Tensor] (variable-length 1D) or Tensor [B, S] (same-length)
        Returns:
            logits: Tensor [num_seqs, vocab_size] — last-token logits per sequence
        """
        # Normalize to list of 1D tensors
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2:
            input_ids = [input_ids[i] for i in range(input_ids.shape[0])]

        empty_dev = self.pp_devices[0] if self.pp_size > 1 else self.device
        return self.step(
            decode_seq_ids=[],
            decode_token_ids=torch.empty(0, dtype=torch.long,
                                         device=empty_dev),
            prefill_seq_ids=list(seq_ids),
            prefill_input_ids=list(input_ids))

    # ── Mixed Batch Support ──────────────────────────────────────────

    def _plan_flashinfer_decode_for_subset(self, seq_ids):
        """Plan FlashInfer decode for a subset of sequence slots.

        Uses _copy_page_indices_kernel to flatten block table pages on GPU
        (no GPU->CPU->GPU roundtrip). Pre-allocated buffers, numpy for small
        CPU metadata, non_blocking H2D copies.

        When called from _step_piecewise, _idx_mapping_buf[:B] already has
        seq_ids on GPU (from section 2a). For eager fallback, fills GPU index
        buffer directly.
        """
        B = len(seq_ids)
        if B == 0:
            return

        # 1. Compute indptr and last_page_len on CPU (numpy)
        seq_ids_np = np.array(seq_ids, dtype=np.int64)
        seq_lens_np = self._seq_lens_cpu.numpy()[seq_ids_np].astype(np.int32)
        num_blocks_np = (seq_lens_np + self.page_size - 1) // self.page_size

        self._plan_indptr_np[0] = 0
        np.cumsum(num_blocks_np, out=self._plan_indptr_np[1:B + 1])
        num_actual_pages = int(self._plan_indptr_np[B])

        lpl_np = seq_lens_np % self.page_size
        self._plan_lpl_np[:B] = np.where(
            (lpl_np == 0) & (seq_lens_np != 0), self.page_size, lpl_np)

        # 2. Copy indptr to GPU for Triton kernel
        self._plan_indptr_gpu[:B + 1].copy_(
            self._plan_indptr_cpu[:B + 1], non_blocking=True)

        # 3. Gather block table rows on GPU, flatten via Triton kernel
        if hasattr(self, '_idx_mapping_buf'):
            # Piecewise path: reuse _idx_mapping_buf[:B] (already on GPU)
            self._plan_idx_long[:B].copy_(self._idx_mapping_buf[:B])
        else:
            # Eager fallback: fill directly
            self._plan_idx_long[:B] = torch.tensor(
                seq_ids, dtype=torch.int64, device=self.device)

        torch.index_select(
            self.block_table, 0, self._plan_idx_long[:B],
            out=self._plan_gathered_bt[:B])

        _copy_page_indices_kernel[(B,)](
            self._plan_indices_gpu,
            self._plan_gathered_bt,
            self._plan_gathered_bt.stride(0),
            self._plan_indptr_gpu,
            BLOCK_SIZE=1024)

        # 4. Call plan() with CPU indptr/lpl + GPU indices
        self._decode_wrapper.plan(
            self._plan_indptr_cpu[:B + 1],
            self._plan_indices_gpu[:num_actual_pages],
            self._plan_lpl_cpu[:B],
            num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, page_size=self.page_size,
            pos_encoding_mode="NONE", q_data_type=self.dtype)

    def _plan_flashinfer_decode_pp(self, seq_ids, gpu_idx):
        """Plan FlashInfer decode for a subset of slots on a specific PP GPU.

        Same optimization as single-GPU but uses per-GPU buffers and block table.
        CPU metadata (indptr, lpl) computed once for gpu_idx==0, reused for others.
        """
        B = len(seq_ids)
        if B == 0:
            return

        bt = self.block_table[gpu_idx]

        # 1. CPU metadata — only compute once for gpu_idx==0
        if gpu_idx == 0:
            seq_ids_np = np.array(seq_ids, dtype=np.int64)
            seq_lens_np = self._seq_lens_cpu.numpy()[seq_ids_np].astype(np.int32)
            num_blocks_np = (seq_lens_np + self.page_size - 1) // self.page_size

            self._plan_indptr_np[0] = 0
            np.cumsum(num_blocks_np, out=self._plan_indptr_np[1:B + 1])

            lpl_np = seq_lens_np % self.page_size
            self._plan_lpl_np[:B] = np.where(
                (lpl_np == 0) & (seq_lens_np != 0), self.page_size, lpl_np)

        num_actual_pages = int(self._plan_indptr_np[B])

        # 2. Copy indptr to this GPU
        indptr_gpu = self._plan_indptr_gpu_pp[gpu_idx]
        indptr_gpu[:B + 1].copy_(
            self._plan_indptr_cpu[:B + 1], non_blocking=True)

        # 3. Gather block table rows on this GPU, flatten
        idx_long = self._plan_idx_long_pp[gpu_idx]
        gathered = self._plan_gathered_bt_pp[gpu_idx]
        indices = self._plan_indices_gpu_pp[gpu_idx]

        idx_long[:B].copy_(self._idx_mapping_buf[:B])

        torch.index_select(bt, 0, idx_long[:B], out=gathered[:B])

        _copy_page_indices_kernel[(B,)](
            indices, gathered, gathered.stride(0), indptr_gpu, BLOCK_SIZE=1024)

        # 4. plan()
        self._decode_wrappers[gpu_idx].plan(
            self._plan_indptr_cpu[:B + 1],
            indices[:num_actual_pages],
            self._plan_lpl_cpu[:B],
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
        if self.is_mla:
            return self._layer_mixed_mla(hidden, layer, positions, slot_mapping,
                                          num_decode_tokens,
                                          prefill_cu_seqlens, prefill_max_seqlen)
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

        # Q/K norm (if model has it)
        if self.has_qk_norm:
            q = F.rms_norm(q, (H,), self.q_norm[layer], self.rms_norm_eps)
            k = F.rms_norm(k, (kv_dim,), self.k_norm[layer], self.rms_norm_eps)

        # reshape creates contiguous copies; rotary_embedding modifies in-place
        q_3d = q.reshape(N, self.num_heads, self.head_dim)
        k_3d = k.reshape(N, self.num_kv_heads, self.head_dim)
        v_3d = v.reshape(N, self.num_kv_heads, self.head_dim)

        # RoPE — in-place on bf16 tensors (cos_sin_cache is also bf16)
        cos_sin = self._get_cos_sin_cache(layer)
        _vllm_ops.rotary_embedding(
            positions.long(), q_3d, k_3d, self.head_dim, cos_sin, True)

        # Write K,V to paged cache for ALL tokens
        k_scale, v_scale = self._get_kv_scales(layer)
        _vllm_ops.reshape_and_cache_flash(
            k_3d, v_3d,
            self.k_cache[layer], self.v_cache[layer],
            slot_mapping, "auto", k_scale, v_scale)

        # ── Split attention ──
        # Decode tokens [0:D]: FlashInfer BatchDecode (reads paged KV cache)
        if D > 0 and N > D:
            # Mixed: both decode and prefill
            q_decode = q_3d[:D]
            decode_out = decode_wrapper.run(
                q_decode, (self.k_cache[layer], self.v_cache[layer]))
            decode_out = decode_out.reshape(D, H)

            # Prefill tokens [D:]: FA3 (stateless) instead of FlashInfer —
            # FlashInfer's fmha_varlen_plan() allocates fresh GPU tensors per
            # call, so addresses baked into a CUDA graph get freed by the next
            # plan() call.
            N_pf = N - D
            q_pf = q_3d[D:]
            k_pf = k_3d[D:]
            v_pf = v_3d[D:]
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
            q_decode = q_3d[:D]
            attn_out = decode_wrapper.run(
                q_decode, (self.k_cache[layer], self.v_cache[layer]))
            attn_out = attn_out.reshape(D, H)

        else:
            # Pure prefill — FA3 (stateless), same reason as mixed path above
            N_pf = N
            q_pf = q_3d
            k_pf = k_3d
            v_pf = v_3d
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

        topk_weights, topk_ids = self._route(hidden, layer)

        if self.ep_size > 1:
            orig_N = hidden.shape[0]
            local_N = torch.tensor([orig_N], device=self.device)
            dist.all_reduce(local_N, op=dist.ReduceOp.MAX,
                            group=self.ep_group)
            max_N = int(local_N.item())
            if orig_N < max_N:
                pad = max_N - orig_N
                hidden = F.pad(hidden, (0, 0, 0, pad))
                topk_weights = F.pad(topk_weights, (0, 0, 0, pad))
                topk_ids = F.pad(topk_ids, (0, 0, 0, pad))

            gathered_h = ep_allgather(hidden, self.ep_group, self.ep_size)
            gathered_w = ep_allgather(topk_weights, self.ep_group, self.ep_size)
            gathered_ids = ep_allgather(
                topk_ids.to(torch.int32), self.ep_group, self.ep_size)
            partial = self._moe_experts(
                gathered_h, self.w1[layer], self.w2[layer],
                gathered_w, gathered_ids, self._ep_expert_map_gpu)
            hidden = ep_reducescatter(partial, self.ep_group, self.ep_size)
            hidden = hidden[:orig_N]
        else:
            expert_map = (self.expert_map[layer]
                          if self.experts_per_layer is not None else None)
            hidden = self._moe_experts(hidden, self.w1[layer], self.w2[layer],
                                       topk_weights, topk_ids, expert_map)

        return residual + hidden

    def _layer_mixed_mla(self, hidden, layer, positions, slot_mapping,
                          D, prefill_cu_seqlens, prefill_max_seqlen):
        """Eager MLA forward for warmup and fallback (no CUDA graphs).

        Args:
            hidden: [N_total, H]
            layer: layer index
            positions: [N_total] int32
            slot_mapping: [N_total] int64
            D: num_decode_tokens — hidden[:D] are decode, [D:] are prefill
            prefill_cu_seqlens: [num_prefill_reqs + 1] int32, for FA3
            prefill_max_seqlen: max prefill seq length
        """
        N, H = hidden.shape[0], self.hidden_size

        # --- Stage 1: RMSNorm + Q proj + KV compress + layernorm + RoPE + cache write ---
        residual = hidden
        hidden = F.rms_norm(hidden, (H,), self.input_layernorm[layer], self.rms_norm_eps)

        if self.q_lora_rank is not None:
            q_compressed = F.linear(hidden, self.q_a_proj[layer])
            q_compressed = F.rms_norm(q_compressed, (self.q_lora_rank,),
                                      self.q_a_layernorm[layer], self.rms_norm_eps)
            q_full = F.linear(q_compressed, self.q_b_proj[layer])
        else:
            q_full = F.linear(hidden, self.q_proj[layer])
        q_full = q_full.view(N, self.num_heads, self.qk_head_dim)
        q_nope = q_full[:, :, :self.qk_nope_head_dim]
        q_pe = q_full[:, :, self.qk_nope_head_dim:]

        kv_a = F.linear(hidden, self.kv_a_proj[layer])
        c_kv = kv_a[:, :self.kv_lora_rank]
        k_pe = kv_a[:, self.kv_lora_rank:]
        c_kv = F.rms_norm(c_kv, (self.kv_lora_rank,), self.kv_a_layernorm[layer], self.rms_norm_eps)

        cos_sin = self._get_cos_sin_cache(layer)
        q_pe_3d = q_pe.contiguous()            # [N, num_heads, R]
        k_pe_3d = k_pe.unsqueeze(1)            # [N, 1, R]
        _vllm_ops.rotary_embedding(
            positions.long(), q_pe_3d, k_pe_3d,
            self.qk_rope_head_dim, cos_sin, False)
        q_pe_rot = q_pe_3d                     # [N, num_heads, R]
        k_pe_rot = k_pe_3d.squeeze(1)          # [N, R]

        safe_slots = torch.where(slot_mapping >= 0, slot_mapping, self._scratch_slot)
        self.ckv_flat[layer][safe_slots] = c_kv
        self.kpe_flat[layer][safe_slots] = k_pe_rot

        # --- Stage 2: MLA decode attention (tokens :D) ---
        if D > 0:
            if self.pp_size > 1:
                gpu_idx = self.pp_layer_gpu[layer]
                mla_wrapper = self._mla_wrappers[gpu_idx]
            else:
                mla_wrapper = self._mla_decode_wrapper
            q_absorbed = torch.einsum('bhp,hpc->bhc', q_nope[:D], self.W_UK_T[layer])
            attn_latent = mla_wrapper.run(
                q_absorbed, q_pe_rot[:D], self.ckv_cache[layer], self.kpe_cache[layer])
            attn_v = torch.einsum('bhc,hcv->bhv', attn_latent, self.W_UV[layer])
            decode_out = attn_v.reshape(D, self.num_heads * self.v_head_dim)

        # --- Stage 3: MLA prefill attention (tokens D:N) ---
        N_pf = N - D
        prefill_out = None
        if N_pf > 0:
            q_nope_pf = q_nope[D:]    # [N_pf, H, 128]
            q_pe_pf = q_pe_rot[D:]    # [N_pf, H, 64]
            c_kv_pf = c_kv[D:]        # [N_pf, 512]
            k_pe_pf = k_pe_rot[D:]    # [N_pf, 64]

            # Decompress via kv_b_proj: [N_pf, 512] -> [N_pf, H*(P+V)]
            kv_decompressed = F.linear(c_kv_pf, self.kv_b_proj[layer])
            kv_reshaped = kv_decompressed.view(N_pf, self.num_heads,
                                                self.qk_nope_head_dim + self.v_head_dim)
            k_nope_pf = kv_reshaped[:, :, :self.qk_nope_head_dim]  # [N_pf, H, 128]
            v_pf = kv_reshaped[:, :, self.qk_nope_head_dim:]       # [N_pf, H, 128]

            # k_pe broadcast to all heads: [N_pf, 64] -> [N_pf, H, 64]
            k_pe_expanded = k_pe_pf.unsqueeze(1).expand(-1, self.num_heads, -1)
            k_full = torch.cat([k_nope_pf, k_pe_expanded], dim=-1)   # [N_pf, H, 192]
            q_full_pf = torch.cat([q_nope_pf, q_pe_pf], dim=-1)      # [N_pf, H, 192]

            # FA3 on Hopper natively handles Q/K dim=192 vs V dim=128
            pf_out = flash_attn_varlen_func(
                q_full_pf.contiguous(), k_full.contiguous(), v_pf.contiguous(),
                cu_seqlens_q=prefill_cu_seqlens,
                cu_seqlens_k=prefill_cu_seqlens,
                max_seqlen_q=prefill_max_seqlen,
                max_seqlen_k=prefill_max_seqlen,
                softmax_scale=self._mla_sm_scale,
                causal=True, fa_version=3)
            # pf_out: [N_pf, H, V=128]
            prefill_out = pf_out.reshape(N_pf, self.num_heads * self.v_head_dim)

        # Combine decode and prefill outputs
        if D > 0 and prefill_out is not None:
            attn_out = torch.cat([decode_out, prefill_out], dim=0)
        elif D > 0:
            attn_out = decode_out
        else:
            attn_out = prefill_out

        # --- O projection + residual ---
        hidden = residual + F.linear(attn_out, self.o_proj[layer])

        # --- Stage 4: Post-attn norm + FFN ---
        residual = hidden
        hidden = F.rms_norm(hidden, (H,), self.post_attn_layernorm[layer], self.rms_norm_eps)

        if self.router[layer] is None:
            # Dense layer: SwiGLU MLP
            gate_up = F.linear(hidden, self.dense_w1[layer])
            I = self.intermediate_size
            hidden = F.linear(F.silu(gate_up[:, :I]) * gate_up[:, I:], self.dense_w2[layer])
        elif self.ep_size > 1:
            topk_weights, topk_ids = self._route(hidden, layer)
            orig_N = hidden.shape[0]
            local_N = torch.tensor([orig_N], device=self.device)
            dist.all_reduce(local_N, op=dist.ReduceOp.MAX,
                            group=self.ep_group)
            max_N = int(local_N.item())

            hidden_padded = hidden
            topk_weights_padded = topk_weights
            topk_ids_padded = topk_ids
            if orig_N < max_N:
                pad = max_N - orig_N
                hidden_padded = F.pad(hidden, (0, 0, 0, pad))
                topk_weights_padded = F.pad(topk_weights, (0, 0, 0, pad))
                topk_ids_padded = F.pad(topk_ids, (0, 0, 0, pad))

            gathered_h = ep_allgather(hidden_padded, self.ep_group, self.ep_size)
            gathered_w = ep_allgather(topk_weights_padded, self.ep_group, self.ep_size)
            gathered_ids = ep_allgather(
                topk_ids_padded.to(torch.int32), self.ep_group, self.ep_size)
            partial = self._moe_experts(
                gathered_h, self.w1[layer], self.w2[layer],
                gathered_w, gathered_ids, self._ep_expert_map_gpu)
            routed = ep_reducescatter(partial, self.ep_group, self.ep_size)[:orig_N]

            shared_gu = F.linear(hidden, self.shared_w1[layer])
            shared_I = self.moe_intermediate_size * self.n_shared_experts
            shared = F.linear(F.silu(shared_gu[:, :shared_I]) * shared_gu[:, shared_I:],
                              self.shared_w2[layer])
            hidden = routed + shared
        else:
            # MoE: router + fused_moe + shared experts
            topk_weights, topk_ids = self._route(hidden, layer)
            routed = self._moe_experts(hidden, self.w1[layer], self.w2[layer],
                                        topk_weights, topk_ids)
            shared_gu = F.linear(hidden, self.shared_w1[layer])
            shared_I = self.moe_intermediate_size * self.n_shared_experts
            shared = F.linear(F.silu(shared_gu[:, :shared_I]) * shared_gu[:, shared_I:],
                              self.shared_w2[layer])
            hidden = routed + shared

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

        if self.has_qk_norm:
            q = F.rms_norm(q, (H,), self.q_norm[layer], self.rms_norm_eps)
            k = F.rms_norm(k, (kv_dim,), self.k_norm[layer], self.rms_norm_eps)

        # reshape creates contiguous copies from non-contiguous split views
        q_3d = q.reshape(N, self.num_heads, self.head_dim)
        k_3d = k.reshape(N, self.num_kv_heads, self.head_dim)
        v_3d = v.reshape(N, self.num_kv_heads, self.head_dim)

        # RoPE — in-place on bf16 tensors (cos_sin_cache is also bf16)
        cos_sin = self._get_cos_sin_cache(layer)
        _vllm_ops.rotary_embedding(
            positions.long(), q_3d, k_3d, self.head_dim, cos_sin, True)

        k_scale, v_scale = self._get_kv_scales(layer)
        _vllm_ops.reshape_and_cache_flash(
            k_3d, v_3d,
            self.k_cache[layer], self.v_cache[layer],
            slot_mapping, "auto", k_scale, v_scale)

        q_buf.copy_(q_3d)
        k_buf.copy_(k_3d)
        v_buf.copy_(v_3d)
        residual_buf.copy_(hidden)

    def _layer_stage1_mla(self, hidden, positions, slot_mapping, layer,
                           q_nope_buf, q_fused_buf,
                           ckv_buf, kpe_buf, residual_buf):
        """Stage 1 for MLA: RMSNorm -> Q proj -> W_UK absorption -> KV compress
        -> KV norm -> RoPE (pe only) -> KV cache write.

        Takes q_fused_buf [N, H, 576] (not separate q_absorbed/q_pe views)
        because torch.compile can't handle aliased tensor inputs — passing two
        views of the same storage as separate args fails synthetic base tracking.
        """
        N = hidden.shape[0]
        H = self.hidden_size

        # RMSNorm
        normed = F.rms_norm(hidden, (H,), self.input_layernorm[layer], self.rms_norm_eps)

        # Q projection (with optional compression for DS-V2)
        if self.q_lora_rank is not None:
            q_compressed = F.linear(normed, self.q_a_proj[layer])
            q_compressed = F.rms_norm(q_compressed, (self.q_lora_rank,),
                                      self.q_a_layernorm[layer], self.rms_norm_eps)
            q_full = F.linear(q_compressed, self.q_b_proj[layer])
        else:
            q_full = F.linear(normed, self.q_proj[layer])
        q_full = q_full.view(N, self.num_heads, self.qk_head_dim)  # [N, H, P+R]
        q_nope = q_full[:, :, :self.qk_nope_head_dim]  # [N, H, P=128]
        q_pe = q_full[:, :, self.qk_nope_head_dim:]     # [N, H, R=64]

        # KV compression: [N, 576] -> split c_kv [N, 512] and k_pe [N, 64]
        kv_compressed = F.linear(normed, self.kv_a_proj[layer])  # [N, C+R]
        c_kv = kv_compressed[:, :self.kv_lora_rank]        # [N, 512]
        k_pe = kv_compressed[:, self.kv_lora_rank:]         # [N, 64]

        # LayerNorm on compressed KV
        c_kv_normed = F.rms_norm(c_kv, (self.kv_lora_rank,),
                                  self.kv_a_layernorm[layer], self.rms_norm_eps)

        # RoPE on pe dimensions only (GPT-J interleaved style)
        cos_sin = self._get_cos_sin_cache(layer)
        q_pe_3d = q_pe.contiguous()            # [N, num_heads, R]
        k_pe_3d = k_pe.unsqueeze(1)            # [N, 1, R]
        _vllm_ops.rotary_embedding(
            positions.long(), q_pe_3d, k_pe_3d,
            self.qk_rope_head_dim, cos_sin, False)
        k_pe_rot = k_pe_3d.squeeze(1)          # [N, R]

        # Write to MLA KV cache (CUDA-graph safe scatter with sentinel handling)
        safe_slots = torch.where(slot_mapping >= 0, slot_mapping, self._scratch_slot)
        self.ckv_flat[layer][safe_slots] = c_kv_normed   # [N, 512]
        self.kpe_flat[layer][safe_slots] = k_pe_rot       # [N, 64]

        # W_UK absorption: q_absorbed = q_nope @ W_UK_T (was eager, now in graph)
        q_absorbed = torch.einsum('bhp,hpc->bhc', q_nope, self.W_UK_T[layer])

        # Output to buffers — q_fused_buf[:, :, :512] = absorbed, [:, :, 512:] = pe
        q_nope_buf.copy_(q_nope)        # [N, H, P=128] (still needed for prefill)
        q_fused_buf[:, :, self.kv_lora_rank:].copy_(q_pe_3d)     # [N, H, R=64]
        q_fused_buf[:, :, :self.kv_lora_rank].copy_(q_absorbed)  # [N, H, C=512]
        ckv_buf.copy_(c_kv_normed)      # [N, 512]
        kpe_buf.copy_(k_pe_rot)         # [N, 64]
        residual_buf.copy_(hidden)

    def _layer_stage2_mla_out(self, attn_latent_buf, layer, attn_out_buf):
        """Stage 2 output for MLA: W_UV projection from attn_latent to attn_out.

        Captured as a CUDA graph. Operates on full buffer (graph_N tokens).
        For decode-only: valid results in [:D], rest is garbage/zero.
        For mixed: stages 3a/3b overwrite [D:D+P] and [D+P:D+P+C] after this.
        """
        N = attn_latent_buf.shape[0]
        attn_v = torch.einsum('bhc,hcv->bhv', attn_latent_buf, self.W_UV[layer])
        attn_out_buf.copy_(attn_v.reshape(N, self.num_heads * self.v_head_dim))

    def _final_norm_lmhead(self, hidden_buf, logits_buf):
        """Final: RMSNorm + lm_head. Captured as CUDA graph for decode."""
        normed = F.rms_norm(hidden_buf, (self.hidden_size,),
                             self.final_norm, self.rms_norm_eps)
        logits = F.linear(normed, self.lm_head)
        logits_buf.copy_(logits)

    def _layer_stage4a_router(self, attn_out, residual, layer,
                              moe_input_buf, moe_residual_buf,
                              topk_weights_buf, topk_ids_buf,
                              token_expert_indices):
        """Stage 4a: O proj -> residual -> norm -> router -> topk.

        Writes routing decisions into buffers for CPU-side inspection between
        stage4a and stage4b. Used for expert offloading (demand loading between
        the two sub-stages).

        Standard models: vLLM fused topk_softmax kernel (softmax + topk +
        renormalize in one CUDA kernel). DS-V2 grouped routing (n_group > 1):
        falls back to _route() with ops.grouped_topk.
        """
        H = self.hidden_size
        hidden = residual + F.linear(attn_out, self.o_proj[layer])
        moe_residual_buf.copy_(hidden)
        hidden = F.rms_norm(hidden, (H,), self.post_attn_layernorm[layer],
                            self.rms_norm_eps)
        moe_input_buf.copy_(hidden)

        if self.router[layer] is None:
            # Dense layer: no routing. topk buffers stay zero (unused).
            return

        if self.n_group > 1:
            # DS-V2 grouped routing — topk_softmax doesn't support groups
            topk_weights, topk_ids = self._route(hidden, layer)
            topk_weights_buf.copy_(topk_weights)
            topk_ids_buf.copy_(topk_ids.to(torch.int32))
        else:
            # Fused softmax + topk + optional renormalize — writes directly
            # into topk_weights_buf and topk_ids_buf (no intermediate tensors).
            router_logits = F.linear(hidden, self.router[layer])
            _vllm_ops.topk_softmax(topk_weights_buf, topk_ids_buf,
                                   token_expert_indices,
                                   router_logits.float(), self.norm_topk_prob)
            if not self.norm_topk_prob and self.is_mla:
                topk_weights_buf.mul_(self.routed_scaling_factor)

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
        is_dense = self.is_mla and layer < self.first_k_dense_replace

        if is_dense:
            # Dense SwiGLU MLP
            gate_up = F.linear(moe_input_buf, self.dense_w1[layer])
            I = self.intermediate_size
            gate, up = gate_up[:, :I], gate_up[:, I:]
            hidden = F.linear(F.silu(gate) * up, self.dense_w2[layer])
            hidden_out.copy_(moe_residual_buf + hidden)
            return

        if self.offloading:
            w1, w2 = self.w1_buf, self.w2_buf
            expert_map = self.expert_map_buf
        elif self.ep_size > 1:
            w1, w2 = self.w1[layer], self.w2[layer]
            expert_map = self._ep_expert_map_gpu
        else:
            w1, w2 = self.w1[layer], self.w2[layer]
            expert_map = None
        routed = self._moe_experts(moe_input_buf, w1, w2,
                                   topk_weights_buf, topk_ids_buf, expert_map)

        # Shared experts (MLA only, non-dense layers)
        if self.is_mla and self.n_shared_experts > 0:
            shared_gate_up = F.linear(moe_input_buf, self.shared_w1[layer])
            shared_I = self.moe_intermediate_size * self.n_shared_experts
            sg, su = shared_gate_up[:, :shared_I], shared_gate_up[:, shared_I:]
            shared = F.linear(F.silu(sg) * su, self.shared_w2[layer])
            hidden_out.copy_(moe_residual_buf + routed + shared)
        else:
            hidden_out.copy_(moe_residual_buf + routed)

    def _full_mixed_graph_body(self, all_token_ids, positions, slot_mapping,
                               num_decode_tokens, decode_wrapper,
                               prefill_cu_seqlens, prefill_max_seqlen):
        """Full mixed forward — embed + layers (no final norm/lm_head).

        Designed for torch.compile(dynamic=True) to fuse RMSNorm + residual
        + RoPE between graph-breaking external kernels.
        Returns hidden states; caller applies selective norm + lm_head.
        """
        hidden = F.embedding(all_token_ids, self.embed_tokens)
        if self.is_mla:
            for layer in range(self.num_layers):
                hidden = self._layer_mixed_mla(
                    hidden, layer, positions, slot_mapping,
                    num_decode_tokens, prefill_cu_seqlens, prefill_max_seqlen)
        else:
            for layer in range(self.num_layers):
                hidden = self._layer_mixed(hidden, layer, positions, slot_mapping,
                                           num_decode_tokens, decode_wrapper,
                                           prefill_cu_seqlens, prefill_max_seqlen)
        return hidden

    def _ep_sync_graph_N(self, local_graph_N):
        """Sync graph_N to max across EP ranks."""
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

    @torch.no_grad()
    def step(self, decode_seq_ids, decode_token_ids,
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

        Output extraction (selective lm_head — only needed tokens get logits):
          - logits[:D] — decode tokens (one per sequence)
          - logits[D:D+num_prefill_seqs] — one row per prefill sequence (last token)
          - logits[D+num_prefill_seqs:] — one row per continuation sequence (last token)
          Each row is already the "next token prediction" logits.

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
            logits: Tensor [D + num_prefill_seqs + num_cont_seqs, vocab_size]
                logits[:D] = decode, then one row per prefill/continuation seq
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
        if self.ep_size > 1:
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
            return self._step_piecewise(
                decode_seq_ids, decode_token_ids,
                prefill_seq_ids, prefill_input_ids,
                continuation_seq_ids, continuation_input_ids,
                continuation_offsets, graph_N)

        if continuation_seq_ids:
            raise NotImplementedError(
                "Continuation prefill requires piecewise CUDA graphs. "
                f"Call capture_cuda_graphs() with sizes >= {N_total}.")

        if self.offload_engine or self.trace_recorder:
            raise RuntimeError(
                f"No piecewise CUDA graph covers {N_total} tokens. "
                f"Offload engine / trace recorder requires piecewise graphs. "
                f"Call capture_cuda_graphs() with sizes >= {N_total}.")

        if self.pp_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism requires piecewise CUDA graphs. "
                f"Call capture_cuda_graphs() with sizes >= {N_total}.")

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

        # ── Increment _seq_lens_cpu for decode BEFORE FlashInfer/MLA plan ──
        if D > 0:
            for sid in decode_seq_ids:
                self._seq_lens_cpu[sid] += 1
                self._num_computed_tokens_gpu[sid] += 1
            if self.is_mla:
                self._plan_mla_decode_for_subset(
                    self._mla_decode_wrapper, decode_seq_ids)
            else:
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
        hidden_all = self._full_mixed_graph_body(
            all_token_ids, positions, slot_mapping,
            D, self._decode_wrapper, prefill_cu, prefill_max)

        # ── Selective norm + lm_head ──
        logit_token_indices = list(range(D))
        offset = D
        for length in prefill_lengths:
            logit_token_indices.append(offset + length - 1)
            offset += length
        num_logit_tokens = len(logit_token_indices)
        if num_logit_tokens < hidden_all.shape[0]:
            idx = torch.tensor(logit_token_indices, dtype=torch.long,
                               device=hidden_all.device)
            hidden = hidden_all[idx]
        else:
            hidden = hidden_all
        hidden = F.rms_norm(hidden, (self.hidden_size,), self.final_norm,
                            self.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head)

        # ── Update GPU seq_lens ──
        for sid in decode_seq_ids:
            self.seq_lens[sid] += 1
        for sid, length in zip(prefill_seq_ids, prefill_lengths):
            self.seq_lens[sid] = length
            self._seq_lens_cpu[sid] = length
            self._num_computed_tokens_gpu[sid] = length

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

        For MLA models, stage 1 produces q_nope_buf, q_pe_buf, q_absorbed_buf,
        ckv_buf, kpe_buf instead of q_buf/k_buf/v_buf. Stage 2 output uses
        attn_latent_buf for the W_UV projection graph.
        """
        total_kv_slots = self.total_pages * self.page_size
        if self.is_mla:
            bufs = {
                'q_nope_buf': torch.zeros(N, self.num_heads, self.qk_nope_head_dim,
                                           dtype=self.dtype, device=device),
                'q_fused_buf': torch.zeros(N, self.num_heads,
                                            self.kv_lora_rank + self.qk_rope_head_dim,
                                            dtype=self.dtype, device=device),
                'attn_latent_buf': torch.zeros(N, self.num_heads, self.kv_lora_rank,
                                                dtype=self.dtype, device=device),
                'ckv_buf': torch.zeros(N, self.kv_lora_rank,
                                        dtype=self.dtype, device=device),
                'kpe_buf': torch.zeros(N, self.qk_rope_head_dim,
                                        dtype=self.dtype, device=device),
                'attn_out_buf': torch.zeros(N, self.num_heads * self.v_head_dim,
                                             dtype=self.dtype, device=device),
                'residual_buf': torch.zeros(N, self.hidden_size, dtype=self.dtype, device=device),
                'hidden_buf': torch.zeros(N, self.hidden_size, dtype=self.dtype, device=device),
                'moe_input_buf': torch.zeros(N, self.hidden_size, dtype=self.dtype, device=device),
                'moe_residual_buf': torch.zeros(N, self.hidden_size, dtype=self.dtype, device=device),
                'topk_weights_buf': torch.zeros(N, self.top_k, dtype=torch.float32, device=device),
                'topk_ids_buf': torch.zeros(N, self.top_k, dtype=torch.int32, device=device),
                'token_expert_indices': torch.zeros(N, self.top_k, dtype=torch.int32, device=device),
                'static_positions': (torch.arange(N, dtype=torch.int64, device=device)
                                     % self.max_seq_len),
                'static_slot_mapping': (torch.arange(N, dtype=torch.long, device=device)
                                        % total_kv_slots),
                'static_token_ids': torch.randint(1, 1000, (N,), device=device),
            }
            # q_absorbed_buf and q_pe_buf are views into q_fused_buf so that
            # stage1 CUDA graph .copy_() writes populate the fused layout and
            # stage2 can pass q_fused_buf directly to FlashMLA without torch.cat.
            bufs['q_absorbed_buf'] = bufs['q_fused_buf'][:, :, :self.kv_lora_rank]
            bufs['q_pe_buf'] = bufs['q_fused_buf'][:, :, self.kv_lora_rank:]
            return bufs
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
            'topk_ids_buf': torch.zeros(N, self.top_k, dtype=torch.int32,
                                        device=device),
            'token_expert_indices': torch.zeros(N, self.top_k, dtype=torch.int32,
                                                device=device),
            'static_positions': (torch.arange(N, dtype=torch.int64,
                                              device=device)
                                 % self.max_seq_len),
            'static_slot_mapping': (torch.arange(N, dtype=torch.long,
                                                 device=device)
                                    % total_kv_slots),
            'static_token_ids': torch.randint(1, 1000, (N,), device=device),
        }

    def capture_cuda_graphs(self, total_token_sizes=None,
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
        stage1_fn = (self._layer_stage1_mla if self.is_mla
                     else self._layer_stage1_pre_attn)
        stage4a_fn = self._layer_stage4a_router
        stage4b_fn = self._layer_stage4b_moe
        stage2_out_fn = self._layer_stage2_mla_out if self.is_mla else None
        final_fn = self._final_norm_lmhead
        if use_torch_compile:
            # Raise dynamo cache limits like vLLM does — default 8/64 is too
            # low for multi-layer models with varying tensor shapes.
            torch._dynamo.config.cache_size_limit = 2048
            torch._dynamo.config.accumulated_cache_size_limit = 8192
            stage1_fn = torch.compile(stage1_fn, fullgraph=False)
            stage4a_fn = torch.compile(stage4a_fn, fullgraph=False)
            stage4b_fn = torch.compile(stage4b_fn, fullgraph=False)
            if stage2_out_fn is not None:
                stage2_out_fn = torch.compile(stage2_out_fn, fullgraph=False)
            final_fn = torch.compile(final_fn, fullgraph=False)

        # Max N for which we capture the final norm+lm_head graph.
        # Beyond this, logits_buf memory becomes excessive (N*vocab*2B).
        _MAX_FINAL_GRAPH_N = 64

        def _call_stage1(stage1_fn, buf, layer):
            """Call stage1_fn with correct signature based on is_mla."""
            if self.is_mla:
                stage1_fn(buf['hidden_buf'],
                          buf['static_positions'],
                          buf['static_slot_mapping'], layer,
                          buf['q_nope_buf'], buf['q_fused_buf'],
                          buf['ckv_buf'], buf['kpe_buf'],
                          buf['residual_buf'])
            else:
                stage1_fn(buf['hidden_buf'],
                          buf['static_positions'],
                          buf['static_slot_mapping'], layer,
                          buf['q_buf'], buf['k_buf'],
                          buf['v_buf'], buf['residual_buf'])

        def _fill_attn_out_dummy(buf, N, layer=0):
            """Fill attn_out_buf with dummy data after stage1 (between stage1 and stage4a).
            For MLA: also fills attn_latent_buf and runs stage2_out to produce attn_out.
            Must pass correct layer for torch.compile warmup."""
            if self.is_mla:
                buf['attn_latent_buf'].copy_(buf['q_absorbed_buf'])
                stage2_out_fn(buf['attn_latent_buf'], layer, buf['attn_out_buf'])
            else:
                buf['attn_out_buf'].copy_(buf['q_buf'].reshape(N, -1))

        # ── Pre-allocate staging buffers for Triton kernel setup path ──
        max_graph_N = max(total_token_sizes)
        primary_dev = self.pp_devices[0] if self.pp_size > 1 else self.device
        self._idx_mapping_buf = torch.zeros(
            max_graph_N, dtype=torch.int32, device=primary_dev)
        self._idx_mapping_cpu = torch.zeros(
            max_graph_N, dtype=torch.int32).pin_memory()
        self._query_start_loc_buf = torch.zeros(
            max_graph_N + 1, dtype=torch.int32, device=primary_dev)
        self._query_start_loc_cpu = torch.zeros(
            max_graph_N + 1, dtype=torch.int32).pin_memory()
        self._seq_lens_gpu_scratch = torch.zeros(
            self.max_seqs, dtype=torch.int32, device=primary_dev)
        # Pre-computed arange for pure-decode query_start_loc:
        # Each decode request = 1 token, so qsl = [0, 1, 2, ..., D].
        # Slice without any CPU work or H2D copy on the hot path.
        self._decode_qsl_gpu = torch.arange(
            max_graph_N + 1, dtype=torch.int32, device=primary_dev)

        # Pre-allocate FlashMLA setup buffers (Phase 5 optimization)
        if self.is_mla and _HAS_FLASHMLA:
            self._flashmla_sid_cpu = torch.zeros(
                max_graph_N, dtype=torch.long).pin_memory()
            self._flashmla_sid_gpu = torch.zeros(
                max_graph_N, dtype=torch.long, device=primary_dev)
            self._flashmla_sl_staging = torch.zeros(
                max_graph_N, dtype=torch.int32).pin_memory()

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
                            _call_stage1(stage1_fn, buf, layer)
                            _fill_attn_out_dummy(buf, N, layer)
                            stage4a_fn(buf['attn_out_buf'],
                                       buf['residual_buf'], layer,
                                       buf['moe_input_buf'],
                                       buf['moe_residual_buf'],
                                       buf['topk_weights_buf'],
                                       buf['topk_ids_buf'],
                                       buf['token_expert_indices'])
                            stage4b_fn(buf['moe_input_buf'],
                                       buf['moe_residual_buf'],
                                       buf['topk_weights_buf'],
                                       buf['topk_ids_buf'],
                                       buf['hidden_buf'], layer)
                # Warmup final graph on last GPU
                if N <= _MAX_FINAL_GRAPH_N:
                    last_gpu = self.pp_layer_gpu[self.num_layers - 1]
                    last_dev = self.pp_devices[last_gpu]
                    last_buf = pp_bufs[last_gpu]
                    logits_buf = torch.zeros(N, self.vocab_size,
                                             dtype=self.dtype, device=last_dev)
                    with torch.cuda.device(last_dev):
                        for _ in range(n_warmup):
                            final_fn(last_buf['hidden_buf'], logits_buf)
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
                stage2_out_graphs = []
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
                            _call_stage1(stage1_fn, buf, layer)
                        stage1_graphs.append(g1)

                    if self.is_mla:
                        buf['attn_latent_buf'].copy_(buf['q_absorbed_buf'])
                        with torch.cuda.device(dev):
                            g2out = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g2out,
                                                  pool=graph_pools[gpu_idx],
                                                  stream=stream):
                                stage2_out_fn(buf['attn_latent_buf'], layer,
                                              buf['attn_out_buf'])
                            stage2_out_graphs.append(g2out)
                    else:
                        _fill_attn_out_dummy(buf, N, layer)

                    with torch.cuda.device(dev):
                        g4a = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g4a, pool=graph_pools[gpu_idx],
                                              stream=stream):
                            stage4a_fn(buf['attn_out_buf'],
                                       buf['residual_buf'], layer,
                                       buf['moe_input_buf'],
                                       buf['moe_residual_buf'],
                                       buf['topk_weights_buf'],
                                       buf['topk_ids_buf'],
                                       buf['token_expert_indices'])
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

                # Capture final graph on last GPU
                if N <= _MAX_FINAL_GRAPH_N:
                    last_gpu = self.pp_layer_gpu[self.num_layers - 1]
                    last_dev = self.pp_devices[last_gpu]
                    stream = pp_streams[last_gpu]
                    with torch.cuda.device(last_dev):
                        g_final = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g_final,
                                              pool=graph_pools[last_gpu],
                                              stream=stream):
                            final_fn(pp_bufs[last_gpu]['hidden_buf'],
                                     logits_buf)

                graph_dict = {
                    'stage1_graphs': stage1_graphs,
                    'stage2_out_graphs': stage2_out_graphs,
                    'stage4a_graphs': stage4a_graphs,
                    'stage4b_graphs': stage4b_graphs,
                    'pp_bufs': pp_bufs,
                }
                if N <= _MAX_FINAL_GRAPH_N:
                    graph_dict['final_graph'] = g_final
                    graph_dict['logits_buf'] = logits_buf
                if self.is_mla:
                    # Save MLA buffers per GPU to prevent GC
                    graph_dict['mla_buffers'] = {
                        gpu_idx: {
                            'q_nope_buf': pp_bufs[gpu_idx]['q_nope_buf'],
                            'q_fused_buf': pp_bufs[gpu_idx]['q_fused_buf'],
                            'q_pe_buf': pp_bufs[gpu_idx]['q_pe_buf'],
                            'q_absorbed_buf': pp_bufs[gpu_idx]['q_absorbed_buf'],
                            'attn_latent_buf': pp_bufs[gpu_idx]['attn_latent_buf'],
                            'ckv_buf': pp_bufs[gpu_idx]['ckv_buf'],
                            'kpe_buf': pp_bufs[gpu_idx]['kpe_buf'],
                        }
                        for gpu_idx in range(self.pp_size)
                    }
                    # FlashMLA: per-GPU sched_meta + block_table + seqlens
                    if _HAS_FLASHMLA:
                        mla_hd = self.kv_lora_rank + self.qk_rope_head_dim
                        pp_flashmla = {}
                        for gpu_idx in range(self.pp_size):
                            dev = self.pp_devices[gpu_idx]
                            sm, _ = get_mla_metadata()
                            bt = torch.zeros(N, self.max_pages_per_seq,
                                             dtype=torch.int32, device=dev)
                            sl = torch.zeros(N, dtype=torch.int32, device=dev)
                            _q_init = torch.zeros(
                                N, 1, self.num_heads, mla_hd,
                                dtype=self.dtype, device=dev)
                            flash_mla_with_kvcache(
                                q=_q_init,
                                k_cache=self.mla_cache[0].to(dev),
                                block_table=bt, cache_seqlens=sl,
                                head_dim_v=self.kv_lora_rank,
                                tile_scheduler_metadata=sm,
                                softmax_scale=self._mla_sm_scale,
                                causal=True)
                            del _q_init
                            pp_flashmla[gpu_idx] = {
                                'sched_meta': sm,
                                'block_table': bt,
                                'seqlens': sl,
                            }
                        graph_dict['pp_flashmla'] = pp_flashmla
                self._piecewise_graphs[N] = graph_dict

                n_graphs = self.num_layers * (4 if self.is_mla else 3)
                if N <= _MAX_FINAL_GRAPH_N:
                    n_graphs += 1
                compile_str = " + torch.compile" if use_torch_compile else ""
                print(f"  Piecewise CUDA graphs{compile_str} captured for "
                      f"N={N} ({n_graphs} graphs, "
                      f"PP={self.pp_size})")
            else:
                # Single-GPU path (original)
                bufs = self._create_intermediate_buffers(N, self.device)

                # EP: pre-allocate dispatch/combine buffers
                if self.ep_size > 1:
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

                # Allocate logits_buf for final graph (outside bufs dict)
                logits_buf = None
                if N <= _MAX_FINAL_GRAPH_N:
                    logits_buf = torch.zeros(N, self.vocab_size,
                                             dtype=self.dtype,
                                             device=self.device)

                # ── Warmup all layers ──
                n_warmup = 5 if use_torch_compile else 3
                for _ in range(n_warmup):
                    bufs['hidden_buf'].copy_(
                        F.embedding(bufs['static_token_ids'],
                                    self.embed_tokens))
                    for layer in range(self.num_layers):
                        _call_stage1(stage1_fn, bufs, layer)
                        _fill_attn_out_dummy(bufs, N, layer)
                        stage4a_fn(bufs['attn_out_buf'],
                                   bufs['residual_buf'], layer,
                                   bufs['moe_input_buf'],
                                   bufs['moe_residual_buf'],
                                   bufs['topk_weights_buf'],
                                   bufs['topk_ids_buf'],
                                   bufs['token_expert_indices'])
                        if self.offloading and self.router[layer] is not None:
                            self.expert_map_buf.copy_(
                                self.expert_map_abs[layer])
                        stage4b_fn(bufs['moe_input_buf'],
                                   bufs['moe_residual_buf'],
                                   bufs['topk_weights_buf'],
                                   bufs['topk_ids_buf'],
                                   bufs['hidden_buf'], layer)
                    # Warmup final graph
                    if logits_buf is not None:
                        final_fn(bufs['hidden_buf'], logits_buf)
                torch.cuda.synchronize()

                # ── Capture per-layer graphs ──
                # Use explicit stream to avoid cross-device CUDA graph pool issues
                capture_stream = torch.cuda.Stream(device=self.device)
                stage1_graphs = []
                stage2_out_graphs = []
                stage4a_graphs = []
                stage4b_graphs = []

                bufs['hidden_buf'].copy_(
                    F.embedding(bufs['static_token_ids'],
                                self.embed_tokens))

                for layer in range(self.num_layers):
                    g1 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g1, pool=graph_pool,
                                          stream=capture_stream):
                        _call_stage1(stage1_fn, bufs, layer)
                    stage1_graphs.append(g1)

                    if self.is_mla:
                        bufs['attn_latent_buf'].copy_(
                            bufs['q_absorbed_buf'])
                        g2out = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g2out, pool=graph_pool,
                                              stream=capture_stream):
                            stage2_out_fn(bufs['attn_latent_buf'], layer,
                                          bufs['attn_out_buf'])
                        stage2_out_graphs.append(g2out)
                    else:
                        _fill_attn_out_dummy(bufs, N, layer)

                    g4a = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g4a, pool=graph_pool,
                                          stream=capture_stream):
                        stage4a_fn(bufs['attn_out_buf'],
                                   bufs['residual_buf'], layer,
                                   bufs['moe_input_buf'],
                                   bufs['moe_residual_buf'],
                                   bufs['topk_weights_buf'],
                                   bufs['topk_ids_buf'],
                                   bufs['token_expert_indices'])
                    stage4a_graphs.append(g4a)

                    if self.offloading and self.router[layer] is not None:
                        self.expert_map_buf.copy_(
                            self.expert_map_abs[layer])

                    _layer_is_moe = (not self.is_mla
                                     or layer >= self.first_k_dense_replace)
                    if self.ep_size > 1 and _layer_is_moe:
                        # EP: stage4b runs eagerly (NCCL not capturable)
                        stage4b_graphs.append(None)
                    else:
                        g4b = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g4b, pool=graph_pool,
                                              stream=capture_stream):
                            stage4b_fn(bufs['moe_input_buf'],
                                       bufs['moe_residual_buf'],
                                       bufs['topk_weights_buf'],
                                       bufs['topk_ids_buf'],
                                       bufs['hidden_buf'], layer)
                        stage4b_graphs.append(g4b)

                # Capture final norm+lm_head graph
                if logits_buf is not None:
                    g_final = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_final, pool=graph_pool,
                                          stream=capture_stream):
                        final_fn(bufs['hidden_buf'], logits_buf)

                # Store buffers directly in dict for single-GPU compat
                graph_dict = {
                    'stage1_graphs': stage1_graphs,
                    'stage2_out_graphs': stage2_out_graphs,
                    'stage4a_graphs': stage4a_graphs,
                    'stage4b_graphs': stage4b_graphs,
                    **bufs,
                }
                if logits_buf is not None:
                    graph_dict['final_graph'] = g_final
                    graph_dict['logits_buf'] = logits_buf
                if self.is_mla:
                    # Save MLA buffers explicitly to prevent GC
                    graph_dict['mla_buffers'] = {
                        'q_nope_buf': bufs['q_nope_buf'],
                        'q_fused_buf': bufs['q_fused_buf'],
                        'q_pe_buf': bufs['q_pe_buf'],
                        'q_absorbed_buf': bufs['q_absorbed_buf'],
                        'attn_latent_buf': bufs['attn_latent_buf'],
                        'ckv_buf': bufs['ckv_buf'],
                        'kpe_buf': bufs['kpe_buf'],
                    }
                    # FlashMLA: sched_meta + block_table + seqlens
                    if _HAS_FLASHMLA:
                        mla_hd = self.kv_lora_rank + self.qk_rope_head_dim
                        flashmla_sched_meta, _ = get_mla_metadata()
                        flashmla_block_table = torch.zeros(
                            N, self.max_pages_per_seq,
                            dtype=torch.int32, device=self.device)
                        flashmla_seqlens = torch.zeros(
                            N, dtype=torch.int32, device=self.device)
                        # Initialize sched_meta with one dummy call
                        _q_init = torch.zeros(
                            N, 1, self.num_heads, mla_hd,
                            dtype=self.dtype, device=self.device)
                        flash_mla_with_kvcache(
                            q=_q_init,
                            k_cache=self.mla_cache[0],
                            block_table=flashmla_block_table,
                            cache_seqlens=flashmla_seqlens,
                            head_dim_v=self.kv_lora_rank,
                            tile_scheduler_metadata=flashmla_sched_meta,
                            softmax_scale=self._mla_sm_scale,
                            causal=True)
                        del _q_init
                        graph_dict['flashmla_sched_meta'] = flashmla_sched_meta
                        graph_dict['flashmla_block_table'] = flashmla_block_table
                        graph_dict['flashmla_seqlens'] = flashmla_seqlens
                self._piecewise_graphs[N] = graph_dict

                n_graphs = self.num_layers * (4 if self.is_mla else 3)
                if logits_buf is not None:
                    n_graphs += 1
                compile_str = " + torch.compile" if use_torch_compile else ""
                print(f"  Piecewise CUDA graphs{compile_str} captured for "
                      f"N={N} ({n_graphs} graphs)")

        # EP: validate all ranks captured identical graph sizes
        if self.ep_size > 1:
            local_sizes = torch.tensor(
                sorted(self._piecewise_graphs.keys()),
                dtype=torch.int64, device=self.device)
            n_sizes = torch.tensor([len(local_sizes)], device=self.device)
            all_n = [torch.empty_like(n_sizes) for _ in range(self.ep_size)]
            dist.all_gather(all_n, n_sizes, group=self.ep_group)
            assert all(x.item() == n_sizes.item() for x in all_n), \
                "EP ranks captured different NUMBER of graph sizes"
            all_sizes = [torch.empty_like(local_sizes)
                         for _ in range(self.ep_size)]
            dist.all_gather(all_sizes, local_sizes, group=self.ep_group)
            for r, other in enumerate(all_sizes):
                assert torch.equal(local_sizes, other), (
                    f"EP rank {r} captured different graph sizes: "
                    f"{other.tolist()} vs local {local_sizes.tolist()}")

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
        before and after capture_cuda_graphs().
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
    def _step_piecewise(self, decode_seq_ids, decode_token_ids,
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
        primary_dev = self.pp_devices[0] if pp else self.device
        primary_buf = info['pp_bufs'][0] if pp else info

        if N_actual == 0:
            # EP dummy step: zero all tokens (padding only)
            primary_buf['static_token_ids'].zero_()
        elif P == 0 and C == 0 and D > 0:
            # Pure decode fast path: skip torch.cat
            primary_buf['static_token_ids'][:D].copy_(
                decode_token_ids.to(primary_dev))
            if D < graph_N:
                primary_buf['static_token_ids'][D:].zero_()
        else:
            # Mixed batch: cat as before
            token_parts = []
            if D > 0:
                token_parts.append(decode_token_ids.to(primary_dev))
            for ids in prefill_input_ids:
                token_parts.append(ids.to(primary_dev))
            for ids in continuation_input_ids:
                token_parts.append(ids.to(primary_dev))
            all_token_ids = torch.cat(token_parts)
            primary_buf['static_token_ids'][:N_actual].copy_(all_token_ids)
            if N_actual < graph_N:
                primary_buf['static_token_ids'][N_actual:].zero_()

        # PP: replicate token_ids to other GPUs
        if pp:
            for gpu_idx in range(1, self.pp_size):
                info['pp_bufs'][gpu_idx]['static_token_ids'].copy_(
                    primary_buf['static_token_ids'])

        # ── 2+3+4a. Positions + slot_mapping + seq_lens via Triton kernels ──
        N_reqs = D + len(prefill_seq_ids) + len(continuation_seq_ids)

        if N_reqs == 0:
            # EP dummy step: zero positions and slot_mapping, skip Triton
            primary_buf['static_positions'].zero_()
            primary_buf['static_slot_mapping'].fill_(-1)
            self._num_computed_tokens_gpu.copy_(self._seq_lens_cpu)
        else:
            # 2a. Build idx_mapping: [decode_sids | prefill_sids | cont_sids]
            if P == 0 and C == 0 and D <= 32:
                # Pure decode fast path: write seq_ids directly, skip numpy
                for i, sid in enumerate(decode_seq_ids):
                    self._idx_mapping_cpu[i] = sid
            else:
                seq_ids_all = decode_seq_ids + prefill_seq_ids + continuation_seq_ids
                idx_np = np.array(seq_ids_all, dtype=np.int32)
                self._idx_mapping_cpu[:N_reqs] = torch.from_numpy(idx_np)
            self._idx_mapping_buf[:N_reqs].copy_(
                self._idx_mapping_cpu[:N_reqs], non_blocking=True)

            # 2b. Build query_start_loc: prefix sum of per-request token counts
            if P == 0 and C == 0:
                # Pure decode fast path: each request = 1 token → qsl = arange
                setup_qsl = self._decode_qsl_gpu[:D + 1]
            else:
                # Mixed batch: compute cumulative token counts via numpy
                tokens_per_req = [1] * D + prefill_lengths + cont_lengths
                qsl_np = np.zeros(N_reqs + 1, dtype=np.int32)
                np.cumsum(tokens_per_req, out=qsl_np[1:])
                self._query_start_loc_cpu[:N_reqs + 1] = torch.from_numpy(qsl_np)
                self._query_start_loc_buf[:N_reqs + 1].copy_(
                    self._query_start_loc_cpu[:N_reqs + 1], non_blocking=True)
                setup_qsl = self._query_start_loc_buf[:N_reqs + 1]

            # Safety sync: GPU seq_lens ← CPU (belt-and-suspenders, ~1us)
            self._num_computed_tokens_gpu.copy_(self._seq_lens_cpu)

            # 2c. Compute positions + seq_lens on GPU
            prepare_pos_seq_lens(
                idx_mapping=self._idx_mapping_buf[:N_reqs],
                query_start_loc=setup_qsl,
                num_computed_tokens=self._num_computed_tokens_gpu,
                pos=primary_buf['static_positions'],
                seq_lens=self._seq_lens_gpu_scratch,
            )
            # Zero padding positions for safety
            if N_actual < graph_N:
                primary_buf['static_positions'][N_actual:].zero_()

            # 2d. Compute slot_mapping on GPU
            slot_2d = primary_buf['static_slot_mapping'].unsqueeze(0)
            _compute_slot_mappings_kernel[(1, N_reqs + 1)](
                N_actual,
                graph_N,  # max_num_tokens (pad slots [N_actual:] to PAD_ID)
                self._idx_mapping_buf[:N_reqs],
                setup_qsl,
                primary_buf['static_positions'],
                self._block_table_ptrs,
                self._block_table_strides,
                self._block_sizes_tensor,
                slot_2d,
                slot_2d.stride(0),
                0,  # cp_rank (no context parallelism)
                CP_SIZE=1, CP_INTERLEAVE=1,
                PAD_ID=-1, TRITON_BLOCK_SIZE=1024,
            )

        # PP: replicate positions + slot_mapping to other GPUs
        if pp:
            for gpu_idx in range(1, self.pp_size):
                b = info['pp_bufs'][gpu_idx]
                b['static_positions'].copy_(primary_buf['static_positions'])
                b['static_slot_mapping'].copy_(primary_buf['static_slot_mapping'])

        # ── 4. Increment decode seq_lens and plan FlashInfer / MLA ──
        # GPU mirror is kept in sync by the safety sync at step start
        # (self._num_computed_tokens_gpu.copy_(self._seq_lens_cpu)),
        # so we only need to update _seq_lens_cpu here.
        if D > 0:
            for sid in decode_seq_ids:
                self._seq_lens_cpu[sid] += 1
            if self.is_mla and _HAS_FLASHMLA:
                # FlashMLA: copy block_table + seqlens using pre-allocated buffers
                sid_buf = self._flashmla_sid_cpu
                for i, sid in enumerate(decode_seq_ids):
                    sid_buf[i] = sid
                sid_sl = sid_buf[:D]

                # Gather seqlens on CPU into pinned staging buffer
                torch.index_select(self._seq_lens_cpu, 0, sid_sl,
                                   out=self._flashmla_sl_staging[:D])

                if pp:
                    for gpu_idx in range(self.pp_size):
                        fm = info['pp_flashmla'][gpu_idx]
                        fm['seqlens'][:D].copy_(
                            self._flashmla_sl_staging[:D], non_blocking=True)
                        if D < graph_N:
                            fm['seqlens'][D:].zero_()
                        # Copy sid to GPU, gather block_table rows
                        self._flashmla_sid_gpu[:D].copy_(sid_sl, non_blocking=True)
                        torch.index_select(
                            self.block_table[gpu_idx], 0,
                            self._flashmla_sid_gpu[:D],
                            out=fm['block_table'][:D])
                        if D < graph_N:
                            fm['block_table'][D:].zero_()
                else:
                    # Seqlens: pinned H2D copy
                    info['flashmla_seqlens'][:D].copy_(
                        self._flashmla_sl_staging[:D], non_blocking=True)
                    if D < graph_N:
                        info['flashmla_seqlens'][D:].zero_()
                    # Sid to GPU + gather block_table via index_select
                    self._flashmla_sid_gpu[:D].copy_(sid_sl, non_blocking=True)
                    torch.index_select(
                        self.block_table, 0, self._flashmla_sid_gpu[:D],
                        out=info['flashmla_block_table'][:D])
                    if D < graph_N:
                        info['flashmla_block_table'][D:].zero_()
            elif self.is_mla:
                # Fallback: FlashInfer MLA plan (no FlashMLA available)
                if pp:
                    for gpu_idx in range(self.pp_size):
                        self._plan_mla_decode_for_subset(
                            self._mla_wrappers[gpu_idx], decode_seq_ids,
                            gpu_idx=gpu_idx)
                else:
                    self._plan_mla_decode_for_subset(
                        self._mla_decode_wrapper, decode_seq_ids)
            elif pp:
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
            self._num_computed_tokens_gpu[sid] = total_sl
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
        # MLA uses _mla_decode_wrapper planned per-layer inside the loop (step 6b).
        cont_prefill_wrappers = None
        if C > 0 and not self.is_mla:
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

            if self.is_mla:
                # MLA: Stage 2 (decode) using FlashMLA + Stage 3a (prefill)
                # FlashMLA runs on all graph_N tokens (padded batch); padding
                # tokens have cache_seqlens=0 and produce exact zero output.
                if D > 0 and _HAS_FLASHMLA:
                    if _nvtx: torch.cuda.nvtx.range_push("stage2_mla")
                    # q_fused_buf is already [graph_N, H, 576] with q_absorbed
                    # and q_pe as contiguous views — no torch.cat needed
                    q_fused = buf['q_fused_buf'].unsqueeze(1)  # [graph_N, 1, H, 576]
                    if pp:
                        gpu_idx = self.pp_layer_gpu[layer]
                        fm = info['pp_flashmla'][gpu_idx]
                        o, _lse = flash_mla_with_kvcache(
                            q=q_fused,
                            k_cache=self.mla_cache[layer],
                            block_table=fm['block_table'],
                            cache_seqlens=fm['seqlens'],
                            head_dim_v=self.kv_lora_rank,
                            tile_scheduler_metadata=fm['sched_meta'],
                            softmax_scale=self._mla_sm_scale,
                            causal=True)
                    else:
                        o, _lse = flash_mla_with_kvcache(
                            q=q_fused,
                            k_cache=self.mla_cache[layer],
                            block_table=info['flashmla_block_table'],
                            cache_seqlens=info['flashmla_seqlens'],
                            head_dim_v=self.kv_lora_rank,
                            tile_scheduler_metadata=info['flashmla_sched_meta'],
                            softmax_scale=self._mla_sm_scale,
                            causal=True)
                    # o: [graph_N, 1, H, 512] -> squeeze -> [graph_N, H, 512]
                    buf['attn_latent_buf'][:D].copy_(o.squeeze(1)[:D])
                    if _nvtx: torch.cuda.nvtx.range_pop()  # stage2_mla
                elif D > 0:
                    # Fallback: FlashInfer MLA plan+run
                    if _nvtx: torch.cuda.nvtx.range_push("stage2_mla")
                    if pp:
                        gpu_idx = self.pp_layer_gpu[layer]
                        mla_wrapper = self._mla_wrappers[gpu_idx]
                    else:
                        mla_wrapper = self._mla_decode_wrapper
                    q_absorbed_sl = buf['q_absorbed_buf'][:D]
                    q_pe_sl = buf['q_pe_buf'][:D]
                    attn_latent = mla_wrapper.run(
                        q_absorbed_sl, q_pe_sl,
                        self.ckv_cache[layer], self.kpe_cache[layer])
                    buf['attn_latent_buf'][:D].copy_(attn_latent)
                    if _nvtx: torch.cuda.nvtx.range_pop()  # stage2_mla
                # Stage 2 output: W_UV projection (CUDA graph)
                if D > 0:
                    info['stage2_out_graphs'][layer].replay()

                # Stage 3a: MLA prefill (FA3 decompressed) for new prefills
                if P > 0:
                    if _nvtx: torch.cuda.nvtx.range_push("stage3a_mla")
                    cu = (prefill_cu[self.pp_layer_gpu[layer]]
                          if pp else prefill_cu_single)
                    q_nope_pf = buf['q_nope_buf'][D:D + P]
                    q_pe_pf = buf['q_pe_buf'][D:D + P]
                    c_kv_pf = buf['ckv_buf'][D:D + P]
                    k_pe_pf = buf['kpe_buf'][D:D + P]

                    kv_decompressed = F.linear(c_kv_pf, self.kv_b_proj[layer])
                    kv_reshaped = kv_decompressed.view(P, self.num_heads,
                                                        self.qk_nope_head_dim + self.v_head_dim)
                    k_nope_pf = kv_reshaped[:, :, :self.qk_nope_head_dim]
                    v_pf = kv_reshaped[:, :, self.qk_nope_head_dim:]
                    k_pe_expanded = k_pe_pf.unsqueeze(1).expand(-1, self.num_heads, -1)
                    k_full = torch.cat([k_nope_pf, k_pe_expanded], dim=-1)
                    q_full_pf = torch.cat([q_nope_pf, q_pe_pf], dim=-1)
                    pf_out = flash_attn_varlen_func(
                        q_full_pf.contiguous(), k_full.contiguous(), v_pf.contiguous(),
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=prefill_max,
                        max_seqlen_k=prefill_max,
                        softmax_scale=self._mla_sm_scale,
                        causal=True, fa_version=3)
                    attn_out_buf[D:D + P].copy_(
                        pf_out.reshape(P, self.num_heads * self.v_head_dim))
                    if _nvtx: torch.cuda.nvtx.range_pop()  # stage3a_mla

                # Stage 3b: MLA continuation prefill — absorbed path via
                # BatchMLAPagedAttentionWrapper with multi-token qo_indptr.
                # Attends to full paged KV cache (prior chunks + current chunk).
                if C > 0:
                    if _nvtx: torch.cuda.nvtx.range_push("stage3b_mla")
                    # q_absorbed already computed by stage1 graph
                    q_absorbed_cont = buf['q_absorbed_buf'][D + P:D + P + C]
                    q_pe_cont = buf['q_pe_buf'][D + P:D + P + C]       # [C, H, R]

                    # Plan MLA wrapper for continuation (overwrites decode plan)
                    if pp:
                        gpu_idx_c = self.pp_layer_gpu[layer]
                        wrapper_c = self._mla_wrappers[gpu_idx_c]
                        self._plan_mla_continuation(
                            wrapper_c, continuation_seq_ids,
                            cont_lengths, cont_total_seq_lens,
                            gpu_idx=gpu_idx_c)
                    else:
                        wrapper_c = self._mla_decode_wrapper
                        self._plan_mla_continuation(
                            wrapper_c, continuation_seq_ids,
                            cont_lengths, cont_total_seq_lens)

                    attn_latent_cont = wrapper_c.run(
                        q_absorbed_cont, q_pe_cont,
                        self.ckv_cache[layer], self.kpe_cache[layer])
                    # attn_latent_cont: [C, H, kv_lora_rank=512]
                    attn_v_cont = torch.einsum(
                        'bhc,hcv->bhv', attn_latent_cont, self.W_UV[layer])
                    attn_out_buf[D + P:D + P + C].copy_(
                        attn_v_cont.reshape(C, self.num_heads * self.v_head_dim))

                    # If also decode tokens, restore decode plan for next layer
                    # (FlashMLA doesn't need plan restoration — metadata is
                    # set once in setup and stays valid across all layers)
                    if D > 0 and not _HAS_FLASHMLA:
                        if pp:
                            for _gidx in range(self.pp_size):
                                self._plan_mla_decode_for_subset(
                                    self._mla_wrappers[_gidx],
                                    decode_seq_ids, gpu_idx=_gidx)
                        else:
                            self._plan_mla_decode_for_subset(
                                self._mla_decode_wrapper, decode_seq_ids)
                    if _nvtx: torch.cuda.nvtx.range_pop()  # stage3b_mla
            else:
                # Standard (non-MLA) attention path
                q_buf = buf['q_buf']
                k_buf = buf['k_buf']
                v_buf = buf['v_buf']

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
                if self.trace_recorder and self.router[layer] is not None:
                    self.trace_recorder.process_layer(
                        layer, buf['topk_ids_buf'], N_actual,
                        router_input_buf=buf['moe_input_buf'])
            elif self.trace_recorder:
                if self.router[layer] is not None:
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

            # Stage 4b: MoE
            if _nvtx: torch.cuda.nvtx.range_push("stage4b")
            _layer_is_moe = (not self.is_mla
                             or layer >= self.first_k_dense_replace)
            if self.ep_size > 1 and _layer_is_moe:
                # EP: AllGather → local fused_experts → ReduceScatter
                gathered_h = ep_allgather(
                    buf['moe_input_buf'], self.ep_group, self.ep_size,
                    out=buf['ep_gathered_hidden'])
                gathered_w = ep_allgather(
                    buf['topk_weights_buf'], self.ep_group, self.ep_size,
                    out=buf['ep_gathered_weights'])
                gathered_ids = ep_allgather(
                    buf['topk_ids_buf'], self.ep_group, self.ep_size,
                    out=buf['ep_gathered_ids'])

                w1, w2 = self.w1[layer], self.w2[layer]
                partial_out = self._moe_experts(
                    gathered_h, w1, w2, gathered_w, gathered_ids,
                    self._ep_expert_map_gpu)

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
                # Original: stage4b CUDA graph replay
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
        hidden_all = last_buf['hidden_buf']

        if P == 0 and C == 0 and 'final_graph' in info:
            # Pure decode — use captured final graph (zero dispatch overhead)
            info['final_graph'].replay()
            logits = info['logits_buf'][:N_actual]
            if pp:
                logits = logits.to(self.pp_devices[0])
        else:
            # Mixed batch — eager selective lm_head
            # Only compute lm_head for tokens that need logits: all decode
            # tokens, plus the last token of each prefill/continuation seq.
            logit_token_indices = list(range(D))  # all decode tokens
            offset = D
            for length in prefill_lengths:
                logit_token_indices.append(offset + length - 1)
                offset += length
            for length in cont_lengths:
                logit_token_indices.append(offset + length - 1)
                offset += length
            num_logit_tokens = len(logit_token_indices)

            if num_logit_tokens < N_actual:
                idx = torch.tensor(logit_token_indices, dtype=torch.long,
                                   device=hidden_all.device)
                hidden = hidden_all[idx]
                hidden = F.rms_norm(hidden, (self.hidden_size,),
                                    self.final_norm, self.rms_norm_eps)
                logits = F.linear(hidden, self.lm_head)
            else:
                hidden = hidden_all[:N_actual]
                hidden = F.rms_norm(hidden, (self.hidden_size,),
                                    self.final_norm, self.rms_norm_eps)
                logits = F.linear(hidden, self.lm_head)
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
            self._num_computed_tokens_gpu[sid] = length
        # _seq_lens_cpu for continuations already set in step 4b

        # When selective lm_head was used, logits is already the right size:
        # [D + num_prefill_seqs + num_cont_seqs, vocab_size]
        # Layout: logits[:D] = decode, then one row per prefill seq, one per cont seq.
        if P > 0 or C > 0:
            num_logit_tokens = D + len(prefill_lengths) + len(cont_lengths)
            if num_logit_tokens < N_actual:
                return logits
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
        self._num_computed_tokens_gpu[:B] = 0

        logits = self.prefill(input_ids)  # [B, vocab_size]
        next_token = logits.argmax(dim=-1)
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

    def fill_kv_random(self, std=0.01):
        """Fill KV caches with random data (benchmarking warmup)."""
        if self.is_mla:
            for l in range(self.num_layers):
                self.ckv_flat[l].normal_(0, std)
                self.kpe_flat[l].normal_(0, std)
        else:
            for l in range(self.num_layers):
                self.k_cache[l].normal_(0, std)
                self.v_cache[l].normal_(0, std)

    def kv_snapshot(self):
        """Snapshot KV caches for later restoration."""
        if self.is_mla:
            return ([c.clone() for c in self.ckv_flat],
                    [k.clone() for k in self.kpe_flat])
        return ([kc.clone() for kc in self.k_cache],
                [vc.clone() for vc in self.v_cache])

    def kv_restore(self, snapshot):
        """Restore KV caches from a snapshot created by kv_snapshot()."""
        a, b = snapshot
        if self.is_mla:
            for c, s in zip(self.ckv_flat, a):
                c.copy_(s)
            for k, s in zip(self.kpe_flat, b):
                k.copy_(s)
        else:
            for kc, s in zip(self.k_cache, a):
                kc.copy_(s)
            for vc, s in zip(self.v_cache, b):
                vc.copy_(s)

    def reset(self):
        """Reset KV cache and sequence state."""
        if self.pp_size > 1:
            for sl in self.seq_lens:
                sl.zero_()
        else:
            self.seq_lens.zero_()
        self._seq_lens_cpu.zero_()
        if hasattr(self, '_num_computed_tokens_gpu'):
            self._num_computed_tokens_gpu.zero_()
        if self.is_mla:
            for l in range(self.num_layers):
                self.ckv_flat[l].zero_()
                self.kpe_flat[l].zero_()
        else:
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
        if hasattr(self, '_num_computed_tokens_gpu'):
            self._num_computed_tokens_gpu[seq_id] = 0
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

        Called by the scheduler before step() for every sequence
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

    def capture_decode_cuda_graph(self, batch_size, warmup_seq_len=128,
                                    max_decode_tokens=256, use_torch_compile=None):
        """Capture CUDA graphs for decode — delegates to capture_cuda_graphs.

        Retained for backward compatibility. Internally captures piecewise
        per-layer graphs that handle both decode and mixed batches.
        warmup_seq_len and max_decode_tokens are ignored (not needed for
        piecewise graphs where attention runs eagerly outside the graph).

        Args:
            batch_size: Fixed batch size for this graph (becomes total_token_size)
            warmup_seq_len: Ignored (kept for API compat)
            max_decode_tokens: Ignored (kept for API compat)
            use_torch_compile: Override instance default (self.use_torch_compile)
        """
        self.capture_cuda_graphs(
            total_token_sizes=[batch_size],
            use_torch_compile=use_torch_compile)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/OLMoE-1B-7B"
    engine = MoEEngine(model_path, max_seqs=4, max_seq_len=512,
                       use_torch_compile=False)

    # Capture piecewise graphs (handles both prefill and decode)
    engine.capture_cuda_graphs(
        total_token_sizes=[1, 4, 128, 256, 512],
        use_torch_compile=False)

    # Quick smoke test
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    print("\nSmoke test: prefill...")
    logits = engine.prefill(input_ids)
    print(f"  Prefill logits shape: {logits.shape}")
    print(f"  Top token: {logits[0].argmax().item()}")

    print("Smoke test: generate...")
    engine.reset()
    tokens = engine.generate(input_ids, max_new_tokens=10)
    print(f"  Generated shape: {tokens.shape}")
    print(f"  Tokens: {tokens[0].tolist()}")
