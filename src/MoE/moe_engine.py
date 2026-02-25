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
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


def rope_pytorch(q, k, cos_sin_cache, positions, num_heads, head_dim):
    """Pure PyTorch NeoX-style RoPE — fully compilable by torch.compile.

    Args:
        q, k: [N, num_heads * head_dim] (flat, after Q/K norm)
        cos_sin_cache: [max_seq, head_dim] (first half cos, second half sin)
        positions: [N] int32/int64
        num_heads: number of attention heads
        head_dim: dimension per head
    Returns:
        q_rot, k_rot: [N, num_heads * head_dim] (same shape as input)
    """
    orig_dtype = q.dtype
    N = q.shape[0]
    half = head_dim // 2
    cos = cos_sin_cache[positions.long(), :half].unsqueeze(1)   # [N, 1, half]
    sin = cos_sin_cache[positions.long(), half:].unsqueeze(1)   # [N, 1, half]
    q_h = q.float().view(N, num_heads, head_dim)
    k_h = k.float().view(N, num_heads, head_dim)
    q_rot = torch.cat([q_h[..., :half] * cos - q_h[..., half:] * sin,
                        q_h[..., :half] * sin + q_h[..., half:] * cos], dim=-1)
    k_rot = torch.cat([k_h[..., :half] * cos - k_h[..., half:] * sin,
                        k_h[..., :half] * sin + k_h[..., half:] * cos], dim=-1)
    return q_rot.reshape(N, -1).to(orig_dtype), k_rot.reshape(N, -1).to(orig_dtype)

# ── Patch vLLM's broken _moe_C ops (requires glibc 2.29, we have 2.28) ──
# Load _C.so (works fine — has silu_and_mul, etc.)
import vllm._custom_ops as _vllm_ops
_vllm_so = Path(_vllm_ops.__file__).parent / "_C.abi3.so"
if _vllm_so.exists():
    torch.ops.load_library(str(_vllm_so))


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


# Apply monkey-patches
_vllm_ops.moe_align_block_size = _moe_align_block_size_torch
_vllm_ops.moe_sum = _moe_sum_torch
_vllm_ops.topk_softmax = _topk_softmax_torch
# Also patch the module-level reference in moe_align_block_size.py
import vllm.model_executor.layers.fused_moe.moe_align_block_size as _mabs_mod
_mabs_mod.ops.moe_align_block_size = _moe_align_block_size_torch
# Patch the module-level reference in fused_moe.py (used by vllm_topk_softmax)
import vllm.model_executor.layers.fused_moe.fused_moe as _fused_moe_mod
_fused_moe_mod.ops.topk_softmax = _topk_softmax_torch
_fused_moe_mod.ops.moe_sum = _moe_sum_torch

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


class MoEEngine:
    """Custom MoE inference engine — FlashInfer decode + torch.compile."""

    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        page_size: int = 16,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_torch_compile: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        self.use_torch_compile = use_torch_compile

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
        self.norm_topk_prob = cfg.get("norm_topk_prob", False)

        # Unsupported features — raise early rather than produce wrong results
        if cfg.get("sliding_window") is not None:
            raise NotImplementedError(
                f"Sliding window attention (window={cfg['sliding_window']}) is not "
                f"yet implemented.")
        if cfg.get("rope_scaling") is not None:
            raise NotImplementedError(
                f"RoPE scaling ({cfg['rope_scaling']}) is not yet implemented.")

        # Load weights
        print("Loading weights...")
        self._load_weights(model_path)

        # RoPE cos/sin cache (must be float32)
        self.cos_sin_cache = self._build_rope_cache().to(device)

        # Attention scale and KV quantization scales
        self._attn_scale = 1.0 / math.sqrt(self.head_dim)
        self._k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

        # KV cache — flat NHD layout (compatible with reshape_and_cache_flash + flash_attn)
        # key/value_cache: [L, num_blocks, block_size, num_kv_heads, head_dim]
        self.max_pages_per_seq = math.ceil(max_seq_len / page_size)
        self.total_pages = max_batch_size * self.max_pages_per_seq
        self.k_cache = torch.zeros(
            self.num_layers, self.total_pages, page_size,
            self.num_kv_heads, self.head_dim,
            dtype=dtype, device=device)
        self.v_cache = torch.zeros(
            self.num_layers, self.total_pages, page_size,
            self.num_kv_heads, self.head_dim,
            dtype=dtype, device=device)

        # Block table: contiguous page allocation per sequence
        self.block_table = torch.zeros(
            max_batch_size, self.max_pages_per_seq, dtype=torch.int32, device=device)
        for i in range(max_batch_size):
            self.block_table[i] = torch.arange(
                i * self.max_pages_per_seq,
                (i + 1) * self.max_pages_per_seq,
                dtype=torch.int32, device=device)

        # Sequence length tracker
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        self._seq_lens_cpu = torch.zeros(max_batch_size, dtype=torch.int32)

        # FlashInfer decode workspace (128 MB) and non-graph wrapper
        self._workspace_buf = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self._workspace_buf, kv_layout="NHD", use_cuda_graph=False)

        # Guard: check model fits on GPU
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if mem_gb > gpu_total_gb * 0.95:
            raise NotImplementedError(
                f"Model requires {mem_gb:.1f} GB but GPU has {gpu_total_gb:.0f} GB. "
                f"Expert offloading is not yet implemented.")

        model_type = cfg.get("model_type", "unknown")
        print(f"MoEEngine ready ({model_type}): {self.num_layers}L, "
              f"{self.num_experts}E (top-{self.top_k}), "
              f"hidden={self.hidden_size}, heads={self.num_heads}, "
              f"qk_norm={self.has_qk_norm}, norm_topk={self.norm_topk_prob}, "
              f"torch_compile={use_torch_compile}")

    # ── Weight Loading ───────────────────────────────────────────────

    def _load_weights(self, model_path: str):
        model_path = Path(model_path)
        weights = {}
        for shard in sorted(model_path.glob("model-*.safetensors")):
            with safe_open(str(shard), framework="pt", device=self.device) as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key).to(self.dtype)

        # Global weights
        self.embed_tokens = weights.pop("model.embed_tokens.weight")
        self.final_norm = weights.pop("model.norm.weight")
        self.lm_head = weights.pop("lm_head.weight")

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
        self.w1 = []
        self.w2 = []

        # Detect Q/K norm from first layer's weights
        self.has_qk_norm = f"model.layers.0.self_attn.q_norm.weight" in weights

        for l in range(self.num_layers):
            p = f"model.layers.{l}"
            self.input_layernorm.append(weights.pop(f"{p}.input_layernorm.weight"))
            self.post_attn_layernorm.append(weights.pop(f"{p}.post_attention_layernorm.weight"))
            self.q_proj.append(weights.pop(f"{p}.self_attn.q_proj.weight"))
            self.k_proj.append(weights.pop(f"{p}.self_attn.k_proj.weight"))
            self.v_proj.append(weights.pop(f"{p}.self_attn.v_proj.weight"))
            self.o_proj.append(weights.pop(f"{p}.self_attn.o_proj.weight"))
            if self.has_qk_norm:
                self.q_norm.append(weights.pop(f"{p}.self_attn.q_norm.weight"))
                self.k_norm.append(weights.pop(f"{p}.self_attn.k_norm.weight"))
            # Router key: OLMoE uses "mlp.gate", Mixtral uses "block_sparse_moe.gate"
            router_key = f"{p}.mlp.gate.weight"
            if router_key not in weights:
                router_key = f"{p}.block_sparse_moe.gate.weight"
            self.router.append(weights.pop(router_key))

            # Fuse expert weights: w1 = cat(gate, up) [E, 2*I, H], w2 = down [E, H, I]
            # Key format: OLMoE: "mlp.experts.{e}", Mixtral: "block_sparse_moe.experts.{e}"
            w1_list, w2_list = [], []
            for e in range(self.num_experts):
                expert_prefix = f"{p}.mlp.experts.{e}"
                if f"{expert_prefix}.gate_proj.weight" not in weights:
                    expert_prefix = f"{p}.block_sparse_moe.experts.{e}"
                gate = weights.pop(f"{expert_prefix}.gate_proj.weight")
                up = weights.pop(f"{expert_prefix}.up_proj.weight")
                down = weights.pop(f"{expert_prefix}.down_proj.weight")
                w1_list.append(torch.cat([gate, up], dim=0))
                w2_list.append(down)
            self.w1.append(torch.stack(w1_list))
            self.w2.append(torch.stack(w2_list))
            print(f"  Layer {l}: w1 {self.w1[-1].shape}, w2 {self.w2[-1].shape}")

        del weights
        torch.cuda.empty_cache()

        # Shape verification (before stacking)
        assert self.embed_tokens.shape == (self.vocab_size, self.hidden_size)
        assert self.lm_head.shape == (self.vocab_size, self.hidden_size)
        assert self.w1[0].shape == (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        assert self.w2[0].shape == (self.num_experts, self.hidden_size, self.intermediate_size)

        # Stack per-layer weights into single tensors for indexed access.
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
        self.w1 = torch.stack(self.w1)            # [L, E, 2*I, H]
        self.w2 = torch.stack(self.w2)            # [L, E, H, I]

    def _build_rope_cache(self) -> torch.Tensor:
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # [max_seq, head_dim]

    # ── Prefill ──────────────────────────────────────────────────────

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill forward pass.
        Args: input_ids [B, S]
        Returns: logits [B, S, vocab_size]
        """
        B, S = input_ids.shape
        hidden = F.embedding(input_ids, self.embed_tokens)
        positions_flat = torch.arange(S, dtype=torch.int32, device=self.device).repeat(B)

        for layer in range(self.num_layers):
            hidden = self._layer_prefill(hidden, layer, positions_flat)

        hidden = F.rms_norm(hidden, (self.hidden_size,), self.final_norm, self.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head)
        self.seq_lens[:B] = S
        self._seq_lens_cpu[:B] = S
        return logits

    def _layer_prefill(self, hidden, layer, positions_flat):
        B, S, H = hidden.shape

        residual = hidden
        hidden = F.rms_norm(hidden, (H,), self.input_layernorm[layer], self.rms_norm_eps)

        # Fused QKV projection (single GEMM)
        kv_dim = self.num_kv_heads * self.head_dim
        qkv = F.linear(hidden, self.qkv_proj[layer])
        q, k, v = qkv.split([H, kv_dim, kv_dim], dim=-1)
        v = v.contiguous()

        # Q/K norm (flat, before head reshape, before RoPE) — OLMoE only
        if self.has_qk_norm:
            q = F.rms_norm(q.reshape(-1, H), (H,), self.q_norm[layer], self.rms_norm_eps).contiguous()
            k = F.rms_norm(k.reshape(-1, kv_dim), (kv_dim,), self.k_norm[layer], self.rms_norm_eps).contiguous()

        # RoPE — use FlashInfer for prefill if available, else PyTorch
        if HAS_FLASHINFER:
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions_flat, query=q, key=k,
                head_size=self.head_dim, cos_sin_cache=self.cos_sin_cache, is_neox=True)
        else:
            q, k = rope_pytorch(q, k, self.cos_sin_cache, positions_flat,
                                self.num_heads, self.head_dim)

        # Write K, V to paged cache for subsequent decode
        k_for_cache = k.view(B, S, self.num_kv_heads, self.head_dim)
        v_for_cache = v.view(B, S, self.num_kv_heads, self.head_dim)
        self._write_kv_prefill(layer, k_for_cache, v_for_cache, B, S)

        # SDPA attention
        q_attn = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k_attn = k_for_cache.transpose(1, 2)
        v_attn = v_for_cache.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, H)

        hidden = residual + F.linear(attn_out, self.o_proj[layer])

        # Post-attention norm + MoE
        residual = hidden
        hidden = F.rms_norm(hidden, (H,), self.post_attn_layernorm[layer], self.rms_norm_eps)

        router_logits = F.linear(hidden, self.router[layer])
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        hidden_flat = fused_experts(
            hidden_states=hidden.reshape(-1, H),
            w1=self.w1[layer], w2=self.w2[layer],
            topk_weights=topk_weights.reshape(-1, self.top_k),
            topk_ids=topk_ids.reshape(-1, self.top_k).to(torch.int32))

        return residual + hidden_flat.view(B, S, H)

    def _write_kv_prefill(self, layer, k, v, B, S):
        """Write prefill K,V to paged cache. k,v: [B, S, nkv, hd]
        Cache layout: [blocks, block_size, nkv, hd] (flat NHD)
        """
        num_pages = math.ceil(S / self.page_size)
        S_padded = num_pages * self.page_size
        if S_padded > S:
            k = F.pad(k, (0, 0, 0, 0, 0, S_padded - S))
            v = F.pad(v, (0, 0, 0, 0, 0, S_padded - S))
        # k,v: [B, S_padded, nkv, hd] → already in [B, pages, ps, nkv, hd] via view
        k_paged = k.view(B, num_pages, self.page_size, self.num_kv_heads, self.head_dim).contiguous()
        v_paged = v.view(B, num_pages, self.page_size, self.num_kv_heads, self.head_dim).contiguous()
        for b in range(B):
            start = b * self.max_pages_per_seq
            self.k_cache[layer][start:start + num_pages] = k_paged[b]
            self.v_cache[layer][start:start + num_pages] = v_paged[b]

    # ── Decode ───────────────────────────────────────────────────────

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Single decode step.
        Args: token_ids [B], positions [B]
        Returns: logits [B, vocab_size]
        """
        B = token_ids.shape[0]
        if hasattr(self, '_cuda_graphs') and B in self._cuda_graphs:
            return self._decode_step_graphed(token_ids, positions)

        pos_i32 = positions.to(torch.int32)
        slot_mapping = self._compute_slot_mapping(pos_i32, B)

        # Plan FlashInfer decode (must be before layer loop / CUDA graph)
        self._plan_flashinfer_decode(self._decode_wrapper, B)

        hidden = F.embedding(token_ids, self.embed_tokens)
        for layer in range(self.num_layers):
            hidden = self._layer_decode(hidden, layer, pos_i32,
                                        slot_mapping, self._decode_wrapper)

        hidden = F.rms_norm(hidden, (self.hidden_size,), self.final_norm, self.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head)
        self.seq_lens[:B] += 1
        self._seq_lens_cpu[:B] += 1
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
        q, k = rope_pytorch(q, k, self.cos_sin_cache, positions,
                            self.num_heads, self.head_dim)

        # Write K,V to cache using reshape_and_cache_flash (flat NHD layout)
        k_write = k.view(B, self.num_kv_heads, self.head_dim)
        v_write = v.reshape(B, self.num_kv_heads, self.head_dim)
        _vllm_ops.reshape_and_cache_flash(
            k_write, v_write,
            self.k_cache[layer], self.v_cache[layer],
            slot_mapping, "auto", self._k_scale, self._v_scale)

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

        hidden = fused_experts(
            hidden_states=hidden, w1=self.w1[layer], w2=self.w2[layer],
            topk_weights=topk_weights, topk_ids=topk_ids.to(torch.int32))

        return residual + hidden

    def _compute_slot_mapping(self, positions, B):
        """Compute flat slot indices for reshape_and_cache_flash kernel.
        slot = page_number * page_size + offset_within_page
        """
        batch_idx = torch.arange(B, device=self.device)
        page_idx = (positions // self.page_size).long()
        offset = (positions % self.page_size).long()
        return (self.block_table[batch_idx, page_idx].long() * self.page_size + offset)

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

    # ── Generation ───────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128) -> torch.Tensor:
        """Greedy autoregressive generation.
        Args: input_ids [B, S]
        Returns: all_tokens [B, S + generated]
        """
        B, S = input_ids.shape
        self.seq_lens[:B] = 0
        self._seq_lens_cpu[:B] = 0

        logits = self.prefill(input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            positions = self.seq_lens[:B].clone()
            logits = self.decode_step(next_token, positions)
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)
            if (next_token == self.eos_token_id).all():
                break

        return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1)

    def reset(self):
        """Reset KV cache and sequence state."""
        self.seq_lens.zero_()
        self._seq_lens_cpu.zero_()
        self.k_cache.zero_()
        self.v_cache.zero_()

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

        # Dummy prefill to populate KV cache
        self.reset()
        dummy = torch.randint(1, 1000, (batch_size, warmup_seq_len), device=self.device)
        with torch.no_grad():
            self.prefill(dummy)

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

        # Re-plan FlashInfer with current seq_lens (plan_info must match page count)
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
        self._seq_lens_cpu[:B] += 1
        return info['static_output']


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "src/MoE/models/OLMoE-1B-7B"
    engine = MoEEngine(model_path, max_batch_size=4, max_seq_len=512,
                       use_torch_compile=False)

    # Quick smoke test
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    print("\nSmoke test: prefill...")
    logits = engine.prefill(input_ids)
    print(f"  Prefill logits shape: {logits.shape}")
    print(f"  Top token: {logits[0, -1].argmax().item()}")

    print("Smoke test: generate...")
    engine.reset()
    tokens = engine.generate(input_ids, max_new_tokens=10)
    print(f"  Generated shape: {tokens.shape}")
    print(f"  Tokens: {tokens[0].tolist()}")
