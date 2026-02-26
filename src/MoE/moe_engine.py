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
from vllm.vllm_flash_attn import flash_attn_varlen_func

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

        # FlashInfer workspace (128 MB) — shared between prefill and decode wrappers
        # (only one is active at a time; each re-plans before use)
        self._workspace_buf = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self._workspace_buf, kv_layout="NHD", use_cuda_graph=False)
        # No prefill wrapper needed — FA3 (flash_attn_varlen_func) is stateless

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
        """Prefill forward pass (CUDA graph only).

        Args: input_ids [B, S]
        Returns: logits [B, S, vocab_size]
        Requires: capture_prefill_cuda_graph() called with sufficient sizes.
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
                (self.max_batch_size + 1,), N,
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
            seq_lengths, self.max_batch_size)
        info['static_cu_seqlens'].copy_(cu_seqlens)

        # 5. Replay graph
        info['graph'].replay()

        # 6. Update seq_lens for all sequences
        for sid, length in zip(seq_ids, seq_lengths):
            self.seq_lens[sid] = length
            self._seq_lens_cpu[sid] = length

        # 7. Return only real token logits
        return info['static_output'][:n_real]

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

    # ── Slot-based Prefill ────────────────────────────────────────────

    @torch.no_grad()
    def prefill_to_slot(self, seq_id: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill a single sequence into a specific KV cache slot (CUDA graph only).

        Args:
            seq_id: slot index in block_table / seq_lens
            input_ids: [S] token IDs (1-D)
        Returns:
            logits: [S, vocab_size]
        Requires: capture_prefill_cuda_graph() called with sufficient sizes.
        """
        S = input_ids.shape[0]
        padded_N = self._find_nearest_prefill_total(S)
        return self._prefill_graphed_flat(input_ids, [S], [seq_id], padded_N)

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
        Requires: capture_prefill_cuda_graph() called with sufficient sizes.
        """
        # Normalize to list of 1D tensors
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2:
            input_ids = [input_ids[i] for i in range(input_ids.shape[0])]

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
        q, k = rope_pytorch(q, k, self.cos_sin_cache, positions,
                            self.num_heads, self.head_dim)

        # Write K,V to paged cache for ALL tokens
        k_write = k.reshape(N, self.num_kv_heads, self.head_dim)
        v_write = v.reshape(N, self.num_kv_heads, self.head_dim)
        _vllm_ops.reshape_and_cache_flash(
            k_write, v_write,
            self.k_cache[layer], self.v_cache[layer],
            slot_mapping, "auto", self._k_scale, self._v_scale)

        # ── Split attention ──
        # Decode tokens [0:D]: FlashInfer BatchDecode (reads paged KV cache)
        if D > 0 and N > D:
            # Mixed: both decode and prefill
            q_decode = q[:D].view(D, self.num_heads, self.head_dim)
            decode_out = decode_wrapper.run(
                q_decode, (self.k_cache[layer], self.v_cache[layer]))
            decode_out = decode_out.reshape(D, H)

            # Prefill tokens [D:]: FA3 self-attention on freshly computed Q/K/V
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
            # Pure prefill
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

        hidden = fused_experts(
            hidden_states=hidden, w1=self.w1[layer], w2=self.w2[layer],
            topk_weights=topk_weights, topk_ids=topk_ids.to(torch.int32))

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

        q, k = rope_pytorch(q, k, self.cos_sin_cache, positions,
                            self.num_heads, self.head_dim)

        k_3d = k.reshape(N, self.num_kv_heads, self.head_dim)
        v_3d = v.reshape(N, self.num_kv_heads, self.head_dim)
        _vllm_ops.reshape_and_cache_flash(
            k_3d, v_3d,
            self.k_cache[layer], self.v_cache[layer],
            slot_mapping, "auto", self._k_scale, self._v_scale)

        q_buf.copy_(q.view(N, self.num_heads, self.head_dim))
        k_buf.copy_(k_3d)
        v_buf.copy_(v_3d)
        residual_buf.copy_(hidden)

    def _layer_stage4_post_attn(self, attn_out, residual, hidden_out, layer):
        """Stage 4: O proj -> residual add -> post-attn norm -> MoE.

        Writes result into hidden_out [N, H] for next layer.
        """
        H = self.hidden_size
        hidden = residual + F.linear(attn_out, self.o_proj[layer])

        residual = hidden
        hidden = F.rms_norm(hidden, (H,), self.post_attn_layernorm[layer],
                            self.rms_norm_eps)

        router_logits = F.linear(hidden, self.router[layer])
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        hidden = fused_experts(
            hidden_states=hidden, w1=self.w1[layer], w2=self.w2[layer],
            topk_weights=topk_weights, topk_ids=topk_ids.to(torch.int32))

        hidden_out.copy_(residual + hidden)

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
                   prefill_seq_ids, prefill_input_ids):
        """Mixed prefill+decode forward pass.

        Concatenates all tokens as [decode | prefill], runs shared compute,
        splits only at attention (FlashInfer decode + FA3 prefill).

        Args:
            decode_seq_ids: list[int] — KV cache slot indices for decode requests
            decode_token_ids: Tensor [D] — one token per decode request
            prefill_seq_ids: list[int] — KV cache slot indices for prefill requests
            prefill_input_ids: list[Tensor] — variable-length prefill sequences
        Returns:
            logits: Tensor [N_total, vocab_size]
                logits[:D] = decode logits, logits[D:] = prefill logits (concat)
        """
        D = len(decode_seq_ids)
        prefill_lengths = [ids.shape[0] for ids in prefill_input_ids]
        N_total = D + sum(prefill_lengths)

        # ── Auto-dispatch to piecewise graphs if available ──
        graph_N = self._find_nearest_piecewise_graph(N_total)
        if graph_N is not None:
            return self._mixed_step_piecewise(
                decode_seq_ids, decode_token_ids,
                prefill_seq_ids, prefill_input_ids, graph_N)

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

    def capture_mixed_cuda_graphs(self, total_token_sizes=None,
                                  use_torch_compile=None):
        """Capture per-layer piecewise CUDA graphs for mixed batches.

        For each N in total_token_sizes, captures 2 graphs per layer:
          - Stage 1: RMSNorm -> QKV -> Q/K norm -> RoPE -> KV cache write
          - Stage 4: O proj -> residual add -> post-attn norm -> MoE

        Stages 2 & 3 (attention) run eagerly — single kernel each.

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

        # Optionally compile stage functions
        stage1_fn = self._layer_stage1_pre_attn
        stage4_fn = self._layer_stage4_post_attn
        if use_torch_compile:
            stage1_fn = torch.compile(stage1_fn, fullgraph=False)
            stage4_fn = torch.compile(stage4_fn, fullgraph=False)

        for N in total_token_sizes:
            self.reset()

            # ── Shared intermediate buffers (fixed addresses for graph replay) ──
            q_buf = torch.zeros(N, self.num_heads, self.head_dim,
                                dtype=self.dtype, device=self.device)
            k_buf = torch.zeros(N, self.num_kv_heads, self.head_dim,
                                dtype=self.dtype, device=self.device)
            v_buf = torch.zeros(N, self.num_kv_heads, self.head_dim,
                                dtype=self.dtype, device=self.device)
            attn_out_buf = torch.zeros(N, self.num_heads * self.head_dim,
                                       dtype=self.dtype, device=self.device)
            residual_buf = torch.zeros(N, self.hidden_size,
                                       dtype=self.dtype, device=self.device)
            hidden_buf = torch.zeros(N, self.hidden_size,
                                     dtype=self.dtype, device=self.device)

            # Static inputs (same for all layers within a step)
            static_positions = (torch.arange(N, dtype=torch.int32,
                                             device=self.device)
                                % self.max_seq_len)
            total_kv_slots = self.total_pages * self.page_size
            static_slot_mapping = torch.arange(
                N, dtype=torch.long, device=self.device) % total_kv_slots
            static_token_ids = torch.randint(1, 1000, (N,),
                                             device=self.device)

            # ── Warmup all layers ──
            n_warmup = 5 if use_torch_compile else 3
            for _ in range(n_warmup):
                hidden_buf.copy_(F.embedding(static_token_ids,
                                             self.embed_tokens))
                for layer in range(self.num_layers):
                    stage1_fn(hidden_buf, static_positions,
                              static_slot_mapping, layer,
                              q_buf, k_buf, v_buf, residual_buf)
                    # Fake attention output for warmup
                    attn_out_buf.copy_(q_buf.reshape(N, -1))
                    stage4_fn(attn_out_buf, residual_buf, hidden_buf, layer)
            torch.cuda.synchronize()

            # ── Capture per-layer graphs ──
            stage1_graphs = []
            stage4_graphs = []

            # Re-init hidden for capture
            hidden_buf.copy_(F.embedding(static_token_ids,
                                         self.embed_tokens))

            for layer in range(self.num_layers):
                # Capture stage 1
                g1 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g1):
                    stage1_fn(hidden_buf, static_positions,
                              static_slot_mapping, layer,
                              q_buf, k_buf, v_buf, residual_buf)
                stage1_graphs.append(g1)

                # Simulate attention (write something into attn_out_buf)
                attn_out_buf.copy_(q_buf.reshape(N, -1))

                # Capture stage 4
                g4 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g4):
                    stage4_fn(attn_out_buf, residual_buf, hidden_buf,
                              layer)
                stage4_graphs.append(g4)

            self._piecewise_graphs[N] = {
                'stage1_graphs': stage1_graphs,
                'stage4_graphs': stage4_graphs,
                'q_buf': q_buf,
                'k_buf': k_buf,
                'v_buf': v_buf,
                'attn_out_buf': attn_out_buf,
                'residual_buf': residual_buf,
                'hidden_buf': hidden_buf,
                'static_positions': static_positions,
                'static_slot_mapping': static_slot_mapping,
                'static_token_ids': static_token_ids,
            }

            compile_str = " + torch.compile" if use_torch_compile else ""
            print(f"  Piecewise CUDA graphs{compile_str} captured for "
                  f"N={N} ({self.num_layers * 2} graphs)")

    def _find_nearest_piecewise_graph(self, total_tokens):
        """Find smallest piecewise graph with N >= total_tokens, or None."""
        if not hasattr(self, '_piecewise_graphs'):
            return None
        candidates = [n for n in self._piecewise_graphs if n >= total_tokens]
        return min(candidates) if candidates else None

    @torch.no_grad()
    def _mixed_step_piecewise(self, decode_seq_ids, decode_token_ids,
                              prefill_seq_ids, prefill_input_ids,
                              graph_N):
        """Replay mixed step using per-layer piecewise CUDA graphs.

        Stage 1 & 4 are CUDA graphs (keyed by N_total).
        Stage 2 (FlashInfer decode) & Stage 3 (FA3 prefill) run eagerly.
        """
        info = self._piecewise_graphs[graph_N]
        D = len(decode_seq_ids)
        prefill_lengths = [ids.shape[0] for ids in prefill_input_ids]
        N_actual = D + sum(prefill_lengths)
        H = self.hidden_size

        # ── 1. Build token_ids and copy into static buffer ──
        token_parts = []
        if D > 0:
            token_parts.append(decode_token_ids)
        for ids in prefill_input_ids:
            token_parts.append(ids)
        all_token_ids = torch.cat(token_parts)

        info['static_token_ids'][:N_actual].copy_(all_token_ids)
        if N_actual < graph_N:
            info['static_token_ids'][N_actual:].zero_()

        # ── 2. Compute positions and copy ──
        decode_positions = (self._seq_lens_cpu[decode_seq_ids].to(torch.int32)
                            .to(self.device)) if D > 0 else torch.empty(
                                0, dtype=torch.int32, device=self.device)
        prefill_pos_parts = [
            torch.arange(s, dtype=torch.int32, device=self.device)
            for s in prefill_lengths
        ]
        positions = torch.cat([decode_positions] + prefill_pos_parts)
        info['static_positions'][:N_actual].copy_(positions)
        if N_actual < graph_N:
            info['static_positions'][N_actual:].zero_()

        # ── 3. Compute slot_mapping and copy ──
        if D > 0:
            d_idx = torch.tensor(decode_seq_ids, device=self.device,
                                 dtype=torch.long)
            d_page = (decode_positions // self.page_size).long()
            d_offset = (decode_positions % self.page_size).long()
            decode_slots = (self.block_table[d_idx, d_page].long()
                            * self.page_size + d_offset)
        else:
            decode_slots = torch.empty(0, dtype=torch.long,
                                       device=self.device)

        prefill_slot_parts = []
        for sid, length in zip(prefill_seq_ids, prefill_lengths):
            pos = torch.arange(length, device=self.device)
            pg = (pos // self.page_size).long()
            off = (pos % self.page_size).long()
            prefill_slot_parts.append(
                self.block_table[sid, pg].long() * self.page_size + off)

        slot_mapping = torch.cat([decode_slots] + prefill_slot_parts)
        info['static_slot_mapping'][:N_actual].copy_(slot_mapping)
        if N_actual < graph_N:
            info['static_slot_mapping'][N_actual:].fill_(-1)

        # ── 4. Increment decode seq_lens and plan FlashInfer ──
        if D > 0:
            for sid in decode_seq_ids:
                self._seq_lens_cpu[sid] += 1
            self._plan_flashinfer_decode_for_subset(decode_seq_ids)

        # ── 5. Build prefill cu_seqlens for FA3 ──
        prefill_cu = torch.zeros(len(prefill_lengths) + 1, dtype=torch.int32,
                                 device=self.device)
        for i, s in enumerate(prefill_lengths):
            prefill_cu[i + 1] = prefill_cu[i] + s
        prefill_max = max(prefill_lengths) if prefill_lengths else 0

        # ── 6. Embed tokens into hidden_buf ──
        info['hidden_buf'].copy_(
            F.embedding(info['static_token_ids'], self.embed_tokens))

        # ── 7. Per-layer piecewise replay ──
        q_buf = info['q_buf']
        k_buf = info['k_buf']
        v_buf = info['v_buf']
        attn_out_buf = info['attn_out_buf']

        for layer in range(self.num_layers):
            # Stage 1: pre-attention (CUDA graph)
            info['stage1_graphs'][layer].replay()

            # Stage 2: FlashInfer decode on q_buf[:D]
            if D > 0:
                q_decode = q_buf[:D]  # [D, num_heads, head_dim]
                decode_out = self._decode_wrapper.run(
                    q_decode,
                    (self.k_cache[layer], self.v_cache[layer]))
                attn_out_buf[:D].copy_(decode_out.reshape(D, H))

            # Stage 3: FA3 prefill on q_buf[D:N_actual]
            if prefill_lengths:
                N_pf = N_actual - D
                q_pf = q_buf[D:N_actual]
                k_pf = k_buf[D:N_actual]
                v_pf = v_buf[D:N_actual]
                prefill_out = flash_attn_varlen_func(
                    q_pf, k_pf, v_pf,
                    cu_seqlens_q=prefill_cu,
                    cu_seqlens_k=prefill_cu,
                    max_seqlen_q=prefill_max,
                    max_seqlen_k=prefill_max,
                    causal=True, fa_version=3)
                attn_out_buf[D:N_actual].copy_(prefill_out.reshape(N_pf, H))

            # Zero padding region of attn_out_buf
            if N_actual < graph_N:
                attn_out_buf[N_actual:].zero_()

            # Stage 4: post-attention (CUDA graph)
            info['stage4_graphs'][layer].replay()

        # ── 8. Final norm + lm_head (eager — single kernel each) ──
        hidden = info['hidden_buf']
        hidden = F.rms_norm(hidden, (self.hidden_size,), self.final_norm,
                            self.rms_norm_eps)
        logits = F.linear(hidden, self.lm_head)

        # ── 9. Update GPU seq_lens ──
        for sid in decode_seq_ids:
            self.seq_lens[sid] += 1
        for sid, length in zip(prefill_seq_ids, prefill_lengths):
            self.seq_lens[sid] = length
            self._seq_lens_cpu[sid] = length

        return logits[:N_actual]

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
    engine = MoEEngine(model_path, max_batch_size=4, max_seq_len=512,
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
