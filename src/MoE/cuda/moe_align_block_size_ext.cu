/*
 * Patched moe_align_block_size kernel — removes the 1024-expert limit.
 *
 * This is a standalone PyTorch CUDA extension that provides a drop-in
 * replacement for vLLM's ops.moe_align_block_size.  It can be loaded at
 * runtime via torch.utils.cpp_extension.load() without recompiling vLLM.
 *
 * Changes vs the vLLM original (v0.17.1):
 *   1. shared_counts zeroing — stride loop instead of 1-warp-per-32-experts
 *   2. prefix sum — serial fallback when num_experts >= 1024
 *   3. cumsum write — handled by serial path
 *   4. expert_ids fill — stride loop instead of if (threadIdx.x < num_experts)
 *   5. TORCH_CHECK(padded_num_experts < 1024) — removed
 *
 * For num_experts < 1024 the code takes the ORIGINAL BlockScan path,
 * producing bit-identical results.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace moe_ext {

// ═══════════════════════════════════════════════════════════════════
// Device function: _moe_align_block_size  (patched)
// ═══════════════════════════════════════════════════════════════════
template <typename scalar_t>
__device__ void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map,
    int32_t  num_experts,
    int32_t  padded_num_experts,
    int32_t  experts_per_warp,   // == WARP_SIZE
    int32_t  block_size,
    size_t   numel,
    int32_t* __restrict__ cumsum,
    int32_t  max_num_tokens_padded,
    int32_t  max_num_m_blocks,
    bool     has_expert_map) {

  extern __shared__ int32_t shared_counts[];

  // ── Block 1 (odd blockIdx.x): fill sorted_token_ids with sentinel ──
  if (blockIdx.x % 2) {
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
      sorted_token_ids[it] = numel;
    }
    return;
  }

  // ── LIMIT #1 FIX: zero shared_counts with stride loop ──
  // Original: each warp zeros exactly 32 entries → only 1024 total.
  // Patched: all threads cooperate with stride to zero any count.
  for (int i = threadIdx.x; i < padded_num_experts; i += blockDim.x) {
    shared_counts[i] = 0;
  }

  __syncthreads();

  // ── Token counting (unchanged — already uses stride loop) ──
  const size_t tid    = threadIdx.x;
  const size_t stride = blockDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) continue;
    }
    // Index into shared_counts — same layout as original
    int warp_idx      = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }

  __syncthreads();

  // ── LIMIT #2 FIX: prefix sum over expert counts ──
  // Branch: use original BlockScan when it fits, serial scan otherwise.

  if (padded_num_experts < 1024) {
    // ────── ORIGINAL PATH (bit-identical for num_experts < 1024) ──────
    using BlockScan = cub::BlockScan<int32_t, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert_count = 0;
    int eid = threadIdx.x;
    if (eid < num_experts) {
      int warp_idx      = eid / experts_per_warp;
      int expert_offset = eid % experts_per_warp;
      expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
      expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    int cumsum_val;
    BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);

    // LIMIT #3: write cumsum — same as original
    if (eid <= num_experts) {
      cumsum[eid] = cumsum_val;
    }
    if (eid == num_experts) {
      total_tokens_post_pad[0] = cumsum_val;
    }
  } else {
    // ────── SERIAL SCAN (for num_experts >= 1024) ──────
    // One thread computes the full prefix sum.  For 1920 experts this is
    // ~1920 additions ≈ 1-2 μs — negligible vs the MoE GEMM.
    if (threadIdx.x == 0) {
      int running = 0;
      for (int e = 0; e < num_experts; ++e) {
        int warp_idx      = e / experts_per_warp;
        int expert_offset = e % experts_per_warp;
        int cnt = shared_counts[warp_idx * experts_per_warp + expert_offset];
        cnt = CEILDIV(cnt, block_size) * block_size;
        cumsum[e] = running;
        running += cnt;
      }
      cumsum[num_experts] = running;
      total_tokens_post_pad[0] = running;
    }
  }

  __syncthreads();

  // ── LIMIT #4 FIX: fill expert_ids with stride loop ──
  // Original: if (threadIdx.x < num_experts) — only threads 0..1023.
  // Patched: stride loop so all experts get processed.
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
    for (int i = cumsum[e]; i < cumsum[e + 1]; i += block_size) {
      expert_ids[i / block_size] = e;
    }
  }

  // Fill remaining expert_ids with -1 (unchanged — already uses stride)
  const size_t fill_start_idx =
      cumsum[num_experts] / block_size + threadIdx.x;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += blockDim.x) {
    expert_ids[i] = -1;
  }
}

// ═══════════════════════════════════════════════════════════════════
// Global kernel wrappers
// ═══════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map,
    int32_t  num_experts,
    int32_t  padded_num_experts,
    int32_t  experts_per_warp,
    int32_t  block_size,
    size_t   numel,
    int32_t* __restrict__ cumsum,
    int32_t  max_num_tokens_padded,
    int32_t  topk_num,
    bool     has_expert_map) {
  _moe_align_block_size(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad,
      expert_map, num_experts, padded_num_experts, experts_per_warp,
      block_size, numel, cumsum, max_num_tokens_padded,
      CEILDIV(max_num_tokens_padded, block_size), has_expert_map);
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map,
    size_t   numel,
    int32_t  num_experts,
    int32_t  max_num_tokens_padded,
    int32_t  topk_num,
    bool     has_expert_map) {
  const size_t tid    = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= num_experts) continue;
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) continue;
    }
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

}  // namespace moe_ext


// ═══════════════════════════════════════════════════════════════════
// Host-side launch function  (same signature as vLLM's moe_align_block_size)
// ═══════════════════════════════════════════════════════════════════

void moe_align_block_size_patched(
    torch::Tensor topk_ids,
    int64_t       num_experts,
    int64_t       block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    std::optional<torch::Tensor> maybe_expert_map) {

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;

  // ── LIMIT #5 FIX: removed TORCH_CHECK(padded_num_experts < 1024) ──

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
  bool has_expert_map = maybe_expert_map.has_value();
  torch::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::empty({0}, options_int);
  }

  // Type dispatch — we only need int32 and int64 (the common topk_ids dtypes)
  auto scalar_type = topk_ids.scalar_type();

  // Helper lambda to launch for a given scalar type
  auto launch = [&]<typename scalar_t>() {
    torch::Tensor cumsum_buffer = torch::empty({num_experts + 1}, options_int);

    size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
    size_t shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

    // Kernel 1: align (2 threadblocks — block 0 counts+scans, block 1 fills sentinel)
    moe_ext::moe_align_block_size_kernel<scalar_t>
        <<<2, threads, shared_mem_size, stream>>>(
        topk_ids.data_ptr<scalar_t>(),
        sorted_token_ids.data_ptr<int32_t>(),
        experts_ids.data_ptr<int32_t>(),
        num_tokens_post_pad.data_ptr<int32_t>(),
        expert_map.data_ptr<int32_t>(),
        num_experts, padded_num_experts, experts_per_warp, block_size,
        topk_ids.numel(),
        cumsum_buffer.data_ptr<int32_t>(),
        sorted_token_ids.size(0),
        topk_ids.size(1),
        has_expert_map);

    // Kernel 2: place tokens into sorted positions using atomicAdd on cumsum
    const int block_threads = std::min(256, threads);
    const int num_blocks = (topk_ids.numel() + block_threads - 1) / block_threads;
    const int max_blocks = 65535;
    const int actual_blocks = std::min(num_blocks, max_blocks);
    dim3 gridDims(1, actual_blocks);

    moe_ext::count_and_sort_expert_tokens_kernel<scalar_t>
        <<<gridDims, block_threads, 0, stream>>>(
        topk_ids.data_ptr<scalar_t>(),
        sorted_token_ids.data_ptr<int32_t>(),
        cumsum_buffer.data_ptr<int32_t>(),
        expert_map.data_ptr<int32_t>(),
        topk_ids.numel(), num_experts, sorted_token_ids.size(0),
        topk_ids.size(1), has_expert_map);
  };

  // Dispatch on topk_ids dtype
  if (scalar_type == torch::kInt32) {
    launch.operator()<int32_t>();
  } else if (scalar_type == torch::kInt64) {
    launch.operator()<int64_t>();
  } else if (scalar_type == torch::kUInt32) {
    launch.operator()<uint32_t>();
  } else {
    TORCH_CHECK(false, "moe_align_block_size_patched: unsupported topk_ids dtype: ",
                scalar_type);
  }
}


// ═══════════════════════════════════════════════════════════════════
// Python binding
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align_block_size", &moe_align_block_size_patched,
        "Patched moe_align_block_size without 1024-expert limit",
        py::arg("topk_ids"),
        py::arg("num_experts"),
        py::arg("block_size"),
        py::arg("sorted_token_ids"),
        py::arg("experts_ids"),
        py::arg("num_tokens_post_pad"),
        py::arg("expert_map") = py::none());
}
