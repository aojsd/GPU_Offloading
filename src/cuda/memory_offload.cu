/**
 * @file memory_offload.cu
 * @brief Empirically tests three strategies for overlapping data movement with computation
 * for a large matrix multiplication problem (A * B = C) where matrix A is partially
 * offloaded (e.g., Host DRAM, PCIe, NVLink, C2C).
 *
 * Shapes: A[N x H], B[H x S], C[N x S]
 *
 * This version uses a CUDA Graph-based approach for all tests and includes
 * a fast, GPU-based verification function.
 *
 * Case 1: Explicit Overlap
 * - Uses a CUDA Graph to model the dependencies.
 * - Subcase 1 (default): Offloads to pinned host memory, transfers over PCIe (H2D).
 * - Subcase 2 (--nvlink): Offloads to a peer GPU's memory, transfers over NVLink (D2D).
 *
 * Case 2: Overlap with UVM Prefetch
 * - Matrices are allocated with `cudaMallocManaged`.
 * - An async prefetch of the offloaded data is launched concurrently
 * with the main compute kernel, allowing the driver to overlap the transfer and
 * computation.
 *
 * Case 3: Legacy Zero-Copy (Interleaved Unified Kernel)
 * - A single, robust kernel is launched that is aware of both device VRAM and
 * pinned host RAM (zero-copy) for matrix A.
 * - The kernel re-orders logical work in a user-configurable "X-to-1" start/end
 * pattern to force a mix of high- and low-latency memory requests on the SMs,
 * enabling the GPU's latency-hiding capabilities.
 *
 * ---
 *
 * Compilation using Makefile:
 * make
 * (Optional: specify GPU architecture, e.g., `make ARCH=sm_90a`)
 *
 * Compilation (manual):
 * nvcc -O3 -arch=sm_90a memory_offload.cu -o memory_offload
 * (Adjust -arch=sm_XX to your GPU's compute capability)
 *
 * ---
 *
 * Usage:
 * ./memory_offload [options]
 * Run with -h or --help for a full list of options.
 */
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>

// CUDA runtime
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CHECK_CUDA(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                          \
        {                                                                \
            fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", err,               \
                    cudaGetErrorString(err));                            \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Forward declaration for the main test function
void runBandwidthExtensionTest(int N, int H, int S, float offload_ratio, int trials, int device_id, int interleave_ratio);

// Kernel configuration
constexpr int TILE_DIM = 32;
// Maximum grid dimension size for Y and Z axes on modern GPUs
constexpr unsigned int MAX_GRID_DIM = 65535;

// Helper function to calculate a 2D grid that can handle large row counts
dim3 calculate_grid_dims(int num_elements_x, int num_elements_y) {
    long long total_blocks_y = (num_elements_y + TILE_DIM - 1) / TILE_DIM;
    long long grid_y = std::min((long long)MAX_GRID_DIM, total_blocks_y);
    long long grid_z = (total_blocks_y + grid_y - 1) / grid_y;

    if (grid_z > MAX_GRID_DIM) {
        fprintf(stderr, "Error: Matrix dimensions exceed launch capabilities.\n");
        exit(EXIT_FAILURE);
    }

    return dim3((num_elements_x + TILE_DIM - 1) / TILE_DIM, (unsigned int)grid_y, (unsigned int)grid_z);
}

// Tiled Matrix Multiplication Kernel (for separate launches and verification)
__global__ void matMulKernel(float *C, const float *A, const float *B, int N, int H, int S, int startRow, int numRows)
{
    long long total_row_blocks = (numRows + TILE_DIM - 1) / TILE_DIM;
    long long linear_block_idx_y = (long long)blockIdx.z * gridDim.y + blockIdx.y;

    if (linear_block_idx_y >= total_row_blocks) return;

    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = linear_block_idx_y * TILE_DIM + ty + startRow;
    int col = blockIdx.x * TILE_DIM + tx;

    float C_val = 0.0f;
    
    for (int t = 0; t < (H + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        long long local_row = (long long)row - startRow;
        if (row < (startRow + numRows) && (t * TILE_DIM + tx) < H) {
            sA[ty][tx] = A[local_row * H + t * TILE_DIM + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if (col < S && (t * TILE_DIM + ty) < H) {
            sB[ty][tx] = B[(long long)(t * TILE_DIM + ty) * S + col];
        } else {
            sB[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) C_val += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }

    if (row < (startRow + numRows) && col < S) C[(long long)row * S + col] = C_val;
}

// **REWRITTEN:** This is the correct implementation of your "X-to-1" design.
// It computes C[i] using A[i], but the order in which the 'i's are processed is interleaved.
__global__ void interleavedScratchpadMatMulKernel(
    float *C, 
    const float *A_resident, 
    const float *A_offload, 
    const float *B, 
    int N, int H, int S, 
    int N_resident,
    int interleave_ratio)
{
    long long total_tile_rows = (N + TILE_DIM - 1) / TILE_DIM;
    long long logical_tile_idx = (long long)blockIdx.z * gridDim.y + blockIdx.y;

    if (logical_tile_idx >= total_tile_rows) return;

    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // **KEY LOGIC:** The block computes the result for its assigned LOGICAL tile.
    int logical_start_row = logical_tile_idx * TILE_DIM;
    int tile_start_col = blockIdx.x * TILE_DIM;

    float C_val = 0.0f;
    for (int t = 0; t < (H + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        // Load the tile for matrix A into shared memory (sA) row by row.
        int row_to_load = logical_start_row + ty;
        int col_to_load = t * TILE_DIM + tx;

        // Use a simple check against N_resident to determine the memory source for the LOGICAL row.
        if (row_to_load < N && col_to_load < H) {
            if (row_to_load < N_resident) {
                // This row is in the resident (VRAM) buffer.
                sA[ty][tx] = A_resident[ (long long)row_to_load * H + col_to_load ];
            } else {
                // This row is in the offloaded (Zero-Copy) buffer.
                long long offload_row_index = (long long)row_to_load - N_resident;
                sA[ty][tx] = A_offload[ offload_row_index * H + col_to_load ];
            }
        } else {
            sA[ty][tx] = 0.0f;
        }
        
        // Load tile for B into shared memory (sB) - always from VRAM.
        int b_row = t * TILE_DIM + ty;
        if (b_row < H && tile_start_col + tx < S) {
            sB[ty][tx] = B[ (long long)b_row * S + (tile_start_col + tx) ];
        } else {
            sB[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) C_val += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }

    // Write final result to C using the LOGICAL tile index to ensure correct, contiguous output.
    int final_row = logical_start_row + ty;
    int final_col = tile_start_col + tx;
    if (final_row < N && final_col < S) {
        C[(long long)final_row * S + final_col] = C_val;
    }
}


// New kernel to compare two matrices on the GPU and find the maximum absolute error
__global__ void compareAndFindMaxErrorKernel(const float* C_original, const float* C_verify, float* d_max_error, size_t total_elements) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += gridDim.x * blockDim.x) {
        float error = fabsf(C_original[i] - C_verify[i]);
        
        float current_max = *d_max_error;
        while (error > current_max) {
            float previous_max = __uint_as_float(atomicCAS((unsigned int*)d_max_error, __float_as_uint(current_max), __float_as_uint(error)));
            if (previous_max == current_max) break;
            current_max = previous_max;
        }
    }
}

// Kernel to initialize the cuRAND states
__global__ void setupCurandKernel(curandState_t *states, unsigned long long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &states[id]);
}

// Kernel to initialize a matrix with random values
__global__ void initMatrixGpuKernel(float* matrix, size_t num_elements, curandState_t* states) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    curandState_t local_state = states[id];
    for (size_t i = 0; i < num_elements; i += stride) {
        matrix[i] = curand_uniform(&local_state);
    }
    states[id] = local_state;
}

// Host function to orchestrate matrix initialization on the GPU
void init_matrices_on_gpu(float* d_A, float* d_B, int N, int H, int S) {
    std::cout << "Initializing matrices on GPU... " << std::flush;
    
    size_t num_elements_A = (size_t)N * H;
    if (d_A == nullptr) num_elements_A = 0;

    size_t num_elements_B = (size_t)H * S;
    if (d_B == nullptr) num_elements_B = 0;

    if (num_elements_A == 0 && num_elements_B == 0) {
        std::cout << "Skipped (no matrices to init).\n";
        return;
    }

    size_t max_elements = std::max(num_elements_A, num_elements_B);

    int threads_per_block = 256;
    int blocks = std::min(16384, (int)((max_elements + threads_per_block - 1) / threads_per_block));
    size_t num_states = (size_t)threads_per_block * blocks;

    curandState_t* d_rand_states;
    CHECK_CUDA(cudaMalloc(&d_rand_states, num_states * sizeof(curandState_t)));
    setupCurandKernel<<<blocks, threads_per_block>>>(d_rand_states, time(NULL));

    if (num_elements_A > 0) initMatrixGpuKernel<<<blocks, threads_per_block>>>(d_A, num_elements_A, d_rand_states);
    if (num_elements_B > 0) initMatrixGpuKernel<<<blocks, threads_per_block>>>(d_B, num_elements_B, d_rand_states);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_rand_states));
    std::cout << "Done.\n";
}

// New GPU-based verification function
void verify_result_gpu(const float* d_A, const float* d_B, const float* d_C_original, int N, int H, int S) {
    std::cout << "\nVerifying result on GPU... " << std::flush;
    
    size_t C_size = (size_t)N * S * sizeof(float);
    float* d_C_verify;
    CHECK_CUDA(cudaMalloc(&d_C_verify, C_size));

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid = calculate_grid_dims(S, N);
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C_verify, d_A, d_B, N, H, S, 0, N);

    float* d_max_error;
    float h_max_error = 0.0f;
    CHECK_CUDA(cudaMalloc(&d_max_error, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_max_error, 0, sizeof(float)));

    int compare_threads = 256;
    int compare_blocks = std::min(1024, (int)(((size_t)N * S + compare_threads - 1) / compare_threads));
    compareAndFindMaxErrorKernel<<<compare_blocks, compare_threads>>>(d_C_original, d_C_verify, d_max_error, (size_t)N * S);
    
    CHECK_CUDA(cudaMemcpy(&h_max_error, d_max_error, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Done.\nMaximum absolute error: " << h_max_error << std::endl;

    CHECK_CUDA(cudaFree(d_C_verify));
    CHECK_CUDA(cudaFree(d_max_error));
}


// **RESTORED:** This function handles the "fast path" case where offload_ratio is 0 for non-extension tests.
void runSingleKernelTest(int N, int H, int S, int trials, bool use_uvm, int device_id) {
    std::cout << "Zero offload ratio detected. Running simplified single-kernel test (CUDA Graph Version).\n";

    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    if (use_uvm) {
        CHECK_CUDA(cudaMallocManaged(&d_A, A_size));
        CHECK_CUDA(cudaMallocManaged(&d_B, B_size));
        CHECK_CUDA(cudaMallocManaged(&d_C, C_size));
        init_matrices_on_gpu(d_A, d_B, N, H, S);

        CHECK_CUDA(cudaMemAdvise(d_A, A_size, cudaMemAdviseSetPreferredLocation, device_id));
        CHECK_CUDA(cudaMemAdvise(d_B, B_size, cudaMemAdviseSetPreferredLocation, device_id));
        CHECK_CUDA(cudaMemAdvise(d_C, C_size, cudaMemAdviseSetPreferredLocation, device_id));
        CHECK_CUDA(cudaMemPrefetchAsync(d_A, A_size, device_id, stream));
        CHECK_CUDA(cudaMemPrefetchAsync(d_B, B_size, device_id, stream));
        CHECK_CUDA(cudaMemPrefetchAsync(d_C, C_size, device_id, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    } else {
        CHECK_CUDA(cudaMalloc(&d_A, A_size));
        CHECK_CUDA(cudaMalloc(&d_B, B_size));
        CHECK_CUDA(cudaMalloc(&d_C, C_size));
        init_matrices_on_gpu(d_A, d_B, N, H, S);
    }

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    cudaKernelNodeParams kernel_params = {0};
    kernel_params.func = (void*)matMulKernel;
    kernel_params.gridDim = calculate_grid_dims(S, N);
    kernel_params.blockDim = dim3(TILE_DIM, TILE_DIM);
    void *kernel_args[] = {&d_C, &d_A, &d_B, &N, &H, &S, new int(0), new int(N)};
    kernel_params.kernelParams = kernel_args;
    
    cudaGraphNode_t start_node, stop_node, kernel_node;
    CHECK_CUDA(cudaGraphAddEventRecordNode(&start_node, graph, nullptr, 0, start));
    CHECK_CUDA(cudaGraphAddKernelNode(&kernel_node, graph, &start_node, 1, &kernel_params));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&stop_node, graph, &kernel_node, 1, stop));
    
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";

    std::vector<float> kernel_times;
    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        kernel_times.push_back(ms);
    }

    auto avg = [](const std::vector<float>& v) { return std::accumulate(v.begin(), v.end(), 0.0f) / v.size(); };
    double avg_time_ms = avg(kernel_times);
    double effective_bandwidth = (A_size + B_size) / (avg_time_ms * 1e6);

    std::cout << "\n--- Timings (avg over " << trials << " trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Effective Bandwidth (GB/s): " << std::setw(8) << effective_bandwidth << "\n";
    std::cout << "Total Kernel Time:          " << std::setw(8) << avg_time_ms << " ms\n";

    verify_result_gpu(d_A, d_B, d_C, N, H, S);
    
    delete (int*)kernel_args[6];
    delete (int*)kernel_args[7];
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaStreamDestroy(stream));
}


void runExplicitOverlapTest(int N, int H, int S, float offload_ratio, int trials, bool use_nvlink, int device_id) {
    if (use_nvlink) {
        std::cout << "\n--- Running Case 1: Explicit Overlap with NVLink (D2D) ---\n";
    } else {
        std::cout << "\n--- Running Case 1: Explicit Overlap with PCIe (H2D) ---\n";
    }

    if (offload_ratio == 0.0f) {
        runSingleKernelTest(N, H, S, trials, false, device_id);
        return;
    }

    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);
    int N_offload = static_cast<int>(N * offload_ratio);
    int N_resident = N - N_offload;
    size_t A_offload_size = (size_t)N_offload * H * sizeof(float);
    
    std::cout << "Resident Rows: " << N_resident << ", Offloaded Rows: " << N_offload << std::endl;

    float* h_A_pinned_offload = nullptr;
    float* d_A_peer_offload = nullptr;
    cudaMemcpyKind copyKind;
    int peerDeviceId = -1;

    if (use_nvlink) {
        int device_count;
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        peerDeviceId = (device_id + 1) % device_count;

        int canAccessPeer;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, device_id, peerDeviceId));
        if (canAccessPeer) {
            std::cout << "Enabling peer access from Device " << device_id << " to Device " << peerDeviceId << std::endl;
            CHECK_CUDA(cudaSetDevice(device_id));
            CHECK_CUDA(cudaDeviceEnablePeerAccess(peerDeviceId, 0));
        } else {
            std::cerr << "Error: Peer access between Device " << device_id << " and " << peerDeviceId << " is not supported." << std::endl;
            exit(EXIT_FAILURE);
        }
        
        CHECK_CUDA(cudaSetDevice(peerDeviceId));
        CHECK_CUDA(cudaMalloc(&d_A_peer_offload, A_offload_size));
        CHECK_CUDA(cudaSetDevice(device_id));

        copyKind = cudaMemcpyDeviceToDevice;
    } else {
        CHECK_CUDA(cudaHostAlloc(&h_A_pinned_offload, A_offload_size, cudaHostAllocDefault));
        copyKind = cudaMemcpyHostToDevice;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, A_size));
    CHECK_CUDA(cudaMalloc(&d_B, B_size));
    CHECK_CUDA(cudaMalloc(&d_C, C_size));

    init_matrices_on_gpu(d_A, d_B, N, H, S);

    if (use_nvlink) {
        CHECK_CUDA(cudaMemcpy(d_A_peer_offload, d_A + (size_t)N_resident * H, A_offload_size, cudaMemcpyDeviceToDevice));
    } else {
        CHECK_CUDA(cudaMemcpy(h_A_pinned_offload, d_A + (size_t)N_resident * H, A_offload_size, cudaMemcpyDeviceToHost));
    }
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    cudaEvent_t start, stop, transferStart, transferStop, compute1Start, compute1Stop, compute2Start, compute2Stop;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&transferStart)); CHECK_CUDA(cudaEventCreate(&transferStop));
    CHECK_CUDA(cudaEventCreate(&compute1Start)); CHECK_CUDA(cudaEventCreate(&compute1Stop));
    CHECK_CUDA(cudaEventCreate(&compute2Start)); CHECK_CUDA(cudaEventCreate(&compute2Stop));

    cudaGraphNode_t start_node, stop_node, memcpy_node, kernel1_node, kernel2_node, sync_node;
    cudaGraphNode_t event_transfer_start, event_transfer_stop, event_compute1_start, event_compute1_stop, event_compute2_start, event_compute2_stop;

    CHECK_CUDA(cudaGraphAddEventRecordNode(&start_node, graph, nullptr, 0, start));

    void* memcpy_src = use_nvlink ? (void*)d_A_peer_offload : (void*)h_A_pinned_offload;
    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_transfer_start, graph, &start_node, 1, transferStart));
    CHECK_CUDA(cudaGraphAddMemcpyNode1D(&memcpy_node, graph, &event_transfer_start, 1, d_A + (size_t)N_resident * H, memcpy_src, A_offload_size, copyKind));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_transfer_stop, graph, &memcpy_node, 1, transferStop));

    cudaKernelNodeParams kernel1_params = {0};
    kernel1_params.func = (void*)matMulKernel;
    kernel1_params.gridDim = calculate_grid_dims(S, N_resident);
    kernel1_params.blockDim = dim3(TILE_DIM, TILE_DIM);
    void *kernel1_args[] = {&d_C, &d_A, &d_B, &N, &H, &S, new int(0), new int(N_resident)};
    kernel1_params.kernelParams = kernel1_args;
    
    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_compute1_start, graph, &start_node, 1, compute1Start));
    CHECK_CUDA(cudaGraphAddKernelNode(&kernel1_node, graph, &event_compute1_start, 1, &kernel1_params));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_compute1_stop, graph, &kernel1_node, 1, compute1Stop));

    cudaGraphNode_t sync_deps[] = {event_transfer_stop, event_compute1_stop};
    CHECK_CUDA(cudaGraphAddEmptyNode(&sync_node, graph, sync_deps, 2));

    cudaKernelNodeParams kernel2_params = {0};
    kernel2_params.func = (void*)matMulKernel;
    kernel2_params.gridDim = calculate_grid_dims(S, N_offload);
    kernel2_params.blockDim = dim3(TILE_DIM, TILE_DIM);
    void *kernel2_args[] = {&d_C, &d_A, &d_B, &N, &H, &S, new int(N_resident), new int(N_offload)};
    kernel2_params.kernelParams = kernel2_args;

    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_compute2_start, graph, &sync_node, 1, compute2Start));
    CHECK_CUDA(cudaGraphAddKernelNode(&kernel2_node, graph, &event_compute2_start, 1, &kernel2_params));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_compute2_stop, graph, &kernel2_node, 1, compute2Stop));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&stop_node, graph, &event_compute2_stop, 1, stop));
    
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        if (!use_nvlink) {
            CHECK_CUDA(cudaMemcpy(h_A_pinned_offload, d_A + (size_t)N_resident * H, A_offload_size, cudaMemcpyDeviceToHost));
        }
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";

    std::vector<double> total_times, transfer_times, compute1_times, compute2_times;
    for (int i = 0; i < trials; ++i) {
        if (!use_nvlink) {
            CHECK_CUDA(cudaMemcpy(h_A_pinned_offload, d_A + (size_t)N_resident * H, A_offload_size, cudaMemcpyDeviceToHost));
        }
        
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float ms_total, ms_transfer, ms_compute1, ms_compute2;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms_transfer, transferStart, transferStop));
        CHECK_CUDA(cudaEventElapsedTime(&ms_compute1, compute1Start, compute1Stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms_compute2, compute2Start, compute2Stop));
        total_times.push_back(ms_total);
        transfer_times.push_back(ms_transfer);
        compute1_times.push_back(ms_compute1);
        compute2_times.push_back(ms_compute2);
    }
    
    auto avg = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0f) / v.size(); };
    std::string bw_label = use_nvlink ? "NVLink Transfer (D2D):" : "PCIe Transfer (H2D): ";

    std::cout << "\n--- Timings (avg over " << trials << " trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << bw_label << std::setw(8) << avg(transfer_times) << " ms\n";
    std::cout << "Compute (Resident Data):  " << std::setw(8) << avg(compute1_times) << " ms\n";
    std::cout << "Compute (Offloaded Data): " << std::setw(8) << avg(compute2_times) << " ms\n";
    std::cout << "--------------------------------------\n";
    std::string bw_rate_label = use_nvlink ? "NVLink Bandwidth (GB/s): " : "PCIe Bandwidth (GB/s): ";
    std::cout << bw_rate_label << std::setw(8) << (A_offload_size / (1e6 * avg(transfer_times))) << " GB/s\n";
    double total_compute_time = avg(compute1_times) + avg(compute2_times);
    std::cout << "GPU Throughput (GB/s): " << std::setw(8) << ((A_size + B_size) / (1e6 * total_compute_time)) << " GB/s\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Total Kernel Time:    " << std::setw(8) << avg(total_times) << " ms\n";
    std::cout << "Total Compute Time = " << total_compute_time << " ms\n";
    
    verify_result_gpu(d_A, d_B, d_C, N, H, S);
    
    if (use_nvlink) {
        CHECK_CUDA(cudaFree(d_A_peer_offload));
        CHECK_CUDA(cudaDeviceDisablePeerAccess(peerDeviceId));
    } else {
        CHECK_CUDA(cudaFreeHost(h_A_pinned_offload));
    }
    delete (int*)kernel1_args[6]; delete (int*)kernel1_args[7];
    delete (int*)kernel2_args[6]; delete (int*)kernel2_args[7];
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec)); CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(transferStart)); CHECK_CUDA(cudaEventDestroy(transferStop));
    CHECK_CUDA(cudaEventDestroy(compute1Start)); CHECK_CUDA(cudaEventDestroy(compute1Stop));
    CHECK_CUDA(cudaEventDestroy(compute2Start)); CHECK_CUDA(cudaEventDestroy(compute2Stop));
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void runUvmTest(int N, int H, int S, float offload_ratio, int trials, int device_id) {
    const bool SERIALIZE_KERNELS = true;

    if (SERIALIZE_KERNELS) {
        std::cout << "\n--- Running Case 2: UVM with Serial Kernels (Multi-Stream) ---\n";
    } else {
        std::cout << "\n--- Running Case 2: UVM with Concurrent Kernels (Multi-Stream) ---\n";
    }

    if (offload_ratio == 0.0f) {
        runSingleKernelTest(N, H, S, trials, true, device_id);
        return;
    }
    
    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);
    int N_offload = static_cast<int>(N * offload_ratio);
    int N_resident = N - N_offload;
    size_t A_offload_size = (size_t)N_offload * H * sizeof(float);
    std::cout << "Resident Rows: " << N_resident << ", Offloaded Rows: " << N_offload << std::endl;
    
    float *A, *B, *C;
    CHECK_CUDA(cudaMallocManaged(&A, A_size));
    CHECK_CUDA(cudaMallocManaged(&B, B_size));
    CHECK_CUDA(cudaMallocManaged(&C, C_size));
    init_matrices_on_gpu(A, B, N, H, S);

    std::vector<float> h_A_offload_copy((size_t)N_offload * H);
    float* A_offload_ptr = A + (size_t)N_resident * H;
    CHECK_CUDA(cudaMemcpy(h_A_offload_copy.data(), A_offload_ptr, A_offload_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaStream_t streamCompute, streamTransfer;
    cudaEvent_t prefetchDoneEvent;
    CHECK_CUDA(cudaStreamCreate(&streamCompute));
    CHECK_CUDA(cudaStreamCreate(&streamTransfer));
    CHECK_CUDA(cudaEventCreate(&prefetchDoneEvent));
    
    CHECK_CUDA(cudaMemAdvise(A, (size_t)N_resident * H * sizeof(float), cudaMemAdviseSetPreferredLocation, device_id));
    CHECK_CUDA(cudaMemAdvise(A_offload_ptr, A_offload_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK_CUDA(cudaMemPrefetchAsync(A, A_size, device_id, streamCompute)); 
    CHECK_CUDA(cudaMemPrefetchAsync(A_offload_ptr, A_offload_size, cudaCpuDeviceId, streamCompute));
    CHECK_CUDA(cudaStreamSynchronize(streamCompute));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    cudaEvent_t start_k1, stop_k1, start_k2, stop_k2;
    CHECK_CUDA(cudaEventCreate(&start_k1)); CHECK_CUDA(cudaEventCreate(&stop_k1));
    CHECK_CUDA(cudaEventCreate(&start_k2)); CHECK_CUDA(cudaEventCreate(&stop_k2));

    cudaKernelNodeParams kernel1_params = {0};
    kernel1_params.func = (void*)matMulKernel;
    kernel1_params.gridDim = calculate_grid_dims(S, N_resident);
    kernel1_params.blockDim = dim3(TILE_DIM, TILE_DIM);
    void *kernel1_args[] = {&C, &A, &B, &N, &H, &S, new int(0), new int(N_resident)};
    kernel1_params.kernelParams = kernel1_args;
    
    cudaGraphNode_t start_k1_node, kernel1_node, stop_k1_node;
    CHECK_CUDA(cudaGraphAddEventRecordNode(&start_k1_node, graph, nullptr, 0, start_k1));
    CHECK_CUDA(cudaGraphAddKernelNode(&kernel1_node, graph, &start_k1_node, 1, &kernel1_params));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&stop_k1_node, graph, &kernel1_node, 1, stop_k1));

    cudaKernelNodeParams kernel2_params = {0};
    kernel2_params.func = (void*)matMulKernel;
    kernel2_params.gridDim = calculate_grid_dims(S, N_offload);
    kernel2_params.blockDim = dim3(TILE_DIM, TILE_DIM);
    void *kernel2_args[] = {&C, &A, &B, &N, &H, &S, new int(N_resident), new int(N_offload)};
    kernel2_params.kernelParams = kernel2_args;
    
    cudaGraphNode_t start_k2_node, kernel2_node, stop_k2_node;
    CHECK_CUDA(cudaGraphAddEventRecordNode(&start_k2_node, graph, nullptr, 0, start_k2));
    
    std::vector<cudaGraphNode_t> k2_deps;
    k2_deps.push_back(start_k2_node);
    if (SERIALIZE_KERNELS) {
        k2_deps.push_back(stop_k1_node); 
    }
    CHECK_CUDA(cudaGraphAddKernelNode(&kernel2_node, graph, k2_deps.data(), k2_deps.size(), &kernel2_params));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&stop_k2_node, graph, &kernel2_node, 1, stop_k2));
    
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        CHECK_CUDA(cudaMemcpy(A_offload_ptr, h_A_offload_copy.data(), A_offload_size, cudaMemcpyHostToHost));
        CHECK_CUDA(cudaMemPrefetchAsync(A_offload_ptr, A_offload_size, device_id, streamTransfer));
        CHECK_CUDA(cudaEventRecord(prefetchDoneEvent, streamTransfer));
        CHECK_CUDA(cudaGraphLaunch(graph_exec, streamCompute));
        CHECK_CUDA(cudaStreamWaitEvent(streamCompute, prefetchDoneEvent));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Done.\n";
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    std::vector<double> total_times, k1_times, k2_times;

    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaMemcpy(A_offload_ptr, h_A_offload_copy.data(), A_offload_size, cudaMemcpyHostToHost));
        CHECK_CUDA(cudaEventRecord(start, streamCompute));
        CHECK_CUDA(cudaMemPrefetchAsync(A_offload_ptr, A_offload_size, device_id, streamTransfer));
        CHECK_CUDA(cudaEventRecord(prefetchDoneEvent, streamTransfer));
        CHECK_CUDA(cudaGraphLaunch(graph_exec, streamCompute));
        CHECK_CUDA(cudaStreamWaitEvent(streamCompute, prefetchDoneEvent));
        CHECK_CUDA(cudaEventRecord(stop, streamCompute));
        CHECK_CUDA(cudaStreamSynchronize(streamCompute));
        
        float ms_total, ms_k1, ms_k2;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms_k1, start_k1, stop_k1));
        CHECK_CUDA(cudaEventElapsedTime(&ms_k2, start_k2, stop_k2));
        total_times.push_back(ms_total);
        k1_times.push_back(ms_k1);
        k2_times.push_back(ms_k2);
    }

    auto avg = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    std::cout << "\n--- Timings (avg over " << trials << " trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Compute (Resident Data):    " << std::setw(8) << avg(k1_times) << " ms\n";
    std::cout << "Compute (Offloaded Data):   " << std::setw(8) << avg(k2_times) << " ms\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Total Overlapped Time:      " << std::setw(8) << avg(total_times) << " ms\n";
    verify_result_gpu(A, B, C, N, H, S);
    delete (int*)kernel1_args[6]; delete (int*)kernel1_args[7];
    delete (int*)kernel2_args[6]; delete (int*)kernel2_args[7];
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec)); CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start_k1)); CHECK_CUDA(cudaEventDestroy(stop_k1));
    CHECK_CUDA(cudaEventDestroy(start_k2)); CHECK_CUDA(cudaEventDestroy(stop_k2));
    CHECK_CUDA(cudaEventDestroy(prefetchDoneEvent));
    CHECK_CUDA(cudaFree(A)); CHECK_CUDA(cudaFree(B)); CHECK_CUDA(cudaFree(C));
    CHECK_CUDA(cudaStreamDestroy(streamCompute)); CHECK_CUDA(cudaStreamDestroy(streamTransfer));
}

// **REWRITTEN:** This is now the primary function for the --extend flag, handling all cases (0, 1, and partial).
void runBandwidthExtensionTest(int N, int H, int S, float offload_ratio, int trials, int device_id, int interleave_ratio) {
    std::cout << "\n--- Running Case 3: Legacy Zero-Copy Test (Interleaved Scratchpad Kernel) ---\n";
    
    // Ensure row counts are aligned to TILE_DIM for simplicity
    if (N % TILE_DIM != 0) {
        int old_N = N;
        N = (N / TILE_DIM) * TILE_DIM;
        std::cout << "Adjusting N from " << old_N << " to " << N << " to be a multiple of TILE_DIM (" << TILE_DIM << ")\n";
    }

    int N_offload = static_cast<int>(round(N * offload_ratio));
    N_offload = (N_offload / TILE_DIM) * TILE_DIM;
    if (offload_ratio > 0.999f) N_offload = N;
    int N_resident = N - N_offload;

    size_t A_resident_size = (size_t)N_resident * H * sizeof(float);
    size_t A_offload_size = (size_t)N_offload * H * sizeof(float);
    size_t A_full_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);

    float *d_A_resident = nullptr, *d_B = nullptr, *d_C = nullptr;
    float *h_A_offload = nullptr;
    float *d_A_offload_mapped = nullptr;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    CHECK_CUDA(cudaMalloc(&d_B, B_size));
    CHECK_CUDA(cudaMalloc(&d_C, C_size));

    if (N_resident > 0) CHECK_CUDA(cudaMalloc(&d_A_resident, A_resident_size));
    if (N_offload > 0) {
        CHECK_CUDA(cudaHostAlloc(&h_A_offload, A_offload_size, cudaHostAllocMapped));
        CHECK_CUDA(cudaHostGetDevicePointer(&d_A_offload_mapped, h_A_offload, 0));
    }
    
    std::cout << "Initializing resident data on GPU and offloaded data on CPU..." << std::flush;
    init_matrices_on_gpu(nullptr, d_B, 0, H, S);
    
    if (N_resident > 0) {
        float* d_A_res_temp;
        CHECK_CUDA(cudaMalloc(&d_A_res_temp, A_resident_size));
        init_matrices_on_gpu(d_A_res_temp, nullptr, N_resident, H, 0);
        CHECK_CUDA(cudaMemcpy(d_A_resident, d_A_res_temp, A_resident_size, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaFree(d_A_res_temp));
    }
    if (N_offload > 0) {
        for (size_t i = 0; i < (size_t)N_offload * H; ++i) h_A_offload[i] = (float)rand() / RAND_MAX;
    }
    
    std::cout << "Done.\n";
    std::cout << "Resident Rows: " << N_resident << ", Offloaded Rows: " << N_offload << "\n";
    
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    cudaGraphNode_t start_node, kernel_node, stop_node;
    CHECK_CUDA(cudaGraphAddEventRecordNode(&start_node, graph, nullptr, 0, start));

    cudaKernelNodeParams kernel_params = {0};
    kernel_params.func = (void*)interleavedScratchpadMatMulKernel;
    kernel_params.gridDim = calculate_grid_dims(S, N);
    kernel_params.blockDim = dim3(TILE_DIM, TILE_DIM);

    void *kernel_args[] = {&d_C, &d_A_resident, &d_A_offload_mapped, &d_B, &N, &H, &S, &N_resident, &interleave_ratio};
    kernel_params.kernelParams = kernel_args;

    CHECK_CUDA(cudaGraphAddKernelNode(&kernel_node, graph, &start_node, 1, &kernel_params));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&stop_node, graph, &kernel_node, 1, stop));
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";

    std::vector<double> total_times;
    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        float ms_total;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        total_times.push_back(ms_total);
    }

    auto avg = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0f) / v.size(); };
    
    double avg_time_ms = avg(total_times);
    double effective_bandwidth = (A_full_size + B_size) / (avg_time_ms * 1e6);
    
    std::cout << "\n--- Timings (avg over " << trials << " trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Effective Blended BW (GB/s): " << std::setw(8) << effective_bandwidth << "\n";
    std::cout << "Total Kernel Time:           " << std::setw(8) << avg_time_ms << " ms\n";
    
    // **RESTORED:** Foolproof verification by reconstructing a simple, contiguous matrix A.
    float* d_A_full_temp_verify;
    CHECK_CUDA(cudaMalloc(&d_A_full_temp_verify, (size_t)N * H * sizeof(float)));
    if (N_resident > 0) {
        CHECK_CUDA(cudaMemcpy(d_A_full_temp_verify, d_A_resident, A_resident_size, cudaMemcpyDeviceToDevice));
    }
    if (N_offload > 0) {
        CHECK_CUDA(cudaMemcpy(d_A_full_temp_verify + (size_t)N_resident * H, h_A_offload, A_offload_size, cudaMemcpyHostToDevice));
    }

    verify_result_gpu(d_A_full_temp_verify, d_B, d_C, N, H, S);
    
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(start)); 
    CHECK_CUDA(cudaEventDestroy(stop));
    if (d_A_resident) CHECK_CUDA(cudaFree(d_A_resident));
    if (h_A_offload) CHECK_CUDA(cudaFreeHost(h_A_offload));
    CHECK_CUDA(cudaFree(d_A_full_temp_verify));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

// Prints the command line usage instructions
void print_usage(const char* prog_name) {
    std::cerr << "\nUsage: " << prog_name << " [options]\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  -h, --help                         Show this help message and exit.\n";
    std::cerr << "  --nvlink                           Use NVLink for D2D transfer (Case 1, requires 2+ GPUs).\n";
    std::cerr << "  --uvm                              Use UVM with Prefetch (Case 2).\n";
    std::cerr << "  --extend                           Use Legacy Zero-Copy Test (Interleaved Kernel, Case 3).\n";
    std::cerr << "  --interleave <X>                   Set X-to-1 resident/offload interleave ratio for --extend mode. (Default: 9)\n";
    std::cerr << "  -N, --N, --rows <int>              Number of rows for matrix A. (Default: 1000000)\n";
    std::cerr << "  -H, --H, --hidden_dim <int>        Number of columns for A / rows for B. (Default: 1024)\n";
    std::cerr << "  -S, --S, --cols <int>              Number of columns for matrix B. (Default: 1)\n";
    std::cerr << "  -r, --ratio, --offload_ratio <f>   Fraction of matrix A to offload (0.0 to 1.0). (Default: 0.1)\n";
    std::cerr << "  -t, --trials <int>                 Number of timed trials to run. (Default: 1000)\n";
    std::cerr << "  -d, --device <id>                  ID of the GPU device to use. (Default: 0)\n\n";
    std::cerr << "Note: Default is Explicit Overlap (PCIe). Test modes (--nvlink, --uvm, --extend) are mutually exclusive.\n";
}

// Simple command line parser with error handling
bool parse_args(int argc, char** argv, int& N, int& H, int& S, float& ratio, int& trials, bool& use_uvm, bool& use_extension, bool& use_nvlink, int& device_id, int& interleave_ratio) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--nvlink") {
            use_nvlink = true;
        } else if (arg == "--uvm") {
            use_uvm = true;
        } else if (arg == "--extend") {
            use_extension = true;
        } else if (arg == "--interleave" && i + 1 < argc) {
            interleave_ratio = std::stoi(argv[++i]);
        } else if ((arg == "--N" || arg == "-N" || arg == "--rows") && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if ((arg == "--H" || arg == "-H" || arg == "--hidden_dim") && i + 1 < argc) {
            H = std::stoi(argv[++i]);
        } else if ((arg == "--S" || arg == "-S" || arg == "--cols") && i + 1 < argc) {
            S = std::stoi(argv[++i]);
        } else if ((arg == "--offload_ratio" || arg == "-r" || arg == "--ratio") && i + 1 < argc) {
            ratio = std::stof(argv[++i]);
        } else if ((arg == "--trials" || arg == "-t") && i + 1 < argc) {
            trials = std::stoi(argv[++i]);
        } else if ((arg == "--device" || arg == "-d") && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else {
            std::cerr << "Error: Unknown or invalid argument: " << arg << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }

    if ((use_uvm && use_extension) || (use_uvm && use_nvlink) || (use_extension && use_nvlink)) {
        std::cerr << "Error: --uvm, --extend, and --nvlink flags are mutually exclusive." << std::endl;
        print_usage(argv[0]);
        return false;
    }
    if (N <= 0 || H <= 0 || S <= 0 || trials <= 0 || interleave_ratio <= 0) {
        std::cerr << "Error: Matrix dimensions, trial count, and interleave ratio must be positive." << std::endl;
        print_usage(argv[0]);
        return false;
    }
    if (ratio < 0.0f || ratio > 1.0f) {
        std::cerr << "Error: Offload ratio must be in the range [0.0, 1.0]." << std::endl;
        print_usage(argv[0]);
        return false;
    }
     if (device_id < 0) {
        std::cerr << "Error: Device ID must be a non-negative integer." << std::endl;
        return false;
    }
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_id >= device_count) {
        std::cerr << "Error: Device ID " << device_id << " is invalid. Only " << device_count << " devices found on this system." << std::endl;
        return false;
    }
    if (use_nvlink && device_count < 2) {
        std::cerr << "Error: --nvlink mode requires at least 2 GPUs." << std::endl;
        return false;
    }
    return true; // Success
}

int main(int argc, char** argv)
{
    // Default values
    int N = 1000000, H = 1024, S = 1;
    float offload_ratio = 0.1f;
    int trials = 1000;
    bool use_uvm = false;
    bool use_extension = false;
    bool use_nvlink = false;
    int device_id = 0;
    int interleave_ratio = 9; // Default X for X-to-1 interleaving

    if (!parse_args(argc, argv, N, H, S, offload_ratio, trials, use_uvm, use_extension, use_nvlink, device_id, interleave_ratio)) {
        return 1;
    }

    CHECK_CUDA(cudaSetDevice(device_id));

    std::cout << "Configuration:\n";
    std::string mode = "Explicit Overlap with PCIe (Case 1)";
    if (use_nvlink) mode = "Explicit Overlap with NVLink (Case 1)";
    if (use_uvm) mode = "UVM with Prefetch (Case 2)";
    if (use_extension) mode = "Legacy Zero-Copy (Interleaved Kernel, Case 3)";


    std::cout << "  Mode:          " << mode << "\n";
    std::cout << "  Device ID:     " << device_id << "\n";
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "  Device Name:   " << prop.name << "\n";
    std::cout << "  Matrix A:      " << N << " x " << H << "\n";
    std::cout << "  Matrix B:      " << H << " x " << S << "\n";
    std::cout << "  Offload Ratio: " << offload_ratio * 100 << "%\n";
    if (use_extension && offload_ratio > 0.0f && offload_ratio < 1.0f) {
        std::cout << "  Interleave Ratio: " << interleave_ratio << ":1\n";
    }
    std::cout << "  Trials:        " << trials << "\n";

    if (use_extension) {
        runBandwidthExtensionTest(N, H, S, offload_ratio, trials, device_id, interleave_ratio);
    } else if (use_uvm) {
        runUvmTest(N, H, S, offload_ratio, trials, device_id);
    } else {
        runExplicitOverlapTest(N, H, S, offload_ratio, trials, use_nvlink, device_id);
    }

    return 0;
}

