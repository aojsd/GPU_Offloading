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

// cuBLAS library
#include <cublas_v2.h>

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

#define CHECK_CUBLAS(call) do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d code=%d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief GPU-side timestamping kernel.
 * Reads the 64-bit nanosecond-resolution global timer and writes it to memory.
 */
__global__ void recordTimestamp(unsigned long long* timestamp_out) {
    asm("mov.u64 %0, %%globaltimer;" : "=l"(*timestamp_out));
}

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
    // --- 1. Remapping Logic: Map Physical Block to Logical Tile ---
    
    long long total_tile_rows = (N + TILE_DIM - 1) / TILE_DIM;
    long long linear_block_id = (long long)blockIdx.z * gridDim.y + blockIdx.y;

    if (linear_block_id >= total_tile_rows) return;

    long long logical_tile_idx;

    // If interleave_ratio is 0 or less, disable interleaving for baseline tests.
    if (interleave_ratio <= 0) {
        // No interleaving: physical block ID maps directly to logical tile ID.
        logical_tile_idx = linear_block_id;
    } else {
        // Interleaving is enabled: use the two-pointer spatial logic.
        long long group_size = interleave_ratio + 1;
        long long group_id = linear_block_id / group_size;
        long long idx_in_group = linear_block_id % group_size;

        if (idx_in_group < interleave_ratio) {
            // This block is assigned a tile from the FRONT of the matrix.
            logical_tile_idx = group_id * interleave_ratio + idx_in_group;
        } else {
            // This block is assigned a tile from the BACK of the matrix.
            logical_tile_idx = (total_tile_rows - 1) - group_id;
        }
    }

    // --- 2. Shared Memory and Thread ID Setup ---

    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // --- 3. Tile Computation ---

    int logical_start_row = logical_tile_idx * TILE_DIM;
    int tile_start_col = blockIdx.x * TILE_DIM;

    float C_val = 0.0f;

    for (int t = 0; t < (H + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        // --- 3a. Cooperative Data Loading ---
        
        int row_to_load = logical_start_row + ty;
        int col_to_load = t * TILE_DIM + tx;

        if (row_to_load < N && col_to_load < H) {
            if (row_to_load < N_resident) {
                sA[ty][tx] = A_resident[ (long long)row_to_load * H + col_to_load ];
            } else {
                long long offload_row_index = (long long)row_to_load - N_resident;
                sA[ty][tx] = A_offload[ offload_row_index * H + col_to_load ];
            }
        } else {
            sA[ty][tx] = 0.0f;
        }
        
        int b_row = t * TILE_DIM + ty;
        if (b_row < H && tile_start_col + tx < S) {
            sB[ty][tx] = B[ (long long)b_row * S + (tile_start_col + tx) ];
        } else {
            sB[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        // --- 3b. Shared Memory Computation ---

        for (int k = 0; k < TILE_DIM; ++k) {
            C_val += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // --- 4. Write Result to Global Memory ---

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

void verify_result_gpu(const float* d_A, const float* d_B, const float* d_C_original, int N, int H, int S) {
    std::cout << "\nVerifying result on GPU using cuBLAS... " << std::flush;
    
    size_t C_size = (size_t)N * S * sizeof(float);
    float* d_C_verify;
    CHECK_CUDA(cudaMalloc(&d_C_verify, C_size));

    // --- Use cuBLAS for verification ---
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Using the default stream (0) for verification is sufficient.
    // The cublasSgemm call will be synchronous with respect to the host after it returns.
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             S, N, H, &alpha,
                             d_B, S,
                             d_A, H, &beta,
                             d_C_verify, S));

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

    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaFree(d_C_verify));
    CHECK_CUDA(cudaFree(d_max_error));
}


/**
 * @brief Measures the baseline performance of a single, fully-resident matrix multiplication
 * using a reusable CUDA Graph and precise GPU-side timers.
 */
void runSingleKernelTest(int N, int H, int S, int trials, bool use_uvm, int device_id) {
    std::cout << "Zero offload ratio detected. Running simplified single-kernel test (CUDA Graph Version).\n";

    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));

    // --- Memory Allocation (UVM or Standard) ---
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

    // --- Graph Capture of the sgemm operation ---
    unsigned long long* d_timestamps;
    CHECK_CUDA(cudaMalloc(&d_timestamps, 2 * sizeof(unsigned long long)));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    recordTimestamp<<<1, 1, 0, stream>>>(d_timestamps);
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             S, N, H, &alpha,
                             d_B, S,
                             d_A, H, &beta,
                             d_C, S));
    recordTimestamp<<<1, 1, 0, stream>>>(d_timestamps + 1);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    // --- Warm-up and Timed Trials using Graph Launch ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";

    std::vector<double> kernel_times_ns;
    unsigned long long h_timestamps[2];
    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpy(h_timestamps, d_timestamps, sizeof(h_timestamps), cudaMemcpyDeviceToHost));
        kernel_times_ns.push_back((double)(h_timestamps[1] - h_timestamps[0]));
    }

    // --- Calculate and Report Metrics (Identical Output) ---
    double avg_time_ns = std::accumulate(kernel_times_ns.begin(), kernel_times_ns.end(), 0.0) / kernel_times_ns.size();
    double avg_time_ms = avg_time_ns / 1.0e6;
    
    double effective_bandwidth = (A_size) / (avg_time_ms / 1000.0) / 1e9;
    double gflops = (2.0 * N * S * H) / (avg_time_ms / 1000.0) / 1e9;

    std::cout << "\n--- Timings (avg over " << trials << " graph trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GFLOPS:                   " << std::setw(8) << gflops << "\n";
    std::cout << "Effective Bandwidth (GB/s): " << std::setw(8) << effective_bandwidth << "\n";
    std::cout << "Total Kernel Time:        " << std::setw(8) << avg_time_ms << " ms\n";

    verify_result_gpu(d_A, d_B, d_C, N, H, S);
    
    // --- Cleanup ---
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaFree(d_timestamps));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaStreamDestroy(stream));
}


/**
 * @brief Models a scenario where a portion of matrix A must be transferred from a remote source
 * (another GPU via NVLink or Host RAM via PCIe) while computation on already-resident
 * data occurs simultaneously.
 *
 * @param N Number of rows in matrix A and C.
 * @param H Number of columns in A / rows in B.
 * @param S Number of columns in matrix B and C.
 * @param offload_ratio The fraction of matrix A that is not resident and must be transferred.
 * @param trials The number of timed trials to run.
 * @param use_nvlink If true, the offloaded data comes from a peer GPU. If false, from pinned host memory.
 * @param device_id The primary GPU device to run the computation on.
 */
void runExplicitOverlapTest(int N, int H, int S, float offload_ratio, int trials, bool use_nvlink, int device_id) {
    if (use_nvlink) {
        std::cout << "\n--- Running Case 1: Explicit Overlap with NVLink (CUDA Graph Version) ---\n";
    } else {
        std::cout << "\n--- Running Case 1: Explicit Overlap with PCIe (CUDA Graph Version) ---\n";
    }

    if (offload_ratio == 0.0f) {
        runSingleKernelTest(N, H, S, trials, false, device_id);
        return;
    }

    // --- 1. Initial Setup and Data Partitioning ---
    const int ALIGNMENT = 32;
    if (N % ALIGNMENT != 0) {
        int old_N = N;
        N = (N / ALIGNMENT) * ALIGNMENT;
        std::cout << "Adjusting N from " << old_N << " to " << N << " to be a multiple of " << ALIGNMENT << "\n";
    }
    int N_offload = static_cast<int>(round(N * offload_ratio));
    N_offload = (N_offload / ALIGNMENT) * ALIGNMENT;
    if (offload_ratio > 0.999f) N_offload = N;
    int N_resident = N - N_offload;

    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);
    size_t A_offload_size = (size_t)N_offload * H * sizeof(float);
    std::cout << "Resident Rows: " << N_resident << ", Offloaded Rows: " << N_offload << std::endl;

    float* h_A_pinned_offload = nullptr;
    float* d_A_peer_offload = nullptr;
    cudaMemcpyKind copyKind;
    int peerDeviceId = -1;

    if (use_nvlink) {
        int device_count;
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        if (device_count < 2) { std::cerr << "Error: NVLink test requires at least 2 GPUs." << std::endl; exit(EXIT_FAILURE); }
        peerDeviceId = (device_id + 1) % device_count;
        int canAccessPeer;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, device_id, peerDeviceId));
        if (canAccessPeer) {
            std::cout << "Enabling peer access from Device " << device_id << " to Device " << peerDeviceId << std::endl;
            CHECK_CUDA(cudaSetDevice(device_id));
            CHECK_CUDA(cudaDeviceEnablePeerAccess(peerDeviceId, 0));
        } else { std::cerr << "Error: Peer access is not supported." << std::endl; exit(EXIT_FAILURE); }
        CHECK_CUDA(cudaSetDevice(peerDeviceId));
        CHECK_CUDA(cudaMalloc(&d_A_peer_offload, A_offload_size));
        CHECK_CUDA(cudaSetDevice(device_id));
        copyKind = cudaMemcpyDeviceToDevice;
    } else {
        CHECK_CUDA(cudaHostAlloc(&h_A_pinned_offload, A_offload_size, cudaHostAllocDefault));
        copyKind = cudaMemcpyHostToDevice;
    }

    // --- 2. Allocate and Initialize Matrices ---
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

    // --- 3. Setup Streams, cuBLAS, and Graph Primitives ---
    cudaStream_t copy_stream, compute_stream;
    CHECK_CUDA(cudaStreamCreate(&copy_stream));
    CHECK_CUDA(cudaStreamCreate(&compute_stream));

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    unsigned long long* d_timestamps;
    const int NUM_TIMESTAMPS = 8;
    CHECK_CUDA(cudaMalloc(&d_timestamps, NUM_TIMESTAMPS * sizeof(unsigned long long)));

    // --- 4. Capture the Operations into a CUDA Graph ---
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    
    cudaEvent_t ghost_link_event, copy_finished_event;
    CHECK_CUDA(cudaEventCreate(&ghost_link_event));
    CHECK_CUDA(cudaEventCreate(&copy_finished_event));

    CHECK_CUDA(cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeThreadLocal));

    // -- Step A: Create a "ghost" dependency to pull the copy_stream into the graph --
    recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + 0); // Overall Start
    CHECK_CUDA(cudaEventRecord(ghost_link_event, compute_stream));
    CHECK_CUDA(cudaStreamWaitEvent(copy_stream, ghost_link_event, 0));

    // -- Step B: Enqueue all the real concurrent work --
    void* memcpy_src = use_nvlink ? (void*)d_A_peer_offload : (void*)h_A_pinned_offload;
    recordTimestamp<<<1, 1, 0, copy_stream>>>(d_timestamps + 1); // Transfer Start
    CHECK_CUDA(cudaMemcpyAsync(d_A + (size_t)N_resident * H, memcpy_src, A_offload_size, copyKind, copy_stream));
    recordTimestamp<<<1, 1, 0, copy_stream>>>(d_timestamps + 2); // Transfer Stop
    CHECK_CUDA(cudaEventRecord(copy_finished_event, copy_stream));

    CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));
    recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + 3); // Compute1 Start
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, S, N_resident, H, &alpha, d_B, S, d_A, H, &beta, d_C, S));
    recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + 4); // Compute1 Stop

    // -- Step C: Establish the real dependency, which is now valid --
    CHECK_CUDA(cudaStreamWaitEvent(compute_stream, copy_finished_event, 0));
    
    // -- Step D: Enqueue the final dependent work --
    recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + 5); // Compute2 Start
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, S, N_offload, H, &alpha,
                             d_B, S, d_A + (size_t)N_resident * H, H, &beta,
                             d_C + (size_t)N_resident * S, S));
    recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + 6); // Compute2 Stop
    recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + 7); // Overall Stop

    CHECK_CUDA(cudaStreamEndCapture(compute_stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    // --- 5. Execute the Graph and Get Timings ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, compute_stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(compute_stream));
    std::cout << "Done.\n";

    std::vector<double> total_times, transfer_times, compute1_times, compute2_times;
    unsigned long long h_timestamps[NUM_TIMESTAMPS];

    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, compute_stream));
        CHECK_CUDA(cudaStreamSynchronize(compute_stream));
        CHECK_CUDA(cudaMemcpy(h_timestamps, d_timestamps, sizeof(h_timestamps), cudaMemcpyDeviceToHost));

        transfer_times.push_back((double)(h_timestamps[2] - h_timestamps[1]));
        compute1_times.push_back((double)(h_timestamps[4] - h_timestamps[3]));
        compute2_times.push_back((double)(h_timestamps[6] - h_timestamps[5]));
        total_times.push_back((double)(h_timestamps[7] - h_timestamps[0]));
    }
    
    // --- 6. Report Averaged Results ---
    auto avg_ns_to_ms = [](const std::vector<double>& v) {
        return (std::accumulate(v.begin(), v.end(), 0.0) / v.size()) / 1.0e6;
    };
    double avg_transfer_ms = avg_ns_to_ms(transfer_times);
    double avg_compute1_ms = avg_ns_to_ms(compute1_times);
    double avg_compute2_ms = avg_ns_to_ms(compute2_times);
    double avg_total_kernel_ms = avg_ns_to_ms(total_times);
    double total_compute_time = avg_compute1_ms + avg_compute2_ms;

    std::string bw_label = use_nvlink ? "NVLink Transfer (D2D):" : "PCIe Transfer (H2D): ";
    std::cout << "\n--- Timings (avg over " << trials << " graph trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << bw_label << std::setw(8) << avg_transfer_ms << " ms\n";
    std::cout << "Compute (Resident Data):  " << std::setw(8) << avg_compute1_ms << " ms\n";
    std::cout << "Compute (Offloaded Data): " << std::setw(8) << avg_compute2_ms << " ms\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Comm. Bandwidth (GB/s):   " << std::setw(8) << (A_offload_size / (1e6 * avg_transfer_ms)) << "\n";
    std::cout << "GPU Throughput (GB/s):    " << std::setw(8) << (A_size / (1e6 * total_compute_time)) << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Total Kernel Time:        " << std::setw(8) << avg_total_kernel_ms << " ms\n";
    std::cout << "Total Compute Time:       " << total_compute_time << " ms\n";
    
    verify_result_gpu(d_A, d_B, d_C, N, H, S);

    // --- 7. Cleanup ---
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    if (use_nvlink) { CHECK_CUDA(cudaFree(d_A_peer_offload)); CHECK_CUDA(cudaDeviceDisablePeerAccess(peerDeviceId)); } 
    else { CHECK_CUDA(cudaFreeHost(h_A_pinned_offload)); }
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaEventDestroy(ghost_link_event));
    CHECK_CUDA(cudaEventDestroy(copy_finished_event));
    CHECK_CUDA(cudaStreamDestroy(copy_stream));
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaFree(d_timestamps));
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
}

/**
 * @brief Models a scenario using Unified Memory where data resident on the CPU is prefetched
 * to the GPU, overlapping with computation on data already resident on the GPU.
 *
 * @param N Number of rows in matrix A and C.
 * @param H Number of columns in A / rows in B.
 * @param S Number of columns in matrix B and C.
 * @param offload_ratio The fraction of matrix A that is initially resident on the CPU.
 * @param trials The number of timed trials to run.
 * @param device_id The primary GPU device to run the computation on.
 */
void runUvmTest(int N, int H, int S, float offload_ratio, int trials, int device_id) {
    // This flag controls whether the two compute kernels can run concurrently
    // or if the second must wait for the first to complete.
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

    // Ensure row counts are aligned to a reasonable number for simplicity (e.g., 32)
    const int ALIGNMENT = 32;
    if (N % ALIGNMENT != 0) {
        int old_N = N;
        N = (N / ALIGNMENT) * ALIGNMENT;
        std::cout << "Adjusting N from " << old_N << " to " << N << " to be a multiple of " << ALIGNMENT << "\n";
    }

    int N_offload = static_cast<int>(round(N * offload_ratio));
    N_offload = (N_offload / ALIGNMENT) * ALIGNMENT;
    if (offload_ratio > 0.999f) N_offload = N;
    int N_resident = N - N_offload;
    
    // --- 1. Initial Setup and UVM Allocation ---
    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);
    size_t A_offload_size = (size_t)N_offload * H * sizeof(float);
    std::cout << "Resident Rows: " << N_resident << ", Offloaded Rows: " << N_offload << std::endl;
    
    float *A, *B, *C;
    CHECK_CUDA(cudaMallocManaged(&A, A_size));
    CHECK_CUDA(cudaMallocManaged(&B, B_size));
    CHECK_CUDA(cudaMallocManaged(&C, C_size));
    init_matrices_on_gpu(A, B, N, H, S);

    // Create a host-side copy of the data that will be "offloaded"
    std::vector<float> h_A_offload_copy((size_t)N_offload * H);
    float* A_offload_ptr = A + (size_t)N_resident * H;
    CHECK_CUDA(cudaMemcpy(h_A_offload_copy.data(), A_offload_ptr, A_offload_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- 2. Setup Streams, Events, and cuBLAS Handle ---
    cudaStream_t streamCompute, streamTransfer;
    cudaEvent_t prefetchDoneEvent, k1_done_event;
    CHECK_CUDA(cudaStreamCreate(&streamCompute));
    CHECK_CUDA(cudaStreamCreate(&streamTransfer));
    CHECK_CUDA(cudaEventCreate(&prefetchDoneEvent));
    CHECK_CUDA(cudaEventCreate(&k1_done_event));

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    
    // --- 3. Set Initial Memory Locations with UVM Hints ---
    // Resident part of A prefers the GPU, offloaded part prefers the CPU initially.
    CHECK_CUDA(cudaMemAdvise(A, (size_t)N_resident * H * sizeof(float), cudaMemAdviseSetPreferredLocation, device_id));
    CHECK_CUDA(cudaMemAdvise(A_offload_ptr, A_offload_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK_CUDA(cudaMemAdvise(B, B_size, cudaMemAdviseSetPreferredLocation, device_id));
    CHECK_CUDA(cudaMemAdvise(C, C_size, cudaMemAdviseSetPreferredLocation, device_id));

    // Prefetch all data to their preferred locations to establish a clean state.
    CHECK_CUDA(cudaMemPrefetchAsync(A, A_size, device_id, streamCompute));
    CHECK_CUDA(cudaMemPrefetchAsync(B, B_size, device_id, streamCompute));
    CHECK_CUDA(cudaMemPrefetchAsync(C, C_size, device_id, streamCompute));
    // Crucially, move the offloaded part back to the CPU after the initial prefetch.
    CHECK_CUDA(cudaMemPrefetchAsync(A_offload_ptr, A_offload_size, cudaCpuDeviceId, streamCompute));
    CHECK_CUDA(cudaStreamSynchronize(streamCompute));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // --- 4. Warm-up Runs ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        // Reset the offloaded data on the host side
        CHECK_CUDA(cudaMemcpy(A_offload_ptr, h_A_offload_copy.data(), A_offload_size, cudaMemcpyHostToHost));
        // Prefetch the offloaded data to the GPU in the transfer stream
        CHECK_CUDA(cudaMemPrefetchAsync(A_offload_ptr, A_offload_size, device_id, streamTransfer));
        // Launch resident computation in the compute stream
        CHECK_CUBLAS(cublasSetStream(cublas_handle, streamCompute));
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, S, N_resident, H, &alpha, B, S, A, H, &beta, C, S));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Done.\n";
    
    // --- 5. Timed Trials ---
    cudaEvent_t start, stop, start_k1, stop_k1, start_k2, stop_k2;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&start_k1)); CHECK_CUDA(cudaEventCreate(&stop_k1));
    CHECK_CUDA(cudaEventCreate(&start_k2)); CHECK_CUDA(cudaEventCreate(&stop_k2));
    
    std::vector<double> total_times, k1_times, k2_times;
    for (int i = 0; i < trials; ++i) {
        // Reset the offloaded data on the host before each trial
        CHECK_CUDA(cudaMemcpy(A_offload_ptr, h_A_offload_copy.data(), A_offload_size, cudaMemcpyHostToHost));
        
        CHECK_CUDA(cudaEventRecord(start, streamCompute));

        // Operation 1: Start prefetching the offloaded data from CPU to GPU
        CHECK_CUDA(cudaMemPrefetchAsync(A_offload_ptr, A_offload_size, device_id, streamTransfer));
        CHECK_CUDA(cudaEventRecord(prefetchDoneEvent, streamTransfer));

        // Operation 2: Launch computation on resident data, overlapping with the prefetch
        CHECK_CUBLAS(cublasSetStream(cublas_handle, streamCompute));
        CHECK_CUDA(cudaEventRecord(start_k1, streamCompute));
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, S, N_resident, H, &alpha, B, S, A, H, &beta, C, S));
        CHECK_CUDA(cudaEventRecord(stop_k1, streamCompute));
        
        // If serializing, record an event that the second kernel must wait on.
        if (SERIALIZE_KERNELS) {
            CHECK_CUDA(cudaEventRecord(k1_done_event, streamCompute));
        }

        // --- Create Dependencies and Launch Final Operation ---
        // The compute stream must wait for the prefetch to finish before the second kernel.
        CHECK_CUDA(cudaStreamWaitEvent(streamCompute, prefetchDoneEvent, 0));
        if (SERIALIZE_KERNELS) {
            CHECK_CUDA(cudaStreamWaitEvent(streamCompute, k1_done_event, 0));
        }

        // Operation 3: Launch computation on the now-resident offloaded data
        CHECK_CUDA(cudaEventRecord(start_k2, streamCompute));
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 S, N_offload, H, &alpha,
                                 B, S,
                                 A_offload_ptr, H, &beta,
                                 C + (size_t)N_resident * S, S));
        CHECK_CUDA(cudaEventRecord(stop_k2, streamCompute));

        CHECK_CUDA(cudaEventRecord(stop, streamCompute));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        // --- Collect Timings ---
        float ms_total, ms_k1, ms_k2;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms_k1, start_k1, stop_k1));
        CHECK_CUDA(cudaEventElapsedTime(&ms_k2, start_k2, stop_k2));
        total_times.push_back(ms_total);
        k1_times.push_back(ms_k1);
        k2_times.push_back(ms_k2);
    }

    // --- 6. Report Results ---
    auto avg = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    std::cout << "\n--- Timings (avg over " << trials << " trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Compute (Resident Data):    " << std::setw(8) << avg(k1_times) << " ms\n";
    std::cout << "Compute (Offloaded Data):   " << std::setw(8) << avg(k2_times) << " ms\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Total Overlapped Time:      " << std::setw(8) << avg(total_times) << " ms\n";
    
    verify_result_gpu(A, B, C, N, H, S);

    // --- 7. Cleanup ---
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start_k1)); CHECK_CUDA(cudaEventDestroy(stop_k1));
    CHECK_CUDA(cudaEventDestroy(start_k2)); CHECK_CUDA(cudaEventDestroy(stop_k2));
    CHECK_CUDA(cudaEventDestroy(prefetchDoneEvent));
    CHECK_CUDA(cudaEventDestroy(k1_done_event));
    CHECK_CUDA(cudaFree(A)); CHECK_CUDA(cudaFree(B)); CHECK_CUDA(cudaFree(C));
    CHECK_CUDA(cudaStreamDestroy(streamCompute)); CHECK_CUDA(cudaStreamDestroy(streamTransfer));
}

/**
 * @brief Adaptation of the simultaneous memory access test using two concurrent cuBLAS kernels
 * instead of a single custom kernel. One kernel accesses VRAM while the other accesses
 * zero-copy host memory, aiming to measure the aggregate bandwidth.
 *
 * @param N Number of rows in matrix A and C.
 * @param H Number of columns in A / rows in B.
 * @param S Number of columns in matrix B and C.
 * @param offload_ratio The fraction of matrix A that is resident in host memory.
 * @param trials The number of timed trials to run.
 * @param device_id The primary GPU device to run the computation on.
 * @param interleave_ratio (Note: This parameter is no longer used by the kernels but is kept for signature consistency).
 */
void runBandwidthExtensionTest(int N, int H, int S, float offload_ratio, int trials, int device_id, int interleave_ratio) {
    std::cout << "\n--- Running Case 3: Sim. Zero-Copy w/ Concurrent cuBLAS Kernels ---\n";
    
    // Ensure row counts are aligned to a reasonable number for simplicity (e.g., 32)
    const int ALIGNMENT = 32;
    if (N % ALIGNMENT != 0) {
        int old_N = N;
        N = (N / ALIGNMENT) * ALIGNMENT;
        std::cout << "Adjusting N from " << old_N << " to " << N << " to be a multiple of " << ALIGNMENT << "\n";
    }

    int N_offload = static_cast<int>(round(N * offload_ratio));
    N_offload = (N_offload / ALIGNMENT) * ALIGNMENT;
    if (offload_ratio > 0.999f) N_offload = N;
    int N_resident = N - N_offload;

    // --- 1. Data Allocation ---
    size_t A_resident_size = (size_t)N_resident * H * sizeof(float);
    size_t A_offload_size = (size_t)N_offload * H * sizeof(float);
    size_t A_full_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);

    float *d_A_resident = nullptr, *d_B = nullptr, *d_C = nullptr;
    float *h_A_offload = nullptr;
    float *d_A_offload_mapped = nullptr;

    CHECK_CUDA(cudaMalloc(&d_B, B_size));
    CHECK_CUDA(cudaMalloc(&d_C, C_size));

    if (N_resident > 0) CHECK_CUDA(cudaMalloc(&d_A_resident, A_resident_size));
    if (N_offload > 0) {
        CHECK_CUDA(cudaHostAlloc(&h_A_offload, A_offload_size, cudaHostAllocMapped));
        CHECK_CUDA(cudaHostGetDevicePointer(&d_A_offload_mapped, h_A_offload, 0));
    }
    
    // --- 2. Data Initialization ---
    std::cout << "Initializing resident data on GPU and offloaded data on CPU..." << std::flush;
    init_matrices_on_gpu(nullptr, d_B, 0, H, S);
    
    float* d_A_full_temp; // Temporary full matrix on GPU for easy initialization
    CHECK_CUDA(cudaMalloc(&d_A_full_temp, A_full_size));
    init_matrices_on_gpu(d_A_full_temp, nullptr, N, H, 0);

    if (N_resident > 0) {
        CHECK_CUDA(cudaMemcpy(d_A_resident, d_A_full_temp, A_resident_size, cudaMemcpyDeviceToDevice));
    }
    if (N_offload > 0) {
        CHECK_CUDA(cudaMemcpy(h_A_offload, d_A_full_temp + (size_t)N_resident * H, A_offload_size, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaFree(d_A_full_temp));
    std::cout << "Done.\n";
    std::cout << "Resident Rows: " << N_resident << ", Offloaded Rows: " << N_offload << "\n";
    
    // --- 3. Setup Streams, Events, and cuBLAS ---
    cudaStream_t stream_resident, stream_offload;
    CHECK_CUDA(cudaStreamCreate(&stream_resident));
    CHECK_CUDA(cudaStreamCreate(&stream_offload));

    cublasHandle_t handle_resident, handle_offload;
    CHECK_CUBLAS(cublasCreate(&handle_resident));
    CHECK_CUBLAS(cublasCreate(&handle_offload));
    CHECK_CUBLAS(cublasSetStream(handle_resident, stream_resident));
    CHECK_CUBLAS(cublasSetStream(handle_offload, stream_offload));

    cudaEvent_t start, stop, start_res, stop_res, start_off, stop_off;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&start_res)); CHECK_CUDA(cudaEventCreate(&stop_res));
    CHECK_CUDA(cudaEventCreate(&start_off)); CHECK_CUDA(cudaEventCreate(&stop_off));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // --- 4. Warm-up Runs ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        if (N_resident > 0) {
             CHECK_CUBLAS(cublasSgemm(handle_resident, CUBLAS_OP_N, CUBLAS_OP_N, S, N_resident, H, &alpha, d_B, S, d_A_resident, H, &beta, d_C, S));
        }
        if (N_offload > 0) {
             CHECK_CUBLAS(cublasSgemm(handle_offload, CUBLAS_OP_N, CUBLAS_OP_N, S, N_offload, H, &alpha, d_B, S, d_A_offload_mapped, H, &beta, d_C + (size_t)N_resident * S, S));
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Done.\n";

    // --- 5. Timed Trials ---
    std::vector<double> total_times, resident_times, offload_times;
    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaEventRecord(start, stream_resident)); // Overall start time

        if (N_resident > 0) {
            CHECK_CUDA(cudaEventRecord(start_res, stream_resident));
            CHECK_CUBLAS(cublasSgemm(handle_resident, CUBLAS_OP_N, CUBLAS_OP_N, S, N_resident, H, &alpha, d_B, S, d_A_resident, H, &beta, d_C, S));
            CHECK_CUDA(cudaEventRecord(stop_res, stream_resident));
        }
        if (N_offload > 0) {
            CHECK_CUDA(cudaEventRecord(start_off, stream_offload));
            CHECK_CUBLAS(cublasSgemm(handle_offload, CUBLAS_OP_N, CUBLAS_OP_N, S, N_offload, H, &alpha, d_B, S, d_A_offload_mapped, H, &beta,
                                     d_C + (size_t)(N_resident * S), S));
            CHECK_CUDA(cudaEventRecord(stop_off, stream_offload));
        }
        
        // Record overall stop time in the stream that is expected to finish last, or sync both.
        // Syncing both streams is the most robust way to measure total time.
        CHECK_CUDA(cudaStreamSynchronize(stream_resident));
        CHECK_CUDA(cudaStreamSynchronize(stream_offload));
        CHECK_CUDA(cudaEventRecord(stop, stream_resident));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms_total = 0, ms_resident = 0, ms_offload = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        if (N_resident > 0) CHECK_CUDA(cudaEventElapsedTime(&ms_resident, start_res, stop_res));
        if (N_offload > 0) CHECK_CUDA(cudaEventElapsedTime(&ms_offload, start_off, stop_off));
        
        total_times.push_back(ms_total);
        resident_times.push_back(ms_resident);
        offload_times.push_back(ms_offload);
    }

    // --- 6. Report Results ---
    auto avg = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0f) / v.size(); };
    
    double avg_time_ms = avg(total_times);
    double effective_bandwidth = (A_full_size + B_size) / (avg_time_ms * 1e6);
    
    std::cout << "\n--- Timings (avg over " << trials << " trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Compute (Resident Data):       " << std::setw(8) << avg(resident_times) << " ms\n";
    std::cout << "Compute (Offloaded Data):      " << std::setw(8) << avg(offload_times) << " ms\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Total Overlapped Time:         " << std::setw(8) << avg_time_ms << " ms\n";
    std::cout << "Effective Blended BW (GB/s):   " << std::setw(8) << effective_bandwidth << "\n";
    
    // --- 7. Verification and Cleanup ---
    float* d_A_full_temp_verify;
    CHECK_CUDA(cudaMalloc(&d_A_full_temp_verify, (size_t)N * H * sizeof(float)));
    if (N_resident > 0) {
        CHECK_CUDA(cudaMemcpy(d_A_full_temp_verify, d_A_resident, A_resident_size, cudaMemcpyDeviceToDevice));
    }
    if (N_offload > 0) {
        CHECK_CUDA(cudaMemcpy(d_A_full_temp_verify + (size_t)N_resident * H, h_A_offload, A_offload_size, cudaMemcpyHostToDevice));
    }

    verify_result_gpu(d_A_full_temp_verify, d_B, d_C, N, H, S);
    
    CHECK_CUBLAS(cublasDestroy(handle_resident));
    CHECK_CUBLAS(cublasDestroy(handle_offload));
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start_res)); CHECK_CUDA(cudaEventDestroy(stop_res));
    CHECK_CUDA(cudaEventDestroy(start_off)); CHECK_CUDA(cudaEventDestroy(stop_off));
    if (d_A_resident) CHECK_CUDA(cudaFree(d_A_resident));
    if (h_A_offload) CHECK_CUDA(cudaFreeHost(h_A_offload));
    CHECK_CUDA(cudaFree(d_A_full_temp_verify));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaStreamDestroy(stream_resident));
    CHECK_CUDA(cudaStreamDestroy(stream_offload));
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
    if (N <= 0 || H <= 0 || S <= 0 || trials <= 0) {
        std::cerr << "Error: Matrix dimensions, and trial count must be positive." << std::endl;
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

