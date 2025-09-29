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
 * - Uses two CUDA streams: one for computation, one for data transfer.
 * - The offloaded portion of A is allocated in pinned host memory.
 * - An async copy is launched on the transfer stream.
 * - Can occur over PCIe, NVLink, or C2C (for GH200)
 * - The first compute kernel (on resident data) is launched on the compute stream.
 * - A CUDA event ensures the compute stream waits for the transfer to finish before
 * launching the second compute kernel (on the now-transferred data).
 *
 * Case 2: Overlap with UVM Prefetch
 * - Matrices are allocated with `cudaMallocManaged`.
 * - An async prefetch of the offloaded data is launched concurrently
 * with the main compute kernel, allowing the driver to overlap the transfer and
 * computation.
 *
 * Case 3: Bandwidth Extension (UVM on Grace Hopper)
 * - Allocate matrices with UVM.
 * - Use UVM to allow the GPU to directly access data in CPU memory simultaneously
 * with data in its own HBM. This is not a data migration strategy.
 * - Goal is to show the effective memory bandwidth increase from the unified memory pool.
 * - Applicable only to Grace-Hopper systems with shared address translation services (ATS).
 * - E.g., GH200 superchip
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
        // This would only happen for astronomically large matrices
        fprintf(stderr, "Error: Matrix dimensions exceed launch capabilities.\n");
        exit(EXIT_FAILURE);
    }

    return dim3((num_elements_x + TILE_DIM - 1) / TILE_DIM, (unsigned int)grid_y, (unsigned int)grid_z);
}

// Tiled Matrix Multiplication Kernel
// Computes C = A * B for a specific subset of rows in A and C.
// Now handles a 2D (y,z) grid for rows to support N > ~2 million.
__global__ void matMulKernel(float *C, const float *A, const float *B, int N, int H, int S, int startRow, int numRows)
{
    long long total_row_blocks = (numRows + TILE_DIM - 1) / TILE_DIM;
    
    // Reconstruct the linear block index for the Y-dimension from the 2D grid
    long long linear_block_idx_y = (long long)blockIdx.z * gridDim.y + blockIdx.y;

    // Guard against excess blocks from grid folding
    if (linear_block_idx_y >= total_row_blocks) {
        return;
    }

    // Shared memory for tiles of A and B
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block index, mapped to the output matrix C
    int row = linear_block_idx_y * TILE_DIM + ty + startRow;
    int col = blockIdx.x * TILE_DIM + tx;

    float C_val = 0.0f;
    
    // Loop over the tiles of A and B required to compute one tile of C
    for (int t = 0; t < (H + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        // Load tile of A into shared memory
        if (row < (startRow + numRows) && (t * TILE_DIM + tx) < H) {
            sA[ty][tx] = A[(long long)row * H + t * TILE_DIM + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        if (col < S && (t * TILE_DIM + ty) < H) {
            sB[ty][tx] = B[(long long)(t * TILE_DIM + ty) * S + col];
        } else {
            sB[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        // Multiply tiles
        for (int k = 0; k < TILE_DIM; ++k)
        {
            C_val += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Write final value to C
    if (row < (startRow + numRows) && col < S)
    {
        C[(long long)row * S + col] = C_val;
    }
}

// New kernel to compare two matrices on the GPU and find the maximum absolute error
__global__ void compareAndFindMaxErrorKernel(const float* C_original, const float* C_verify, float* d_max_error, size_t total_elements) {
    // Grid-stride loop to have each thread process multiple elements
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += gridDim.x * blockDim.x) {
        float error = fabsf(C_original[i] - C_verify[i]);
        
        // atomicMax is not natively supported for floats in global memory.
        // We implement it using a compare-and-swap (CAS) loop.
        float current_max = *d_max_error;
        // Loop until our value is not greater than the current max
        while (error > current_max) {
            // Try to swap our error value in, assuming the max is still current_max
            float previous_max = __uint_as_float(atomicCAS((unsigned int*)d_max_error, __float_as_uint(current_max), __float_as_uint(error)));
            // If the CAS succeeded, previous_max will be equal to current_max. We are done with this iteration.
            if (previous_max == current_max) {
                break;
            }
            // If it failed, another thread wrote a larger value. Update our current_max and retry.
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
    for (size_t i = id; i < num_elements; i += stride) {
        matrix[i] = curand_uniform(&local_state);
    }
    states[id] = local_state;
}

// Host function to orchestrate matrix initialization on the GPU
void init_matrices_on_gpu(float* d_A, float* d_B, int N, int H, int S) {
    std::cout << "Initializing matrices on GPU... " << std::flush;
    
    size_t num_elements_A = (size_t)N * H;
    size_t num_elements_B = (size_t)H * S;
    size_t max_elements = std::max(num_elements_A, num_elements_B);

    // Setup cuRAND states
    int threads_per_block = 256;
    int blocks = std::min(16384, (int)((max_elements + threads_per_block - 1) / threads_per_block));
    size_t num_states = (size_t)threads_per_block * blocks;

    curandState_t* d_rand_states;
    CHECK_CUDA(cudaMalloc(&d_rand_states, num_states * sizeof(curandState_t)));
    setupCurandKernel<<<blocks, threads_per_block>>>(d_rand_states, time(NULL));

    // Initialize matrices
    initMatrixGpuKernel<<<blocks, threads_per_block>>>(d_A, num_elements_A, d_rand_states);
    initMatrixGpuKernel<<<blocks, threads_per_block>>>(d_B, num_elements_B, d_rand_states);
    
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

    // Re-calculate the full result on the GPU without any offloading
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid = calculate_grid_dims(S, N);
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C_verify, d_A, d_B, N, H, S, 0, N);

    // Prepare for reduction kernel to find max error
    float* d_max_error;
    float h_max_error = 0.0f;
    CHECK_CUDA(cudaMalloc(&d_max_error, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_max_error, 0, sizeof(float)));

    // Launch comparison kernel
    int compare_threads = 256;
    int compare_blocks = std::min(1024, (int)(((size_t)N * S + compare_threads - 1) / compare_threads));
    compareAndFindMaxErrorKernel<<<compare_blocks, compare_threads>>>(d_C_original, d_C_verify, d_max_error, (size_t)N * S);
    
    // Copy the single float result back
    CHECK_CUDA(cudaMemcpy(&h_max_error, d_max_error, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure copy is complete

    std::cout << "Done.\nMaximum absolute error: " << h_max_error << std::endl;

    // Clean up temporary buffers
    CHECK_CUDA(cudaFree(d_C_verify));
    CHECK_CUDA(cudaFree(d_max_error));
}


// This function handles the "fast path" case where offload_ratio is 0.
void runSingleKernelTest(int N, int H, int S, int trials, bool use_uvm, int deviceId) {
    std::cout << "Zero offload ratio detected. Running simplified single-kernel test (CUDA Graph Version).\n";

    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Handle memory allocation based on the mode
    if (use_uvm) {
        CHECK_CUDA(cudaMallocManaged(&d_A, A_size));
        CHECK_CUDA(cudaMallocManaged(&d_B, B_size));
        CHECK_CUDA(cudaMallocManaged(&d_C, C_size));
        init_matrices_on_gpu(d_A, d_B, N, H, S);

        // Advise and prefetch everything to the GPU since nothing is offloaded
        CHECK_CUDA(cudaMemAdvise(d_A, A_size, cudaMemAdviseSetPreferredLocation, deviceId));
        CHECK_CUDA(cudaMemAdvise(d_B, B_size, cudaMemAdviseSetPreferredLocation, deviceId));
        CHECK_CUDA(cudaMemAdvise(d_C, C_size, cudaMemAdviseSetPreferredLocation, deviceId));
        CHECK_CUDA(cudaMemPrefetchAsync(d_A, A_size, deviceId, stream));
        CHECK_CUDA(cudaMemPrefetchAsync(d_B, B_size, deviceId, stream));
        CHECK_CUDA(cudaMemPrefetchAsync(d_C, C_size, deviceId, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    } else {
        CHECK_CUDA(cudaMalloc(&d_A, A_size));
        CHECK_CUDA(cudaMalloc(&d_B, B_size));
        CHECK_CUDA(cudaMalloc(&d_C, C_size));
        init_matrices_on_gpu(d_A, d_B, N, H, S);
    }

    // --- CUDA Graph Setup ---
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
    // --- End Graph Setup ---

    // --- Warm-up Section ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";
    // --- End Warm-up ---

    std::vector<float> kernel_times;
    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        kernel_times.push_back(ms);
    }

    auto avg = [](const std::vector<float>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    };

    double avg_time_ms = avg(kernel_times);
    double effective_bandwidth = (A_size + B_size) / (avg_time_ms * 1e6);

    std::cout << "\n--- Timings (avg over " << trials << " trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Effective Bandwidth (GB/s): " << std::setw(8) << effective_bandwidth << "\n";
    std::cout << "Total Kernel Time:          " << std::setw(8) << avg_time_ms << " ms\n";

    verify_result_gpu(d_A, d_B, d_C, N, H, S);
    
    // Cleanup
    delete (int*)kernel_args[6]; // Clean up dynamically allocated startRow
    delete (int*)kernel_args[7]; // Clean up dynamically allocated numRows
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaStreamDestroy(stream));
}


void runExplicitOverlapTest(int N, int H, int S, float offload_ratio, int trials) {
    std::cout << "\n--- Running Case 1: Explicit Overlap (CUDA Graph Version) ---\n";
    if (offload_ratio == 0.0f) {
        runSingleKernelTest(N, H, S, trials, false, 0);
        return;
    }

    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);
    int N_offload = static_cast<int>(N * offload_ratio);
    int N_resident = N - N_offload;
    size_t A_offload_size = (size_t)N_offload * H * sizeof(float);
    
    std::cout << "Resident Rows: " << N_resident << ", Offloaded Rows: " << N_offload << std::endl;

    float *d_A, *d_B, *d_C;
    float* h_A_pinned_offload;
    CHECK_CUDA(cudaMalloc(&d_A, A_size));
    CHECK_CUDA(cudaMalloc(&d_B, B_size));
    CHECK_CUDA(cudaMalloc(&d_C, C_size));
    CHECK_CUDA(cudaHostAlloc(&h_A_pinned_offload, A_offload_size, cudaHostAllocDefault));

    init_matrices_on_gpu(d_A, d_B, N, H, S);
    CHECK_CUDA(cudaMemcpy(h_A_pinned_offload, d_A + (size_t)N_resident * H, A_offload_size, cudaMemcpyDeviceToHost));
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // --- CUDA Graph Setup ---
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    // Create all events once
    cudaEvent_t start, stop, transferStart, transferStop, compute1Start, compute1Stop, compute2Start, compute2Stop;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&transferStart)); CHECK_CUDA(cudaEventCreate(&transferStop));
    CHECK_CUDA(cudaEventCreate(&compute1Start)); CHECK_CUDA(cudaEventCreate(&compute1Stop));
    CHECK_CUDA(cudaEventCreate(&compute2Start)); CHECK_CUDA(cudaEventCreate(&compute2Stop));

    // Define nodes
    cudaGraphNode_t start_node, stop_node, memcpy_node, kernel1_node, kernel2_node, sync_node;
    cudaGraphNode_t event_transfer_start, event_transfer_stop, event_compute1_start, event_compute1_stop, event_compute2_start, event_compute2_stop;

    CHECK_CUDA(cudaGraphAddEventRecordNode(&start_node, graph, nullptr, 0, start));

    // Branch 1: Transfer
    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_transfer_start, graph, &start_node, 1, transferStart));
    CHECK_CUDA(cudaGraphAddMemcpyNode1D(&memcpy_node, graph, &event_transfer_start, 1, d_A + (size_t)N_resident * H, h_A_pinned_offload, A_offload_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_transfer_stop, graph, &memcpy_node, 1, transferStop));

    // Branch 2: Compute on Resident Data
    cudaKernelNodeParams kernel1_params = {0};
    kernel1_params.func = (void*)matMulKernel;
    kernel1_params.gridDim = calculate_grid_dims(S, N_resident);
    kernel1_params.blockDim = dim3(TILE_DIM, TILE_DIM);
    void *kernel1_args[] = {&d_C, &d_A, &d_B, &N, &H, &S, new int(0), new int(N_resident)};
    kernel1_params.kernelParams = kernel1_args;
    
    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_compute1_start, graph, &start_node, 1, compute1Start));
    CHECK_CUDA(cudaGraphAddKernelNode(&kernel1_node, graph, &event_compute1_start, 1, &kernel1_params));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&event_compute1_stop, graph, &kernel1_node, 1, compute1Stop));

    // Synchronization Point: Wait for both branches to complete
    cudaGraphNode_t sync_deps[] = {event_transfer_stop, event_compute1_stop};
    CHECK_CUDA(cudaGraphAddEmptyNode(&sync_node, graph, sync_deps, 2));

    // Branch 3: Compute on Offloaded Data
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
    // --- End Graph Setup ---

    // --- Warm-up Section ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        // Must reset state before each run, even for warm-ups
        CHECK_CUDA(cudaMemcpy(h_A_pinned_offload, d_A + (size_t)N_resident * H, A_offload_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";
    // --- End Warm-up ---

    std::vector<double> total_times, transfer_times, compute1_times, compute2_times;
    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaMemcpy(h_A_pinned_offload, d_A + (size_t)N_resident * H, A_offload_size, cudaMemcpyDeviceToHost));
        
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
    
    // Reporting is unchanged
    auto avg = [](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0f) / v.size(); };
    std::cout << "\n--- Timings (avg over " << trials << " trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "PCIe Transfer (HtoD):     " << std::setw(8) << avg(transfer_times) << " ms\n";
    std::cout << "Compute (Resident Data):  " << std::setw(8) << avg(compute1_times) << " ms\n";
    std::cout << "Compute (Offloaded Data): " << std::setw(8) << avg(compute2_times) << " ms\n";
    std::cout << "--------------------------------------\n";
    std::cout << "PCIe Bandwidth (GB/s): " << std::setw(8) << (A_offload_size / (1e6 * avg(transfer_times))) << " GB/s\n";
    double total_compute_time = avg(compute1_times) + avg(compute2_times);
    std::cout << "GPU Throughput (GB/s): " << std::setw(8) << ((A_size + B_size) / (1e6 * total_compute_time)) << " GB/s\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Total Kernel Time:    " << std::setw(8) << avg(total_times) << " ms\n";
    std::cout << "Total Compute Time = " << total_compute_time << " ms\n";
    
    verify_result_gpu(d_A, d_B, d_C, N, H, S);
    
    // Cleanup
    delete (int*)kernel1_args[6]; delete (int*)kernel1_args[7];
    delete (int*)kernel2_args[6]; delete (int*)kernel2_args[7];
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec)); CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(transferStart)); CHECK_CUDA(cudaEventDestroy(transferStop));
    CHECK_CUDA(cudaEventDestroy(compute1Start)); CHECK_CUDA(cudaEventDestroy(compute1Stop));
    CHECK_CUDA(cudaEventDestroy(compute2Start)); CHECK_CUDA(cudaEventDestroy(compute2Stop));
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFreeHost(h_A_pinned_offload)); CHECK_CUDA(cudaStreamDestroy(stream));
}

void runUvmTest(int N, int H, int S, float offload_ratio, int trials, int deviceId) {
    // --- KERNEL SCHEDULING TOGGLE ---
    // Set to 'true' to force Kernel 2 to wait for Kernel 1 (serial execution).
    // Set to 'false' to allow both kernels to run concurrently.
    const bool SERIALIZE_KERNELS = true;

    if (SERIALIZE_KERNELS) {
        std::cout << "\n--- Running Case 2: UVM with Serial Kernels (Multi-Stream) ---\n";
    } else {
        std::cout << "\n--- Running Case 2: UVM with Concurrent Kernels (Multi-Stream) ---\n";
    }

    if (offload_ratio == 0.0f) {
        runSingleKernelTest(N, H, S, trials, true, deviceId);
        return;
    }
    
    // --- Setup ---
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
    
    // Initial memory placement
    CHECK_CUDA(cudaMemAdvise(A, (size_t)N_resident * H * sizeof(float), cudaMemAdviseSetPreferredLocation, deviceId));
    CHECK_CUDA(cudaMemAdvise(A_offload_ptr, A_offload_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK_CUDA(cudaMemPrefetchAsync(A, A_size, deviceId, streamCompute)); 
    CHECK_CUDA(cudaMemPrefetchAsync(A_offload_ptr, A_offload_size, cudaCpuDeviceId, streamCompute));
    CHECK_CUDA(cudaStreamSynchronize(streamCompute));

    // --- Graph Setup ---
    // The graph is built with two main branches for the compute kernels.
    // A boolean flag controls whether a dependency is added to serialize them.
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    // Create event pairs for timing each kernel individually
    cudaEvent_t start_k1, stop_k1, start_k2, stop_k2;
    CHECK_CUDA(cudaEventCreate(&start_k1)); CHECK_CUDA(cudaEventCreate(&stop_k1));
    CHECK_CUDA(cudaEventCreate(&start_k2)); CHECK_CUDA(cudaEventCreate(&stop_k2));

    // -- Branch 1: Kernel on Resident Data --
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

    // -- Branch 2: Kernel on Offloaded Data --
    cudaKernelNodeParams kernel2_params = {0};
    kernel2_params.func = (void*)matMulKernel;
    kernel2_params.gridDim = calculate_grid_dims(S, N_offload);
    kernel2_params.blockDim = dim3(TILE_DIM, TILE_DIM);
    void *kernel2_args[] = {&C, &A, &B, &N, &H, &S, new int(N_resident), new int(N_offload)};
    kernel2_params.kernelParams = kernel2_args;
    
    cudaGraphNode_t start_k2_node, kernel2_node, stop_k2_node;
    CHECK_CUDA(cudaGraphAddEventRecordNode(&start_k2_node, graph, nullptr, 0, start_k2));
    
    // -- Add Kernel 2 with Conditional Dependency --
    std::vector<cudaGraphNode_t> k2_deps;
    k2_deps.push_back(start_k2_node); // Kernel 2 always depends on its own start event.
    if (SERIALIZE_KERNELS) {
        // If serialized, add a dependency on the completion of Kernel 1.
        k2_deps.push_back(stop_k1_node); 
    }
    CHECK_CUDA(cudaGraphAddKernelNode(&kernel2_node, graph, k2_deps.data(), k2_deps.size(), &kernel2_params));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&stop_k2_node, graph, &kernel2_node, 1, stop_k2));
    
    // -- Instantiate the Graph for Execution --
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    
    // --- Warm-up Section ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        // Reset state by copying from the host buffer to the UVM allocation.
        // This forces the UVM driver to make the pages resident on the CPU.
        CHECK_CUDA(cudaMemcpy(A_offload_ptr, h_A_offload_copy.data(), A_offload_size, cudaMemcpyHostToHost));
        
        // 1. Start the asynchronous prefetch on the transfer stream.
        CHECK_CUDA(cudaMemPrefetchAsync(A_offload_ptr, A_offload_size, deviceId, streamTransfer));
        CHECK_CUDA(cudaEventRecord(prefetchDoneEvent, streamTransfer));
        
        // 2. Launch the compute graph on the compute stream. It can start executing
        //    the resident kernel (Kernel 1) concurrently with the prefetch.
        CHECK_CUDA(cudaGraphLaunch(graph_exec, streamCompute));
        
        // 3. Make the compute stream wait for the prefetch to complete before proceeding.
        CHECK_CUDA(cudaStreamWaitEvent(streamCompute, prefetchDoneEvent));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Done.\n";
    
    // --- Timed Trials ---
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    std::vector<double> total_times, k1_times, k2_times;

    for (int i = 0; i < trials; ++i) {
        // Reset state by copying from the host buffer to the UVM allocation.
        CHECK_CUDA(cudaMemcpy(A_offload_ptr, h_A_offload_copy.data(), A_offload_size, cudaMemcpyHostToHost));

        // Record a start event for the total overlapped time.
        CHECK_CUDA(cudaEventRecord(start, streamCompute));

        // 1. Start the asynchronous prefetch on the transfer stream.
        CHECK_CUDA(cudaMemPrefetchAsync(A_offload_ptr, A_offload_size, deviceId, streamTransfer));
        CHECK_CUDA(cudaEventRecord(prefetchDoneEvent, streamTransfer));
        
        // 2. Launch the compute graph on the compute stream to overlap with the prefetch.
        CHECK_CUDA(cudaGraphLaunch(graph_exec, streamCompute));
        
        // 3. Make the compute stream wait for the prefetch to complete. This ensures the
        //    'stop' event below only fires after both the transfer and all compute are done.
        CHECK_CUDA(cudaStreamWaitEvent(streamCompute, prefetchDoneEvent));
        
        // Record a stop event to capture the total time.
        CHECK_CUDA(cudaEventRecord(stop, streamCompute));
        
        // Wait for all work on the compute stream to finish before reading timers.
        CHECK_CUDA(cudaStreamSynchronize(streamCompute));
        
        // Collect timings for this run.
        float ms_total, ms_k1, ms_k2;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms_k1, start_k1, stop_k1));
        CHECK_CUDA(cudaEventElapsedTime(&ms_k2, start_k2, stop_k2));
        total_times.push_back(ms_total);
        k1_times.push_back(ms_k1);
        k2_times.push_back(ms_k2);
    }

    // --- Reporting & Cleanup ---
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

// Special case for Bandwidth Extension test where 100% of A is offloaded.
void runFullCpuOffloadTest(int N, int H, int S, int trials, int deviceId) {
    std::cout << "100% offload ratio detected. Running simplified single-kernel CPU offload test.\n";

    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);

    float *A, *B, *C;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Allocate all UVM memory
    CHECK_CUDA(cudaMallocManaged(&A, A_size));
    CHECK_CUDA(cudaMallocManaged(&B, B_size));
    CHECK_CUDA(cudaMallocManaged(&C, C_size));
    
    // Initialize data on the GPU (will pull pages to GPU)
    init_matrices_on_gpu(A, B, N, H, S);

    // CRITICAL: Force A and B to be resident on the CPU. C will be written by GPU.
    std::cout << "Forcing matrices A and B to CPU memory..." << std::flush;
    CHECK_CUDA(cudaMemAdvise(A, A_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK_CUDA(cudaMemAdvise(B, B_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK_CUDA(cudaMemAdvise(C, C_size, cudaMemAdviseSetPreferredLocation, deviceId));

    // This is the enforcement step. Physically move the data.
    CHECK_CUDA(cudaMemPrefetchAsync(A, A_size, cudaCpuDeviceId, stream));
    CHECK_CUDA(cudaMemPrefetchAsync(B, B_size, cudaCpuDeviceId, stream));
    CHECK_CUDA(cudaMemPrefetchAsync(C, C_size, deviceId, stream)); // Prefetch C to GPU
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";

    // --- CUDA Graph Setup ---
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
    void *kernel_args[] = {&C, &A, &B, &N, &H, &S, new int(0), new int(N)};
    kernel_params.kernelParams = kernel_args;
    
    cudaGraphNode_t start_node, stop_node, kernel_node;
    CHECK_CUDA(cudaGraphAddEventRecordNode(&start_node, graph, nullptr, 0, start));
    CHECK_CUDA(cudaGraphAddKernelNode(&kernel_node, graph, &start_node, 1, &kernel_params));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&stop_node, graph, &kernel_node, 1, stop));
    
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    // --- End Graph Setup ---

    // --- Warm-up Section ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";
    // --- End Warm-up ---

    std::vector<float> kernel_times;
    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        kernel_times.push_back(ms);
    }

    auto avg = [](const std::vector<float>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    };

    double avg_time_ms = avg(kernel_times);
    double effective_bandwidth = (A_size + B_size) / (avg_time_ms * 1e6);

    std::cout << "\n--- Timings (avg over " << trials << " trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Effective Bandwidth (GB/s): " << std::setw(8) << effective_bandwidth << "\n";
    std::cout << "Total Kernel Time:          " << std::setw(8) << avg_time_ms << " ms\n";

    verify_result_gpu(A, B, C, N, H, S);
    
    // Cleanup
    delete (int*)kernel_args[6];
    delete (int*)kernel_args[7];
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

// Test case for UVM bandwidth extension on integrated systems like Grace Hopper.
void runBandwidthExtensionTest(int N, int H, int S, float offload_ratio, int trials, int deviceId) {
    std::cout << "\n--- Running Case 3: Bandwidth Extension (UVM, Single Kernel) ---\n";
    
    // Handle special cases
    if (offload_ratio == 0.0f) {
        runSingleKernelTest(N, H, S, trials, true, deviceId);
        return;
    }
    if (offload_ratio == 1.0f) {
        runFullCpuOffloadTest(N, H, S, trials, deviceId);
        return;
    }
    
    // --- Setup ---
    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);

    int N_offload = static_cast<int>(N * offload_ratio);
    int N_resident = N - N_offload;
    size_t A_resident_size = (size_t)N_resident * H * sizeof(float);
    size_t A_offload_size = (size_t)N_offload * H * sizeof(float);

    float *A, *B, *C;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Allocate all UVM memory
    CHECK_CUDA(cudaMallocManaged(&A, A_size));
    CHECK_CUDA(cudaMallocManaged(&B, B_size));
    CHECK_CUDA(cudaMallocManaged(&C, C_size));
    
    // Initialize data on the GPU (will pull pages to GPU)
    init_matrices_on_gpu(A, B, N, H, S);

    // CRITICAL: Set memory policy and enforce it
    float* A_offload_ptr = A + (size_t)N_resident * H;
    std::cout << "Forcing " << offload_ratio * 100 << "% of A to CPU memory..." << std::flush;

    // 1. Advise the driver on our intent
    CHECK_CUDA(cudaMemAdvise(A, A_resident_size, cudaMemAdviseSetPreferredLocation, deviceId));
    CHECK_CUDA(cudaMemAdvise(A_offload_ptr, A_offload_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK_CUDA(cudaMemAdvise(B, B_size, cudaMemAdviseSetPreferredLocation, deviceId));
    CHECK_CUDA(cudaMemAdvise(C, C_size, cudaMemAdviseSetPreferredLocation, deviceId));

    // 2. Enforce the location by physically moving the data
    CHECK_CUDA(cudaMemPrefetchAsync(A, A_resident_size, deviceId, stream));
    CHECK_CUDA(cudaMemPrefetchAsync(A_offload_ptr, A_offload_size, cudaCpuDeviceId, stream));
    CHECK_CUDA(cudaMemPrefetchAsync(B, B_size, deviceId, stream));
    CHECK_CUDA(cudaMemPrefetchAsync(C, C_size, deviceId, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";
    std::cout << "Resident Rows: " << N_resident << ", Offloaded Rows: " << N_offload << std::endl;

    // --- CUDA Graph Setup ---
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    cudaKernelNodeParams kernel_params = {0};
    kernel_params.func = (void*)matMulKernel;
    kernel_params.gridDim = calculate_grid_dims(S, N); // Single kernel covers all rows
    kernel_params.blockDim = dim3(TILE_DIM, TILE_DIM);
    void *kernel_args[] = {&C, &A, &B, &N, &H, &S, new int(0), new int(N)};
    kernel_params.kernelParams = kernel_args;
    
    cudaGraphNode_t start_node, stop_node, kernel_node;
    CHECK_CUDA(cudaGraphAddEventRecordNode(&start_node, graph, nullptr, 0, start));
    CHECK_CUDA(cudaGraphAddKernelNode(&kernel_node, graph, &start_node, 1, &kernel_params));
    CHECK_CUDA(cudaGraphAddEventRecordNode(&stop_node, graph, &kernel_node, 1, stop));
    
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    // --- End Graph Setup ---

    // --- Warm-up Section ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";
    // --- End Warm-up ---

    std::vector<float> kernel_times;
    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        kernel_times.push_back(ms);
    }

    auto avg = [](const std::vector<float>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    };
    
    double avg_time_ms = avg(kernel_times);
    double effective_bandwidth = (A_size + B_size) / (avg_time_ms * 1e6);

    std::cout << "\n--- Timings (avg over " << trials << " trials) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Effective Bandwidth (GB/s): " << std::setw(8) << effective_bandwidth << "\n";
    std::cout << "Total Kernel Time:          " << std::setw(8) << avg_time_ms << " ms\n";
    
    verify_result_gpu(A, B, C, N, H, S);
    
    // Cleanup
    delete (int*)kernel_args[6];
    delete (int*)kernel_args[7];
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

// Prints the command line usage instructions
void print_usage(const char* prog_name) {
    std::cerr << "\nUsage: " << prog_name << " [options]\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  -h, --help                         Show this help message and exit.\n";
    std::cerr << "  --uvm                              Use UVM with Prefetch (Case 2).\n";
    std::cerr << "  --extend                           Use UVM for Bandwidth Extension (Case 3, for Grace Hopper).\n";
    std::cerr << "  -N, --N, --rows <int>              Number of rows for matrix A. (Default: 1000000)\n";
    std::cerr << "  -H, --H, --hidden_dim <int>        Number of columns for A / rows for B. (Default: 1024)\n";
    std::cerr << "  -S, --S, --cols <int>              Number of columns for matrix B. (Default: 1)\n";
    std::cerr << "  -r, --ratio, --offload_ratio <f>   Fraction of matrix A to offload (0.0 to 1.0). (Default: 0.1)\n";
    std::cerr << "  -t, --trials <int>                 Number of timed trials to run. (Default: 1000)\n";
    std::cerr << "  -d, --device <id>                  ID of the GPU device to use. (Default: 0)\n\n";
    std::cerr << "Note: Default is Explicit Overlap (Case 1). --uvm and --extend are mutually exclusive.\n";
}

// Simple command line parser with error handling
bool parse_args(int argc, char** argv, int& N, int& H, int& S, float& ratio, int& trials, bool& use_uvm, bool& use_extension, int& device_id) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false; // Signal to exit
        } else if (arg == "--uvm") {
            use_uvm = true;
        } else if (arg == "--extend") {
            use_extension = true;
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
            return false; // Signal to exit
        }
    }

    // Validate inputs
    if (use_uvm && use_extension) {
        std::cerr << "Error: --uvm and --extend flags are mutually exclusive." << std::endl;
        print_usage(argv[0]);
        return false;
    }
    if (N <= 0 || H <= 0 || S <= 0 || trials <= 0) {
        std::cerr << "Error: Matrix dimensions and trial count must be positive." << std::endl;
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
    int device_id = 0;

    if (!parse_args(argc, argv, N, H, S, offload_ratio, trials, use_uvm, use_extension, device_id)) {
        return 1; // Exit if args are invalid or help was requested
    }

    CHECK_CUDA(cudaSetDevice(device_id));

    std::cout << "Configuration:\n";
    std::string mode = "Explicit Overlap (Case 1)";
    if (use_uvm) mode = "UVM with Prefetch (Case 2)";
    if (use_extension) mode = "Bandwidth Extension (Case 3)";

    std::cout << "  Mode:          " << mode << "\n";
    std::cout << "  Device ID:     " << device_id << "\n";
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "  Device Name:   " << prop.name << "\n";
    if (use_extension && prop.major < 9) {
        std::cout << "\nWarning: Bandwidth Extension test is designed for Hopper (sm_90) or newer GPUs\n";
        std::cout << "         on integrated systems like Grace Hopper. Results may not be meaningful.\n";
    }
    std::cout << "  Matrix A:      " << N << " x " << H << "\n";
    std::cout << "  Matrix B:      " << H << " x " << S << "\n";
    std::cout << "  Offload Ratio: " << offload_ratio * 100 << "%\n";
    std::cout << "  Trials:        " << trials << "\n";

    if (use_extension) {
        runBandwidthExtensionTest(N, H, S, offload_ratio, trials, device_id);
    } else if (use_uvm) {
        runUvmTest(N, H, S, offload_ratio, trials, device_id);
    } else {
        runExplicitOverlapTest(N, H, S, offload_ratio, trials);
    }

    return 0;
}
