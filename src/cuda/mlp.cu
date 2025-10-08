/**
 * @file mlp.cu
 * @brief Empirically tests a pipelined offloading strategy for the WEIGHTS of a multi-layer MLP.
 *
 * This version implements the multiplication as W(H,H) * A(H,N) and partitions
 * the weight matrix W row-wise, which is a cleaner and more robust formulation.
 */
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)
#define CHECK_CUBLAS(call) do { cublasStatus_t status = call; if (status != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at %s:%d code=%d\n", __FILE__, __LINE__, status); exit(EXIT_FAILURE); } } while (0)

__global__ void recordTimestamp(unsigned long long* timestamp_out) {
    asm("mov.u64 %0, %%globaltimer;" : "=l"(*timestamp_out));
}

__global__ void reluKernel(float* data, size_t num_elements) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < num_elements; i += stride) {
        data[i] = fmaxf(0.0f, data[i]);
    }
}

/**
 * @brief Runs a baseline, non-offloaded MLP forward pass to measure pure compute performance.
 *
 * This function builds a CUDA graph that contains only the cublasSgemm and reluKernel
 * operations for each layer, with all data resident on the GPU. It serves as the
 * performance baseline to compare against the offloading strategies.
 */
void runBaselineMlpTest(int H, int N, int num_layers, int trials) {
    std::cout << "\n--- Running Non-Offloaded Baseline MLP Test ---" << std::endl;
    
    // --- 1. Memory Allocation ---
    std::vector<float*> d_weights(num_layers);
    for (int i = 0; i < num_layers; ++i) CHECK_CUDA(cudaMalloc(&d_weights[i], (size_t)H * H * sizeof(float)));
    float *d_activations_A, *d_activations_B;
    CHECK_CUDA(cudaMalloc(&d_activations_A, (size_t)H * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_activations_B, (size_t)H * N * sizeof(float)));
    
    std::vector<float*> p_input_activations(num_layers);
    std::vector<float*> p_output_activations(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        p_input_activations[i] = (i % 2 == 0) ? d_activations_A : d_activations_B;
        p_output_activations[i] = (i % 2 == 0) ? d_activations_B : d_activations_A;
    }

    // --- 2. CUDA Graph Construction ---
    std::cout << "Building Baseline MLP CUDA Graph..." << std::flush;
    cudaStream_t compute_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle)); 
    CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));
    const float alpha = 1.0f, beta = 0.0f;

    const int TIMESTAMPS_PER_LAYER = 2; // Start and End for each layer
    unsigned long long* d_timestamps;
    CHECK_CUDA(cudaMalloc(&d_timestamps, num_layers * TIMESTAMPS_PER_LAYER * sizeof(unsigned long long)));

    cudaGraph_t graph; 
    cudaGraphExec_t graph_exec;
    
    CHECK_CUDA(cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < num_layers; ++i) {
        recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 0); // Layer i Start
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, H, H, &alpha, p_input_activations[i], N, d_weights[i], H, &beta, p_output_activations[i], N));
        reluKernel<<<((size_t)H * N + 255) / 256, 256, 0, compute_stream>>>(p_output_activations[i], (size_t)H * N);
        recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 1); // Layer i End
    }
    CHECK_CUDA(cudaStreamEndCapture(compute_stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    std::cout << " Done.\n";

    // --- 3. Execution and Timing ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs..." << std::flush;
    for(int i = 0; i < WARMUP_COUNT; ++i) CHECK_CUDA(cudaGraphLaunch(graph_exec, compute_stream));
    CHECK_CUDA(cudaStreamSynchronize(compute_stream));
    std::cout << " Done.\n";

    unsigned long long* h_timestamps = new unsigned long long[num_layers * TIMESTAMPS_PER_LAYER];
    std::vector<double> all_layer_times, all_forward_pass_times;

    std::cout << "Running " << trials << " timed trials..." << std::flush;
    for(int t = 0; t < trials; ++t) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, compute_stream));
        CHECK_CUDA(cudaStreamSynchronize(compute_stream));
        CHECK_CUDA(cudaMemcpy(h_timestamps, d_timestamps, num_layers * TIMESTAMPS_PER_LAYER * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        double current_pass_time_ns = 0;
        for (int i = 0; i < num_layers; ++i) {
            double layer_time = (double)(h_timestamps[i*TIMESTAMPS_PER_LAYER+1] - h_timestamps[i*TIMESTAMPS_PER_LAYER+0]);
            all_layer_times.push_back(layer_time);
            current_pass_time_ns += layer_time;
        }
        all_forward_pass_times.push_back(current_pass_time_ns);
    }
    std::cout << " Done.\n";

    // --- 4. Reporting with GPU Throughput ---
    auto avg_ns_to_ms = [](const std::vector<double>& v) { 
        if (v.empty()) return 0.0; 
        return (std::accumulate(v.begin(), v.end(), 0.0) / v.size()) / 1.0e6; 
    };
    
    double avg_per_layer_ms = avg_ns_to_ms(all_layer_times);
    double avg_forward_pass_ms = avg_ns_to_ms(all_forward_pass_times);

    // ✅ Add throughput calculation
    double total_weights_size_gb = (double)num_layers * H * H * sizeof(float) / 1.0e9;
    double observed_gpu_throughput = avg_forward_pass_ms > 0 ? (total_weights_size_gb / (avg_forward_pass_ms / 1000.0)) : 0.0;

    std::cout << "\n--- Baseline Performance (No Offloading) ---\n";
    std::cout << "\n--- Average Timings per Layer (over " << trials * num_layers << " samples) ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Compute1 (Resident):      " << std::setw(8) << avg_per_layer_ms << " ms\n";
    std::cout << "Compute2 (Offloaded):     " << std::setw(8) << 0 << " ms\n";
    std::cout << "Transfer1 (Main):         " << std::setw(8) << 0 << " ms\n";
    std::cout << "Transfer2 (Pre-fetch):    " << std::setw(8) << 0 << " ms\n";
    
    std::cout << "\n--- Overall Performance Metrics ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Observed Transfer BW:     " << std::setw(8) << 0 << " GB/s\n";
    std::cout << "Observed GPU Throughput:  " << std::setw(8) << observed_gpu_throughput << " GB/s\n";
    std::cout << "Avg. Per-Layer Time:      " << std::setw(8) << avg_per_layer_ms << " ms (Wall Clock)\n";
    std::cout << "Avg. Forward Pass Time:   " << std::setw(8) << avg_forward_pass_ms << " ms (Wall Clock)\n";


    // --- 5. Cleanup ---
    delete[] h_timestamps;
    for (auto& p : d_weights) CHECK_CUDA(cudaFree(p));
    CHECK_CUDA(cudaFree(d_activations_A)); CHECK_CUDA(cudaFree(d_activations_B));
    CHECK_CUDA(cudaFree(d_timestamps));
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
}

void runMlpTest(int H, int N, int num_layers, double x, bool use_nvlink, int trials) {
    int device_id = 0;
    CHECK_CUDA(cudaSetDevice(device_id));

    std::cout << "\n--- Test Configuration ---\n";
    std::cout << "Mode:                         " << (use_nvlink ? "NVLink (D2D)" : "PCIe (H2D)") << "\n";
    std::cout << "Layers:                       " << num_layers << "\n";
    std::cout << "Hidden Dim (H):               " << H << "\n";
    std::cout << "Batch Size (N):               " << N << "\n";
    std::cout << "Trials:                       " << trials << "\n";
    std::cout << "Target Comm. Ratio (x):       " << x << "\n";
    std::cout << "------------------------------------------\n";
    if (x <= 0.0) {
        runBaselineMlpTest(H, N, num_layers, trials);
        return;
    }

    // --- 1. Calculate Sizes & Offload Strategy (Based on H) ---
    const int ALIGNMENT = 32;
    int H_offload_L1 = static_cast<int>(round((double)H / (x + 1.0) / ALIGNMENT) * ALIGNMENT);
    int H_resident_L1 = H - H_offload_L1;

    int H_offload_L_plus = static_cast<int>(round((double)H / x / ALIGNMENT) * ALIGNMENT);
    int H_resident_L_plus = H - H_offload_L_plus;

    int H_transfer2_rows = static_cast<int>(round((double)H / (x * (x + 1.0))));
    size_t transfer2_size_bytes = (size_t)H_transfer2_rows * H * sizeof(float);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Layer 1 Offload Rows:         " << H_offload_L1 << " (" << 100.0 * H_offload_L1 / H << "% of a layer)\n";
    std::cout << "Subsequent Offload Rows:      " << H_offload_L_plus << " (" << 100.0 * H_offload_L_plus / H << "% of a layer)\n";
    std::cout << "Pre-fetch (Transfer2) Rows:   " << H_transfer2_rows << " (" << 100.0 * H_transfer2_rows / H << "% of a layer)\n";

    // --- 2. Memory Allocation ---
    cudaMemcpyKind copyKind = use_nvlink ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    int peerDeviceId = -1;

    // GPU allocations
    std::vector<float*> d_weights(num_layers);
    for (int i = 0; i < num_layers; ++i) CHECK_CUDA(cudaMalloc(&d_weights[i], (size_t)H * H * sizeof(float)));
    float *d_activations_A, *d_activations_B;
    CHECK_CUDA(cudaMalloc(&d_activations_A, (size_t)H * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_activations_B, (size_t)H * N * sizeof(float)));
    
    // Offload buffer allocations
    std::vector<void*> offload_buffers_src(num_layers);
    std::vector<size_t> offload_sizes_bytes(num_layers);
    offload_sizes_bytes[0] = (size_t)H_offload_L1 * H * sizeof(float);
    for (int i = 1; i < num_layers; ++i) offload_sizes_bytes[i] = (size_t)H_offload_L_plus * H * sizeof(float);

    if (use_nvlink) {
        int device_count;
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        if (device_count < 2) { std::cerr << "Error: --nvlink requires at least 2 GPUs." << std::endl; exit(EXIT_FAILURE); }
        peerDeviceId = (device_id + 1) % device_count;
        int canAccessPeer;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, device_id, peerDeviceId));
        if (!canAccessPeer) { std::cerr << "Error: Peer access is not supported." << std::endl; exit(EXIT_FAILURE); }
        CHECK_CUDA(cudaDeviceEnablePeerAccess(peerDeviceId, 0));
        
        CHECK_CUDA(cudaSetDevice(peerDeviceId));
        for (int i = 0; i < num_layers; ++i) {
            if (offload_sizes_bytes[i] > 0) CHECK_CUDA(cudaMalloc(&offload_buffers_src[i], offload_sizes_bytes[i]));
        }
        CHECK_CUDA(cudaSetDevice(device_id));
    } else {
        for (int i = 0; i < num_layers; ++i) {
            if (offload_sizes_bytes[i] > 0) CHECK_CUDA(cudaHostAlloc(&offload_buffers_src[i], offload_sizes_bytes[i], cudaHostAllocDefault));
        }
    }
    
    // --- 3. Pre-calculate Pointers for Graph Capture ---
    std::vector<float*> p_input_activations(num_layers);
    std::vector<float*> p_output_activations(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        p_input_activations[i] = (i % 2 == 0) ? d_activations_A : d_activations_B;
        p_output_activations[i] = (i % 2 == 0) ? d_activations_B : d_activations_A;
    }

    // --- 4. CUDA Graph Construction with Per-Layer Timers ---
    std::cout << "Building MLP CUDA Graph..." << std::flush;
    cudaStream_t compute_stream, copy_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream)); CHECK_CUDA(cudaStreamCreate(&copy_stream));
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle)); CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));
    const float alpha = 1.0f, beta = 0.0f;
    
    // ✅ Allocate space for 5 categories of timers per layer (C1, C2, T1, T2, and LayerTotal)
    const int TIMESTAMPS_PER_LAYER = 10;
    unsigned long long* d_timestamps;
    CHECK_CUDA(cudaMalloc(&d_timestamps, num_layers * TIMESTAMPS_PER_LAYER * sizeof(unsigned long long)));

    cudaGraph_t graph; cudaGraphExec_t graph_exec;
    std::vector<cudaEvent_t> transfer1_events(num_layers), transfer2_events(num_layers - 1);
    cudaEvent_t ghost_event; CHECK_CUDA(cudaEventCreate(&ghost_event));
    for (auto& e : transfer1_events) CHECK_CUDA(cudaEventCreate(&e));
    for (auto& e : transfer2_events) CHECK_CUDA(cudaEventCreate(&e));
    
    CHECK_CUDA(cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeThreadLocal));
    CHECK_CUDA(cudaEventRecord(ghost_event, compute_stream));
    CHECK_CUDA(cudaStreamWaitEvent(copy_stream, ghost_event, 0));

    for (int i = 0; i < num_layers; ++i) {
        int H_resident = (i == 0) ? H_resident_L1 : H_resident_L_plus;
        int H_offload = (i == 0) ? H_offload_L1 : H_offload_L_plus;
        size_t transfer1_size_bytes = (size_t)H_offload * H * sizeof(float);
        float* weight_dst_T1 = d_weights[i] + (size_t)H_resident * H; // Column-major pointer arithmetic is complex, but this is for W', so it's row-wise
        float* output_dst_C2 = p_output_activations[i] + H_resident; // Column-major output; C(row, col) = base + row + col*ld
        
        // ✅ Record the true start of the layer's work
        recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 8); // Layer i Start

        // Phase 1
        recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 0); // Compute1 Start
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, H_resident, H, &alpha, p_input_activations[i], N, d_weights[i], H, &beta, p_output_activations[i], N));
        recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 1); // Compute1 Stop

        recordTimestamp<<<1, 1, 0, copy_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 4); // Transfer1 Start
        if (transfer1_size_bytes > 0) CHECK_CUDA(cudaMemcpyAsync(d_weights[i] + (size_t)H_resident, offload_buffers_src[i], transfer1_size_bytes, copyKind, copy_stream));
        recordTimestamp<<<1, 1, 0, copy_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 5); // Transfer1 Stop
        
        CHECK_CUDA(cudaEventRecord(transfer1_events[i], copy_stream));
        CHECK_CUDA(cudaStreamWaitEvent(compute_stream, transfer1_events[i], 0));

        // Phase 2
        recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 2); // Compute2 Start
        if (H_offload > 0) CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, H_offload, H, &alpha, p_input_activations[i], N, d_weights[i] + H_resident, H, &beta, output_dst_C2, N));
        reluKernel<<<((size_t)H * N + 255) / 256, 256, 0, compute_stream>>>(p_output_activations[i], (size_t)H * N);
        recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 3); // Compute2 Stop

        if (i < num_layers - 1) {
            recordTimestamp<<<1, 1, 0, copy_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 6); // Transfer2 Start
            if (transfer2_size_bytes > 0) CHECK_CUDA(cudaMemcpyAsync(d_weights[i+1], offload_buffers_src[i+1], transfer2_size_bytes, copyKind, copy_stream));
            recordTimestamp<<<1, 1, 0, copy_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 7); // Transfer2 Stop
            
            CHECK_CUDA(cudaEventRecord(transfer2_events[i], copy_stream));
            CHECK_CUDA(cudaStreamWaitEvent(compute_stream, transfer2_events[i], 0));
        }
        
        // ✅ Record the true end of the layer's work (after all syncs)
        recordTimestamp<<<1, 1, 0, compute_stream>>>(d_timestamps + i * TIMESTAMPS_PER_LAYER + 9); // Layer i End
    }

    CHECK_CUDA(cudaStreamEndCapture(compute_stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    std::cout << " Done.\n";

    // --- 5. Execution and Timing ---
    const int WARMUP_COUNT = 5;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs..." << std::flush;
    for(int i = 0; i < WARMUP_COUNT; ++i) CHECK_CUDA(cudaGraphLaunch(graph_exec, compute_stream));
    CHECK_CUDA(cudaStreamSynchronize(compute_stream));
    std::cout << " Done.\n";

    unsigned long long* h_timestamps = new unsigned long long[num_layers * TIMESTAMPS_PER_LAYER];
    std::vector<double> all_compute1, all_compute2, all_transfer1, all_transfer2, all_layer_times, all_pass_times;

    std::cout << "Running " << trials << " timed trials..." << std::flush;
    for(int t = 0; t < trials; ++t) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, compute_stream));
        CHECK_CUDA(cudaStreamSynchronize(compute_stream));
        CHECK_CUDA(cudaMemcpy(h_timestamps, d_timestamps, num_layers * TIMESTAMPS_PER_LAYER * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < num_layers; ++i) {
            all_compute1.push_back((double)(h_timestamps[i*TIMESTAMPS_PER_LAYER+1] - h_timestamps[i*TIMESTAMPS_PER_LAYER+0]));
            all_compute2.push_back((double)(h_timestamps[i*TIMESTAMPS_PER_LAYER+3] - h_timestamps[i*TIMESTAMPS_PER_LAYER+2]));
            all_transfer1.push_back((double)(h_timestamps[i*TIMESTAMPS_PER_LAYER+5] - h_timestamps[i*TIMESTAMPS_PER_LAYER+4]));
            if (i < num_layers - 1) {
                all_transfer2.push_back((double)(h_timestamps[i*TIMESTAMPS_PER_LAYER+7] - h_timestamps[i*TIMESTAMPS_PER_LAYER+6]));
            }
            // ✅ Collect the true, overlapped layer time
            all_layer_times.push_back((double)(h_timestamps[i*TIMESTAMPS_PER_LAYER+9] - h_timestamps[i*TIMESTAMPS_PER_LAYER+8]));
        }
        all_pass_times.push_back((double)(h_timestamps[num_layers * TIMESTAMPS_PER_LAYER - 1] - h_timestamps[8]));
    }
    std::cout << " Done.\n";

    // --- 6. Reporting with Corrected Metrics ---
    auto avg_ns_to_ms = [](const std::vector<double>& v) { if (v.empty()) return 0.0; return (std::accumulate(v.begin(), v.end(), 0.0) / v.size()) / 1.0e6; };
    
    double avg_c1_ms = avg_ns_to_ms(all_compute1);
    double avg_c2_ms = avg_ns_to_ms(all_compute2);
    double avg_t1_ms = avg_ns_to_ms(all_transfer1);
    double avg_t2_ms = avg_ns_to_ms(all_transfer2);
    
    // ✅ Calculate new aggregate metrics based on direct measurements
    double avg_per_layer_ms = avg_ns_to_ms(all_layer_times);
    double avg_forward_pass_ms = avg_ns_to_ms(all_pass_times);
    
    double total_bytes_transferred = trials * (offload_sizes_bytes[0] + (num_layers - 1) * offload_sizes_bytes[1] + (num_layers - 1) * transfer2_size_bytes);
    double total_transfer_time_s = (std::accumulate(all_transfer1.begin(), all_transfer1.end(), 0.0) + std::accumulate(all_transfer2.begin(), all_transfer2.end(), 0.0)) / 1.0e9;
    double observed_transfer_bw = total_transfer_time_s > 0 ? (total_bytes_transferred / total_transfer_time_s / 1.0e9) : 0.0;
    
    double total_weights_size_gb = (double)num_layers * H * H * sizeof(float) / 1.0e9;
    double observed_gpu_throughput = avg_forward_pass_ms > 0 ? (total_weights_size_gb / (avg_forward_pass_ms / 1000.0)) : 0.0;

    std::cout << "\n--- Average Timings per Layer (over " << trials * num_layers << " samples) ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Compute1 (Resident):      " << std::setw(8) << avg_c1_ms << " ms\n";
    std::cout << "Compute2 (Offloaded):     " << std::setw(8) << avg_c2_ms << " ms\n";
    std::cout << "Transfer1 (Main):         " << std::setw(8) << avg_t1_ms << " ms\n";
    std::cout << "Transfer2 (Pre-fetch):    " << std::setw(8) << avg_t2_ms << " ms\n";
    
    std::cout << "\n--- Overall Performance Metrics ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Observed Transfer BW:     " << std::setw(8) << observed_transfer_bw << " GB/s\n";
    std::cout << "Observed GPU Throughput:  " << std::setw(8) << observed_gpu_throughput << " GB/s\n";
    std::cout << "Avg. Per-Layer Time:      " << std::setw(8) << avg_per_layer_ms << " ms (Wall Clock)\n";
    std::cout << "Avg. Forward Pass Time:   " << std::setw(8) << avg_forward_pass_ms << " ms (Wall Clock)\n";


    // --- 7. Cleanup ---
    delete[] h_timestamps;
    for (auto& p : d_weights) CHECK_CUDA(cudaFree(p));
    CHECK_CUDA(cudaFree(d_activations_A)); CHECK_CUDA(cudaFree(d_activations_B));
    CHECK_CUDA(cudaFree(d_timestamps));
    if (use_nvlink && peerDeviceId != -1) CHECK_CUDA(cudaDeviceDisablePeerAccess(peerDeviceId));
    for (auto& buf : offload_buffers_src) { if (buf) { if (use_nvlink) { CHECK_CUDA(cudaSetDevice(peerDeviceId)); CHECK_CUDA(cudaFree(buf)); CHECK_CUDA(cudaSetDevice(device_id)); } else { CHECK_CUDA(cudaFreeHost(buf)); } } }
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec)); CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(ghost_event));
    for (auto& e : transfer1_events) CHECK_CUDA(cudaEventDestroy(e));
    for (auto& e : transfer2_events) CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaStreamDestroy(compute_stream)); CHECK_CUDA(cudaStreamDestroy(copy_stream));
}

void printUsage(const char* prog_name) {
    std::cerr << "\nUsage: " << prog_name << " [options]\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  -h, --help                 Show this help message and exit.\n";
    std::cerr << "  -H, --hidden_dim <int>     Hidden dimension of MLP layers. (Default: 12288)\n";
    std::cerr << "  -B, --batch_size <int>     Batch size for the input. (Default: 8)\n";
    std::cerr << "  -N, --num_layers <int>     Number of MLP layers. (Default: 10)\n";
    std::cerr << "  -x, --comm_ratio <float>   Target ratio of HBM to PCIe/NVLink bandwidth. (Default: 10.0)\n";
    std::cerr << "  -t, --trials <int>         Number of timed trials to run. (Default: 1000)\n";
    std::cerr << "  --nvlink                   Use NVLink for D2D transfer instead of PCIe H2D.\n\n";
}

bool parseArgs(int argc, char** argv, int& H, int& N, int& num_layers, double& x, bool& use_nvlink, int& trials) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return false;
        }
        if ((arg == "-H" || arg == "--hidden_dim") && i + 1 < argc) H = std::stoi(argv[++i]);
        else if ((arg == "-B" || arg == "--batch_size") && i + 1 < argc) N = std::stoi(argv[++i]);
        else if ((arg == "-N" || arg == "--num_layers") && i + 1 < argc) num_layers = std::stoi(argv[++i]);
        else if ((arg == "-x" || arg == "--comm_ratio") && i + 1 < argc) x = std::stod(argv[++i]);
        else if ((arg == "-t" || arg == "--trials") && i + 1 < argc) trials = std::stoi(argv[++i]);
        else if (arg == "--nvlink") use_nvlink = true;
        else {
            std::cerr << "Error: Unknown or invalid argument: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

// Then, replace the existing main() function with this simpler version:
int main(int argc, char** argv) {
    int H = 12288, N = 8, num_layers = 10, trials = 1000;
    double x = 10.0;
    bool use_nvlink = false;

    if (!parseArgs(argc, argv, H, N, num_layers, x, use_nvlink, trials)) {
        return 1;
    }
    
    runMlpTest(H, N, num_layers, x, use_nvlink, trials);
    
    return 0;
}