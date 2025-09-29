#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate

// Error checking macro
#define CHECK(call)                                                            \
    do {                                                                       \
        const cudaError_t error_code = call;                                   \
        if (error_code != cudaSuccess) {                                       \
            printf("CUDA Error:\n");                                           \
            printf("    File:       %s\n", __FILE__);                           \
            printf("    Line:       %d\n", __LINE__);                           \
            printf("    Error code: %d\n", error_code);                         \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// A compute-intensive kernel for testing timing
__global__ void intensive_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ITERATIONS = 2500;

    if (idx < n) {
        float val = (float)data[idx];
        for (int i = 0; i < ITERATIONS; ++i) {
            val = sinf(val) * 0.5f + cosf(val) * 0.8f;
        }
        data[idx] = (int)val;
    }
}

int main() {
    // 1. Setup
    int n = 4 * 1024 * 1024;
    size_t bytes = n * sizeof(int);

    std::vector<int> h_data(n);
    std::vector<int> h_result(n);
    for (int i = 0; i < n; ++i) {
        h_data[i] = i;
    }

    int *d_data;
    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // NEW: Create event pairs for each kernel and for the total graph
    cudaEvent_t start_A, stop_A, start_B, stop_B, start_C, stop_C;
    cudaEvent_t start_graph, stop_graph;
    CHECK(cudaEventCreate(&start_A)); CHECK(cudaEventCreate(&stop_A));
    CHECK(cudaEventCreate(&start_B)); CHECK(cudaEventCreate(&stop_B));
    CHECK(cudaEventCreate(&start_C)); CHECK(cudaEventCreate(&stop_C));
    CHECK(cudaEventCreate(&start_graph)); CHECK(cudaEventCreate(&stop_graph));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CHECK(cudaGraphCreate(&graph, 0));

    // Define the kernel launch parameters (reused for all 3 kernels)
    cudaKernelNodeParams kernel_params = {0};
    kernel_params.func = (void*)intensive_kernel;
    kernel_params.gridDim = dim3((n + 255) / 256, 1, 1);
    kernel_params.blockDim = dim3(256, 1, 1);
    void *kernel_args[] = {&d_data, &n};
    kernel_params.kernelParams = kernel_args;

    // --- NEW: Build Graph with Event Nodes Around Each Kernel ---
    cudaGraphNode_t node_A, node_B, node_C;
    cudaGraphNode_t event_start_A, event_stop_A, event_start_B, event_stop_B, event_start_C, event_stop_C;

    // Chain 1: Kernel A
    CHECK(cudaGraphAddEventRecordNode(&event_start_A, graph, nullptr, 0, start_A));
    CHECK(cudaGraphAddKernelNode(&node_A, graph, &event_start_A, 1, &kernel_params));
    CHECK(cudaGraphAddEventRecordNode(&event_stop_A, graph, &node_A, 1, stop_A));

    // Chain 2: Kernel B (depends on Kernel A finishing)
    CHECK(cudaGraphAddEventRecordNode(&event_start_B, graph, &event_stop_A, 1, start_B));
    CHECK(cudaGraphAddKernelNode(&node_B, graph, &event_start_B, 1, &kernel_params));
    CHECK(cudaGraphAddEventRecordNode(&event_stop_B, graph, &node_B, 1, stop_B));

    // Chain 3: Kernel C (depends on Kernel B finishing)
    CHECK(cudaGraphAddEventRecordNode(&event_start_C, graph, &event_stop_B, 1, start_C));
    CHECK(cudaGraphAddKernelNode(&node_C, graph, &event_start_C, 1, &kernel_params));
    CHECK(cudaGraphAddEventRecordNode(&event_stop_C, graph, &node_C, 1, stop_C));

    // --- Instantiate, Launch, and Time ---
    CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    // NEW: Record events on the stream around the graph launch
    CHECK(cudaEventRecord(start_graph, stream));
    CHECK(cudaGraphLaunch(graph_exec, stream));
    CHECK(cudaEventRecord(stop_graph, stream));

    CHECK(cudaStreamSynchronize(stream));

    // --- NEW: Calculate and Report All Timings ---
    float time_A = 0, time_B = 0, time_C = 0, time_graph = 0;
    CHECK(cudaEventElapsedTime(&time_A, start_A, stop_A));
    CHECK(cudaEventElapsedTime(&time_B, start_B, stop_B));
    CHECK(cudaEventElapsedTime(&time_C, start_C, stop_C));
    CHECK(cudaEventElapsedTime(&time_graph, start_graph, stop_graph));

    CHECK(cudaMemcpy(h_result.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    std::cout.precision(4);
    std::cout << std::fixed;
    std::cout << "--- Timing Results ---" << std::endl;
    std::cout << "Kernel A Execution Time: \t" << time_A << " ms" << std::endl;
    std::cout << "Kernel B Execution Time: \t" << time_B << " ms" << std::endl;
    std::cout << "Kernel C Execution Time: \t" << time_C << " ms" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    float sum_kernels = time_A + time_B + time_C;
    std::cout << "Sum of Kernel Times: \t\t" << sum_kernels << " ms" << std::endl;
    std::cout << "Total Graph Execution Time: \t" << time_graph << " ms" << std::endl;

    // --- Cleanup ---
    CHECK(cudaEventDestroy(start_A)); CHECK(cudaEventDestroy(stop_A));
    CHECK(cudaEventDestroy(start_B)); CHECK(cudaEventDestroy(stop_B));
    CHECK(cudaEventDestroy(start_C)); CHECK(cudaEventDestroy(stop_C));
    CHECK(cudaEventDestroy(start_graph)); CHECK(cudaEventDestroy(stop_graph));
    CHECK(cudaGraphExecDestroy(graph_exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_data));

    return 0;
}