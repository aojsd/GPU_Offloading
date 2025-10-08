/**
 * @file gemm_row_split.cu
 * @brief Tests the performance of a matrix multiplication A * B (N x H * H x S)
 * by splitting matrix A row-wise and executing two sequential cuBLAS GEMM calls
 * within a single CUDA Graph.
 *
 * This test measures the effective throughput at which the GPU consumes the data
 * from matrix A, providing a baseline for scenarios where A might be partitioned
 * for streaming or offloading.
 */
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include <cuda_runtime.h>
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

// GPU kernel to compare two matrices and find the maximum absolute error
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

// Host function to verify the result using a single large cuBLAS call
void verify_result_gpu(const float* d_A, const float* d_B, const float* d_C_test, int N, int H, int S) {
    std::cout << "\nVerifying result on GPU using cuBLAS... " << std::flush;
    
    size_t C_size = (size_t)N * S * sizeof(float);
    float* d_C_verify;
    CHECK_CUDA(cudaMalloc(&d_C_verify, C_size));

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
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
    compareAndFindMaxErrorKernel<<<compare_blocks, compare_threads>>>(d_C_test, d_C_verify, d_max_error, (size_t)N * S);
    
    CHECK_CUDA(cudaMemcpy(&h_max_error, d_max_error, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Done.\nMaximum absolute error: " << h_max_error << std::endl;

    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaFree(d_C_verify));
    CHECK_CUDA(cudaFree(d_max_error));
}

/**
 * @brief Measures the performance of a row-split matrix multiplication using a
 * reusable CUDA Graph and precise GPU-side timers.
 */
void runRowSplitTest(int N, int H, int S, float split_ratio, int trials) {
    std::cout << "\n--- Running Row-Split GEMM Test (CUDA Graph Version) ---\n";

    // --- 1. Data Partitioning ---
    const int ALIGNMENT = 32;
    if (N % ALIGNMENT != 0) {
        int old_N = N;
        N = (N / ALIGNMENT) * ALIGNMENT;
        std::cout << "Adjusting N from " << old_N << " to " << N << " to be a multiple of " << ALIGNMENT << "\n";
    }
    // N2 is the smaller partition, based on the ratio. N1 is the rest.
    int N2 = (static_cast<int>(round(N * split_ratio)) / ALIGNMENT) * ALIGNMENT;
    int N1 = N - N2;

    if (N1 < 0 || (N1 == 0 && N2 == 0)) {
        std::cerr << "Error: Split ratio " << split_ratio << " results in an invalid partition. Aborting." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Partitioning A[" << N << "x" << H << "] into A1[" << N1 << "x" << H << "] (large) and A2[" << N2 << "x" << H << "] (small)\n";

    // --- 2. Memory Allocation and Initialization ---
    size_t A_size = (size_t)N * H * sizeof(float);
    size_t B_size = (size_t)H * S * sizeof(float);
    size_t C_size = (size_t)N * S * sizeof(float);

    std::vector<float> h_A(A_size / sizeof(float));
    std::vector<float> h_B(B_size / sizeof(float));
    for(size_t i = 0; i < h_A.size(); ++i) h_A[i] = (float)rand() / RAND_MAX;
    for(size_t i = 0; i < h_B.size(); ++i) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, A_size));
    CHECK_CUDA(cudaMalloc(&d_B, B_size));
    CHECK_CUDA(cudaMalloc(&d_C, C_size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), A_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), B_size, cudaMemcpyHostToDevice));

    // --- 3. Graph Construction ---
    std::cout << "Building CUDA Graph... " << std::flush;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));

    const int NUM_TIMESTAMPS = 6;
    unsigned long long* d_timestamps;
    CHECK_CUDA(cudaMalloc(&d_timestamps, NUM_TIMESTAMPS * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_timestamps, 0, NUM_TIMESTAMPS * sizeof(unsigned long long)));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    
    recordTimestamp<<<1, 1, 0, stream>>>(d_timestamps + 0); // Total Start

    // First GEMM on A1
    recordTimestamp<<<1, 1, 0, stream>>>(d_timestamps + 1); // GEMM1 Start
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             S, N1, H, &alpha,
                             d_B, S,
                             d_A, H, &beta,
                             d_C, S));
    recordTimestamp<<<1, 1, 0, stream>>>(d_timestamps + 2); // GEMM1 Stop

    // Second GEMM on A2, only if N2 is non-zero
    if (N2 > 0) {
        recordTimestamp<<<1, 1, 0, stream>>>(d_timestamps + 3); // GEMM2 Start
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 S, N2, H, &alpha,
                                 d_B, S,
                                 d_A + (size_t)N1 * H, H, &beta,
                                 d_C + (size_t)N1 * S, S));
        recordTimestamp<<<1, 1, 0, stream>>>(d_timestamps + 4); // GEMM2 Stop
    }

    recordTimestamp<<<1, 1, 0, stream>>>(d_timestamps + 5); // Total Stop
    
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    std::cout << "Done.\n";

    // --- 4. Execution and Timing ---
    const int WARMUP_COUNT = 1;
    std::cout << "Performing " << WARMUP_COUNT << " warm-up runs... " << std::flush;
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "Done.\n";

    std::vector<double> total_times_ns, gemm1_times_ns, gemm2_times_ns;
    unsigned long long h_timestamps[NUM_TIMESTAMPS];

    std::cout << "Running " << trials << " timed trials... " << std::flush;
    for (int i = 0; i < trials; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpy(h_timestamps, d_timestamps, sizeof(h_timestamps), cudaMemcpyDeviceToHost));
        
        total_times_ns.push_back((double)(h_timestamps[5] - h_timestamps[0]));
        gemm1_times_ns.push_back((double)(h_timestamps[2] - h_timestamps[1]));
        if (N2 > 0) {
            gemm2_times_ns.push_back((double)(h_timestamps[4] - h_timestamps[3]));
        } else {
            gemm2_times_ns.push_back(0.0);
        }
    }
    std::cout << "Done.\n";

    // --- 5. Reporting ---
    auto avg_ns_to_ms = [](const std::vector<double>& v) {
        return (std::accumulate(v.begin(), v.end(), 0.0) / v.size()) / 1.0e6;
    };
    double avg_total_ms = avg_ns_to_ms(total_times_ns);
    double avg_gemm1_ms = avg_ns_to_ms(gemm1_times_ns);
    double avg_gemm2_ms = avg_ns_to_ms(gemm2_times_ns);

    double effective_throughput_gbs = (A_size) / (avg_total_ms / 1000.0) / 1e9;

    std::cout << "\n--- Timings (avg over " << trials << " graph trials) ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GEMM 1 (N1=" << N1 << ", large):  " << std::setw(8) << avg_gemm1_ms << " ms";
        std::cout << "\t(Throughput: " << std::setw(8) << (N1 * H * sizeof(float) / (avg_gemm1_ms / 1000.0)) / 1e9 << " GB/s)\n";
    std::cout << "GEMM 2 (N2=" << N2 << ", small):  " << std::setw(8) << avg_gemm2_ms << " ms";
        std::cout << "\t(Throughput: " << std::setw(8) << (N2 * H * sizeof(float) / (avg_gemm2_ms / 1000.0)) / 1e9 << " GB/s)\n";
    std::cout << "Total Operation Time:     " << std::setw(8) << avg_total_ms << " ms\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Effective GPU Throughput: " << std::setw(8) << effective_throughput_gbs << " GB/s\n";

    // --- 6. Verification and Cleanup ---
    verify_result_gpu(d_A, d_B, d_C, N, H, S);
    
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaFree(d_timestamps));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void print_usage(const char* prog_name) {
    std::cerr << "\nUsage: " << prog_name << " [options]\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  -h, --help                 Show this help message and exit.\n";
    std::cerr << "  -N, --rows <int>           Number of rows for matrix A. (Default: 1000000)\n";
    std::cerr << "  -H, --hidden_dim <int>     Number of columns for A / rows for B. (Default: 1024)\n";
    std::cerr << "  -S, --cols <int>           Number of columns for matrix B. (Default: 1)\n";
    std::cerr << "  -r, --split_ratio <f>      Fraction of matrix A in the smaller partition (0.0 to 0.5). (Default: 0.5)\n";
    std::cerr << "  -t, --trials <int>         Number of timed trials to run. (Default: 1000)\n";
    std::cerr << "  -d, --device <id>          ID of the GPU device to use. (Default: 0)\n\n";
}

bool parse_args(int argc, char** argv, int& N, int& H, int& S, float& ratio, int& trials, int& device_id) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if ((arg == "-N" || arg == "--rows") && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if ((arg == "-H" || arg == "--hidden_dim") && i + 1 < argc) {
            H = std::stoi(argv[++i]);
        } else if ((arg == "-S" || arg == "--cols") && i + 1 < argc) {
            S = std::stoi(argv[++i]);
        } else if ((arg == "-r" || arg == "--split_ratio") && i + 1 < argc) {
            ratio = std::stof(argv[++i]);
        } else if ((arg == "-t" || arg == "--trials") && i + 1 < argc) {
            trials = std::stoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else {
            std::cerr << "Error: Unknown or invalid argument: " << arg << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }
    if (N <= 0 || H <= 0 || S <= 0 || trials <= 0) {
        std::cerr << "Error: Matrix dimensions and trial count must be positive." << std::endl; return false;
    }
    if (ratio < 0.0f || ratio > 0.5f) {
        std::cerr << "Error: Split ratio must be in the range [0.0, 0.5]." << std::endl; return false;
    }
    return true;
}

int main(int argc, char** argv) {
    // Default values
    int N = 12288, H = 12288, S = 8;
    float split_ratio = 0.25f;
    int trials = 1000;
    int device_id = 0;

    if (!parse_args(argc, argv, N, H, S, split_ratio, trials, device_id)) {
        return 1;
    }

    CHECK_CUDA(cudaSetDevice(device_id));

    std::cout << "Configuration:\n";
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "  Device Name:   " << prop.name << "\n";
    std::cout << "  Matrix A:      " << N << " x " << H << "\n";
    std::cout << "  Matrix B:      " << H << " x " << S << "\n";
    std::cout << "  Split Ratio:   " << split_ratio << "\n";
    std::cout << "  Trials:        " << trials << "\n";

    runRowSplitTest(N, H, S, split_ratio, trials);

    return 0;
}