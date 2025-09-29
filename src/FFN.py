import torch
import time
import os
import numpy as np
import argparse

def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "="*80)
    print(f"| {title:^76} |")
    print("="*80)

def profile_cpu_gemm_throughput(N_values, num_repeats=100):
    """
    Profiles CPU throughput for matrix-vector multiplication (GEMM).

    Args:
        N_values (list): A list of matrix dimensions (N) to test.
        num_repeats (int): The number of times to repeat the operation for stable timing.
    """
    print_header("CPU Throughput Profiling (Matrix-Vector)")
    print(f"{'Matrix Size (N x N)':<22} | {'Data Size (MB)':<18} | {'Time per op (ms)':<20} | {'GFLOPS':<15}")
    print("-" * 85)

    for N in N_values:
        data_size_mb = 0
        try:
            mat = torch.randn(N, N, dtype=torch.float32)
            vec = torch.randn(N, 1, dtype=torch.float32)
            
            # Calculate data size
            total_elements = mat.numel() + vec.numel()
            data_size_mb = (total_elements * 4) / (1024 * 1024)

            # Warm-up run to load libraries, JIT compile, etc.
            _ = torch.mm(mat, vec)

            start_time = time.perf_counter()
            for _ in range(num_repeats):
                torch.mm(mat, vec)
            end_time = time.perf_counter()

            total_time = end_time - start_time
            time_per_op = (total_time / num_repeats) * 1000  # Convert to milliseconds

            # For an N x N matrix and N x 1 vector, mv is approx. 2 * N^2 FLOPs
            flops_per_op = 2 * N * N
            gflops = (flops_per_op / (time_per_op / 1000)) / 1e9 # Giga-FLOPs per second

            print(f"{N:<22} | {data_size_mb:<18.2f} | {time_per_op:<20.4f} | {gflops:<15.2f}")
        except RuntimeError as e:
            if data_size_mb == 0:
                total_elements = (N * N) + N
                data_size_mb = (total_elements * 4) / (1024 * 1024)
            print(f"{N:<22} | {data_size_mb:<18.2f} | {'OOM Error':<20} | {str(e)}")
            break # Stop if we run out of memory

def profile_gpu_gemm_throughput(N_values, cuda_device=0, num_repeats=100):
    """
    Profiles GPU throughput for matrix-vector multiplication (GEMM).

    Args:
        N_values (list): A list of matrix dimensions (N) to test.
        cuda_device (int): The CUDA device ID to use for the tests.
        num_repeats (int): The number of times to repeat the operation for stable timing.
    """
    print_header("GPU Throughput Profiling (Matrix-Vector)")
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU tests.")
        return

    if cuda_device >= torch.cuda.device_count():
        print(f"Error: CUDA device {cuda_device} is not available. Found {torch.cuda.device_count()} devices.")
        return

    device = torch.device(f"cuda:{cuda_device}")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"{'Matrix Size (N x N)':<22} | {'Data Size (MB)':<18} | {'Time per op (ms)':<20} | {'GFLOPS':<15}")
    print("-" * 85)

    for N in N_values:
        data_size_mb = 0
        try:
            # Pre-calculate size in case allocation fails
            total_elements = (N * N) + N
            data_size_mb = (total_elements * 4) / (1024 * 1024)

            mat = torch.randn(N, N, dtype=torch.float32).to(device)
            vec = torch.randn(N, 1, dtype=torch.float32).to(device)

            # Warm-up run
            _ = torch.mm(mat, vec)
            torch.cuda.synchronize() # Wait for the warm-up op to finish

            start_time = time.perf_counter()
            for _ in range(num_repeats):
                torch.mm(mat, vec)
            # Crucial: Wait for all GPU operations to complete before stopping the timer
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            time_per_op = (total_time / num_repeats) * 1000  # ms

            flops_per_op = 2 * N * N
            gflops = (flops_per_op / (time_per_op / 1000)) / 1e9

            print(f"{N:<22} | {data_size_mb:<18.2f} | {time_per_op:<20.4f} | {gflops:<15.2f}")
        except RuntimeError as e:
            print(f"{N:<22} | {data_size_mb:<18.2f} | {'OOM Error':<20} | Check GPU memory.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch System Profiler for CPU and GPU.")
    parser.add_argument('--run', type=str, default='all', choices=['cpu', 'gpu', 'all'],
                        help="Specify which tests to run: 'cpu', 'gpu', or 'all'. (default: all)")
    parser.add_argument('--cuda', type=int, default=0,
                        help="The CUDA device ID to use for GPU tests. (default: 0)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Show detailed analysis text for CPU and GPU bound sections.")
    args = parser.parse_args()

    # Set PyTorch to use all available CPU cores
    num_cores = 24
    torch.set_num_threads(num_cores)
    print(f"Using {num_cores} CPU threads for PyTorch operations.")

    # Define the range of matrix sizes (N) to test.
    cpu_n_values = [2**i for i in range(10, 18)] # 1024x1024 up to 131072x131072
    gpu_n_values = [2**i for i in range(10, 17)] # 1024x1024 up to 65536x65536

    if args.run in ['cpu', 'all']:
        profile_cpu_gemm_throughput(cpu_n_values)
    
    if args.run in ['gpu', 'all']:
        profile_gpu_gemm_throughput(gpu_n_values, cuda_device=args.cuda)