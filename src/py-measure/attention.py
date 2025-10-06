import torch
import time
import argparse

def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "="*85)
    print(f"| {title:^81} |")
    print("="*85)

def profile_cpu_attention_matmul(N_values, S_values, h, num_repeats=100):
    """
    Profiles CPU throughput for an attention-style matrix multiplication.

    Args:
        N_values (list): A list of matrix dimensions (N) to test.
        S_values (list): A list of matrix dimensions (S) to test.
        h (int): The shared hidden dimension.
        num_repeats (int): The number of times to repeat the operation for stable timing.
    """
    print_header(f"CPU Throughput Profiling (MatMul, H={h})")
    print(f"{'Matrix (NxH x HxS)':<25} | {'Data Size (MB)':<18} | {'Time per op (ms)':<20} | {'GFLOPS':<15}")
    print("-" * 85)

    for S in S_values:
        for N in N_values:
            data_size_mb = 0
            try:
                mat1 = torch.randn(N, h, dtype=torch.float32)
                mat2 = torch.randn(h, S, dtype=torch.float32)

                # Calculate data size
                total_elements = mat1.numel() + mat2.numel()
                data_size_mb = (total_elements * 4) / (1024 * 1024)

                # Warm-up run
                _ = torch.mm(mat1, mat2)

                start_time = time.perf_counter()
                for _ in range(num_repeats):
                    torch.mm(mat1, mat2)
                end_time = time.perf_counter()

                total_time = end_time - start_time
                time_per_op = (total_time / num_repeats) * 1000  # ms

                # FLOPs for (N, H) x (H, S) is approx. 2 * N * H * S
                flops_per_op = 2 * N * h * S
                gflops = (flops_per_op / (time_per_op / 1000)) / 1e9

                dims_str = f"{N}x{h} x {h}x{S}"
                print(f"{dims_str:<25} | {data_size_mb:<18.2f} | {time_per_op:<20.4f} | {gflops:<15.2f}")

            except RuntimeError as e:
                dims_str = f"{N}x{h} x {h}x{S}"
                if data_size_mb == 0:
                    total_elements = (N * h) + (h * S)
                    data_size_mb = (total_elements * 4) / (1024 * 1024)
                print(f"{dims_str:<25} | {data_size_mb:<18.2f} | {'OOM Error':<20} | {str(e)}")
                break # Stop testing this S value if we run out of memory
        print()

def profile_gpu_attention_matmul(N_values, S_values, h, cuda_device=0, num_repeats=1000):
    """
    Profiles GPU throughput for an attention-style matrix multiplication using CUDA Graphs.
    """
    print_header(f"2. GPU Throughput Profiling (MatMul, H={h}, CUDA Graph)")
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU tests.")
        return

    if cuda_device >= torch.cuda.device_count():
        print(f"Error: CUDA device {cuda_device} is not available. Found {torch.cuda.device_count()} devices.")
        return

    device = torch.device(f"cuda:{cuda_device}")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"{'Matrix (NxH x HxS)':<25} | {'Data Size (MB)':<18} | {'Time per op (ms)':<20} | {'GFLOPS':<15}")
    print("-" * 85)

    torch.cuda.set_device(device)
    s = torch.cuda.Stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # --- Pre-allocate memory buffers outside the main loops ---
    max_N = max(N_values)
    max_S = max(S_values)
    
    # Allocate the largest possible matrices once and fill them with random data.
    mat1_buffer = torch.randn(max_N, h, dtype=torch.float32, device=device)
    mat2_buffer = torch.randn(h, max_S, dtype=torch.float32, device=device)

    for S in S_values:
        for N in N_values:
            try:
                mat1 = mat1_buffer[:N, :].clone().to(device)
                mat2 = mat2_buffer[:, :S].clone().to(device)

                # Warm-up is still good practice
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _ = torch.mm(mat1, mat2)
                torch.cuda.synchronize()

                # --- CAPTURE THE GRAPH ---
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    # The 'for' loop is unrolled during capture, creating a single graph
                    # that contains 'num_repeats' matmul operations.
                    for _ in range(num_repeats):
                        torch.mm(mat1, mat2)
                
                # --- 3. FIX: REPLAY & TIMING PHASE ---
                # Now, time the execution of the entire graph replay.
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # Record events around the g.replay() call.
                start_event.record()
                g.replay()
                end_event.record()
                torch.cuda.synchronize()  # Wait for the events to be recorded
                
                total_time_ms = start_event.elapsed_time(end_event)
                time_per_op = total_time_ms / num_repeats

                flops_per_op = 2 * N * h * S
                gflops = (flops_per_op / (time_per_op / 1000)) / 1e9
                
                dims_str = f"{N}x{h} x {h}x{S}"
                data_size_mb = ((N * h) + (h * S)) * 4 / (1024*1024)
                print(f"{dims_str:<25} | {data_size_mb:<18.2f} | {time_per_op:<20.4f} | {gflops:<15.2f}")

            except RuntimeError as e:
                dims_str = f"{N}x{h} x {h}x{S}"
                data_size_mb = ((N * h) + (h * S)) * 4 / (1024*1024)
                print(f"{dims_str:<25} | {data_size_mb:<18.2f} | {'OOM Error':<20} | Check GPU memory.")
                break
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Profiler for Attention-style Matrix Multiplication.")
    parser.add_argument('-H', '--hdim', type=int, required=True,
                        help="The shared dimension 'h' for the (N,h)x(h,S) matrix multiplication. REQUIRED.")
    parser.add_argument('--run', type=str, default='all', choices=['cpu', 'gpu', 'all'],
                        help="Specify which tests to run: 'cpu', 'gpu', or 'all'. (default: all)")
    parser.add_argument('--cuda', type=int, default=0,
                        help="The CUDA device ID to use for GPU tests. (default: 0)")
    args = parser.parse_args()

    num_cores = 24
    torch.set_num_threads(num_cores)
    print(f"Using {num_cores} CPU threads for PyTorch operations.")

    # Define the range of matrix sizes (N) to test.
    n_values = [2**i for i in range(10, 21)] # 1024 up to 1M
    s_values = [2**i for i in range(0, 11)]  # 1 up to 1024

    if args.run in ['cpu', 'all']:
        profile_cpu_attention_matmul(n_values, s_values, h=args.hdim)
    
    if args.run in ['gpu', 'all']:
        profile_gpu_attention_matmul(n_values, s_values, h=args.hdim, cuda_device=args.cuda)