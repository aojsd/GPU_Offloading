import torch
import time
import argparse

def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "="*80)
    print(f"| {title:^76} |")
    print("="*80)


def profile_pcie_bandwidth(size_kb_values, num_repeats=50, cuda_device=0):
    """
    Profiles PCIe bandwidth for CPU-GPU data transfers.

    Args:
        size_kb_values (list): List of tensor sizes in kilobytes.
        num_repeats (int): Number of repeats for stable timing.
    """
    print_header("PCIe Bandwidth & Latency Profiling")
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping PCIe tests.")
        return

    device = torch.device(f"cuda:{cuda_device}")
    print("--- PCIe Bandwidth ---")
    print(f"{'Tensor Size (MB)':<20} | {'CPU->GPU (GB/s)':<20} | {'GPU->CPU (GB/s)':<20}")
    print("-" * 80)

    for size_kb in size_kb_values:
        # Calculate the number of elements for a float32 tensor
        num_elements = int((size_kb * 1024) / 4)
        cpu_tensor = torch.randn(num_elements, dtype=torch.float32)
        gpu_tensor = torch.randn(num_elements, dtype=torch.float32).to(device)

        # --- Profile CPU to GPU transfer ---
        # Warm-up
        _ = cpu_tensor.to(device)
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(num_repeats):
            cpu_tensor.to(device)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time_h2d = end_time - start_time
        # Bandwidth = Total Bytes / Total Time
        bandwidth_h2d = (num_repeats * size_kb / 1024 / 1024) / total_time_h2d # GB/s

        # --- Profile GPU to CPU transfer ---
        # Warm-up
        _ = gpu_tensor.to("cpu")
        torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(num_repeats):
            gpu_tensor.to("cpu")
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_time_d2h = end_time - start_time
        bandwidth_d2h = (num_repeats * size_kb / 1024 / 1024) / total_time_d2h # GB/s
        size_mb = size_kb / 1024
        print(f"{size_mb:<20.2f} | {bandwidth_h2d:<20.2f} | {bandwidth_d2h:<20.2f}")

    # --- Profile Latency ---
    print("\n--- PCIe Latency (Approximate) ---")
    # Use a tiny tensor to minimize data transfer time, isolating the call overhead.
    tiny_tensor_cpu = torch.zeros(1, dtype=torch.float32)
    latency_repeats = 1000
    
    # Warm-up
    _ = tiny_tensor_cpu.to(device)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(latency_repeats):
        tiny_tensor_cpu.to(device)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    h2d_latency_ms = ((end_time - start_time) / latency_repeats) * 1000
    
    print(f"Host to Device (CPU->GPU) latency: {h2d_latency_ms:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch System Profiler for CPU, GPU, and PCIe.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Show detailed analysis text for CPU and GPU bound sections.")
    parser.add_argument('--cuda', type=int, default=0,
                        help="The CUDA device ID to use for GPU tests. (default: 0)")
    args = parser.parse_args()

    # Define tensor sizes for PCIe bandwidth test in KB
    pcie_test_sizes_kb = [2**i for i in range(2, 23, 2)] # 4KB to 8GB

    # --- Run PCIe Tests ---
    profile_pcie_bandwidth(pcie_test_sizes_kb, cuda_device=args.cuda)

