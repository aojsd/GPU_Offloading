"""Test PCIe bandwidth with concurrent H2D transfers to multiple GPUs.

Measures single-GPU and multi-GPU concurrent transfer bandwidth to verify
there's no contention when running experiments on different GPUs in parallel.
"""
import torch
import torch.cuda
import threading
import time


def measure_h2d_bandwidth(gpu_id: int, size_mb: int = 336, n_transfers: int = 10):
    """Measure H2D bandwidth for a single GPU. Returns GB/s."""
    torch.cuda.set_device(gpu_id)
    src = torch.randn(size_mb * 1024 * 1024 // 4, dtype=torch.float32, pin_memory=True)
    dst = torch.empty_like(src, device=f"cuda:{gpu_id}")
    stream = torch.cuda.Stream(device=gpu_id)

    # Warmup
    with torch.cuda.stream(stream):
        for _ in range(3):
            dst.copy_(src, non_blocking=True)
    stream.synchronize()

    # Timed
    start = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(n_transfers):
            dst.copy_(src, non_blocking=True)
    stream.synchronize()
    elapsed = time.perf_counter() - start

    total_gb = size_mb * n_transfers / 1024
    bw = total_gb / elapsed
    return bw, elapsed


def run_concurrent(gpu_ids: list[int], size_mb: int = 336, n_transfers: int = 10):
    """Run H2D transfers concurrently on multiple GPUs. Returns {gpu_id: (bw, elapsed)}."""
    results = {}
    barrier = threading.Barrier(len(gpu_ids))

    def worker(gpu_id):
        torch.cuda.set_device(gpu_id)
        src = torch.randn(size_mb * 1024 * 1024 // 4, dtype=torch.float32, pin_memory=True)
        dst = torch.empty_like(src, device=f"cuda:{gpu_id}")
        stream = torch.cuda.Stream(device=gpu_id)

        # Warmup
        with torch.cuda.stream(stream):
            for _ in range(3):
                dst.copy_(src, non_blocking=True)
        stream.synchronize()

        # Sync all threads before timed region
        barrier.wait()

        start = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(n_transfers):
                dst.copy_(src, non_blocking=True)
        stream.synchronize()
        elapsed = time.perf_counter() - start

        total_gb = size_mb * n_transfers / 1024
        results[gpu_id] = (total_gb / elapsed, elapsed)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpu_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main():
    n_gpus = torch.cuda.device_count()
    gpu_ids = list(range(n_gpus))
    print(f"GPUs: {n_gpus} x {torch.cuda.get_device_name(0)}")

    # Expert-sized transfer: 336 MB (Mixtral expert = 3 * 8*14336*4096 * 2B = 336 MB)
    size_mb = 336
    n_transfers = 10
    print(f"Transfer size: {size_mb} MB, {n_transfers} transfers per GPU\n")

    # Sequential: each GPU alone
    print("=== Sequential (one GPU at a time) ===")
    solo_bw = {}
    for g in gpu_ids:
        bw, elapsed = measure_h2d_bandwidth(g, size_mb, n_transfers)
        solo_bw[g] = bw
        print(f"  GPU {g}: {bw:.2f} GB/s  ({elapsed:.3f}s)")

    # Concurrent: all GPUs simultaneously
    print(f"\n=== Concurrent ({n_gpus} GPUs simultaneously) ===")
    conc = run_concurrent(gpu_ids, size_mb, n_transfers)
    for g in gpu_ids:
        bw, elapsed = conc[g]
        ratio = bw / solo_bw[g]
        print(f"  GPU {g}: {bw:.2f} GB/s  ({elapsed:.3f}s)  ratio vs solo: {ratio:.2f}x")

    avg_solo = sum(solo_bw.values()) / len(solo_bw)
    avg_conc = sum(bw for bw, _ in conc.values()) / len(conc)
    print(f"\n  Avg solo: {avg_solo:.2f} GB/s")
    print(f"  Avg concurrent: {avg_conc:.2f} GB/s")
    print(f"  Contention factor: {avg_conc / avg_solo:.2f}x (1.00 = no contention)")


if __name__ == "__main__":
    main()
