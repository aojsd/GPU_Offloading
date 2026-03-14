"""Profile PCIe and NVLink interconnect bandwidth with contention analysis.

Measures bandwidth across three interconnect types and their interactions:

  1. PCIe (Host ↔ GPU):
     - Sequential H2D per GPU
     - Concurrent H2D across all GPUs
     - Bidirectional H2D + D2H per GPU
     - Bidirectional concurrent across all GPUs

  2. NVLink (GPU ↔ GPU):
     - Unidirectional per pair
     - Simultaneous bidirectional per pair

  3. PCIe × NVLink cross-contention (multi-GPU only):
     - For each GPU A with peer GPU B: simultaneous H2D (host→A) + GPU
       copy (A→B), measuring whether PCIe and NVLink share bandwidth.

All pinned CPU tensors are NUMA-local to the target GPU (thread affinity +
first-touch + cudaHostRegister via pin_memory()).

Usage:
    python system_profiling/profile_interconnect.py [--size-mb 336] [--transfers 10]
"""
import argparse
import os
import subprocess
import threading

import torch
import torch.cuda


# ═══════════════════════════════════════════════════════════════════════
#  NUMA helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_gpu_numa_node(gpu_id: int) -> int:
    """Return the NUMA node ID for the given GPU via sysfs."""
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=pci.bus_id",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    bdf = result.stdout.strip().lower()  # e.g. "00000000:3b:00.0"
    # sysfs uses 4-digit domain (0000:xx:yy.z), nvidia-smi uses 8-digit
    if len(bdf.split(":")[0]) == 8:
        bdf = bdf[4:]
    with open(f"/sys/bus/pci/devices/{bdf}/numa_node") as f:
        return int(f.read().strip())


def _parse_cpulist(cpulist: str) -> set[int]:
    cpus: set[int] = set()
    for part in cpulist.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _get_numa_cpus(node: int) -> set[int]:
    """Return set of CPU IDs on the given NUMA node."""
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        return _parse_cpulist(f.read().strip())


def build_gpu_numa_map() -> dict[int, set[int]]:
    """Return {gpu_id: set_of_cpus_on_local_numa_node} for all GPUs."""
    n = torch.cuda.device_count()
    mapping = {}
    for g in range(n):
        node = _get_gpu_numa_node(g)
        mapping[g] = _get_numa_cpus(node)
    return mapping


def alloc_pinned_numa(shape, dtype, numa_cpus: set[int]) -> torch.Tensor:
    """Allocate a pinned CPU tensor with pages faulted on the given NUMA node."""
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, numa_cpus)
    try:
        t = torch.empty(shape, dtype=dtype)
        t.zero_()  # first-touch: fault pages on this NUMA node
        return t.pin_memory()
    finally:
        os.sched_setaffinity(0, old)


# ═══════════════════════════════════════════════════════════════════════
#  PCIe benchmarks (Host ↔ GPU)
# ═══════════════════════════════════════════════════════════════════════

def pcie_h2d_sequential(
    gpu_ids: list[int], numa_map: dict[int, set[int]],
    size_mb: int, n_transfers: int,
) -> dict[int, tuple[float, float]]:
    """Measure H2D bandwidth for each GPU sequentially. Returns {gpu: (GB/s, seconds)}."""
    results = {}
    for gpu_id in gpu_ids:
        torch.cuda.set_device(gpu_id)
        n_elems = size_mb * 1024 * 1024 // 4
        src = alloc_pinned_numa(n_elems, torch.float32, numa_map[gpu_id])
        dst = torch.empty(n_elems, dtype=torch.float32, device=f"cuda:{gpu_id}")
        stream = torch.cuda.Stream(device=gpu_id)

        with torch.cuda.stream(stream):
            for _ in range(3):
                dst.copy_(src, non_blocking=True)
        stream.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_ev.record(stream)
            for _ in range(n_transfers):
                dst.copy_(src, non_blocking=True)
            end_ev.record(stream)
        end_ev.synchronize()

        elapsed = start_ev.elapsed_time(end_ev) / 1000.0
        total_gb = size_mb * n_transfers / 1024
        results[gpu_id] = (total_gb / elapsed, elapsed)
    return results


def pcie_h2d_concurrent(
    gpu_ids: list[int], numa_map: dict[int, set[int]],
    size_mb: int, n_transfers: int,
) -> dict[int, tuple[float, float]]:
    """Run H2D transfers concurrently on all GPUs. Returns {gpu: (GB/s, seconds)}."""
    results = {}
    barrier = threading.Barrier(len(gpu_ids))

    def worker(gpu_id):
        torch.cuda.set_device(gpu_id)
        n_elems = size_mb * 1024 * 1024 // 4
        src = alloc_pinned_numa(n_elems, torch.float32, numa_map[gpu_id])
        dst = torch.empty(n_elems, dtype=torch.float32, device=f"cuda:{gpu_id}")
        stream = torch.cuda.Stream(device=gpu_id)

        with torch.cuda.stream(stream):
            for _ in range(3):
                dst.copy_(src, non_blocking=True)
        stream.synchronize()

        barrier.wait()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_ev.record(stream)
            for _ in range(n_transfers):
                dst.copy_(src, non_blocking=True)
            end_ev.record(stream)
        end_ev.synchronize()

        elapsed = start_ev.elapsed_time(end_ev) / 1000.0
        total_gb = size_mb * n_transfers / 1024
        results[gpu_id] = (total_gb / elapsed, elapsed)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpu_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def pcie_bidirectional(
    gpu_ids: list[int], numa_map: dict[int, set[int]],
    size_mb: int, n_transfers: int,
) -> dict[int, tuple[float, float]]:
    """Simultaneous H2D + D2H per GPU (sequential across GPUs). Returns {gpu: (h2d_bw, d2h_bw)}."""
    results = {}
    for gpu_id in gpu_ids:
        torch.cuda.set_device(gpu_id)
        n_elems = size_mb * 1024 * 1024 // 4
        cpus = numa_map[gpu_id]

        h2d_src = alloc_pinned_numa(n_elems, torch.float32, cpus)
        h2d_dst = torch.empty(n_elems, dtype=torch.float32, device=f"cuda:{gpu_id}")
        d2h_src = torch.randn(n_elems, dtype=torch.float32, device=f"cuda:{gpu_id}")
        d2h_dst = alloc_pinned_numa(n_elems, torch.float32, cpus)

        h2d_stream = torch.cuda.Stream(device=gpu_id)
        d2h_stream = torch.cuda.Stream(device=gpu_id)

        with torch.cuda.stream(h2d_stream):
            for _ in range(3):
                h2d_dst.copy_(h2d_src, non_blocking=True)
        with torch.cuda.stream(d2h_stream):
            for _ in range(3):
                d2h_dst.copy_(d2h_src, non_blocking=True)
        h2d_stream.synchronize()
        d2h_stream.synchronize()

        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        d2h_start = torch.cuda.Event(enable_timing=True)
        d2h_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(h2d_stream):
            h2d_start.record(h2d_stream)
            for _ in range(n_transfers):
                h2d_dst.copy_(h2d_src, non_blocking=True)
            h2d_end.record(h2d_stream)
        with torch.cuda.stream(d2h_stream):
            d2h_start.record(d2h_stream)
            for _ in range(n_transfers):
                d2h_dst.copy_(d2h_src, non_blocking=True)
            d2h_end.record(d2h_stream)
        h2d_end.synchronize()
        d2h_end.synchronize()

        total_gb = size_mb * n_transfers / 1024
        h2d_elapsed = h2d_start.elapsed_time(h2d_end) / 1000.0
        d2h_elapsed = d2h_start.elapsed_time(d2h_end) / 1000.0
        results[gpu_id] = (total_gb / h2d_elapsed, total_gb / d2h_elapsed)
    return results


def pcie_bidirectional_concurrent(
    gpu_ids: list[int], numa_map: dict[int, set[int]],
    size_mb: int, n_transfers: int,
) -> dict[int, tuple[float, float]]:
    """Simultaneous H2D + D2H on all GPUs concurrently. Returns {gpu: (h2d_bw, d2h_bw)}."""
    results = {}
    barrier = threading.Barrier(len(gpu_ids))

    def worker(gpu_id):
        torch.cuda.set_device(gpu_id)
        n_elems = size_mb * 1024 * 1024 // 4
        cpus = numa_map[gpu_id]

        h2d_src = alloc_pinned_numa(n_elems, torch.float32, cpus)
        h2d_dst = torch.empty(n_elems, dtype=torch.float32, device=f"cuda:{gpu_id}")
        d2h_src = torch.randn(n_elems, dtype=torch.float32, device=f"cuda:{gpu_id}")
        d2h_dst = alloc_pinned_numa(n_elems, torch.float32, cpus)

        h2d_stream = torch.cuda.Stream(device=gpu_id)
        d2h_stream = torch.cuda.Stream(device=gpu_id)

        with torch.cuda.stream(h2d_stream):
            for _ in range(3):
                h2d_dst.copy_(h2d_src, non_blocking=True)
        with torch.cuda.stream(d2h_stream):
            for _ in range(3):
                d2h_dst.copy_(d2h_src, non_blocking=True)
        h2d_stream.synchronize()
        d2h_stream.synchronize()

        barrier.wait()

        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        d2h_start = torch.cuda.Event(enable_timing=True)
        d2h_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(h2d_stream):
            h2d_start.record(h2d_stream)
            for _ in range(n_transfers):
                h2d_dst.copy_(h2d_src, non_blocking=True)
            h2d_end.record(h2d_stream)
        with torch.cuda.stream(d2h_stream):
            d2h_start.record(d2h_stream)
            for _ in range(n_transfers):
                d2h_dst.copy_(d2h_src, non_blocking=True)
            d2h_end.record(d2h_stream)
        h2d_end.synchronize()
        d2h_end.synchronize()

        total_gb = size_mb * n_transfers / 1024
        h2d_elapsed = h2d_start.elapsed_time(h2d_end) / 1000.0
        d2h_elapsed = d2h_start.elapsed_time(d2h_end) / 1000.0
        results[gpu_id] = (total_gb / h2d_elapsed, total_gb / d2h_elapsed)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpu_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ═══════════════════════════════════════════════════════════════════════
#  NVLink benchmarks (GPU ↔ GPU)
# ═══════════════════════════════════════════════════════════════════════

def nvlink_unidirectional(
    gpu_ids: list[int], size_mb: int, n_transfers: int,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Measure unidirectional GPU→GPU bandwidth for all pairs. Returns {(src,dst): (GB/s, seconds)}."""
    results = {}
    for src in gpu_ids:
        for dst in gpu_ids:
            if src == dst:
                continue
            src_t = torch.randn(
                size_mb * 1024 * 1024 // 4, dtype=torch.float32,
                device=f"cuda:{src}")
            dst_t = torch.empty_like(src_t, device=f"cuda:{dst}")
            stream = torch.cuda.Stream(device=dst)

            with torch.cuda.stream(stream):
                for _ in range(5):
                    dst_t.copy_(src_t, non_blocking=True)
            stream.synchronize()

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start_ev.record(stream)
                for _ in range(n_transfers):
                    dst_t.copy_(src_t, non_blocking=True)
                end_ev.record(stream)
            end_ev.synchronize()

            elapsed = start_ev.elapsed_time(end_ev) / 1000.0
            total_gb = size_mb * n_transfers / 1024
            results[(src, dst)] = (total_gb / elapsed, elapsed)
    return results


def nvlink_bidirectional(
    gpu_ids: list[int], size_mb: int, n_transfers: int,
) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """Simultaneous bidirectional per pair. Returns {(a,b): (a→b GB/s, b→a GB/s, a→b sec, b→a sec)}."""
    results = {}
    for i, a in enumerate(gpu_ids):
        for b in gpu_ids[i + 1:]:
            src_ab = torch.randn(
                size_mb * 1024 * 1024 // 4, dtype=torch.float32,
                device=f"cuda:{a}")
            dst_ab = torch.empty_like(src_ab, device=f"cuda:{b}")
            src_ba = torch.randn(
                size_mb * 1024 * 1024 // 4, dtype=torch.float32,
                device=f"cuda:{b}")
            dst_ba = torch.empty_like(src_ba, device=f"cuda:{a}")

            stream_ab = torch.cuda.Stream(device=b)
            stream_ba = torch.cuda.Stream(device=a)

            pair_results = {}
            barrier = threading.Barrier(2)

            def transfer(label, src, dst, stream):
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        dst.copy_(src, non_blocking=True)
                stream.synchronize()

                barrier.wait()

                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    start_ev.record(stream)
                    for _ in range(n_transfers):
                        dst.copy_(src, non_blocking=True)
                    end_ev.record(stream)
                end_ev.synchronize()

                elapsed = start_ev.elapsed_time(end_ev) / 1000.0
                total_gb = size_mb * n_transfers / 1024
                pair_results[label] = (total_gb / elapsed, elapsed)

            t1 = threading.Thread(
                target=transfer,
                args=(f"{a}->{b}", src_ab, dst_ab, stream_ab))
            t2 = threading.Thread(
                target=transfer,
                args=(f"{b}->{a}", src_ba, dst_ba, stream_ba))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            bw_ab, el_ab = pair_results[f"{a}->{b}"]
            bw_ba, el_ba = pair_results[f"{b}->{a}"]
            results[(a, b)] = (bw_ab, bw_ba, el_ab, el_ba)
    return results


# ═══════════════════════════════════════════════════════════════════════
#  PCIe × NVLink cross-contention
# ═══════════════════════════════════════════════════════════════════════

def pcie_nvlink_cross_contention(
    gpu_ids: list[int], numa_map: dict[int, set[int]],
    size_mb: int, n_transfers: int,
) -> dict[int, dict]:
    """For each GPU A, measure simultaneous PCIe H2D (host→A) + NVLink (A→B).

    Tests whether PCIe and NVLink transfers from/to the same GPU contend.
    For each GPU A, picks the first available peer B != A.

    Returns:
        {gpu_a: {
            'peer': gpu_b,
            'pcie_h2d_solo': float,     # GB/s, H2D alone
            'nvlink_solo': float,       # GB/s, A→B alone
            'pcie_h2d_combined': float, # GB/s, H2D during simultaneous NVLink
            'nvlink_combined': float,   # GB/s, A→B during simultaneous PCIe
        }}
    """
    if len(gpu_ids) < 2:
        return {}

    results = {}
    for gpu_a in gpu_ids:
        # Pick a peer GPU
        gpu_b = next(g for g in gpu_ids if g != gpu_a)
        n_elems = size_mb * 1024 * 1024 // 4
        total_gb = size_mb * n_transfers / 1024

        # Allocate buffers
        pcie_src = alloc_pinned_numa(n_elems, torch.float32, numa_map[gpu_a])
        pcie_dst = torch.empty(n_elems, dtype=torch.float32,
                               device=f"cuda:{gpu_a}")
        nvlink_src = torch.randn(n_elems, dtype=torch.float32,
                                 device=f"cuda:{gpu_a}")
        nvlink_dst = torch.empty(n_elems, dtype=torch.float32,
                                 device=f"cuda:{gpu_b}")

        pcie_stream = torch.cuda.Stream(device=gpu_a)
        nvlink_stream = torch.cuda.Stream(device=gpu_b)

        # Warmup both paths
        with torch.cuda.stream(pcie_stream):
            for _ in range(3):
                pcie_dst.copy_(pcie_src, non_blocking=True)
        with torch.cuda.stream(nvlink_stream):
            for _ in range(3):
                nvlink_dst.copy_(nvlink_src, non_blocking=True)
        pcie_stream.synchronize()
        nvlink_stream.synchronize()

        # ── Solo PCIe H2D ──
        pcie_start = torch.cuda.Event(enable_timing=True)
        pcie_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(pcie_stream):
            pcie_start.record(pcie_stream)
            for _ in range(n_transfers):
                pcie_dst.copy_(pcie_src, non_blocking=True)
            pcie_end.record(pcie_stream)
        pcie_end.synchronize()
        pcie_solo = total_gb / (pcie_start.elapsed_time(pcie_end) / 1000.0)

        # ── Solo NVLink A→B ──
        nv_start = torch.cuda.Event(enable_timing=True)
        nv_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(nvlink_stream):
            nv_start.record(nvlink_stream)
            for _ in range(n_transfers):
                nvlink_dst.copy_(nvlink_src, non_blocking=True)
            nv_end.record(nvlink_stream)
        nv_end.synchronize()
        nv_solo = total_gb / (nv_start.elapsed_time(nv_end) / 1000.0)

        # ── Simultaneous PCIe H2D + NVLink A→B ──
        combined = {}
        barrier = threading.Barrier(2)

        def run_pcie():
            barrier.wait()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(pcie_stream):
                s.record(pcie_stream)
                for _ in range(n_transfers):
                    pcie_dst.copy_(pcie_src, non_blocking=True)
                e.record(pcie_stream)
            e.synchronize()
            combined['pcie'] = total_gb / (s.elapsed_time(e) / 1000.0)

        def run_nvlink():
            barrier.wait()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(nvlink_stream):
                s.record(nvlink_stream)
                for _ in range(n_transfers):
                    nvlink_dst.copy_(nvlink_src, non_blocking=True)
                e.record(nvlink_stream)
            e.synchronize()
            combined['nvlink'] = total_gb / (s.elapsed_time(e) / 1000.0)

        t1 = threading.Thread(target=run_pcie)
        t2 = threading.Thread(target=run_nvlink)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        results[gpu_a] = {
            'peer': gpu_b,
            'pcie_h2d_solo': pcie_solo,
            'nvlink_solo': nv_solo,
            'pcie_h2d_combined': combined['pcie'],
            'nvlink_combined': combined['nvlink'],
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Profile PCIe and NVLink interconnect bandwidth")
    parser.add_argument("--size-mb", type=int, default=336,
                        help="Transfer size in MB (default: 336)")
    parser.add_argument("--transfers", type=int, default=10,
                        help="Number of transfers per measurement (default: 10)")
    args = parser.parse_args()

    size_mb = args.size_mb
    n_transfers = args.transfers
    n_gpus = torch.cuda.device_count()
    gpu_ids = list(range(n_gpus))

    print(f"GPUs: {n_gpus}x {torch.cuda.get_device_name(0)}")
    numa_map = build_gpu_numa_map()
    for g in gpu_ids:
        node = _get_gpu_numa_node(g)
        cpus = sorted(numa_map[g])
        print(f"  GPU {g} -> NUMA node {node}, CPUs {cpus[0]}-{cpus[-1]}")
    print(f"Transfer size: {size_mb} MB, {n_transfers} transfers per measurement")

    # ── 1. PCIe: sequential H2D ──────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  PCIe H2D — Sequential (one GPU at a time)")
    print(f"{'=' * 65}")
    solo = pcie_h2d_sequential(gpu_ids, numa_map, size_mb, n_transfers)
    for g in gpu_ids:
        bw, elapsed = solo[g]
        print(f"  GPU {g}: {bw:7.2f} GB/s  ({elapsed:.3f}s)")
    avg_solo = sum(bw for bw, _ in solo.values()) / len(solo)

    # ── 2. PCIe: concurrent H2D ──────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  PCIe H2D — Concurrent ({n_gpus} GPUs simultaneously)")
    print(f"{'=' * 65}")
    conc = pcie_h2d_concurrent(gpu_ids, numa_map, size_mb, n_transfers)
    for g in gpu_ids:
        bw, elapsed = conc[g]
        ratio = bw / solo[g][0]
        print(f"  GPU {g}: {bw:7.2f} GB/s  ({elapsed:.3f}s)  "
              f"ratio vs solo: {ratio:.2f}x")
    avg_conc = sum(bw for bw, _ in conc.values()) / len(conc)
    print(f"\n  Avg solo:       {avg_solo:.2f} GB/s")
    print(f"  Avg concurrent: {avg_conc:.2f} GB/s")
    print(f"  Contention:     {avg_conc / avg_solo:.2f}x (1.00 = none)")

    # ── 3. PCIe: bidirectional per GPU ───────────────────────────────
    print(f"\n{'=' * 65}")
    print("  PCIe Bidirectional (H2D + D2H simultaneously, per GPU)")
    print(f"{'=' * 65}")
    bidi = pcie_bidirectional(gpu_ids, numa_map, size_mb, n_transfers)
    for g in gpu_ids:
        h2d, d2h = bidi[g]
        print(f"  GPU {g}:  H2D {h2d:7.2f} GB/s  |  D2H {d2h:7.2f} GB/s  |  "
              f"sum {h2d + d2h:7.2f} GB/s")
    avg_h2d_bi = sum(h for h, _ in bidi.values()) / len(bidi)
    avg_d2h_bi = sum(d for _, d in bidi.values()) / len(bidi)
    print(f"\n  Avg H2D (bidir): {avg_h2d_bi:.2f} GB/s  (solo: {avg_solo:.2f})")
    print(f"  Avg D2H (bidir): {avg_d2h_bi:.2f} GB/s")
    print(f"  H2D contention:  {avg_h2d_bi / avg_solo:.2f}x")

    # ── 4. PCIe: bidirectional concurrent ────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  PCIe Bidirectional Concurrent ({n_gpus} GPUs simultaneously)")
    print(f"{'=' * 65}")
    bidi_conc = pcie_bidirectional_concurrent(
        gpu_ids, numa_map, size_mb, n_transfers)
    for g in gpu_ids:
        h2d, d2h = bidi_conc[g]
        print(f"  GPU {g}:  H2D {h2d:7.2f} GB/s  |  D2H {d2h:7.2f} GB/s  |  "
              f"sum {h2d + d2h:7.2f} GB/s")
    avg_h2d_bc = sum(h for h, _ in bidi_conc.values()) / len(bidi_conc)
    avg_d2h_bc = sum(d for _, d in bidi_conc.values()) / len(bidi_conc)
    print(f"\n  Avg H2D: {avg_h2d_bc:.2f} GB/s")
    print(f"  Avg D2H: {avg_d2h_bc:.2f} GB/s")
    print(f"  Contention vs solo bidir — "
          f"H2D: {avg_h2d_bc / avg_h2d_bi:.2f}x  "
          f"D2H: {avg_d2h_bc / avg_d2h_bi:.2f}x")

    # ── 5. NVLink: unidirectional ────────────────────────────────────
    if n_gpus >= 2:
        print(f"\n{'=' * 65}")
        print("  NVLink — Unidirectional")
        print(f"{'=' * 65}")
        uni = nvlink_unidirectional(gpu_ids, size_mb, n_transfers)
        for (src, dst), (bw, elapsed) in sorted(uni.items()):
            print(f"  GPU {src} -> GPU {dst}:  {bw:7.2f} GB/s  ({elapsed:.3f}s)")
        avg_uni = sum(bw for bw, _ in uni.values()) / len(uni)

        # ── 6. NVLink: bidirectional ─────────────────────────────────
        print(f"\n{'=' * 65}")
        print("  NVLink — Simultaneous Bidirectional")
        print(f"{'=' * 65}")
        bidi_nv = nvlink_bidirectional(gpu_ids, size_mb, n_transfers)
        for (a, b), (bw_ab, bw_ba, _, _) in sorted(bidi_nv.items()):
            print(f"  GPU {a} <-> GPU {b}:  "
                  f"{a}->{b} {bw_ab:7.2f} GB/s  |  "
                  f"{b}->{a} {bw_ba:7.2f} GB/s  |  "
                  f"sum {bw_ab + bw_ba:7.2f} GB/s")
        avg_bidi_nv = sum(a + b for a, b, _, _ in bidi_nv.values()) / len(bidi_nv) / 2
        print(f"\n  Avg unidirectional:          {avg_uni:7.2f} GB/s")
        print(f"  Avg bidirectional (per dir):  {avg_bidi_nv:7.2f} GB/s")
        print(f"  Bidir / Unidir ratio:         {avg_bidi_nv / avg_uni:.2f}x "
              f"(1.00 = no contention)")

        # ── 7. PCIe x NVLink cross-contention ────────────────────────
        print(f"\n{'=' * 65}")
        print("  PCIe x NVLink Cross-Contention")
        print(f"  (simultaneous Host->A via PCIe + A->B via NVLink)")
        print(f"{'=' * 65}")
        cross = pcie_nvlink_cross_contention(
            gpu_ids, numa_map, size_mb, n_transfers)
        for gpu_a in gpu_ids:
            r = cross[gpu_a]
            gpu_b = r['peer']
            pcie_ratio = r['pcie_h2d_combined'] / r['pcie_h2d_solo']
            nv_ratio = r['nvlink_combined'] / r['nvlink_solo']
            print(f"  GPU {gpu_a} (peer={gpu_b}):")
            print(f"    PCIe  H2D:  solo {r['pcie_h2d_solo']:7.2f} GB/s  "
                  f"combined {r['pcie_h2d_combined']:7.2f} GB/s  "
                  f"ratio {pcie_ratio:.2f}x")
            print(f"    NVLink {gpu_a}->{gpu_b}: solo {r['nvlink_solo']:7.2f} GB/s  "
                  f"combined {r['nvlink_combined']:7.2f} GB/s  "
                  f"ratio {nv_ratio:.2f}x")

        avg_pcie_ratio = sum(
            r['pcie_h2d_combined'] / r['pcie_h2d_solo']
            for r in cross.values()) / len(cross)
        avg_nv_ratio = sum(
            r['nvlink_combined'] / r['nvlink_solo']
            for r in cross.values()) / len(cross)
        print(f"\n  Avg PCIe retention:  {avg_pcie_ratio:.2f}x (1.00 = no contention)")
        print(f"  Avg NVLink retention: {avg_nv_ratio:.2f}x (1.00 = no contention)")
    else:
        print("\n  (Skipping NVLink and cross-contention tests: single GPU)")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  Summary")
    print(f"{'=' * 65}")
    print(f"  PCIe H2D solo avg:              {avg_solo:7.2f} GB/s")
    print(f"  PCIe H2D concurrent avg:        {avg_conc:7.2f} GB/s  "
          f"({avg_conc / avg_solo:.2f}x)")
    print(f"  PCIe bidir H2D avg:             {avg_h2d_bi:7.2f} GB/s  "
          f"({avg_h2d_bi / avg_solo:.2f}x)")
    if n_gpus >= 2:
        print(f"  NVLink unidirectional avg:      {avg_uni:7.2f} GB/s")
        print(f"  NVLink bidirectional avg:       {avg_bidi_nv:7.2f} GB/s  "
              f"({avg_bidi_nv / avg_uni:.2f}x)")
        print(f"  Cross-contention PCIe retention: {avg_pcie_ratio:.2f}x")
        print(f"  Cross-contention NVLink retention: {avg_nv_ratio:.2f}x")


if __name__ == "__main__":
    main()
