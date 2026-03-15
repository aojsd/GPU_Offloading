#!/usr/bin/env python3
"""
Test suite for the patched moe_align_block_size CUDA kernel.

Tests both correctness (bit-identical to vLLM original for <1024 experts,
correct output for >1024) and performance (eager + CUDA graph captured).

Run inside the container:
    bash GH_GB200_env/pytorch_ngc.sh "cd /workspace/GPU_Offloading/src/MoE && \
        python -m cuda.tests.test_moe_align_patch"
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cuda.patch_moe_align import get_extension, moe_align_block_size_replacement
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size as original_moe_align,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod

MODELS = {
    "OLMoE":     {"E": 64,  "topk": 8, "I": 1024,  "H": 2048},
    "Mixtral":   {"E": 8,   "topk": 2, "I": 14336, "H": 4096},
    "DSv2-Lite": {"E": 64,  "topk": 6, "I": 1408,  "H": 2048},
    "DSv2":      {"E": 160, "topk": 6, "I": 1536,  "H": 5120},
}

BATCH_SIZES = [1, 4, 16, 32, 64, 128]
WARMUP = 10
ITERS = 50
TRIALS = 3  # run each measurement this many times and take the median

_original = None


def setup():
    global _original
    print("Building patched CUDA extension...")
    ext = get_extension()
    print(f"  Extension: {ext}")
    print(f"  Compiled to: /tmp/moe_align_ext/moe_align_ext.so")
    _original = fused_moe_mod.moe_align_block_size


def make_expert_map(E, cache_size, device="cuda"):
    expert_map = torch.full((E,), -1, dtype=torch.int32, device=device)
    perm = torch.randperm(cache_size, device=device)[:E]
    expert_map[:E] = perm.to(torch.int32)
    return expert_map


# ═══════════════════════════════════════════════════════════════════
# Correctness tests
# ═══════════════════════════════════════════════════════════════════

def compare_kernels(name, topk_ids, block_size, num_experts, expert_map=None):
    """Compare original vs patched kernel output."""
    kwargs = {"ignore_invalid_experts": True} if expert_map is not None else {}
    if expert_map is not None:
        kwargs["expert_map"] = expert_map

    s1, e1, n1 = original_moe_align(topk_ids, block_size, num_experts, **kwargs)
    s2, e2, n2 = moe_align_block_size_replacement(topk_ids, block_size, num_experts, **kwargs)

    n1v, n2v = n1.item(), n2.item()
    maxn = max(n1v, n2v)
    nb = maxn // block_size

    npost_ok = n1v == n2v
    expert_ok = torch.equal(e1[:nb], e2[:nb])
    # sorted_token_ids may differ in within-expert order (atomicAdd non-determinism)
    # This is expected and does NOT affect fused_experts output.

    ok = npost_ok and expert_ok
    status = "PASS" if ok else "FAIL"
    print(f"    {name}: {status}  n_post={n1v}/{n2v}  expert_ids={'ok' if expert_ok else 'MISMATCH'}")
    if not ok:
        print(f"      orig expert[:8]  = {e1[:8].tolist()}")
        print(f"      ptch expert[:8]  = {e2[:8].tolist()}")
    return ok


def test_correctness():
    print("\n=== Correctness: kernel output (expert_ids + num_tokens_post_pad) ===")
    all_pass = True

    # Basic tests
    topk = torch.tensor([[2,3],[1,2],[1,3],[1,2]], dtype=torch.int32, device="cuda")
    all_pass &= compare_kernels("minimal, no map", topk, 4, 4)

    emap = torch.tensor([99, 5, 10, 15], dtype=torch.int32, device="cuda")
    all_pass &= compare_kernels("minimal, w/ map", topk, 4, 100, emap)

    # Per-model tests
    for mname, cfg in MODELS.items():
        E, topk_k = cfg["E"], cfg["topk"]
        for bs in [1, 4, 32]:
            ids = torch.randint(0, E, (bs, topk_k), dtype=torch.int32, device="cuda")
            all_pass &= compare_kernels(f"{mname} bs={bs}", ids, 64, E)

    # Offloading tests (with expert_map)
    for E, cs in [(64, 256), (64, 960), (160, 640), (160, 960)]:
        emap = make_expert_map(E, cs, "cuda")
        ids = torch.randint(0, E, (32, 6), dtype=torch.int32, device="cuda")
        all_pass &= compare_kernels(f"offload E={E} cs={cs}", ids, 64, cs, emap)

    return all_pass


def test_large_expert_counts():
    print("\n=== Large expert counts (>1024, patched kernel only) ===")
    all_pass = True

    for ne in [1024, 1344, 1920, 3000, 5000]:
        ids = torch.randint(0, ne, (32, 6), dtype=torch.int32, device="cuda")
        s, e, n = moe_align_block_size_replacement(ids, 64, ne)
        nv = n.item()

        sentinel = ids.numel()
        real_tokens = s[:nv][s[:nv] < sentinel]
        nb = nv // 64
        eids = e[:nb]

        checks = []
        checks.append(("n_post > 0", nv > 0))
        checks.append(("n_post <= sorted_ids size", nv <= s.shape[0]))
        checks.append(("token ids in range", real_tokens.max().item() < sentinel if real_tokens.numel() > 0 else True))
        checks.append(("expert ids in range", (eids >= 0).all().item() and (eids < ne).all().item()))
        checks.append(("all tokens placed", real_tokens.shape[0] == ids.numel()))

        ok = all(c[1] for c in checks)
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        fails = [c[0] for c in checks if not c[1]]
        extra = f"  FAILED: {fails}" if fails else ""
        print(f"    num_experts={ne}: {status}  n_post={nv}  tokens={real_tokens.shape[0]}/{ids.numel()}{extra}")

    return all_pass


def test_fused_experts_e2e():
    print("\n=== End-to-end fused_experts (patched sort → same Triton GEMM) ===")
    all_pass = True

    for mname, cfg in MODELS.items():
        E, topk_k, I, H = cfg["E"], cfg["topk"], cfg["I"], cfg["H"]
        for bs in [1, 8, 32]:
            hidden = torch.randn(bs, H, dtype=torch.bfloat16, device="cuda")
            w1 = torch.randn(E, 2*I, H, dtype=torch.bfloat16, device="cuda")
            w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda")
            ids = torch.randint(0, E, (bs, topk_k), dtype=torch.int32, device="cuda")
            tw = torch.softmax(torch.randn(bs, topk_k, dtype=torch.float32, device="cuda"), -1)

            fused_moe_mod.moe_align_block_size = _original
            out1 = fused_experts(hidden, w1, w2, tw, ids)

            fused_moe_mod.moe_align_block_size = moe_align_block_size_replacement
            out2 = fused_experts(hidden, w1, w2, tw, ids)

            fused_moe_mod.moe_align_block_size = _original

            diff = (out1 - out2).abs().max().item()
            ok = diff == 0.0
            all_pass &= ok
            status = "PASS" if ok else f"FAIL (diff={diff:.6f})"
            print(f"    {mname} bs={bs}: {status}")

            del hidden, w1, w2
            torch.cuda.empty_cache()

    # With expert_map
    E, cs, I, H, topk_k, bs = 64, 256, 1408, 2048, 6, 16
    hidden = torch.randn(bs, H, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(cs, 2*I, H, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(cs, H, I, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, E, (bs, topk_k), dtype=torch.int32, device="cuda")
    tw = torch.softmax(torch.randn(bs, topk_k, dtype=torch.float32, device="cuda"), -1)
    emap = make_expert_map(E, cs, "cuda")

    fused_moe_mod.moe_align_block_size = _original
    out1 = fused_experts(hidden, w1, w2, tw, ids, expert_map=emap, global_num_experts=cs)
    fused_moe_mod.moe_align_block_size = moe_align_block_size_replacement
    out2 = fused_experts(hidden, w1, w2, tw, ids, expert_map=emap, global_num_experts=cs)
    fused_moe_mod.moe_align_block_size = _original

    diff = (out1 - out2).abs().max().item()
    ok = diff == 0.0
    all_pass &= ok
    print(f"    offload E={E} cs={cs}: {'PASS' if ok else f'FAIL (diff={diff})'}")
    del hidden, w1, w2
    torch.cuda.empty_cache()

    return all_pass


# ═══════════════════════════════════════════════════════════════════
# Performance tests
# ═══════════════════════════════════════════════════════════════════

def _run_bench_eager(E, topk, I, H, batch_size, cache_size, use_patch):
    """Single trial of eager benchmark. Returns ms or None."""
    device = "cuda"
    hidden = torch.randn(batch_size, H, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(cache_size, 2 * I, H, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(cache_size, H, I, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, E, (batch_size, topk), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn(batch_size, topk, dtype=torch.float32, device=device), -1)

    kwargs = {}
    if cache_size != E:
        kwargs['expert_map'] = make_expert_map(E, cache_size, device)
        kwargs['global_num_experts'] = cache_size

    fused_moe_mod.moe_align_block_size = moe_align_block_size_replacement if use_patch else _original

    try:
        for _ in range(WARMUP):
            fused_experts(hidden, w1, w2, topk_weights, topk_ids, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fused_experts(hidden, w1, w2, topk_weights, topk_ids, **kwargs)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / ITERS * 1000
    except RuntimeError as e:
        if "padded_num_experts" in str(e):
            ms = None
        else:
            raise
    finally:
        fused_moe_mod.moe_align_block_size = _original
    del w1, w2, hidden
    torch.cuda.empty_cache()
    return ms


def _run_bench_graph(E, topk, I, H, batch_size, cache_size, use_patch):
    """Single trial of graph-captured benchmark. Returns ms or None."""
    device = "cuda"
    inp = torch.randn(batch_size, H, dtype=torch.bfloat16, device=device)
    res = torch.randn(batch_size, H, dtype=torch.bfloat16, device=device)
    tw = torch.softmax(torch.randn(batch_size, topk, dtype=torch.float32, device=device), -1)
    ti = torch.randint(0, E, (batch_size, topk), dtype=torch.int32, device=device)
    out = torch.empty(batch_size, H, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(cache_size, 2 * I, H, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(cache_size, H, I, dtype=torch.bfloat16, device=device)

    kwargs = {}
    if cache_size != E:
        kwargs['expert_map'] = make_expert_map(E, cache_size, device)
        kwargs['global_num_experts'] = cache_size

    fused_moe_mod.moe_align_block_size = moe_align_block_size_replacement if use_patch else _original

    def fn():
        r = fused_experts(inp, w1, w2, tw, ti, **kwargs)
        out.copy_(res + r)

    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            g.replay()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / ITERS * 1000
    except Exception:
        ms = None
    finally:
        fused_moe_mod.moe_align_block_size = _original
    del w1, w2, inp, res, out
    torch.cuda.empty_cache()
    return ms


def _median_bench(bench_fn, *args, trials=TRIALS):
    """Run bench_fn `trials` times and return the median."""
    results = []
    for _ in range(trials):
        v = bench_fn(*args)
        if v is not None:
            results.append(v)
    if not results:
        return None
    results.sort()
    return results[len(results) // 2]


def test_performance():
    print(f"\n=== Performance: original vs patched ({TRIALS} trials × {ITERS} iters, median) ===")
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    for mname, cfg in MODELS.items():
        E, topk, I, H = cfg["E"], cfg["topk"], cfg["I"], cfg["H"]
        print(f"\n  {mname} (E={E}, topk={topk}, I={I}, H={H}):")

        # Test at cache=E (baseline) and cache=1920 (beyond the 992 limit)
        cache_sizes = [E]
        if E <= 960:
            cache_sizes.append(1920)

        for cs in cache_sizes:
            mem_mb = (cs * 2 * I * H * 2 + cs * H * I * 2) / 1e6
            if mem_mb > gpu_mem_gb * 1000 * 0.70:
                print(f"\n    cache={cs}: SKIP ({mem_mb:.0f} MB > 70% of GPU)")
                continue
            limit = "ok" if cs <= 992 else ">992"
            print(f"\n    cache={cs} ({limit}):")
            print(f"      {'bs':>5s} {'orig_eager':>11s} {'patch_eager':>11s} {'orig_graph':>11s} {'patch_graph':>11s}  {'e_ratio':>8s} {'g_ratio':>8s}")

            for bs in BATCH_SIZES:
                oe = _median_bench(_run_bench_eager, E, topk, I, H, bs, cs, False)
                pe = _median_bench(_run_bench_eager, E, topk, I, H, bs, cs, True)
                og = _median_bench(_run_bench_graph, E, topk, I, H, bs, cs, False)
                pg = _median_bench(_run_bench_graph, E, topk, I, H, bs, cs, True)

                def f(v): return f"{v:>10.3f}ms" if v else f"{'N/A':>11s}"
                def r(a, b):
                    if a and b: return f"{b/a:>7.2f}x"
                    return f"{'':>8s}"
                print(f"      {bs:>5d} {f(oe)} {f(pe)} {f(og)} {f(pg)}  {r(oe,pe)} {r(og,pg)}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    setup()

    p1 = test_correctness()
    p2 = test_large_expert_counts()
    p3 = test_fused_experts_e2e()
    test_performance()

    print(f"\n{'='*60}")
    all_ok = p1 and p2 and p3
    print(f"CORRECTNESS: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
