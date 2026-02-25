"""Nsight Systems profiling: Custom CUDA Graph vs vLLM SOTA across sequence lengths.

Runs each (engine, seq_len) pair as a separate process under nsys.
Outputs per-seq-len .prof files to profiling/<model_name>/.

Usage:
    python src/MoE/nsys_seq_len_comparison.py custom 128     # one run
    python src/MoE/nsys_seq_len_comparison.py vllm 1024      # one run
    python src/MoE/nsys_seq_len_comparison.py all             # all combos
    python src/MoE/nsys_seq_len_comparison.py all --model path/to/model
"""

import os
import sys
import subprocess
import sqlite3
import textwrap
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = str(SCRIPT_DIR / "models" / "OLMoE-1B-7B")
PROFILE_BASE = SCRIPT_DIR / "profiling"
NSYS = os.path.expanduser("~/software/cuda-12.8/bin/nsys")


def _profile_dir(model_dir):
    """Return profiling output directory for a given model, e.g. profiling/OLMoE-1B-7B/."""
    model_name = Path(model_dir).name
    return PROFILE_BASE / model_name

SEQ_LENS = [128, 256, 512, 1024, 2048]
NUM_WARMUP = 10
NUM_DECODE = 30  # decode steps to profile (enough for stats, short enough for traces)


# =====================================================================
#  Kernel classification (same rules as profile_and_export.py)
# =====================================================================

KERNEL_RULES = [
    ("fused_moe_kernel",                "MoE (fused_experts)"),
    ("act_and_mul_kernel",              "MoE (SiLU gate)"),
    ("paged_attention",                 "Attention (paged_attn_v2)"),
    ("flash_fwd",                       "Attention (FlashAttention)"),
    ("fmha",                            "Attention (FlashAttention)"),
    ("BatchDecodeWithPagedKVCacheKernel", "Attention (FlashInfer)"),
    ("scaled_dot_product",              "Attention (SDPA)"),
    ("reshape_and_cache",              "KV cache store"),
    ("RotaryPosIds",                    "RoPE (FlashInfer)"),
    ("rope",                            "RoPE"),
    ("triton_red_fused",                "torch.compile fused"),
    ("triton_poi_fused",                "torch.compile fused"),
    ("triton_per_fused",                "torch.compile fused"),
    ("device_kernel",                   "Attention (FlashInfer graph)"),
    ("prepare_varlen_num_blocks",       "Attention (FlashInfer meta)"),
    ("layer_norm_kernel",               "RMSNorm"),
    ("rms_norm",                        "RMSNorm"),
    ("nvjet",                           "Linear (cuBLAS)"),
    ("gemm",                            "Linear (cuBLAS)"),
    ("cutlass",                         "Linear (cuBLAS)"),
    ("cublasLt",                        "Linear (cuBLAS)"),
    ("ampere_sgemm",                    "Linear (cuBLAS)"),
    ("gatherTopK",                      "Router (topk)"),
    ("radixSort",                       "Router (sort)"),
    ("bitonicSort",                     "Router (sort)"),
    ("softmax",                         "Router (softmax)"),
    ("searchsorted",                    "MoE align (patched)"),
    ("DeviceScan",                      "MoE align (patched)"),
    ("scatter",                         "KV cache indexing"),
    ("index_elementwise",               "KV cache indexing"),
    ("index_put",                       "KV cache indexing"),
    ("indexSelect",                     "KV cache indexing"),
    ("reduce_kernel",                   "Reduce / sum"),
    ("copy_kernel",                     "Copy / cast"),
    ("direct_copy",                     "Copy / cast"),
    ("LoadWithCast",                    "Copy / cast"),
    ("StoreWithCast",                   "Copy / cast"),
    ("FillFunctor",                     "Elementwise"),
    ("arange",                          "Elementwise"),
    ("elementwise",                     "Elementwise"),
    ("vectorized_elementwise",          "Elementwise"),
    ("memset",                          "Memset / memcpy"),
    ("memcpy",                          "Memset / memcpy"),
]


def classify_kernel(name):
    name_lower = name.lower()
    for substr, cat in KERNEL_RULES:
        if substr.lower() in name_lower:
            return cat
    return "Other"


# =====================================================================
#  Driver script generation
# =====================================================================

def write_custom_driver(path, seq_len, model_dir):
    """Driver: custom engine CUDA graph decode at target seq_len."""
    max_seq = max(seq_len + 256, 2560)  # headroom
    code = textwrap.dedent(f"""\
        import sys, torch
        sys.path.insert(0, "{SCRIPT_DIR}")
        from moe_engine import MoEEngine

        SEQ_LEN = {seq_len}
        MAX_SEQ = {max_seq}
        NUM_WARMUP = {NUM_WARMUP}
        NUM_DECODE = {NUM_DECODE}

        engine = MoEEngine("{model_dir}", max_batch_size=1, max_seq_len=MAX_SEQ)
        engine.capture_decode_cuda_graph(
            batch_size=1, warmup_seq_len=128,
            max_decode_tokens=MAX_SEQ - 128)

        # Fill KV cache with random data to simulate target seq_len
        engine.k_cache.normal_(0, 0.01)
        engine.v_cache.normal_(0, 0.01)

        token = torch.tensor([100], device="cuda")
        pos = torch.tensor([SEQ_LEN], dtype=torch.int32, device="cuda")

        # Warmup (graph already captured, just exercise replay)
        for _ in range(NUM_WARMUP):
            engine.seq_lens[0] = SEQ_LEN
            engine._seq_lens_cpu[0] = SEQ_LEN
            engine._decode_step_graphed(token, pos)
        torch.cuda.synchronize()

        # Profiled region
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(NUM_DECODE):
            engine.seq_lens[0] = SEQ_LEN
            engine._seq_lens_cpu[0] = SEQ_LEN
            engine._decode_step_graphed(token, pos)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    """)
    Path(path).write_text(code)


def write_vllm_driver(path, seq_len, model_dir):
    """Driver: vLLM SOTA decode at target seq_len."""
    code = textwrap.dedent(f"""\
        import os, sys, torch
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

        sys.path.insert(0, "{SCRIPT_DIR}")
        import moe_engine  # glibc patches

        from vllm import LLM, SamplingParams

        SEQ_LEN = {seq_len}
        NUM_WARMUP = {NUM_WARMUP}
        NUM_DECODE = {NUM_DECODE}

        llm = LLM(
            model="{model_dir}",
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.95,
        )
        sp = SamplingParams(max_tokens=1, temperature=0)

        # Warmup: trigger CUDA graph capture
        for i in range(3):
            tokens = torch.randint(1, 1000, (256,)).tolist()
            llm.llm_engine.add_request(
                request_id=f"warmup_{{i}}",
                prompt={{"prompt_token_ids": tokens}},
                params=sp,
            )
            while llm.llm_engine.has_unfinished_requests():
                llm.llm_engine.step()
        torch.cuda.synchronize()

        # Generate: prefill at target seq_len, then decode
        total_tokens = NUM_WARMUP + NUM_DECODE
        sp_gen = SamplingParams(max_tokens=total_tokens, temperature=0)
        prompt_ids = list(range(100, 100 + SEQ_LEN))
        llm.llm_engine.add_request(
            request_id="bench",
            prompt={{"prompt_token_ids": prompt_ids}},
            params=sp_gen,
        )

        step = 0
        while llm.llm_engine.has_unfinished_requests():
            if step == 1 + NUM_WARMUP:  # 1 prefill + warmup decodes
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("decode_profile")
            llm.llm_engine.step()
            step += 1

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print(f"Total steps: {{step}} (1 prefill + {{step-1}} decode)")
    """)
    Path(path).write_text(code)


# =====================================================================
#  nsys capture + parse
# =====================================================================

def run_nsys(driver_path, rep_prefix, mode):
    """Run nsys profile, return .nsys-rep path."""
    cmd = [
        NSYS, "profile",
        "-o", rep_prefix,
        "-f", "true",
    ]
    if mode == "custom":
        cmd += [
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-graph-trace=node",
        ]
    else:
        cmd += [
            "--trace=cuda,nvtx",
            "--cuda-graph-trace=node",
        ]
    cmd += [sys.executable, str(driver_path)]
    print(f"    CMD: {' '.join(cmd[-5:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    STDERR (last 2000 chars):\n{result.stderr[-2000:]}")
        raise RuntimeError(f"nsys profile failed (rc={result.returncode})")
    return rep_prefix + ".nsys-rep"


def export_sqlite(rep_path):
    """Export .nsys-rep to .sqlite."""
    sqlite_path = rep_path.replace(".nsys-rep", ".sqlite")
    cmd = [NSYS, "export", "--type", "sqlite",
           "--output", sqlite_path, rep_path, "--force-overwrite=true"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"    Export STDERR: {result.stderr[-500:]}")
    return sqlite_path


def parse_kernels(sqlite_path, num_steps, nvtx_filter=None):
    """Parse kernel data from nsys SQLite export.

    Returns dict with categories, totals, and per-kernel data.
    """
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    # NVTX time window filter
    time_filter = ""
    if nvtx_filter:
        try:
            cur.execute("SELECT start, end FROM NVTX_EVENTS WHERE text = ?",
                        (nvtx_filter,))
            rows = cur.fetchall()
            if rows:
                nvtx_start = min(r[0] for r in rows)
                nvtx_end = max(r[1] for r in rows)
                time_filter = f"AND k.start >= {nvtx_start} AND k.start <= {nvtx_end}"
                print(f"      NVTX '{nvtx_filter}': {(nvtx_end-nvtx_start)/1e6:.1f} ms")
        except sqlite3.OperationalError:
            pass

    # GPU kernel summary
    cur.execute(f"""
        SELECT
            s.value as name,
            COUNT(*) as cnt,
            SUM(k.end - k.start) as total_dur,
            AVG(k.end - k.start) as avg_dur,
            MIN(k.end - k.start) as min_dur,
            MAX(k.end - k.start) as max_dur
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE 1=1 {time_filter}
        GROUP BY k.shortName
        ORDER BY total_dur DESC
    """)
    kern_rows = cur.fetchall()

    total_kernel_ns = 0
    total_launches = 0
    categories = defaultdict(lambda: {"total_ns": 0, "launches": 0})
    per_kernel = []

    for name, cnt, total_dur, avg_dur, min_dur, max_dur in kern_rows:
        cat = classify_kernel(name)
        categories[cat]["total_ns"] += total_dur
        categories[cat]["launches"] += cnt
        total_kernel_ns += total_dur
        total_launches += cnt
        per_kernel.append({
            "name": name, "count": cnt, "total_ns": total_dur,
            "avg_ns": avg_dur, "min_ns": min_dur, "max_ns": max_dur,
            "category": cat,
        })

    conn.close()

    return {
        "categories": dict(categories),
        "total_kernel_ns": total_kernel_ns,
        "total_launches": total_launches,
        "num_steps": num_steps,
        "per_kernel": per_kernel,
    }


# =====================================================================
#  Single profiling run
# =====================================================================

def profile_one(mode, seq_len, model_dir=DEFAULT_MODEL_DIR):
    """Profile one (mode, seq_len) combination. Returns parsed data dict."""
    prof_dir = _profile_dir(model_dir)
    prof_dir.mkdir(parents=True, exist_ok=True)
    name = f"{mode}_seq{seq_len}"
    rep_prefix = str(prof_dir / name)

    # Write driver
    driver_path = prof_dir / f"_driver_{name}.py"
    if mode == "custom":
        write_custom_driver(driver_path, seq_len, model_dir)
    else:
        write_vllm_driver(driver_path, seq_len, model_dir)

    print(f"\n  [{mode} seq_len={seq_len}]")
    print(f"    Driver: {driver_path}")

    # Run nsys
    print(f"    [1/3] Capturing nsys trace...")
    rep_path = run_nsys(str(driver_path), rep_prefix, mode)
    print(f"    Trace: {rep_path}")

    # Export to SQLite
    print(f"    [2/3] Exporting to SQLite...")
    sqlite_path = export_sqlite(rep_path)

    # Parse
    print(f"    [3/3] Parsing kernel data...")
    nvtx_filter = "decode_profile" if mode == "vllm" else None
    data = parse_kernels(sqlite_path, NUM_DECODE, nvtx_filter=nvtx_filter)
    data["mode"] = mode
    data["seq_len"] = seq_len

    # Keep top 30 kernels for .prof generation
    data["top_kernels"] = data.pop("per_kernel")[:30]

    gpu_ms = data["total_kernel_ns"] / 1e6 / NUM_DECODE
    launches = data["total_launches"] / NUM_DECODE
    print(f"    GPU kernel time: {gpu_ms:.2f} ms/step, {launches:.0f} launches/step")

    # Clean up intermediate files
    driver_path.unlink(missing_ok=True)
    Path(rep_path).unlink(missing_ok=True)
    Path(sqlite_path).unlink(missing_ok=True)

    return data


# =====================================================================
#  .prof file generation (per sequence length, side-by-side comparison)
# =====================================================================

def _fmt_time(us):
    """Format microseconds as 'Xus' or 'X.XXms'."""
    if us < 1000:
        return f"{us:.0f}us"
    return f"{us/1000:.2f}ms"


def _fmt_delta(us):
    """Format delta microseconds as '+Xus' or '+X.XXms'."""
    if abs(us) < 1000:
        return f"{us:+.0f}us"
    return f"{us/1000:+.2f}ms"


def _write_engine_section(lines, label, data, w):
    """Append one engine's kernel breakdown to lines."""
    n = data["num_steps"]
    gpu_ms = data["total_kernel_ns"] / 1e6 / n
    launches = data["total_launches"] / n

    lines.append(f"  {label}")
    lines.append(f"  GPU kernel time / step: {gpu_ms:.2f}ms")
    lines.append(f"  Kernel launches / step: {launches:.0f}")
    lines.append("")

    lines.append(f"  {'Category':<40s}  {'Time/step':>10s}  {'% GPU':>7s}  {'Launches':>9s}")
    lines.append(f"  {'-' * 70}")

    cats = data["categories"]
    total_ns = data["total_kernel_ns"]
    rows = []
    for cat, info in cats.items():
        us = info["total_ns"] / 1e3 / n
        pct = info["total_ns"] / total_ns * 100 if total_ns else 0
        l = info["launches"] / n
        rows.append((cat, us, pct, l))
    rows.sort(key=lambda x: -x[1])

    for cat, us, pct, l in rows:
        lines.append(f"  {cat:<40s}  {_fmt_time(us):>10s}  {pct:>6.1f}%  {l:>8.0f}/step")
    lines.append(f"  {'-' * 70}")
    lines.append(f"  {'TOTAL':<40s}  {_fmt_time(gpu_ms * 1000):>10s}  {'100.0':>6s}%  {launches:>8.0f}/step")
    lines.append("")

    # Top 20 kernels
    top = sorted(data.get("top_kernels", []), key=lambda x: -x["total_ns"])[:20]
    lines.append(f"  Top 20 individual kernels:")
    lines.append(f"    {'#':>3s}  {'Kernel':<55s}  {'Total':>9s}  {'Count':>6s}  {'Avg':>8s}  Category")
    lines.append(f"  {'-' * (w - 4)}")
    for i, k in enumerate(top, 1):
        name = k["name"][:53] + ".." if len(k["name"]) > 55 else k["name"]
        total_ms = k["total_ns"] / 1e6
        avg_us = k["avg_ns"] / 1e3
        lines.append(f"    {i:>3d}  {name:<55s}  {total_ms:>8.2f}ms  {k['count']:>6d}  "
                      f"{avg_us:>7.1f}us  {k['category']}")
    lines.append("")


def write_comparison_prof(seq_len, custom_data, vllm_data, model_dir=DEFAULT_MODEL_DIR):
    """Write a seq{N}.prof file comparing both engines at one seq_len."""
    w = 100
    lines = []
    n_steps = custom_data["num_steps"] if custom_data else vllm_data["num_steps"]

    model_name = Path(model_dir).name

    lines.append("=" * w)
    lines.append(f"  {model_name} Decode Kernel Profile — seq_len={seq_len}, batch=1, {n_steps} decode steps")
    lines.append(f"  Custom CUDA Graph vs vLLM SOTA (CUDA graphs + torch.compile)")
    lines.append("=" * w)
    lines.append("")

    # ── Comparison summary ──
    if custom_data and vllm_data:
        c_ms = custom_data["total_kernel_ns"] / 1e6 / custom_data["num_steps"]
        v_ms = vllm_data["total_kernel_ns"] / 1e6 / vllm_data["num_steps"]
        c_l = custom_data["total_launches"] / custom_data["num_steps"]
        v_l = vllm_data["total_launches"] / vllm_data["num_steps"]
        gap_ms = c_ms - v_ms
        gap_pct = gap_ms / v_ms * 100

        lines.append("COMPARISON SUMMARY")
        lines.append(f"  {'Metric':<35s}  {'Custom':>10s}  {'vLLM':>10s}  {'Delta':>12s}")
        lines.append(f"  {'-' * 72}")
        lines.append(f"  {'GPU kernel time / step':<35s}  {c_ms:>9.2f}ms  {v_ms:>9.2f}ms  "
                      f"{_fmt_delta(gap_ms * 1000):>10s} ({gap_pct:+.1f}%)")
        lines.append(f"  {'Kernel launches / step':<35s}  {c_l:>10.0f}  {v_l:>10.0f}  "
                      f"{c_l - v_l:>+10.0f}")
        lines.append("")

        # Category comparison
        all_cats = set()
        all_cats.update(custom_data["categories"].keys())
        all_cats.update(vllm_data["categories"].keys())

        cn = custom_data["num_steps"]
        vn = vllm_data["num_steps"]
        rows = []
        for cat in all_cats:
            c_us = custom_data["categories"].get(cat, {}).get("total_ns", 0) / 1e3 / cn
            v_us = vllm_data["categories"].get(cat, {}).get("total_ns", 0) / 1e3 / vn
            rows.append((cat, c_us, v_us, c_us - v_us))
        rows.sort(key=lambda x: -abs(x[3]))  # sort by largest delta

        lines.append("KERNEL CATEGORY COMPARISON (per decode step)")
        lines.append(f"  {'Category':<35s}  {'Custom':>10s}  {'vLLM':>10s}  {'Delta':>10s}")
        lines.append(f"  {'-' * 70}")
        for cat, c_us, v_us, delta in rows:
            lines.append(f"  {cat:<35s}  {_fmt_time(c_us):>10s}  {_fmt_time(v_us):>10s}  "
                          f"{_fmt_delta(delta):>10s}")
        lines.append(f"  {'-' * 70}")
        c_total_us = custom_data["total_kernel_ns"] / 1e3 / cn
        v_total_us = vllm_data["total_kernel_ns"] / 1e3 / vn
        lines.append(f"  {'TOTAL':<35s}  {_fmt_time(c_total_us):>10s}  {_fmt_time(v_total_us):>10s}  "
                      f"{_fmt_delta(c_total_us - v_total_us):>10s}")
        lines.append("")

    # ── Individual engine breakdowns ──
    if custom_data:
        lines.append("=" * w)
        _write_engine_section(lines, "CUSTOM ENGINE — CUDA Graph Decode", custom_data, w)

    if vllm_data:
        lines.append("=" * w)
        _write_engine_section(lines, "vLLM SOTA — CUDA Graphs + torch.compile", vllm_data, w)

    lines.append("=" * w)

    prof_dir = _profile_dir(model_dir)
    prof_dir.mkdir(parents=True, exist_ok=True)
    prof_path = prof_dir / f"seq{seq_len}.prof"
    prof_path.write_text("\n".join(lines) + "\n")
    print(f"    Written: {prof_path}")


def generate_all_prof_files(results, model_dir=DEFAULT_MODEL_DIR):
    """Generate .prof files for all profiled sequence lengths."""
    seq_lens_seen = set()
    for (mode, sl) in results:
        seq_lens_seen.add(sl)

    for sl in sorted(seq_lens_seen):
        c = results.get(("custom", sl))
        v = results.get(("vllm", sl))
        if c or v:
            write_comparison_prof(sl, c, v, model_dir=model_dir)


# =====================================================================
#  Summary / comparison
# =====================================================================

def _key(data):
    return (data["mode"], data["seq_len"])


def print_comparison(results):
    """Print side-by-side comparison table."""
    w = 100
    print()
    print("=" * w)
    print("  NSIGHT SYSTEMS: CUSTOM CUDA GRAPH vs vLLM SOTA — PER SEQUENCE LENGTH")
    print("=" * w)

    # Overview table
    print(f"\n{'seq_len':>8s}  {'Custom GPU ms':>14s}  {'vLLM GPU ms':>12s}  "
          f"{'Gap ms':>8s}  {'Gap %':>7s}  {'Custom launches':>16s}  {'vLLM launches':>14s}")
    print("-" * w)

    for sl in SEQ_LENS:
        c = results.get(("custom", sl))
        v = results.get(("vllm", sl))
        c_ms = c["total_kernel_ns"] / 1e6 / c["num_steps"] if c else float('nan')
        v_ms = v["total_kernel_ns"] / 1e6 / v["num_steps"] if v else float('nan')
        gap = c_ms - v_ms
        gap_pct = gap / v_ms * 100 if v else float('nan')
        c_l = c["total_launches"] / c["num_steps"] if c else float('nan')
        v_l = v["total_launches"] / v["num_steps"] if v else float('nan')
        print(f"{sl:>8d}  {c_ms:>14.2f}  {v_ms:>12.2f}  "
              f"{gap:>+8.2f}  {gap_pct:>+6.1f}%  {c_l:>16.0f}  {v_l:>14.0f}")

    # Per-category comparison at each seq_len
    for sl in SEQ_LENS:
        c = results.get(("custom", sl))
        v = results.get(("vllm", sl))
        if not c or not v:
            continue

        print(f"\n{'='*w}")
        print(f"  seq_len={sl}: KERNEL CATEGORY BREAKDOWN (per decode step)")
        print(f"{'='*w}")

        # Merge categories from both
        all_cats = set()
        if c: all_cats.update(c["categories"].keys())
        if v: all_cats.update(v["categories"].keys())

        rows = []
        for cat in all_cats:
            c_ns = c["categories"].get(cat, {}).get("total_ns", 0) if c else 0
            v_ns = v["categories"].get(cat, {}).get("total_ns", 0) if v else 0
            c_us = c_ns / 1e3 / c["num_steps"] if c else 0
            v_us = v_ns / 1e3 / v["num_steps"] if v else 0
            rows.append((cat, c_us, v_us, c_us - v_us))

        rows.sort(key=lambda x: -max(x[1], x[2]))

        print(f"  {'Category':<35s}  {'Custom':>10s}  {'vLLM':>10s}  {'Delta':>10s}")
        print(f"  {'-'*70}")
        c_total = c["total_kernel_ns"] / 1e3 / c["num_steps"]
        v_total = v["total_kernel_ns"] / 1e3 / v["num_steps"]
        for cat, c_us, v_us, delta in rows:
            c_str = f"{c_us:.0f}us" if c_us < 1000 else f"{c_us/1000:.2f}ms"
            v_str = f"{v_us:.0f}us" if v_us < 1000 else f"{v_us/1000:.2f}ms"
            d_str = f"{delta:+.0f}us" if abs(delta) < 1000 else f"{delta/1000:+.2f}ms"
            print(f"  {cat:<35s}  {c_str:>10s}  {v_str:>10s}  {d_str:>10s}")
        print(f"  {'-'*70}")
        c_total_str = f"{c_total/1000:.2f}ms"
        v_total_str = f"{v_total/1000:.2f}ms"
        d_total = (c_total - v_total) / 1000
        print(f"  {'TOTAL':<35s}  {c_total_str:>10s}  {v_total_str:>10s}  {d_total:+.2f}ms")

    # Top kernel differences at seq_len=128 (most informative for non-attention gap)
    print(f"\n{'='*w}")
    print(f"  TOP 15 KERNELS — Custom CUDA Graph (seq_len=128)")
    print(f"{'='*w}")
    c128 = results.get(("custom", 128))
    if c128:
        top = sorted(c128["top_kernels"], key=lambda x: -x["total_ns"])[:15]
        print(f"  {'#':>3s}  {'Kernel':<50s}  {'Total':>9s}  {'Count':>6s}  {'Avg':>8s}  {'Category'}")
        print(f"  {'-'*95}")
        for i, k in enumerate(top, 1):
            name = k["name"][:48] + ".." if len(k["name"]) > 50 else k["name"]
            total_ms = k["total_ns"] / 1e6
            avg_us = k["avg_ns"] / 1e3
            print(f"  {i:>3d}  {name:<50s}  {total_ms:>8.2f}ms  {k['count']:>6d}  "
                  f"{avg_us:>7.1f}us  {k['category']}")

    print(f"\n{'='*w}")
    print(f"  TOP 15 KERNELS — vLLM SOTA (seq_len=128)")
    print(f"{'='*w}")
    v128 = results.get(("vllm", 128))
    if v128:
        top = sorted(v128["top_kernels"], key=lambda x: -x["total_ns"])[:15]
        print(f"  {'#':>3s}  {'Kernel':<50s}  {'Total':>9s}  {'Count':>6s}  {'Avg':>8s}  {'Category'}")
        print(f"  {'-'*95}")
        for i, k in enumerate(top, 1):
            name = k["name"][:48] + ".." if len(k["name"]) > 50 else k["name"]
            total_ms = k["total_ns"] / 1e6
            avg_us = k["avg_ns"] / 1e3
            print(f"  {i:>3d}  {name:<50s}  {total_ms:>8.2f}ms  {k['count']:>6d}  "
                  f"{avg_us:>7.1f}us  {k['category']}")


# =====================================================================
#  Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nsight Systems profiling: Custom vs vLLM")
    parser.add_argument("command", choices=["custom", "vllm", "all"],
                        help="Which engine(s) to profile")
    parser.add_argument("seq_len", nargs="?", type=int, default=None,
                        help="Sequence length (required for custom/vllm)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path to HuggingFace model directory")
    args = parser.parse_args()

    if args.command == "all":
        results = {}
        for sl in SEQ_LENS:
            data = profile_one("custom", sl, args.model)
            results[_key(data)] = data
        for sl in SEQ_LENS:
            data = profile_one("vllm", sl, args.model)
            results[_key(data)] = data
        print_comparison(results)
        generate_all_prof_files(results, model_dir=args.model)
        return

    if args.command in ("custom", "vllm"):
        if args.seq_len is None:
            parser.error(f"seq_len is required for '{args.command}' command")
        data = profile_one(args.command, args.seq_len, args.model)
        c = data if args.command == "custom" else None
        v = data if args.command == "vllm" else None
        write_comparison_prof(args.seq_len, c, v, model_dir=args.model)
        return


if __name__ == "__main__":
    main()
