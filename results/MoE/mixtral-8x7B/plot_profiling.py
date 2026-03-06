#!/usr/bin/env python3
"""Plot per-layer compute time vs expert PCIe fetch latency for Mixtral-8x7B-20L.

Reads data from profiling.md and generates a two-panel PDF:
  - Left:  Decode scaling (seq position 128..262K)
  - Right: Prefill scaling (seq length 128..2048)

Each panel shows:
  - Total layer compute time
  - Compute without MoE (total - stage4b)
  - Expert fetch latency (red dotted horizontal line)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12})
import numpy as np
from pathlib import Path

# ── Data from profiling.md (per-layer averages, microseconds) ──

# Decode: 1 token at given sequence position
decode_seq_pos = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
decode_total   = [472.6, 473.6, 474.0, 474.3, 476.2, 477.4, 484.4, 498.3, 522.2, 571.4, 665.1, 846.0]
decode_4b      = [327.1, 327.6, 327.5, 327.5, 327.6, 327.6, 327.7, 327.9, 327.9, 327.7, 327.8, 327.5]

# Prefill: 1 sequence of given length
prefill_seq_len = [128, 256, 512, 1024, 2048]
prefill_total   = [1296, 1447, 1498, 2160, 3716]
prefill_4b      = [1126, 1252, 1252, 1736, 2919]

# Expert PCIe fetch: 336 MB BF16, 6314 us at 52 GB/s
expert_fetch_us = 6314

# ── Derived ──

decode_no_moe  = [t - s4b for t, s4b in zip(decode_total, decode_4b)]
prefill_no_moe = [t - s4b for t, s4b in zip(prefill_total, prefill_4b)]

# ── Plot ──

fig, (ax_dec, ax_pre) = plt.subplots(1, 2, figsize=(12, 5))

expert_fetch_label = f"Expert fetch over PCIe ({expert_fetch_us:,} us)"

# --- Decode panel ---
ax_dec.plot(decode_seq_pos, decode_total, "o-", color="tab:blue", label="Total layer compute")
ax_dec.plot(decode_seq_pos, decode_no_moe, "s-", color="tab:green", label="Without MoE kernel")
ax_dec.axhline(expert_fetch_us, color="red", linestyle=":", linewidth=1.5, label=expert_fetch_label)
ax_dec.set_xscale("log", base=2)
ax_dec.set_xlabel("Sequence position (tokens)")
ax_dec.set_ylabel("Per-layer time (us)")
ax_dec.set_title("Decode: 1 token")
ax_dec.legend(fontsize=13, loc="center left")
ax_dec.grid(True, alpha=0.3)
ax_dec.set_xticks(decode_seq_pos)
ax_dec.set_xticklabels([f"{x//1024}K" if x >= 1024 else str(x) for x in decode_seq_pos],
                       rotation=45, fontsize=9)

# --- Prefill panel ---
ax_pre.plot(prefill_seq_len, prefill_total, "o-", color="tab:blue")
ax_pre.plot(prefill_seq_len, prefill_no_moe, "s-", color="tab:green")
ax_pre.axhline(expert_fetch_us, color="red", linestyle=":", linewidth=1.5)
ax_pre.set_xscale("log", base=2)
ax_pre.set_xlabel("Sequence length (tokens)")
ax_pre.set_ylabel("Per-layer time (us)")
ax_pre.set_title("Prefill: 1 sequence")
ax_pre.grid(True, alpha=0.3)
ax_pre.set_xticks(prefill_seq_len)
ax_pre.set_xticklabels([str(x) for x in prefill_seq_len])

fig.suptitle("Mixtral-8x7B-20L: Layer Compute vs Expert Fetch Latency (H100, PCIe)", fontsize=14)
fig.tight_layout()

out_path = Path(__file__).resolve().parent / "profiling.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved {out_path}")
