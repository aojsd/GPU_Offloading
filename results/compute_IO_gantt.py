#!/usr/bin/env python3
"""Gantt-style comparison of per-layer compute vs expert I/O for Mixtral-8x7B.

Three cases:
  1. Demand load (sync): stages 1–4a, then blocking I/O, then stage4b.
  2. Start-of-layer prefetch: I/O starts at stage1, overlaps 1–4a, 4b waits.
  3. 1-layer-ahead prefetch: I/O overlaps prior layer's stage4b; next
     layer's stage4b waits.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-layer compute breakdown (ms) — 80% cache, No-Prefetch average
# ---------------------------------------------------------------------------
STAGE1    = 0.085   # QKV projections
ATTENTION = 0.044   # self-attention
STAGE4A   = 0.044   # expert routing
STAGE4B   = 0.976   # MoE kernel
EXPERT_IO = 6.0     # 336 MB / 56 GB/s

COLORS = {
    "stage1":    "#4C72B0",
    "attention": "#55A868",
    "stage4a":   "#C44E52",
    "stage4b":   "#8172B2",
    "io":        "#CCB974",
    "idle":      "#D3D3D3",
}

LABELS = {
    "stage1":    "QKV Projections",
    "attention": "Attention",
    "stage4a":   "Expert Routing",
    "stage4b":   "MoE Kernel",
    "io":        "Expert I/O (PCIe)",
    "idle":      "Idle (stalled on I/O)",
}

BAR_HEIGHT = 0.6
_STALL_BBOX = dict(boxstyle="round,pad=0.15", fc="#D3D3D3", ec="none", alpha=0.85)

# ---------------------------------------------------------------------------
# Layout parameters (tweak these to adjust spacing)
# ---------------------------------------------------------------------------
SUBPLOT_HSPACE  = 0.4    # vertical space between subplots
TITLE_Y         = .99    # main title vertical position (figure coords)
LEGEND_Y        = -0.025  # legend vertical position (figure coords)


# ---------------------------------------------------------------------------
def _bar(ax, row, start, dur, color, hatch=None):
    ax.broken_barh(
        [(start, dur)], (row - BAR_HEIGHT / 2, BAR_HEIGHT),
        facecolors=color, edgecolors="black", linewidth=0.7, hatch=hatch,
    )


def _time_label(ax, row, start, dur, text, fontsize=7, min_width=0.25,
                color="white", bbox=None):
    """Place a centred label inside a bar; skip if bar is too narrow."""
    if dur < min_width:
        return
    ax.text(
        start + dur / 2, row, text,
        ha="center", va="center", fontsize=fontsize, color=color,
        fontweight="bold", zorder=5, bbox=bbox,
    )


def _draw_stages_1_to_4a(ax, row, t0):
    """Draw stage1, attention, stage4a sequentially. Returns time after 4a."""
    t = t0
    for stage, dur in [("stage1", STAGE1), ("attention", ATTENTION),
                       ("stage4a", STAGE4A)]:
        _bar(ax, row, t, dur, COLORS[stage])
        t += dur
    return t


# ---------------------------------------------------------------------------
def case1_demand(ax):
    """Case 1: Synchronous demand load — no overlap between I/O and compute."""
    C, IO = 1, 0

    # Compute stages 1–4a
    t = 0.0
    t = _draw_stages_1_to_4a(ax, C, t)

    # Synchronous I/O — compute stalls, I/O active
    io_start = t
    _bar(ax, IO, io_start, EXPERT_IO, COLORS["io"])
    _time_label(ax, IO, io_start, EXPERT_IO, f"{EXPERT_IO:.1f} ms", fontsize=8)

    _bar(ax, C, t, EXPERT_IO, COLORS["idle"], hatch="///")
    _time_label(ax, C, t, EXPERT_IO, f"stall {EXPERT_IO:.1f} ms", fontsize=8,
                color="#333333", bbox=_STALL_BBOX)
    t += EXPERT_IO

    # stage4b
    _bar(ax, C, t, STAGE4B, COLORS["stage4b"])
    _time_label(ax, C, t, STAGE4B, f"{STAGE4B:.2f}", fontsize=7)
    t += STAGE4B

    ax.set_title("Case 1 — Demand Load (single layer)", fontweight="bold")
    _format(ax, t, is_last=False)
    return t


# ---------------------------------------------------------------------------
def case2_start_of_layer(ax):
    """Case 2: I/O starts at beginning of stage1; stage4b waits for I/O."""
    C, IO = 1, 0

    # I/O starts at t=0
    _bar(ax, IO, 0, EXPERT_IO, COLORS["io"])
    _time_label(ax, IO, 0, EXPERT_IO, f"{EXPERT_IO:.1f} ms", fontsize=8)

    # Compute stages 1–4a overlap with I/O
    t = 0.0
    t = _draw_stages_1_to_4a(ax, C, t)

    # Idle until I/O finishes
    idle = EXPERT_IO - t
    _bar(ax, C, t, idle, COLORS["idle"], hatch="///")
    _time_label(ax, C, t, idle, f"stall {idle:.2f} ms", fontsize=8,
                color="#333333", bbox=_STALL_BBOX)
    t = EXPERT_IO

    # stage4b
    _bar(ax, C, t, STAGE4B, COLORS["stage4b"])
    _time_label(ax, C, t, STAGE4B, f"{STAGE4B:.2f}", fontsize=7)
    t += STAGE4B

    ax.set_title("Case 2 — Start-of-Layer Prefetch (single layer)",
                 fontweight="bold")
    _format(ax, t, is_last=False)
    return t


# ---------------------------------------------------------------------------
def case3_one_layer_ahead(ax):
    """Case 3: I/O for layer N+1 overlaps with stage4b of layer N.

    X-axis is shifted so Layer N+1's stage1 starts at t=0.
    All drawing uses pre-shifted coordinates (no post-hoc patch shifting).
    """
    C, IO = 1, 0
    PRE = STAGE1 + ATTENTION + STAGE4A  # stages before MoE

    # Offset: Layer N+1's stage1 is at x=0, so Layer N's stage4b is at -STAGE4B,
    # and Layer N's prefetch I/O starts at -(STAGE4B) (issued after Layer N's 4a).
    # In absolute time: Layer N 4a ends at PRE, prefetch starts at PRE,
    # Layer N 4b runs PRE..PRE+STAGE4B, Layer N+1 stage1 at PRE+STAGE4B.
    # Offset = PRE + STAGE4B (so Layer N+1 stage1 maps to 0).
    off = PRE + STAGE4B

    # ── Layer N (only stage4b visible) ──
    n_4b_start = PRE - off          # negative
    _bar(ax, C, n_4b_start, STAGE4B, COLORS["stage4b"])
    _time_label(ax, C, n_4b_start, STAGE4B, f"{STAGE4B:.2f}", fontsize=7)

    # I/O: prefetch issued after Layer N's stage4a (absolute t=PRE)
    io_start = PRE - off            # negative
    io_end = io_start + EXPERT_IO
    _bar(ax, IO, io_start, EXPERT_IO, COLORS["io"])
    _time_label(ax, IO, io_start, EXPERT_IO, f"{EXPERT_IO:.1f} ms", fontsize=8)

    # ── Layer N+1 ──
    t = 0.0  # stage1 starts at x=0
    for stage, dur in [("stage1", STAGE1), ("attention", ATTENTION),
                       ("stage4a", STAGE4A)]:
        _bar(ax, C, t, dur, COLORS[stage])
        t += dur

    # Stall if I/O hasn't finished
    if io_end > t:
        idle = io_end - t
        _bar(ax, C, t, idle, COLORS["idle"], hatch="///")
        _time_label(ax, C, t, idle, f"stall {idle:.2f} ms",
                    fontsize=8, color="#333333", bbox=_STALL_BBOX)
        t = io_end

    _bar(ax, C, t, STAGE4B, COLORS["stage4b"])
    _time_label(ax, C, t, STAGE4B, f"{STAGE4B:.2f}", fontsize=7)
    t += STAGE4B

    # Layer boundary marker at x=0
    ax.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.text(n_4b_start + STAGE4B / 2, C + BAR_HEIGHT / 2 + 0.12,
            "Layer N", ha="center", fontsize=8, fontstyle="italic")
    ax.text(t / 2, C + BAR_HEIGHT / 2 + 0.12,
            "Layer N+1", ha="center", fontsize=8, fontstyle="italic")

    ax.set_title("Case 3 — Prefetch During MoE Kernel",
                 fontweight="bold")

    # Formatting: show Layer N's 4b in negative region, ticks only ≥ 0
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["PCIe I/O", "Compute"])
    ax.set_xlim(n_4b_start - 0.15, t + 0.8)
    ax.set_ylim(-0.8, 1.7)
    ax.set_xlabel("Time (ms)")  # case3 is always the last subplot
    ax.grid(axis="x", alpha=0.25)

    all_ticks = ax.get_xticks()
    ax.set_xticks([tk for tk in all_ticks if tk >= 0])

    ax.annotate(
        f"Per layer: {t:.2f} ms", xy=(t, 0.7),
        xytext=(t, -0.55), fontsize=9, fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )


# ---------------------------------------------------------------------------
def _format(ax, xmax, is_last=False):
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["PCIe I/O", "Compute"])
    ax.set_xlim(-0.15, xmax + 0.8)
    ax.set_ylim(-0.8, 1.7)
    if is_last:
        ax.set_xlabel("Time (ms)")
    ax.grid(axis="x", alpha=0.25)
    # total annotation — arrow points straight up from below
    ax.annotate(
        f"Total: {xmax:.2f} ms", xy=(xmax, 0.7),
        xytext=(xmax, -0.55), fontsize=9, fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )


# ---------------------------------------------------------------------------
def main():
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2))
    fig.subplots_adjust(top=0.92, bottom=0.12, hspace=SUBPLOT_HSPACE)

    case1_demand(axes[0])
    case2_start_of_layer(axes[1])
    case3_one_layer_ahead(axes[2])

    # shared legend — two rows at bottom center, with execution times
    handles = [
        mpatches.Patch(fc=COLORS["stage1"],    ec="black",
                       label=f"{LABELS['stage1']}  ({STAGE1*1000:.0f} \u00b5s)"),
        mpatches.Patch(fc=COLORS["attention"], ec="black",
                       label=f"{LABELS['attention']}  ({ATTENTION*1000:.0f} \u00b5s)"),
        mpatches.Patch(fc=COLORS["stage4a"],   ec="black",
                       label=f"{LABELS['stage4a']}  ({STAGE4A*1000:.0f} \u00b5s)"),
        mpatches.Patch(fc=COLORS["stage4b"],   ec="black",
                       label=f"{LABELS['stage4b']}  ({STAGE4B*1000:.0f} \u00b5s)"),
        mpatches.Patch(fc=COLORS["io"],        ec="black",
                       label=f"{LABELS['io']}  ({EXPERT_IO:.1f} ms)"),
        mpatches.Patch(fc=COLORS["idle"],       ec="black",
                       label=LABELS["idle"], hatch="///"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, LEGEND_Y), framealpha=0.9)

    fig.suptitle(
        "Mixtral-8x7B — Per-Layer Compute vs Expert I/O  (H100 PCIe 5.0, 56 GB/s)",
        fontsize=12, fontweight="bold", y=TITLE_Y,
    )

    out = Path(__file__).resolve().parent / "MoE" / "mixtral-8x7B" / "gantt.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
