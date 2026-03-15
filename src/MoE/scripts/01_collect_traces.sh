#!/usr/bin/env bash
# Phase 1: GPU-based batched trace collection.
# Collects expert traces with continuous batching, one run per cache fraction.
#
# ── Relationship between cache%, KV budget, and EPL ──────────────────────
#
#   cache_pct  ──►  expert_cache_size  ──►  KV budget  ──►  optimal EPL
#              │                        │               │
#              │  cache_pct determines  │  GPU memory   │  EPL is the largest
#              │  how many expert slots  │  remaining    │  value such that the
#              │  reside on GPU during   │  after expert │  EPL-based buffer
#              │  Phase 3 replay.       │  cache + non- │  (L*EPL + scratchpad)
#              │                        │  expert model │  + the KV budget still
#              │                        │  = KV budget. │  fits on GPU.
#
#   The KV budget controls batching and preemption in the scheduler.
#   Trace collection (Phase 1) must use the SAME KV budget as replay (Phase 3)
#   so that batch compositions are identical. EPL only affects the physical
#   expert buffer during collection — it must be large enough for inference but
#   small enough that the buffer + KV budget fit in GPU memory together.
#
# ── GPU configuration ────────────────────────────────────────────────────
#   - Single large GPU (GH200): experts-per-layer offloading, default 70-97.5%
#   - Multi-GPU (H100):  PP=NUM_GPUS, default cache percents 60,70,80
#
# Usage:
#   bash scripts/01_collect_traces.sh                          # auto-detect defaults
#   bash scripts/01_collect_traces.sh --cache-pct 70,80,90     # explicit sweep
#   bash scripts/01_collect_traces.sh --cache-pct 85 -n 100    # single point, 100 convos
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

MODEL=${MODEL:-../../models/Mixtral-8x7B}
DATASET=../../datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json
NUM_CONVERSATIONS=200
MAX_OUTPUT_TOKENS=4096
MAX_SEQS=32

# Parse args
USER_CACHE_PCTS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-pct) USER_CACHE_PCTS="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        -n)          NUM_CONVERSATIONS="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done
# Re-derive MODEL_TAG after arg parsing (--model may have changed it)
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
OUTPUT_BASE=../../datasets/ShareGPT_Vicuna/expert_traces/${MODEL_TAG}

# Detect GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi

# Auto-detect GPU memory
GPU_MEM_GB=$(python3 -c "
import subprocess
out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True)
mem_mib = int(out.strip().split('\n')[0])
print(mem_mib // 1024)
" 2>/dev/null || echo "80")
echo "GPU memory: ${GPU_MEM_GB} GB, ${NUM_GPUS} GPU(s)"

# Select mode based on GPU configuration
if [ "$NUM_GPUS" -eq 1 ] && [ "$GPU_MEM_GB" -ge 90 ]; then
    MODE="offload"
    DEFAULT_CACHE_PCTS="70,80,90"
else
    MODE="pp"
    DEFAULT_CACHE_PCTS="60,70,80"
fi

# Use user-specified cache percents or defaults
CACHE_PCTS="${USER_CACHE_PCTS:-$DEFAULT_CACHE_PCTS}"
# Convert comma-separated percents to space-separated fractions
CACHE_FRACTIONS=$(python3 -c "print(' '.join(str(float(p)/100) for p in '${CACHE_PCTS}'.split(',')))")
echo "Mode: ${MODE}, cache percents: ${CACHE_PCTS}"

for frac in $CACHE_FRACTIONS; do
    pct=$(python3 -c "v=$frac*100; print(int(v) if v==int(v) else f'{v:g}')")
    out_dir="$OUTPUT_BASE/cache${pct}pct"
    echo ""
    echo "=== Collecting traces for cache ${pct}% (fraction=$frac) ==="

    COLLECT_ARGS=(
        --model "$MODEL"
        --dataset "$DATASET"
        --output-dir "$out_dir"
        --cache-fraction "$frac"
        --num-conversations "$NUM_CONVERSATIONS"
        --max-output-tokens "$MAX_OUTPUT_TOKENS"
        --max-seqs "$MAX_SEQS"
        --gpu-memory-gb "$GPU_MEM_GB"
        --resume
    )
    if [ "$MODE" = "offload" ]; then
        # Compute optimal EPL for this cache fraction.
        # Flow: cache_pct → KV budget (compute_replay_kv_budget) → optimal EPL.
        # See collect_batched_traces.py:compute_optimal_epl for details.
        EPL=$(python3 -c "
import json, os, sys
sys.path.insert(0, '.')
from trace_construction.collect_batched_traces import (
    compute_replay_kv_budget, compute_optimal_epl)
cfg = json.load(open(os.path.join('$MODEL', 'config.json')))
page_size = 64 if cfg.get('kv_lora_rank') is not None else 16
mem = compute_replay_kv_budget(
    os.path.join('$MODEL', 'config.json'),
    cache_fraction=$frac, page_size=page_size,
    gpu_memory_gb=$GPU_MEM_GB,
)
print(compute_optimal_epl(mem))
")
        echo "  cache ${pct}%: experts_per_layer=$EPL (derived from KV budget)"
        COLLECT_ARGS+=(--experts-per-layer "$EPL")
    else
        COLLECT_ARGS+=(--pp "$NUM_GPUS")
    fi

    python3 trace_construction/collect_batched_traces.py "${COLLECT_ARGS[@]}"
done

echo ""
echo "Phase 1 complete. Traces in $OUTPUT_BASE/cache*pct/"
