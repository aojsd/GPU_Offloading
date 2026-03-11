#!/usr/bin/env bash
# Full pipeline: collect traces → simulate policies → GPU replay for a single cache fraction.
#
# Usage:
#   bash scripts/full_pipeline.sh 0.95              # run 95% cache fraction
#   bash scripts/full_pipeline.sh 0.95 --resume     # skip completed replay jobs
#
# Equivalent to running 01 → 02 → 03 for a single cache%, useful for ad-hoc experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

CACHE_FRAC="${1:?Usage: full_pipeline.sh <cache_fraction> [--resume]}"
shift
RESUME_FLAG=""
for arg in "$@"; do
    [ "$arg" = "--resume" ] && RESUME_FLAG="--resume"
done

PCT=$(python3 -c "print(int($CACHE_FRAC * 100))")
echo "=== Full pipeline for cache ${PCT}% (fraction=$CACHE_FRAC) ==="

MODEL=${MODEL:-models/Mixtral-8x7B}
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
DATASET=datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json
OUTPUT_BASE=datasets/ShareGPT_Vicuna/expert_traces/${MODEL_TAG}
NUM_CONVERSATIONS=200
MAX_OUTPUT_TOKENS=4096
MAX_SEQS=32

# Detect GPU
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM_GB=$(python3 -c "
import subprocess
out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True)
mem_mib = int(out.strip().split('\n')[0])
print(mem_mib // 1024)
" 2>/dev/null || echo "80")
echo "GPU memory: ${GPU_MEM_GB} GB, ${NUM_GPUS} GPU(s)"

if [ "$NUM_GPUS" -eq 1 ] && [ "$GPU_MEM_GB" -ge 90 ]; then
    MODE="offload"
    EPL=5
else
    MODE="pp"
fi

# ── Phase 1: Collect traces ──────────────────────────────────────────────
OUT_DIR="$OUTPUT_BASE/cache${PCT}pct"
if [ -f "$OUT_DIR/batched_trace.json" ]; then
    echo ""
    echo "Phase 1: SKIPPED (trace already exists: $OUT_DIR/batched_trace.json)"
else
    echo ""
    echo "Phase 1: Collecting traces for cache ${PCT}%..."
    COLLECT_ARGS=(
        --model "$MODEL"
        --dataset "$DATASET"
        --output-dir "$OUT_DIR"
        --cache-fraction "$CACHE_FRAC"
        --num-conversations "$NUM_CONVERSATIONS"
        --max-output-tokens "$MAX_OUTPUT_TOKENS"
        --max-seqs "$MAX_SEQS"
        --gpu-memory-gb "$GPU_MEM_GB"
        --resume
    )
    if [ "$MODE" = "offload" ]; then
        COLLECT_ARGS+=(--experts-per-layer "$EPL")
    else
        COLLECT_ARGS+=(--pp "$NUM_GPUS")
    fi
    python3 trace_construction/collect_batched_traces.py "${COLLECT_ARGS[@]}"
    echo "Phase 1 complete."
fi

# ── Phase 2: Policy simulation ───────────────────────────────────────────
echo ""
echo "Phase 2: Simulating policies for cache ${PCT}%..."
python3 scripts/run_all_policies.py --cache-pct "$PCT" --model "$MODEL"
echo "Phase 2 complete."

# ── Phase 3: GPU replay ─────────────────────────────────────────────────
echo ""
echo "Phase 3: GPU replay for cache ${PCT}%..."
REPLAY_ARGS=(--cache-pct "$PCT")
[ -n "$RESUME_FLAG" ] && REPLAY_ARGS+=("$RESUME_FLAG")
bash scripts/03_gpu_replay.sh "${REPLAY_ARGS[@]}"
echo ""
echo "Full pipeline complete for cache ${PCT}%."
