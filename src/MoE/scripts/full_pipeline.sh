#!/usr/bin/env bash
# Full pipeline: collect traces → simulate policies → GPU replay.
#
# Usage:
#   bash scripts/full_pipeline.sh --cache-pct 70,80,90                          # sweep multiple
#   bash scripts/full_pipeline.sh --cache-pct 95 --model ../../models/MyModel   # custom model
#   bash scripts/full_pipeline.sh --cache-pct 70,80 --resume                    # skip completed
#
# Equivalent to running 01 → 02 → 03 for the given cache percents.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

# Parse args
CACHE_PCT_ARG=""
RESUME_FLAG=""
MODEL_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-pct) CACHE_PCT_ARG="$2"; shift 2 ;;
        --model)     MODEL_ARG="$2"; shift 2 ;;
        --resume)    RESUME_FLAG="--resume"; shift ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$CACHE_PCT_ARG" ]; then
    echo "Usage: full_pipeline.sh --cache-pct <pct[,pct,...]> [--model <path>] [--resume]"
    exit 1
fi

# Export MODEL so sub-scripts pick it up
export MODEL="${MODEL_ARG:-${MODEL:-../../models/Mixtral-8x7B}}"
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
echo "=== Full pipeline for ${MODEL_TAG}, cache percents: ${CACHE_PCT_ARG} ==="

# ── Phase 1: Collect traces ──────────────────────────────────────────────
echo ""
echo "Phase 1: Collecting traces..."
bash scripts/01_collect_traces.sh --cache-pct "$CACHE_PCT_ARG"
echo "Phase 1 complete."

# ── Phase 2: Policy simulation ───────────────────────────────────────────
echo ""
echo "Phase 2: Simulating policies..."
bash scripts/02_policy_simulate.sh --cache-pct "$CACHE_PCT_ARG"
echo "Phase 2 complete."

# ── Phase 3: GPU replay ─────────────────────────────────────────────────
echo ""
echo "Phase 3: GPU replay..."
REPLAY_ARGS=(--cache-pct "$CACHE_PCT_ARG")
[ -n "$RESUME_FLAG" ] && REPLAY_ARGS+=("$RESUME_FLAG")
bash scripts/03_gpu_replay.sh "${REPLAY_ARGS[@]}"
echo ""
echo "Full pipeline complete for cache percents: ${CACHE_PCT_ARG}."
