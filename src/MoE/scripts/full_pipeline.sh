#!/usr/bin/env bash
# Full pipeline: collect traces → simulate policies → GPU replay.
#
# Usage:
#   bash scripts/full_pipeline.sh --cache-pct 70,80,90          # sweep multiple
#   bash scripts/full_pipeline.sh --cache-pct 95                # single point
#   bash scripts/full_pipeline.sh --cache-pct 70,80 --resume    # skip completed replay jobs
#
# Equivalent to running 01 → 02 → 03 for the given cache percents.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

# Parse args
CACHE_PCT_ARG=""
RESUME_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-pct) CACHE_PCT_ARG="$2"; shift 2 ;;
        --resume)    RESUME_FLAG="--resume"; shift ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$CACHE_PCT_ARG" ]; then
    echo "Usage: full_pipeline.sh --cache-pct <pct[,pct,...]> [--resume]"
    exit 1
fi

echo "=== Full pipeline for cache percents: ${CACHE_PCT_ARG} ==="

MODEL=${MODEL:-../../models/Mixtral-8x7B}

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
