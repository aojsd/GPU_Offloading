#!/usr/bin/env bash
# Phase 2: Run all cache × prefetch policy simulations.
# CPU-only — no GPU required. Produces GPUReplayTrace files in each cache%pct/ dir.
# Runs all cache fractions in parallel (one process per cache%).
#
# Usage:
#   bash scripts/02_policy_simulate.sh                        # all auto-detected cache%
#   bash scripts/02_policy_simulate.sh --cache-pct 70,80,90   # specific cache percents
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

MODEL=${MODEL:-../../models/Mixtral-8x7B}

# Parse args
CACHE_PCT_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-pct) CACHE_PCT_ARG="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')

SIM_ARGS=(--parallel --model "$MODEL")
if [ -n "$CACHE_PCT_ARG" ]; then
    SIM_ARGS+=(--cache-pct "$CACHE_PCT_ARG")
fi

python3 scripts/run_all_policies.py "${SIM_ARGS[@]}"

echo ""
echo "Phase 2 complete. Replay traces in ../../datasets/ShareGPT_Vicuna/expert_traces/${MODEL_TAG}/cache*pct/"
