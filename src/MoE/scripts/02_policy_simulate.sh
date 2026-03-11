#!/usr/bin/env bash
# Phase 2: Run all cache × prefetch policy simulations.
# CPU-only — no GPU required. Produces GPUReplayTrace files in each cache%pct/ dir.
# Runs all cache fractions in parallel (one process per cache%).
#
# Usage: bash scripts/02_policy_simulate.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

MODEL=${MODEL:-models/Mixtral-8x7B}
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
python3 scripts/run_all_policies.py --parallel --model "$MODEL"

echo ""
echo "Phase 2 complete. Replay traces in datasets/ShareGPT_Vicuna/expert_traces/${MODEL_TAG}/cache*pct/"
