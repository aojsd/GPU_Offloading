#!/usr/bin/env bash
# Phase 1: Collect per-conversation expert traces for Mixtral-8x7B (full 32L).
# Uses pipeline parallelism across all available GPUs.
#
# Usage: bash scripts/01_collect_traces.sh [NUM_CONVERSATIONS]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

MODEL=models/Mixtral-8x7B
DATASET=datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json
OUTPUT_DIR=datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b
NUM_CONVERSATIONS=${1:-200}
MAX_OUTPUT_TOKENS=4096

# Detect GPUs for pipeline parallelism
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi
echo "Detected $NUM_GPUS GPU(s), using PP=$NUM_GPUS"

python3 trace_construction/collect_traces.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --num-conversations "$NUM_CONVERSATIONS" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --pipeline-parallel "$NUM_GPUS"

echo ""
echo "Phase 1 complete. Traces in $OUTPUT_DIR/requests/"
