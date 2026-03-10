#!/usr/bin/env bash
# Phase 1: GPU-based batched trace collection for Mixtral-8x7B.
# Collects expert traces with continuous batching, one run per cache fraction.
# Each fraction constrains KV budget to match single-GPU replay memory.
#
# Usage: bash scripts/01_collect_traces.sh [NUM_CONVERSATIONS]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

MODEL=models/Mixtral-8x7B
DATASET=datasets/ShareGPT_Vicuna/ShareGPT_V3_unfiltered_cleaned_split.json
OUTPUT_BASE=datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b
NUM_CONVERSATIONS=${1:-200}
MAX_OUTPUT_TOKENS=4096
MAX_SEQS=32

# Detect GPUs for pipeline parallelism
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi
echo "Detected $NUM_GPUS GPU(s), using PP=$NUM_GPUS"

CACHE_FRACTIONS="0.6 0.7 0.8"

for frac in $CACHE_FRACTIONS; do
    pct=$(python3 -c "print(int($frac * 100))")
    out_dir="$OUTPUT_BASE/cache${pct}pct"
    echo ""
    echo "=== Collecting traces for cache ${pct}% (fraction=$frac) ==="
    python3 trace_construction/collect_batched_traces.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output-dir "$out_dir" \
        --cache-fraction "$frac" \
        --num-conversations "$NUM_CONVERSATIONS" \
        --max-output-tokens "$MAX_OUTPUT_TOKENS" \
        --max-seqs "$MAX_SEQS" \
        --pp "$NUM_GPUS" \
        --resume
done

echo ""
echo "Phase 1 complete. Traces in $OUTPUT_BASE/cache*pct/"
