#!/usr/bin/env bash
# Phase 1: GPU-based batched trace collection for Mixtral-8x7B.
# Collects expert traces with continuous batching, one run per cache fraction.
# Each fraction constrains KV budget to match single-GPU replay memory.
#
# Auto-detects GPU configuration:
#   - Multi-GPU (H100):  PP=NUM_GPUS, cache fractions 0.6/0.7/0.8
#   - Single large GPU (GH200): experts-per-layer offloading, fractions 0.7/0.8/0.9
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
# Single large GPU (e.g. GH200 96GB): use expert offloading for collection
# Multi-GPU (e.g. 2x H100 80GB): use pipeline parallelism
if [ "$NUM_GPUS" -eq 1 ] && [ "$GPU_MEM_GB" -ge 90 ]; then
    MODE="offload"
    EPL=5  # experts_per_layer: keeps ~57GB of experts on GPU, leaves room for KV
    CACHE_FRACTIONS="0.7 0.8 0.9"
    echo "Mode: single-GPU offloading (epl=$EPL), fractions: $CACHE_FRACTIONS"
else
    MODE="pp"
    CACHE_FRACTIONS="0.6 0.7 0.8"
    echo "Mode: pipeline parallel (PP=$NUM_GPUS), fractions: $CACHE_FRACTIONS"
fi

for frac in $CACHE_FRACTIONS; do
    pct=$(python3 -c "print(int($frac * 100))")
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
        COLLECT_ARGS+=(--experts-per-layer "$EPL")
    else
        COLLECT_ARGS+=(--pp "$NUM_GPUS")
    fi

    python3 trace_construction/collect_batched_traces.py "${COLLECT_ARGS[@]}"
done

echo ""
echo "Phase 1 complete. Traces in $OUTPUT_BASE/cache*pct/"
