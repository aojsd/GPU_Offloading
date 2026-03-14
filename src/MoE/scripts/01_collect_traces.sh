#!/usr/bin/env bash
# Phase 1: GPU-based batched trace collection for Mixtral-8x7B.
# Collects expert traces with continuous batching, one run per cache fraction.
# Each fraction constrains KV budget to match single-GPU replay memory.
#
# Auto-detects GPU configuration:
#   - Multi-GPU (H100):  PP=NUM_GPUS, default cache percents 60,70,80
#   - Single large GPU (GH200): experts-per-layer offloading, default 70,80,90
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
# Single large GPU (e.g. GH200 96GB): use expert offloading for collection
# Multi-GPU (e.g. 2x H100 80GB): use pipeline parallelism
if [ "$NUM_GPUS" -eq 1 ] && [ "$GPU_MEM_GB" -ge 90 ]; then
    MODE="offload"
    # experts_per_layer ≈ 60% of total experts (enough for collection, leaves room for KV)
    EPL=$(python3 -c "
import json, os
cfg = json.load(open(os.path.join('$MODEL', 'config.json')))
n_exp = cfg.get('n_routed_experts') or cfg.get('num_local_experts', 8)
print(round(n_exp * 0.6))
")
    echo "experts_per_layer = $EPL (60% of total)"
    DEFAULT_CACHE_PCTS="70,80,90"
else
    MODE="pp"
    DEFAULT_CACHE_PCTS="60,70,80"
fi

# Use user-specified cache percents or defaults
CACHE_PCTS="${USER_CACHE_PCTS:-$DEFAULT_CACHE_PCTS}"
# Convert comma-separated percents to space-separated fractions
CACHE_FRACTIONS=$(python3 -c "print(' '.join(str(float(p)/100) for p in '${CACHE_PCTS}'.split(',')))")
echo "Mode: ${MODE}, cache percents: ${CACHE_PCTS}, fractions: ${CACHE_FRACTIONS}"

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
        COLLECT_ARGS+=(--experts-per-layer "$EPL")
    else
        COLLECT_ARGS+=(--pp "$NUM_GPUS")
    fi

    python3 trace_construction/collect_batched_traces.py "${COLLECT_ARGS[@]}"
done

echo ""
echo "Phase 1 complete. Traces in $OUTPUT_BASE/cache*pct/"
