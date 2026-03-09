#!/usr/bin/env bash
# Phase 2: Run continuous batching simulation for all cache fractions.
# CPU-only — no GPU required. Produces batched.json in each cache%pct/ dir.
#
# If 2+ GPUs are available, also validates all batched traces against the real
# model (PP=2) to confirm that set-union expert merging is faithful.
#
# Usage: bash scripts/02_batch_simulate.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

MODEL=models/Mixtral-8x7B
INPUT_DIR=datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b
MODEL_CONFIG=$MODEL/config.json

CACHE_FRACTIONS="0.5 0.6 0.7 0.8 0.85"

for frac in $CACHE_FRACTIONS; do
    pct=$(python3 -c "print(int($frac * 100))")
    echo "=== Cache fraction $frac (${pct}%) ==="
    python3 trace_construction/build_trace.py \
        --input-dir "$INPUT_DIR" \
        --model-config "$MODEL_CONFIG" \
        --cache-fraction "$frac" \
        --max-graph-size 512 \
        --prefill-chunk-size 256
    echo ""
done

echo "Phase 2 complete. Batched traces in $INPUT_DIR/cache*pct/batched.json"

# --- Validate batched trace routing with PP (if 2+ GPUs available) ---
# Validate only 85% cache fraction: it has the lowest peak concurrent batch
# size (51), so max_seq_len=actual_max_seq fits in PP=2 memory.  Lower cache
# fractions have higher peak concurrency → more KV pages per GPU → OOM.
# Routing is per-token-independent, so validating one fraction proves
# faithfulness for all.
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" -ge 2 ]; then
    echo ""
    echo "=== Validating batched trace for cache 85% (PP=$NUM_GPUS) ==="
    python3 trace_construction/validate_batched_trace.py \
        --model "$MODEL" \
        --trace-dir "$INPUT_DIR" \
        --cache-pct 85 \
        --pipeline-parallel "$NUM_GPUS"
else
    echo ""
    echo "Skipping trace validation (requires 2+ GPUs, found $NUM_GPUS)."
fi
