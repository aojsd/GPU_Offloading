#!/usr/bin/env bash
# Phase 3: GPU replay of all policy traces across all cache fractions.
# Auto-detects GPU count and distributes work across GPUs.
#
# Execution order: policies first (SF → Belady → LFU → LRU), then cache sizes
# within each policy group (80 → 70 → 60). LRU runs last since it has by far
# the most demand loads and takes the longest.
#
# Each (policy_group, cache%) job is dispatched to the next free GPU.
#
# Usage:
#   bash scripts/03_gpu_replay.sh                    # auto-detect GPUs
#   bash scripts/03_gpu_replay.sh --cache-pct 80     # single cache%
#   bash scripts/03_gpu_replay.sh --resume           # skip completed jobs
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

MODEL=models/Mixtral-8x7B
WARMUP=100
TRACE_BASE=datasets/ShareGPT_Vicuna/expert_traces/mixtral-8x7b

# Policy groups in priority order
POLICY_GROUPS=(
    "StaticFreq-None,StaticFreq-Oracle,StaticFreq-Oracle(1)"
    "Belady-None,Belady-Oracle,Belady-Oracle(1)"
    "LFU-None,LFU-Oracle,LFU-Oracle(1)"
    "LRU-None,LRU-Oracle,LRU-Oracle(1)"
)
GROUP_NAMES=("SF" "Belady" "LFU" "LRU")

# Cache percentages in priority order
CACHE_PCTS=(80 70 60)

# Results directory for resume checking
RESULTS_TMP="../../results/MoE/mixtral-8x7B/tmp"

# Parse args
SINGLE_PCT=""
RESUME=0
for arg in "$@"; do
    if [ "$arg" = "--cache-pct" ]; then
        shift_next=1
        continue
    fi
    if [ "$arg" = "--resume" ]; then
        RESUME=1
        continue
    fi
    if [ "${shift_next:-}" = "1" ]; then
        SINGLE_PCT="$arg"
        shift_next=0
    fi
done

# Detect GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi
echo "Detected $NUM_GPUS GPU(s)"

# Read cache_size from trace metadata for a given cache%
get_cache_size() {
    local pct=$1
    local trace_dir="$TRACE_BASE/cache${pct}pct"
    local batched="$trace_dir/batched_trace.json"
    if [ ! -f "$batched" ]; then
        echo "ERROR: Batched trace not found: $batched" >&2
        echo "Run scripts/01_collect_traces.sh first." >&2
        return 1
    fi
    python3 -c "
import json, sys
with open('$batched') as f:
    d = json.load(f)
cs = d.get('scheduling', {}).get('cache_size')
if cs is None:
    print('ERROR: cache_size not found in trace scheduling config', file=sys.stderr)
    sys.exit(1)
print(cs)
"
}

# Single cache% mode (all policies)
if [ -n "$SINGLE_PCT" ]; then
    echo "Running single cache% = $SINGLE_PCT"
    CS=$(get_cache_size "$SINGLE_PCT")
    python3 -u scripts/batched_replay.py \
        --model "$MODEL" \
        --trace-dir "$TRACE_BASE/cache${SINGLE_PCT}pct" \
        --batched-trace "$TRACE_BASE/cache${SINGLE_PCT}pct/batched_trace.json" \
        --cache-size "$CS" \
        --warmup-steps "$WARMUP" \
        --output-dir "$RESULTS_TMP"
    exit 0
fi

# Pre-check: all trace dirs exist and have cache_size
declare -A CACHE_SIZE_MAP
for pct in "${CACHE_PCTS[@]}"; do
    CS=$(get_cache_size "$pct") || exit 1
    CACHE_SIZE_MAP[$pct]=$CS
    echo "  cache${pct}%: cache_size=$CS"
done

# Build job list: (policy_group_idx, cache_pct) in priority order
JOBS=()
SKIPPED=0
for ((g=0; g<${#POLICY_GROUPS[@]}; g++)); do
    for pct in "${CACHE_PCTS[@]}"; do
        if [ "$RESUME" -eq 1 ]; then
            all_done=1
            IFS=',' read -ra policies <<< "${POLICY_GROUPS[$g]}"
            for pol in "${policies[@]}"; do
                if [ ! -f "$RESULTS_TMP/cache${pct}pct-${pol}.json" ]; then
                    all_done=0
                    break
                fi
            done
            if [ "$all_done" -eq 1 ]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi
        JOBS+=("$g:$pct")
    done
done
if [ "$RESUME" -eq 1 ] && [ "$SKIPPED" -gt 0 ]; then
    echo "Resuming: skipped $SKIPPED completed jobs"
fi

echo ""
echo "Job queue (${#JOBS[@]} jobs, ${NUM_GPUS} GPUs):"
for job in "${JOBS[@]}"; do
    g="${job%%:*}"
    pct="${job##*:}"
    echo "  ${GROUP_NAMES[$g]} @ cache${pct}%"
done
echo ""

# Dispatch jobs to GPUs
declare -a GPU_PID
declare -a GPU_LOG
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PID[$g]=0
done

FAILED=0
JOB_IDX=0

launch_on_gpu() {
    local gpu=$1
    local job=${JOBS[$JOB_IDX]}
    local gidx="${job%%:*}"
    local pct="${job##*:}"
    local policies="${POLICY_GROUPS[$gidx]}"
    local gname="${GROUP_NAMES[$gidx]}"
    local cs="${CACHE_SIZE_MAP[$pct]}"
    local logfile="/tmp/gpu_replay_gpu${gpu}_${gname}_cache${pct}.log"

    echo "[job $((JOB_IDX+1))/${#JOBS[@]}] GPU $gpu: ${gname} @ cache${pct}% (CS=$cs) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/batched_replay.py \
        --model "$MODEL" \
        --trace-dir "$TRACE_BASE/cache${pct}pct" \
        --batched-trace "$TRACE_BASE/cache${pct}pct/batched_trace.json" \
        --cache-size "$cs" \
        --policies "$policies" \
        --warmup-steps "$WARMUP" \
        --output-dir "$RESULTS_TMP" \
        > "$logfile" 2>&1 &
    GPU_PID[$gpu]=$!
    GPU_LOG[$gpu]="$logfile"
    JOB_IDX=$((JOB_IDX + 1))
}

wait_for_any_gpu() {
    while true; do
        for ((g=0; g<NUM_GPUS; g++)); do
            local pid=${GPU_PID[$g]}
            if [ "$pid" -eq 0 ]; then
                continue
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                if wait "$pid" 2>/dev/null; then
                    echo "  GPU $g finished: $(basename ${GPU_LOG[$g]}) [OK]"
                    tail -5 "${GPU_LOG[$g]}" | sed 's/^/    /'
                else
                    echo "  GPU $g FAILED: $(basename ${GPU_LOG[$g]})"
                    tail -20 "${GPU_LOG[$g]}" | sed 's/^/    /'
                    FAILED=1
                fi
                GPU_PID[$g]=0
                return 0
            fi
        done
        sleep 5
    done
}

# Main dispatch loop
while [ $JOB_IDX -lt ${#JOBS[@]} ]; do
    free_gpu=-1
    for ((g=0; g<NUM_GPUS; g++)); do
        if [ "${GPU_PID[$g]}" -eq 0 ]; then
            free_gpu=$g
            break
        fi
    done

    if [ $free_gpu -ge 0 ]; then
        launch_on_gpu $free_gpu
    else
        wait_for_any_gpu
    fi
done

# Wait for remaining jobs
while true; do
    active=0
    for ((g=0; g<NUM_GPUS; g++)); do
        if [ "${GPU_PID[$g]}" -ne 0 ]; then
            active=1
            break
        fi
    done
    [ $active -eq 0 ] && break
    wait_for_any_gpu
done

echo ""
if [ "$FAILED" -eq 1 ]; then
    echo "WARNING: Some jobs failed. Check logs above."
    exit 1
fi
echo "Phase 3 complete. Results in $RESULTS_TMP/"
