#!/usr/bin/env bash
# Phase 4: GPU replay of all policy traces across all cache fractions.
# Auto-detects GPU count and distributes work across GPUs.
#
# Execution order: policies first (SF → Belady → LFU → LRU), then cache sizes
# within each policy group (80 → 70 → 60 → 50 → 85). LRU runs last since it
# has by far the most demand loads and takes the longest.
#
# Each (policy_group, cache%) job is dispatched to the next free GPU.
#
# Usage:
#   bash scripts/04_gpu_replay.sh                    # auto-detect GPUs
#   bash scripts/04_gpu_replay.sh --cache-pct 85     # single cache%
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

# Cache sizes in priority order
CACHE_PCTS=(80 70 60 50 85)

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

# Single cache% mode (all policies)
if [ -n "$SINGLE_PCT" ]; then
    echo "Running single cache% = $SINGLE_PCT"
    python3 -u scripts/batched_replay.py \
        --model "$MODEL" \
        --trace-dir "$TRACE_BASE" \
        --cache-pct "$SINGLE_PCT" \
        --warmup-steps "$WARMUP" \
        --output-dir "$RESULTS_TMP"
    exit 0
fi

# Build job list: (policy_group_idx, cache_pct) in priority order
# Policies first, then cache sizes within each policy group
# With --resume, skip jobs where all policy result files already exist
JOBS=()
SKIPPED=0
for ((g=0; g<${#POLICY_GROUPS[@]}; g++)); do
    for pct in "${CACHE_PCTS[@]}"; do
        if [ "$RESUME" -eq 1 ]; then
            # Check if all policies in this group have result files
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

# Dispatch jobs to GPUs: launch up to NUM_GPUS in parallel, wait for
# any to finish before launching the next.
declare -a GPU_PID     # PID running on each GPU (0 = free)
declare -a GPU_LOG     # log file for current job on each GPU
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PID[$g]=0
done

FAILED=0
JOB_IDX=0

# Launch a job on a specific GPU
launch_on_gpu() {
    local gpu=$1
    local job=${JOBS[$JOB_IDX]}
    local gidx="${job%%:*}"
    local pct="${job##*:}"
    local policies="${POLICY_GROUPS[$gidx]}"
    local gname="${GROUP_NAMES[$gidx]}"
    local logfile="/tmp/gpu_replay_gpu${gpu}_${gname}_cache${pct}.log"

    echo "[job $((JOB_IDX+1))/${#JOBS[@]}] GPU $gpu: ${gname} @ cache${pct}% -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/batched_replay.py \
        --model "$MODEL" \
        --trace-dir "$TRACE_BASE" \
        --cache-pct "$pct" \
        --policies "$policies" \
        --warmup-steps "$WARMUP" \
        --output-dir "$RESULTS_TMP" \
        > "$logfile" 2>&1 &
    GPU_PID[$gpu]=$!
    GPU_LOG[$gpu]="$logfile"
    JOB_IDX=$((JOB_IDX + 1))
}

# Wait for any GPU to become free, report result
wait_for_any_gpu() {
    while true; do
        for ((g=0; g<NUM_GPUS; g++)); do
            local pid=${GPU_PID[$g]}
            if [ "$pid" -eq 0 ]; then
                continue
            fi
            # Check if process finished (non-blocking)
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process finished, get exit code
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
    # Find a free GPU
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
        # All GPUs busy, wait for one to finish
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
echo "All jobs complete. Merging results..."

if [ "$FAILED" -eq 1 ]; then
    echo "WARNING: Some jobs failed. Check logs above."
    exit 1
fi
echo ""
echo "Phase 4 complete. Results in results/MoE/mixtral-8x7B/tmp/"
