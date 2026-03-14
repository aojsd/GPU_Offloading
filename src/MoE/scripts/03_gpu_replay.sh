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
#   bash scripts/03_gpu_replay.sh                        # auto-detect GPUs + cache%
#   bash scripts/03_gpu_replay.sh --cache-pct 70,80,90   # specific cache percents
#   bash scripts/03_gpu_replay.sh --cache-pct 80         # single cache%
#   bash scripts/03_gpu_replay.sh --resume               # skip completed jobs
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

MODEL=${MODEL:-../../models/Mixtral-8x7B}
WARMUP=100

# Policy groups in priority order
POLICY_GROUPS=(
    "StaticFreq-None,StaticFreq-Oracle,StaticFreq-Oracle(1)"
    "Belady-None,Belady-Oracle,Belady-Oracle(1)"
    "LFU-None,LFU-Oracle,LFU-Oracle(1)"
    "LRU-None,LRU-Oracle,LRU-Oracle(1)"
)
GROUP_NAMES=("SF" "Belady" "LFU" "LRU")

# Parse args
USER_CACHE_PCTS=""
RESUME=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-pct) USER_CACHE_PCTS="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --resume)    RESUME=1; shift ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done
# Re-derive MODEL_TAG after arg parsing
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
TRACE_BASE=../../datasets/ShareGPT_Vicuna/expert_traces/${MODEL_TAG}
RESULTS_TMP="../../results/MoE/${MODEL_TAG}/tmp"

# Cache percentages: use user-specified or auto-detect from existing trace directories
if [ -n "$USER_CACHE_PCTS" ]; then
    IFS=',' read -ra CACHE_PCTS <<< "$USER_CACHE_PCTS"
else
    CACHE_PCTS=()
    for d in "$TRACE_BASE"/cache*pct; do
        [ -d "$d" ] || continue
        [ -f "$d/batched_trace.json" ] || continue
        pct="${d##*/}"            # cache80pct
        pct="${pct#cache}"        # 80pct
        pct="${pct%pct}"          # 80
        CACHE_PCTS+=("$pct")
    done
fi
# Sort descending (highest cache% first = fastest jobs first)
IFS=$'\n' CACHE_PCTS=($(printf '%s\n' "${CACHE_PCTS[@]}" | sort -rn)); unset IFS
if [ ${#CACHE_PCTS[@]} -eq 0 ]; then
    echo "ERROR: No trace directories found in $TRACE_BASE/cache*pct/"
    echo "Run scripts/01_collect_traces.sh first."
    exit 1
fi
echo "Cache percentages: ${CACHE_PCTS[*]}"

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

# Results aggregation: rebuild results_<GPU>.md from all JSONs so far
RESULTS_DIR="../../results/MoE/${MODEL_TAG}"
aggregate_results() {
    python3 -c "
import json, glob, os, sys
tmp_dir = '$RESULTS_TMP'
results_dir = '$RESULTS_DIR'
jsons = sorted(glob.glob(os.path.join(tmp_dir, 'cache*pct-*.json')))
if not jsons:
    sys.exit(0)
with open(jsons[0]) as f:
    d = json.load(f)
env = d.get('env', {})
gpu_name = env.get('gpu_name', 'unknown')
gpu_tag = 'unknown'
for token in gpu_name.split():
    if any(c.isdigit() for c in token) and len(token) >= 3:
        gpu_tag = token
        break
md_path = os.path.join(results_dir, f'results_{gpu_tag}.md')
os.makedirs(results_dir, exist_ok=True)
header = (
    f'### GPU Replay: Wall-Clock Timing ({gpu_name})\n\n'
    '| Cache% | Policy | ms/step | Compute% | Demands | Prefetches |\n'
    '|--------|--------|---------|----------|---------|------------|\n'
)
rows = []
for jp in jsons:
    with open(jp) as f:
        r = json.load(f)
    rows.append(
        f'| {r[\"cache_pct\"]}%    '
        f'| {r[\"policy\"]:<20s} '
        f'| {r[\"ms_per_step\"]:>7.2f} '
        f'| {r.get(\"compute_pct\", 0):>6.1f}% '
        f'| {r.get(\"demands\", 0):>7d} '
        f'| {r.get(\"prefetches\", 0):>10d} |'
    )
with open(md_path, 'w') as f:
    f.write(header)
    f.write('\n'.join(rows) + '\n')
print(f'  results: {len(rows)} rows -> {md_path}')
" 2>/dev/null
}

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
                aggregate_results
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
