#!/bin/bash
# Container runner for vLLM on RHEL 8 / H100 via Apptainer.
#
# Usage:
#   bash vllm_apptainer.sh                         # interactive shell
#   bash vllm_apptainer.sh "cd /workspace/... && python script.py"
#
# Environment variables (all optional):
#   VLLM_VERSION   vLLM version tag              (default: v0.17.1)
#   SIF_DIR        Where to store the .sif image (default: ~/software/containers)
#   MODELS_DIR     Host path to model weights    (auto-detected from symlinks)
#   DATASETS_DIR   Host path to datasets         (auto-detected from symlinks)

set -euo pipefail

# =========================================================
#  PATH RESOLUTION
# =========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# =========================================================
#  CONFIGURATION
# =========================================================

VLLM_VERSION="${VLLM_VERSION:-v0.17.1}"
SIF_DIR="${SIF_DIR:-$HOME/software/containers}"
SIF_FILE="$SIF_DIR/vllm-openai_${VLLM_VERSION}.sif"
CONTAINER_WORKSPACE="/workspace/GPU_Offloading"

# Apptainer cache — keep off of small $HOME quota
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/scratch/.apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$HOME/scratch/.apptainer/tmp}"

# Auto-detect a storage directory by resolving the first symlink found
# in a given directory (e.g. src/MoE/models/Mixtral-8x7B -> /gpfs/.../models/Mixtral-8x7B
# yields /gpfs/.../models).
auto_detect_dir() {
    local search_dir="$1"
    for link in "$search_dir"/*; do
        if [ -L "$link" ]; then
            local target
            target="$(readlink -f "$link")"
            if [ -e "$target" ]; then
                dirname "$target"
                return
            fi
        fi
    done
}

# Try repo-root models/ first, fall back to src/MoE/models/
MODELS_DIR="${MODELS_DIR:-$(auto_detect_dir "$REPO_ROOT/models" 2>/dev/null \
    || auto_detect_dir "$REPO_ROOT/src/MoE/models" 2>/dev/null || true)}"
DATASETS_DIR="${DATASETS_DIR:-$(auto_detect_dir "$REPO_ROOT/datasets" 2>/dev/null \
    || auto_detect_dir "$REPO_ROOT/src/MoE/datasets" 2>/dev/null || true)}"

# =========================================================
#  BUILD BIND-MOUNT ARGUMENTS
# =========================================================

BIND_ARGS="$REPO_ROOT:$CONTAINER_WORKSPACE"

if [ -n "$MODELS_DIR" ] && [ -d "$MODELS_DIR" ]; then
    # Mount at the same host path so existing symlinks resolve inside the container
    BIND_ARGS="$BIND_ARGS,$MODELS_DIR:$MODELS_DIR"
else
    echo "WARNING: No models directory detected."
    echo "  Set MODELS_DIR or create symlinks in src/MoE/models/ first."
fi

if [ -n "$DATASETS_DIR" ] && [ -d "$DATASETS_DIR" ]; then
    BIND_ARGS="$BIND_ARGS,$DATASETS_DIR:$DATASETS_DIR"
else
    echo "WARNING: No datasets directory detected."
fi

# Also bind GPFS root so absolute paths in configs/scripts work
if [ -d /gpfs ]; then
    BIND_ARGS="$BIND_ARGS,/gpfs:/gpfs"
fi

# =========================================================
#  PULL IMAGE (if needed)
# =========================================================

if [ ! -f "$SIF_FILE" ]; then
    echo "--------------------------------------------------------------------------------"
    echo "Image not found at: $SIF_FILE"
    echo "Pulling vllm/vllm-openai:${VLLM_VERSION} from Docker Hub ..."
    echo "--------------------------------------------------------------------------------"

    mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

    apptainer pull "$SIF_FILE" "docker://vllm/vllm-openai:${VLLM_VERSION}"

    echo "Pull complete: $SIF_FILE"
else
    echo "Image found: $SIF_FILE"
fi

# =========================================================
#  RUN LOGIC
# =========================================================

echo "Starting container: vllm-openai:${VLLM_VERSION}"
echo "  Repo root   : $REPO_ROOT"
echo "  Models dir  : ${MODELS_DIR:-(not set)}"
echo "  Datasets dir: ${DATASETS_DIR:-(not set)}"

# The base image only has python3; create a temp dir with a python symlink
# so scripts that invoke `python` work without rebuilding the SIF.
_PY_SHIM_DIR="$(mktemp -d)"
ln -sf /usr/bin/python3 "$_PY_SHIM_DIR/python"

if [ $# -gt 0 ]; then
    echo "  Command     : $*"
    apptainer exec --nv \
        --bind "$BIND_ARGS,$_PY_SHIM_DIR:$_PY_SHIM_DIR" \
        --pwd "$CONTAINER_WORKSPACE" \
        "$SIF_FILE" \
        bash -c "export PATH=\"$_PY_SHIM_DIR:\$PATH\" && $*"
else
    apptainer shell --nv \
        --bind "$BIND_ARGS,$_PY_SHIM_DIR:$_PY_SHIM_DIR" \
        --pwd "$CONTAINER_WORKSPACE" \
        "$SIF_FILE"
fi

rm -rf "$_PY_SHIM_DIR"
