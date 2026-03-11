#!/bin/bash
# Container runner for the NGC PyTorch + vLLM environment.
#
# Usage:
#   bash pytorch_ngc.sh                         # interactive shell
#   bash pytorch_ngc.sh "cd /workspace/... && python script.py"
#
# Environment variables (all optional):
#   DOCKER_IMAGE   Docker image name            (default: pytorch-gh200)
#   MODELS_DIR     Host path to model weights   (auto-detected from symlinks)
#   DATASETS_DIR   Host path to datasets        (auto-detected from symlinks)

set -euo pipefail

# =========================================================
#  PATH RESOLUTION
# =========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# =========================================================
#  CONFIGURATION
# =========================================================

IMAGE_NAME="${DOCKER_IMAGE:-pytorch-gh200}"
CONTAINER_WORKSPACE="/workspace/GPU_Offloading"

# Auto-detect a storage directory by resolving the first symlink found
# in a given directory (e.g. src/MoE/models/Mixtral-8x7B -> /data/models/Mixtral-8x7B
# yields /data/models).
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

MODELS_DIR="${MODELS_DIR:-$(auto_detect_dir "$REPO_ROOT/src/MoE/models" 2>/dev/null || true)}"
DATASETS_DIR="${DATASETS_DIR:-$(auto_detect_dir "$REPO_ROOT/src/MoE/datasets" 2>/dev/null || true)}"

# Build optional volume arguments — only mount if the directory exists
VOLUME_ARGS=()
VOLUME_ARGS+=(-v "$REPO_ROOT:$CONTAINER_WORKSPACE")
VOLUME_ARGS+=(-v /tmp:/tmp)

if [ -n "$MODELS_DIR" ] && [ -d "$MODELS_DIR" ]; then
    # Mount at the same host path so existing symlinks resolve inside the container
    VOLUME_ARGS+=(-v "$MODELS_DIR:$MODELS_DIR")
else
    echo "WARNING: No models directory detected."
    echo "  Set MODELS_DIR or run src/MoE/models/download.sh first."
fi

if [ -n "$DATASETS_DIR" ] && [ -d "$DATASETS_DIR" ]; then
    VOLUME_ARGS+=(-v "$DATASETS_DIR:$DATASETS_DIR")
else
    echo "WARNING: No datasets directory detected."
    echo "  Set DATASETS_DIR or run src/MoE/datasets/download.sh first."
fi

# =========================================================
#  AUTOMATED BUILD LOGIC
# =========================================================

if [[ "$(sudo docker images -q "$IMAGE_NAME" 2>/dev/null)" == "" ]]; then
    echo "--------------------------------------------------------------------------------"
    echo "Image '$IMAGE_NAME' not found."
    echo "Building image from Dockerfile in $SCRIPT_DIR ..."
    echo "--------------------------------------------------------------------------------"

    if [ ! -f "$SCRIPT_DIR/Dockerfile" ]; then
        echo "ERROR: Dockerfile not found in $SCRIPT_DIR. Cannot build image."
        exit 1
    fi

    sudo docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Docker build failed."
        exit 1
    fi

    echo "Build complete."
else
    echo "Image '$IMAGE_NAME' found. Skipping build."
fi

# =========================================================
#  RUN LOGIC
# =========================================================

echo "Starting container: $IMAGE_NAME"
echo "  Repo root  : $REPO_ROOT"
echo "  Models dir : ${MODELS_DIR:-(not set)}"
echo "  Datasets dir: ${DATASETS_DIR:-(not set)}"

if [ $# -gt 0 ]; then
    echo "  Command    : $*"
    sudo docker run --gpus all --ipc=host \
        "${VOLUME_ARGS[@]}" \
        --privileged \
        --ulimit memlock=-1 --ulimit stack=67108864 --rm \
        "$IMAGE_NAME" bash -c "$*"
else
    sudo docker run --gpus all --ipc=host \
        "${VOLUME_ARGS[@]}" \
        --privileged \
        --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
        "$IMAGE_NAME"
fi
