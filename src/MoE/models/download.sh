#!/usr/bin/env bash
# Download MoE model weights and create a symlink in this directory.
#
# Usage:
#   ./download.sh <target_dir> [model]
#
# Models:
#   olmoe     (default) allenai/OLMoE-1B-7B-0924
#   mixtral              mistralai/Mixtral-8x7B-Instruct-v0.1
#   deepseek2            deepseek-ai/DeepSeek-V2
#   deepseek2-lite       deepseek-ai/DeepSeek-V2-Lite
#
# Examples:
#   ./download.sh ~/project/models                   # downloads OLMoE
#   ./download.sh ~/project/models mixtral            # downloads Mixtral-8x7B
#   ./download.sh ~/project/models deepseek2          # downloads DeepSeek-V2
#   ./download.sh ~/project/models deepseek2-lite     # downloads DeepSeek-V2-Lite
#
# Downloads into <target_dir>/<model_name> and creates a symlink in this
# directory pointing to it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Model registry ───────────────────────────────────────────────────
declare -A HF_REPOS=(
    [olmoe]="allenai/OLMoE-1B-7B-0924"
    [mixtral]="mistralai/Mixtral-8x7B-Instruct-v0.1"
    [deepseek2]="deepseek-ai/DeepSeek-V2"
    [deepseek2-lite]="deepseek-ai/DeepSeek-V2-Lite"
)
declare -A LOCAL_NAMES=(
    [olmoe]="OLMoE-1B-7B"
    [mixtral]="Mixtral-8x7B"
    [deepseek2]="DeepSeek-V2"
    [deepseek2-lite]="DeepSeek-V2-Lite"
)

# ── Parse args ───────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: $0 <target_dir> [model]"
    echo ""
    echo "Models:"
    for key in "${!HF_REPOS[@]}"; do
        echo "  ${key}  ->  ${HF_REPOS[$key]}  (local: ${LOCAL_NAMES[$key]})"
    done
    exit 1
fi

TARGET_DIR="$(realpath "$1")"
MODEL_KEY="${2:-olmoe}"

if [ -z "${HF_REPOS[$MODEL_KEY]+x}" ]; then
    echo "Error: Unknown model '${MODEL_KEY}'. Available: ${!HF_REPOS[*]}"
    exit 1
fi

HF_REPO="${HF_REPOS[$MODEL_KEY]}"
MODEL_NAME="${LOCAL_NAMES[$MODEL_KEY]}"
MODEL_DIR="${TARGET_DIR}/${MODEL_NAME}"

# ── Download ─────────────────────────────────────────────────────────
if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    echo "Model already exists at ${MODEL_DIR}, skipping download."
else
    echo "Downloading ${HF_REPO} to ${MODEL_DIR}..."
    mkdir -p "${TARGET_DIR}"
    huggingface-cli download "${HF_REPO}" --local-dir "${MODEL_DIR}"
fi

# ── Create symlink ───────────────────────────────────────────────────
LINK_PATH="${SCRIPT_DIR}/${MODEL_NAME}"
if [ -L "${LINK_PATH}" ]; then
    rm "${LINK_PATH}"
elif [ -e "${LINK_PATH}" ]; then
    echo "Error: ${LINK_PATH} exists and is not a symlink. Remove it manually."
    exit 1
fi

ln -s "${MODEL_DIR}" "${LINK_PATH}"
echo "Symlink created: ${LINK_PATH} -> ${MODEL_DIR}"

# ── Post-download: truncate Mixtral ──────────────────────────────────
if [ "${MODEL_KEY}" = "mixtral" ]; then
    TRUNCATED_NAME="Mixtral-8x7B-20L"
    TRUNCATED_DIR="${TARGET_DIR}/${TRUNCATED_NAME}"
    TRUNCATED_LINK="${SCRIPT_DIR}/${TRUNCATED_NAME}"

    if [ -d "${TRUNCATED_DIR}" ] && [ -f "${TRUNCATED_DIR}/config.json" ]; then
        echo "Truncated model already exists at ${TRUNCATED_DIR}, skipping."
    else
        echo ""
        echo "Creating truncated Mixtral (20 layers) at ${TRUNCATED_DIR}..."
        # Use the same Python that has safetensors installed (conda env)
        PYTHON="${PYTHON:-python3}"
        "${PYTHON}" "${SCRIPT_DIR}/truncate_model.py" \
            "${MODEL_DIR}" "${TRUNCATED_DIR}" --num-layers 20
    fi

    # Symlink for truncated model
    if [ -L "${TRUNCATED_LINK}" ]; then
        rm "${TRUNCATED_LINK}"
    elif [ -e "${TRUNCATED_LINK}" ]; then
        echo "Error: ${TRUNCATED_LINK} exists and is not a symlink."
        exit 1
    fi

    ln -s "${TRUNCATED_DIR}" "${TRUNCATED_LINK}"
    echo "Symlink created: ${TRUNCATED_LINK} -> ${TRUNCATED_DIR}"
fi
