#!/usr/bin/env bash
# Download OLMoE-1B-7B model weights and create a symlink in this directory.
#
# Usage:
#   ./download.sh <target_dir>
#
# Example:
#   ./download.sh ~/project/models
#
# This downloads allenai/OLMoE-1B-7B-0924 into <target_dir>/OLMoE-1B-7B
# and creates a symlink: models/OLMoE-1B-7B -> <target_dir>/OLMoE-1B-7B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="OLMoE-1B-7B"
HF_REPO="allenai/OLMoE-1B-7B-0924"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <target_dir>"
    echo "  Downloads ${HF_REPO} into <target_dir>/${MODEL_NAME}"
    echo "  and symlinks it into this directory."
    exit 1
fi

TARGET_DIR="$(realpath "$1")"
MODEL_DIR="${TARGET_DIR}/${MODEL_NAME}"

# Download
if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    echo "Model already exists at ${MODEL_DIR}, skipping download."
else
    echo "Downloading ${HF_REPO} to ${MODEL_DIR}..."
    mkdir -p "${TARGET_DIR}"
    huggingface-cli download "${HF_REPO}" --local-dir "${MODEL_DIR}"
fi

# Create symlink
LINK_PATH="${SCRIPT_DIR}/${MODEL_NAME}"
if [ -L "${LINK_PATH}" ]; then
    rm "${LINK_PATH}"
elif [ -e "${LINK_PATH}" ]; then
    echo "Error: ${LINK_PATH} exists and is not a symlink. Remove it manually."
    exit 1
fi

ln -s "${MODEL_DIR}" "${LINK_PATH}"
echo "Symlink created: ${LINK_PATH} -> ${MODEL_DIR}"
