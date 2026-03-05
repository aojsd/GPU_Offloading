#!/usr/bin/env bash
# Download datasets and create a symlink in this directory.
#
# Usage:
#   ./download.sh <target_dir> [dataset]
#
# Datasets:
#   sharegpt  (default) anon8231489123/ShareGPT_Vicuna_unfiltered
#
# Examples:
#   ./download.sh ~/project/datasets              # downloads ShareGPT
#
# Downloads into <target_dir>/<dataset_name> and creates a symlink in this
# directory pointing to it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Dataset registry ─────────────────────────────────────────────────
declare -A HF_REPOS=(
    [sharegpt]="anon8231489123/ShareGPT_Vicuna_unfiltered"
)
declare -A LOCAL_NAMES=(
    [sharegpt]="ShareGPT_Vicuna"
)
# Specific files to download (avoids pulling wheels, notebooks, etc.)
declare -A DOWNLOAD_FILES=(
    [sharegpt]="ShareGPT_V3_unfiltered_cleaned_split.json README.md"
)

# ── Parse args ───────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: $0 <target_dir> [dataset]"
    echo ""
    echo "Datasets:"
    for key in "${!HF_REPOS[@]}"; do
        echo "  ${key}  ->  ${HF_REPOS[$key]}  (local: ${LOCAL_NAMES[$key]})"
    done
    exit 1
fi

TARGET_DIR="$(realpath "$1")"
DATASET_KEY="${2:-sharegpt}"

if [ -z "${HF_REPOS[$DATASET_KEY]+x}" ]; then
    echo "Error: Unknown dataset '${DATASET_KEY}'. Available: ${!HF_REPOS[*]}"
    exit 1
fi

HF_REPO="${HF_REPOS[$DATASET_KEY]}"
DATASET_NAME="${LOCAL_NAMES[$DATASET_KEY]}"
DATASET_DIR="${TARGET_DIR}/${DATASET_NAME}"
FILES="${DOWNLOAD_FILES[$DATASET_KEY]}"

# ── Download ─────────────────────────────────────────────────────────
if [ -d "${DATASET_DIR}" ] && [ -f "${DATASET_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
    echo "Dataset already exists at ${DATASET_DIR}, skipping download."
else
    echo "Downloading ${HF_REPO} to ${DATASET_DIR}..."
    mkdir -p "${TARGET_DIR}"

    # Download each file explicitly
    for fname in ${FILES}; do
        echo "  Fetching ${fname}..."
        huggingface-cli download "${HF_REPO}" "${fname}" \
            --repo-type dataset \
            --local-dir "${DATASET_DIR}"
    done
fi

# ── Create symlink ───────────────────────────────────────────────────
LINK_PATH="${SCRIPT_DIR}/${DATASET_NAME}"
if [ -L "${LINK_PATH}" ]; then
    rm "${LINK_PATH}"
elif [ -e "${LINK_PATH}" ]; then
    echo "Error: ${LINK_PATH} exists and is not a symlink. Remove it manually."
    exit 1
fi

ln -s "${DATASET_DIR}" "${LINK_PATH}"
echo "Symlink created: ${LINK_PATH} -> ${DATASET_DIR}"
echo ""
echo "Dataset files:"
ls -lh "${DATASET_DIR}"/*.json 2>/dev/null || echo "  (no JSON files found)"
