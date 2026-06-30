#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

CONFIG="${CONFIG:-configs/r7_3class_v100_tuned.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/SAGE_SAM_R7_3Class_V100_Tuned_SAMAgreeKD}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
RESUME="${RESUME:-}"

python tools/validate_dataset.py \
  --config "${CONFIG}"

python tools/verify_real_sam.py \
  --config "${CONFIG}"

mkdir -p "${OUTPUT_DIR}"

train_args=(--config "${CONFIG}" --output-dir "${OUTPUT_DIR}")
if [[ -n "${MAX_ITERATIONS}" ]]; then
  train_args+=(--max-iterations "${MAX_ITERATIONS}")
fi
if [[ -n "${RESUME}" ]]; then
  train_args+=(--resume "${RESUME}")
fi

python train_r7.py "${train_args[@]}" "$@" \
  2>&1 | tee "${OUTPUT_DIR}/stdout.log"
