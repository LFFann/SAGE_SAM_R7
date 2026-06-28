#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

CONFIG="${CONFIG:-configs/r6_3class_v100_tuned.yaml}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1500}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/SAGE_SAM_R6_Diagnostic_${MAX_ITERATIONS}}"

python tools/validate_dataset.py \
  --config "${CONFIG}"

python tools/verify_real_sam.py \
  --config "${CONFIG}"

mkdir -p "${OUTPUT_DIR}"
python train_r6.py \
  --config "${CONFIG}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/stdout.log"

python tools/check_r6_diagnostics.py \
  --output-dir "${OUTPUT_DIR}" \
  --config "${OUTPUT_DIR}/resolved_config.yaml" \
  --min-train-rows 20 \
  --tail 200 \
  --correlation-after 2000 \
  --report "${OUTPUT_DIR}/diagnostic_report.json"
