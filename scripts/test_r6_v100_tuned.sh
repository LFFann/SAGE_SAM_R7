#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/SAGE_SAM_R6_3Class_V100_Tuned}"
CONFIG="${CONFIG:-${OUTPUT_DIR}/resolved_config.yaml}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoints/best_val_dice.pth}"
if [[ ! -f "${CHECKPOINT}" && -f "${OUTPUT_DIR}/checkpoints/latest.pth" ]]; then
  CHECKPOINT="${OUTPUT_DIR}/checkpoints/latest.pth"
fi

python validate_r6.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  "$@" \
  2>&1 | tee "${OUTPUT_DIR}/validate_stdout.log"

python test_r6.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --save-pred \
  2>&1 | tee "${OUTPUT_DIR}/test_stdout.log"

python export_deploy_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT_DIR}/checkpoints/deploy_student.pth" \
  2>&1 | tee "${OUTPUT_DIR}/export_stdout.log"
