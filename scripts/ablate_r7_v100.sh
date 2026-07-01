#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

CONFIG="${CONFIG:-configs/r7_3class_v100_tuned.yaml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablations_r7}"
MAX_ITERATIONS="${MAX_ITERATIONS:-8000}"
ABLATIONS="${ABLATIONS:-full no_sam no_prior_feedback no_copy_paste no_strong_consistency no_trust_conditioned_floor no_ssl_class_balance no_topology_filter no_prompt_consistency no_eval_topology no_boundary no_class_balance}"

python tools/validate_dataset.py --config "${CONFIG}"

for name in ${ABLATIONS}; do
  output_dir="${BASE_OUTPUT_DIR}/${name}"
  mkdir -p "${output_dir}"
  args=(--config "${CONFIG}" --output-dir "${output_dir}" --max-iterations "${MAX_ITERATIONS}")
  case "${name}" in
    full)
      python tools/verify_real_sam.py --config "${CONFIG}"
      ;;
    no_sam)
      args+=(--no-sam)
      ;;
    no_prior_feedback)
      python tools/verify_real_sam.py --config "${CONFIG}"
      args+=(--opts prior_feedback.enabled false)
      ;;
    no_copy_paste)
      python tools/verify_real_sam.py --config "${CONFIG}"
      args+=(--opts copy_paste.enabled false)
      ;;
    no_strong_consistency)
      python tools/verify_real_sam.py --config "${CONFIG}"
      args+=(--opts losses.strong_view_consistency.enabled false)
      ;;
    no_trust_conditioned_floor)
      python tools/verify_real_sam.py --config "${CONFIG}"
      args+=(--opts sam.losses.trust_conditioned_floor false)
      ;;
    no_ssl_class_balance)
      python tools/verify_real_sam.py --config "${CONFIG}"
      args+=(--opts losses.ssl_class_balance.enabled false)
      ;;
    no_topology_filter)
      python tools/verify_real_sam.py --config "${CONFIG}"
      args+=(--opts pseudo.topology_candidate_filter_enabled false)
      ;;
    no_prompt_consistency)
      python tools/verify_real_sam.py --config "${CONFIG}"
      args+=(--opts sam.losses.prompt_consistency_weight 0.0)
      ;;
    no_eval_topology)
      python tools/verify_real_sam.py --config "${CONFIG}"
      args+=(--opts eval.topology_postprocess.enabled false)
      ;;
    no_boundary)
      python tools/verify_real_sam.py --config "${CONFIG}"
      args+=(--opts model.use_boundary_head false losses.supervised_boundary_weight 0.0 sam.losses.sam_boundary_weight 0.0 pseudo.boundary_weight 0.0)
      ;;
    no_class_balance)
      python tools/verify_real_sam.py --config "${CONFIG}"
      args+=(--opts losses.class_balanced_ce.enabled false)
      ;;
    *)
      echo "Unknown ablation: ${name}" >&2
      exit 2
      ;;
  esac

  echo "=== R7 ablation: ${name} -> ${output_dir} ==="
  python train_r7.py "${args[@]}" 2>&1 | tee "${output_dir}/stdout.log"
done

compare_args=()
for name in ${ABLATIONS}; do
  compare_args+=("${BASE_OUTPUT_DIR}/${name}")
done
python tools/compare_r7_runs.py "${compare_args[@]}" --report "${BASE_OUTPUT_DIR}/compare_report.json"
