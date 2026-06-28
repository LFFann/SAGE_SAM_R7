#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
python test_r6.py --config outputs/SAGE_SAM_R6_3Class/resolved_config.yaml --checkpoint outputs/SAGE_SAM_R6_3Class/checkpoints/best_val_dice.pth --save-pred --split test
