#!/bin/bash
# Evaluate checkpoints on GSM8K + MATH via lm-evaluation-harness.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CKPT_PATH [BATCH_SIZE] [LIMIT]" >&2
  exit 1
fi

CKPT_PATH=$(realpath "$1")
BATCH_SIZE=${2:-4}
LIMIT=${3:-}

WORK_DIR="/home/hungpv/projects/lab/DTW-v2"
SAVE_ROOT="$(dirname "${CKPT_PATH}")/lm_eval/math/$(basename "${CKPT_PATH}")"

CMD=(
  python "${WORK_DIR}/code/run_lm_eval_math.py"
  --model-path "${CKPT_PATH}"
  --save-dir "${SAVE_ROOT}"
  --tasks gsm8k_cot hendrycks_math
  --batch-size "${BATCH_SIZE}"
  --dtype bfloat16
  --trust-remote-code
  --log-samples
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "Executing: ${CMD[*]}"
"${CMD[@]}"

