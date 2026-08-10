#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" || ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "usage: $0 RUN_ID" >&2
  exit 2
fi

CONFIG="/workspace/project/configs/training/so101_act_v0_0_3_pilot.json"
SOURCE_ROOT="/workspace/datasets/source/wenyixu101/farpoint-so101/v0.0.3"
VIEW_ROOT="/workspace/datasets/views/${RUN_ID}"
OUTPUT_DIR="/workspace/models/${RUN_ID}"
PREFLIGHT_REPORT="/workspace/logs/${RUN_ID}/preflight.json"
VALIDATION_REPORT="/workspace/logs/${RUN_ID}/validation.json"

"${PROJECT_ROOT}/scripts/run_so101_training.sh" \
  python /workspace/project/scripts/preflight_policy_training.py \
  --config "${CONFIG}" \
  --source-root "${SOURCE_ROOT}" \
  --view-root "${VIEW_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --report "${PREFLIGHT_REPORT}" \
  --profile pilot \
  --run-profile

"${PROJECT_ROOT}/scripts/run_so101_training.sh" \
  python /workspace/project/scripts/evaluate_act_checkpoints.py \
  --config "${CONFIG}" \
  --dataset-root "${VIEW_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --preflight-report "${PREFLIGHT_REPORT}" \
  --report "${VALIDATION_REPORT}"
