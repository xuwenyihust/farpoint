#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-}"
CONFIG_NAME="${2:-}"
PROFILE="${3:-}"
SOURCE_REVISION="${4:-}"
RESUME_CHECKPOINT="${5:-}"

if [[ -z "${RUN_ID}" || ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "usage: $0 RUN_ID CONFIG_NAME PROFILE SOURCE_REVISION [RESUME_CHECKPOINT]" >&2
  exit 2
fi
if [[ ! "${CONFIG_NAME}" =~ ^[a-zA-Z0-9._-]+\.json$ ]]; then
  echo "CONFIG_NAME must be a basename under configs/training" >&2
  exit 2
fi
if [[ ! "${PROFILE}" =~ ^(smoke|pilot|training)$ ]]; then
  echo "PROFILE must be smoke, pilot, or training" >&2
  exit 2
fi
if [[ -z "${SOURCE_REVISION}" || ! "${SOURCE_REVISION}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "SOURCE_REVISION must be a safe dataset revision path component" >&2
  exit 2
fi
if [[ -n "${RESUME_CHECKPOINT}" ]] && [[ "${RESUME_CHECKPOINT}" != "/workspace/resume-checkpoint" ]]; then
  echo "resume checkpoint must use the read-only /workspace/resume-checkpoint mount" >&2
  exit 2
fi
if [[ -n "${RESUME_CHECKPOINT}" && -z "${FARPOINT_TRAINING_RESUME_CHECKPOINT_ROOT:-}" ]]; then
  echo "FARPOINT_TRAINING_RESUME_CHECKPOINT_ROOT is required for continuation" >&2
  exit 2
fi

CONFIG="/workspace/project/configs/training/${CONFIG_NAME}"
if [[ -n "${FARPOINT_TRAINING_IMMUTABLE_SOURCE_ROOT:-}" ]]; then
  SOURCE_ROOT="/workspace/source-dataset"
else
  SOURCE_ROOT="/workspace/datasets/source/wenyixu101/farpoint-so101/${SOURCE_REVISION}"
fi
if [[ "${SOURCE_ROOT}" != "/workspace/source-dataset" ]] && \
  { [[ ! "${SOURCE_ROOT}" =~ ^/workspace/datasets/[a-zA-Z0-9._/-]+$ ]] || [[ "${SOURCE_ROOT}" == *".."* ]]; }; then
  echo "training source must be the immutable mount or a safe path under /workspace/datasets" >&2
  exit 2
fi
VIEW_ROOT="/workspace/datasets/views/${RUN_ID}"
OUTPUT_DIR="/workspace/models/${RUN_ID}"
PREFLIGHT_REPORT="/workspace/logs/${RUN_ID}/preflight.json"
VALIDATION_REPORT="/workspace/logs/${RUN_ID}/validation.json"

preflight_args=(
  python /workspace/project/scripts/preflight_policy_training.py
  --config "${CONFIG}"
  --source-root "${SOURCE_ROOT}"
  --view-root "${VIEW_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --report "${PREFLIGHT_REPORT}"
  --profile "${PROFILE}"
  --run-profile
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  preflight_args+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
"${PROJECT_ROOT}/scripts/run_so101_training.sh" "${preflight_args[@]}"

if [[ "${PROFILE}" != "smoke" ]]; then
  "${PROJECT_ROOT}/scripts/run_so101_training.sh" \
    python /workspace/project/scripts/evaluate_act_checkpoints.py \
    --config "${CONFIG}" \
    --dataset-root "${VIEW_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --preflight-report "${PREFLIGHT_REPORT}" \
    --report "${VALIDATION_REPORT}" \
    --profile "${PROFILE}"
fi
