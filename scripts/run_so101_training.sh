#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FARPOINT_SO101_TRAINING_IMAGE:-farpoint-so101-lerobot-training:0.4.4}"
DATA_ROOT="${FARPOINT_TRAINING_DATA_ROOT:-${PROJECT_ROOT}/.cache/farpoint/training/datasets}"
MODEL_ROOT="${FARPOINT_TRAINING_MODEL_ROOT:-${PROJECT_ROOT}/.cache/farpoint/training/models}"
LOG_ROOT="${FARPOINT_TRAINING_LOG_ROOT:-${PROJECT_ROOT}/.cache/farpoint/training/logs}"
CACHE_ROOT="${FARPOINT_TRAINING_CACHE_ROOT:-${PROJECT_ROOT}/.cache/farpoint/training/cache}"
IMMUTABLE_SOURCE_ROOT="${FARPOINT_TRAINING_IMMUTABLE_SOURCE_ROOT:-}"
RESUME_CHECKPOINT_ROOT="${FARPOINT_TRAINING_RESUME_CHECKPOINT_ROOT:-}"
GIT_COMMIT="${FARPOINT_GIT_COMMIT:-}"

if [[ -z "${GIT_COMMIT}" ]] && git -C "${PROJECT_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
fi
if [[ ! "${GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FARPOINT_GIT_COMMIT must identify the exact 40-character source commit" >&2
  exit 2
fi
IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi
mkdir -p "${DATA_ROOT}" "${MODEL_ROOT}" "${LOG_ROOT}" "${CACHE_ROOT}"

docker_args=(
  --rm --gpus all --ipc=host --network host
  --user "$(id -u):$(id -g)"
  --workdir /workspace/project
  --env HOME=/workspace/cache/home
  --env HF_HOME=/workspace/cache/huggingface
  --env PYTHONPATH=/workspace/project/src
  --env PYTHONUNBUFFERED=1
  --env FARPOINT_GIT_COMMIT="${GIT_COMMIT}"
  --env FARPOINT_TRAINING_IMAGE_ID="${IMAGE_ID}"
  --volume "${PROJECT_ROOT}:/workspace/project:ro"
  --volume "${DATA_ROOT}:/workspace/datasets:rw"
  --volume "${MODEL_ROOT}:/workspace/models:rw"
  --volume "${LOG_ROOT}:/workspace/logs:rw"
  --volume "${CACHE_ROOT}:/workspace/cache:rw"
)
if [[ -n "${IMMUTABLE_SOURCE_ROOT}" ]]; then
  if [[ ! -d "${IMMUTABLE_SOURCE_ROOT}" ]]; then
    echo "FARPOINT_TRAINING_IMMUTABLE_SOURCE_ROOT must name an existing directory" >&2
    exit 2
  fi
  docker_args+=(--volume "${IMMUTABLE_SOURCE_ROOT}:/workspace/source-dataset:ro")
fi
if [[ -n "${RESUME_CHECKPOINT_ROOT}" ]]; then
  if [[ ! -d "${RESUME_CHECKPOINT_ROOT}" ]]; then
    echo "FARPOINT_TRAINING_RESUME_CHECKPOINT_ROOT must name an existing directory" >&2
    exit 2
  fi
  docker_args+=(--volume "${RESUME_CHECKPOINT_ROOT}:/workspace/resume-checkpoint:ro")
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_args+=(--env HF_TOKEN)
fi

docker run "${docker_args[@]}" "${IMAGE}" "$@"
