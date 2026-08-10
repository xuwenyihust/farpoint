#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FARPOINT_SO101_TRAINING_IMAGE:-farpoint-so101-lerobot-training:0.4.4}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "SO-101 training image must be built on DGX Spark (aarch64), got $(uname -m)" >&2
  exit 2
fi

docker build \
  --pull \
  --file "${PROJECT_ROOT}/docker/so101-lerobot-training/Dockerfile" \
  --tag "${IMAGE}" \
  "${PROJECT_ROOT}"

docker image inspect "${IMAGE}" --format '{{.Id}} {{.Architecture}}'
