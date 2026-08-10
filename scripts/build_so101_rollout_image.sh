#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${FARPOINT_SO101_IMAGE:-farpoint-so101-isaaclab:3.0-beta2}"
IMAGE="${FARPOINT_SO101_ROLLOUT_IMAGE:-farpoint-so101-act-rollout:0.4.4}"
EXPECTED_BASE_ID="sha256:ddcd4daa68cef3ece67f4fbad4eb8f5257d8236a55aba04d0697b55e7679fd04"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "SO-101 rollout image must be built on DGX Spark (aarch64)" >&2
  exit 2
fi
ACTUAL_BASE_ID="$(docker image inspect "${BASE_IMAGE}" --format '{{.Id}}')"
if [[ "${ACTUAL_BASE_ID}" != "${EXPECTED_BASE_ID}" ]]; then
  echo "Isaac base image ID mismatch: ${ACTUAL_BASE_ID} != ${EXPECTED_BASE_ID}" >&2
  exit 2
fi

docker build \
  --file "${PROJECT_ROOT}/docker/so101-act-rollout/Dockerfile" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --tag "${IMAGE}" \
  "${PROJECT_ROOT}"

docker image inspect "${IMAGE}" --format '{{.Id}} {{.Architecture}}'
