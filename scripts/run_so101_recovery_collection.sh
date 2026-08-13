#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-headless}"
shift || true
if [[ "${MODE}" != "headless" && "${MODE}" != "viewer" ]]; then
  echo "usage: $0 [headless|viewer] [collector arguments...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${FARPOINT_DATA_ROOT:-${PROJECT_ROOT}/outputs}"
POLICY_IMAGE="${FARPOINT_SO101_TRAINING_IMAGE:-farpoint-so101-lerobot-training:0.4.4}"
CHECKPOINT="${FARPOINT_ACT_CHECKPOINT:-}"
POLICY_CACHE="${FARPOINT_POLICY_CACHE_ROOT:-${PROJECT_ROOT}/.cache/farpoint/act-policy}"
POLICY_PORT="${FARPOINT_ACT_POLICY_PORT:-8766}"
GIT_COMMIT="${FARPOINT_GIT_COMMIT:-}"
COLLECTOR_ARGS=("$@")
RUNTIME_ARGUMENT=""
for ((index = 0; index < ${#COLLECTOR_ARGS[@]}; index++)); do
  if [[ "${COLLECTOR_ARGS[index]}" == "--recovery-runtime" ]]; then
    RUNTIME_ARGUMENT="${COLLECTOR_ARGS[index + 1]:-}"
  fi
done
if [[ ! "${GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FARPOINT_GIT_COMMIT must identify the exact source commit" >&2
  exit 2
fi
if [[ -z "${CHECKPOINT}" || ! -s "${CHECKPOINT}/model.safetensors" ]]; then
  echo "FARPOINT_ACT_CHECKPOINT must name a complete pretrained_model directory" >&2
  exit 2
fi
case "${RUNTIME_ARGUMENT}" in
  "${DATA_ROOT}"/*) HOST_RUNTIME="${RUNTIME_ARGUMENT}" ;;
  /workspace/farpoint-data/*) HOST_RUNTIME="${DATA_ROOT}/${RUNTIME_ARGUMENT#/workspace/farpoint-data/}" ;;
  *) echo "--recovery-runtime must be below FARPOINT_DATA_ROOT" >&2; exit 2 ;;
esac
if [[ ! -s "${HOST_RUNTIME}" ]]; then
  echo "recovery runtime does not exist: ${HOST_RUNTIME}" >&2
  exit 2
fi
REPLAN_INTERVAL_STEPS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["control"]["replan_interval_steps"])' "${HOST_RUNTIME}")"
MODEL_SHA256="$(sha256sum "${CHECKPOINT}/model.safetensors" | awk '{print $1}')"
EXPECTED_MODEL_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_policy"]["model_sha256"])' "${HOST_RUNTIME}")"
if [[ "${MODEL_SHA256}" != "${EXPECTED_MODEL_SHA256}" ]]; then
  echo "checkpoint does not match frozen recovery runtime" >&2
  exit 2
fi
if curl --silent --fail "http://127.0.0.1:${POLICY_PORT}/health" >/dev/null 2>&1; then
  echo "policy port ${POLICY_PORT} is already in use" >&2
  exit 2
fi

mkdir -p "${POLICY_CACHE}"
POLICY_IMAGE_ID="$(docker image inspect "${POLICY_IMAGE}" --format '{{.Id}}')"
POLICY_CONTAINER="farpoint-recovery-policy-${GIT_COMMIT:0:12}-$$"
cleanup() {
  status=$?
  trap - EXIT
  docker logs "${POLICY_CONTAINER}" >&2 2>/dev/null || true
  docker stop "${POLICY_CONTAINER}" >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT

docker run -d --rm --gpus all --ipc=host --network host \
  --name "${POLICY_CONTAINER}" \
  --user "$(id -u):$(id -g)" \
  --workdir /workspace/project \
  --env HOME=/workspace/cache/home \
  --env HF_HOME=/workspace/cache/huggingface \
  --env PYTHONPATH=/workspace/project/src \
  --env PYTHONUNBUFFERED=1 \
  --env FARPOINT_POLICY_IMAGE_ID="${POLICY_IMAGE_ID}" \
  --volume "${PROJECT_ROOT}:/workspace/project:ro" \
  --volume "${CHECKPOINT}:/workspace/policy:ro" \
  --volume "${POLICY_CACHE}:/workspace/cache:rw" \
  "${POLICY_IMAGE}" \
  python /workspace/project/scripts/serve_so101_act_policy.py \
  --checkpoint /workspace/policy \
  --expected-model-sha256 "${MODEL_SHA256}" \
  --port "${POLICY_PORT}" \
  --replan-interval-steps "${REPLAN_INTERVAL_STEPS}" >/dev/null

policy_ready=false
for _ in $(seq 1 120); do
  if curl --silent --fail "http://127.0.0.1:${POLICY_PORT}/health" >/dev/null 2>&1; then
    policy_ready=true
    break
  fi
  if ! docker inspect "${POLICY_CONTAINER}" >/dev/null 2>&1; then
    echo "ACT policy server exited during startup" >&2
    exit 1
  fi
  sleep 1
done
if [[ "${policy_ready}" != true ]]; then
  echo "ACT policy server did not become ready within 120 seconds" >&2
  exit 1
fi

export FARPOINT_ACT_POLICY_URL="http://127.0.0.1:${POLICY_PORT}"
"${PROJECT_ROOT}/scripts/run_so101_isaaclab.sh" "${MODE}" "${COLLECTOR_ARGS[@]}"
