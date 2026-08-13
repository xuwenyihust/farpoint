#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-headless}"
shift || true
if [[ "${MODE}" != "headless" && "${MODE}" != "viewer" ]]; then
  echo "usage: $0 [headless|viewer] [rollout arguments...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_IMAGE="${FARPOINT_SO101_IMAGE:-farpoint-so101-isaaclab:3.0-beta2}"
POLICY_IMAGE="${FARPOINT_SO101_TRAINING_IMAGE:-farpoint-so101-lerobot-training:0.4.4}"
DATA_ROOT="${FARPOINT_DATA_ROOT:-${PROJECT_ROOT}/outputs}"
POLICY_CACHE="${FARPOINT_POLICY_CACHE_ROOT:-${PROJECT_ROOT}/.cache/farpoint/act-policy}"
CHECKPOINT="${FARPOINT_ACT_CHECKPOINT:-}"
ASSET="${FARPOINT_SO101_ASSET:-${PROJECT_ROOT}/.cache/farpoint/assets/so101/ce807d99724cb65671abec01f908a2fcb4a6eab7/SO-ARM101-USD.usd}"
POLICY_PORT="${FARPOINT_ACT_POLICY_PORT:-8766}"
CONTAINER_ASSET=/workspace/farpoint-assets/SO-ARM101-USD.usd
CONTAINER_CHECKPOINT=/workspace/policy
GIT_COMMIT="${FARPOINT_GIT_COMMIT:-}"
ROLLOUT_ARGUMENTS=("$@")
CONTAINER_OUTPUT_ROOT=""
CONTAINER_SPEC=""

for ((argument_index = 0; argument_index < ${#ROLLOUT_ARGUMENTS[@]}; argument_index++)); do
  if [[ "${ROLLOUT_ARGUMENTS[argument_index]}" == "--output-root" ]]; then
    CONTAINER_OUTPUT_ROOT="${ROLLOUT_ARGUMENTS[argument_index + 1]:-}"
  elif [[ "${ROLLOUT_ARGUMENTS[argument_index]}" == "--spec" ]]; then
    CONTAINER_SPEC="${ROLLOUT_ARGUMENTS[argument_index + 1]:-}"
  fi
done

if [[ ! "${GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FARPOINT_GIT_COMMIT must identify the exact 40-character source commit" >&2
  exit 2
fi
if [[ -z "${CHECKPOINT}" || ! -s "${CHECKPOINT}/model.safetensors" ]]; then
  echo "FARPOINT_ACT_CHECKPOINT must name a complete pretrained_model directory" >&2
  exit 2
fi
if [[ "${CONTAINER_OUTPUT_ROOT}" != /workspace/farpoint-data/* ]]; then
  echo "--output-root must be below /workspace/farpoint-data" >&2
  exit 2
fi
HOST_OUTPUT_ROOT="${DATA_ROOT}/${CONTAINER_OUTPUT_ROOT#/workspace/farpoint-data/}"
case "${CONTAINER_SPEC}" in
  /workspace/project/*)
    HOST_SPEC="${PROJECT_ROOT}/${CONTAINER_SPEC#/workspace/project/}"
    ;;
  /workspace/farpoint-data/*)
    HOST_SPEC="${DATA_ROOT}/${CONTAINER_SPEC#/workspace/farpoint-data/}"
    ;;
  *)
    echo "--spec must be below /workspace/project or /workspace/farpoint-data" >&2
    exit 2
    ;;
esac
if [[ ! -s "${HOST_SPEC}" ]]; then
  echo "rollout spec does not exist: ${HOST_SPEC}" >&2
  exit 2
fi
REPLAN_INTERVAL_STEPS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["control"].get("replan_interval_steps", ""))' "${HOST_SPEC}")"
POLICY_REPLAN_ARGS=()
if [[ -n "${REPLAN_INTERVAL_STEPS}" ]]; then
  POLICY_REPLAN_ARGS+=(--replan-interval-steps "${REPLAN_INTERVAL_STEPS}")
fi
REPLAY_MANIFEST="${FARPOINT_ACTION_REPLAY_MANIFEST:-}"
POLICY_REPLAY_ARGS=()
POLICY_REPLAY_MOUNT=()
if [[ -n "${REPLAY_MANIFEST}" ]]; then
  case "${REPLAY_MANIFEST}" in
    "${DATA_ROOT}"/*) ;;
    *) echo "FARPOINT_ACTION_REPLAY_MANIFEST must be below FARPOINT_DATA_ROOT" >&2; exit 2 ;;
  esac
  if [[ ! -s "${REPLAY_MANIFEST}" ]]; then
    echo "expert action replay manifest does not exist: ${REPLAY_MANIFEST}" >&2
    exit 2
  fi
  CONTAINER_REPLAY_MANIFEST="/workspace/farpoint-data/${REPLAY_MANIFEST#${DATA_ROOT}/}"
  POLICY_REPLAY_ARGS+=(--replay-manifest "${CONTAINER_REPLAY_MANIFEST}")
  POLICY_REPLAY_MOUNT+=(--volume "${DATA_ROOT}:/workspace/farpoint-data:ro")
fi
if [[ ! -f "${ASSET}" ]]; then
  python3 "${PROJECT_ROOT}/scripts/fetch_so101_assets.py" --destination "${ASSET}"
fi
if curl --silent --fail "http://127.0.0.1:${POLICY_PORT}/health" >/dev/null 2>&1; then
  echo "policy port ${POLICY_PORT} is already in use" >&2
  exit 2
fi

mkdir -p "${DATA_ROOT}" "${POLICY_CACHE}"
ISAAC_IMAGE_ID="$(docker image inspect "${ISAAC_IMAGE}" --format '{{.Id}}')"
POLICY_IMAGE_ID="$(docker image inspect "${POLICY_IMAGE}" --format '{{.Id}}')"
MODEL_SHA256="$(sha256sum "${CHECKPOINT}/model.safetensors" | awk '{print $1}')"
POLICY_CONTAINER="farpoint-act-policy-${GIT_COMMIT:0:12}-$$"

cleanup() {
  status=$?
  trap - EXIT
  docker logs "${POLICY_CONTAINER}" >&2 2>/dev/null || true
  docker stop "${POLICY_CONTAINER}" >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT

policy_docker_args=(
  -d --rm --gpus all --ipc=host --network host
  --name "${POLICY_CONTAINER}"
  --user "$(id -u):$(id -g)"
  --workdir /workspace/project
  --env HOME=/workspace/cache/home
  --env HF_HOME=/workspace/cache/huggingface
  --env PYTHONPATH=/workspace/project/src
  --env PYTHONUNBUFFERED=1
  --env FARPOINT_POLICY_IMAGE_ID="${POLICY_IMAGE_ID}"
  --volume "${PROJECT_ROOT}:/workspace/project:ro"
  --volume "${CHECKPOINT}:${CONTAINER_CHECKPOINT}:ro"
  --volume "${POLICY_CACHE}:/workspace/cache:rw"
)
if (( ${#POLICY_REPLAY_MOUNT[@]} )); then
  policy_docker_args+=("${POLICY_REPLAY_MOUNT[@]}")
fi
policy_docker_args+=(
  "${POLICY_IMAGE}"
  python /workspace/project/scripts/serve_so101_act_policy.py
  --checkpoint "${CONTAINER_CHECKPOINT}"
  --expected-model-sha256 "${MODEL_SHA256}"
  --port "${POLICY_PORT}"
)
if (( ${#POLICY_REPLAN_ARGS[@]} )); then
  policy_docker_args+=("${POLICY_REPLAN_ARGS[@]}")
fi
if (( ${#POLICY_REPLAY_ARGS[@]} )); then
  policy_docker_args+=("${POLICY_REPLAY_ARGS[@]}")
fi
docker run "${policy_docker_args[@]}" >/dev/null

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

docker_args=(
  --rm --gpus all --network host
  -w /workspace/project
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y
  -e FARPOINT_GIT_COMMIT="${GIT_COMMIT}"
  -e FARPOINT_ISAAC_IMAGE_ID="${ISAAC_IMAGE_ID}"
  -e FARPOINT_POLICY_IMAGE_ID="${POLICY_IMAGE_ID}"
  -e FARPOINT_SO101_BASE_IMAGE_ID="${ISAAC_IMAGE_ID}"
  -e FARPOINT_ACT_POLICY_URL="http://127.0.0.1:${POLICY_PORT}"
  -e FARPOINT_SO101_USD="${CONTAINER_ASSET}"
  -v "${PROJECT_ROOT}:/workspace/project:ro"
  -v "${DATA_ROOT}:/workspace/farpoint-data:rw"
  -v "${ASSET}:${CONTAINER_ASSET}:ro"
  -v "${CHECKPOINT}:${CONTAINER_CHECKPOINT}:ro"
)
if [[ "${MODE}" == "viewer" ]]; then
  docker_args+=(-e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi

launcher_args=(--enable_cameras)
if [[ "${MODE}" == "viewer" ]]; then
  launcher_args+=(
    --livestream 2
    --visualizer kit
    --kit_args "--/app/window/width=1280 --/app/window/height=720 --no-window"
  )
fi
rollout_command=(
  /workspace/IsaacLab/isaaclab.sh
  -p
  /workspace/project/examples/isaaclab_so101_pick_place/rollout.py
  --mode "${MODE}"
  --checkpoint "${CONTAINER_CHECKPOINT}"
  "${launcher_args[@]}"
  "${ROLLOUT_ARGUMENTS[@]}"
)
printf -v rollout_command_q '%q ' "${rollout_command[@]}"

set +e
docker run "${docker_args[@]}" "${ISAAC_IMAGE}" "${rollout_command_q}"
rollout_status=$?
set -e
if [[ ! -s "${HOST_OUTPUT_ROOT}/report.json" ]]; then
  echo "rollout did not produce report.json" >&2
  rollout_status=1
else
  report_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${HOST_OUTPUT_ROOT}/report.json")"
  if [[ "${report_status}" != "PASS" ]]; then
    rollout_status=2
  fi
fi
echo "FARPOINT_ACT_ROLLOUT_EXIT status=${rollout_status}" >&2
exit "${rollout_status}"
