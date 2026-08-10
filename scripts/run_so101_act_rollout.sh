#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-headless}"
shift || true
if [[ "${MODE}" != "headless" && "${MODE}" != "viewer" ]]; then
  echo "usage: $0 [headless|viewer] [rollout arguments...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FARPOINT_SO101_ROLLOUT_IMAGE:-farpoint-so101-act-rollout:0.4.4}"
BASE_IMAGE="${FARPOINT_SO101_IMAGE:-farpoint-so101-isaaclab:3.0-beta2}"
DATA_ROOT="${FARPOINT_DATA_ROOT:-${PROJECT_ROOT}/outputs}"
CHECKPOINT="${FARPOINT_ACT_CHECKPOINT:-}"
ASSET="${FARPOINT_SO101_ASSET:-${PROJECT_ROOT}/.cache/farpoint/assets/so101/ce807d99724cb65671abec01f908a2fcb4a6eab7/SO-ARM101-USD.usd}"
CONTAINER_ASSET=/workspace/farpoint-assets/SO-ARM101-USD.usd
CONTAINER_CHECKPOINT=/workspace/policy

if [[ -z "${CHECKPOINT}" || ! -s "${CHECKPOINT}/model.safetensors" ]]; then
  echo "FARPOINT_ACT_CHECKPOINT must name a complete pretrained_model directory" >&2
  exit 2
fi
if [[ ! -f "${ASSET}" ]]; then
  python3 "${PROJECT_ROOT}/scripts/fetch_so101_assets.py" --destination "${ASSET}"
fi
mkdir -p "${DATA_ROOT}"
ROLLOUT_IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
BASE_IMAGE_ID="$(docker image inspect "${BASE_IMAGE}" --format '{{.Id}}')"

docker_args=(
  --rm --gpus all --network host
  -w /workspace/project
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y
  -e FARPOINT_GIT_COMMIT="${FARPOINT_GIT_COMMIT:-unknown}"
  -e FARPOINT_ROLLOUT_IMAGE_ID="${ROLLOUT_IMAGE_ID}"
  -e FARPOINT_SO101_BASE_IMAGE_ID="${BASE_IMAGE_ID}"
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
  "$@"
)
printf -v rollout_command_q '%q ' "${rollout_command[@]}"

docker run "${docker_args[@]}" "${IMAGE}" "${rollout_command_q}"
