#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-headless}"
shift || true
if [[ "${MODE}" != "headless" && "${MODE}" != "viewer" ]]; then
  echo "usage: $0 [headless|viewer] [collector arguments...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FARPOINT_SO101_IMAGE:-farpoint-so101-isaaclab:3.0-beta2}"
DATA_ROOT="${FARPOINT_DATA_ROOT:-${PROJECT_ROOT}/outputs}"
ASSET="${PROJECT_ROOT}/.cache/farpoint/assets/so101/ce807d99724cb65671abec01f908a2fcb4a6eab7/SO-ARM101-USD.usd"
if [[ ! -f "${ASSET}" ]]; then
  python3 "${PROJECT_ROOT}/scripts/fetch_so101_assets.py" --destination "${ASSET}"
fi
mkdir -p "${DATA_ROOT}"

docker_args=(
  --rm --gpus all --network host
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y
  -e FARPOINT_GIT_COMMIT="${FARPOINT_GIT_COMMIT:-unknown}"
  -e FARPOINT_SO101_USD=/workspace/project/.cache/farpoint/assets/so101/ce807d99724cb65671abec01f908a2fcb4a6eab7/SO-ARM101-USD.usd
  -v "${PROJECT_ROOT}:/workspace/project:rw"
  -v "${DATA_ROOT}:/workspace/farpoint-data:rw"
)
if [[ "${MODE}" == "viewer" ]]; then
  docker_args+=(-e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi

launcher_args=""
if [[ "${MODE}" == "headless" ]]; then
  # Isaac Lab 3.0 requires this explicit switch whenever a camera is spawned,
  # including headless RGB capture.
  launcher_args=" --enable_cameras"
fi

docker run "${docker_args[@]}" "${IMAGE}" \
  "/workspace/IsaacLab/isaaclab.sh -p /workspace/project/examples/isaaclab_so101_pick_place/collect.py --mode ${MODE}${launcher_args} $*"
