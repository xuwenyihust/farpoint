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
SIMULATOR_IMAGE_DIGEST="${FARPOINT_SIMULATOR_IMAGE_DIGEST:-$(docker image inspect --format '{{.Id}}' "${IMAGE}")}"
if [[ ! "${SIMULATOR_IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "invalid simulator image digest: ${SIMULATOR_IMAGE_DIGEST}" >&2
  exit 2
fi
DATA_ROOT="${FARPOINT_DATA_ROOT:-${PROJECT_ROOT}/outputs}"
ASSET="${FARPOINT_SO101_ASSET:-${PROJECT_ROOT}/.cache/farpoint/assets/so101/ce807d99724cb65671abec01f908a2fcb4a6eab7/SO-ARM101-USD.usd}"
CONTAINER_ASSET=/workspace/farpoint-assets/SO-ARM101-USD.usd
if [[ ! -f "${ASSET}" ]]; then
  python3 "${PROJECT_ROOT}/scripts/fetch_so101_assets.py" --destination "${ASSET}"
fi
mkdir -p "${DATA_ROOT}"

# Workflow status reports host-absolute evidence paths. Translate only paths
# inside the explicitly mounted data root; other arguments remain byte-for-byte
# unchanged. This makes the emitted COLLECT command directly executable.
collector_user_args=()
for arg in "$@"; do
  if [[ "${arg}" == "${DATA_ROOT}" ]]; then
    collector_user_args+=(/workspace/farpoint-data)
  elif [[ "${arg}" == "${DATA_ROOT}/"* ]]; then
    collector_user_args+=("/workspace/farpoint-data/${arg#"${DATA_ROOT}/"}")
  else
    collector_user_args+=("${arg}")
  fi
done

docker_args=(
  --rm --gpus all --network host
  -w /workspace/project
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y
  -e FARPOINT_GIT_COMMIT="${FARPOINT_GIT_COMMIT:-unknown}"
  -e FARPOINT_SIMULATOR_IMAGE_DIGEST="${SIMULATOR_IMAGE_DIGEST}"
  -e FARPOINT_SO101_USD="${CONTAINER_ASSET}"
  -v "${PROJECT_ROOT}:/workspace/project:rw"
  -v "${DATA_ROOT}:/workspace/farpoint-data:rw"
  -v "${ASSET}:${CONTAINER_ASSET}:ro"
)
if [[ "${MODE}" == "viewer" ]]; then
  docker_args+=(-e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi

has_option() {
  local expected="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${expected}" || "${arg}" == "${expected}="* ]]; then
      return 0
    fi
  done
  return 1
}

launcher_args=()
if ! has_option --enable_cameras "${collector_user_args[@]}"; then
  # Isaac Lab 3.0 requires this explicit switch whenever a camera is spawned,
  # including headless RGB capture and WebRTC inspection.
  launcher_args+=(--enable_cameras)
fi

if [[ "${MODE}" == "viewer" ]]; then
  livestream=""
  previous=""
  for arg in "${collector_user_args[@]}"; do
    if [[ "${previous}" == "--livestream" ]]; then
      livestream="${arg}"
      previous=""
      continue
    fi
    case "${arg}" in
      --livestream)
        previous="--livestream"
        ;;
      --livestream=*)
        livestream="${arg#*=}"
        ;;
    esac
  done
  if [[ -z "${livestream}" ]]; then
    livestream="2"
    launcher_args+=(--livestream "${livestream}")
  fi

  # KitVisualizer pumps app updates itself.  That prevents the RTX sensor
  # synchronizer from waiting 30 seconds per control step during WebRTC runs.
  # Match the renderer and negotiated stream sizes so frames are accepted.
  if [[ "${livestream}" == "1" || "${livestream}" == "2" ]]; then
    if ! has_option --visualizer "${collector_user_args[@]}"; then
      launcher_args+=(--visualizer kit)
    fi
    if ! has_option --kit_args "${collector_user_args[@]}"; then
      launcher_args+=(
        --kit_args
        "--/app/window/width=1280 --/app/window/height=720 --no-window"
      )
    fi
  fi
fi

collector_command=(
  /workspace/IsaacLab/isaaclab.sh
  -p
  /workspace/project/examples/isaaclab_so101_pick_place/collect.py
  --mode "${MODE}"
  "${launcher_args[@]}"
  "${collector_user_args[@]}"
)
printf -v collector_command_q '%q ' "${collector_command[@]}"

docker run "${docker_args[@]}" "${IMAGE}" \
  "${collector_command_q}"
