#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_PATH="${1:-examples/isaac_cube_scene}"
DGX_HOST="${FARPOINT_REMOTE_HOST:-${DGX_SPARK_HOST:-}}"
DGX_HOSTNAME_OVERRIDE="${FARPOINT_REMOTE_HOSTNAME:-${DGX_SPARK_HOSTNAME:-}}"
DGX_HOST_KEY_ALIAS="${FARPOINT_REMOTE_KEY_ALIAS:-${DGX_SPARK_HOST_KEY_ALIAS:-}}"
REMOTE_ROOT="${FARPOINT_REMOTE_ROOT:-${HOME}/farpoint}"
REMOTE_RUNTIME="${FARPOINT_REMOTE_RUNTIME:-${HOME}/.cache/farpoint/isaac-sim}"
ISAAC_IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
EPISODE_SEED="${FARPOINT_EPISODE_SEED:-0}"
VARIATION_ID="${FARPOINT_VARIATION_ID:-}"
FRAME_LIMIT="${FARPOINT_FRAME_LIMIT:-}"
BENCHMARK_ID="${FARPOINT_BENCHMARK_ID:-}"
BENCHMARK_REPEAT="${FARPOINT_BENCHMARK_REPEAT:-0}"
RUN_TIMEOUT_SECONDS="${FARPOINT_RUN_TIMEOUT_SECONDS:-300}"
STARTUP_TIMEOUT_SECONDS="${FARPOINT_STARTUP_TIMEOUT_SECONDS:-120}"
SYNC_MODE="${FARPOINT_SYNC_MODE:-summary}"
DASHBOARD_PORT="${FARPOINT_DASHBOARD_PORT:-8765}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
REMOTE_RUN_RUNTIME="${REMOTE_RUNTIME}"
SSH_OPTIONS=()
RSYNC_SSH_COMMAND="ssh"
if [[ -z "${DGX_HOST}" ]]; then
  echo "Set FARPOINT_REMOTE_HOST to the GPU host before running an Isaac Sim example." >&2
  exit 2
fi
if [[ -n "${DGX_HOSTNAME_OVERRIDE}" ]]; then
  if [[ ! "${DGX_HOSTNAME_OVERRIDE}" =~ ^[a-zA-Z0-9._:-]+$ ]]; then
    echo "Invalid DGX_SPARK_HOSTNAME: ${DGX_HOSTNAME_OVERRIDE}" >&2
    exit 2
  fi
  if [[ ! "${DGX_HOST_KEY_ALIAS}" =~ ^[a-zA-Z0-9._:-]+$ ]]; then
    echo "Invalid DGX_SPARK_HOST_KEY_ALIAS: ${DGX_HOST_KEY_ALIAS}" >&2
    exit 2
  fi
  SSH_OPTIONS=(
    -o "HostName=${DGX_HOSTNAME_OVERRIDE}"
    -o "HostKeyAlias=${DGX_HOST_KEY_ALIAS}"
  )
  RSYNC_SSH_COMMAND="ssh -o HostName=${DGX_HOSTNAME_OVERRIDE} -o HostKeyAlias=${DGX_HOST_KEY_ALIAS}"
fi
if [[ -n "${BENCHMARK_ID}" ]]; then
  REMOTE_RUN_RUNTIME="${REMOTE_RUNTIME}/benchmark-runs/${RUN_ID}"
fi

case "${EXAMPLE_PATH}" in
  examples/isaac_cube_scene|examples/isaac_tabletop_scene|examples/isaac_robot_arm_scene|examples/isaac_ur10e_robotiq_scene|examples/isaac_perception_contact_scene)
    ;;
  *)
  echo "Unsupported example: ${EXAMPLE_PATH}" >&2
  echo "Supported examples: examples/isaac_cube_scene, examples/isaac_tabletop_scene, examples/isaac_robot_arm_scene, examples/isaac_ur10e_robotiq_scene, examples/isaac_perception_contact_scene" >&2
  exit 2
    ;;
esac

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE_NAME="$(basename "${EXAMPLE_PATH}")"
LOCAL_PHASE_DIR="${LOCAL_ROOT}/outputs/episodes/_phases"
LOCAL_PHASE_LOG="${LOCAL_PHASE_DIR}/${EXAMPLE_NAME}_${RUN_ID}_local_phase_events.jsonl"

record_local_phase() {
  local phase="$1"
  shift || true
  mkdir -p "${LOCAL_PHASE_DIR}"
  python3 - "$LOCAL_PHASE_LOG" "$phase" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
phase = sys.argv[2]
fields = {}
for item in sys.argv[3:]:
    key, _, value = item.partition("=")
    fields[key] = value
payload = {
    "time": datetime.now(timezone.utc).isoformat(),
    "phase": phase,
    **fields,
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
PY
}

echo "Farpoint"
echo "Example: ${EXAMPLE_PATH}"
echo "Remote host: ${DGX_HOST}"
if [[ -n "${DGX_HOSTNAME_OVERRIDE}" ]]; then
  echo "Remote hostname override: ${DGX_HOSTNAME_OVERRIDE}"
fi
echo "Remote root: ${REMOTE_ROOT}"
echo "Isaac image: ${ISAAC_IMAGE}"
echo "Episode seed: ${EPISODE_SEED}"
if [[ -n "${BENCHMARK_ID}" ]]; then
  echo "Benchmark: ${BENCHMARK_ID}"
fi

record_local_phase "local_runner_start" run_id="${RUN_ID}" example="${EXAMPLE_PATH}"
record_local_phase "remote_prepare_start" host="${DGX_HOST}"
ssh "${SSH_OPTIONS[@]}" "${DGX_HOST}" "mkdir -p '${REMOTE_ROOT}' '${REMOTE_RUN_RUNTIME}/cache' '${REMOTE_RUN_RUNTIME}/compute' '${REMOTE_RUN_RUNTIME}/config' '${REMOTE_RUN_RUNTIME}/data' '${REMOTE_RUN_RUNTIME}/logs' '${REMOTE_RUN_RUNTIME}/pkg' '${REMOTE_RUN_RUNTIME}/hub' && chmod a+rwx '${REMOTE_RUN_RUNTIME}' '${REMOTE_RUN_RUNTIME}/cache' '${REMOTE_RUN_RUNTIME}/compute' '${REMOTE_RUN_RUNTIME}/config' '${REMOTE_RUN_RUNTIME}/data' '${REMOTE_RUN_RUNTIME}/logs' '${REMOTE_RUN_RUNTIME}/pkg' '${REMOTE_RUN_RUNTIME}/hub'"
record_local_phase "remote_prepare_end" host="${DGX_HOST}"

record_local_phase "project_sync_start" host="${DGX_HOST}" remote_root="${REMOTE_ROOT}"
rsync -az -e "${RSYNC_SSH_COMMAND}" \
  --exclude ".git" \
  --exclude ".generated-skills" \
  --exclude "outputs" \
  --exclude ".codex" \
  --exclude "__pycache__" \
  "${LOCAL_ROOT}/README.md" \
  "${LOCAL_ROOT}/docs" \
  "${LOCAL_ROOT}/examples" \
  "${LOCAL_ROOT}/configs" \
  "${LOCAL_ROOT}/scripts" \
  "${LOCAL_ROOT}/src" \
  "${DGX_HOST}:${REMOTE_ROOT}/"
record_local_phase "project_sync_end" host="${DGX_HOST}" remote_root="${REMOTE_ROOT}"

REMOTE_LOG_DIR="${REMOTE_ROOT}/outputs/episodes/_logs"
REMOTE_RESOURCE_DIR="${REMOTE_ROOT}/outputs/episodes/_resources"
ssh "${SSH_OPTIONS[@]}" "${DGX_HOST}" "mkdir -p '${REMOTE_ROOT}/outputs/episodes' '${REMOTE_LOG_DIR}' '${REMOTE_RESOURCE_DIR}' && chmod a+rwx '${REMOTE_ROOT}/outputs' '${REMOTE_ROOT}/outputs/episodes' '${REMOTE_LOG_DIR}' '${REMOTE_RESOURCE_DIR}'"

REMOTE_LOG="${REMOTE_LOG_DIR}/${EXAMPLE_NAME}_${RUN_ID}.log"

set +e
record_local_phase "remote_example_start" host="${DGX_HOST}" log="${REMOTE_LOG}"
ssh "${SSH_OPTIONS[@]}" "${DGX_HOST}" "bash -lc 'cd \"${REMOTE_ROOT}\" && set -o pipefail && FARPOINT_RUN_TIMEOUT_SECONDS=\"${RUN_TIMEOUT_SECONDS}\" FARPOINT_STARTUP_TIMEOUT_SECONDS=\"${STARTUP_TIMEOUT_SECONDS}\" FARPOINT_VARIATION_ID=\"${VARIATION_ID}\" FARPOINT_FRAME_LIMIT=\"${FRAME_LIMIT}\" bash scripts/run_remote_isaac_example.sh \"${EXAMPLE_PATH}\" \"${ISAAC_IMAGE}\" \"${REMOTE_RUN_RUNTIME}\" \"${RUN_ID}\" \"${EPISODE_SEED}\" \"${BENCHMARK_ID}\" \"${BENCHMARK_REPEAT}\" \"${VARIATION_ID}\" 2>&1 | tee \"${REMOTE_LOG}\"'"
STATUS=$?
record_local_phase "remote_example_end" host="${DGX_HOST}" status="${STATUS}"
set -e

if [[ "${SYNC_MODE}" == "summary" ]]; then
  mkdir -p "${LOCAL_ROOT}/outputs/episodes"
  record_local_phase "summary_sync_start" host="${DGX_HOST}"
  rsync -az --prune-empty-dirs -e "${RSYNC_SSH_COMMAND}" \
    --include "episode_*/" \
    --include "episode_*/metadata.json" \
    --include "episode_*/metrics.json" \
    --exclude "*" \
    "${DGX_HOST}:${REMOTE_ROOT}/outputs/episodes/" \
    "${LOCAL_ROOT}/outputs/episodes/"
  record_local_phase "summary_sync_end" host="${DGX_HOST}"
elif [[ "${SYNC_MODE}" != "none" ]]; then
  echo "Unsupported FARPOINT_SYNC_MODE: ${SYNC_MODE}; use summary or none" >&2
  exit 2
fi

if [[ "${STATUS}" -ne 0 ]]; then
  echo "Example failed with status ${STATUS}. Artifacts remain on the remote GPU host." >&2
  exit "${STATUS}"
fi

if ! ssh "${SSH_OPTIONS[@]}" "${DGX_HOST}" \
  "grep -q 'SMOKE_TEST_RESULT: PASS' '${REMOTE_LOG}'"; then
  echo "Example completed, but PASS marker was not found in the DGX log." >&2
  exit 1
fi

echo "Example completed successfully."
echo "Authoritative artifacts: ${DGX_HOST}:${REMOTE_ROOT}/outputs"
echo "Dashboard: http://${DGX_HOST}:${DASHBOARD_PORT}/"
record_local_phase "local_runner_end" status="${STATUS}"
