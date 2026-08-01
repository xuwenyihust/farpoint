#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_PATH="${1:?usage: run_remote_isaac_example.sh EXAMPLE_PATH ISAAC_IMAGE RUN_RUNTIME RUN_ID}"
ISAAC_IMAGE="${2:?usage: run_remote_isaac_example.sh EXAMPLE_PATH ISAAC_IMAGE RUN_RUNTIME RUN_ID}"
RUN_RUNTIME="${3:?usage: run_remote_isaac_example.sh EXAMPLE_PATH ISAAC_IMAGE RUN_RUNTIME RUN_ID}"
RUN_ID="${4:?usage: run_remote_isaac_example.sh EXAMPLE_PATH ISAAC_IMAGE RUN_RUNTIME RUN_ID}"
EPISODE_SEED="${5:-0}"
BENCHMARK_ID="${6:-}"
BENCHMARK_REPEAT="${7:-0}"
VARIATION_ID="${8:-${FARPOINT_VARIATION_ID:-}}"
POSITION_PLAN="${9:-${FARPOINT_POSITION_PLAN:-}}"
TRIAL_ID="${10:-${FARPOINT_TRIAL_ID:-}}"
RESERVE_INDEX="${11:-${FARPOINT_RESERVE_INDEX:-0}}"
if [[ -n "${POSITION_PLAN}" && "${POSITION_PLAN}" != /* ]]; then
  POSITION_PLAN="/workspace/project/${POSITION_PLAN}"
fi
RUN_TIMEOUT_SECONDS="${FARPOINT_RUN_TIMEOUT_SECONDS:-300}"
STARTUP_TIMEOUT_SECONDS="${FARPOINT_STARTUP_TIMEOUT_SECONDS:-120}"

CONTAINER_NAME="farpoint_${RUN_ID}"
EXAMPLE_NAME="$(basename "${EXAMPLE_PATH}")"
RESOURCE_DIR="outputs/episodes/_resources"
RESOURCE_LOG="${RESOURCE_DIR}/${EXAMPLE_NAME}_${RUN_ID}.csv"
RESOURCE_SUMMARY="${RESOURCE_DIR}/${EXAMPLE_NAME}_${RUN_ID}_summary.json"
PHASE_DIR="outputs/episodes/_phases"
RUN_PHASE_LOG="${PHASE_DIR}/${EXAMPLE_NAME}_${RUN_ID}_runner_phase_events.jsonl"
CONTAINER_OUTPUT_LOG="${RUN_RUNTIME}/container-output-${RUN_ID}.log"
MONITOR_PID=""
RUN_REGISTERED=0
RUN_FINISHED=0

record_phase() {
  local phase="$1"
  shift || true
  python3 - "$RUN_PHASE_LOG" "$phase" "$@" <<'PY'
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

stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
    record_phase "resource_monitor_stop_start"
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${MONITOR_PID}" >/dev/null 2>&1 || true
    record_phase "resource_monitor_stop_end"
  fi
  MONITOR_PID=""
}

stop_container() {
  if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    record_phase "docker_stop_start" container="${CONTAINER_NAME}"
    docker stop --timeout 10 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    record_phase "docker_stop_end" container="${CONTAINER_NAME}"
  fi
}

cleanup() {
  stop_monitor
  stop_container
  if (( RUN_REGISTERED )) && (( ! RUN_FINISHED )); then
    python3 scripts/data_platform_cli.py run-finish \
      --run-id "${RUN_ID}" \
      --status INCOMPLETE \
      --failure-reason "runner exited before completion" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

mkdir -p "${RESOURCE_DIR}" "${PHASE_DIR}"
python3 scripts/data_platform_cli.py run-start \
  --run-id "${RUN_ID}" \
  --task-name "${EXAMPLE_NAME}" \
  --seed "${EPISODE_SEED}" \
  --benchmark-id "${BENCHMARK_ID}" \
  --benchmark-repeat "${BENCHMARK_REPEAT}" >/dev/null
RUN_REGISTERED=1
record_phase "remote_runner_start" run_id="${RUN_ID}" example="${EXAMPLE_PATH}"
record_phase "runtime_lock_archive_start" runtime="${RUN_RUNTIME}"
docker run --rm --user 0:0 --entrypoint bash \
  -v "${RUN_RUNTIME}:/runtime:rw" \
  "${ISAAC_IMAGE}" \
  -lc '
    backup="/runtime/stale-locks/$1"
    mkdir -p "${backup}"
    archive_lock() {
      source_path="$1"
      target_name="$2"
      if [[ -e "${source_path}" ]]; then
        mv "${source_path}" "${backup}/${target_name}"
      fi
    }
    archive_lock /runtime/cache/ov/_cache.lock cache.lock
    archive_lock /runtime/hub/hub.lock hub.lock
    archive_lock /runtime/logs/omni.telemetry.transmitter.lock telemetry.lock
  ' _ "${RUN_ID}"
record_phase "runtime_lock_archive_end" runtime="${RUN_RUNTIME}"
record_phase "resource_monitor_start" output="${RESOURCE_LOG}"
bash scripts/monitor_resources.sh "${RESOURCE_LOG}" "${CONTAINER_NAME}" 2 &
MONITOR_PID="$!"
record_phase "resource_monitor_started" pid="${MONITOR_PID}"

set +e
record_phase "docker_run_start" container="${CONTAINER_NAME}" image="${ISAAC_IMAGE}"
timeout --signal=TERM --kill-after=20 "${RUN_TIMEOUT_SECONDS}" \
  docker run --name "${CONTAINER_NAME}" --rm --gpus all --network=host --entrypoint bash \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNI_ENV_PRIVACY_CONSENT=Y \
  -e PYTHONUNBUFFERED=1 \
  -e MPLCONFIGDIR=/isaac-sim/.cache/matplotlib \
  -e WARP_CACHE_PATH=/isaac-sim/.cache/warp \
  -e FARPOINT_EPISODE_SEED="${EPISODE_SEED}" \
  -e FARPOINT_VARIATION_ID="${VARIATION_ID}" \
  -e FARPOINT_POSITION_PLAN="${POSITION_PLAN}" \
  -e FARPOINT_TRIAL_ID="${TRIAL_ID}" \
  -e FARPOINT_RESERVE_INDEX="${RESERVE_INDEX}" \
  -e FARPOINT_FRAME_LIMIT="${FARPOINT_FRAME_LIMIT:-}" \
  -e FARPOINT_RUN_ID="${RUN_ID}" \
  -e FARPOINT_BENCHMARK_ID="${BENCHMARK_ID}" \
  -e FARPOINT_BENCHMARK_REPEAT="${BENCHMARK_REPEAT}" \
  -v "${RUN_RUNTIME}/cache:/isaac-sim/.cache:rw" \
  -v "${RUN_RUNTIME}/compute:/isaac-sim/.nv/ComputeCache:rw" \
  -v "${RUN_RUNTIME}/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
  -v "${RUN_RUNTIME}/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "${RUN_RUNTIME}/data:/isaac-sim/.local/share/ov/data:rw" \
  -v "${RUN_RUNTIME}/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
  -v "${RUN_RUNTIME}/hub:/var/cache/hub:rw" \
  -v "$(pwd):/workspace/project:rw" \
  "${ISAAC_IMAGE}" \
  -lc "./python.sh /workspace/project/${EXAMPLE_PATH}/scene.py" \
  2>&1 | tee "${CONTAINER_OUTPUT_LOG}" &
RUN_PIPE_PID="$!"
STARTUP_READY=0
STARTUP_TIMED_OUT=0
STARTUP_DEADLINE=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while kill -0 "${RUN_PIPE_PID}" >/dev/null 2>&1; do
  if grep -q "Simulation App Startup Complete" "${CONTAINER_OUTPUT_LOG}" 2>/dev/null; then
    STARTUP_READY=1
    record_phase "simulation_app_startup_ready"
    break
  fi
  if (( SECONDS >= STARTUP_DEADLINE )); then
    STARTUP_TIMED_OUT=1
    record_phase \
      "simulation_app_startup_timeout" \
      timeout_seconds="${STARTUP_TIMEOUT_SECONDS}"
    stop_container
    break
  fi
  sleep 2
done
wait "${RUN_PIPE_PID}"
STATUS=$?
if (( STARTUP_TIMED_OUT )); then
  STATUS=124
elif (( ! STARTUP_READY )) && grep -q \
  "Simulation App Startup Complete" \
  "${CONTAINER_OUTPUT_LOG}" 2>/dev/null; then
  STARTUP_READY=1
elif (( ! STARTUP_READY )) && (( STATUS == 0 )); then
  record_phase "simulation_app_startup_marker_missing"
  STATUS=125
fi
record_phase "docker_run_end" status="${STATUS}"
set -e

stop_container
stop_monitor
record_phase "resource_summary_start" input="${RESOURCE_LOG}" output="${RESOURCE_SUMMARY}"
python3 scripts/summarize_resources.py "${RESOURCE_LOG}" "${RESOURCE_SUMMARY}"
record_phase "resource_summary_end" output="${RESOURCE_SUMMARY}"

echo "Resource trace written: ${RESOURCE_LOG}"
record_phase "remote_runner_end" status="${STATUS}"
if (( STATUS == 0 )); then
  RUN_STATUS="PASS"
  RUN_FAILURE_REASON=""
else
  RUN_STATUS="FAIL"
  RUN_FAILURE_REASON="remote runner exited with status ${STATUS}"
fi
python3 scripts/data_platform_cli.py run-finish \
  --run-id "${RUN_ID}" \
  --status "${RUN_STATUS}" \
  --return-code "${STATUS}" \
  --failure-reason "${RUN_FAILURE_REASON}" >/dev/null
RUN_FINISHED=1
exit "${STATUS}"
