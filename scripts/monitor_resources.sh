#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:?usage: monitor_resources.sh OUTPUT_PATH [CONTAINER_NAME] [INTERVAL_SECONDS]}"
CONTAINER_NAME="${2:-}"
INTERVAL_SECONDS="${3:-2}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "timestamp,gpu_name,gpu_util_percent,gpu_memory_available,gpu_memory_note,gpu_mem_util_percent,gpu_mem_used_mib,gpu_mem_total_mib,gpu_power_w,gpu_temp_c,load_1m,load_5m,load_15m,mem_used_mib,mem_total_mib,docker_cpu_percent,docker_mem_usage,docker_net_io,docker_block_io,docker_pids" >"${OUTPUT_PATH}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

normalize_metric() {
  local value
  value="$(trim "$1")"
  if [[ "${value}" == "[N/A]" || "${value}" == "N/A" ]]; then
    printf ''
    return
  fi
  printf '%s' "${value}"
}

while true; do
  timestamp="$(date --iso-8601=seconds)"

  gpu_row="$(
    nvidia-smi \
      --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true
  )"
  if [[ -z "${gpu_row}" ]]; then
    gpu_name=""
    gpu_util=""
    gpu_mem_util=""
    gpu_mem_used=""
    gpu_mem_total=""
    gpu_power=""
    gpu_temp=""
  else
    IFS=',' read -r gpu_name gpu_util gpu_mem_util gpu_mem_used gpu_mem_total gpu_power gpu_temp <<<"${gpu_row}"
    gpu_name="$(trim "${gpu_name}")"
    gpu_util="$(normalize_metric "${gpu_util}")"
    gpu_mem_util="$(normalize_metric "${gpu_mem_util}")"
    gpu_mem_used="$(normalize_metric "${gpu_mem_used}")"
    gpu_mem_total="$(normalize_metric "${gpu_mem_total}")"
    gpu_power="$(normalize_metric "${gpu_power}")"
    gpu_temp="$(normalize_metric "${gpu_temp}")"
  fi

  gpu_memory_available="true"
  gpu_memory_note=""
  if [[ -z "${gpu_mem_used}" || -z "${gpu_mem_total}" ]]; then
    gpu_memory_available="false"
    gpu_memory_note="gpu memory accounting unavailable from nvidia-smi on this platform"
    gpu_mem_util=""
    gpu_mem_used=""
    gpu_mem_total=""
  fi

  read -r load_1m load_5m load_15m _ < /proc/loadavg
  mem_total_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
  mem_available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  mem_total_mib="$((mem_total_kib / 1024))"
  mem_used_mib="$(((mem_total_kib - mem_available_kib) / 1024))"

  docker_row=",,,,"
  if [[ -n "${CONTAINER_NAME}" ]]; then
    docker_row="$(
      docker stats --no-stream \
        --format "{{.CPUPerc}},{{.MemUsage}},{{.NetIO}},{{.BlockIO}},{{.PIDs}}" \
        "${CONTAINER_NAME}" 2>/dev/null || true
    )"
    if [[ -z "${docker_row}" ]]; then
      docker_row=",,,,"
    fi
  fi

  echo "${timestamp},${gpu_name},${gpu_util},${gpu_memory_available},${gpu_memory_note},${gpu_mem_util},${gpu_mem_used},${gpu_mem_total},${gpu_power},${gpu_temp},${load_1m},${load_5m},${load_15m},${mem_used_mib},${mem_total_mib},${docker_row}" >>"${OUTPUT_PATH}"
  sleep "${INTERVAL_SECONDS}"
done
