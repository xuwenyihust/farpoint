#!/usr/bin/env bash
set -euo pipefail

# Sync only source/configuration to DGX Spark; generated datasets and caches stay remote.
HOST="${FARPOINT_DGX_HOST:-dgx-spark}"
REMOTE_ROOT="${FARPOINT_DGX_ROOT:-/home/wenyixu/projects/farpoint}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -az \
  --exclude '.git' \
  --exclude '.cache' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  --exclude 'outputs' \
  --exclude 'artifacts' \
  "${PROJECT_ROOT}/" "${HOST}:${REMOTE_ROOT}/"

ssh "${HOST}" "cd '${REMOTE_ROOT}' && git status --short || true"
echo "Synced SO-101 source to ${HOST}:${REMOTE_ROOT}"
