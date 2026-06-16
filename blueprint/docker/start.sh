#!/usr/bin/env bash
set -euo pipefail

: "${ATTRIBUTION_DB:=/data/attributions.db}"
: "${ATTRIBUTION_SERVICE_PORT:=8000}"
: "${ATTRIBUTION_WEB_PORT:=8080}"

service_pid=""
web_pid=""
db_dir="$(dirname "${ATTRIBUTION_DB}")"

shutdown() {
  local pids=()
  [[ -n "${web_pid}" ]] && pids+=("${web_pid}")
  [[ -n "${service_pid}" ]] && pids+=("${service_pid}")

  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

trap shutdown INT TERM EXIT

mkdir -p "${db_dir}"

if [[ ! -f "${ATTRIBUTION_DB}" ]]; then
  uv run --project /opt/blueprint --frozen python -c \
    "import os; from attribution_blueprint.connectors.outputs.sqlite import migrate; migrate(os.environ['ATTRIBUTION_DB'])"
fi
db_owner="$(stat -c '%u:%g' "${db_dir}")"
chown "${db_owner}" "${ATTRIBUTION_DB}" "${ATTRIBUTION_DB}-wal" "${ATTRIBUTION_DB}-shm" 2>/dev/null || true
chmod 755 "${db_dir}" 2>/dev/null || true
chmod 644 "${ATTRIBUTION_DB}" "${ATTRIBUTION_DB}-wal" "${ATTRIBUTION_DB}-shm" 2>/dev/null || true

/opt/attribution-service/.venv/bin/uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${ATTRIBUTION_SERVICE_PORT}" &
service_pid="$!"

service_healthy=0
for attempt in $(seq 0 60); do
  if ! kill -0 "${service_pid}" 2>/dev/null; then
    service_status=0
    wait "${service_pid}" 2>/dev/null || service_status=$?
    if [[ "${service_status}" -eq 0 ]]; then
      service_status=1
    fi
    echo "attribution service exited before becoming healthy (status=${service_status})" >&2
    exit "${service_status}"
  fi
  if uv run --project /opt/blueprint --frozen python -c \
    "import os; import urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['ATTRIBUTION_SERVICE_PORT']}/v2/healthz\", timeout=1)" \
    >/dev/null 2>&1; then
    service_healthy=1
    break
  fi
  if [[ "${attempt}" -lt 60 ]]; then
    sleep 1
  fi
done

if [[ "${service_healthy}" -ne 1 ]]; then
  echo "attribution service did not become healthy within 60 seconds" >&2
  exit 1
fi

/opt/blueprint/.venv/bin/nvidia-attribution-blueprint-web \
  --db "${ATTRIBUTION_DB}" \
  --host 0.0.0.0 \
  --port "${ATTRIBUTION_WEB_PORT}" &
web_pid="$!"

wait -n "${service_pid}" "${web_pid}"
