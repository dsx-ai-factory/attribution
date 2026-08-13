#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
expected_runs="${QUICKSTART_EXPECTED_RUNS:-118}"
wait_seconds="${QUICKSTART_WAIT_SECONDS:-180}"
service_port="${ATTRIBUTION_SERVICE_PORT:-18000}"
web_port="${ATTRIBUTION_WEB_PORT:-18080}"
run_suffix="${GITHUB_RUN_ID:-$$}"
service_image="attribution-service-test:${run_suffix}"
blueprint_image="attribution-blueprint-test:${run_suffix}"
container_name="attribution-test-${run_suffix}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/attribution-quickstart.XXXXXX")"
database_dir="${work_dir}/data"
database_path="${database_dir}/attributions.db"
output_path="${work_dir}/attribution.log"

cleanup() {
    local status=$?
    docker rm --force "${container_name}" >/dev/null 2>&1 || true
    docker image rm --force "${service_image}" "${blueprint_image}" >/dev/null 2>&1 || true
    if [[ -n "${work_dir}" && "${work_dir}" == "${TMPDIR:-/tmp}"/attribution-quickstart.* ]]; then
        rm -rf -- "${work_dir}"
    fi
    exit "${status}"
}
trap cleanup EXIT

wait_for_url() {
    local url="$1"
    local attempt
    for ((attempt = 1; attempt <= wait_seconds; attempt++)); do
        if curl --fail --silent "${url}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    curl --fail --silent --show-error "${url}" >/dev/null
}

cd "${repository_root}"
mkdir -p "${database_dir}"

docker build \
    --file service/service-app/Dockerfile \
    --tag "${service_image}" \
    .
docker build \
    --file blueprint/docker/Dockerfile \
    --tag "${blueprint_image}" \
    .
docker run --detach --rm \
    --name "${container_name}" \
    --env ATTRIBUTION_DB=/data/attributions.db \
    --publish "127.0.0.1:${service_port}:8000" \
    --publish "127.0.0.1:${web_port}:8080" \
    --volume "${database_dir}:/data" \
    "${blueprint_image}" >/dev/null

wait_for_url "http://127.0.0.1:${service_port}/v2/healthz"
wait_for_url "http://127.0.0.1:${web_port}/"

uv run --project blueprint --frozen nvidia-attribution-blueprint \
    --configs blueprint/examples/static/configs \
    --connector attribution service_url "http://127.0.0.1:${service_port}" \
    --connector output db sqlite path "${database_path}" \
    2>&1 | tee "${output_path}"

if ! grep -F "Attribution complete: ${expected_runs}/${expected_runs} execution contexts succeeded." \
    "${output_path}" >/dev/null; then
    echo "quickstart did not complete ${expected_runs}/${expected_runs} contexts" >&2
    exit 1
fi

QUICKSTART_DATABASE_PATH="${database_path}" \
QUICKSTART_EXPECTED_RUNS="${expected_runs}" \
QUICKSTART_WEB_PORT="${web_port}" \
uv run --project blueprint --frozen python - <<'PY'
import json
import os
import sqlite3
import urllib.request
from pathlib import Path

database_path = os.environ["QUICKSTART_DATABASE_PATH"]
expected_runs = int(os.environ["QUICKSTART_EXPECTED_RUNS"])
manifest_path = Path("blueprint/docs/sample-data/static/sample-jobs.json")

with sqlite3.connect(database_path) as connection:
    run_count = connection.execute("select count(*) from attribution_run").fetchone()[0]
    actual = dict(
        connection.execute(
            "select execution_context.job_id, attribution_run.outcome "
            "from attribution_run join execution_context "
            "on execution_context.id = attribution_run.execution_context_id"
        )
    )
    responsible_node_count = connection.execute(
        "select count(*) from attribution_run "
        "where responsible_nodes_json is not null and responsible_nodes_json != '[]'"
    ).fetchone()[0]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {job["id"]: job["expected_outcome"] for job in manifest["jobs"]}
missing = sorted(set(expected) - set(actual))
extra = sorted(set(actual) - set(expected))
mismatches = sorted(
    (job_id, expected[job_id], actual[job_id])
    for job_id in set(expected) & set(actual)
    if expected[job_id] != actual[job_id]
)

if run_count != expected_runs:
    raise SystemExit(f"expected {expected_runs} attribution runs, found {run_count}")
if missing != [] or extra != [] or mismatches != []:
    raise SystemExit(
        f"sample outcome mismatch: missing={missing}, extra={extra}, mismatches={mismatches}"
    )
if responsible_node_count == 0:
    raise SystemExit("expected at least one attribution run with responsible nodes")

web_port = os.environ["QUICKSTART_WEB_PORT"]
with urllib.request.urlopen(f"http://127.0.0.1:{web_port}/api/v1/dashboard", timeout=10) as response:
    web_payload = json.load(response)
web_run_count = int(web_payload["meta"]["totalRuns"])
if web_run_count != expected_runs:
    raise SystemExit(f"expected {expected_runs} web UI runs, found {web_run_count}")

print(
    f"quickstart_results={run_count} outcome_mismatches=0 "
    f"responsible_nodes={responsible_node_count} webui_total_runs={web_run_count}"
)
PY

if docker logs "${container_name}" 2>&1 \
    | grep -E "500 Internal Server Error|exceptions must derive|IncompatibleObservation|ingest_observation failed"; then
    echo "quickstart service logs contain ingestion errors" >&2
    exit 1
fi

echo "quickstart validation passed"
