#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
expected_runs="${END_TO_END_EXPECTED_RUNS:-118}"
wait_seconds="${END_TO_END_WAIT_SECONDS:-180}"
service_port="${ATTRIBUTION_SERVICE_PORT:-18000}"
web_port="${ATTRIBUTION_WEB_PORT:-18080}"
run_suffix="${GITHUB_RUN_ID:-$$}"
service_image="attribution-service-test:${run_suffix}"
blueprint_image="attribution-blueprint-test:${run_suffix}"
container_name="attribution-test-${run_suffix}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/attribution-end-to-end.XXXXXX")"
source_root="${work_dir}/source"
database_dir="${work_dir}/data"
database_path="${database_dir}/attributions.db"
output_path="${work_dir}/attribution.log"

cleanup() {
    local status=$?
    docker rm --force "${container_name}" >/dev/null 2>&1 || true
    docker image rm --force "${service_image}" "${blueprint_image}" >/dev/null 2>&1 || true
    if [[ -n "${work_dir}" && "${work_dir}" == "${TMPDIR:-/tmp}"/attribution-end-to-end.* ]]; then
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

assert_image_license() {
    local image="$1"
    local actual_label
    actual_label="$(
        docker image inspect \
            --format '{{ index .Config.Labels "com.nvidia.attribution.project.license" }}' \
            "${image}"
    )"
    if [[ "${actual_label}" != "Apache-2.0" ]]; then
        echo "${image}: expected Apache-2.0 project license label, found ${actual_label}" >&2
        return 1
    fi
    docker run --rm \
        --entrypoint python \
        --volume "${source_root}/LICENSE:/expected/LICENSE:ro" \
        "${image}" \
        -c 'from pathlib import Path; raise SystemExit(Path("/licenses/LICENSE").read_bytes() != Path("/expected/LICENSE").read_bytes())'
}

cd "${repository_root}"
mkdir -p "${source_root}"
git checkout-index --all --prefix="${source_root}/"
mkdir -p "${database_dir}"

docker build \
    --file "${source_root}/service/service-app/Dockerfile" \
    --tag "${service_image}" \
    "${source_root}"
docker build \
    --file "${source_root}/blueprint/docker/Dockerfile" \
    --tag "${blueprint_image}" \
    "${source_root}"
assert_image_license "${service_image}"
assert_image_license "${blueprint_image}"
docker run --detach --rm \
    --name "${container_name}" \
    --env ATTRIBUTION_DB=/data/attributions.db \
    --publish "127.0.0.1:${service_port}:8000" \
    --publish "127.0.0.1:${web_port}:8080" \
    --volume "${database_dir}:/data" \
    "${blueprint_image}" >/dev/null

wait_for_url "http://127.0.0.1:${service_port}/v2/healthz"
wait_for_url "http://127.0.0.1:${web_port}/"

ATTRIBUTION_TEST_SERVICE_URL="http://127.0.0.1:${service_port}" \
uv run --project blueprint --frozen --extra dev pytest \
    blueprint/tests/connectors/attribution/test_session_lifecycle.py -q

uv run --project blueprint --frozen nvidia-attribution-blueprint \
    --configs blueprint/examples/static/configs \
    --connector attribution service_url "http://127.0.0.1:${service_port}" \
    --connector output db sqlite path "${database_path}" \
    2>&1 | tee "${output_path}"

if ! grep -F "Attribution complete: ${expected_runs}/${expected_runs} execution contexts succeeded." \
    "${output_path}" >/dev/null; then
    echo "end-to-end tests did not complete ${expected_runs}/${expected_runs} contexts" >&2
    exit 1
fi

END_TO_END_DATABASE_PATH="${database_path}" \
END_TO_END_EXPECTED_RUNS="${expected_runs}" \
END_TO_END_WEB_PORT="${web_port}" \
uv run --project blueprint --frozen python - <<'PY'
import json
import os
import sqlite3
import urllib.request
from pathlib import Path

database_path = os.environ["END_TO_END_DATABASE_PATH"]
expected_runs = int(os.environ["END_TO_END_EXPECTED_RUNS"])
manifest_path = Path("blueprint/docs/sample-data/static/sample-jobs.json")

with sqlite3.connect(database_path) as connection:
    run_count = connection.execute("select count(*) from attribution_run").fetchone()[0]
    rows = connection.execute(
        "select execution_context.job_id, attribution_run.outcome, "
        "attribution_run.responsible_nodes_json, attribution_run.raw_result_json "
        "from attribution_run join execution_context "
        "on execution_context.id = attribution_run.execution_context_id"
    ).fetchall()
    responsible_node_count = connection.execute(
        "select count(*) from attribution_run "
        "where responsible_nodes_json is not null and responsible_nodes_json != '[]'"
    ).fetchone()[0]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
actual = {
    job_id: {
        "outcome": outcome,
        "responsible_nodes": json.loads(responsible_nodes_json),
        "raw_result": json.loads(raw_result_json),
    }
    for job_id, outcome, responsible_nodes_json, raw_result_json in rows
}
expected = {job["id"]: job["expected_outcome"] for job in manifest["jobs"]}
responsible_node_assertion_workloads = set(manifest["constraints"]["responsible_node_assertion_workloads"])
missing = sorted(set(expected) - set(actual))
extra = sorted(set(actual) - set(expected))
mismatches = sorted(
    (job_id, expected[job_id], actual[job_id]["outcome"])
    for job_id in set(expected) & set(actual)
    if expected[job_id] != actual[job_id]["outcome"]
)
responsible_node_mismatches = sorted(
    (job["id"], job["responsible_nodes"], actual[job["id"]]["responsible_nodes"])
    for job in manifest["jobs"]
    if job["id"] in responsible_node_assertion_workloads
    and job["id"] in actual
    and job["responsible_nodes"] != actual[job["id"]]["responsible_nodes"]
)
cascade_mismatches = []
for job in manifest["jobs"]:
    discarded_prefixes = job.get("discarded_symptom_prefixes", [])
    if discarded_prefixes == [] or job["id"] not in actual:
        continue
    actual_symptoms = actual[job["id"]]["raw_result"].get("symptoms", {})
    for prefix in discarded_prefixes:
        retained = sorted(symptom for symptom in actual_symptoms if symptom.startswith(prefix))
        if retained != []:
            cascade_mismatches.append((job["id"], prefix, retained))

bad_pair_mismatches = []
for job in manifest["jobs"]:
    expected_bad_pairs = job.get("expected_bad_pairs")
    if expected_bad_pairs is None or job["id"] not in actual:
        continue
    collectors = actual[job["id"]]["raw_result"].get("collectors", {})
    actual_bad_pairs = collectors.get("bad_pairs") if isinstance(collectors, dict) else None
    if expected_bad_pairs != actual_bad_pairs:
        bad_pair_mismatches.append((job["id"], expected_bad_pairs, actual_bad_pairs))

if run_count != expected_runs:
    raise SystemExit(f"expected {expected_runs} attribution runs, found {run_count}")
if missing != [] or extra != [] or mismatches != []:
    raise SystemExit(
        f"sample outcome mismatch: missing={missing}, extra={extra}, mismatches={mismatches}"
    )
if responsible_node_mismatches != []:
    raise SystemExit(f"sample responsible-node mismatch: {responsible_node_mismatches}")
if cascade_mismatches != []:
    raise SystemExit(f"sample cascade mismatch: {cascade_mismatches}")
if bad_pair_mismatches != []:
    raise SystemExit(f"sample bad-pair mismatch: {bad_pair_mismatches}")
if responsible_node_count == 0:
    raise SystemExit("expected at least one attribution run with responsible nodes")

web_port = os.environ["END_TO_END_WEB_PORT"]
with urllib.request.urlopen(f"http://127.0.0.1:{web_port}/api/v1/dashboard", timeout=10) as response:
    web_payload = json.load(response)
web_run_count = int(web_payload["meta"]["dbTotalRuns"])
if web_run_count != expected_runs:
    raise SystemExit(f"expected {expected_runs} web UI runs, found {web_run_count}")

print(
    f"end_to_end_results={run_count} outcome_mismatches=0 "
    f"responsible_node_mismatches=0 cascade_mismatches=0 bad_pair_mismatches=0 "
    f"responsible_nodes={responsible_node_count} webui_total_runs={web_run_count}"
)
PY

if docker logs "${container_name}" 2>&1 \
    | grep -E "500 Internal Server Error|exceptions must derive|IncompatibleObservation|ingest_observation failed"; then
    echo "end-to-end service logs contain ingestion errors" >&2
    exit 1
fi

echo "end-to-end tests passed"
