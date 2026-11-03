# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full completion-event contract through disposable Kafka, Service, and SQLite.

Run against disposable local services:
    ATTRIBUTION_TEST_SERVICE_URL=http://127.0.0.1:18000 \
    ATTRIBUTION_TEST_KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:39092 \
    uv run --project blueprint --frozen --extra dev --extra kafka pytest \
    blueprint/tests/connectors/execution_contexts/kafka/test_end_to_end.py -q

The test creates and deletes its own topic; it never uses an existing topic.
"""

import json
import sqlite3
from datetime import datetime

import pytest

from attribution_blueprint.connectors.attribution import AttributionConnector
from attribution_blueprint.connectors.execution_contexts.kafka import _build
from attribution_blueprint.connectors.logs.local_file import LocalFileLogConnector
from attribution_blueprint.connectors.outputs.sqlite import SQLiteOutput
from attribution_blueprint.orchestrator import AttributionOrchestrator
from attribution_blueprint.types import Source

pytestmark = pytest.mark.integration

XID_SYMPTOM = "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS"


EXPECTED_FIELDS = {
    "id",
    "job_id",
    "attempt_id",
    "scheduler",
    "workload_framework",
    "name",
    "user_name",
    "state",
    "start_time",
    "end_time",
    "timezone",
    "time_limit",
    "nodes",
    "gpu_count",
    "failed_nodes",
    "cancelled_by",
    "namespace",
    "pod_names",
    "pod_nodes",
}


def _write_evidence(path, node):
    """Write an XID inside the event window on a node declared by the event."""
    evidence = path
    evidence.write_text(
        json.dumps(
            {
                "body": "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.",
                "ts": int(datetime.fromisoformat("2026-10-21T17:38:12+00:00").timestamp()) * 1_000_000_000,
                "resources": [{"kind": "node", "hostname": node}],
            }
        )
        + "\n"
    )
    return evidence


def _attribute(adapter, service_url, evidence, database, monkeypatch):
    """Use real collection/HTTP/output, recording the workload sent to the Service."""
    collector = LocalFileLogConnector(log_path_templates=[str(evidence)], source=Source.SYSLOG)
    attribution = AttributionConnector(service_url)
    sent_workloads = []
    create_attributor = attribution.create_attributor

    def record_create(info):
        # Observe the request and delegate to the real HTTP client.
        sent_workloads.append(info.workload.model_dump(mode="json"))
        return create_attributor(info)

    monkeypatch.setattr(attribution, "create_attributor", record_create)
    pipeline = AttributionOrchestrator(
        execution_context_adapters=[adapter],
        log_connectors=[collector],
        metrics_connectors=[],
        attribution=attribution,
        outputs=[SQLiteOutput(str(database))],
        gpus_per_node=8,
    )
    results = pipeline.analyze()
    assert pipeline.last_run_stats == {"attempted": 1, "succeeded": 1, "conflicted": 0}
    assert len(results) == 1
    assert len(sent_workloads) == 1
    return results[0], sent_workloads[0]


def _assert_workload(workload, data):
    """Check scheduler fields, time window, and rank placement at the HTTP boundary."""
    for context_field, wire_field in {
        "id": "id",
        "state": "status",
        "name": "name",
        "user_name": "username",
        "workload_framework": "framework",
        "timezone": "timezone",
        "time_limit": "time_limit",
    }.items():
        assert workload[wire_field] == data[context_field]
    for context_field, wire_field in {"start_time": "job_start_time", "end_time": "job_end_time"}.items():
        assert datetime.fromisoformat(workload[wire_field]) == datetime.fromisoformat(data[context_field])
    assert workload["failed_node"] == min(data["failed_nodes"])
    assert workload["cancelled_by"] == (data["cancelled_by"] or None)
    assert workload["nodes"] == {node: list(range(i * 8, (i + 1) * 8)) for i, node in enumerate(data["nodes"])}
    # Connector-only scheduler provenance stays local; the generic example
    # value is not a supported service scheduler enum.
    assert workload["scheduler"] is None


def _assert_stored(database, event, expected_result, evidence_node):
    """Check every input field and the hardware verdict in the persisted result."""
    data = event["data"]
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        contexts = connection.execute("SELECT * FROM execution_context").fetchall()
        runs = connection.execute("SELECT * FROM attribution_run").fetchall()
    assert len(contexts) == len(runs) == 1
    stored = contexts[0]
    raw = json.loads(stored["raw_scheduler_json"])
    for field, value in data.items():
        if field in {"start_time", "end_time"}:
            assert datetime.fromisoformat(raw[field]) == datetime.fromisoformat(value)
        elif field == "cancelled_by" and not value:
            assert field not in raw  # Empty cancellation is normalized to unset.
        else:
            assert raw[field] == value, field
    assert raw["event_id"] == event["id"]
    assert raw["failed_node"] == min(data["failed_nodes"])
    assert stored["job_id"] == data["job_id"]
    assert stored["attempt_id"] == data["attempt_id"]
    assert stored["scheduler_type"] == data["scheduler"]
    assert json.loads(stored["nodes_json"]) == data["nodes"]
    assert runs[0]["execution_context_id"] == stored["id"]
    result = json.loads(runs[0]["raw_result_json"])
    assert result == expected_result
    assert XID_SYMPTOM in result["symptoms"]
    assert json.loads(runs[0]["responsible_nodes_json"]) == [evidence_node]
    assert any(report["status"] == "ok" for report in result["connector_reports"])
    assert result["observation_context"]


@pytest.mark.parametrize("encoding", ["structured", "binary"])
@pytest.mark.parametrize("multiple_failed_nodes", [False, True])
@pytest.mark.parametrize("cancelled_by", ["", "example-operator"])
def test_full_completion_event_reaches_service_and_sqlite(
    kafka_topic,
    kafka_spec,
    completion_event,
    tmp_path,
    monkeypatch,
    cancelled_by,
    encoding,
    multiple_failed_nodes,
    publish_event,
):
    """Prove full-payload attribution and restart from the committed group offset."""
    event = completion_event
    data = event["data"]
    data["cancelled_by"] = cancelled_by
    if multiple_failed_nodes:
        data["failed_nodes"] = list(reversed(data["nodes"]))
    event["dataschema"] = "https://example.com/schemas/job-completed-v1.json"
    assert set(data) == EXPECTED_FIELDS
    publish_event(event, encoding=encoding)
    evidence_node = min(data["failed_nodes"])
    evidence = _write_evidence(tmp_path / "syslog.jsonl", evidence_node)
    database = tmp_path / "attributions.db"
    adapter = _build(kafka_spec, {})
    try:
        result, workload = _attribute(adapter, kafka_topic[1], evidence, database, monkeypatch)
    finally:
        adapter.close()
    _assert_workload(workload, data)
    _assert_stored(database, event, result, evidence_node)
    restarted = _build(kafka_spec, {})
    try:
        assert restarted.discover() == []
    finally:
        restarted.close()
