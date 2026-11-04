# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One completion-event smoke test through real Kafka, Service, and SQLite.

Use disposable endpoints as documented in README.md. Expanded pipeline variants
are separate Blueprint integration follow-up work.
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
        # Discovery must release Kafka ownership before the Service is called.
        assert adapter._closed
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


def _assert_smoke_result(database, event, result, evidence_node):
    """Check persisted correlation, node scope, and the real hardware verdict."""
    with sqlite3.connect(database) as connection:
        [stored] = connection.execute(
            "SELECT ec.raw_scheduler_json, ar.raw_result_json, ar.responsible_nodes_json "
            "FROM execution_context ec JOIN attribution_run ar ON ar.execution_context_id = ec.id"
        ).fetchall()
    raw = json.loads(stored[0])
    assert (raw["event_source"], raw["event_id"], raw["event_subject"]) == (
        event["source"],
        event["id"],
        event["subject"],
    )
    for field in ("id", "job_id", "attempt_id", "nodes", "cluster"):
        assert raw[field] == event["data"][field]
    assert json.loads(stored[1]) == result
    assert json.loads(stored[2]) == [evidence_node]
    assert XID_SYMPTOM in result["symptoms"]


def test_completion_event_reaches_service_and_sqlite(
    kafka_topic, kafka_spec, completion_event, publish_event, tmp_path, monkeypatch
):
    """Attribute one structured completion and persist its correlation metadata."""
    event = completion_event
    event["data"]["cluster"] = "example-cluster"
    event["subject"] = f"custom/jobs/{event['data']['id']}"
    publish_event(event)
    evidence_node = min(event["data"]["failed_nodes"])
    evidence = _write_evidence(tmp_path / "syslog.jsonl", evidence_node)
    database = tmp_path / "attributions.db"
    adapter = _build(kafka_spec, {})
    try:
        result, workload = _attribute(adapter, kafka_topic[1], evidence, database, monkeypatch)
    finally:
        adapter.close()
    assert workload["id"] == event["data"]["id"]
    assert set(workload["nodes"]) == set(event["data"]["nodes"])
    _assert_smoke_result(database, event, result, evidence_node)
    assert _build(kafka_spec, {}).discover() == []
