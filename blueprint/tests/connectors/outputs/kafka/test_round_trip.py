# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One opt-in smoke test: completion event -> real Service -> result event.

Set ATTRIBUTION_TEST_KAFKA_BOOTSTRAP_SERVERS and ATTRIBUTION_TEST_SERVICE_URL
to disposable local services. This test creates and deletes its own topics.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from attribution_blueprint.connectors.attribution import AttributionConnector
from attribution_blueprint.connectors.execution_contexts.kafka import _build as build_input
from attribution_blueprint.connectors.logs.local_file import LocalFileLogConnector
from attribution_blueprint.connectors.outputs.kafka import _build as build_output
from attribution_blueprint.orchestrator import AttributionOrchestrator
from attribution_blueprint.types import Source

pytestmark = pytest.mark.integration


@pytest.fixture
def services():
    """Provision only test-owned topics on explicitly selected disposable services."""
    brokers = os.environ.get("ATTRIBUTION_TEST_KAFKA_BOOTSTRAP_SERVERS")
    service = os.environ.get("ATTRIBUTION_TEST_SERVICE_URL")
    if not brokers or not service:
        pytest.skip("set Kafka and Service test endpoints to run the result round trip")
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": brokers})
    topics = [f"blueprint-{name}-{uuid4().hex}" for name in ("completions", "results")]
    futures = admin.create_topics(
        [NewTopic(t, num_partitions=1, replication_factor=1, config={"max.message.bytes": "3000000"}) for t in topics]
    )
    for future in futures.values():
        future.result(20)
    try:
        yield brokers, service, topics
    finally:
        for future in admin.delete_topics(topics).values():
            future.result(20)


def _read_result(brokers, topic):
    """Read the emitted record with a fresh group and a bounded wait."""
    from confluent_kafka import Consumer

    consumer = Consumer(
        {
            "bootstrap.servers": brokers,
            "group.id": uuid4().hex,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            record = consumer.poll(1)
            if record is not None:
                assert record.error() is None, record.error()
                return record
        pytest.fail("no result arrived before the deadline")
    finally:
        consumer.close()


@pytest.fixture
def completion_event():
    """Build a unique RFC-shaped completion with distinct run and job identities."""
    path = Path(__file__).parents[3] / "fixtures" / "kafka_completion_events.json"
    event = json.loads(path.read_text())[0]
    event.update(id=str(uuid4()), type="com.nvidia.mfn.job.completed.v1", source="/mfn/runapi")
    event["data"].update(
        id=f"run-{uuid4().hex}", job_id=f"job-{uuid4().hex}", scheduler="runapi", cluster="example-cluster"
    )
    event["subject"] = f"jobs/{event['data']['job_id']}"
    return event


def _publish_completion(brokers, topic, event):
    """Require delivery of the trigger before running discovery."""
    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": brokers})
    delivery = []
    producer.produce(
        topic,
        key=event["data"]["id"],
        value=json.dumps(event),
        on_delivery=lambda error, _record: delivery.append(error),
    )
    assert producer.flush(15) == 0
    assert delivery == [None]


@pytest.fixture
def evidence(tmp_path, completion_event):
    """Use one synthetic XID line inside the supplied completion window."""
    evidence = tmp_path / "syslog.jsonl"
    evidence.write_text(
        json.dumps(
            {
                "body": "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.",
                "ts": int(datetime.fromisoformat("2026-10-21T17:38:12+00:00").timestamp()) * 1_000_000_000,
                "resources": [{"kind": "node", "hostname": completion_event["data"]["nodes"][0]}],
            }
        )
        + "\n"
    )
    return evidence


@pytest.fixture
def pipeline(services, evidence):
    """Wire production connectors and close resources even if the assertion fails."""
    brokers, service, (input_topic, output_topic) = services
    eca = build_input(
        {
            "bootstrap_servers": brokers,
            "topic": input_topic,
            "group_id": uuid4().hex,
            "max_messages": 1,
            "discovery_timeout_seconds": 20,
        },
        {},
    )
    output = build_output({"bootstrap_servers": brokers, "topic": output_topic}, {})
    pipeline = AttributionOrchestrator(
        execution_context_adapters=[eca],
        log_connectors=[LocalFileLogConnector(log_path_templates=[str(evidence)], source=Source.SYSLOG)],
        metrics_connectors=[],
        attribution=AttributionConnector(service),
        outputs=[output],
        gpus_per_node=8,
    )
    try:
        yield pipeline
    finally:
        eca.close()
        output.close()


def _assert_result(record, event, result):
    """Compare correlation, actual service verdict, and collected evidence on wire."""
    emitted = json.loads(record.value())
    assert record.key() == event["data"]["id"].encode()
    assert ("content-type", b"application/cloudevents+json") in record.headers()
    assert emitted["type"] == "com.nvidia.attribution.job.attributed.v1"
    assert emitted["source"] == "/attribution/blueprint"
    assert emitted["subject"] == event["subject"]
    data = emitted["data"]
    assert data["trigger_event_id"] == event["id"]
    for key in ("job_id", "attempt_id", "cluster", "scheduler"):
        assert data[key] == event["data"][key]
    assert data["terminal_state"] == result["terminal_state"] == "fail"
    assert data["sub_domain"] == result["sub_domain"] == "hardware"
    assert data["result"]["verdict"] == result["verdict"]
    assert data["result"]["symptoms"] == result["symptoms"]
    assert "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS" in data["result"]["symptoms"]
    assert data["connector_reports"] == result["connector_reports"]
    assert data["ingestion_reports"] == result["ingestion_reports"]


def test_completion_to_service_to_result_event(services, completion_event, pipeline):
    """Verify native delivery and RFC correlation around a real XID attribution."""
    brokers, _, (input_topic, output_topic) = services
    _publish_completion(brokers, input_topic, completion_event)
    [result] = pipeline.analyze()
    assert pipeline.last_run_stats == {"attempted": 1, "succeeded": 1, "conflicted": 0}
    _assert_result(_read_result(brokers, output_topic), completion_event, result)


def test_result_larger_than_native_default_reaches_broker(services):
    """A raised size limit permits a real record larger than the native default."""
    brokers, _, (_, topic) = services
    output = build_output({"bootstrap_servers": brokers, "topic": topic, "max_message_bytes": 2_000_000}, {})
    verdict = {"detail": "x" * 1_100_000}
    try:
        output.write({"id": "large-result-run"}, {"verdict": verdict, "symptoms": {}})
    finally:
        output.close()
    record = _read_result(brokers, topic)
    assert len(record.value()) > 1_000_000
    assert record.key() == b"large-result-run"
    assert json.loads(record.value())["data"]["result"]["verdict"] == verdict
