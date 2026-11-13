# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result mapping and acknowledgement behavior, using native callback semantics."""

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from attribution_blueprint.connectors.outputs.kafka import KafkaOutput, KafkaOutputError, _build
from attribution_blueprint.exceptions import MCAError

FIXTURE = Path(__file__).parents[3] / "fixtures" / "kafka_attribution_result.json"


class FakeProducer:
    """Deliver only from flush(), which returns the count of queued records."""

    def __init__(self, error=None, deliver=True):
        self.error = error
        self.deliver = deliver
        self.queued = []
        self.messages = []
        self.flushes = []

    def produce(self, topic, **record):
        self.queued.append((topic, record))

    def flush(self, timeout):
        self.flushes.append(timeout)
        if self.deliver:
            for topic, record in self.queued:
                self.messages.append((topic, record))
                record["on_delivery"](self.error, None)
            self.queued.clear()
        return len(self.queued)


@pytest.fixture
def rfc():
    """Use the RFC output example as the expected wire contract."""
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def context(rfc):
    """Keep run ID distinct from job ID to test both correlation and routing."""
    data = rfc["data"]
    return {
        "id": "run-123",
        **{k: data[k] for k in ("job_id", "attempt_id", "scheduler", "cluster")},
        "event_id": data["trigger_event_id"],
        "event_subject": rfc["subject"],
    }


@pytest.fixture
def attribution(rfc):
    """Represent the enriched result passed to every output by the orchestrator."""
    data = rfc["data"]
    return {
        **data["result"],
        **{
            k: data[k]
            for k in (
                "terminal_state",
                "sub_domain",
                "tenant_sre_action",
                "cloud_operator_sre_action",
                "connector_reports",
            )
        },
    }


def test_result_matches_rfc_and_waits_for_delivery(rfc, context, attribution):
    producer = FakeProducer()
    output = KafkaOutput(producer=producer, topic="results")
    before = datetime.now(timezone.utc)
    original = copy.deepcopy(attribution)

    output.write(context, attribution)

    [(topic, record)] = producer.messages
    event = json.loads(record["value"])
    assert topic == "results"
    assert record["key"] == b"run-123"
    assert record["headers"] == [("content-type", "application/cloudevents+json")]
    assert producer.flushes == [30.0]
    assert UUID(event["id"]).version == 4
    assert event["id"] != context["event_id"]
    assert before <= datetime.fromisoformat(event["time"]) <= datetime.now(timezone.utc)
    assert {k: v for k, v in event.items() if k not in ("id", "time")} == {
        k: v for k, v in rfc.items() if k not in ("id", "time")
    }
    assert attribution == original


@pytest.mark.parametrize(
    "run_id,expected_key",
    [
        pytest.param("provider-run", b"provider-run", id="provider-run"),
        pytest.param(None, b"processing-1", id="missing"),
        pytest.param("", b"processing-1", id="empty"),
        pytest.param(3, b"processing-1", id="integer"),
        pytest.param(True, b"processing-1", id="boolean"),
    ],
)
def test_processing_id_keeps_original_run_routing(
    run_id: object, expected_key: bytes, context: dict[str, object], attribution: dict[str, object]
) -> None:
    """Use an explicit run identity for routing without changing result correlation."""
    producer = FakeProducer()
    ctx = {**context, "id": "processing-1", "run_id": run_id}
    KafkaOutput(producer=producer, topic="results").write(ctx, attribution)

    record = producer.messages[0][1]
    event = json.loads(record["value"])
    assert record["key"] == expected_key
    assert event["data"]["job_id"] == context["job_id"]
    assert event["data"]["attempt_id"] == context["attempt_id"]
    assert event["data"]["trigger_event_id"] == context["event_id"]
    assert event["subject"] == context["event_subject"]


def test_original_run_supplies_missing_job_identity(attribution: dict[str, object]) -> None:
    """A processing key does not leak into fallback job identity or subject."""
    producer = FakeProducer()
    KafkaOutput(producer=producer, topic="results").write({"id": "processing-1", "run_id": "original-run"}, attribution)

    record = producer.messages[0][1]
    event = json.loads(record["value"])
    assert record["key"] == b"original-run"
    assert event["data"]["job_id"] == "original-run"
    assert event["subject"] == "jobs/original-run"


def test_preserves_service_fields_and_evidence_extensions(context, attribution):
    producer = FakeProducer()
    attribution.update(
        collectors={"custom": {"node-a": [1]}},
        observation_context=["obs-1"],
        evidence_status="partial",
        ingestion_reports=[{"source": "syslog", "status": "empty"}],
    )
    KafkaOutput(producer=producer, topic="results").write(context, attribution)

    data = json.loads(producer.messages[0][1]["value"])["data"]
    assert data["result"]["collectors"] == attribution["collectors"]
    assert data["result"]["observation_context"] == ["obs-1"]
    assert data["ingestion_reports"] == attribution["ingestion_reports"]
    assert data["evidence_status"] == "partial"


@pytest.mark.parametrize("subject", [None, "", 42])
def test_non_kafka_context_uses_defaults(subject, attribution):
    producer = FakeProducer()
    KafkaOutput(producer=producer, topic="results", cluster="fallback").write(
        {"id": "job-1", "event_subject": subject}, attribution
    )
    event = json.loads(producer.messages[0][1]["value"])
    assert event["subject"] == "jobs/job-1"
    assert {k: event["data"][k] for k in ("job_id", "attempt_id", "cluster", "scheduler", "trigger_event_id")} == {
        "job_id": "job-1",
        "attempt_id": "0",
        "cluster": "fallback",
        "scheduler": None,
        "trigger_event_id": None,
    }


def test_new_emit_gets_new_id_and_preserves_string_attempt(context, attribution):
    producer = FakeProducer()
    output = KafkaOutput(producer=producer, topic="results", cluster="fallback")
    context["attempt_id"] = "2.1"
    output.write(context, attribution)
    output.write(context, attribution)
    first, second = [json.loads(record["value"]) for _, record in producer.messages]
    assert first["id"] != second["id"]
    assert first["data"]["attempt_id"] == "2.1"
    assert first["data"]["cluster"] == context["cluster"]


@pytest.mark.parametrize(
    "error,deliver,match",
    [("authorization failed", True, "authorization failed"), (None, False, "delivery unconfirmed")],
)
def test_delivery_must_be_confirmed(error, deliver, match, context, attribution):
    output = KafkaOutput(producer=FakeProducer(error, deliver), topic="results")
    with pytest.raises(KafkaOutputError, match=match):
        output.write(context, attribution)


def test_late_report_cannot_acknowledge_next_record(context, attribution):
    producer = FakeProducer(deliver=False)
    output = KafkaOutput(producer=producer, topic="results")
    with pytest.raises(KafkaOutputError, match="unconfirmed"):
        output.write(context, attribution)
    # Only the old record gets a callback during the second write's flush.
    old_flush = producer.flush

    def flush_old(timeout):
        pending = producer.queued.pop()
        producer.deliver = True
        old_flush(timeout)
        producer.queued.append(pending)
        return 1

    producer.flush = flush_old
    with pytest.raises(KafkaOutputError, match="unconfirmed"):
        output.write(context, attribution)


@pytest.mark.parametrize("operation", ["produce", "flush"])
def test_native_exceptions_propagate(operation, mocker, context, attribution):
    native = pytest.importorskip("confluent_kafka")
    producer = FakeProducer()
    error = native.KafkaException(native.KafkaError(native.KafkaError._TRANSPORT))
    mocker.patch.object(producer, operation, side_effect=error)
    with pytest.raises(native.KafkaException) as caught:
        KafkaOutput(producer=producer, topic="results").write(context, attribution)
    assert caught.value is error


def test_oversized_result_fails_without_truncation(context, attribution):
    producer = FakeProducer()
    original = copy.deepcopy(attribution)
    with pytest.raises(KafkaOutputError, match="over max_message_bytes"):
        KafkaOutput(producer=producer, topic="results", max_message_bytes=100).write(context, attribution)
    assert not producer.queued
    assert attribution == original


@pytest.mark.parametrize("via_registry", [False, True])
def test_default_size_guardrail_can_be_overridden(via_registry, mocker, context):
    """Both construction paths reject large results by default and allow an override."""
    producer = FakeProducer()
    if via_registry:
        native = pytest.importorskip("confluent_kafka")
        mocker.patch.object(native, "Producer", return_value=producer)

    def build(**options):
        """Construct the selected path with identical options."""
        if via_registry:
            return _build({"bootstrap_servers": "localhost:9092", "topic": "results", **options}, {})
        return KafkaOutput(producer=producer, topic="results", **options)

    attribution = {"verdict": {"detail": "x" * 65536}, "symptoms": {}}
    with pytest.raises(KafkaOutputError, match="max_message_bytes=65536"):
        build().write(context, attribution)
    assert not producer.queued

    build(max_message_bytes=128 * 1024).write(context, attribution)
    assert json.loads(producer.messages[0][1]["value"])["data"]["result"] == attribution


@pytest.mark.parametrize("pending", [False, True])
def test_close_is_idempotent_and_prevents_reuse(pending, context, attribution):
    producer = FakeProducer(deliver=not pending)
    output = KafkaOutput(producer=producer, topic="results")
    if pending:
        with pytest.raises(KafkaOutputError):
            output.write(context, attribution)
        with pytest.raises(KafkaOutputError, match="pending"):
            output.close()
    else:
        output.close()
    flushes = len(producer.flushes)
    output.close()
    assert len(producer.flushes) == flushes
    with pytest.raises(KafkaOutputError, match="closed"):
        output.write(context, attribution)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), float("-inf"), 2147483.648, 1e308])
def test_invalid_timeout_rejected_before_producer_allocation(value, mocker):
    native = pytest.importorskip("confluent_kafka")
    factory = mocker.patch.object(native, "Producer")
    with pytest.raises(MCAError, match="delivery_timeout_seconds"):
        _build({"bootstrap_servers": "localhost:9092", "topic": "results", "delivery_timeout_seconds": value}, {})
    factory.assert_not_called()


def test_builder_creates_idempotent_producer_from_registry_config(mocker, monkeypatch):
    native = pytest.importorskip("confluent_kafka")
    producer = FakeProducer()
    factory = mocker.patch.object(native, "Producer", return_value=producer)
    monkeypatch.setenv("TEST_KAFKA_USER", "example-user")
    monkeypatch.setenv("TEST_KAFKA_PASSWORD", "example-password")
    from attribution_blueprint.mca import build_config

    config = build_config(
        {
            "eca": {
                "component": "static",
                "static": {
                    "id": "job",
                    "nodes": ["node-a"],
                    "start_time": "2026-10-21T14:05:31Z",
                    "end_time": "2026-10-21T17:41:58Z",
                },
            },
            "attribution": {"service_url": "http://localhost:8000"},
            "output": {
                "events": {
                    "component": "kafka",
                    "kafka": {
                        "bootstrap_servers": "localhost:9092",
                        "topic": "results",
                        "security_protocol": "SASL_SSL",
                        "sasl_mechanism": "SCRAM-SHA-512",
                        "sasl_username_env": "TEST_KAFKA_USER",
                        "sasl_password_env": "TEST_KAFKA_PASSWORD",
                        "ssl_ca_location": "/example/ca.pem",
                    },
                }
            },
        }
    )
    assert len(config.outputs) == 1
    assert isinstance(config.outputs[0], KafkaOutput)
    factory.assert_called_once_with(
        {
            "bootstrap.servers": "localhost:9092",
            "acks": "all",
            "enable.idempotence": True,
            "message.timeout.ms": 30000,
            "message.max.bytes": 1000000,
            "security.protocol": "SASL_SSL",
            "sasl.mechanism": "SCRAM-SHA-512",
            "sasl.username": "example-user",
            "sasl.password": "example-password",
            "ssl.ca.location": "/example/ca.pem",
        }
    )


def test_optional_client_missing_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "confluent_kafka", None)
    with pytest.raises(MCAError, match="requires the kafka extra"):
        _build({"bootstrap_servers": "localhost:9092", "topic": "results"}, {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_message_bytes", 0),
        ("max_message_bytes", True),
        ("max_message_bytes", 1.5),
        ("max_message_bytes", 1_000_000_001),
        ("source", "bad source"),
        ("event_type", ""),
        ("topic", ""),
    ],
)
def test_invalid_options_are_rejected(field, value):
    with pytest.raises(MCAError, match=field):
        _build({"bootstrap_servers": "localhost:9092", "topic": "results", field: value}, {})


def test_builder_requires_broker_and_topic():
    with pytest.raises(MCAError, match="bootstrap_servers.*topic"):
        _build({}, {})


def test_sasl_requires_environment_credentials(monkeypatch):
    monkeypatch.delenv("TEST_MISSING_KAFKA_USER", raising=False)
    with pytest.raises(MCAError, match="unset environment variable"):
        _build(
            {
                "bootstrap_servers": "localhost:9092",
                "topic": "results",
                "security_protocol": "SASL_SSL",
                "sasl_mechanism": "SCRAM-SHA-512",
                "sasl_username_env": "TEST_MISSING_KAFKA_USER",
            },
            {},
        )


def test_buffer_full_is_propagated(mocker, context, attribution):
    producer = FakeProducer()
    mocker.patch.object(producer, "produce", side_effect=BufferError("queue full"))
    with pytest.raises(BufferError, match="queue full"):
        KafkaOutput(producer=producer, topic="results").write(context, attribution)
    assert producer.flushes == []


def test_native_configuration_error_is_reported(mocker):
    native = pytest.importorskip("confluent_kafka")
    error = native.KafkaException(native.KafkaError(native.KafkaError._INVALID_ARG, "bad CA file"))
    mocker.patch.object(native, "Producer", side_effect=error)
    with pytest.raises(MCAError, match="invalid producer configuration") as caught:
        _build({"bootstrap_servers": "localhost:9092", "topic": "results"}, {})
    assert caught.value.__cause__ is error


def test_native_limits_match_largest_supported_options(mocker):
    """Accept native upper boundaries without overflow or leaving the size default."""
    native = pytest.importorskip("confluent_kafka")
    factory = mocker.patch.object(native, "Producer", return_value=FakeProducer())
    _build(
        {
            "bootstrap_servers": "localhost:9092",
            "topic": "results",
            "delivery_timeout_seconds": 2147483.647,
            "max_message_bytes": 1_000_000_000,
        },
        {},
    )
    config = factory.call_args.args[0]
    assert config["message.timeout.ms"] == 2147483647
    assert config["message.max.bytes"] == 1_000_000_000


@pytest.mark.parametrize("terminal_state", ["pass", "cancelled", "unknown"])
def test_non_failure_verdicts_with_empty_evidence_are_published(terminal_state, context):
    producer = FakeProducer()
    attribution = {
        "terminal_state": terminal_state,
        "sub_domain": "none",
        "verdict": {"root_symptom": None},
        "symptoms": {},
        "connector_reports": [],
    }
    KafkaOutput(producer=producer, topic="results").write(context, attribution)
    data = json.loads(producer.messages[0][1]["value"])["data"]
    assert data["terminal_state"] == terminal_state
    assert data["result"] == {"verdict": {"root_symptom": None}, "symptoms": {}}
    assert data["connector_reports"] == []
    assert data["tenant_sre_action"] == data["cloud_operator_sre_action"] == ""


@pytest.mark.parametrize("extra_bytes", [-1, 0])
def test_encoded_event_size_boundary(extra_bytes, mocker, context):
    """Measure encoded JSON bytes; an event exactly at the limit is permitted."""
    mocker.patch("attribution_blueprint.connectors.outputs.kafka.connector._event", return_value={"data": "é"})
    encoded = b'{"data":"\\u00e9"}'
    producer = FakeProducer()
    output = KafkaOutput(producer=producer, topic="results", max_message_bytes=len(encoded) + extra_bytes)
    if extra_bytes < 0:
        with pytest.raises(KafkaOutputError, match="over max_message_bytes"):
            output.write(context, {})
        assert not producer.queued
    else:
        output.write(context, {})
        assert producer.messages[0][1]["value"] == encoded


def test_close_exception_still_prevents_reuse(mocker, context, attribution):
    producer = FakeProducer()
    error = RuntimeError("flush failed")
    flush = mocker.patch.object(producer, "flush", side_effect=error)
    output = KafkaOutput(producer=producer, topic="results")
    with pytest.raises(RuntimeError) as caught:
        output.close()
    assert caught.value is error
    with pytest.raises(KafkaOutputError, match="closed"):
        output.write(context, attribution)
    output.close()
    flush.assert_called_once()
