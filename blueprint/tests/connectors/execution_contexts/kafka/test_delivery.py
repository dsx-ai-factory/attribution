# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State and failure contracts over real Kafka and a disposable Service."""

import copy
import json
import os
import sqlite3
import subprocess
from uuid import uuid4

import pytest

from attribution_blueprint.connectors.attribution import AttributionConnector
from attribution_blueprint.connectors.execution_contexts.kafka import _build, _consumer_config
from attribution_blueprint.connectors.execution_contexts.kafka.client import KafkaConsumerClient
from attribution_blueprint.connectors.execution_contexts.kafka.connector import (
    KafkaConsumerError,
    KafkaExecutionContextAdapter,
)
from attribution_blueprint.connectors.outputs.sqlite import SQLiteOutput
from attribution_blueprint.orchestrator import AttributionOrchestrator

pytestmark = pytest.mark.integration


def _assert_drained(spec):
    restarted = _build(spec, {})
    try:
        assert restarted.discover() == []
    finally:
        restarted.close()


@pytest.mark.parametrize(
    "state,cancelled_by,outcome,terminal_state,symptom,time_limit",
    [
        ("COMPLETED", "", "Pass", "pass", None, 480),
        ("FAILED", "", "Fail User", "fail", "job failed", 480),
        ("CANCELLED", "example-user", "Cancelled by User", "cancelled", "Cancelled by user", 480),
        (
            "CANCELLED",
            "site-automation-account",
            "Cancelled by cluster operators",
            "cancelled",
            "Cancelled by cluster operators",
            480,
        ),
        # Existing Service policy classifies timeout-only jobs with a known
        # wall-clock limit as Pass. Omitting the limit keeps the Fail User fallback.
        ("TIMEOUT", "", "Pass", "pass", "Cancelled by Timeout", 480),
        ("TIMEOUT", "", "Fail User", "fail", "Cancelled by Timeout", None),
    ],
)
def test_state_without_log_evidence(
    kafka_topic,
    kafka_spec,
    completion_event,
    publish_event,
    tmp_path,
    state,
    cancelled_by,
    outcome,
    terminal_state,
    symptom,
    time_limit,
):
    completion_event["data"].update(state=state, cancelled_by=cancelled_by, failed_nodes=[])
    if time_limit is None:
        completion_event["data"].pop("time_limit")
    else:
        completion_event["data"]["time_limit"] = time_limit
    publish_event(completion_event)
    database = tmp_path / "state.db"
    adapter = _build(kafka_spec, {})
    try:
        pipeline = AttributionOrchestrator(
            execution_context_adapters=[adapter],
            log_connectors=[],
            metrics_connectors=[],
            attribution=AttributionConnector(kafka_topic[1]),
            outputs=[SQLiteOutput(str(database))],
        )
        results = pipeline.analyze()
        assert pipeline.last_run_stats == {"attempted": 1, "succeeded": 1, "conflicted": 0}
    finally:
        adapter.close()
    assert len(results) == 1
    result = results[0]
    assert result["verdict"]["attribution_outcome"] == outcome
    assert result["terminal_state"] == terminal_state
    assert result["evidence_status"] == "none"
    if symptom:
        assert symptom in result["symptoms"]
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT outcome, raw_result_json FROM attribution_run").fetchone()
        raw = json.loads(connection.execute("SELECT raw_scheduler_json FROM execution_context").fetchone()[0])
    assert row[0] == outcome
    assert json.loads(row[1]) == result
    assert raw["state"] == state
    if cancelled_by:
        assert raw["cancelled_by"] == cancelled_by
    _assert_drained(kafka_spec)


@pytest.mark.parametrize("failure", ["service", "sqlite"])
def test_cli_failure_after_discovery_is_nonzero_and_event_is_not_replayed(
    failure,
    kafka_spec,
    completion_event,
    publish_event,
    run_kafka_cli,
    unavailable_endpoint,
    tmp_path,
):
    publish_event(completion_event)
    if failure == "service":
        completed = run_kafka_cli(service=f"http://{unavailable_endpoint}")
        assert "processing failed" in completed.stdout + completed.stderr
    else:
        # A directory cannot be opened as a SQLite database.
        completed = run_kafka_cli(database=tmp_path)
        assert "write failed" in completed.stdout + completed.stderr
    assert completed.returncode == 1
    # This explicitly locks in the documented best-effort delivery tradeoff:
    # attribution/output failure happens after discovery has committed.
    _assert_drained(kafka_spec)


def test_unreachable_broker_returns_nonzero(kafka_spec, run_kafka_cli, unavailable_endpoint):
    spec = {
        **kafka_spec,
        "bootstrap_servers": unavailable_endpoint,
        "poll_timeout_seconds": 0.2,
        "assignment_timeout_seconds": 1,
    }
    completed = run_kafka_cli(spec=spec)
    assert completed.returncode == 1
    assert "kafka ECA:" in completed.stdout + completed.stderr


def test_broker_interruption_preserves_event_for_retry(
    kafka_spec,
    completion_event,
    publish_event,
    run_kafka_cli,
):
    # Opt in only for a broker dedicated to this sequential test run.
    container = os.environ.get("ATTRIBUTION_TEST_KAFKA_CONTAINER")
    if not container:
        pytest.skip("set ATTRIBUTION_TEST_KAFKA_CONTAINER to a disposable Docker broker")
    publish_event(completion_event)
    subprocess.run(["docker", "pause", container], check=True, capture_output=True, timeout=10)
    try:
        completed = run_kafka_cli(
            spec={
                **kafka_spec,
                "assignment_timeout_seconds": 1,
                "poll_timeout_seconds": 0.2,
            }
        )
    finally:
        subprocess.run(["docker", "unpause", container], check=True, capture_output=True, timeout=10)
    assert completed.returncode == 1
    adapter = _build(kafka_spec, {})
    try:
        assert [ctx["event_id"] for ctx in adapter.discover()] == [completion_event["id"]]
    finally:
        adapter.close()
    _assert_drained(kafka_spec)


class ConsumerProxy:
    """Delegate to a real consumer; subclasses inject only a commit failure."""

    def __init__(self, consumer):
        self.consumer = consumer

    def __getattr__(self, name):
        return getattr(self.consumer, name)


def _wrapped_adapter(spec, wrapper):
    from confluent_kafka import Consumer, TopicPartition

    return KafkaExecutionContextAdapter(
        consumer=KafkaConsumerClient(wrapper(Consumer(_consumer_config(spec)))),
        topic=spec["topic"],
        topic_partition=TopicPartition,
        event_types=spec["event_types"],
        poll_timeout_seconds=spec.get("poll_timeout_seconds", 1),
        fetch_timeout_seconds=spec.get("fetch_timeout_seconds", 60),
    )


def test_partial_commit_failure_rewinds_only_failed_partition(kafka_spec, completion_event, publish_event):
    from confluent_kafka import TopicPartition

    first = completion_event
    second = copy.deepcopy(first)
    second["id"] = str(uuid4())
    second["data"]["id"] += "-other"
    second["data"]["job_id"] += "-other"
    second["subject"] = f"jobs/{second['data']['id']}"
    publish_event(first, partition=0)
    publish_event(second, partition=1)

    class FailOnePartitionOnce(ConsumerProxy):
        failed = False

        def commit(self, *, offsets, asynchronous):
            assert len(offsets) == 1
            if offsets[0].partition == 1 and not self.failed:
                self.failed = True
                # Ask the real broker for an invalid partition. The native
                # client raises KafkaException, rather than returning a fake
                # per-partition error object.
                tp = offsets[0]
                return self.consumer.commit(
                    offsets=[TopicPartition(tp.topic, 999, tp.offset)], asynchronous=asynchronous
                )
            return self.consumer.commit(offsets=offsets, asynchronous=asynchronous)

    adapter = _wrapped_adapter(kafka_spec, FailOnePartitionOnce)
    try:
        assert [ctx["event_id"] for ctx in adapter.discover()] == [first["id"]]
        assert [ctx["event_id"] for ctx in adapter.discover()] == [second["id"]]
    finally:
        adapter.close()
    _assert_drained(kafka_spec)


@pytest.mark.parametrize("restart", [False, True])
def test_commit_exception_replays_event_on_retry(kafka_spec, completion_event, publish_event, restart):
    publish_event(completion_event)

    class FailCommitOnce(ConsumerProxy):
        failed = False

        def commit(self, **kwargs):
            if not self.failed:
                self.failed = True
                raise KafkaConsumerError("injected coordinator timeout")
            return self.consumer.commit(**kwargs)

    adapter = _wrapped_adapter(kafka_spec, FailCommitOnce)
    try:
        with pytest.raises(KafkaConsumerError, match="offset commit failed"):
            adapter.discover()
        if restart:
            # close() invokes a real revoke callback; it must not commit
            # the batch abandoned by the unsuccessful discover call.
            adapter.close()
            adapter = _build(kafka_spec, {})
        assert [ctx["event_id"] for ctx in adapter.discover()] == [completion_event["id"]]
    finally:
        adapter.close()
    _assert_drained(kafka_spec)


def test_topic_authorization_failure_is_nonzero_and_does_not_commit(
    kafka_topic,
    kafka_spec,
    completion_event,
    publish_event,
    run_kafka_cli,
):
    sasl_brokers = os.environ.get("ATTRIBUTION_TEST_KAFKA_SASL_BOOTSTRAP_SERVERS")
    username = os.environ.get("ATTRIBUTION_TEST_KAFKA_USERNAME")
    if not sasl_brokers or not username or not os.environ.get("ATTRIBUTION_TEST_KAFKA_PASSWORD"):
        pytest.skip("configure the disposable SASL listener and test username/password")
    from confluent_kafka.admin import (
        AclBinding,
        AclBindingFilter,
        AclOperation,
        AclPermissionType,
        AdminClient,
        ResourcePatternType,
        ResourceType,
    )

    brokers, _, topic = kafka_topic
    admin = AdminClient({"bootstrap.servers": brokers})
    acl = AclBinding(
        ResourceType.TOPIC,
        topic,
        ResourcePatternType.LITERAL,
        f"User:{username}",
        "*",
        AclOperation.READ,
        AclPermissionType.DENY,
    )
    acl_filter = AclBindingFilter(
        ResourceType.TOPIC,
        topic,
        ResourcePatternType.LITERAL,
        f"User:{username}",
        "*",
        AclOperation.READ,
        AclPermissionType.DENY,
    )
    for future in admin.create_acls([acl], request_timeout=15).values():
        future.result(20)
    try:
        publish_event(completion_event)
        completed = run_kafka_cli(
            spec={
                **kafka_spec,
                "bootstrap_servers": sasl_brokers,
                "security_protocol": "SASL_PLAINTEXT",
                "sasl_mechanism": "PLAIN",
                "sasl_username_env": "ATTRIBUTION_TEST_KAFKA_USERNAME",
                "sasl_password_env": "ATTRIBUTION_TEST_KAFKA_PASSWORD",
            }
        )
        assert completed.returncode == 1
        assert "TOPIC_AUTHORIZATION_FAILED" in completed.stdout + completed.stderr
    finally:
        for future in admin.delete_acls([acl_filter], request_timeout=15).values():
            future.result(20)
    # Authorized access in the same group still sees the rejected event.
    adapter = _build(kafka_spec, {})
    try:
        assert [ctx["event_id"] for ctx in adapter.discover()] == [completion_event["id"]]
    finally:
        adapter.close()


def test_native_mixed_commit_raises_after_another_partition_commits(kafka_spec, completion_event, publish_event):
    from confluent_kafka import Consumer, KafkaException, TopicPartition

    publish_event(completion_event)
    consumer = Consumer(_consumer_config(kafka_spec))
    topic = kafka_spec["topic"]
    try:
        with pytest.raises(KafkaException):
            consumer.commit(offsets=[TopicPartition(topic, 0, 1), TopicPartition(topic, 999, 1)], asynchronous=False)
        # This is why the adapter must not treat a mixed-call exception as
        # proof that none of its partitions committed.
        assert consumer.committed([TopicPartition(topic, 0)], timeout=10)[0].offset == 1
    finally:
        consumer.close()


def test_short_polls_wait_for_delayed_real_fetch(kafka_spec, completion_event, publish_event):
    publish_event(completion_event)

    class DelayedFetch(ConsumerProxy):
        def __init__(self, consumer):
            super().__init__(consumer)
            self.paused = []
            self.empty_polls = 0

        def subscribe(self, topics, **callbacks):
            def assigned(consumer, partitions):
                consumer.assign(partitions)
                consumer.pause(partitions)
                self.paused = partitions

            self.consumer.subscribe(topics, on_assign=assigned, **callbacks)

        def poll(self, timeout):
            message = self.consumer.poll(timeout)
            if message is None and self.paused:
                self.empty_polls += 1
                if self.empty_polls == 8:
                    self.consumer.resume(self.paused)
                    self.paused = []
            return message

    adapter = _wrapped_adapter({**kafka_spec, "poll_timeout_seconds": 0.001}, DelayedFetch)
    try:
        assert [ctx["event_id"] for ctx in adapter.discover()] == [completion_event["id"]]
        assert adapter._consumer._consumer.empty_polls >= 8
    finally:
        adapter.close()


def test_poll_error_rewinds_for_same_consumer_retry(kafka_spec, completion_event, publish_event):
    publish_event(completion_event)

    class FailAfterReadOnce(ConsumerProxy):
        fail_next = False
        failed = False

        def poll(self, timeout):
            if self.fail_next:
                self.fail_next = False
                self.failed = True
                raise KafkaConsumerError("injected read error after a real record")
            message = self.consumer.poll(timeout)
            if message is not None and message.error() is None and not self.failed:
                self.fail_next = True
            return message

    adapter = _wrapped_adapter(kafka_spec, FailAfterReadOnce)
    try:
        with pytest.raises(KafkaConsumerError, match="injected read error"):
            adapter.discover()
        assert [ctx["event_id"] for ctx in adapter.discover()] == [completion_event["id"]]
        assert adapter.discovery_incomplete is False
    finally:
        adapter.close()
    _assert_drained(kafka_spec)


def test_cli_reports_filtered_only_schema_change(kafka_spec, completion_event, publish_event, run_kafka_cli):
    completion_event["type"] = "com.example.job.completed.v2"
    publish_event(completion_event)
    completed = run_kafka_cli()
    output = completed.stdout + completed.stderr
    assert completed.returncode == 1
    assert completion_event["type"] in output
    assert "'filtered': 1" in output
    assert "'accepted': 0" in output
    _assert_drained(kafka_spec)  # Filtered events retain the documented commit policy.


def test_cli_surfaces_filtered_count_in_mixed_batch(
    completion_event,
    publish_event,
    run_kafka_cli,
    tmp_path,
):
    other = copy.deepcopy(completion_event)
    other["type"] = "com.example.job.completed.v2"
    other["id"] = str(uuid4())
    publish_event(other, partition=0)
    publish_event(completion_event, partition=1)
    database = tmp_path / "mixed.db"
    completed = run_kafka_cli(database=database)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0
    assert other["type"] in output
    assert "'filtered': 1" in output
    assert "'accepted': 1" in output
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM attribution_run").fetchone()[0] == 1


def test_latest_reset_uses_eof_without_skipping_future_records(kafka_spec, completion_event, publish_event):
    publish_event(completion_event)
    adapter = _build({**kafka_spec, "auto_offset_reset": "latest"}, {})
    try:
        assert adapter.discover() == []
        assert adapter.last_discovery_stats["consumed"] == 0
        later = copy.deepcopy(completion_event)
        later["id"] = str(uuid4())
        publish_event(later)
        assert [ctx["event_id"] for ctx in adapter.discover()] == [later["id"]]
    finally:
        adapter.close()


def test_unread_native_partition_has_bounded_fetch_wait(kafka_spec, completion_event, publish_event):
    publish_event(completion_event)

    class PausedFetch(ConsumerProxy):
        def subscribe(self, topics, **callbacks):
            def assigned(consumer, partitions):
                consumer.assign(partitions)
                consumer.pause(partitions)

            self.consumer.subscribe(topics, on_assign=assigned, **callbacks)

    adapter = _wrapped_adapter({**kafka_spec, "poll_timeout_seconds": 0.001, "fetch_timeout_seconds": 0.2}, PausedFetch)
    try:
        with pytest.raises(KafkaConsumerError, match="unread records|end offsets"):
            adapter.discover()
        assert adapter.last_discovery_stats["returned"] == 0
    finally:
        adapter.close()
