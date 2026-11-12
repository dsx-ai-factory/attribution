# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check Kafka consumer measurements with fake consumers and an in-memory reader."""

import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from attribution_blueprint.connectors.execution_contexts.kafka import connector as connector_module
from attribution_blueprint.connectors.execution_contexts.kafka.connector import KafkaConsumerError

from .connectors.execution_contexts.kafka.test_connector import (
    TOPIC,
    TP,
    FakeError,
    FakeMessage,
    _events,
    _make_eca,
    _structured,
)


def points(reader, name, group):
    """Select one consumer group's data points for a Kafka consumer metric.

    Args:
        reader: Metric reader for this process.
        name: Metric suffix after `attribution.kafka.consumer.`.
        group: Consumer group label that isolates one test.

    Returns:
        Matching data points.
    """
    data = reader.get_metrics_data()
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == f"attribution.kafka.consumer.{name}"
        for point in metric.data.data_points
        if point.attributes.get("consumer_group") == group
    ]


def by_partition(reader, name, group):
    """Map each partition to its value.

    Args:
        reader: Metric reader for this process.
        name: Metric suffix after `attribution.kafka.consumer.`.
        group: Consumer group label that isolates one test.

    Returns:
        Values keyed by partition.
    """
    return {point.attributes["partition"]: point.value for point in points(reader, name, group)}


@pytest.fixture
def clock(monkeypatch):
    """Control monotonic time for delivery ages and refresh intervals.

    Args:
        monkeypatch: Pytest patch helper.

    Returns:
        A namespace whose `now` attribute is the current monotonic time.
    """
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    return clock


@pytest.fixture
def stream():
    """Create streaming adapters and close them after the test.

    Returns:
        A function with the arguments of `_make_eca`.
    """
    adapters = []

    def create(script, **kwargs):
        adapter, consumer = _make_eca(script, **kwargs)
        adapters.append(adapter)
        return adapter, consumer

    yield create
    for adapter in adapters:
        adapter.close()


def test_poll_and_ack_count_consumed_skipped_accepted_acknowledged_and_committed_records(reader, stream):
    """Count each record once by outcome, including repeat acknowledgements.

    Args:
        reader: Metric reader for this process.
        stream: Adapter factory.
    """
    group = "counts"
    filtered = {**_events()[0], "type": "com.example.other.v1"}
    adapter, consumer = stream(
        [
            *_structured(_events()[:1]),
            FakeMessage(b"not JSON", 1),
            *_structured([filtered], first_offset=2),
            *_structured(_events()[1:], first_offset=3),
        ],
        group_id=group,
    )
    first, last = adapter.poll(4, 0)
    adapter.ack(first)
    adapter.ack(last)
    adapter.ack(last)

    assert consumer.commits == [{0: 1}, {0: 4}]
    assert by_partition(reader, "consumed", group) == {0: 4}
    assert by_partition(reader, "accepted", group) == {0: 2}
    assert {point.attributes["reason"]: point.value for point in points(reader, "skipped", group)} == {
        "filtered": 1,
        "malformed": 1,
    }
    assert by_partition(reader, "acknowledged", group) == {0: 2}
    assert by_partition(reader, "commits", group) == {0: 2}
    assert points(reader, "commit.errors", group) == []
    (commit,) = points(reader, "commits", group)
    assert commit.attributes == {"topic": TOPIC, "partition": 0, "consumer_group": group}


def test_discovery_counts_records_and_commits(reader):
    """Count bounded discovery reads with the same instruments as streaming.

    Args:
        reader: Metric reader for this process.
    """
    group = "discovery"
    adapter, _ = _make_eca(_structured(_events()), group_id=group)
    adapter.discover()

    assert by_partition(reader, "consumed", group) == {0: 2}
    assert by_partition(reader, "accepted", group) == {0: 2}
    assert by_partition(reader, "commits", group) == {0: 1}


def test_unacknowledged_record_reports_lag_backlog_and_age_until_committed(reader, stream, clock):
    """Show a commit held behind the oldest unacknowledged record.

    Args:
        reader: Metric reader for this process.
        stream: Adapter factory.
        clock: Controlled monotonic time.
    """
    group = "backlog"
    adapter, _ = stream(_structured(_events()), group_id=group)
    first, last = adapter.poll(2, 0)
    adapter.ack(last)
    clock.now = 160.0

    assert by_partition(reader, "lag", group) == {0: 2}
    assert by_partition(reader, "position.lag", group) == {0: 0}
    assert by_partition(reader, "unacknowledged", group) == {0: 1}
    assert by_partition(reader, "oldest_unacknowledged.age", group) == {0: 60.0}

    adapter.ack(first)
    assert by_partition(reader, "lag", group) == {0: 0}
    assert by_partition(reader, "unacknowledged", group) == {0: 0}
    assert by_partition(reader, "oldest_unacknowledged.age", group) == {0: 0.0}


def test_rebalance_counts_changes_and_reports_only_owned_partitions(reader, stream):
    """Stop reporting a partition after it is revoked or lost.

    Args:
        reader: Metric reader for this process.
        stream: Adapter factory.
    """
    group = "rebalance"

    def assign(consumer):
        consumer.on_assign(consumer, [TP(TOPIC, 0), TP(TOPIC, 1)])

    adapter, consumer = stream(
        [assign, *_structured(_events()[:1], partition=0), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
        group_id=group,
    )
    assert len(adapter.poll(2, 0)) == 2
    assert by_partition(reader, "unacknowledged", group) == {0: 1, 1: 1}

    consumer.on_revoke(consumer, [TP(TOPIC, 1)])
    assert by_partition(reader, "unacknowledged", group) == {0: 1}
    consumer.on_lost(consumer, [TP(TOPIC, 0)])
    assert by_partition(reader, "unacknowledged", group) == {}
    assert {
        (point.attributes["partition"], point.attributes["change"]): point.value
        for point in points(reader, "rebalances", group)
    } == {(0, "assigned"): 1, (1, "assigned"): 1, (1, "revoked"): 1, (0, "lost"): 1}


def test_commit_failure_counts_error_type_and_closed_adapter_stops_gauges(reader):
    """Label a failed commit with its exception class, then drop gauges on close.

    Args:
        reader: Metric reader for this process.
    """
    group = "commit-failure"
    adapter, consumer = _make_eca(_structured(_events()[:1]), commit_raises=True, group_id=group)
    (delivery,) = adapter.poll(1, 0)
    assert by_partition(reader, "unacknowledged", group) == {0: 1}

    with pytest.raises(KafkaConsumerError):
        adapter.ack(delivery)
    assert consumer.closed
    (failure,) = points(reader, "commit.errors", group)
    assert failure.value == 1
    assert failure.attributes["error.type"] == "KafkaConsumerError"
    assert points(reader, "commits", group) == []
    assert points(reader, "unacknowledged", group) == []


def test_poll_error_counts_error_name_but_not_partition_eof(reader, stream):
    """Count native poll errors by name and ignore end-of-partition events.

    Args:
        reader: Metric reader for this process.
        stream: Adapter factory.
    """
    group = "poll-error"
    adapter, _ = stream(
        [FakeMessage(None, 0, error=FakeError("_PARTITION_EOF")), FakeMessage(None, 0, error=FakeError("_TRANSPORT"))],
        group_id=group,
    )
    with pytest.raises(KafkaConsumerError, match="_TRANSPORT"):
        adapter.poll(2, 0)
    assert {point.attributes["error.type"]: point.value for point in points(reader, "errors", group)} == {
        "_TRANSPORT": 1
    }


def test_offset_refresh_reads_cached_watermarks_at_most_once_per_interval(stream, clock, mocker):
    """Read client-side watermarks only, and not on every poll.

    Args:
        stream: Adapter factory.
        clock: Controlled monotonic time.
        mocker: Pytest mock helper.
    """
    adapter, consumer = stream([], group_id="refresh")
    watermarks = mocker.spy(consumer, "get_watermark_offsets")
    adapter.poll(1, 0)
    clock.now = 104.9
    adapter.poll(1, 0)
    assert watermarks.call_count == 1
    assert watermarks.call_args.kwargs == {"cached": True}
    clock.now = 105.0
    adapter.poll(1, 0)
    assert watermarks.call_count == 2


def test_offset_refresh_failure_keeps_polling_and_logs_warning(stream, mocker):
    """Keep the consumer open when the lag refresh fails.

    Args:
        stream: Adapter factory.
        mocker: Pytest mock helper.
    """
    adapter, consumer = stream(_structured(_events()[:1]), group_id="refresh-failure")
    mocker.patch.object(consumer, "get_watermark_offsets", side_effect=KafkaConsumerError("cache unavailable"))
    warning = mocker.patch.object(connector_module.logger, "warning")

    assert len(adapter.poll(1, 0)) == 1
    assert not consumer.closed
    warning.assert_called_once_with("kafka ECA: consumer lag refresh failed: cache unavailable")


def test_prometheus_endpoint_renders_kafka_consumer_metrics():
    """Verify the scrape names and labels in a separate provider process."""
    pytest.importorskip("opentelemetry.exporter.prometheus")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import socket
from urllib.request import urlopen
from attribution_blueprint.connectors.execution_contexts.kafka.receipts import Backlog
from attribution_blueprint.connectors.execution_contexts.kafka.telemetry import (
    ConsumerMetrics,
    PartitionChange,
    SkipReason,
)
from attribution_blueprint.telemetry import enable_prometheus_endpoint

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
enable_prometheus_endpoint("127.0.0.1", port)
key = ("jobs", 0)
consumer = ConsumerMetrics(topic="jobs", group_id="scrape", backlog=lambda: {key: Backlog(1, 0.0)})
consumer.partitions_changed(PartitionChange.ASSIGNED, [key])
consumer.consumed(key, 5)
consumer.accepted(key)
consumer.skipped(key, SkipReason.MALFORMED)
consumer.acknowledged(key)
consumer.committed(key, 6)
consumer.commit_failed(key, "KafkaException")
consumer.consumer_error("_TRANSPORT")
consumer.positions(key, position=7, high=9)
with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
    print(response.read().decode())
""",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    for name in (
        "attribution_kafka_consumer_consumed_total",
        "attribution_kafka_consumer_accepted_total",
        "attribution_kafka_consumer_skipped_total",
        "attribution_kafka_consumer_acknowledged_total",
        "attribution_kafka_consumer_commits_total",
        "attribution_kafka_consumer_commit_errors_total",
        "attribution_kafka_consumer_errors_total",
        "attribution_kafka_consumer_rebalances_total",
        "attribution_kafka_consumer_lag{",
        "attribution_kafka_consumer_position_lag{",
        "attribution_kafka_consumer_unacknowledged{",
        "attribution_kafka_consumer_oldest_unacknowledged_age_seconds{",
    ):
        assert name in result.stdout
    for label in (
        'topic="jobs"',
        'partition="0"',
        'consumer_group="scrape"',
        'reason="malformed"',
        'change="assigned"',
        'error_type="_TRANSPORT"',
    ):
        assert label in result.stdout
