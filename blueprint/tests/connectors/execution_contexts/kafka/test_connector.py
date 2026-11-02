# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from attribution_blueprint.connectors.execution_contexts.kafka import (
    KafkaExecutionContextAdapter,
    _build as build_kafka,
)
from attribution_blueprint.connectors.execution_contexts.kafka.connector import KafkaConsumerError
from attribution_blueprint.exceptions import MCAError

FIXTURE = Path(__file__).parents[3] / "fixtures" / "kafka_completion_events.json"
EVENT_TYPE = "com.example.job.completed.v1"
TOPIC = "job-events"


def _events() -> list[dict]:
    return json.loads(FIXTURE.read_text())


class TP:
    """Stand-in for `confluent_kafka.TopicPartition`."""

    def __init__(self, topic, partition, offset=-1001, error=None):
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.error = error


class FakeError:
    def __init__(self, text, *, retriable=False, fatal=False):
        self._text = text
        self._retriable = retriable
        self._fatal = fatal

    def retriable(self):
        return self._retriable

    def fatal(self):
        return self._fatal

    def __str__(self):
        return self._text


class FakeMessage:
    def __init__(self, value, offset, partition=0, headers=None, error=None):
        self._value = value
        self._offset = offset
        self._partition = partition
        self._headers = headers
        self._error = error

    def error(self):
        return self._error

    def value(self):
        return self._value

    def headers(self):
        return self._headers

    def topic(self):
        return TOPIC

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset


class FakeConsumer:
    """Replays `script`: a message, None for an empty poll, or a callable run
    inside `poll` (a rebalance callback). Partitions are assigned after
    `unassigned_polls` polls, like a fresh group join."""

    def __init__(self, script, unassigned_polls=0, partitions=(0,), commit_errors=(), commit_raises=False):
        self._script = list(script)
        self._unassigned_polls = unassigned_polls
        self.partitions = set(partitions)
        self._commit_errors = set(commit_errors)
        self._commit_raises = commit_raises
        self.polls = 0
        self.commits = []
        self.seeks = []
        self.closed = False

    def subscribe(self, topics, *, on_revoke, on_lost):
        assert topics == [TOPIC]
        self.on_revoke = on_revoke
        self.on_lost = on_lost

    def poll(self, timeout):
        self.polls += 1
        while self._script and callable(self._script[0]):
            self._script.pop(0)(self)
        return self._script.pop(0) if self._script else None

    def assignment(self):
        if self.polls <= self._unassigned_polls:
            return []
        return [TP(TOPIC, p) for p in sorted(self.partitions)]

    def commit(self, *, offsets, asynchronous):
        assert asynchronous is False
        if self._commit_raises:
            raise RuntimeError("coordinator unavailable")
        self.commits.append({tp.partition: tp.offset for tp in offsets})
        return [
            TP(tp.topic, tp.partition, tp.offset, "commit failed" if tp.partition in self._commit_errors else None)
            for tp in offsets
        ]

    def seek(self, tp):
        self.seeks.append((tp.partition, tp.offset))

    def close(self):
        assert not self.closed, "consumer closed twice"
        self.closed = True


def revoke(partition):
    def run(consumer):
        consumer.partitions.discard(partition)
        consumer.on_revoke(consumer, [TP(TOPIC, partition)])

    return run


def lose(partition):
    def run(consumer):
        consumer.partitions.discard(partition)
        consumer.on_lost(consumer, [TP(TOPIC, partition)])

    return run


def _structured(events, partition=0, first_offset=0) -> list[FakeMessage]:
    return [FakeMessage(json.dumps(event).encode(), first_offset + i, partition) for i, event in enumerate(events)]


def _make_eca(script, unassigned_polls=0, partitions=(0,), commit_errors=(), commit_raises=False, **overrides):
    consumer = FakeConsumer(script, unassigned_polls, partitions, commit_errors, commit_raises)
    kwargs = {
        "consumer": consumer,
        "topic": TOPIC,
        "topic_partition": TP,
        "event_types": [EVENT_TYPE],
        "poll_timeout_seconds": 0.01,
    }
    kwargs.update(overrides)
    return KafkaExecutionContextAdapter(**kwargs), consumer


def test_discover_maps_structured_event_to_context():
    eca, consumer = _make_eca(_structured(_events()))

    contexts = eca.discover()

    assert [ctx["id"] for ctx in contexts] == ["job-a1", "job-b7"]
    ctx = contexts[0]
    assert ctx["job_id"] == "job-a1"
    assert ctx["attempt_id"] == "2"
    assert ctx["scheduler"] == "example"
    assert ctx["workload_type"] == ""
    assert ctx["state"] == "FAILED"
    assert ctx["event_id"] == "0b7e5a52-1c1e-4b7e-9f51-6a7d0c2b9e01"
    assert ctx["start_time"] == datetime(2026, 10, 21, 14, 5, 31, tzinfo=timezone.utc)
    assert ctx["end_time"] == datetime(2026, 10, 21, 17, 41, 58, tzinfo=timezone.utc)
    assert ctx["timezone"] == "America/Los_Angeles"
    assert ctx["nodes"] == ["example-node-001", "example-node-002"]
    assert ctx["gpu_count"] == 16
    assert ctx["time_limit"] == 480
    assert ctx["failed_node"] == "example-node-002"
    assert ctx["failed_nodes"] == ["example-node-002"]
    assert ctx["namespace"] == "example-team"
    assert ctx["pod_nodes"]["job-a1-worker-1"] == "example-node-002"
    assert "cancelled_by" not in ctx
    assert consumer.commits == [{0: 2}]


def test_discover_reads_binary_mode_event():
    event = _events()[1]
    headers = [
        ("ce_specversion", b"1.0"),
        ("ce_id", event["id"].encode()),
        ("ce_type", EVENT_TYPE.encode()),
        ("ce_source", b"/example/scheduler"),
    ]
    eca, _ = _make_eca([FakeMessage(json.dumps(event["data"]).encode(), 0, headers=headers)])

    contexts = eca.discover()

    assert [ctx["id"] for ctx in contexts] == ["job-b7"]
    assert contexts[0]["event_id"] == event["id"]


def test_discover_drops_redelivery_of_same_attempt():
    first = _events()[0]
    resend = copy.deepcopy(first)
    resend["id"] = "a-new-id-per-send"
    eca, _ = _make_eca(_structured([first, resend]))

    contexts = eca.discover()

    assert len(contexts) == 1
    assert contexts[0]["event_id"] == first["id"]


def test_discover_keeps_distinct_attempts():
    first = _events()[0]
    retry = copy.deepcopy(first)
    retry["id"] = "retry-event"
    retry["data"]["attempt_id"] = "3"
    eca, _ = _make_eca(_structured([first, retry]))

    assert [ctx["attempt_id"] for ctx in eca.discover()] == ["2", "3"]


def test_discover_ignores_other_event_types():
    other = _events()[0]
    other["type"] = "com.example.job.started.v1"
    eca, consumer = _make_eca(_structured([other, _events()[1]]))

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.commits == [{0: 2}]


def test_discover_accepts_every_type_when_unfiltered():
    other = _events()[0]
    other["type"] = "com.example.anything.v1"
    eca, _ = _make_eca(_structured([other]), event_types=None)

    assert len(eca.discover()) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e["data"].pop("nodes"),
        lambda e: e["data"].update(nodes=[]),
        lambda e: e["data"].pop("id"),
        lambda e: e["data"].update(start_time="2026-10-21T14:05:31"),
        lambda e: e["data"].update(end_time="not-a-time"),
        lambda e: e["data"].update(end_time="2026-10-21T00:00:00Z"),
        lambda e: e["data"].update(timezone="Mars/Olympus"),
        # A non-hour offset without an IANA `timezone` fails at the service.
        lambda e: e["data"].update(
            timezone="", start_time="2026-10-21T14:05:31+05:30", end_time="2026-10-21T17:41:58+05:30"
        ),
        lambda e: e["data"].update(gpu_count=-1),
        lambda e: e["data"].update(pod_nodes=["not", "a", "map"]),
        lambda e: e.update(specversion="0.3"),
        lambda e: e.pop("id"),
        lambda e: e.update(data="not-an-object"),
    ],
)
def test_malformed_event_is_skipped_without_blocking(mutate):
    bad = _events()[0]
    mutate(bad)
    eca, consumer = _make_eca(_structured([bad, _events()[1]]))

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.commits == [{0: 2}]


def test_non_hour_offset_accepted_with_iana_timezone():
    event = _events()[0]
    event["data"].update(
        timezone="Asia/Kolkata", start_time="2026-10-21T14:05:31+05:30", end_time="2026-10-21T17:41:58+05:30"
    )
    eca, _ = _make_eca(_structured([event]))

    assert [ctx["timezone"] for ctx in eca.discover()] == ["Asia/Kolkata"]


def test_non_json_value_is_skipped():
    eca, _ = _make_eca([FakeMessage(b"\xff not json", 0), *_structured(_events()[1:])])

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]


def test_non_retriable_consumer_error_raises_without_commit():
    error = FakeError("topic authorization failed")
    eca, consumer = _make_eca([*_structured(_events()[:1]), FakeMessage(None, 0, error=error)])

    with pytest.raises(KafkaConsumerError, match="topic authorization failed"):
        eca.discover()
    assert consumer.commits == []


def test_fatal_consumer_error_raises():
    error = FakeError("fenced", retriable=True, fatal=True)
    eca, _ = _make_eca([FakeMessage(None, 0, error=error)])

    with pytest.raises(KafkaConsumerError, match="fenced"):
        eca.discover()


def test_retriable_consumer_errors_are_retried():
    error = FakeError("broker transport failure", retriable=True)
    eca, _ = _make_eca([FakeMessage(None, 0, error=error), *_structured(_events())])

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]


def test_retriable_consumer_errors_are_bounded():
    error = FakeError("broker transport failure", retriable=True)
    eca, consumer = _make_eca([FakeMessage(None, 0, error=error)] * 3, max_transient_errors=2)

    with pytest.raises(KafkaConsumerError, match="broker transport failure"):
        eca.discover()
    assert consumer.commits == []


def test_discover_waits_for_partition_assignment():
    eca, _ = _make_eca([None, None, *_structured(_events())], unassigned_polls=2)

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]


def test_discover_polls_again_when_assignment_completes_on_empty_poll():
    eca, consumer = _make_eca([None, *_structured(_events())])

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]
    assert consumer.commits == [{0: 2}]


def test_discover_raises_when_never_assigned():
    eca, consumer = _make_eca([], unassigned_polls=10**9, assignment_timeout_seconds=0)

    with pytest.raises(KafkaConsumerError, match="no partitions assigned"):
        eca.discover()
    assert consumer.polls == 1


def test_commits_explicit_offsets_per_partition():
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0, first_offset=7), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]
    assert consumer.commits == [{0: 8, 1: 1}]


def test_revoked_partition_is_committed_before_release():
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0), revoke(0), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]
    assert consumer.commits == [{0: 1}, {1: 1}]


def test_revoked_partition_with_failed_commit_is_dropped():
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0), revoke(0), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
        commit_errors={0},
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.seeks == []  # no longer owned; the new owner rereads it


def test_lost_partition_is_dropped_and_not_committed():
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0), lose(0), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.commits == [{1: 1}]


def test_reassigned_dropped_partition_is_ignored_and_rewound():
    def reassign(consumer):
        consumer.partitions.add(0)

    first, later = _events()
    eca, consumer = _make_eca(
        [
            *_structured([first], partition=0, first_offset=4),
            lose(0),
            reassign,
            *_structured([later], partition=0, first_offset=4),
        ],
    )

    assert eca.discover() == []
    assert consumer.commits == []
    assert consumer.seeks == [(0, 4)]


def test_partial_commit_failure_drops_and_rewinds_that_partition():
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0, first_offset=3), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
        commit_errors={0},
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.seeks == [(0, 3)]


def test_commit_exception_raises():
    eca, _ = _make_eca(_structured(_events()), commit_raises=True)

    with pytest.raises(KafkaConsumerError, match="offset commit failed"):
        eca.discover()


def test_defaults_when_optional_fields_absent():
    event = _events()[1]
    for key in ("job_id", "attempt_id", "scheduler", "state"):
        event["data"].pop(key)
    eca, _ = _make_eca(_structured([event]), scheduler="fallback", workload_type="kubernetes")

    ctx = eca.discover()[0]
    assert ctx["job_id"] == "job-b7"
    assert ctx["attempt_id"] == "0"
    assert ctx["scheduler"] == "fallback"
    assert ctx["state"] == "UNKNOWN"
    assert ctx["workload_type"] == "kubernetes"


def test_discover_stops_at_max_messages():
    eca, consumer = _make_eca(_structured(_events()), max_messages=1)

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1"]
    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.commits == [{0: 1}, {0: 2}]


def test_empty_topic_returns_nothing_and_does_not_commit():
    eca, consumer = _make_eca([])

    assert eca.discover() == []
    assert consumer.commits == []


def test_discover_by_ids_not_supported():
    eca, _ = _make_eca([])
    with pytest.raises(NotImplementedError):
        eca.discover_by_ids(["job-a1"])


def test_discover_by_time_range_not_supported():
    eca, _ = _make_eca([])
    with pytest.raises(NotImplementedError):
        eca.discover_by_time_range(datetime.now(timezone.utc), datetime.now(timezone.utc))


def test_close_closes_consumer_once():
    eca, consumer = _make_eca([])
    eca.close()
    eca.close()
    assert consumer.closed


def test_consumer_closed_when_adapter_is_collected():
    eca, consumer = _make_eca([])
    del eca
    assert consumer.closed


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_messages", 0),
        ("poll_timeout_seconds", 0),
        ("assignment_timeout_seconds", -1),
        ("max_transient_errors", -1),
    ],
)
def test_constructor_rejects_non_positive_limits(field, value):
    with pytest.raises(ValueError, match=field):
        _make_eca([], **{field: value})


def test_build_requires_connection_keys():
    with pytest.raises(MCAError, match="bootstrap_servers"):
        build_kafka({"topic": "job-events", "group_id": "attribution"}, {})


def test_build_requires_sasl_env(monkeypatch):
    monkeypatch.delenv("EXAMPLE_KAFKA_USER", raising=False)
    spec = {
        "bootstrap_servers": "localhost:9092",
        "topic": "job-events",
        "group_id": "attribution",
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "SCRAM-SHA-512",
        "sasl_username_env": "EXAMPLE_KAFKA_USER",
        "sasl_password_env": "EXAMPLE_KAFKA_PASSWORD",
    }
    with pytest.raises(MCAError, match="EXAMPLE_KAFKA_USER"):
        build_kafka(spec, {})


def test_build_rejects_unknown_sasl_mechanism():
    spec = {
        "bootstrap_servers": "localhost:9092",
        "topic": "job-events",
        "group_id": "attribution",
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "GSSAPI",
    }
    with pytest.raises(MCAError, match="sasl_mechanism"):
        build_kafka(spec, {})
