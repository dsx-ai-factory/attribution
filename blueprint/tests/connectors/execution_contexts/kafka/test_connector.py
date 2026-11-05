# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import errno
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from attribution_blueprint.connectors.execution_contexts.kafka import (
    KafkaExecutionContextAdapter,
    _build as build_kafka,
    events as events_module,
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
    def __init__(self, text):
        self._text = text

    def name(self):
        return self._text

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
        self._positions = {}
        self._highs = {}
        for message in self._script:
            if isinstance(message, FakeMessage) and message.error() is None:
                partition = message.partition()
                self._highs[partition] = max(self._highs.get(partition, 0), message.offset() + 1)
        self._unassigned_polls = unassigned_polls
        self.partitions = set(partitions)
        self._commit_errors = set(commit_errors)
        self._commit_raises = commit_raises
        self.polls = 0
        self.commits = []
        self.closed = False
        self.paused_partitions = set()

    def subscribe(self, topics, *, on_revoke, on_lost, on_assign=None):
        assert topics == [TOPIC]
        self.on_assign = on_assign
        self.on_revoke = on_revoke
        self.on_lost = on_lost

    def poll(self, timeout):
        self.polls += 1
        while self._script and callable(self._script[0]):
            self._script.pop(0)(self)
        if self.partitions and self.partitions <= self.paused_partitions:
            return None
        message = self._script.pop(0) if self._script else None
        if message is not None and message.error() is None:
            self._positions[message.partition()] = message.offset() + 1
        return message

    def assignment(self):
        if self.polls <= self._unassigned_polls:
            return []
        return [TP(TOPIC, p) for p in sorted(self.partitions)]

    def assign(self, partitions):
        """Replace assignment when the native on_assign callback requests it."""
        self.partitions = {tp.partition for tp in partitions}

    def pause(self, partitions):
        """Retain queued records while servicing rebalance callbacks."""
        self.paused_partitions.update(tp.partition for tp in partitions)

    def resume(self, partitions):
        """Allow queued records from these partitions again."""
        self.paused_partitions.difference_update(tp.partition for tp in partitions)

    def commit(self, *, offsets, asynchronous):
        assert asynchronous is False
        if self._commit_raises:
            raise KafkaConsumerError("coordinator unavailable")
        self.commits.append({tp.partition: tp.offset for tp in offsets})
        if any(tp.partition in self._commit_errors for tp in offsets):
            # Match the native synchronous client: a failed partition can
            # raise, including when other partitions in a mixed call committed.
            raise KafkaConsumerError("partition commit failed")
        return [TP(tp.topic, tp.partition, tp.offset) for tp in offsets]

    def position(self, partitions):
        return [TP(tp.topic, tp.partition, self._positions.get(tp.partition, -1001)) for tp in partitions]

    def get_watermark_offsets(self, tp, *, timeout, cached):
        assert timeout > 0
        assert cached is False
        return 0, self._highs.get(tp.partition, 0)

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


def _binary(event, partition=0, offset=0) -> FakeMessage:
    """Carry envelope attributes as `ce_` headers and `data` as the value."""
    headers = [(f"ce_{key}", value.encode()) for key, value in event.items() if key not in {"data", "datacontenttype"}]
    return FakeMessage(json.dumps(event["data"]).encode(), offset, partition, headers=headers)


def _two_partitions(*between) -> list:
    """First fixture event on partition 0, then `between` steps, then the second on partition 1."""
    return [*_structured(_events()[:1], partition=0), *between, *_structured(_events()[1:], partition=1)]


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
    eca, _ = _make_eca([_binary(event)])

    contexts = eca.discover()

    assert [ctx["id"] for ctx in contexts] == ["job-b7"]
    assert contexts[0]["event_id"] == event["id"]


def test_discover_drops_exact_redelivery_before_attempt_supersession():
    first = _events()[0]
    resend = copy.deepcopy(first)
    resend["id"] = "new-event-id"
    eca, consumer = _make_eca(_structured([first, resend, first]))

    [context] = eca.discover()

    assert context["event_id"] == resend["id"]
    assert eca.last_discovery_stats["redeliveries"] == 1
    assert eca.last_discovery_stats["superseded"] == 1
    assert consumer.commits == [{0: 3}]


@pytest.mark.parametrize("encoding", ["structured", "binary"])
@pytest.mark.parametrize("same_source", [False, True])
def test_redelivery_identity_is_scoped_to_source(encoding, same_source):
    first, second = _events()
    first["source"] = "/scheduler/a"
    second = copy.deepcopy(first) if same_source else second
    second["source"] = first["source"] if same_source else "/scheduler/b"
    second["id"] = first["id"]
    events = [first, second]
    messages = _structured(events) if encoding == "structured" else [_binary(e, offset=i) for i, e in enumerate(events)]
    eca, consumer = _make_eca(messages)

    contexts = eca.discover()

    expected = events[:1] if same_source else events
    assert [ctx["job_id"] for ctx in contexts] == [e["data"]["job_id"] for e in expected]
    assert [ctx["event_source"] for ctx in contexts] == [e["source"] for e in expected]
    assert eca.last_discovery_stats["redeliveries"] == int(same_source)
    assert eca.last_discovery_stats["superseded"] == 0
    assert consumer.commits == [{0: 2}]


@pytest.mark.parametrize("encoding", ["structured", "binary"])
@pytest.mark.parametrize("source", [None, "", "/scheduler with spaces"])
def test_missing_or_empty_source_is_skipped(encoding, source):
    event = _events()[0]
    if source is None:
        event.pop("source")
    else:
        event["source"] = source
    messages = _structured([event]) if encoding == "structured" else [_binary(event)]
    eca, consumer = _make_eca(messages)

    assert eca.discover() == []
    assert eca.last_discovery_stats["malformed"] == 1
    assert consumer.commits == [{0: 1}]


@pytest.mark.parametrize("source", [42, False, {}, []])
def test_structured_source_must_be_a_string(source):
    event = _events()[0]
    event["source"] = source
    eca, _ = _make_eca(_structured([event]))

    assert eca.discover() == []
    assert eca.last_discovery_stats["malformed"] == 1


@pytest.mark.parametrize("field,value", [("id", "new-event-id"), ("source", "/scheduler/re-send")])
def test_discover_keeps_higher_offset_for_same_attempt_in_first_seen_order(field, value):
    first, other = _events()
    resend = copy.deepcopy(first)
    resend[field] = value
    resend["subject"] = "jobs/new-trigger-subject"
    eca, consumer = _make_eca(_structured([first, other, resend], first_offset=4))

    contexts = eca.discover()

    assert [ctx["event_id"] for ctx in contexts] == [resend["id"], other["id"]]
    assert contexts[0]["event_source"] == resend["source"]
    assert contexts[0]["event_subject"] == resend["subject"]
    assert eca.last_discovery_stats["redeliveries"] == 0
    assert eca.last_discovery_stats["superseded"] == 1
    assert consumer.commits == [{0: 7}]


def test_cross_partition_attempt_uses_commit_order_and_warns(monkeypatch):
    first = _events()[0]
    resend = copy.deepcopy(first)
    resend["id"] = "event-on-partition-one"
    warnings = []
    monkeypatch.setattr(
        "attribution_blueprint.connectors.execution_contexts.kafka.connector.logger.warning", warnings.append
    )
    # Poll partition 1 first; partition 1 still wins because it commits last.
    eca, consumer = _make_eca(
        [*_structured([resend], partition=1), *_structured([first], partition=0)], partitions=(0, 1)
    )

    [context] = eca.discover()

    assert context["event_id"] == resend["id"]
    assert eca.last_discovery_stats["superseded"] == 1
    assert eca.last_discovery_stats["redeliveries"] == 0
    assert consumer.commits == [{0: 1}, {1: 1}]
    assert len(warnings) == 1
    assert "spans partitions" in warnings[0]
    assert "commit order" in warnings[0]
    assert "message key" in warnings[0]


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


@pytest.mark.parametrize(
    "raw",
    [b"9" * 10_000, b"[" * 100_000 + b"0" + b"]" * 100_000],
    ids=["integer-digit-limit", "nesting-limit"],
)
def test_json_parser_limits_do_not_block_following_events(raw):
    """Skip parser-limit failures and commit past them with the next valid event."""
    eca, consumer = _make_eca([FakeMessage(raw, 0), *_structured(_events()[1:], first_offset=1)])

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert eca.last_discovery_stats["malformed"] == 1
    assert eca.last_discovery_stats["accepted"] == 1
    assert consumer.commits == [{0: 2}]
    assert consumer.closed


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


@pytest.mark.parametrize("read_first", [False, True])
@pytest.mark.parametrize("failure_kind", ["message", "native_poll"])
def test_consumer_error_closes_without_commit_and_leaves_retry_to_a_new_consumer(read_first, failure_kind):
    """Message errors and raised native poll errors leave the batch uncommitted."""
    error_type = KafkaConsumerError
    native_error = None
    failure = FakeMessage(None, 0, error=FakeError("broker transport failure"))
    if failure_kind == "native_poll":
        native = pytest.importorskip("confluent_kafka")
        error_type = native.KafkaException
        native_error = error_type(native.KafkaError(native.KafkaError._TRANSPORT))

        def fail_poll(_consumer):
            """Raise the exact native exception from inside the fake poll."""
            raise native_error

        failure = fail_poll
    script = [*_structured(_events()[:1]), failure] if read_first else [failure, *_structured(_events())]
    eca, consumer = _make_eca(script, consumer_error=error_type)

    with pytest.raises(error_type, match="transport failure") as caught:
        eca.discover()
    if native_error is not None:
        assert caught.value is native_error
    assert consumer.polls == (2 if read_first else 1)
    assert eca.discovery_incomplete is True
    assert consumer.closed
    assert consumer.commits == []


def test_discover_waits_for_partition_assignment():
    eca, _ = _make_eca([None, None, *_structured(_events())], unassigned_polls=2)

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]


def test_discover_polls_again_when_assignment_completes_on_empty_poll():
    eca, consumer = _make_eca([None, *_structured(_events())])

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]
    assert consumer.commits == [{0: 2}]


def test_discover_raises_when_never_assigned(monkeypatch, mocker):
    """Poll for assignment before timing out, closing without committing."""
    eca, consumer = _make_eca([], unassigned_polls=10**9, discovery_timeout_seconds=1)
    assignment = mocker.spy(consumer, "assignment")
    tick = iter(range(100))
    monkeypatch.setattr(
        "attribution_blueprint.connectors.execution_contexts.kafka.connector.time.monotonic", lambda: next(tick) / 10
    )
    with pytest.raises(KafkaConsumerError, match="assignment or unread records"):
        eca.discover()
    assert consumer.polls > 1
    assignment.assert_called()
    assert consumer.commits == []
    assert consumer.closed


def test_commits_explicit_offsets_per_partition():
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0, first_offset=7), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]
    assert consumer.commits == [{0: 8}, {1: 1}]


def test_revoked_partition_is_released_without_committing():
    eca, consumer = _make_eca(
        _two_partitions(revoke(0)),
        partitions=(0, 1),
    )
    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.commits == [{1: 1}]


def test_failure_after_revoke_commits_nothing_past_unreturned_records():
    # A revoke followed by a failing discover must not leave the revoked
    # partition's records committed but never returned.
    eca, consumer = _make_eca(
        [
            *_structured(_events(), partition=0, first_offset=4),
            revoke(0),
            FakeMessage(None, 0, partition=1, error=FakeError("topic authorization failed")),
        ],
        partitions=(0, 1),
    )

    with pytest.raises(KafkaConsumerError, match="topic authorization failed"):
        eca.discover()
    assert consumer.commits == []


def test_lost_partition_is_dropped_and_not_committed():
    eca, consumer = _make_eca(
        _two_partitions(lose(0)),
        partitions=(0, 1),
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.commits == [{1: 1}]


@pytest.mark.parametrize("release, commits", [(lose, [{0: 5}]), (revoke, [{0: 5}])])
def test_reassigned_partition_is_reread_from_last_commit(release, commits):
    # An eager rebalance revokes every partition and may hand one back; the
    # consumer then fetches it again from the last commit.
    def reassign(consumer):
        consumer.partitions.add(0)

    eca, consumer = _make_eca(
        [
            *_structured(_events()[:1], partition=0, first_offset=4),
            release(0),
            reassign,
            *_structured(_events()[:1], partition=0, first_offset=4),
        ],
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1"]
    assert consumer.commits == commits
    assert eca.last_discovery_stats["redeliveries"] == 0


def test_commit_exception_raises_and_revoke_cannot_commit_aborted_batch():
    eca, consumer = _make_eca(_structured(_events()), commit_raises=True)

    with pytest.raises(KafkaConsumerError, match="offset commit failed"):
        eca.discover()

    consumer._commit_raises = False
    # Invoke the callback the real Kafka consumer calls during close.
    consumer.on_revoke(consumer, [TP(TOPIC, 0)])
    eca.close()
    assert consumer.commits == []


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


def test_discover_stops_at_max_messages_and_closes():
    eca, consumer = _make_eca(_structured(_events()), max_messages=1)
    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1"]
    assert consumer.commits == [{0: 1}]
    assert consumer.closed
    with pytest.raises(KafkaConsumerError, match="closed"):
        eca.discover()


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


@pytest.mark.parametrize("field", ["max_messages", "poll_timeout_seconds", "discovery_timeout_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_constructor_rejects_invalid_limits(field, value):
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


def test_empty_polls_do_not_end_discovery_before_partition_end():
    eca, consumer = _make_eca([None, None, *_structured(_events())], poll_timeout_seconds=0.0001)

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]
    assert consumer.polls == 5


def test_unread_partition_times_out_instead_of_returning_empty(monkeypatch, mocker):
    """Check unread offsets after polling, then time out without committing."""
    eca, consumer = _make_eca([], discovery_timeout_seconds=0.01)
    consumer._highs[0] = 1
    watermark = mocker.spy(consumer, "get_watermark_offsets")
    tick = iter(range(1000))
    monkeypatch.setattr(
        "attribution_blueprint.connectors.execution_contexts.kafka.connector.time.monotonic",
        lambda: next(tick) / 1000,
    )

    with pytest.raises(KafkaConsumerError, match="unread records"):
        eca.discover()
    assert consumer.polls > 1
    watermark.assert_called()
    assert consumer.commits == []
    assert consumer.closed


def test_discovery_counters_surface_filtered_and_malformed_events():
    other = _events()[0]
    other["type"] = "com.example.job.completed.v2"
    eca, consumer = _make_eca([*_structured([other]), FakeMessage(b"not json", 1)])

    assert eca.discover() == []
    assert eca.last_discovery_stats["consumed"] == 2
    assert eca.last_discovery_stats["filtered"] == 1
    assert eca.last_discovery_stats["malformed"] == 1
    assert eca.last_discovery_stats["returned"] == 0
    assert eca.last_unmatched_event_types == {"com.example.job.completed.v2": 1}
    assert eca.discovery_incomplete is False
    assert consumer.commits == [{0: 2}]


def test_first_commit_failure_does_not_attempt_remaining_partitions():
    eca, consumer = _make_eca(_two_partitions(), partitions=(0, 1), commit_errors={0})
    with pytest.raises(KafkaConsumerError, match="offset commit failed"):
        eca.discover()
    assert consumer.commits == [{0: 1}]
    assert consumer.closed
    assert eca.last_discovery_stats["commit_failed_partitions"] == 1


def test_discovery_counters_remain_available_after_close():
    eca, consumer = _make_eca(_structured([_events()[0], _events()[0]]))
    assert len(eca.discover()) == 1
    assert consumer.closed
    assert eca.last_discovery_stats["redeliveries"] == 1
    assert eca.last_discovery_stats["superseded"] == 0
    assert eca.last_discovery_stats["returned"] == 1
    assert eca.discovery_incomplete is False


def test_partition_eof_can_confirm_committed_end_without_data_position():
    class EndOfPartition(FakeError):
        def name(self):
            return "_PARTITION_EOF"

    eca, consumer = _make_eca([FakeMessage(None, 3, error=EndOfPartition("end"))])
    consumer._highs[0] = 3

    assert eca.discover() == []
    assert consumer.commits == []
    assert eca.last_discovery_stats["consumed"] == 0


@pytest.mark.parametrize(
    "error", [IsADirectoryError(errno.EISDIR, "Is a directory"), OSError(errno.ENAMETOOLONG, "File name too long")]
)
def test_timezone_lookup_os_error_skips_event(monkeypatch, error):
    # With the tzdata package installed, ZoneInfo raises OSError subclasses
    # for directory names ("America") or overlong keys instead of
    # ZoneInfoNotFoundError. Such an event must be skipped, not block the topic.
    def zone_info(key):
        if key == "America":
            raise error
        return ZoneInfo(key)

    monkeypatch.setattr(events_module, "ZoneInfo", zone_info)
    bad = _events()[0]
    bad["data"]["timezone"] = "America"
    eca, consumer = _make_eca(_structured([bad, _events()[1]]))

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.commits == [{0: 2}]
    assert eca.last_discovery_stats["malformed"] == 1


@pytest.mark.parametrize("error", [OSError(errno.EIO, "I/O error"), PermissionError(errno.EACCES, "Permission denied")])
def test_timezone_read_failure_fails_discovery_without_commit(monkeypatch, error):
    # A host problem reading a valid zone file is not a malformed event:
    # fail the run uncommitted so the event is retried.
    def zone_info(key):
        if key == "America/Los_Angeles":
            raise error
        return ZoneInfo(key)

    monkeypatch.setattr(events_module, "ZoneInfo", zone_info)
    eca, consumer = _make_eca(_structured(_events()))

    with pytest.raises(OSError) as caught:
        eca.discover()
    assert caught.value is error
    assert consumer.commits == []


def test_revoke_after_successful_discover_does_not_move_commit_backwards():
    # close() triggers a revoke after the batch committed; re-committing the
    # batch's first offset there would make the next run reprocess it.
    eca, consumer = _make_eca(_structured(_events(), first_offset=4))

    assert len(eca.discover()) == 2
    consumer.on_revoke(consumer, [TP(TOPIC, 0)])
    assert consumer.commits == [{0: 6}]


@pytest.mark.parametrize("operation", ["poll", "commit", "position", "get_watermark_offsets"])
def test_unexpected_client_bug_is_reraised_instead_of_treated_as_kafka_failure(operation):
    eca, consumer = _make_eca(_structured(_events()))
    bug = AttributeError(f"unexpected {operation} bug")

    def broken(*args, **kwargs):
        raise bug

    setattr(consumer, operation, broken)
    with pytest.raises(AttributeError) as caught:
        eca.discover()
    assert caught.value is bug
    assert eca.discovery_incomplete is True
    assert consumer.commits == []
    assert consumer.closed


def test_unexpected_parser_bug_does_not_commit_or_become_malformed(monkeypatch):
    eca, consumer = _make_eca(_structured(_events()))

    def broken(*args, **kwargs):
        raise TypeError("unexpected parser bug")

    monkeypatch.setattr(
        "attribution_blueprint.connectors.execution_contexts.kafka.connector._context_from_data", broken
    )
    with pytest.raises(TypeError, match="unexpected parser bug"):
        eca.discover()
    assert consumer.commits == []
    assert eca.last_discovery_stats["malformed"] == 0


def test_close_failure_preserves_original_read_error():
    eca, consumer = _make_eca(
        [
            *_structured(_events()[:1]),
            FakeMessage(None, 1, error=FakeError("authorization failed")),
        ]
    )
    original_close = eca.close

    def close():
        consumer.on_revoke(consumer, [TP(TOPIC, 0)])
        original_close()
        raise KafkaConsumerError("injected close failure")

    eca.close = close
    with pytest.raises(KafkaConsumerError, match="authorization failed"):
        eca.discover()
    assert consumer.commits == []
    assert consumer.closed


def test_closed_adapter_rejects_discovery():
    eca, _ = _make_eca([])
    eca.close()
    with pytest.raises(KafkaConsumerError, match="closed"):
        eca.discover()


def test_native_commit_failure_keeps_successful_partition():
    native = pytest.importorskip("confluent_kafka")
    eca, consumer = _make_eca(_two_partitions(), partitions=(0, 1), consumer_error=native.KafkaException)
    commit = consumer.commit

    def fail_one_partition(*, offsets, asynchronous):
        if offsets[0].partition == 1:
            raise native.KafkaException(native.KafkaError(native.KafkaError._TIMED_OUT))
        return commit(offsets=offsets, asynchronous=asynchronous)

    consumer.commit = fail_one_partition
    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1"]
    assert eca.discovery_incomplete is True
    assert consumer.closed


def test_build_rejects_latest_before_opening_consumer():
    with pytest.raises(MCAError, match="supports only auto_offset_reset=earliest"):
        build_kafka(
            {"bootstrap_servers": "localhost:9092", "topic": TOPIC, "group_id": "test", "auto_offset_reset": "latest"},
            {},
        )


@pytest.mark.parametrize("encoding", ["structured", "binary"])
@pytest.mark.parametrize("cluster", [None, "example-cluster"])
def test_correlation_metadata_is_preserved(encoding, cluster):
    event = _events()[0]
    event["subject"] = "custom/subject/that-is-not-derived-from-job-id"
    if cluster is not None:
        event["data"]["cluster"] = cluster
    eca, consumer = _make_eca(_structured([event]) if encoding == "structured" else [_binary(event)])
    [context] = eca.discover()
    assert context["event_subject"] == event["subject"]
    assert context.get("cluster") == cluster
    assert consumer.closed


@pytest.mark.parametrize(
    "value",
    [[], "", False, {"worker": None}, {"worker": 2}, {"worker": ""}, {"worker": " "}, {"": "node"}, {" ": "node"}],
)
def test_invalid_pod_nodes_is_skipped_without_coercion(value):
    event = _events()[0]
    event["data"]["pod_nodes"] = value
    eca, consumer = _make_eca(_structured([event, _events()[1]]))
    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert eca.last_discovery_stats["malformed"] == 1
    assert consumer.commits == [{0: 2}]


@pytest.mark.parametrize("subject", ["absent", "", 7, None])
def test_missing_or_unusable_subject_is_treated_as_absent(subject):
    event = _events()[0]
    if subject == "absent":
        event.pop("subject", None)
    else:
        event["subject"] = subject
    eca, _ = _make_eca(_structured([event]))
    [context] = eca.discover()
    assert "event_subject" not in context
    assert eca.last_discovery_stats["malformed"] == 0


def test_valueless_binary_subject_header_is_treated_as_absent():
    message = _binary(_events()[0])
    message._headers = [(key, None if key == "ce_subject" else value) for key, value in message._headers]
    eca, _ = _make_eca([message])
    [context] = eca.discover()
    assert "event_subject" not in context


def test_filtered_only_batch_succeeds_and_closes():
    event = _events()[0]
    event["type"] = "com.example.job.started.v1"
    eca, consumer = _make_eca(_structured([event]))
    assert eca.discover() == []
    assert eca.discovery_incomplete is False
    assert eca.last_unmatched_event_types == {event["type"]: 1}
    assert consumer.commits == [{0: 1}]
    assert consumer.closed


def test_close_failure_preserves_committed_contexts():
    eca, consumer = _make_eca(_structured(_events()))
    original_close = eca.close

    def failed_close():
        original_close()
        raise KafkaConsumerError("injected close failure")

    eca.close = failed_close
    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]
    assert consumer.commits == [{0: 2}]
    assert consumer.closed
    assert eca.discovery_incomplete is True


def test_uncommitted_resend_cannot_supersede_a_committed_event():
    first = _events()[0]
    resend = copy.deepcopy(first)
    resend["id"] = "uncommitted-resend"
    eca, consumer = _make_eca(
        [*_structured([first], partition=0), *_structured([resend], partition=1)],
        partitions=(0, 1),
        commit_errors={1},
    )

    [context] = eca.discover()

    assert context["event_id"] == first["id"]
    assert eca.discovery_incomplete is True
    assert eca.last_discovery_stats["superseded"] == 0
    assert eca.last_discovery_stats["redeliveries"] == 0
    assert consumer.commits == [{0: 1}, {1: 1}]
