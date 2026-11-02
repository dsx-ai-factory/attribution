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
    connector as connector_module,
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
        message = self._script.pop(0) if self._script else None
        if message is not None and message.error() is None:
            self._positions[message.partition()] = message.offset() + 1
        return message

    def assignment(self):
        if self.polls <= self._unassigned_polls:
            return []
        return [TP(TOPIC, p) for p in sorted(self.partitions)]

    def commit(self, *, offsets, asynchronous):
        assert asynchronous is False
        if self._commit_raises:
            raise RuntimeError("coordinator unavailable")
        self.commits.append({tp.partition: tp.offset for tp in offsets})
        if any(tp.partition in self._commit_errors for tp in offsets):
            # Match the native synchronous client: a failed partition can
            # raise, including when other partitions in a mixed call committed.
            raise RuntimeError("partition commit failed")
        return [TP(tp.topic, tp.partition, tp.offset) for tp in offsets]

    def position(self, partitions):
        return [TP(tp.topic, tp.partition, self._positions.get(tp.partition, -1001)) for tp in partitions]

    def get_watermark_offsets(self, tp, *, timeout, cached):
        assert timeout > 0
        assert cached is False
        return 0, self._highs.get(tp.partition, 0)

    def seek(self, tp):
        self._positions[tp.partition] = tp.offset
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
    assert consumer.commits == [{0: 8}, {1: 1}]


def test_revoked_partition_is_released_with_a_bookmark_commit():
    # The revoked partition's contexts are not returned; its first unreturned
    # offset is committed so the next owner starts there, even for a new
    # group whose reset policy is `latest`.
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0), revoke(0), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.commits == [{0: 0}, {1: 1}]
    assert consumer.seeks == []  # no longer owned; the new owner rereads it


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
    assert consumer.commits == [{0: 4}]


def test_failed_bookmark_commit_marks_discovery_incomplete():
    # The revoked records are still reread by the next owner, but without the
    # bookmark a new `latest` group could skip them, so the run must not look
    # complete.
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0), revoke(0), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
        commit_errors={0},
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert eca.discovery_incomplete is True
    assert eca.last_discovery_stats["commit_failed_partitions"] == 1


def test_bookmark_partition_error_result_marks_discovery_incomplete():
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0), revoke(0), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
    )
    commit = consumer.commit

    def commit_with_partition_error(*, offsets, asynchronous):
        if offsets[0].partition == 0:
            return [TP(TOPIC, 0, offsets[0].offset, error="not coordinator")]
        return commit(offsets=offsets, asynchronous=asynchronous)

    consumer.commit = commit_with_partition_error

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert eca.discovery_incomplete is True


def test_lost_partition_is_dropped_and_not_committed():
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0), lose(0), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.commits == [{1: 1}]


@pytest.mark.parametrize("release, commits", [(lose, [{0: 5}]), (revoke, [{0: 4}, {0: 5}])])
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
    assert consumer.seeks == []
    assert eca.last_discovery_stats["duplicates"] == 0


def test_partial_commit_failure_drops_and_rewinds_that_partition():
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0, first_offset=3), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
        commit_errors={0},
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.seeks == [(0, 3)]


def test_commit_exception_raises_and_revoke_cannot_commit_aborted_batch():
    eca, consumer = _make_eca(_structured(_events()), commit_raises=True)

    with pytest.raises(KafkaConsumerError, match="offset commit failed"):
        eca.discover()
    assert consumer.seeks == [(0, 0)]

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
        ("fetch_timeout_seconds", 0),
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


def test_empty_polls_do_not_end_discovery_before_partition_end():
    eca, consumer = _make_eca([None, None, *_structured(_events())], poll_timeout_seconds=0.0001)

    assert [ctx["id"] for ctx in eca.discover()] == ["job-a1", "job-b7"]
    assert consumer.polls == 5


def test_unread_partition_times_out_instead_of_returning_empty(monkeypatch):
    eca, consumer = _make_eca([], fetch_timeout_seconds=0.01)
    consumer._highs[0] = 1
    tick = iter(range(1000))
    monkeypatch.setattr(
        "attribution_blueprint.connectors.execution_contexts.kafka.connector.time.monotonic",
        lambda: next(tick) / 100,
    )

    with pytest.raises(KafkaConsumerError, match="unread records"):
        eca.discover()
    assert consumer.commits == []


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
    assert eca.discovery_incomplete is True
    assert consumer.commits == [{0: 2}]


def test_partial_commit_failure_is_reported_but_keeps_successful_contexts():
    eca, consumer = _make_eca(
        [*_structured(_events()[:1], partition=0), *_structured(_events()[1:], partition=1)],
        partitions=(0, 1),
        commit_errors={0},
    )

    assert [ctx["id"] for ctx in eca.discover()] == ["job-b7"]
    assert consumer.commits == [{0: 1}, {1: 1}]
    assert eca.last_discovery_stats["commit_failed_partitions"] == 1
    assert eca.discovery_incomplete is True


def test_discovery_counters_reset_between_batches():
    eca, _ = _make_eca(_structured([_events()[0], _events()[0]]))
    assert len(eca.discover()) == 1
    assert eca.last_discovery_stats["duplicates"] == 1
    assert eca.discover() == []
    assert not any(eca.last_discovery_stats.values())
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

    monkeypatch.setattr(connector_module, "ZoneInfo", zone_info)
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

    monkeypatch.setattr(connector_module, "ZoneInfo", zone_info)
    eca, consumer = _make_eca(_structured(_events()))

    with pytest.raises(KafkaConsumerError, match="cannot read timezone"):
        eca.discover()
    assert consumer.commits == []


def test_revoke_after_successful_discover_does_not_move_commit_backwards():
    # close() triggers a revoke after the batch committed; re-committing the
    # batch's first offset there would make the next run reprocess it.
    eca, consumer = _make_eca(_structured(_events(), first_offset=4))

    assert len(eca.discover()) == 2
    consumer.on_revoke(consumer, [TP(TOPIC, 0)])
    assert consumer.commits == [{0: 6}]
