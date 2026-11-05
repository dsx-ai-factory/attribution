# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit acknowledgement receipts exercised through the adapter's poll API."""

from dataclasses import replace

import pytest

from attribution_blueprint.connectors.execution_contexts import (
    AcknowledgingExecutionContextAdapter,
    AcknowledgingExecutionContextSource,
    InvalidReceiptError,
)
from attribution_blueprint.connectors.execution_contexts.kafka import (
    KafkaReceipt,
    connector as connector_module,
)
from attribution_blueprint.connectors.execution_contexts.kafka.connector import KafkaConsumerError

from .test_connector import (
    TOPIC,
    TP,
    FakeConsumer,
    FakeError,
    FakeMessage,
    _binary,
    _events,
    _make_eca,
    _structured,
    _two_partitions,
    lose,
    revoke,
)


def _one(adapter):
    """Require one delivered record without hiding batch-length regressions."""
    [receipt] = adapter.poll(1, 0)
    return receipt


@pytest.fixture
def stream():
    """Reuse the existing fake consumer and always close created adapters."""
    adapters = []

    def create(script, **kwargs):
        adapter, consumer = _make_eca(script, **kwargs)
        adapters.append(adapter)
        return adapter, consumer

    yield create
    for adapter in adapters:
        adapter.close()


def test_poll_exposes_context_and_receipt_without_committing(stream):
    adapter, consumer = stream(_structured(_events()[:1]))
    delivery = _one(adapter)

    assert isinstance(adapter, AcknowledgingExecutionContextAdapter)
    assert isinstance(delivery, KafkaReceipt)
    assert delivery.event == _events()[0]
    assert (delivery.topic, delivery.partition, delivery.offset) == (TOPIC, 0, 0)
    assert delivery.context["event_id"] == _events()[0]["id"]
    assert delivery.context["event_source"] == _events()[0]["source"]
    assert delivery.context["event_subject"] == _events()[0]["subject"]
    assert "receipt" not in delivery.context
    assert consumer.commits == []
    assert not consumer.closed


def test_out_of_order_acknowledgements_commit_only_the_accepted_prefix(stream):
    adapter, consumer = stream(_structured(_events()))
    first, second = _one(adapter), _one(adapter)

    adapter.ack(second)
    assert consumer.commits == []
    adapter.ack(first)
    assert consumer.commits == [{0: 2}]
    adapter.ack(first)
    adapter.ack(second)
    assert consumer.commits == [{0: 2}]
    assert not consumer.closed


def test_acknowledgements_are_independent_between_partitions(stream):
    adapter, consumer = stream(_two_partitions(), partitions=(0, 1))
    first, second = _one(adapter), _one(adapter)

    adapter.ack(second)
    assert consumer.commits == [{1: 1}]
    adapter.ack(first)
    assert consumer.commits == [{1: 1}, {0: 1}]


def test_offset_gaps_do_not_block_a_fully_acknowledged_prefix(stream):
    adapter, consumer = stream(
        [*_structured(_events()[:1], first_offset=3), *_structured(_events()[1:], first_offset=9)]
    )
    first, second = _one(adapter), _one(adapter)

    adapter.ack(second)
    assert consumer.commits == []
    adapter.ack(first)
    assert consumer.commits == [{0: 10}]


@pytest.mark.parametrize("kind", ["malformed", "filtered"])
def test_skips_do_not_commit_past_an_unacknowledged_delivery(stream, kind):
    """A skipped offset cannot let a later receipt bypass pending work."""
    event = {**_events()[0], "type": "unmatched.event.v2"}
    skipped = FakeMessage(b"not JSON", 1) if kind == "malformed" else _structured([event], first_offset=1)[0]
    adapter, consumer = stream([*_structured(_events()[:1]), skipped, *_structured(_events()[1:], first_offset=2)])
    first = _one(adapter)
    assert adapter.poll() == []
    last = _one(adapter)
    assert adapter.last_discovery_stats[kind] == 1
    adapter.ack(last)
    assert consumer.commits == []
    adapter.ack(first)
    assert consumer.commits == [{0: 3}]


def test_skip_without_pending_work_commits_and_polling_continues(stream):
    adapter, consumer = stream([FakeMessage(b"not JSON", 0), *_structured(_events()[:1], first_offset=1)])
    assert adapter.poll() == []
    assert consumer.commits == []
    delivery = _one(adapter)
    assert consumer.commits == []
    adapter.ack(delivery)
    assert consumer.commits == [{0: 2}]


def test_stream_mismatch_diagnostics_have_bounded_cardinality(stream, monkeypatch):
    """A long-running filtered stream cannot retain a new label per message."""
    monkeypatch.setattr(connector_module, "_MAX_UNMATCHED_EVENT_TYPES", 2)
    event = _events()[0]
    adapter, _ = stream(_structured([{**event, "type": f"unknown-{i}"} for i in range(5)]))
    for _ in range(5):
        assert adapter.poll() == []
    assert adapter.last_discovery_stats["filtered"] == 5
    assert adapter.last_unmatched_event_types == {"unknown-0": 1, "unknown-1": 1, "<other>": 3}


def test_streaming_delivers_duplicate_events_with_separate_receipts(stream):
    event = _events()[0]
    adapter, consumer = stream(_structured([event, event]))
    first, second = _one(adapter), _one(adapter)
    assert first.context == second.context
    assert first.offset != second.offset
    adapter.ack(first)
    assert consumer.commits == [{0: 1}]
    adapter.ack(second)
    assert consumer.commits == [{0: 1}, {0: 2}]


@pytest.mark.parametrize("release", [revoke, lose])
def test_reassignment_invalidates_old_receipts_even_for_the_same_offset(stream, release):
    """An old owner cannot acknowledge a replayed offset in a new assignment."""
    adapter, consumer = stream(
        [
            *_structured(_events()[:1]),
            release(0),
            lambda client: client.partitions.add(0),
            *_structured(_events()[:1]),
        ]
    )
    first, replay = _one(adapter), _one(adapter)
    assert first.assignment_id != replay.assignment_id
    with pytest.raises(InvalidReceiptError, match="stale or foreign"):
        adapter.ack(first)
    assert consumer.commits == []
    adapter.ack(replay)
    assert consumer.commits == [{0: 1}]
    assert adapter.last_discovery_stats["dropped_partitions"] == 1


def test_lost_partition_does_not_invalidate_another_partitions_receipts(stream):
    adapter, consumer = stream([*_two_partitions(), lose(0)], partitions=(0, 1))
    first, second = _one(adapter), _one(adapter)
    assert adapter.poll() == []
    with pytest.raises(InvalidReceiptError):
        adapter.ack(first)
    adapter.ack(second)
    assert consumer.commits == [{1: 1}]


def test_foreign_and_unknown_receipts_are_rejected_without_closing_intake(stream):
    """Invalid receipt metadata must not poison the current owner."""
    adapter, consumer = stream(_structured(_events()[:1]))
    other, _ = stream(_structured(_events()[:1]))
    delivery, foreign = _one(adapter), _one(other)
    for invalid in (
        foreign,
        object(),
        replace(delivery, offset=100),
        replace(delivery, offset=True),
        replace(delivery, offset=0.0),
        replace(delivery, partition=0.0),
        replace(delivery, topic=[]),
    ):
        with pytest.raises(InvalidReceiptError):
            adapter.ack(invalid)
    assert consumer.commits == []
    assert not consumer.closed
    adapter.ack(delivery)
    assert consumer.commits == [{0: 1}]


def test_close_does_not_commit_pending_receipts_and_rejects_later_calls(stream):
    adapter, consumer = stream(_structured(_events()[:1]))
    delivery = _one(adapter)
    adapter.close()
    adapter.close()
    assert consumer.closed
    assert consumer.commits == []
    with pytest.raises(InvalidReceiptError, match="closed"):
        adapter.ack(delivery)
    with pytest.raises(KafkaConsumerError, match="closed"):
        adapter.poll()


def test_polling_idle_or_at_eof_keeps_the_consumer_open(stream):
    eof = FakeMessage(None, 0, error=FakeError("_PARTITION_EOF"))
    adapter, consumer = stream([None, eof, *_structured(_events()[:1])])
    assert adapter.poll(timeout_seconds=0) == []
    assert adapter.poll(timeout_seconds=0) == []
    assert len(adapter.poll(timeout_seconds=0)) == 1
    assert not consumer.closed
    assert consumer.commits == []


def test_discovery_cannot_switch_to_automatic_commit_after_polling(stream):
    adapter, consumer = stream(_structured(_events()[:1]))
    delivery = _one(adapter)
    with pytest.raises(KafkaConsumerError, match="cannot mix"):
        adapter.discover()
    assert consumer.commits == []
    adapter.ack(delivery)
    assert consumer.commits == [{0: 1}]


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan"), 1e308, 2_147_483.648, True, "1"])
def test_invalid_poll_timeout_does_not_touch_the_native_client(stream, timeout):
    adapter, consumer = stream([])
    with pytest.raises(ValueError, match="finite and in"):
        adapter.poll(timeout_seconds=timeout)
    assert consumer.polls == 0


def test_commit_failure_closes_intake_and_preserves_the_error(stream):
    adapter, consumer = stream(_structured(_events()), commit_errors=(0,))
    first, second = _one(adapter), _one(adapter)
    adapter.ack(second)
    with pytest.raises(KafkaConsumerError, match="acknowledgement commit failed") as error:
        adapter.ack(first)
    assert error.value.__cause__ is not None
    assert consumer.closed
    assert adapter.discovery_incomplete
    assert adapter.last_discovery_stats["commit_failed_partitions"] == 1
    with pytest.raises(InvalidReceiptError):
        adapter.ack(second)


@pytest.mark.parametrize("after_record", [False, True])
def test_poll_error_propagates_and_closes_without_committing(stream, after_record):
    """A native failure preserves its cause and leaves all deliveries replayable."""
    error = KafkaConsumerError("original polling failure")

    def fail(consumer):
        raise error

    script = _structured(_events()[:1]) if after_record else []
    adapter, consumer = stream([*script, fail])
    if after_record:
        adapter.poll()
    with pytest.raises(KafkaConsumerError) as caught:
        adapter.poll()
    assert caught.value is error
    assert consumer.closed
    assert consumer.commits == []


@pytest.mark.parametrize("error_type", [TypeError, KeyboardInterrupt])
def test_decoder_failure_or_cancellation_closes_intake_without_hiding_the_error(stream, monkeypatch, error_type):
    adapter, consumer = stream(_structured(_events()[:1]))

    def fail(message):
        raise error_type("decoder interrupted")

    monkeypatch.setattr(adapter, "_decode_record", fail)
    with pytest.raises(error_type, match="decoder interrupted"):
        adapter.poll()
    assert consumer.closed
    assert consumer.commits == []


def test_poll_error_record_closes_without_accepting_or_committing_it(stream):
    message = FakeMessage(None, 0, error=FakeError("_ALL_BROKERS_DOWN"))
    adapter, consumer = stream([message])
    with pytest.raises(KafkaConsumerError, match="_ALL_BROKERS_DOWN"):
        adapter.poll()
    assert consumer.closed
    assert consumer.commits == []
    assert adapter.last_discovery_stats["consumed"] == 0


@pytest.mark.parametrize("maximum", [0, -1, True, 1.0, "1"])
def test_invalid_batch_size_does_not_poll(stream, maximum):
    adapter, consumer = stream([])
    with pytest.raises(ValueError, match="max_records"):
        adapter.poll(maximum, 0)
    assert consumer.polls == 0


def test_batch_poll_bounds_native_reads_and_waits_only_once(stream, monkeypatch):
    """One wait is followed by bounded nonblocking reads, with no poll commits."""
    adapter, consumer = stream(_structured(_events()))
    timeouts = []
    native_poll = consumer.poll

    def poll(timeout):
        timeouts.append(timeout)
        return native_poll(timeout)

    monkeypatch.setattr(consumer, "poll", poll)
    batch = adapter.poll(2, 0.25)
    assert [receipt.event["id"] for receipt in batch] == [event["id"] for event in _events()]
    assert timeouts == [0.25, 0]
    assert consumer.commits == []
    adapter.ack(batch[1])
    assert consumer.commits == []
    adapter.ack(batch[0])
    assert consumer.commits == [{0: 2}]


def test_binary_delivery_keeps_the_cloudevent_envelope(stream):
    event = _events()[0]
    message = _binary(event)
    message._headers.append(("content-type", event["datacontenttype"].encode()))
    adapter, _ = stream([message])
    receipt = _one(adapter)
    assert receipt.event == event
    assert receipt.event["type"] == event["type"]
    assert receipt.context["start_time"] != receipt.event["data"]["start_time"]


def test_filtered_only_poll_does_not_commit_or_accumulate_receipts(stream):
    event = {**_events()[0], "type": "unknown.v2"}
    adapter, consumer = stream(_structured([event] * 20))
    assert adapter.poll(20, 0) == []
    assert consumer.polls == 20
    assert consumer.commits == []
    assert adapter._acknowledgements._partitions == {}


def test_pause_services_polling_and_keeps_pending_receipts_until_resume(stream):
    """Pausing preserves both pending receipts and the unconsumed queue."""
    adapter, consumer = stream(_structured(_events()))
    first = _one(adapter)
    adapter.pause()
    polls = consumer.polls
    for _ in range(3):
        assert adapter.poll(10, 0) == []
    assert consumer.polls == polls + 3
    assert not consumer.closed
    assert consumer.commits == []
    adapter.ack(first)
    assert consumer.commits == [{0: 1}]
    adapter.resume()
    second = _one(adapter)
    assert second.offset == 1
    adapter.ack(second)
    assert consumer.commits == [{0: 1}, {0: 2}]


def test_assignment_while_paused_stays_paused_and_old_receipts_are_invalid(stream):
    def reassign(consumer):
        consumer.on_assign(consumer, [TP(TOPIC, 0)])

    adapter, consumer = stream([*_structured(_events()[:1]), revoke(0), reassign, *_structured(_events()[:1])])
    first = _one(adapter)
    adapter.pause()
    assert adapter.poll(5, 0) == []
    with pytest.raises(InvalidReceiptError):
        adapter.ack(first)
    adapter.resume()
    replay = _one(adapter)
    assert replay.offset == first.offset
    adapter.ack(replay)
    assert consumer.commits == [{0: 1}]


@pytest.mark.parametrize("operation", ["pause", "resume"])
def test_native_flow_control_failure_closes_and_propagates(stream, monkeypatch, operation):
    adapter, consumer = stream([])
    adapter.poll(1, 0)
    error = KafkaConsumerError("flow control failed")

    def fail(partitions):
        raise error

    monkeypatch.setattr(consumer, operation, fail)
    with pytest.raises(KafkaConsumerError) as caught:
        getattr(adapter, operation)()
    assert caught.value is error
    assert consumer.closed
    assert consumer.commits == []


def test_registered_source_opens_fresh_consumers_without_an_idle_group_member(monkeypatch):
    """Factory reconnects must not add an unused source consumer to the group."""
    from attribution_blueprint.connectors.execution_contexts.kafka import _build

    native = pytest.importorskip("confluent_kafka")
    opened = []

    def create(config):
        assert config["group.id"] == "test-group"
        assert config["enable.auto.commit"] is False
        consumer = FakeConsumer(_structured(_events()[:1]))
        opened.append(consumer)
        return consumer

    monkeypatch.setattr(native, "Consumer", create)
    source = _build({"bootstrap_servers": "localhost:9092", "topic": TOPIC, "group_id": "test-group"}, {})
    assert isinstance(source, AcknowledgingExecutionContextSource)
    assert opened == []
    first = source.open_consumer()
    original = _one(first)
    first.close()
    second = source.open_consumer()
    try:
        replay = _one(second)
        assert replay.context == original.context
        with pytest.raises(InvalidReceiptError):
            second.ack(original)
        second.ack(replay)
        assert opened[1].commits == [{0: 1}]
    finally:
        second.close()
        source.close()
    assert len(opened) == 2
    assert all(consumer.closed for consumer in opened)


def test_acknowledgements_remain_valid_across_idle_polls_and_shutdown_is_noncommitting(stream):
    """Time spent polling does not expire receipts or acknowledge unfinished work."""
    adapter, consumer = stream(_structured(_events()))
    first, second = _one(adapter), _one(adapter)
    for _ in range(100):
        assert adapter.poll(5, 0) == []
    adapter.ack(second)
    assert consumer.commits == []
    adapter.close()
    assert consumer.commits == []
    with pytest.raises(InvalidReceiptError):
        adapter.ack(first)


@pytest.mark.parametrize("limit", [None, 0, 120])
def test_optional_time_limit_is_service_compatible_without_changing_the_event(stream, limit):
    """Zero is not a Service deadline; keep the original CloudEvent unchanged."""
    event = _events()[0]
    event["data"]["time_limit"] = limit
    adapter, consumer = stream(_structured([event]))
    receipt = _one(adapter)
    assert receipt.event["data"]["time_limit"] == limit
    if limit:
        assert receipt.context["time_limit"] == limit
    else:
        assert "time_limit" not in receipt.context
    assert consumer.commits == []
