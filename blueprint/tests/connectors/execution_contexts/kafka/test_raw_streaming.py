# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw payload custody and acknowledgement boundaries against disposable Kafka."""

import time

import pytest

from attribution_blueprint.connectors.execution_contexts import InvalidReceiptError
from attribution_blueprint.connectors.execution_contexts.kafka import _build

from .test_streaming_delivery import _committed

pytestmark = pytest.mark.integration


def _next_raw(consumer):
    """Poll within a bounded wait while Kafka assigns the isolated test group."""
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        receipts = consumer.poll_raw(1, 0.5)
        if receipts:
            [receipt] = receipts
            return receipt
    pytest.fail("Kafka did not return the expected raw record")


@pytest.fixture
def publish_raw(kafka_broker_topic):
    """Produce synthetic bytes, null values and duplicate headers without envelopes."""
    from confluent_kafka import Producer

    brokers, topic = kafka_broker_topic
    producer = Producer({"bootstrap.servers": brokers})

    def publish(value, *, headers=None):
        errors = []
        producer.produce(
            topic,
            key=b"record-key",
            value=value,
            headers=headers,
            partition=0,
            on_delivery=lambda error, message: errors.append(error) if error is not None else None,
        )
        assert producer.flush(15) == 0
        assert errors == []

    return publish


def test_raw_bytes_and_tombstones_hold_the_broker_commit_prefix(kafka_spec, publish_raw):
    """A later tombstone cannot commit an earlier undecodable payload."""
    publish_raw(b"\xffnot-json", headers=[("trace", b"first"), ("trace", None)])
    publish_raw(None)
    source = _build(kafka_spec, {})
    consumer = source.open_raw_consumer()
    try:
        first, last = _next_raw(consumer), _next_raw(consumer)
        assert first.value == b"\xffnot-json"
        assert first.key == b"record-key"
        assert first.headers == [("trace", b"first"), ("trace", None)]
        assert last.value is None
        consumer.ack(last)
        assert _committed(consumer, first) < 0
        consumer.ack(first)
        assert _committed(consumer, first) == last.offset + 1
    finally:
        consumer.close()
        source.close()


def test_raw_unacknowledged_record_replays_with_new_ownership(kafka_spec, publish_raw):
    """Restart preserves raw bytes and invalidates the previous receipt."""
    publish_raw(b'{"native_id":42}')
    source = _build(kafka_spec, {})
    first = source.open_raw_consumer()
    try:
        original = _next_raw(first)
        assert _committed(first, original) < 0
    finally:
        first.close()
    restarted = source.open_raw_consumer()
    try:
        replay = _next_raw(restarted)
        assert replay.value == original.value == b'{"native_id":42}'
        assert replay.key == original.key
        with pytest.raises(InvalidReceiptError):
            restarted.ack(original)
        restarted.ack(replay)
        assert _committed(restarted, replay) == replay.offset + 1
    finally:
        restarted.close()
        source.close()
