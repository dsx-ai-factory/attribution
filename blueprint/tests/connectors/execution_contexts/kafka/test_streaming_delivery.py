# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deferred receipts against disposable Kafka; no Attribution Service required."""

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

from attribution_blueprint.connectors.execution_contexts import InvalidReceiptError
from attribution_blueprint.connectors.execution_contexts.kafka import _build

from .test_delivery import _assert_drained

pytestmark = pytest.mark.integration


def _next_delivery(adapter, timeout=25):
    """Keep servicing Kafka until a delivery arrives or the bounded wait expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        delivery = adapter.poll(1, 0.5)
        if delivery:
            return delivery[0]
    pytest.fail("Kafka did not return the expected delivery")


def _committed(adapter, receipt):
    """Read the broker's group offset rather than the client's local position."""
    from confluent_kafka import TopicPartition

    return adapter._consumer.committed([TopicPartition(receipt.topic, receipt.partition)], timeout=10)[0].offset


def test_unacknowledged_delivery_replays_after_close(kafka_spec, completion_event, publish_event):
    """A new factory consumer replays work but rejects the old ownership token."""
    publish_event(completion_event)
    source = _build(kafka_spec, {})
    first = source.open_consumer()
    try:
        original = _next_delivery(first)
        assert _committed(first, original) < 0
    finally:
        first.close()

    restarted = source.open_consumer()
    try:
        replay = _next_delivery(restarted)
        assert replay.context == original.context
        with pytest.raises(InvalidReceiptError):
            restarted.ack(original)
        restarted.ack(replay)
        assert _committed(restarted, replay) == replay.offset + 1
    finally:
        restarted.close()
    _assert_drained(kafka_spec)


def test_out_of_order_acknowledgement_cannot_advance_the_broker_offset(kafka_spec, completion_event, publish_event):
    """Broker offsets stay unchanged until the earlier receipt is handled."""
    second = {**completion_event, "id": str(uuid4())}
    publish_event(completion_event)
    publish_event(second)
    adapter = _build(kafka_spec, {})
    try:
        first, last = _next_delivery(adapter), _next_delivery(adapter)
        assert first.context["event_id"] == completion_event["id"]
        assert last.context["event_id"] == second["id"]
        adapter.ack(last)
        assert _committed(adapter, last) < 0
        adapter.ack(first)
        assert _committed(adapter, last) == last.offset + 1
    finally:
        adapter.close()


def _acknowledge_peer(spec, accepted, stop):
    """Keep the peer alive after acceptance so it does not trigger another rebalance."""
    adapter = _build(spec, {})
    try:
        delivery = _next_delivery(adapter)
        adapter.ack(delivery)
        accepted.set()
        deadline = time.monotonic() + 30
        while not stop.is_set() and time.monotonic() < deadline:
            assert adapter.poll(1, 0.2) == []
        return delivery.context["event_id"]
    finally:
        adapter.close()


def test_rebalance_replays_pending_work_and_fences_previous_receipts(kafka_spec, completion_event, publish_event):
    """Real broker ownership changes cannot let a previous receipt commit work."""
    second = copy.deepcopy(completion_event)
    second["id"] = str(uuid4())
    second["data"]["id"] += "-second"
    second["data"]["job_id"] += "-second"
    publish_event(completion_event, partition=0)
    publish_event(second, partition=1)
    adapter = _build(kafka_spec, {})
    accepted, stop = Event(), Event()
    try:
        originals = [_next_delivery(adapter), _next_delivery(adapter)]
        assert {item.partition for item in originals} == {0, 1}
        with ThreadPoolExecutor(max_workers=1) as executor:
            peer = executor.submit(_acknowledge_peer, kafka_spec, accepted, stop)
            try:
                replay = _next_delivery(adapter)
                for original in originals:
                    with pytest.raises(InvalidReceiptError):
                        adapter.ack(original)
                adapter.ack(replay)
                assert accepted.wait(20), "the peer never acknowledged its delivery"
            finally:
                stop.set()
            observed = {replay.context["event_id"], peer.result(timeout=30)}
        assert observed == {completion_event["id"], second["id"]}
        assert adapter.last_discovery_stats["dropped_partitions"] > 0
    finally:
        stop.set()
        adapter.close()
    _assert_drained(kafka_spec)


def test_pause_before_assignment_and_after_delivery_preserves_records(kafka_spec, completion_event, publish_event):
    """Paused polls service a real group without consuming or committing work."""
    publish_event(completion_event)
    source = _build(kafka_spec, {})
    consumer = source.open_consumer()
    try:
        consumer.pause()
        deadline = time.monotonic() + 25
        while not consumer._consumer.assignment() and time.monotonic() < deadline:
            assert consumer.poll(5, 0.2) == []
        assert consumer._consumer.assignment(), "paused consumer never joined the group"
        for _ in range(3):
            assert consumer.poll(5, 0.2) == []
        consumer.resume()
        first = _next_delivery(consumer)
        assert first.event["type"] == completion_event["type"]
        assert first.event["data"] == completion_event["data"]
        second = {**completion_event, "id": str(uuid4())}
        consumer.pause()
        publish_event(second)
        for _ in range(3):
            assert consumer.poll(5, 0.2) == []
        assert _committed(consumer, first) < 0
        consumer.resume()
        last = _next_delivery(consumer)
        assert last.event["id"] == second["id"]
        consumer.ack(last)
        assert _committed(consumer, last) < 0
        consumer.ack(first)
        assert _committed(consumer, last) == last.offset + 1
    finally:
        consumer.close()
        source.close()
    _assert_drained(kafka_spec)
