# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A real group rebalance while the first adapter holds uncommitted records."""

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

from attribution_blueprint.connectors.execution_contexts.kafka import _build, _consumer_config
from attribution_blueprint.connectors.execution_contexts.kafka.connector import KafkaExecutionContextAdapter

pytestmark = pytest.mark.integration


class ObservedConsumer:
    """Observe native records/commits while holding the first batch for a peer."""

    def __init__(self, spec, first_read, rebalanced):
        from confluent_kafka import Consumer

        self.first_read, self.rebalanced = first_read, rebalanced
        self.consumer = Consumer(_consumer_config(spec))
        self.delivered = []
        self.commits = []
        self.revocations = []
        self.rebalance_deadline = None

    def __getattr__(self, name):
        return getattr(self.consumer, name)

    def subscribe(self, topics, *, on_revoke, on_lost, on_assign):
        def revoke(consumer, partitions):
            committed_before = len(self.commits)
            on_revoke(consumer, partitions)
            self.revocations.append(
                (
                    {tp.partition for tp in partitions},
                    list(self.delivered),
                    self.commits[committed_before:],
                )
            )
            if self.delivered:
                self.rebalanced.set()

        self.consumer.subscribe(topics, on_assign=on_assign, on_revoke=revoke, on_lost=on_lost)

    def commit(self, *, offsets, asynchronous):
        result = self.consumer.commit(offsets=offsets, asynchronous=asynchronous)
        self.commits.append({tp.partition: tp.offset for tp in offsets})
        return result

    def poll(self, timeout):
        # Keep servicing the actual broker's rebalance callbacks while
        # the peer joins. Never fabricate a message or callback.
        while True:
            waiting = self.first_read.is_set() and not self.rebalanced.is_set()
            message = self.consumer.poll(min(timeout, 0.1) if waiting else timeout)
            if message is not None:
                if message.error() is None:
                    self.delivered.append((message.partition(), message.offset()))
                    if not self.first_read.is_set():
                        self.rebalance_deadline = time.monotonic() + 30
                        self.first_read.set()
                return message
            if not waiting or self.rebalanced.is_set():
                return None
            if time.monotonic() >= self.rebalance_deadline:
                raise TimeoutError("the second consumer did not trigger a live rebalance")


def _publish_attempts(completion_event, publish_event):
    """Publish two unique attempts per partition and retain their identities."""
    expected = set()
    partition_of = {}
    for partition in range(2):
        for index in range(2):
            event = copy.deepcopy(completion_event)
            suffix = f"-{partition}-{index}"
            event["id"] = str(uuid4())
            event["data"]["id"] += suffix
            event["data"]["job_id"] += suffix
            event["subject"] = f"jobs/{event['data']['id']}"
            expected.add(event["id"])
            partition_of[event["id"]] = partition
            publish_event(event, partition=partition)

    return expected, partition_of


def _discover_with_assignment(adapter):
    """Snapshot ownership before one-shot discovery closes the native consumer."""
    assignment = set()
    original_close = adapter.close

    def close():
        if not adapter._closed:
            assignment.update(tp.partition for tp in adapter._consumer.assignment())
        original_close()

    adapter.close = close
    return adapter.discover(), assignment


def _peer_run(spec, first_read):
    """Join after the first record and return a separately committed batch."""
    assert first_read.wait(30), "the first consumer never read a record"
    peer = _build({**spec, "poll_timeout_seconds": 3}, {})
    try:
        return _discover_with_assignment(peer)
    finally:
        peer.close()


def _assert_rebalance(
    observed, first, first_assignment, second, second_assignment, expected, partition_of, spec, rebalanced
):
    """Prove no revoke commit, no overlap/loss, and complete broker offsets."""
    from confluent_kafka import Consumer, TopicPartition

    assert rebalanced.is_set()
    relevant = [item for item in observed.revocations if item[0] and item[1]]
    assert relevant
    assert all(commits == [] for _, _, commits in relevant)
    assert {partition_of[ctx["event_id"]] for ctx in first} <= first_assignment
    assert {partition_of[ctx["event_id"]] for ctx in second} <= second_assignment
    # Consumers leave on completion; ownership may overlap at different times.
    actual = [ctx["event_id"] for ctx in first + second]
    assert set(actual) == expected
    assert len(actual) == len(expected)  # No duplicates across consumers.
    reader = Consumer(_consumer_config(spec))
    try:
        committed = reader.committed([TopicPartition(spec["topic"], p) for p in range(2)], timeout=10)
        assert {tp.partition: tp.offset for tp in committed} == {0: 2, 1: 2}
    finally:
        reader.close()


def test_live_rebalance_drops_revoked_partition_without_duplicate_or_missing_events(
    kafka_spec,
    completion_event,
    publish_event,
):
    """Rebalance a real group while its first consumer holds uncommitted records."""
    from confluent_kafka import KafkaException, TopicPartition

    expected, partition_of = _publish_attempts(completion_event, publish_event)
    first_read, rebalanced = Event(), Event()
    observed = ObservedConsumer(kafka_spec, first_read, rebalanced)
    adapter = KafkaExecutionContextAdapter(
        consumer=observed,
        consumer_error=KafkaException,
        topic=kafka_spec["topic"],
        topic_partition=TopicPartition,
        event_types=kafka_spec["event_types"],
        poll_timeout_seconds=3,
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_peer_run, kafka_spec, first_read)
            first, first_assignment = _discover_with_assignment(adapter)
            second, second_assignment = future.result(timeout=35)

        _assert_rebalance(
            observed,
            first,
            first_assignment,
            second,
            second_assignment,
            expected,
            partition_of,
            kafka_spec,
            rebalanced,
        )
    finally:
        adapter.close()

    restarted = _build(kafka_spec, {})
    try:
        assert restarted.discover() == []
    finally:
        restarted.close()
