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


def test_live_rebalance_drops_revoked_partition_without_duplicate_or_missing_events(
    kafka_spec,
    completion_event,
    publish_event,
):
    from confluent_kafka import Consumer, TopicPartition

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

    first_read = Event()
    rebalanced = Event()
    release_peer = Event()

    class ObservedConsumer:
        def __init__(self):
            self.consumer = Consumer(_consumer_config(kafka_spec))
            self.delivered = []
            self.commits = []
            self.revocations = []
            self.rebalance_deadline = None

        def __getattr__(self, name):
            return getattr(self.consumer, name)

        def subscribe(self, topics, *, on_revoke, on_lost):
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
                    rebalanced.set()

            self.consumer.subscribe(topics, on_revoke=revoke, on_lost=on_lost)

        def commit(self, *, offsets, asynchronous):
            result = self.consumer.commit(offsets=offsets, asynchronous=asynchronous)
            self.commits.append({tp.partition: tp.offset for tp in offsets})
            return result

        def poll(self, timeout):
            # Keep servicing the actual broker's rebalance callbacks while
            # the peer joins. Never fabricate a message or callback.
            while True:
                waiting = first_read.is_set() and not rebalanced.is_set()
                message = self.consumer.poll(min(timeout, 0.1) if waiting else timeout)
                if message is not None:
                    if message.error() is None:
                        self.delivered.append((message.partition(), message.offset()))
                        if not first_read.is_set():
                            self.rebalance_deadline = time.monotonic() + 30
                            first_read.set()
                    return message
                if not waiting or rebalanced.is_set():
                    return None
                if time.monotonic() >= self.rebalance_deadline:
                    raise TimeoutError("the second consumer did not trigger a live rebalance")

    observed = ObservedConsumer()
    adapter = KafkaExecutionContextAdapter(
        consumer=observed,
        topic=kafka_spec["topic"],
        topic_partition=TopicPartition,
        event_types=kafka_spec["event_types"],
        poll_timeout_seconds=3,
    )

    def peer_run():
        assert first_read.wait(30), "the first consumer never read a record"
        peer = _build({**kafka_spec, "poll_timeout_seconds": 3}, {})
        try:
            contexts = peer.discover()
            assignment = {tp.partition for tp in peer._consumer.assignment()}
            assert release_peer.wait(30), "the first consumer did not complete its batch"
            return contexts, assignment
        finally:
            peer.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(peer_run)
            try:
                first = adapter.discover()
                first_assignment = {tp.partition for tp in observed.assignment()}
            finally:
                release_peer.set()
            second, second_assignment = future.result(timeout=35)

        assert rebalanced.is_set()
        # Revocation must not commit: records read from a revoked partition
        # are dropped and reread by the partition's next owner.
        relevant = [item for item in observed.revocations if item[0] and item[1]]
        assert relevant
        assert all(commits == [] for _, _, commits in relevant)
        assert {partition_of[ctx["event_id"]] for ctx in first} <= first_assignment
        assert {partition_of[ctx["event_id"]] for ctx in second} <= second_assignment
        assert len(first_assignment) == len(second_assignment) == 1
        assert first_assignment.isdisjoint(second_assignment)
        actual = [ctx["event_id"] for ctx in first + second]
        assert set(actual) == expected
        assert len(actual) == len(expected)  # No duplicates across consumers.
        committed = observed.committed([TopicPartition(kafka_spec["topic"], p) for p in range(2)], timeout=10)
        assert {tp.partition: tp.offset for tp in committed} == {0: 2, 1: 2}
    finally:
        adapter.close()

    restarted = _build(kafka_spec, {})
    try:
        assert restarted.discover() == []
    finally:
        restarted.close()
