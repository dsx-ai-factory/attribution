# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kafka input: read a bounded batch and commit it before attribution.

Kafka owns connection retries and group assignment. This adapter owns payload
validation and the discovery boundary. On failure it closes the consumer;
callers create a new adapter to resume from the group's committed offsets.
Delivery remains best effort because attribution happens after these commits.
"""

from __future__ import annotations

import math
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from attribution_blueprint.connectors.execution_contexts.kafka.events import (
    EventError,
    KafkaExecutionContext,
    _context_from_data,
    _read_cloudevent,
)

if TYPE_CHECKING:
    from confluent_kafka import Consumer, Message

PartitionKey = tuple[str, int]
CommittedContext = tuple[PartitionKey, KafkaExecutionContext]


class KafkaConsumerError(RuntimeError):
    """Discovery failed or a partition's commit outcome is uncertain."""


@dataclass
class _PartitionBatch:
    """Handled records for one currently owned partition; offset is next to read."""

    offset: int
    contexts: list[KafkaExecutionContext] = field(default_factory=list)


class KafkaExecutionContextAdapter:
    """Consume CloudEvents with automatic commits disabled and earliest reset.

    Revocation/loss drops that partition's uncommitted batch; Kafka replays it
    to its next owner. Each discover() call closes the consumer before returning,
    including on failure. Construct a new adapter for each batch.
    """

    def __init__(
        self,
        *,
        consumer: Consumer,
        topic: str,
        topic_partition: Callable[[str, int, int], Any],
        consumer_error: type[Exception] = KafkaConsumerError,
        event_types: list[str] | None = None,
        scheduler: str = "kafka",
        workload_type: str = "",
        max_messages: int = 500,
        poll_timeout_seconds: float = 5.0,
        discovery_timeout_seconds: float = 60.0,
    ) -> None:
        """Subscribe a native consumer (or test double); do not poll yet.

        The factory supplies TopicPartition and KafkaException alongside the
        consumer. Only that specific client exception is recoverable here.
        """
        if not topic:
            raise ValueError("kafka ECA: topic is required")
        for name, value in (
            ("max_messages", max_messages),
            ("poll_timeout_seconds", poll_timeout_seconds),
            ("discovery_timeout_seconds", discovery_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"kafka ECA: {name} must be positive and finite, got {value}")
        self._consumer, self._topic_partition = consumer, topic_partition
        self._consumer_errors = (consumer_error, KafkaConsumerError)
        self._event_types = frozenset(event_types or ())
        self._scheduler, self._workload_type = scheduler, workload_type
        self._max_messages = max_messages
        self._poll_timeout_seconds = poll_timeout_seconds
        self._discovery_timeout_seconds = discovery_timeout_seconds
        self._pending: dict[PartitionKey, _PartitionBatch] = {}
        self._end_offsets: dict[PartitionKey, int] = {}
        self.last_discovery_stats: dict[str, int] = {}
        self.last_unmatched_event_types: dict[str, int] = {}
        self.discovery_incomplete = False
        self._closed = False
        self._close_consumer = weakref.finalize(self, consumer.close)
        consumer.subscribe([topic], on_revoke=_weak(self._release), on_lost=_weak(self._release))

    def discover(self) -> list[KafkaExecutionContext]:
        """Read and commit one batch, then close before attribution can begin.

        Invalid/filtered/duplicate events are intentional skips and commit too.
        A commit failure returns only already committed contexts and flags the
        call incomplete; with no successful contexts it raises instead.
        """
        if self._closed:
            raise KafkaConsumerError("kafka ECA: consumer is closed; create a new adapter")
        self.last_discovery_stats = dict.fromkeys(
            (
                "consumed",
                "accepted",
                "filtered",
                "malformed",
                "redeliveries",
                "superseded",
                "returned",
                "commit_failed_partitions",
                "dropped_partitions",
            ),
            0,
        )
        self.last_unmatched_event_types = {}
        self.discovery_incomplete = False
        try:
            self._read_batch()
            contexts = self._dedupe(self._commit_batch())
            self.last_discovery_stats["returned"] = len(contexts)
            return contexts
        except BaseException:
            self.discovery_incomplete = True
            raise
        finally:
            self._pending.clear()
            self._end_offsets.clear()
            self._close_after_discovery()
            logger.info(f"kafka ECA: discovery summary {self.last_discovery_stats}")
            if self.last_unmatched_event_types:
                logger.warning(f"kafka ECA: filtered event types {self.last_unmatched_event_types}")

    def _read_batch(self) -> None:
        """Read until max_messages or broker end offsets, within one time budget."""
        deadline = time.monotonic() + self._discovery_timeout_seconds
        while self.last_discovery_stats["consumed"] < self._max_messages:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise KafkaConsumerError("kafka ECA: discovery timed out waiting for assignment or unread records")
            message = self._consumer.poll(min(self._poll_timeout_seconds, remaining))
            if message is None:
                if self._is_drained(deadline):
                    return
                continue
            error = message.error()
            if error is None:
                self._accept_record(message)
            elif error.name() == "_PARTITION_EOF":
                self._end_offsets[(message.topic(), message.partition())] = message.offset()
            else:
                raise KafkaConsumerError(f"kafka ECA: consumer error: {error}")

    def _is_drained(self, deadline: float) -> bool:
        """An empty poll is complete only when every assigned partition is at EOF."""
        assigned = self._consumer.assignment()
        if not assigned:
            return False
        positions = self._consumer.position(assigned)
        for position in positions:
            if position.error is not None:
                raise KafkaConsumerError(f"kafka ECA: position lookup failed: {position.error}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            low, high = self._consumer.get_watermark_offsets(position, timeout=remaining, cached=False)
            key = (position.topic, position.partition)
            if low < high and max(position.offset, self._end_offsets.get(key, -1)) < high:
                return False
        return len(positions) == len(assigned)

    def _accept_record(self, message: Message) -> None:
        """Advance this partition's pending offset, catching only invalid payloads."""
        self.last_discovery_stats["consumed"] += 1
        key = (message.topic(), message.partition())
        batch = self._pending.setdefault(key, _PartitionBatch(message.offset() + 1))
        batch.offset = message.offset() + 1
        try:
            event_id, event_type, data, subject = _read_cloudevent(message.value(), message.headers())
            if self._event_types and event_type not in self._event_types:
                self.last_discovery_stats["filtered"] += 1
                self.last_unmatched_event_types[event_type] = self.last_unmatched_event_types.get(event_type, 0) + 1
                return
            context = _context_from_data(
                data,
                event_id=event_id,
                default_scheduler=self._scheduler,
                workload_type=self._workload_type,
                event_subject=subject,
            )
        except EventError as exc:
            self.last_discovery_stats["malformed"] += 1
            logger.warning(f"kafka ECA: skipping {key}@{message.offset()}: {exc}")
            return
        batch.contexts.append(context)
        self.last_discovery_stats["accepted"] += 1

    def _commit_batch(self) -> list[CommittedContext]:
        """Commit one partition per call so earlier successes remain returnable.

        Mixed-partition commits may partially succeed and then raise. Stop at
        the first failure, close the consumer, and leave remaining work to a
        new adapter. A failed commit's broker outcome may still be uncertain.
        """
        contexts: list[CommittedContext] = []
        for key, batch in sorted(self._pending.items()):
            try:
                results = self._consumer.commit(
                    offsets=[self._topic_partition(*key, batch.offset)],
                    asynchronous=False,
                )
            except self._consumer_errors as exc:
                return self._commit_failed(contexts, exc)
            if not results or any(tp.error is not None for tp in results):
                return self._commit_failed(contexts, KafkaConsumerError("commit returned no successful result"))
            contexts.extend((key, context) for context in batch.contexts)
        return contexts

    def _commit_failed(self, contexts: list[CommittedContext], error: Exception) -> list[CommittedContext]:
        """Report partial commits without discarding contexts already acknowledged."""
        self.discovery_incomplete = True
        self.last_discovery_stats["commit_failed_partitions"] = 1
        if not contexts:
            raise KafkaConsumerError(f"kafka ECA: offset commit failed: {error}") from error
        logger.warning(f"kafka ECA: offset commit failed: {error}; returning the committed subset")
        return contexts

    def _close_after_discovery(self) -> None:
        """Report close failures without hiding read errors or committed contexts."""
        try:
            self.close()
        except self._consumer_errors as exc:
            self.discovery_incomplete = True
            logger.warning(f"kafka ECA: consumer close failed: {exc}")

    def _release(self, consumer: Consumer, partitions: list[Any]) -> None:
        """Drop revoked/lost records without committing; their next owner replays."""
        for partition in partitions:
            key = (partition.topic, partition.partition)
            if self._pending.pop(key, None) is not None:
                self.last_discovery_stats["dropped_partitions"] += 1
            self._end_offsets.pop(key, None)

    def _dedupe(self, contexts: list[CommittedContext]) -> list[KafkaExecutionContext]:
        """Drop exact redeliveries, then keep the latest event for each attempt.

        Committed contexts arrive in partition order, then increasing offsets.
        Replacing a dict value preserves the attempt's first-seen output order.
        """
        seen_ids: set[str] = set()
        unique: dict[tuple[str, str, str], CommittedContext] = {}
        for partition, context in contexts:
            if context["event_id"] in seen_ids:
                self.last_discovery_stats["redeliveries"] += 1
                continue
            seen_ids.add(context["event_id"])
            key = (context["scheduler"], context["job_id"], context["attempt_id"])
            previous = unique.get(key)
            if previous is not None:
                self.last_discovery_stats["superseded"] += 1
                if previous[0] != partition:
                    logger.warning(
                        f"kafka ECA: attempt {key} spans partitions {previous[0]} and {partition}; "
                        "keeping the last event in commit order; check the producer's message key"
                    )
            unique[key] = (partition, context)
        return [context for _, context in unique.values()]

    def discover_by_ids(self, ids: list[str]) -> list[KafkaExecutionContext]:
        """Kafka is a completion stream, not a scheduler job index."""
        raise NotImplementedError("kafka ECA cannot look up jobs by id; consume the topic without ids")

    def discover_by_time_range(
        self, start: datetime, end: datetime, nodelist: list[str] | None = None
    ) -> list[KafkaExecutionContext]:
        """Kafka completion events do not support scheduler time-range queries."""
        raise NotImplementedError("kafka ECA does not scan time ranges; consume the topic without start_time")

    def close(self) -> None:
        """Leave the group once; a closed adapter cannot be reused."""
        self._closed = True
        self._close_consumer()


def _weak(method: Callable[..., None]) -> Callable[..., None]:
    """Avoid a callback reference cycle so finalization can close the consumer."""
    ref = weakref.WeakMethod(method)

    def call(*args: Any) -> None:
        """Forward the callback only while its adapter is alive."""
        target = ref()
        if target is not None:
            target(*args)

    return call
