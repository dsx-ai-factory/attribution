# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kafka input with bounded discovery and explicit processing acknowledgements.

Kafka owns connection retries and group assignment. This adapter owns payload
validation and the discovery boundary. On failure it closes the consumer;
callers create a new adapter to resume from the group's committed offsets.
Bounded discovery is best effort: attribution happens after its commits.
Long-lived callers use poll/ack and decide when processing is complete.
"""

from __future__ import annotations

import math
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from attribution_blueprint.connectors.execution_contexts import InvalidReceiptError
from attribution_blueprint.connectors.execution_contexts.kafka.events import (
    EventError,
    KafkaExecutionContext,
    _context_from_data,
    _read_cloudevent,
)
from attribution_blueprint.connectors.execution_contexts.kafka.receipts import (
    KafkaReceipt,
    PartitionKey,
    _Acknowledgements,
)

if TYPE_CHECKING:
    from confluent_kafka import Consumer, Message

CommittedContext = tuple[PartitionKey, KafkaExecutionContext]
_MAX_POLL_TIMEOUT_SECONDS = 2_147_483.647  # Native timeouts use signed 32-bit milliseconds.
_MAX_UNMATCHED_EVENT_TYPES = 100


def _validate_poll_timeout(timeout: float) -> None:
    """Reject invalid or overflowing timeouts before calling the native client."""
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 <= timeout <= _MAX_POLL_TIMEOUT_SECONDS
    ):
        raise ValueError("kafka ECA: poll timeout must be finite and in 0..2147483.647 seconds")


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
    Alternatively, poll returns deliveries without closing or committing valid
    work. Acknowledge receipts when handled, using one owner thread.
    """

    def __init__(
        self,
        *,
        consumer: Consumer | None = None,
        consumer_factory: Callable[[], Consumer] | None = None,
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
        """Configure a native consumer or lazy factory; do not poll yet.

        The factory supplies TopicPartition and KafkaException alongside the
        consumer. Only that specific client exception is recoverable here.
        """
        if consumer is None and consumer_factory is None:
            raise ValueError("kafka ECA: a consumer or consumer factory is required")
        if not topic:
            raise ValueError("kafka ECA: topic is required")
        for name, value in (
            ("max_messages", max_messages),
            ("poll_timeout_seconds", poll_timeout_seconds),
            ("discovery_timeout_seconds", discovery_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"kafka ECA: {name} must be positive and finite, got {value}")
        self._consumer_instance = None
        self._consumer_factory = consumer_factory
        self._topic, self._topic_partition = topic, topic_partition
        self._consumer_error = consumer_error
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
        self._mode: str | None = None
        self._paused = False
        self._acknowledgements = _Acknowledgements()
        self._close_consumer = None
        if consumer is not None:
            self._subscribe(consumer)

    def _subscribe(self, consumer: Consumer) -> None:
        """Attach ownership callbacks and guarantee native resource cleanup."""
        self._consumer_instance = consumer
        self._close_consumer = weakref.finalize(self, consumer.close)
        try:
            consumer.subscribe(
                [self._topic],
                on_assign=_weak(self._on_assign),
                on_revoke=_weak(self._release),
                on_lost=_weak(self._release),
            )
        except BaseException:
            self._close_after_discovery()
            raise

    @property
    def _consumer(self) -> Consumer:
        """Open the configured native consumer only when this instance is used."""
        if self._consumer_instance is None:
            self._subscribe(self._consumer_factory())
        return self._consumer_instance

    def open_consumer(self) -> KafkaExecutionContextAdapter:
        """Open an independent long-lived consumer for the configured topic/group."""
        if self._consumer_factory is None:
            raise KafkaConsumerError("kafka ECA: open_consumer requires a configured consumer factory")
        adapter = KafkaExecutionContextAdapter(
            consumer_factory=self._consumer_factory,
            consumer_error=self._consumer_error,
            topic=self._topic,
            topic_partition=self._topic_partition,
            event_types=list(self._event_types),
            scheduler=self._scheduler,
            workload_type=self._workload_type,
            max_messages=self._max_messages,
            poll_timeout_seconds=self._poll_timeout_seconds,
            discovery_timeout_seconds=self._discovery_timeout_seconds,
        )
        adapter._require_mode("streaming")
        adapter._consumer
        return adapter

    def discover(self) -> list[KafkaExecutionContext]:
        """Read and commit one batch, then close before attribution can begin.

        Invalid/filtered/duplicate events are intentional skips and commit too.
        A commit failure returns only already committed contexts and flags the
        call incomplete; with no successful contexts it raises instead.
        """
        self._require_mode("discovery")
        self._reset_stats()
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

    def poll(self, max_records: int = 1, timeout_seconds: float | None = None) -> list[KafkaReceipt]:
        """Return up to max_records deliveries without committing any offsets.

        Wait for the first native record, then drain immediately available
        records. Intentional skips count toward the bound. Keep polling while
        paused so assignment callbacks and group membership remain serviced.
        """
        if type(max_records) is not int or max_records <= 0:
            raise ValueError("kafka ECA: max_records must be a positive integer")
        timeout = self._poll_timeout_seconds if timeout_seconds is None else timeout_seconds
        _validate_poll_timeout(timeout)
        self._require_mode("streaming")
        try:
            return self._poll_records(max_records, timeout)
        except BaseException:
            # Cleanup covers cancellation too; propagate the original failure.
            self.discovery_incomplete = True
            self._close_after_discovery()
            raise

    def _poll_records(self, max_records: int, timeout: float) -> list[KafkaReceipt]:
        """Drain one bounded poll, including native ownership callbacks."""
        deliveries = []
        for index in range(max_records):
            message = self._consumer.poll(timeout if index == 0 else 0)
            if message is None:
                break
            error = message.error()
            if error is not None:
                if error.name() == "_PARTITION_EOF":
                    continue
                raise KafkaConsumerError(f"kafka ECA: consumer error: {error}")
            if self._paused:
                raise KafkaConsumerError("kafka ECA: native consumer delivered a record while paused")
            delivery = self._delivery(message)
            if delivery is not None:
                deliveries.append(delivery)
        return deliveries

    def ack(self, delivery: object) -> None:
        """Commit only the acknowledged prefix, even across many polls.

        The caller decides when work is handled. On commit failure, recreate
        the consumer and recover replayed work through application state.
        """
        if self._closed:
            raise InvalidReceiptError("kafka ECA: receipt belongs to a closed adapter")
        key = self._acknowledgements.acknowledge(delivery)
        try:
            self._commit_acknowledged(key)
        except BaseException:
            self.discovery_incomplete = True
            self._close_after_discovery()
            raise

    def pause(self) -> None:
        """Pause owned partitions; future assignments inherit the paused state."""
        self._set_paused(True)

    def resume(self) -> None:
        """Resume owned partitions without leaving the consumer group."""
        self._set_paused(False)

    def _set_paused(self, paused: bool) -> None:
        """Apply flow control on the owner thread and propagate native errors."""
        self._require_mode("streaming")
        try:
            partitions = self._consumer.assignment()
            if partitions:
                operation = self._consumer.pause if paused else self._consumer.resume
                operation(partitions)
            self._paused = paused
        except BaseException:
            self._close_after_discovery()
            raise

    def _on_assign(self, consumer: Consumer, partitions: list[Any]) -> None:
        """Apply pauses before records from a new assignment can be delivered."""
        consumer.assign(partitions)
        if self._paused:
            consumer.pause(partitions)

    def _require_mode(self, mode: str) -> None:
        """Keep one-shot and streaming lifecycles exclusive on one consumer."""
        if self._closed:
            raise KafkaConsumerError("kafka ECA: consumer is closed; create a new adapter")
        if self._mode is not None and self._mode != mode:
            raise KafkaConsumerError("kafka ECA: cannot mix discover() and poll() on one adapter")
        if self._mode is None:
            self._mode = mode
            if mode == "streaming":
                self._reset_stats()

    def _reset_stats(self) -> None:
        """Initialize per-discovery counters, or cumulative streaming counters."""
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

    def _delivery(self, message: Message) -> KafkaReceipt | None:
        """Issue receipts only for valid records; skips commit with later acknowledgements."""
        decoded = self._decode_record(message)
        if decoded is None:
            return None
        receipt = self._acknowledgements.register(message.topic(), message.partition(), message.offset())
        context, event = decoded
        self.last_discovery_stats["returned"] += 1
        return KafkaReceipt(receipt.topic, receipt.partition, receipt.offset, receipt.assignment_id, context, event)

    def _commit_acknowledged(self, key: PartitionKey) -> None:
        """Forget receipts only after their safe prefix was broker-confirmed."""
        offset = self._acknowledgements.ready_offset(key)
        if offset is None:
            return
        try:
            self._commit_partition(key, offset)
        except self._consumer_errors as exc:
            self.last_discovery_stats["commit_failed_partitions"] += 1
            raise KafkaConsumerError(f"kafka ECA: acknowledgement commit failed: {exc}") from exc
        self._acknowledgements.committed(key, offset)

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
        """Record a bounded-discovery offset and its successfully decoded context."""
        key = (message.topic(), message.partition())
        batch = self._pending.setdefault(key, _PartitionBatch(message.offset() + 1))
        batch.offset = message.offset() + 1
        decoded = self._decode_record(message)
        if decoded is not None:
            batch.contexts.append(decoded[0])

    def _decode_record(self, message: Message) -> tuple[KafkaExecutionContext, dict] | None:
        """Share payload validation and skip accounting across both lifecycles."""
        self.last_discovery_stats["consumed"] += 1
        try:
            event_id, event_type, data, subject, source, event = _read_cloudevent(message.value(), message.headers())
            if self._event_types and event_type not in self._event_types:
                self.last_discovery_stats["filtered"] += 1
                self._record_filtered_type(event_type)
                return
            context = _context_from_data(
                data,
                event_id=event_id,
                event_source=source,
                default_scheduler=self._scheduler,
                workload_type=self._workload_type,
                event_subject=subject,
            )
        except EventError as exc:
            self.last_discovery_stats["malformed"] += 1
            logger.warning(f"kafka ECA: skipping {(message.topic(), message.partition())}@{message.offset()}: {exc}")
            return
        self.last_discovery_stats["accepted"] += 1
        return context, event

    def _record_filtered_type(self, event_type: str) -> None:
        """Surface stream mismatches without retaining unlimited distinct types."""
        counts = self.last_unmatched_event_types
        if self._mode == "streaming" and event_type not in counts and len(counts) >= _MAX_UNMATCHED_EVENT_TYPES:
            event_type = "<other>"
        counts[event_type] = counts.get(event_type, 0) + 1
        if self._mode == "streaming" and counts[event_type] == 1:
            logger.warning(f"kafka ECA: filtered event type {event_type!r}; check event_types")

    def _commit_partition(self, key: PartitionKey, offset: int) -> None:
        """Commit one partition and require an explicit successful native result."""
        results = self._consumer.commit(offsets=[self._topic_partition(*key, offset)], asynchronous=False)
        if not results or any(tp.error is not None for tp in results):
            raise KafkaConsumerError("commit returned no successful result")

    def _commit_batch(self) -> list[CommittedContext]:
        """Commit one partition per call so earlier successes remain returnable.

        Mixed-partition commits may partially succeed and then raise. Stop at
        the first failure, close the consumer, and leave remaining work to a
        new adapter. A failed commit's broker outcome may still be uncertain.
        """
        contexts: list[CommittedContext] = []
        for key, batch in sorted(self._pending.items()):
            try:
                self._commit_partition(key, batch.offset)
            except self._consumer_errors as exc:
                return self._commit_failed(contexts, exc)
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
            dropped = self._pending.pop(key, None) is not None
            if self._acknowledgements.release(key) or dropped:
                self.last_discovery_stats["dropped_partitions"] += 1
            self._end_offsets.pop(key, None)

    def _dedupe(self, contexts: list[CommittedContext]) -> list[KafkaExecutionContext]:
        """Drop repeated (source, id) pairs, then keep the latest event per attempt.

        Committed contexts arrive in partition order, then increasing offsets.
        Replacing a dict value preserves the attempt's first-seen output order.
        """
        seen_events: set[tuple[str, str]] = set()
        unique: dict[tuple[str, str, str], CommittedContext] = {}
        for partition, context in contexts:
            event_key = (context["event_source"], context["event_id"])
            if event_key in seen_events:
                self.last_discovery_stats["redeliveries"] += 1
                continue
            seen_events.add(event_key)
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
        self._acknowledgements.clear()
        if self._close_consumer is not None:
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
