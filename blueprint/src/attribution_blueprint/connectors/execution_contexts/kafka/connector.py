# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kafka execution-context input, with commits at the discovery boundary.

Like other execution-context adapters, discover() returns normalized contexts.
CloudEvent decoding lives in events.py; batch.py names per-call state.
The adapter coordinates one batch and the rebalance callbacks invoked by poll.

Delivery is best effort: successful partitions commit before contexts reach
attribution. Downstream failure therefore does not replay those events. A
future acknowledgment interface belongs in a separate change.
"""

from __future__ import annotations

import time
import weakref
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from attribution_blueprint.connectors.execution_contexts.kafka.batch import (
    DiscoveryBatch,
    PartitionKey,
    PollWait,
)
from attribution_blueprint.connectors.execution_contexts.kafka.events import (
    EventError,
    KafkaExecutionContext,
    _context_from_data,
    _read_cloudevent,
)

if TYPE_CHECKING:
    from confluent_kafka import Consumer, Message


class KafkaConsumerError(RuntimeError):
    """Discovery failed or only a subset of the batch can be returned."""


class KafkaExecutionContextAdapter:
    """Read bounded batches from a consumer with automatic commits disabled.

    One discovery call owns one DiscoveryBatch. poll() may synchronously revoke
    or lose partitions: those callbacks remove their unreturned records. A
    revoke bookmarks the first unfinished offset; a loss cannot commit. With
    latest reset and no prior group offset, a lost partition can skip records.

    At the end, each partition commits independently. Failed partitions are
    rewound and omitted; successful contexts are returned even if another
    partition fails. An uncertain broker commit cannot guarantee redelivery.
    Failed rewinds close the consumer so a later discover cannot skip records.

    last_discovery_stats and last_unmatched_event_types describe the last call.
    discovery_incomplete flags failures or filtered-only batches for the CLI.
    This adapter is sequential: callers must not overlap discover/close calls.
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
        assignment_timeout_seconds: float = 60.0,
        fetch_timeout_seconds: float = 60.0,
        max_transient_errors: int = 3,
    ) -> None:
        """Validate batch limits and subscribe the supplied consumer once.

        The factory supplies the native Consumer, TopicPartition constructor,
        and KafkaException type. Tests can inject a consumer with the same
        methods and a specific operational exception. Other errors propagate.
        """
        if not topic:
            raise ValueError("kafka ECA: topic is required")
        for name, value in (
            ("max_messages", max_messages),
            ("poll_timeout_seconds", poll_timeout_seconds),
            ("fetch_timeout_seconds", fetch_timeout_seconds),
        ):
            if value <= 0:
                raise ValueError(f"kafka ECA: {name} must be positive, got {value}")
        for name, value in (
            ("assignment_timeout_seconds", assignment_timeout_seconds),
            ("max_transient_errors", max_transient_errors),
        ):
            if value < 0:
                raise ValueError(f"kafka ECA: {name} must be non-negative, got {value}")
        self._consumer = consumer
        self._consumer_errors = (KafkaConsumerError, consumer_error)
        self._topic_partition = topic_partition
        self._event_types = frozenset(event_types or ())
        self._scheduler = scheduler
        self._workload_type = workload_type
        self._max_messages = max_messages
        self._poll_timeout_seconds = poll_timeout_seconds
        self._assignment_timeout_seconds = assignment_timeout_seconds
        self._max_transient_errors = max_transient_errors
        self._fetch_timeout_seconds = fetch_timeout_seconds
        self.last_discovery_stats: dict[str, int] = {}
        self.last_unmatched_event_types: dict[str, int] = {}
        self.discovery_incomplete = False
        self._batch: DiscoveryBatch | None = None
        self._closed = False
        self._close_consumer = weakref.finalize(self, consumer.close)
        # Weak callbacks let the finalizer release the group when the caller
        # drops the adapter. Callbacks only mutate an active discovery batch.
        consumer.subscribe([topic], on_revoke=_weak(self._on_revoke), on_lost=_weak(self._on_lost))

    def discover(self) -> list[KafkaExecutionContext]:
        """Read and commit one batch before returning its contexts.

        Invalid, filtered, and duplicate events are intentionally acknowledged.
        Failure or cancellation rewinds uncommitted records and re-raises;
        successful partition commits cannot be rolled back.
        """
        if self._closed:
            raise KafkaConsumerError("kafka ECA: consumer is closed; create a new adapter")
        batch = DiscoveryBatch()
        self._batch = batch
        self._reset_diagnostics()
        try:
            self._poll_batch(batch)
            contexts = self._finish_batch(batch)
            self.last_discovery_stats["returned"] = len(contexts)
            return contexts
        except BaseException:
            self.discovery_incomplete = True
            # Detach before cleanup: close() may invoke on_revoke.
            self._batch = None
            self._rewind_uncommitted(batch)
            raise
        finally:
            self._batch = None
            self._report_batch(batch)

    def _reset_diagnostics(self) -> None:
        """Start each call with independent counters, including after failure."""
        self.last_discovery_stats = dict.fromkeys(
            (
                "consumed",
                "accepted",
                "filtered",
                "malformed",
                "duplicates",
                "returned",
                "commit_failed_partitions",
                "dropped_partitions",
            ),
            0,
        )
        self.last_unmatched_event_types = {}
        self.discovery_incomplete = False

    def _report_batch(self, batch: DiscoveryBatch) -> None:
        """Expose partial failures without discarding successfully committed work."""
        failures = batch.commit_failures | batch.bookmark_failures
        self.last_discovery_stats["commit_failed_partitions"] = len(failures)
        self.last_discovery_stats["dropped_partitions"] = len(batch.commit_failures) + batch.released_count
        self.discovery_incomplete |= bool(failures) or (
            self.last_discovery_stats["filtered"] > 0 and self.last_discovery_stats["accepted"] == 0
        )
        logger.info(f"kafka ECA: discovery summary {self.last_discovery_stats}")
        if self.last_unmatched_event_types:
            logger.warning(
                f"kafka ECA: filtered event types {dict(sorted(self.last_unmatched_event_types.items()))}; "
                f"configured event_types={sorted(self._event_types)}"
            )

    def _poll_batch(self, batch: DiscoveryBatch) -> None:
        """Poll up to max_messages, or until every owned partition is drained."""
        wait = PollWait(time.monotonic() + self._assignment_timeout_seconds)
        while self.last_discovery_stats["consumed"] < self._max_messages:
            message = self._consumer.poll(self._poll_timeout_seconds)
            if message is None:
                if self._wait_for_records(batch, wait):
                    break
                continue
            wait.assignment_deadline = None
            wait.fetch_deadline = None
            error = message.error()
            if error is not None:
                self._handle_poll_error(batch, wait, message, error)
            else:
                self._accept_record(batch, message)

    def _wait_for_records(self, batch: DiscoveryBatch, wait: PollWait) -> bool:
        """Return True only at broker end offsets; raise on either idle deadline.

        Waiting for assignment and waiting for unread data have separate
        budgets. Short empty polls during group join/fetch are not completion.
        """
        assigned = self._consumer.assignment()
        now = time.monotonic()
        if not assigned:
            wait.fetch_deadline = None
            if wait.assignment_deadline is None:
                wait.assignment_deadline = now + self._assignment_timeout_seconds
            if now >= wait.assignment_deadline:
                raise KafkaConsumerError(
                    f"kafka ECA: no partitions assigned within {self._assignment_timeout_seconds}s"
                )
            return False
        wait.assignment_deadline = None
        if wait.fetch_deadline is None:
            wait.fetch_deadline = now + self._fetch_timeout_seconds
        if self._is_drained(batch, assigned, wait.fetch_deadline):
            return True
        if time.monotonic() >= wait.fetch_deadline:
            raise KafkaConsumerError(
                f"kafka ECA: assigned partitions still have unread records after "
                f"{self._fetch_timeout_seconds}s without progress"
            )
        return False

    def _handle_poll_error(self, batch: DiscoveryBatch, wait: PollWait, message: Message, error: Any) -> None:
        """Record EOF evidence, retry bounded transient errors, or fail discovery."""
        if error.name() == "_PARTITION_EOF":
            batch.end_offsets[(message.topic(), message.partition())] = message.offset()
            return
        wait.transient_errors += 1
        if error.fatal() or not error.retriable() or wait.transient_errors > self._max_transient_errors:
            raise KafkaConsumerError(f"kafka ECA: consumer error: {error}")
        logger.warning(f"kafka ECA: retriable consumer error ({wait.transient_errors}): {error}")

    def _accept_record(self, batch: DiscoveryBatch, message: Message) -> None:
        """Track one offset and parse it; catch only invalid-event errors."""
        self.last_discovery_stats["consumed"] += 1
        key = (message.topic(), message.partition())
        batch.observe(key, message.offset())
        try:
            ctx = self._parse_message(message)
        except EventError as exc:
            self.last_discovery_stats["malformed"] += 1
            logger.warning(f"kafka ECA: skipping event at {key[0]}[{key[1]}]@{message.offset()}: {exc}")
            return
        if ctx is not None:
            self.last_discovery_stats["accepted"] += 1
            batch.contexts.append((key, ctx))

    def _is_drained(self, batch: DiscoveryBatch, assigned: list[Any], deadline: float) -> bool:
        """Compare positions/EOF with fresh high watermarks within the idle budget.

        An unresolved negative position cannot prove a nonempty partition is
        drained. Client failures propagate, rather than looking like emptiness.
        """
        try:
            positions = self._consumer.position(assigned)
        except self._consumer_errors as exc:
            raise KafkaConsumerError(f"kafka ECA: position lookup failed: {exc}") from exc
        positions = {(tp.topic, tp.partition): tp for tp in positions}
        for tp in assigned:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            key = (tp.topic, tp.partition)
            position = positions.get(key)
            if position is not None and position.error is not None:
                raise KafkaConsumerError(f"kafka ECA: position lookup failed: {position.error}")
            try:
                low, high = self._consumer.get_watermark_offsets(tp, timeout=remaining, cached=False)
            except self._consumer_errors as exc:
                raise KafkaConsumerError(f"kafka ECA: cannot check partition end offsets: {exc}") from exc
            next_offset = max(position.offset if position is not None else -1, batch.end_offsets.get(key, -1))
            if low < high and next_offset < high:
                return False
        return True

    def _finish_batch(self, batch: DiscoveryBatch) -> list[KafkaExecutionContext]:
        """Commit each owned partition, rewind failures, and return its successes.

        Native mixed-partition commits can raise after committing a subset, so
        each call names exactly one partition. An operational failure affects
        only that partition; a programming error is allowed to propagate.
        """
        for key, progress in batch.partitions.items():
            try:
                self._commit_offset(key, progress.next_offset)
            except self._consumer_errors as exc:
                batch.commit_failures.add(key)
                logger.warning(f"kafka ECA: commit failed for {key}: {exc}; its contexts will not be returned")
            else:
                batch.committed.add(key)
        contexts = self._dedupe(batch)
        self._rewind_uncommitted(batch)
        if batch.commit_failures and not contexts:
            raise KafkaConsumerError(f"kafka ECA: offset commit failed for {sorted(batch.commit_failures)}")
        return contexts

    def _commit_offset(self, key: PartitionKey, offset: int) -> None:
        """Commit one explicit next offset and require a successful result."""
        results = self._consumer.commit(
            offsets=[self._topic_partition(*key, offset)],
            asynchronous=False,
        )
        if not results or any(tp.error is not None for tp in results):
            raise KafkaConsumerError("kafka ECA: commit returned no successful partition result")

    def _rewind_uncommitted(self, batch: DiscoveryBatch) -> None:
        """Rewind uncommitted records still owned; close after an unsafe rewind.

        Successful partitions are untouched, including after another partition
        fails. Recovery catches only client errors. A failed rewind is surfaced
        as incomplete discovery and makes future calls fail explicitly.
        """
        pending = batch.partitions.keys() - batch.committed
        if self._closed or not pending:
            return
        try:
            owned = {(tp.topic, tp.partition) for tp in self._consumer.assignment()}
        except self._consumer_errors as exc:
            self._close_after_recovery_failure(exc)
            return
        for key in sorted(pending & owned):
            progress = batch.partitions[key]
            try:
                self._consumer.seek(self._topic_partition(*key, progress.first_offset))
            except self._consumer_errors as exc:
                self._close_after_recovery_failure(exc)
                return
            del batch.partitions[key]

    def _close_after_recovery_failure(self, error: Exception) -> None:
        """Invalidate the consumer while preserving any successful batch subset."""
        self.discovery_incomplete = True
        self._batch = None
        logger.warning(f"kafka ECA: cannot rewind aborted batch: {error}")
        try:
            self.close()
        except self._consumer_errors as exc:
            logger.warning(f"kafka ECA: close after aborted batch failed: {exc}")

    def _on_revoke(self, consumer: Any, partitions: list[Any]) -> None:
        """Bookmark first unreturned records, then release the revoked partitions.

        This never advances past an unreturned record. It protects fresh latest
        groups after a record was read, but cannot bookmark an unresolved start
        before the first fetch. No active batch means nothing may be committed.
        """
        batch = self._batch
        if batch is None:
            return
        keys = {(tp.topic, tp.partition) for tp in partitions}
        pending = batch.partitions.keys() - batch.committed
        for key in sorted(keys & pending):
            try:
                self._commit_offset(key, batch.partitions[key].first_offset)
            except self._consumer_errors as exc:
                batch.bookmark_failures.add(key)
                logger.warning(f"kafka ECA: cannot bookmark {key} on revoke: {exc}")
        batch.release(keys)

    def _on_lost(self, consumer: Any, partitions: list[Any]) -> None:
        """Discard lost records without attempting a commit after ownership loss."""
        if self._batch is not None:
            self._batch.release({(tp.topic, tp.partition) for tp in partitions})

    def _dedupe(self, batch: DiscoveryBatch) -> list[KafkaExecutionContext]:
        """Return committed contexts in poll order, once per scheduler/job/attempt."""
        contexts: list[KafkaExecutionContext] = []
        seen: set[tuple[str, str, str]] = set()
        for key, ctx in batch.contexts:
            if key not in batch.committed:
                continue
            attempt = (ctx["scheduler"], ctx["job_id"], ctx["attempt_id"])
            if attempt in seen:
                self.last_discovery_stats["duplicates"] += 1
                logger.info(f"kafka ECA: dropping duplicate event {ctx['event_id']} for {attempt}")
            else:
                seen.add(attempt)
                contexts.append(ctx)
        return contexts

    def _parse_message(self, message: Message) -> KafkaExecutionContext | None:
        """Decode and filter CloudEvents before applying scheduler field mapping."""
        event_id, event_type, data = _read_cloudevent(message.value(), message.headers())
        if self._event_types and event_type not in self._event_types:
            self.last_discovery_stats["filtered"] += 1
            self.last_unmatched_event_types[event_type] = self.last_unmatched_event_types.get(event_type, 0) + 1
            return None
        return _context_from_data(
            data,
            event_id=event_id,
            default_scheduler=self._scheduler,
            workload_type=self._workload_type,
        )

    def discover_by_ids(self, ids: list[str]) -> list[KafkaExecutionContext]:
        """Reject scheduler lookups: this input consumes a topic, not a job index."""
        raise NotImplementedError("kafka ECA cannot look up jobs by id; run without ids to consume the topic")

    def discover_by_time_range(
        self,
        start: datetime,
        end: datetime,
        nodelist: list[str] | None = None,
    ) -> list[KafkaExecutionContext]:
        """Reject time-range scans; consume completion events using discover()."""
        raise NotImplementedError("kafka ECA does not scan time ranges; run without start_time to consume the topic")

    def close(self) -> None:
        """Leave the group once and reject future discovery on this consumer."""
        self._closed = True
        self._close_consumer()


def _weak(method: Callable[..., None]) -> Callable[..., None]:
    """Avoid a consumer/callback/adapter cycle that would prevent finalization."""
    ref = weakref.WeakMethod(method)

    def call(*args: Any) -> None:
        target = ref()
        if target is not None:
            target(*args)

    return call
