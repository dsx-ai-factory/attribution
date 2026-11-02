# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kafka ExecutionContextAdapter.

Event-driven counterpart to the pull-based adapters: a scheduler or control
plane publishes one job-completion event per attempt to a Kafka topic, and
`discover` turns the events read since the last call into ExecutionContexts.

Events are CloudEvents 1.0 in structured mode (the Kafka message value is the
JSON envelope) or binary mode (`ce_*` headers, value is `data`). The `data`
object uses ExecutionContext field names directly:

    data field         ExecutionContext
    -----------------  -----------------------------------------------------
    id                 id (required)
    start_time         start_time (required, ISO-8601 with UTC offset)
    end_time           end_time (required, ISO-8601 with UTC offset)
    nodes              nodes (required, non-empty)
    job_id             job_id (defaults to id)
    attempt_id         attempt_id (defaults to "0")
    scheduler          scheduler (defaults to the adapter's `scheduler`)
    failed_nodes       failed_node = lex-smallest entry (Kueue convention);
                       the full list is kept as `failed_nodes`
    failed_node        failed_node, when the producer sends a single node
    namespace, pod_names, pod_nodes
                       Kubernetes extensions, same keys as the Kueue adapter
    state, name, user_name, workload_framework, cancelled_by, timezone,
    time_limit, gpu_count, std_out, std_err
                       copied as-is (validated where typed)

Malformed events, including timestamps with a fixed offset the service cannot
map to an IANA zone and no `timezone`, are logged and skipped so one bad
message cannot block the topic. Repeated deliveries of the same attempt within
a batch are dropped by (scheduler, job_id, attempt_id), the key SQLiteOutput
upserts on; the CloudEvent `id` is not used because producers may mint a new
one per send. A repeat in a later batch is attributed again.

Delivery is best effort: offsets are committed before `discover` returns, so
an event whose attribution later fails, or whose process dies mid-run, is not
redelivered. `discover_by_ids` and `discover_by_time_range` are unsupported:
a topic cannot be queried by job or time.
"""

from __future__ import annotations

import json
import time
import weakref
from datetime import datetime
from typing import Any, Callable, NotRequired, Protocol, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger

from attribution_blueprint.orchestrator import _service_timezone_name

_OPTIONAL_STR_FIELDS = (
    "name",
    "user_name",
    "workload_framework",
    "cancelled_by",
    "std_out",
    "std_err",
    "namespace",
)


class KafkaExecutionContext(TypedDict):
    """Adapter-specific ExecutionContext shape.

    Same spine as `attribution_blueprint.types.ExecutionContext`; `event_id`,
    `failed_nodes`, `namespace`, `pod_names` and `pod_nodes` are extensions
    stored on the dict for the per-context raw_scheduler_json column.
    """

    id: str
    workload_type: str
    scheduler: str
    nodes: list[str]
    start_time: datetime
    end_time: datetime
    job_id: str
    attempt_id: str
    state: str
    # CloudEvent `id` of the triggering event, for tracing results back to it.
    event_id: str

    timezone: NotRequired[str]
    name: NotRequired[str]
    user_name: NotRequired[str]
    workload_framework: NotRequired[str]
    failed_node: NotRequired[str]
    failed_nodes: NotRequired[list[str]]
    cancelled_by: NotRequired[str]
    time_limit: NotRequired[int]
    gpu_count: NotRequired[int]
    std_out: NotRequired[str]
    std_err: NotRequired[str]
    namespace: NotRequired[str]
    pod_names: NotRequired[list[str]]
    pod_nodes: NotRequired[dict[str, str]]


class KafkaMessage(Protocol):
    def error(self) -> Any: ...

    def value(self) -> bytes | None: ...

    def headers(self) -> list[tuple[str, bytes]] | None: ...

    def topic(self) -> str: ...

    def partition(self) -> int: ...

    def offset(self) -> int: ...


class KafkaConsumer(Protocol):
    """Subset of `confluent_kafka.Consumer` this adapter uses."""

    def subscribe(self, topics: list[str], **callbacks: Callable[..., None]) -> None: ...

    def poll(self, timeout: float) -> KafkaMessage | None: ...

    def commit(self, *, offsets: list[Any], asynchronous: bool) -> list[Any] | None: ...

    def assignment(self) -> list[Any]: ...

    def position(self, partitions: list[Any]) -> list[Any]: ...

    def get_watermark_offsets(self, partition: Any, *, timeout: float, cached: bool) -> tuple[int, int]: ...

    def seek(self, partition: Any) -> None: ...

    def close(self) -> None: ...


PartitionKey = tuple[str, int]


class EventError(ValueError):
    """An event that cannot be turned into an ExecutionContext."""


class KafkaConsumerError(RuntimeError):
    """The consumer cannot read or commit; the run must fail, not return []."""


class KafkaExecutionContextAdapter:
    """ExecutionContextAdapter that reads job-completion CloudEvents from Kafka.

    `consumer` must be created with auto-commit disabled and partition EOF
    events enabled; the adapter subscribes it to `topic`. `topic_partition(topic, partition, offset)`
    builds the commit objects (`confluent_kafka.TopicPartition`).
    `event_types` limits which CloudEvent `type` values are accepted; empty
    accepts every type. `max_messages` bounds one `discover` batch, and
    `poll_timeout_seconds` bounds an individual poll. An empty poll ends the
    batch only when positions have reached the broker end offsets for every
    assigned partition. Unread partitions wait up to `fetch_timeout_seconds`
    without progress; an unassigned consumer waits up to
    `assignment_timeout_seconds`. Exhausting either wait raises.

    Each partition is committed independently: the native client's mixed
    commit can raise after other partitions already committed. Revoked and
    lost partitions are not committed: their contexts are removed from the
    batch and the next owner (possibly this consumer) rereads them from the
    last commit, so only contexts that `discover` returns are ever committed. Failed partitions are also dropped; still-owned ones are
    rewound. Commit errors can have an uncertain broker outcome, so redelivery
    after a process restart is not guaranteed. A failed discover clears pending
    commits before cleanup and rewinds still-owned uncommitted records.

    `last_discovery_stats` counts consumed, accepted, filtered, malformed,
    duplicate and returned events, plus failed and dropped partitions.
    `last_unmatched_event_types` counts filtered types. `discovery_incomplete`
    is true after any failure, any partition commit failure, or a filtered-only
    batch. The CLI processes any successful subset and exits 1 for incomplete
    discovery. Every discover call resets these diagnostics. Retriable consumer
    errors are retried up to `max_transient_errors`; fatal or non-retriable
    errors, and batches with no returned contexts after commit failure, raise.

    The consumer is closed by `close` or at interpreter exit. The CLI never
    calls `close`, and a consumer that exits without leaving its group makes
    the next run wait for the group to evict it before partitions are assigned.
    """

    def __init__(
        self,
        *,
        consumer: KafkaConsumer,
        topic: str,
        topic_partition: Callable[[str, int, int], Any],
        event_types: list[str] | None = None,
        scheduler: str = "kafka",
        workload_type: str = "",
        max_messages: int = 500,
        poll_timeout_seconds: float = 5.0,
        assignment_timeout_seconds: float = 60.0,
        fetch_timeout_seconds: float = 60.0,
        max_transient_errors: int = 3,
    ):
        if not topic:
            raise ValueError("kafka ECA: topic is required")
        if max_messages <= 0:
            raise ValueError(f"kafka ECA: max_messages must be positive, got {max_messages}")
        if poll_timeout_seconds <= 0:
            raise ValueError(f"kafka ECA: poll_timeout_seconds must be positive, got {poll_timeout_seconds}")
        if assignment_timeout_seconds < 0:
            raise ValueError(
                f"kafka ECA: assignment_timeout_seconds must be non-negative, got {assignment_timeout_seconds}"
            )
        if fetch_timeout_seconds <= 0:
            raise ValueError(f"kafka ECA: fetch_timeout_seconds must be positive, got {fetch_timeout_seconds}")
        if max_transient_errors < 0:
            raise ValueError(f"kafka ECA: max_transient_errors must be non-negative, got {max_transient_errors}")
        self._consumer = consumer
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
        self._commit_failures: set[PartitionKey] = set()
        self._end_offsets: dict[PartitionKey, int] = {}
        # Per-batch state, also touched by the rebalance callbacks that run
        # inside `poll`: first uncommitted and next offset per partition, and
        # dropped partitions with the offset to rewind to if still owned.
        self._first: dict[PartitionKey, int] = {}
        self._pending: dict[PartitionKey, int] = {}
        self._dropped: dict[PartitionKey, int] = {}
        self._close_consumer = weakref.finalize(self, consumer.close)
        # Weak callbacks: the consumer must not keep the adapter alive, or the
        # finalizer above (which holds the consumer) would never run early.
        consumer.subscribe([topic], on_revoke=_weak(self._on_revoke), on_lost=_weak(self._on_lost))

    def discover(self) -> list[KafkaExecutionContext]:
        self._first = {}
        self._pending = {}
        self._dropped = {}
        self._batch = []
        self._released = 0
        self._commit_failures = set()
        self._end_offsets = {}
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
        try:
            contexts = self._discover_batch()
            self.last_discovery_stats["returned"] = len(contexts)
            self.discovery_incomplete = bool(self._commit_failures) or (
                self.last_discovery_stats["filtered"] > 0 and self.last_discovery_stats["accepted"] == 0
            )
            return contexts
        except BaseException:
            self.discovery_incomplete = True
            self._abandon_batch()
            raise
        finally:
            self.last_discovery_stats["commit_failed_partitions"] = len(self._commit_failures)
            logger.info(f"kafka ECA: discovery summary {self.last_discovery_stats}")
            if self.last_unmatched_event_types:
                logger.warning(
                    f"kafka ECA: filtered event types {dict(sorted(self.last_unmatched_event_types.items()))}; "
                    f"configured event_types={sorted(self._event_types)}"
                )

    def _abandon_batch(self) -> None:
        # close() can invoke on_revoke. Clear pending offsets before cleanup
        # so an unsuccessful discover cannot commit records it never returned.
        offsets = dict(self._dropped)
        for key in self._pending:
            offsets.setdefault(key, self._first[key])
        self._pending = {}
        self._first = {}
        self._dropped = {}
        if not offsets:
            return
        try:
            owned = {(tp.topic, tp.partition) for tp in self._consumer.assignment()}
            for (topic, partition), offset in sorted(offsets.items()):
                if (topic, partition) in owned:
                    self._rewind(topic, partition, offset)
        except Exception as exc:
            logger.warning(f"kafka ECA: cannot rewind aborted batch: {exc}")
            # An unrecoverable cursor must not be reused on the next call.
            try:
                self.close()
            except Exception as close_exc:
                logger.warning(f"kafka ECA: close after aborted batch failed: {close_exc}")

    def _discover_batch(self) -> list[KafkaExecutionContext]:
        consumed = 0
        transient_errors = 0
        assignment_deadline = time.monotonic() + self._assignment_timeout_seconds
        idle_deadline: float | None = None
        while consumed < self._max_messages:
            message = self._consumer.poll(self._poll_timeout_seconds)
            if message is None:
                assigned = self._consumer.assignment()
                if not assigned:
                    idle_deadline = None
                    if assignment_deadline is None:
                        assignment_deadline = time.monotonic() + self._assignment_timeout_seconds
                    if time.monotonic() >= assignment_deadline:
                        raise KafkaConsumerError(
                            f"kafka ECA: no partitions assigned within {self._assignment_timeout_seconds}s"
                        )
                    continue
                assignment_deadline = None
                if idle_deadline is None:
                    idle_deadline = time.monotonic() + self._fetch_timeout_seconds
                if self._is_drained(assigned, idle_deadline):
                    break
                if time.monotonic() >= idle_deadline:
                    raise KafkaConsumerError(
                        f"kafka ECA: assigned partitions still have unread records after "
                        f"{self._fetch_timeout_seconds}s without progress"
                    )
                continue
            assignment_deadline = None
            idle_deadline = None
            error = message.error()
            if error is not None:
                name = getattr(error, "name", None)
                if callable(name) and name() == "_PARTITION_EOF":
                    self._end_offsets[(message.topic(), message.partition())] = message.offset()
                    continue
                transient_errors += 1
                if not _is_retriable(error) or transient_errors > self._max_transient_errors:
                    raise KafkaConsumerError(f"kafka ECA: consumer error: {error}")
                logger.warning(f"kafka ECA: retriable consumer error ({transient_errors}): {error}")
                continue
            consumed += 1
            self.last_discovery_stats["consumed"] += 1
            key = (message.topic(), message.partition())
            if key in self._dropped:
                continue
            self._first.setdefault(key, message.offset())
            self._pending[key] = message.offset() + 1
            where = f"{key[0]}[{key[1]}]@{message.offset()}"
            try:
                ctx = self._parse_message(message)
            except EventError as exc:
                self.last_discovery_stats["malformed"] += 1
                logger.warning(f"kafka ECA: skipping event at {where}: {exc}")
                continue
            if ctx is not None:
                self.last_discovery_stats["accepted"] += 1
                self._batch.append((key, ctx))

        self._drop(self._commit(self._pending))
        self._pending = {}
        contexts = self._dedupe(self._batch)
        self._batch = []
        self.last_discovery_stats["dropped_partitions"] = len(self._dropped) + self._released
        if self._commit_failures and not contexts:
            raise KafkaConsumerError(f"kafka ECA: offset commit failed for {sorted(self._commit_failures)}")
        if self._dropped:
            owned = {(tp.topic, tp.partition) for tp in self._consumer.assignment()}
            for (topic, partition), offset in sorted(self._dropped.items()):
                if (topic, partition) in owned:
                    self._rewind(topic, partition, offset)
        return contexts

    def _is_drained(self, assigned: list[Any], deadline: float) -> bool:
        try:
            positions = {(tp.topic, tp.partition): tp for tp in self._consumer.position(assigned)}
            for tp in assigned:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                position = positions.get((tp.topic, tp.partition))
                if position is not None and getattr(position, "error", None) is not None:
                    raise KafkaConsumerError(f"kafka ECA: position lookup failed: {position.error}")
                low, high = self._consumer.get_watermark_offsets(tp, timeout=remaining, cached=False)
                # An invalid initial position is not evidence that a nonempty
                # partition is drained. Wait for its fetch/reset to complete.
                next_offset = max(
                    position.offset if position is not None else -1,
                    self._end_offsets.get((tp.topic, tp.partition), -1),
                )
                if low < high and next_offset < high:
                    return False
            return True
        except KafkaConsumerError:
            raise
        except Exception as exc:
            raise KafkaConsumerError(f"kafka ECA: cannot check partition end offsets: {exc}") from exc

    def discover_by_ids(self, ids: list[str]) -> list[KafkaExecutionContext]:
        raise NotImplementedError("kafka ECA cannot look up jobs by id; run without ids to consume the topic")

    def discover_by_time_range(
        self,
        start: datetime,
        end: datetime,
        nodelist: list[str] | None = None,
    ) -> list[KafkaExecutionContext]:
        raise NotImplementedError("kafka ECA does not scan time ranges; run without start_time to consume the topic")

    def close(self) -> None:
        self._close_consumer()

    def _on_revoke(self, consumer: Any, partitions: list[Any]) -> None:
        # Do not commit on revoke: a later failure in this discover() would
        # discard the batch after its offsets were committed. Releasing keeps
        # the invariant that only returned contexts are ever committed; the
        # partition's next owner, possibly this consumer after an eager
        # rebalance, rereads its records from the last commit.
        self._release(partitions, "revoked")

    def _on_lost(self, consumer: Any, partitions: list[Any]) -> None:
        self._release(partitions, "lost")

    def _release(self, partitions: list[Any], reason: str) -> None:
        keys = {(tp.topic, tp.partition) for tp in partitions}
        for key in keys:
            self._end_offsets.pop(key, None)
        released = keys & self._pending.keys()
        for topic, partition in sorted(released):
            logger.info(f"kafka ECA: {reason} {topic}[{partition}]; its uncommitted events will be reread")
        # Not marked dropped: a reassigned partition restarts from its last
        # commit, so this batch can reread the same records.
        self._batch = [(key, ctx) for key, ctx in self._batch if key not in released]
        self._released += len(released)
        for key in released:
            del self._pending[key]
            self._first.pop(key, None)

    def _commit(self, offsets: dict[PartitionKey, int]) -> set[PartitionKey]:
        """Commit partitions independently; native mixed commits can raise
        after some offsets have already been committed, hiding their results.
        """
        failed: set[PartitionKey] = set()
        for (topic, partition), offset in sorted(offsets.items()):
            key = (topic, partition)
            try:
                results = self._consumer.commit(
                    offsets=[self._topic_partition(topic, partition, offset)], asynchronous=False
                )
                if not results or any(getattr(tp, "error", None) is not None for tp in results):
                    raise KafkaConsumerError("commit returned no successful partition result")
            except Exception as exc:
                failed.add(key)
                self._commit_failures.add(key)
                logger.warning(
                    f"kafka ECA: commit failed for {topic}[{partition}] at offset {offset}: {exc}; "
                    "its contexts will not be returned"
                )
        return failed

    def _drop(self, keys: set[PartitionKey]) -> None:
        for key in keys:
            self._dropped.setdefault(key, self._first[key])

    def _rewind(self, topic: str, partition: int, offset: int) -> None:
        try:
            self._consumer.seek(self._topic_partition(topic, partition, offset))
        except Exception as exc:
            raise KafkaConsumerError(f"kafka ECA: cannot rewind {topic}[{partition}] to {offset}: {exc}") from exc

    def _dedupe(self, batch: list[tuple[PartitionKey, KafkaExecutionContext]]) -> list[KafkaExecutionContext]:
        contexts: list[KafkaExecutionContext] = []
        seen: set[tuple[str, str, str]] = set()
        for key, ctx in batch:
            if key in self._dropped:
                continue
            attempt = (ctx["scheduler"], ctx["job_id"], ctx["attempt_id"])
            if attempt in seen:
                self.last_discovery_stats["duplicates"] += 1
                logger.info(f"kafka ECA: dropping duplicate event {ctx['event_id']} for {attempt}")
                continue
            seen.add(attempt)
            contexts.append(ctx)
        return contexts

    def _parse_message(self, message: KafkaMessage) -> KafkaExecutionContext | None:
        event_id, event_type, data = _read_cloudevent(message)
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


def _read_cloudevent(message: KafkaMessage) -> tuple[str, str, dict]:
    """Return (id, type, data) from a structured- or binary-mode CloudEvent."""
    raw = message.value()
    if raw is None:
        raise EventError("message has no value")
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventError(f"value is not JSON: {exc}") from exc

    headers = {key: value for key, value in (message.headers() or [])}
    if "ce_type" in headers:
        attrs = {key[3:]: _header_str(value) for key, value in headers.items() if key.startswith("ce_")}
        data = body
    else:
        if not isinstance(body, dict):
            raise EventError("structured CloudEvent must be a JSON object")
        attrs = body
        data = body.get("data")

    if attrs.get("specversion") != "1.0":
        raise EventError(f"unsupported CloudEvents specversion {attrs.get('specversion')!r}")
    event_id = attrs.get("id")
    event_type = attrs.get("type")
    if not isinstance(event_id, str) or event_id == "":
        raise EventError("CloudEvent id is missing")
    if not isinstance(event_type, str) or event_type == "":
        raise EventError(f"CloudEvent {event_id} type is missing")
    if not isinstance(data, dict):
        raise EventError(f"CloudEvent {event_id} data must be a JSON object")
    return event_id, event_type, data


def _weak(method: Callable[..., None]) -> Callable[..., None]:
    ref = weakref.WeakMethod(method)

    def call(*args: Any) -> None:
        target = ref()
        if target is not None:
            target(*args)

    return call


def _is_retriable(error: Any) -> bool:
    """True for transient librdkafka errors; fatal or unknown errors are not."""
    fatal = getattr(error, "fatal", None)
    retriable = getattr(error, "retriable", None)
    if callable(fatal) and fatal():
        return False
    return bool(callable(retriable) and retriable())


def _header_str(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else value


def _context_from_data(
    data: dict,
    *,
    event_id: str,
    default_scheduler: str,
    workload_type: str,
) -> KafkaExecutionContext:
    """Validate an event `data` object and build an ExecutionContext from it."""
    missing = [key for key in ("id", "start_time", "end_time", "nodes") if data.get(key) in (None, "", [])]
    if missing:
        raise EventError(f"CloudEvent {event_id} data missing required keys: {missing}")

    ctx_id = str(data["id"])
    ctx: KafkaExecutionContext = {
        "id": ctx_id,
        "workload_type": workload_type,
        "scheduler": _str_or_default(data.get("scheduler"), default_scheduler),
        "nodes": _str_list(data["nodes"], "nodes", event_id),
        "start_time": _aware_datetime(data["start_time"], "start_time", event_id),
        "end_time": _aware_datetime(data["end_time"], "end_time", event_id),
        "job_id": _str_or_default(data.get("job_id"), ctx_id),
        "attempt_id": _str_or_default(data.get("attempt_id"), "0"),
        "state": _str_or_default(data.get("state"), "UNKNOWN"),
        "event_id": event_id,
    }
    if ctx["end_time"] < ctx["start_time"]:
        raise EventError(f"CloudEvent {event_id} end_time is before start_time")

    for key in _OPTIONAL_STR_FIELDS:
        value = data.get(key)
        if value is not None and value != "":
            ctx[key] = str(value)

    timezone = data.get("timezone")
    if timezone is not None and timezone != "":
        try:
            ctx["timezone"] = ZoneInfo(str(timezone)).key
        # With the tzdata package, a directory name ("America") or an overlong
        # key raises an OSError subclass instead of ZoneInfoNotFoundError.
        except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
            raise EventError(f"CloudEvent {event_id} timezone {timezone!r} is not an IANA key") from exc

    # The service needs an IANA zone; reject now what the orchestrator would
    # reject after the offset is committed (e.g. +05:30 without `timezone`).
    try:
        _service_timezone_name(ctx["start_time"], ctx["end_time"], ctx.get("timezone"))
    except ValueError as exc:
        raise EventError(f"CloudEvent {event_id}: {exc}") from exc

    for key in ("time_limit", "gpu_count"):
        value = data.get(key)
        if value is not None:
            ctx[key] = _non_negative_int(value, key, event_id)

    failed_nodes = data.get("failed_nodes")
    if failed_nodes:
        ctx["failed_nodes"] = _str_list(failed_nodes, "failed_nodes", event_id)
        ctx["failed_node"] = min(ctx["failed_nodes"])
    elif data.get("failed_node"):
        ctx["failed_node"] = str(data["failed_node"])

    if data.get("pod_names"):
        ctx["pod_names"] = _str_list(data["pod_names"], "pod_names", event_id)
    pod_nodes = data.get("pod_nodes")
    if pod_nodes:
        if not isinstance(pod_nodes, dict):
            raise EventError(f"CloudEvent {event_id} pod_nodes must be an object")
        ctx["pod_nodes"] = {str(pod): str(node) for pod, node in pod_nodes.items()}
    return ctx


def _str_or_default(value: object, default: str) -> str:
    return default if value is None or value == "" else str(value)


def _str_list(value: object, field: str, event_id: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise EventError(f"CloudEvent {event_id} {field} must be a list of non-empty strings")
    return list(value)


def _aware_datetime(value: object, field: str, event_id: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise EventError(f"CloudEvent {event_id} {field}={value!r} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise EventError(f"CloudEvent {event_id} {field} must include a UTC offset")
    return parsed


def _non_negative_int(value: object, field: str, event_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EventError(f"CloudEvent {event_id} {field} must be a non-negative integer")
    return value
