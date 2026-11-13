# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the Kafka execution-context consumer through the optional OpenTelemetry provider.

The owner thread records counters and publishes offset snapshots. Observable
gauges run on the metric reader's thread, so they read only those snapshots
under a lock and never call the native consumer.
"""

import threading
import time
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

from opentelemetry import metrics
from opentelemetry.metrics import CallbackOptions, Observation

from attribution_blueprint.connectors.execution_contexts.kafka.receipts import Backlog, PartitionKey


class SkipReason(StrEnum):
    """Why a consumed record produced no context."""

    FILTERED = "filtered"
    MALFORMED = "malformed"


class PartitionChange(StrEnum):
    """Ownership change reported by a rebalance callback."""

    ASSIGNED = "assigned"
    REVOKED = "revoked"
    LOST = "lost"


# The same meter as the connector request metrics in attribution_blueprint.telemetry.
_meter = metrics.get_meter("attribution_blueprint")
_consumed = _meter.create_counter(
    "attribution.kafka.consumer.consumed", unit="{record}", description="Records read from the broker."
)
_accepted = _meter.create_counter(
    "attribution.kafka.consumer.accepted",
    unit="{record}",
    description="Valid records delivered by poll() or added to a discovery batch.",
)
_skipped = _meter.create_counter(
    "attribution.kafka.consumer.skipped", unit="{record}", description="Records skipped as filtered or malformed."
)
_acknowledged = _meter.create_counter(
    "attribution.kafka.consumer.acknowledged", unit="{record}", description="Receipts acknowledged for the first time."
)
_commits = _meter.create_counter(
    "attribution.kafka.consumer.commits", unit="{commit}", description="Partition offset commits the broker confirmed."
)
_commit_errors = _meter.create_counter(
    "attribution.kafka.consumer.commit.errors", unit="{commit}", description="Partition offset commits that failed."
)
_errors = _meter.create_counter(
    "attribution.kafka.consumer.errors", unit="{error}", description="Errors returned or raised by consumer polls."
)
_rebalances = _meter.create_counter(
    "attribution.kafka.consumer.rebalances",
    unit="{partition}",
    description="Partitions assigned, revoked, or lost in rebalance callbacks.",
)


@dataclass
class _Offsets:
    """Latest offsets for one owned partition; None until known."""

    committed: int | None = None
    position: int | None = None
    high: int | None = None


class PartitionState(NamedTuple):
    """Gauge values for one owned partition at collection time."""

    labels: dict[str, str | int]
    lag: int | None
    position_lag: int | None
    unacknowledged: int
    oldest_unacknowledged_age: float


class ConsumerMetrics:
    """Counters and gauge snapshots for one adapter's consumer.

    The committed offset starts at the first record consumed after assignment,
    which is where the group's committed offset resumed, and then follows this
    adapter's confirmed commits.
    """

    def __init__(self, *, topic: str, group_id: str, backlog: Callable[[], dict[PartitionKey, Backlog]]) -> None:
        """Register for gauge collection until close().

        Args:
            topic: Subscribed topic, used to label errors without a partition.
            group_id: Consumer group used as the `consumer_group` label.
            backlog: Thread-safe view of unacknowledged deliveries.
        """
        self._topic = topic
        self._group_id = group_id
        self._backlog = backlog
        self._lock = threading.Lock()
        self._offsets: dict[PartitionKey, _Offsets] = {}
        with _registry_lock:
            _registry.add(self)

    def _labels(self, key: PartitionKey) -> dict[str, str | int]:
        """Return the fixed labels for one partition."""
        return {"topic": key[0], "partition": key[1], "consumer_group": self._group_id}

    def consumed(self, key: PartitionKey, offset: int) -> None:
        """Count one record and remember where consumption resumed."""
        _consumed.add(1, self._labels(key))
        with self._lock:
            offsets = self._offsets.setdefault(key, _Offsets())
            if offsets.committed is None:
                offsets.committed = offset

    def accepted(self, key: PartitionKey) -> None:
        """Count one valid record."""
        _accepted.add(1, self._labels(key))

    def skipped(self, key: PartitionKey, reason: SkipReason) -> None:
        """Count one record that produced no context."""
        _skipped.add(1, {**self._labels(key), "reason": reason})

    def acknowledged(self, key: PartitionKey) -> None:
        """Count one first-time acknowledgement."""
        _acknowledged.add(1, self._labels(key))

    def committed(self, key: PartitionKey, offset: int) -> None:
        """Count a confirmed commit and use its offset for the committed lag."""
        _commits.add(1, self._labels(key))
        with self._lock:
            self._offsets.setdefault(key, _Offsets()).committed = offset

    def commit_failed(self, key: PartitionKey, error_type: str) -> None:
        """Count a failed or uncertain commit."""
        _commit_errors.add(1, {**self._labels(key), "error.type": error_type})

    def consumer_error(self, error_type: str) -> None:
        """Count a poll error that is not a partition end-of-file event."""
        _errors.add(1, {"topic": self._topic, "consumer_group": self._group_id, "error.type": error_type})

    def partitions_changed(self, change: PartitionChange, keys: Iterable[PartitionKey]) -> None:
        """Count ownership changes and start or stop reporting those partitions."""
        with self._lock:
            for key in keys:
                _rebalances.add(1, {**self._labels(key), "change": change})
                if change is PartitionChange.ASSIGNED:
                    self._offsets.setdefault(key, _Offsets())
                else:
                    self._offsets.pop(key, None)

    def positions(self, key: PartitionKey, *, position: int | None, high: int | None) -> None:
        """Publish the local fetch position and cached high watermark; None leaves a value unchanged."""
        with self._lock:
            offsets = self._offsets.setdefault(key, _Offsets())
            if position is not None:
                offsets.position = position
            if high is not None:
                offsets.high = high

    def close(self) -> None:
        """Stop reporting gauges for this consumer."""
        with _registry_lock:
            _registry.discard(self)
        with self._lock:
            self._offsets.clear()

    def partition_states(self) -> list[PartitionState]:
        """Compute gauge values for owned partitions; safe to call from any thread."""
        backlog = self._backlog()
        now = time.monotonic()
        with self._lock:
            offsets = {
                key: _Offsets(value.committed, value.position, value.high) for key, value in self._offsets.items()
            }
        states = []
        for key in sorted(offsets.keys() | backlog.keys()):
            value = offsets.get(key, _Offsets())
            waiting = backlog.get(key, Backlog(0, None))
            age = 0.0 if waiting.oldest_delivered_at is None else max(now - waiting.oldest_delivered_at, 0.0)
            states.append(
                PartitionState(
                    self._labels(key),
                    _lag(value.high, value.committed),
                    _lag(value.high, value.position),
                    waiting.unacknowledged,
                    age,
                )
            )
        return states


def _lag(high: int | None, offset: int | None) -> int | None:
    """Return records between an offset and the high watermark, if both are known."""
    if high is None or offset is None:
        return None
    return max(high - offset, 0)


_registry: weakref.WeakSet[ConsumerMetrics] = weakref.WeakSet()
_registry_lock = threading.Lock()


def _states() -> list[PartitionState]:
    """Snapshot every live consumer's partitions."""
    with _registry_lock:
        consumers = list(_registry)
    return [state for consumer in consumers for state in consumer.partition_states()]


def _observe_lag(_options: CallbackOptions) -> list[Observation]:
    return [Observation(state.lag, state.labels) for state in _states() if state.lag is not None]


def _observe_position_lag(_options: CallbackOptions) -> list[Observation]:
    return [Observation(state.position_lag, state.labels) for state in _states() if state.position_lag is not None]


def _observe_unacknowledged(_options: CallbackOptions) -> list[Observation]:
    return [Observation(state.unacknowledged, state.labels) for state in _states()]


def _observe_oldest_unacknowledged_age(_options: CallbackOptions) -> list[Observation]:
    return [Observation(state.oldest_unacknowledged_age, state.labels) for state in _states()]


_meter.create_observable_gauge(
    "attribution.kafka.consumer.lag",
    callbacks=[_observe_lag],
    unit="{record}",
    description="High watermark minus the committed offset for each owned partition.",
)
_meter.create_observable_gauge(
    "attribution.kafka.consumer.position.lag",
    callbacks=[_observe_position_lag],
    unit="{record}",
    description="High watermark minus the consumer's fetch position for each owned partition.",
)
_meter.create_observable_gauge(
    "attribution.kafka.consumer.unacknowledged",
    callbacks=[_observe_unacknowledged],
    unit="{record}",
    description="Records delivered by poll() that the caller has not acknowledged.",
)
_meter.create_observable_gauge(
    "attribution.kafka.consumer.oldest_unacknowledged.age",
    callbacks=[_observe_oldest_unacknowledged_age],
    unit="s",
    description="Time since the earliest unacknowledged record in each partition was delivered.",
)
