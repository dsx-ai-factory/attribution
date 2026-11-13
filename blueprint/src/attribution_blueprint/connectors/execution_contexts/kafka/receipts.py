# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Track acknowledged delivery prefixes independently for each Kafka partition."""

import threading
import time
from dataclasses import dataclass, field
from typing import NamedTuple
from uuid import uuid4

from attribution_blueprint.connectors.execution_contexts import InvalidReceiptError
from attribution_blueprint.connectors.execution_contexts.kafka.events import KafkaExecutionContext

PartitionKey = tuple[str, int]


@dataclass(frozen=True)
class _Receipt:
    """Kafka provenance and an opaque token for this partition's ownership.

    Store topic, partition and offset with the accepted context. The assignment
    token cannot be used after ownership is released or the adapter restarts.
    """

    topic: str
    partition: int
    offset: int
    assignment_id: str


@dataclass(frozen=True)
class KafkaReceipt(_Receipt):
    """One delivered record with its context, CloudEvent, and ownership token."""

    context: KafkaExecutionContext
    event: dict


class Backlog(NamedTuple):
    """Delivered receipts in one partition that the caller has not acknowledged."""

    unacknowledged: int
    oldest_delivered_at: float | None  # time.monotonic() of the earliest unacknowledged delivery


@dataclass
class _Partition:
    """Valid delivery offsets in native order; numeric gaps are allowed.

    Each pending offset maps to its monotonic delivery time, or None once acknowledged.
    """

    assignment_id: str = field(default_factory=lambda: uuid4().hex)
    pending: dict[int, float | None] = field(default_factory=dict)
    committed: int = -1


class _Acknowledgements:
    """Keep receipts until broker confirmation or partition ownership loss.

    The owner thread makes every change. The lock lets a metrics reader take a
    consistent backlog snapshot from another thread.
    """

    def __init__(self) -> None:
        """Start without owned delivery state."""
        self._partitions: dict[PartitionKey, _Partition] = {}
        self._lock = threading.Lock()

    def register(self, topic: str, partition: int, offset: int) -> _Receipt:
        """Issue a receipt without assuming numeric offsets are consecutive."""
        key = (topic, partition)
        with self._lock:
            state = self._partitions.get(key)
            if state is None:
                state = self._partitions[key] = _Partition()
            if offset >= state.committed:
                state.pending.setdefault(offset, time.monotonic())
        return _Receipt(topic, partition, offset, state.assignment_id)

    def acknowledge(self, receipt: object) -> tuple[PartitionKey, bool]:
        """Mark one receipt accepted; repeat acknowledgements are harmless.

        Returns:
            The receipt's partition, and whether this call acknowledged it for the first time.
        """
        if not isinstance(receipt, _Receipt):
            raise InvalidReceiptError("kafka ECA: expected a Kafka receipt")
        if not isinstance(receipt.topic, str) or type(receipt.partition) is not int or type(receipt.offset) is not int:
            raise InvalidReceiptError("kafka ECA: receipt requires a topic string and integer partition/offset")
        key = (receipt.topic, receipt.partition)
        state = self._partitions.get(key)
        if state is None or state.assignment_id != receipt.assignment_id or receipt.offset < 0:
            raise InvalidReceiptError("kafka ECA: receipt has stale or foreign partition ownership")
        if receipt.offset < state.committed:
            return key, False
        if receipt.offset not in state.pending:
            raise InvalidReceiptError("kafka ECA: receipt offset was not delivered by this adapter")
        first = state.pending[receipt.offset] is not None
        with self._lock:
            state.pending[receipt.offset] = None
        return key, first

    def ready_offset(self, key: PartitionKey) -> int | None:
        """Return the next offset after the acknowledged delivery prefix."""
        ready = None
        for offset, delivered_at in self._partitions[key].pending.items():
            if delivered_at is not None:
                break
            ready = offset + 1
        return ready

    def committed(self, key: PartitionKey, offset: int) -> None:
        """Forget a prefix only after its synchronous commit was confirmed."""
        state = self._partitions[key]
        with self._lock:
            state.pending = {pending: at for pending, at in state.pending.items() if pending >= offset}
            state.committed = offset

    def release(self, key: PartitionKey) -> bool:
        """Invalidate ownership and report whether pending deliveries existed."""
        with self._lock:
            state = self._partitions.pop(key, None)
        return state is not None and bool(state.pending)

    def clear(self) -> None:
        """Invalidate every receipt when the adapter closes."""
        with self._lock:
            self._partitions.clear()

    def backlog(self) -> dict[PartitionKey, Backlog]:
        """Count unacknowledged deliveries per partition; safe to call from any thread."""
        with self._lock:
            pending = {key: list(state.pending.values()) for key, state in self._partitions.items()}
        backlog = {}
        for key, delivered_at in pending.items():
            waiting = [at for at in delivered_at if at is not None]
            backlog[key] = Backlog(len(waiting), waiting[0] if waiting else None)
        return backlog
