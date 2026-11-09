# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Track acknowledged delivery prefixes independently for each Kafka partition."""

from dataclasses import dataclass, field
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


@dataclass
class _Partition:
    """Valid delivery offsets in native order; numeric gaps are allowed."""

    assignment_id: str = field(default_factory=lambda: uuid4().hex)
    pending: dict[int, bool] = field(default_factory=dict)
    committed: int = -1


class _Acknowledgements:
    """Keep receipts until broker confirmation or partition ownership loss."""

    def __init__(self) -> None:
        """Start without owned delivery state."""
        self._partitions: dict[PartitionKey, _Partition] = {}

    def register(self, topic: str, partition: int, offset: int) -> _Receipt:
        """Issue a receipt without assuming numeric offsets are consecutive."""
        key = (topic, partition)
        state = self._partitions.get(key)
        if state is None:
            state = self._partitions[key] = _Partition()
        if offset >= state.committed:
            state.pending.setdefault(offset, False)
        return _Receipt(topic, partition, offset, state.assignment_id)

    def acknowledge(self, receipt: object) -> PartitionKey:
        """Mark one receipt accepted; repeat acknowledgements are harmless."""
        if not isinstance(receipt, _Receipt):
            raise InvalidReceiptError("kafka ECA: expected a Kafka receipt")
        if not isinstance(receipt.topic, str) or type(receipt.partition) is not int or type(receipt.offset) is not int:
            raise InvalidReceiptError("kafka ECA: receipt requires a topic string and integer partition/offset")
        key = (receipt.topic, receipt.partition)
        state = self._partitions.get(key)
        if state is None or state.assignment_id != receipt.assignment_id or receipt.offset < 0:
            raise InvalidReceiptError("kafka ECA: receipt has stale or foreign partition ownership")
        if receipt.offset < state.committed:
            return key
        if receipt.offset not in state.pending:
            raise InvalidReceiptError("kafka ECA: receipt offset was not delivered by this adapter")
        state.pending[receipt.offset] = True
        return key

    def ready_offset(self, key: PartitionKey) -> int | None:
        """Return the next offset after the acknowledged delivery prefix."""
        ready = None
        for offset, acknowledged in self._partitions[key].pending.items():
            if not acknowledged:
                break
            ready = offset + 1
        return ready

    def committed(self, key: PartitionKey, offset: int) -> None:
        """Forget a prefix only after its synchronous commit was confirmed."""
        state = self._partitions[key]
        state.pending = {pending: acked for pending, acked in state.pending.items() if pending >= offset}
        state.committed = offset

    def release(self, key: PartitionKey) -> bool:
        """Invalidate ownership and report whether pending deliveries existed."""
        state = self._partitions.pop(key, None)
        return state is not None and bool(state.pending)

    def clear(self) -> None:
        """Invalidate every receipt when the adapter closes."""
        self._partitions.clear()
