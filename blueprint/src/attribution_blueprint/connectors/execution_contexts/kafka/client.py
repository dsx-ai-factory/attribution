# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kafka consumer contract and native exception boundary.

The adapter uses this small protocol, as Kubernetes adapters use their client
contract. Only the native implementation needs the optional Kafka package.
Operational Kafka exceptions are translated at individual client calls;
programming errors propagate unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol


class KafkaConsumerError(RuntimeError):
    """A Kafka operation failed; discovery cannot be reported as complete."""


class KafkaMessage(Protocol):
    """Read-only record/error access used by discovery and CloudEvent decoding."""

    def error(self) -> Any: ...

    def value(self) -> bytes | None: ...

    def headers(self) -> list[tuple[str, bytes]] | None: ...

    def topic(self) -> str: ...

    def partition(self) -> int: ...

    def offset(self) -> int: ...


class KafkaConsumer(Protocol):
    """Consumer operations used by discovery.

    Implementations report operational failures as KafkaConsumerError; tests
    implement this contract directly, while KafkaConsumerClient adapts the
    native library. Client/programming errors outside that contract propagate.
    """

    def subscribe(self, topics: list[str], **callbacks: Callable[..., None]) -> None: ...

    def poll(self, timeout: float) -> KafkaMessage | None: ...

    def commit(self, *, offsets: list[Any], asynchronous: bool) -> list[Any] | None: ...

    def assignment(self) -> list[Any]: ...

    def position(self, partitions: list[Any]) -> list[Any]: ...

    def get_watermark_offsets(self, partition: Any, *, timeout: float, cached: bool) -> tuple[int, int]: ...

    def seek(self, partition: Any) -> None: ...

    def close(self) -> None: ...


class KafkaConsumerClient:
    """Translate native KafkaException failures without catching other errors.

    The wrapped consumer has automatic commits disabled. Rebalance callbacks
    are passed through and execute synchronously inside poll/close.
    """

    def __init__(self, consumer: KafkaConsumer) -> None:
        from confluent_kafka import KafkaException

        self._consumer = consumer
        self._error = KafkaException

    def subscribe(self, topics: list[str], **callbacks: Callable[..., None]) -> None:
        """Subscribe once; leave callback exceptions visible to the caller."""
        try:
            self._consumer.subscribe(topics, **callbacks)
        except self._error as exc:
            raise KafkaConsumerError(f"kafka ECA: subscribe failed: {exc}") from exc

    def poll(self, timeout: float) -> KafkaMessage | None:
        """Poll one record or service a rebalance; None means no record yet."""
        try:
            return self._consumer.poll(timeout)
        except self._error as exc:
            raise KafkaConsumerError(f"kafka ECA: poll failed: {exc}") from exc

    def assignment(self) -> list[Any]:
        """Return the partitions currently owned by this consumer."""
        try:
            return self._consumer.assignment()
        except self._error as exc:
            raise KafkaConsumerError(f"kafka ECA: assignment lookup failed: {exc}") from exc

    def position(self, partitions: list[Any]) -> list[Any]:
        """Read the next fetch positions; negative offsets are unresolved."""
        try:
            return self._consumer.position(partitions)
        except self._error as exc:
            raise KafkaConsumerError(f"kafka ECA: position lookup failed: {exc}") from exc

    def get_watermark_offsets(self, partition: Any, *, timeout: float, cached: bool) -> tuple[int, int]:
        """Fetch broker low/high offsets within the caller's idle deadline."""
        try:
            return self._consumer.get_watermark_offsets(partition, timeout=timeout, cached=cached)
        except self._error as exc:
            raise KafkaConsumerError(f"kafka ECA: cannot check partition end offsets: {exc}") from exc

    def commit(self, *, offsets: list[Any], asynchronous: bool) -> list[Any] | None:
        """Commit explicit offsets; a raised error may have an uncertain outcome."""
        try:
            return self._consumer.commit(offsets=offsets, asynchronous=asynchronous)
        except self._error as exc:
            raise KafkaConsumerError(f"kafka ECA: offset commit failed: {exc}") from exc

    def seek(self, partition: Any) -> None:
        """Rewind an owned partition; failures make consumer reuse unsafe."""
        try:
            self._consumer.seek(partition)
        except self._error as exc:
            raise KafkaConsumerError(f"kafka ECA: cannot rewind {partition}: {exc}") from exc

    def close(self) -> None:
        """Leave the group; the adapter controls idempotence and callback state."""
        try:
            self._consumer.close()
        except self._error as exc:
            raise KafkaConsumerError(f"kafka ECA: close failed: {exc}") from exc
