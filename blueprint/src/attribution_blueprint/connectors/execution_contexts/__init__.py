# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.types import ExecutionContext


class ExecutionContextAdapter(Protocol):
    def discover(self) -> list[ExecutionContext]: ...

    def discover_by_ids(self, ids: list[str]) -> list[ExecutionContext]: ...

    def discover_by_time_range(
        self,
        start: datetime,
        end: datetime,
        nodelist: list[str] | None = None,
    ) -> list[ExecutionContext]: ...


class InvalidReceiptError(ValueError):
    """A receipt does not belong to this adapter's current ownership."""


class ExecutionContextDelivery(Protocol):
    """A delivered event, execution context, and provider receipt metadata."""

    @property
    def topic(self) -> str:
        """Topic that delivered the event."""
        ...

    @property
    def partition(self) -> int:
        """Partition that owns the record's offset."""
        ...

    @property
    def offset(self) -> int:
        """Position of the delivered record within its partition."""
        ...

    @property
    def context(self) -> ExecutionContext:
        """Mapped workload context, without transport receipt metadata."""
        ...

    @property
    def event(self) -> Mapping[str, object]:
        """CloudEvent envelope and original JSON data."""
        ...


@runtime_checkable
class AcknowledgingExecutionContextAdapter(Protocol):
    """Optional consumer lifecycle, driven by one owner thread.

    The caller chooses when work is handled and acknowledges it explicitly.
    Keep polling while processing separately, including while paused.
    """

    def poll(self, max_records: int, timeout_seconds: float) -> Sequence[ExecutionContextDelivery]:
        """Read a bounded batch without committing offsets."""
        ...

    def ack(self, receipt: ExecutionContextDelivery) -> None:
        """Commit the acknowledged prefix; reject stale or foreign receipts."""
        ...

    def pause(self) -> None:
        """Stop delivery while continuing to service group membership."""
        ...

    def resume(self) -> None:
        """Resume delivery on the currently owned partitions."""
        ...

    def close(self) -> None:
        """Release ownership without committing unfinished work."""
        ...


@runtime_checkable
class AcknowledgingExecutionContextSource(Protocol):
    """Optional ECA factory for independent long-lived consumers."""

    def open_consumer(self) -> AcknowledgingExecutionContextAdapter:
        """Open a new consumer using the configured topic and consumer group."""
        ...


if "eca" not in _registry._FRAMEWORKS:
    _registry.register_framework(name="eca", aliases=("execution_context",), has_selector=True)


__all__ = [
    "AcknowledgingExecutionContextAdapter",
    "AcknowledgingExecutionContextSource",
    "ExecutionContextAdapter",
    "ExecutionContextDelivery",
    "InvalidReceiptError",
]
