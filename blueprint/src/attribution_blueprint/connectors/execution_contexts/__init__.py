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
    """The adapter cannot acknowledge the supplied delivery."""


class ExecutionContextDelivery(Protocol):
    """A delivered execution context awaiting acknowledgement.

    Providers may expose provenance or acknowledgement handles as extra
    attributes on their delivery types.
    """

    @property
    def context(self) -> ExecutionContext:
        """Mapped workload context, without transport receipt metadata."""
        ...

    @property
    def event(self) -> Mapping[str, object]:
        """Original provider event data, without an assumed envelope format."""
        ...


@runtime_checkable
class AcknowledgingExecutionContextAdapter(Protocol):
    """Optional consumer lifecycle, driven by one owner thread.

    The caller chooses when work is handled and acknowledges it explicitly.
    Keep polling while processing separately, including while paused.
    """

    def poll(self, max_records: int, timeout_seconds: float) -> Sequence[ExecutionContextDelivery]:
        """Return up to max_records deliveries without acknowledging them."""
        ...

    def ack(self, delivery: ExecutionContextDelivery) -> None:
        """Mark one delivery handled; raise InvalidReceiptError if it cannot be."""
        ...

    def pause(self) -> None:
        """Stop returning new deliveries until resume(); keep calling poll()."""
        ...

    def resume(self) -> None:
        """Allow new deliveries after pause()."""
        ...

    def close(self) -> None:
        """Release resources without acknowledging unfinished deliveries."""
        ...


@runtime_checkable
class AcknowledgingExecutionContextSource(Protocol):
    """Optional ECA factory for independent long-lived consumers."""

    def open_consumer(self) -> AcknowledgingExecutionContextAdapter:
        """Open an independent consumer using this adapter's configuration."""
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
