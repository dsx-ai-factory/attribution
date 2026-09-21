# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from typing import Callable, Optional, Protocol

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.types import (
    ExecutionContext,
    Format,
    Observation,
    Source,
    TimeInterval,
    TimeseriesEntry,
)


class LogConnector(Protocol):
    @property
    def source(self) -> Source: ...

    def collect(self, context: ExecutionContext) -> list[Observation]: ...


def entries_from_text(text: str) -> list[TimeseriesEntry]:
    """One ENTRY_TIMESERIES entry per newline-separated line. Only emits the body field.
    Timestamp and resource parsing, if allowed, is handled by the service."""
    return [TimeseriesEntry(body=line) for line in text.split("\n")]


def _unbounded(context: ExecutionContext) -> TimeInterval:
    return TimeInterval(start=None, end=None)


def _context_window(context: ExecutionContext) -> TimeInterval:
    return TimeInterval(start=context["start_time"], end=context["end_time"])


# Per-source default for the observation context's time_interval.
DEFAULT_TIME_INTERVAL: dict[Source, Callable[[ExecutionContext], TimeInterval]] = {
    Source.APPLOG: _unbounded,
    Source.SYSLOG: _context_window,
}


def log_observation(
    *,
    source: Source,
    context: ExecutionContext,
    format: Format = Format.ENTRY_TIMESERIES,
    body: Optional[str] = None,
    entries: Optional[list[TimeseriesEntry]] = None,
    resources: str = "AllJobResources",
    time_interval: Optional[TimeInterval] = None,
) -> Observation:
    """Build a log Observation, defaulting `time_interval` to the per-source contract.

    Args:
        body: Payload already serialized for `format`. Mutually exclusive with `entries`.
        entries: ENTRY_TIMESERIES entries
        time_interval: Override for `DEFAULT_TIME_INTERVAL[source]`, e.g. a tighter
            window when the backend returned unwindowed data.
    """
    if (body is None) == (entries is None):
        raise ValueError("log_observation takes exactly one of body or entries")
    if entries is not None and format is not Format.ENTRY_TIMESERIES:
        raise ValueError(f"entries= requires Format.ENTRY_TIMESERIES, got {format.value!r}")
    if entries is not None:
        body = json.dumps(entries)
    if time_interval is None:
        time_interval = DEFAULT_TIME_INTERVAL[source](context)
    return {
        "context": {
            "time_interval": time_interval,
            "source": source,
            "format": format,
            "resources": resources,
        },
        "body": body,
    }


if "log" not in _registry._FRAMEWORKS:
    _registry.register_framework(name="log", aliases=("logs",), sources=("applog", "syslog"))


__all__ = ["DEFAULT_TIME_INTERVAL", "LogConnector", "entries_from_text", "log_observation"]
