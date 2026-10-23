# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Callable, Optional, Protocol

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.types import (
    ExecutionContext,
    Observation,
    Source,
    TimeInterval,
    TimeseriesEntry,
)


class LogConnector(Protocol):
    @property
    def source(self) -> Source: ...

    def collect(self, context: ExecutionContext) -> list[Observation]: ...


def entries_from_text(text: str, node: Optional[str] = None) -> list[TimeseriesEntry]:
    """One `entry_timeseries` entry per newline-separated line.

    Emits only the body unless `node` is given, in which case every entry declares
    it. Timestamp and any remaining resource parsing is handled by the service.
    """
    entries = [TimeseriesEntry(body=line) for line in text.split("\n")]
    if node is not None and node != "":
        for entry in entries:
            entry["resources"] = [{"kind": "node", "hostname": node}]
    return entries


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
    entries: list[TimeseriesEntry],
    time_interval: Optional[TimeInterval] = None,
) -> Observation:
    """Build an `entry_timeseries` log Observation, defaulting `time_interval` to the per-source contract.

    Args:
        entries: `entry_timeseries` entries, serialized into the observation body.
        time_interval: Override for `DEFAULT_TIME_INTERVAL[source]`, e.g. a tighter
            window when the backend returned unwindowed data.
    """
    if time_interval is None:
        time_interval = DEFAULT_TIME_INTERVAL[source](context)
    return {
        "context": {
            "time_interval": time_interval,
            "source": source,
        },
        "body": json.dumps(entries),
    }


if "log" not in _registry._FRAMEWORKS:
    _registry.register_framework(name="log", aliases=("logs",), sources=("applog", "syslog"))


__all__ = ["DEFAULT_TIME_INTERVAL", "LogConnector", "entries_from_text", "log_observation"]
