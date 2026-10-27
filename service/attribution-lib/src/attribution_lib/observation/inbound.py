# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, NewType, NotRequired, Optional, TypeAlias, TypedDict, TypeVar, cast
from zoneinfo import ZoneInfo

from attribution_lib.exceptions import (
    InvalidTimeIntervalError,
    UnknownObservationSourceError,
)
from attribution_lib.resources import ResourceSet

Id = NewType("Id", int)


class Source(str, Enum):
    """The classes of observations that can be ingested.

    `job_keywords` takes the `string_list` format; every other implemented source takes `entry_timeseries`.
    See `Format` for how each source reads its entries.

    - `slurm_info`

        Automatically attached to JobInfo objects; not supported for other Observations. No relevant Format.

    - `job_keywords`

        Keywords extracted from preprocessing some logs.

    - `applog`

        Application's stdout/stderr.

    - `syslog`

        Data from dmesg/syslog.

    - `health_check`

        Data from health checks.

    - `ufmlog`

        Fabric manager events.

    - `node_metrics`

       Prometheus metrics data for node attribution.

    - `nvlink_metrics`

       Prometheus metrics data for nvlink attribution.

    - `ib_metrics`

       Not implemented.

    - `eth_metrics`

       Not implemented.

    - `network_syslog`

       Not implemented.

    - `csp_event`

       CSP event events indicating infrastructure-level host failures,
       repairs, or maintenance actions reported by the cloud service provider.

    """

    UNDEFINED = ""
    JOB_INFO = "slurm_info"
    JOB_KEYWORDS = "job_keywords"
    APPLOG = "applog"
    SYSLOG = "syslog"
    HEALTH_CHECK = "health_check"
    NODE_METRICS = "node_metrics"  # DGCM + NODE_EXPORTER
    NVLINK_METRICS = "nvlink_metrics"  # i. e. NMX
    IB_METRICS = "ib_metrics"  # i.e. UFM
    ETH_METRICS = "eth_metrics"  # RoCE, etc
    NETWORK_SYSLOG = "network_syslog"
    UFMLOG = "ufmlog"
    CSP_EVENT = "csp_event"


class Format(str, Enum):
    """Specifies the format the 'lines' in an observation are in.

    `job_keywords` takes `string_list`; every other implemented source takes `entry_timeseries`.

    - `string_list`: lines of text separated by newlines. No host associations.

    - `entry_timeseries`: a JSON array of entry objects, each with
        - a required `body` string,
        - an optional `ts` (Unix nanoseconds, meaning "this entry occurred no later than this time") and
        - an optional `resources` (the exact string `"AllJobResources"`, or an array naming exactly one
            `{"kind": "node", "hostname": ...}` optionally alongside `{"kind": "replica", "name": ...}`
            objects).
        Any other key is rejected.

      Entries are time-filtered only by the context's `time_interval`; a null bound is unbounded on that
      side. What each source does with a missing `ts` or node, and what it parses out of the body
      instead, is the per-source `SourceProcessingConfig` registered in `core.py`. A declared node that
      is not usable (several nodes, or a hostname outside the job) skips body parsing: `applog` counts
      the entry against every job node, every other source drops it.

      Per-source behavior and examples are fully documented in
      `service/service-app/docs/api/observation-format.md`.
    """

    UNDEFINED = ""
    STRING_LIST = "string_list"
    ENTRY_TIMESERIES = "entry_timeseries"


@dataclass(frozen=True)
class TimeInterval:
    """The window a producer declares an observation covers; a null bound is unbounded on that side.

    Raises:
        InvalidTimeIntervalError: If both bounds are set and `start` is later than `end`.
    """

    start: Optional[datetime]
    end: Optional[datetime]

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise InvalidTimeIntervalError(self)

    # isoformat keeps the UTC offset in the hash, unlike datetime.__hash__.
    def __hash__(self) -> int:
        return hash(
            (
                self.start.isoformat() if self.start is not None else None,
                self.end.isoformat() if self.end is not None else None,
            )
        )


TimestampNsField: TypeAlias = int

RawLogLine: TypeAlias = str

NodeStr: TypeAlias = str

StringList: TypeAlias = list[str]


class TimeseriesEntry(TypedDict):
    """One entry of an `entry_timeseries` observation. See `Format` for the wire contract."""

    body: RawLogLine
    ts: NotRequired[Optional[TimestampNsField]]
    resources: NotRequired[Optional[ResourceSet]]


TimeseriesEntrySeries: TypeAlias = list[TimeseriesEntry]


@dataclass
class Context:
    """Describes the scope of an observation."""

    time_interval: TimeInterval
    source: Source
    format: Format
    timezone: ZoneInfo
    # Processing directive, not identity.
    pre_filtered: bool = field(default=False, kw_only=True, compare=False)

    def __hash__(self) -> int:
        return hash((self.time_interval, self.source, self.format, self.timezone))

    def __post_init__(self):
        assert isinstance(self.timezone, ZoneInfo)


V = TypeVar("V", bound=Iterable[Any])


class Observation(Generic[V]):
    """Describes association of some observation data with a context."""

    context: Context
    lines: V

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Observation):
            raise NotImplementedError
        # list(...) equality compares length and contents; the previous
        # zip-based comparison silently returned True when one side was a prefix
        # of the other, which was a latent bug.
        #
        # ty does not propagate Observation's TypeVar bound (Iterable) through
        # isinstance-narrowed `other`, so an explicit cast is needed on
        # other.lines. mypy accepts it without the cast.
        return self.context == other.context and list(self.lines) == list(cast("Iterable[Any]", other.lines))

    def __hash__(self) -> int:
        match self.context.format:
            case Format.UNDEFINED:
                raise UnknownObservationSourceError(self.context)
            case _:
                return hash((self.context, json.dumps(self.lines)))

    def __init__(self, context: Context, lines: V):
        self.context = context
        self.lines = lines

    def __repr__(self) -> str:
        return f"Observation(context={self.context}, lines={self.lines})"
