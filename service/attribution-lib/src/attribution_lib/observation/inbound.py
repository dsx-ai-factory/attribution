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
    r"""Specifies the format the 'lines' in an observation are in.

    `job_keywords` takes `string_list`; every other implemented source takes `entry_timeseries`.

    Rank-like prefixes match the following regex: "^\[rank\s*:?\s*(?P<rank>\d+)\]" or have an integer followed by a colon at the start of the line.

    Examples:

        [rank3] Message
         3: another message
        [rank :10] a third message


    - `string_list`:

        Raw text

        Lines of text separated by newlines. No hosts associations.

    - `entry_timeseries`:

      JSON-based.

      Top-Level Structure:
      - An array of entry objects.

      Details:
      - Each entry object contains:
        - `body` (string): *Required*. Matched verbatim. Parsed only as the per-source fallback
          described below.
        - `ts` (int or null): Unix timestamp in **nanoseconds**, or null/absent when the producer
          has no timestamp for this entry. It means "this entry occurred no later than this time".
          May be parsed from body and/or dropped if none given for some sources.
        - `resources` (optional): What this entry is about. Either the exact string `"AllJobResources"`
          or an array naming exactly one `{"kind": "node", "hostname": ...}`, optionally alongside
          `{"kind": "replica", "name": ...}` objects, which are recorded but not attributed on. An array
          naming several nodes, or a hostname outside the job, is invalid and handled like an entry naming
          no node. For application logs, an empty array explicitly declares unknown node identity;
          it prevents body-based node inference. Node may be parsed from body and/or dropped if none
          given for other sources.
      - Any other key in an entry is rejected.

      Entries are time-filtered only by the observation context's `time_interval`. That interval is the
      producer's declared intent: a null bound means unbounded on that side, so an interval of two nulls
      disables time filtering for the observation entirely.

      Key Source Behavior:

      The sources listed below read an entry the same way unless noted; a source not listed requires
      nothing and parses nothing out of the body.
      - `ts` is required: an entry without it is dropped.
      - A declared node is used as-is, so a host specifier in IP form must be resolved by the producer. If
        no node is declared, the node is parsed out of the body instead, keeping any declared replicas. An
        entry that still names no node is dropped.
      - `body` is the line as received. The library never parses its shape beyond the fallbacks
        noted below; what a body must look like is whatever the source's rules match.

      Deviations:
      - `applog`:
        - `ts` can be null, in which case the body is parsed for a timestamp. Year-less formats resolve it
          against the workload's `job_end_time`.
        - When `resources` is absent or null, the body node is parsed from an explicit rank-like or
          node-name text prefix. Message content, timestamps and unrecognized producer prefixes do not
          imply rank zero. An entry that still names no node, or declares an invalid node, is retained
          as job-level evidence with an empty resource set; it does not identify any job node.
        - Explicit `resources: []` preserves unknown node identity without parsing a replacement from
          the body. Explicit `"AllJobResources"` retains its distinct meaning and skips prefix parsing.
        - The caller must establish which execution owns the entry before submitting it. Unknown node
          identity does not prevent its symptom from affecting that execution's outcome.
      - `syslog` & `health_check`: the node is parsed from the hostname field of the body.
      - `ufmlog`:
        - `ts` can be null, in which case the entry is kept and treated as untimed.
        - The body node is parsed from either the link form `(Computer:<node> (<guid>))` or the node
          form `(<node>) node is Down`.
      - `node_metrics` and `nvlink_metrics`:
        - `ts` can be null, in which case the entry is kept and treated as untimed.
        - `resources` must name the node; the body is never parsed for one.
      - `csp_event`:
        - `ts` must be set; the body is never parsed for one. An entry without it is dropped.
        - `resources` must name the node; the body is never parsed for one.
        - There is no received line: producers render the body as
          `csp_event: category=<category>, reason=<reason>, status=<status>`, optionally followed by
          `, source_event=<compact JSON, sorted keys>` carrying the provider payload.

      Example:
      ```json
      [
        {
          "ts": 1744734437830498000,
          "resources": [{"kind": "node", "hostname": "host-a1b2c3d4"}],
          "body": "some log message"
        },
        {
          "body": "an untimed entry; kept or dropped per the source rules above"
        }
      ]
      ```
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
    resources: ResourceSet
    source: Source
    format: Format
    timezone: ZoneInfo
    # Processing directive, not identity.
    pre_filtered: bool = field(default=False, kw_only=True, compare=False)

    def __hash__(self) -> int:
        return hash((self.time_interval, self.resources, self.source, self.format, self.timezone))

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
