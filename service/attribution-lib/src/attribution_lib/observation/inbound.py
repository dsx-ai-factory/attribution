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
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, NamedTuple, NewType, NotRequired, Optional, TypeAlias, TypedDict, TypeVar, cast
from zoneinfo import ZoneInfo

from attribution_lib.exceptions import (
    UnknownObservationSourceError,
)
from attribution_lib.resources import ResourceSet

Id = NewType("Id", int)


class Source(str, Enum):
    """The classes of observations that can be ingested.

    - `slurm_info`

        Automatically attached to JobInfo objects; not supported for other Observations. No relevant Format.

    - `job_keywords`

        Formats: `[string_list]`

        Keywords extracted from preprocessing some logs.

    - `applog`

        Formats: `[raw_log, loki_dict_list, raw_loki_streams]`. Timestamps should in in nanoseconds, entries must have rank-like prefixes.

        Application's stdout/stderr.

    - `syslog`

        Formats: `[loki_dict_list, raw_loki_streams, entry_timeseries]`. Timestamps should be in nanoseconds.
        Host-specifiers required.

        Data from dmesg/syslog

    - `health_check`

        Formats: `[loki_dict_list, raw_loki_streams, entry_timeseries]`. Timestamps should be in milliseconds for
        `loki_dict_list`, nanoseconds otherwise. Host-specifiers required.

        Data from health checks.

    - `node_metrics`

       Formats: `[prometheus_metrics, entry_timeseries]`. Timestamps should be in seconds for `prometheus_metrics` and
       nanoseconds for `entry_timeseries`, entries must have node information.

       Prometheus metrics data for node attribution.

    - `nvlink_metrics`

       Formats: `[prometheus_metrics, entry_timeseries]`. Timestamps should be in seconds for `prometheus_metrics` and
       nanoseconds for `entry_timeseries`, entries must have node information.

       Prometheus metrics data for nvlink attribution.

    - `ib_metrics`

       Not implemented.

    - `eth_metrics`

       Not implemented.

    - `network_syslog`

       Not implemented.

    - `csp_event`

       Formats: `[csp_event_list, entry_timeseries]`. Timestamps should be in seconds for
       `csp_event_list`, nanoseconds otherwise.

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
    r"""Specifies the types of formats the 'lines' in an observation can be in.

    Not all sources can be specified in each format; see above for clarification.

    Furthermore, some sources expect slightly different fields in formats.

    In particular, the `applog` source expects the entry found in the given format to have a rank-like prefix; for them, the host-specifier field, if present in a format, is optional. `entry_timeseries` is the exception: it parses the prefix only out of an entry that declares no `resources`, and a body timestamp only when `ts` is null (see below).

    For `syslog` and `health_check`, the rank-like prefix is not expected, and `host-specifier` fields are mandatory.

    Rank-like prefixes are match the following regex: "^\[rank\s*:?\s*(?P<rank>\d+)\]" or have an integer followed by a colon at the start of the line.

    Examples:

        [rank3] Message
         3: another message
        [rank :10] a third message


    - `string_list`:

        Raw text

        Lines of text separated by newlines. No hosts associations.

    - `raw_log`:

        Raw text

        entries separated by newlines. Rank-like prefix expected for applog.

    - `raw_loki_streams`:

        JSON-based

        Top-Level Structure:
        - An array of objects, each with:
          - `stream`: If host-spcifier needed, *Required*. If present, must contain the `node_name` field (string); which is the host-specifier. All other fields in stream are permitted but ignored.
          - `values`: *Required*. An array of entries.

        Details:
        - `stream` Object. *Required* if host-specifier:
          - Required Field:
            - `node_name` (string): Identifies the node that generated the logs. Host specifier.
            - Other fields: Any other fields within stream are permitted but ignored.
        - `values` Array:
          - Each entry is a 2-element array:
            - Timestamp (string): Unix timestamp; unit scale dictated by Source (see above).
            - Entry (string): the main content. Should have rank-line prefix for applog.
        - Other fields:
          - Any other fields at any level are permitted but ignored.

        Example (`syslog` source):
        ```json
        [
          {
            "stream": {
              "node_name": "host-a1b2c3d4"
            },
            "values": [
              [
                "1744734437830498000",
                "Apr 24 12:04:06 host-a1b2c3d4 kernel: message repeated 8 times: [ NVRM: subdeviceCtrlCmdBusGetNvlinkStatus_IMPL: Failed to collect nvlink status info!]"
              ]
            ]
          }
        ]
        ```

    - `loki_dict_list`:

        JSON-based.

        Top-Level Structure:
        -  An array of objects, each with:
            - `values`: Required. An array of entries.

        Details:
        - `values` Array:
          Each entry is a 2-element array:
          - Timestamp (string): Unix timestamp; unit scale dictated by Source (see above).
          - Content (string): A JSON-encoded object with at least the following fields:
            - `body` (string): entry (required; rank-like prefix expected for APPLOG.)
            - `hostname` (string): The source hostname (required if host-specified needed.)
        - Other fields:
          - The `stream` object is optional and, if present, is ignored.
          - Any other fields in the log message JSON are permitted but ignored.

        Example (`syslog` source)
        ```json
        [
          {
            "values": [
              [
                "1744734421212766000",
                "{\"body\":\"Some log message\",\"hostname\":\"server-01\"}"
              ]
            ]
          }
        ]
        ```

    - `csv_metric_response`:

      Not implemented presently.

    - `prometheus_metrics`:

      JSON-based.

      Top-Level Structure:
      - An array of objects, each representing a metric data point.

      Details:
      - Each object must contain:
        - `timestamp` (int): Unix timestamp in seconds
        - `value` (Union[int, float, string]): Metric value (initially as string, converted later)
        - `metric` (string): Metric name (e.g., "node_infiniband_link_downed_total")
        - `labels` (dict): Prometheus labels including node and device information
        - `node` (string): Node identifier (required, extracted from net_host_name label)
        - `component` (string, optional): Component/device identifier (e.g., "mlx5_0", "mlx5_1")
        - `cluster` (string, optional): Cluster identifier (e.g., "cluster-e5f6g7h8", extracted from s_cluster label)

      Example:
      ```json
      [
        {
          "timestamp": 1753898040,
          "value": "1",
          "metric": "node_infiniband_link_downed_total",
          "labels": {
            "net_host_name": "host-i9j0k1l2",
            "s_cluster": "cluster-e5f6g7h8",
            "device": "mlx5_0",
            "instance": "host-i9j0k1l2:9100"
          },
          "node": "host-i9j0k1l2",
          "component": "mlx5_0",
          "cluster": "cluster-e5f6g7h8"
        }
      ]
      ```

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
          no node. Node may be parsed from body and/or dropped if none given for some sources.
      - Any other key in an entry is rejected.

      Unlike every other format, entries here are only time-filtered by the observation context's
      `time_interval`. That interval is the producer's declared intent: a null bound means unbounded on
      that side, so an interval of two nulls disables time filtering for the observation entirely.

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
        - The body node is parsed from the rank-like or node-name text prefix. An entry that still names no node,
          or names an invalid one, is recorded against every job resource instead of dropped. An explicit
          `"AllJobResources"` skips the prefix parse altogether.
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
    RAW_LOG_LINES = "raw_log"
    RAW_LOKI_STREAMS = "raw_loki_streams"
    LOKI_DICT_LIST = "loki_dict_list"
    CSV_METRIC_RESPONSE = "csv_metric_response"
    UFMLOG = "ufmlog"
    PROMETHEUS_METRICS = "prometheus_metrics"
    CSP_EVENT_LIST = "csp_event_list"
    ENTRY_TIMESERIES = "entry_timeseries"


class TimeInterval(NamedTuple):
    """The window a producer declares an observation covers; a null bound is unbounded on that side."""

    start: Optional[datetime]
    end: Optional[datetime]

    def __hash__(self) -> int:
        return hash(
            (
                self.start.isoformat() if self.start is not None else None,
                self.end.isoformat() if self.end is not None else None,
            )
        )


TimestampNsFieldStr: TypeAlias = str
TimestampNsField: TypeAlias = int
TimestampMsField: TypeAlias = int

RawLogLine: TypeAlias = str

NodeStr: TypeAlias = str


class LokiAppEntry(TypedDict):
    timestamp: TimestampNsFieldStr
    body: RawLogLine
    line: NotRequired[str]
    node: NotRequired[NodeStr]
    replica: NotRequired[str]


LokiAppEntrySeries: TypeAlias = list[LokiAppEntry]


class LokiSyslogEntry(TypedDict):
    timestamp: TimestampNsField
    node: NodeStr
    body: RawLogLine


LokiSyslogEntrySeries: TypeAlias = list[LokiSyslogEntry]


class LokiHealthCheckEntry(TypedDict):
    timestamp: TimestampMsField
    node: NodeStr
    body: RawLogLine


LokiHealthCheckEntrySeries: TypeAlias = list[LokiHealthCheckEntry]

MetricEntry: TypeAlias = str

MetricEntrySeries: TypeAlias = list[MetricEntry]

# Prometheus metric entry type
PrometheusMetricEntry: TypeAlias = dict[str, Any]
PrometheusMetricEntrySeries: TypeAlias = list[PrometheusMetricEntry]


class CspEventSourceEvent(TypedDict):
    source: str
    event_id: str
    event_created_at: str
    event_region: str
    hostname: str
    fault_id: str
    fault_component: NotRequired[str]


class CspEventEntry(TypedDict):
    hostname: NodeStr
    termination_timestamp_s: float
    category: str
    reason: str
    status: str
    version: int
    source_event: NotRequired[CspEventSourceEvent]


CspEventEntrySeries: TypeAlias = list[CspEventEntry]

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
