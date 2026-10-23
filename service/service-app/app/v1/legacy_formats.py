# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy `/v1` observation bodies, decoded into `entry_timeseries`.

The library accepts only `entry_timeseries`. These decoders live in the service, not the library,
solely so callers still on the 1.0.x contract can keep posting their legacy bodies to `/v1`
unchanged.
"""

import json
from collections.abc import Callable, Generator, Iterable
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Optional, TypedDict

from attribution_lib.job_info import JobInfo
from attribution_lib.observation.inbound import (
    Source,
    TimeInterval as LibTimeInterval,
)
from attribution_lib.observation.processing import _extract_timestamp_from_log_line, render_metric_body
from attribution_lib.utils import get_hostname
from pydantic import BaseModel, RootModel, ValidationError

from app.exceptions import IncompatibleObservationSourceFormatError, InvalidObservationBodyError


class Format(StrEnum):
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


# Mirrors the library's source/format matrix; a source absent here is rejected by the library itself.
_FORMATS_BY_SOURCE: dict[Source, frozenset[Format]] = {
    Source.JOB_KEYWORDS: frozenset(),
    Source.APPLOG: frozenset({Format.RAW_LOG_LINES, Format.LOKI_DICT_LIST, Format.RAW_LOKI_STREAMS}),
    Source.SYSLOG: frozenset({Format.LOKI_DICT_LIST, Format.RAW_LOKI_STREAMS}),
    Source.HEALTH_CHECK: frozenset({Format.LOKI_DICT_LIST, Format.RAW_LOKI_STREAMS}),
    Source.UFMLOG: frozenset({Format.UFMLOG}),
    Source.NODE_METRICS: frozenset({Format.PROMETHEUS_METRICS}),
    Source.NVLINK_METRICS: frozenset({Format.PROMETHEUS_METRICS}),
    Source.CSP_EVENT: frozenset({Format.CSP_EVENT_LIST}),
}


_EPILOG_TAIL = timedelta(minutes=10)


def _epilog_time_start(job_info: JobInfo) -> datetime:
    time_start = job_info.interest_start_time
    last_tflops = job_info.last_tflops_time
    # Shared logs can contain progress from a later workload. Only narrow this
    # window with progress inside it; otherwise retain the health-check evidence.
    if last_tflops is not None and time_start < last_tflops <= job_info.interest_end_time + _EPILOG_TAIL:
        time_start = last_tflops
    return time_start


_WindowBound = Callable[[JobInfo], Optional[datetime]]

# Legacy bodies are filtered on these per-source windows; their declared `time_interval` is ignored.
_WINDOWS: dict[Source, tuple[_WindowBound, _WindowBound]] = {
    Source.HEALTH_CHECK: (_epilog_time_start, lambda ji: ji.interest_end_time + _EPILOG_TAIL),
    Source.SYSLOG: (lambda ji: ji.interest_start_time, lambda ji: ji.interest_end_time),
    Source.UFMLOG: (lambda ji: ji.job_start_time, lambda ji: ji.job_end_time),
    Source.NODE_METRICS: (lambda ji: ji.job_start_time, lambda ji: ji.job_end_time),
    Source.NVLINK_METRICS: (lambda ji: ji.interest_start_time, lambda ji: ji.interest_end_time),
    Source.CSP_EVENT: (lambda ji: ji.job_start_time, lambda ji: None),
}


def legacy_window(source: Source, job_info: JobInfo) -> LibTimeInterval:
    """The window a legacy body of `source` is filtered on; a null bound is unbounded on that side."""
    if (window := _WINDOWS.get(source)) is None:
        return LibTimeInterval(start=None, end=None)
    start, end = window
    return LibTimeInterval(start=start(job_info), end=end(job_info))


class RawLokiStreamPair(RootModel):
    root: tuple[str, str]


class RawLokiStream(BaseModel):
    stream: dict[str, Any]
    values: list[RawLokiStreamPair]


class RawLokiStreams(RootModel):
    root: list[RawLokiStream]


class RawLokiDict(TypedDict):
    timestamp: int
    body: str
    node: Optional[str]
    replica: Optional[str]


def extract_body_and_node(parsed: dict, node: Optional[str], raw_payload: str) -> tuple[str, Optional[str]]:
    """Extract body text and hostname from a parsed Loki JSON payload.

    Supports three envelope formats:
    - top-level `body` + `attributes.hostname`
    - top-level `message` + `hostname` stream label
    - CloudEvents: `data.message` + `data.hostname`

    Falls back to `raw_payload` when none of the above yield a non-empty body,
    so callers can still match substrings (e.g. GPU GUIDs).

    Args:
        parsed: The already-parsed JSON dict from the Loki value.
        node: Current node value (may be None). Only overwritten when None.
        raw_payload: The original unparsed string, used as last-resort body.

    Returns:
        (body, node) tuple.
    """
    if node is None and (attributes := parsed.get("attributes")) is not None:
        node = attributes.get("hostname")

    body = parsed.get("body") or parsed.get("message") or ""

    if body == "":
        data = parsed.get("data")
        if isinstance(data, dict):
            body = data.get("message") or ""
            if node is None:
                node = data.get("hostname")

    if body == "":
        body = raw_payload

    return body, node


def _raw_loki_unpack(timestamp: int, payload: str, node: Optional[str], replica: Optional[str]) -> RawLokiDict:
    """Unpack one raw Loki value, keeping the payload itself as the body when it is not a JSON object."""
    body = payload
    try:
        res = json.loads(payload)
        if isinstance(res, dict):
            body, node = extract_body_and_node(res, node, payload)
    except json.JSONDecodeError:
        pass

    return RawLokiDict(node=node, replica=replica, timestamp=timestamp, body=body)


def _raw_loki_results(results: Iterable[dict[str, Any]]) -> Generator[RawLokiDict, None, None]:
    """Flatten Loki streams into one entry per value; a stream's `node_name`/`pod` labels seed node/replica."""
    for result in results:
        node = None
        replica = None
        if "stream" in result:
            node = result["stream"].get("node_name", node)
            replica = result["stream"].get("pod", replica)
        for value in result["values"]:
            if len(value) != 2:
                continue
            try:
                timestamp = int(value[0])
            except ValueError:
                continue
            yield _raw_loki_unpack(timestamp, value[1], node, replica)


_CSP_SOURCE_EVENT_REQUIRED_FIELDS = frozenset(
    {"source", "event_id", "event_created_at", "event_region", "hostname", "fault_id"}
)
_CSP_SOURCE_EVENT_ALLOWED_FIELDS = _CSP_SOURCE_EVENT_REQUIRED_FIELDS | {"fault_component"}


def _serialize_csp_source_event(source_event: Any) -> str:
    """Serialize a CSP event's provider provenance as deterministic compact JSON, or raise ValueError."""
    if not isinstance(source_event, dict):
        raise ValueError("Invalid CSP source_event: expected an object")
    fields = set(source_event)
    if not _CSP_SOURCE_EVENT_REQUIRED_FIELDS <= fields or not fields <= _CSP_SOURCE_EVENT_ALLOWED_FIELDS:
        raise ValueError("Invalid CSP source_event: unexpected or missing fields")
    if any(not isinstance(value, str) or not value.strip() for value in source_event.values()):
        raise ValueError("Invalid CSP source_event: values must be non-empty strings")
    return json.dumps(source_event, sort_keys=True, separators=(",", ":"), allow_nan=False)


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _datetime_ns(moment: datetime) -> int:
    """Whole microseconds as ns, so the library's round-up to microseconds returns `moment` exactly."""
    return (moment - _EPOCH) // timedelta(microseconds=1) * 1_000


def _legacy_ns(seconds: float) -> int:
    """The ns `ts` for a legacy float-seconds timestamp, rounded to the microsecond the library used to."""
    return _datetime_ns(datetime.fromtimestamp(seconds, timezone.utc))


def _entry(body: str, ts: Optional[int], resources: Optional[list[dict[str, str]]] = None) -> dict[str, Any]:
    return {"body": body, "ts": ts, "resources": resources}


def _node_resources(node: Optional[str]) -> Optional[list[dict[str, str]]]:
    return [{"kind": "node", "hostname": node}] if node is not None else None


def _loki_resources(node: Optional[str], replica: Optional[str] = None) -> Optional[list[dict[str, str]]]:
    resources: list[dict[str, str]] = []
    if node is not None:
        if node[0].isdigit():  # IP form; the library verifies a declared node against the job as-is
            node = get_hostname(node)
        resources.append({"kind": "node", "hostname": node})
    if replica is not None:
        resources.append({"kind": "replica", "name": replica})
    return resources or None


def _raw_log_entries(lines: list[str], job_info: JobInfo, declared_end: Optional[datetime]) -> list[dict[str, Any]]:
    entries = []
    for line in lines:
        # Year-less timestamps take their year from the declared interval end, as the library's `raw_log` path did.
        moment = _extract_timestamp_from_log_line(line, timezone=job_info.timezone, job_end_time=declared_end)
        entries.append(_entry(line, _datetime_ns(moment) if moment is not None else None))
    return entries


def _loki_dict_entries(source: Source, items: list[Any]) -> list[dict[str, Any]]:
    match source:
        case Source.APPLOG:
            return [_entry(json.loads(i["line"])["body"], _legacy_ns(float(i["timestamp"]) * 1e-9)) for i in items]
        case Source.SYSLOG:
            return [_entry(i["body"], _legacy_ns(i["timestamp"] * 1e-9), _loki_resources(i["node"])) for i in items]
        case _:  # health_check, whose dict timestamps are milliseconds
            return [_entry(i["body"], _legacy_ns(i["timestamp"] * 1e-3), _loki_resources(i["node"])) for i in items]


def _raw_loki_entries(source: Source, streams: list[Any]) -> list[dict[str, Any]]:
    items = _raw_loki_results(streams)
    if source is Source.APPLOG:
        return [
            _entry(i["body"], _legacy_ns(float(i["timestamp"]) * 1e-9), _loki_resources(i["node"], i["replica"]))
            for i in items
        ]
    return [_entry(i["body"], _legacy_ns(i["timestamp"] * 1e-9), _loki_resources(i["node"])) for i in items]


def _ufmlog_entries(items: list[Any]) -> list[dict[str, Any]]:
    # No resources: the node is parsed out of the body, never the fabric manager host that logged it.
    return [_entry(i.get("body", ""), _legacy_ns(i.get("timestamp", 0) * 1e-9)) for i in items]


def _prometheus_entries(items: list[Any]) -> list[dict[str, Any]]:
    entries = []
    for metric in items:
        # Legacy samples never carried a threshold into the rendered body.
        rendered = render_metric_body({key: value for key, value in metric.items() if key != "threshold"})
        entries.append(_entry(rendered, _legacy_ns(metric["timestamp"]), _node_resources(metric["node"])))
    return entries


def _csp_event_entries(items: list[Any]) -> list[dict[str, Any]]:
    entries = []
    for event in items:
        hostname = event["hostname"]
        seconds = event["termination_timestamp_s"]
        body = (
            f"csp_event: category={event.get('category', '')}, reason={event.get('reason', '')}, "
            f"status={event.get('status', '')}"
        )
        if "source_event" in event:
            body = f"{body}, source_event={_serialize_csp_source_event(event['source_event'])}"
        entries.append(_entry(body, _legacy_ns(seconds), _node_resources(hostname)))
    return entries


def _decode_json(format: Format, body: str) -> Any:
    try:
        if format is Format.RAW_LOKI_STREAMS:
            return RawLokiStreams.model_validate_json(body).model_dump()
        return json.loads(body)
    except (json.JSONDecodeError, ValidationError) as e:
        raise InvalidObservationBodyError.new(format.value).to_http_exception() from e


def to_entry_timeseries(
    source: Source, format: Format, body: str, job_info: JobInfo, declared_end: Optional[datetime]
) -> list[dict[str, Any]]:
    """Decode a legacy body into the `entry_timeseries` entries the library receives.

    Args:
        source: The declared observation source.
        format: The legacy format `body` is in; not `string_list`, `entry_timeseries` or `undefined`.
        body: The body as posted.
        job_info: The job the observation belongs to.
        declared_end: The declared `time_interval.end`; year-less `raw_log` timestamps take their year from it.

    Returns:
        The entries; empty for a source the library rejects regardless of format.

    Raises:
        HTTPException: 400 when the body does not decode, or the source does not accept the format.
    """
    if format in (Format.RAW_LOG_LINES, Format.CSV_METRIC_RESPONSE):
        decoded: Any = body.split("\n")
    else:
        decoded = _decode_json(format, body)

    if (accepted := _FORMATS_BY_SOURCE.get(source)) is None:
        return []
    if format not in accepted:
        raise IncompatibleObservationSourceFormatError.new(source=source, format=format).to_http_exception()

    match format:
        case Format.RAW_LOG_LINES:
            return _raw_log_entries(decoded, job_info, declared_end)
        case Format.LOKI_DICT_LIST:
            return _loki_dict_entries(source, decoded)
        case Format.RAW_LOKI_STREAMS:
            return _raw_loki_entries(source, decoded)
        case Format.UFMLOG:
            return _ufmlog_entries(decoded)
        case Format.PROMETHEUS_METRICS:
            return _prometheus_entries(decoded)
        case _:  # csp_event_list, the only accepted format left
            return _csp_event_entries(decoded)
