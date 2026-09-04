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
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator
from datetime import datetime, timedelta
from typing import (
    Any,
    Generic,
    Optional,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
)
from zoneinfo import ZoneInfo

from pydantic_core import PydanticCustomError

from attribution_lib.exceptions import (
    InvalidTimestampSourceError,
    MismatchedSourceAndFormatError,
    UnhandledObservationFormatError,
    UnidentifiableNodeError,
    UnknownObservationFormatError,
)
from attribution_lib.job_info import JobInfo
from attribution_lib.observation import Context, Observation
from attribution_lib.observation.inbound import (
    CspEventEntry,
    CspEventSourceEvent,
    Format,
    LokiAppEntry,
    LokiHealthCheckEntry,
    LokiSyslogEntry,
    NodeStr,
    RawLogLine,
    Source as ObservationSource,
    TimeseriesEntry,
)
from attribution_lib.resources import get_single_node
from attribution_lib.resources.base import Base as ResourceBase
from attribution_lib.resources.node import Node as NodeResource
from attribution_lib.resources.replica import Replica as ReplicaResource
from attribution_lib.resources.resource_set import ResourceSet, ResourceSetClass, validate_resource_set
from attribution_lib.sources import Source
from attribution_lib.ufmlog_common import extract_ufmlog_line_node
from attribution_lib.utils import get_hostname, method_cache


class EntryBase(ABC):
    """Abstract base for all entry types (log lines, metrics, etc.)."""

    _context: Context
    _pos: int

    def timezone(self) -> ZoneInfo:
        return self._context.timezone

    def context(self) -> Context:
        return self._context

    @abstractmethod
    def time(self) -> Optional[datetime]: ...

    @abstractmethod
    def body(self) -> str: ...

    def attribution_source_base(self, job_info: Optional[JobInfo]) -> Source:
        return Source(
            source_context=self._context,
            line_num=self._pos,
            datum=self.body(),
            time=self.time(),
            resources=self.resources(job_info),
            rule=None,
        )

    def resources(self, job_info: Optional[JobInfo] = None) -> ResourceSet:
        return ResourceSetClass.ANY


# Backward-compatible alias so existing imports of EntryProtocol keep working.
EntryProtocol = EntryBase

E = TypeVar("E")
T_co = TypeVar("T_co", bound=EntryBase, covariant=True)


class EntrySourceBase(ABC, Generic[E, T_co]):
    """Abstract base for observation sources that yield typed entries.

    Type parameters:
        E: Element type that the internal iterator yields and that
           ``make_line_base`` receives.
        T_co: Entry type produced by ``make_line_base`` and yielded by
              iteration (covariant, bound to ``EntryBase``).
    """

    context: Context
    lines: Iterator[E]
    count: int

    @staticmethod
    def from_observation(observation: Observation[Any]) -> "EntrySourceBase[Any, EntryBase]":
        """Factory that dispatches to the correct concrete subclass by format."""
        match observation.context.format:
            case Format.STRING_LIST:
                return JobKeywordEntrySource(observation)
            case Format.RAW_LOG_LINES:
                return RawLineEntrySource(observation)
            case Format.RAW_LOKI_STREAMS:
                return RawLokiStreamsEntrySource(observation)
            case Format.LOKI_DICT_LIST:
                return LokiDictEntrySource(observation)
            case Format.CSV_METRIC_RESPONSE:
                raise UnhandledObservationFormatError(observation.context.format)
            case Format.UFMLOG:
                return UFMEntrySource(observation)
            case Format.PROMETHEUS_METRICS:
                return PrometheusMetricsEntrySource(observation)
            case Format.CSP_EVENT_LIST:
                return CspEventEntrySource(observation)
            case Format.ENTRY_TIMESERIES:
                return TimeseriesEntrySource(observation)
            case _:
                raise UnknownObservationFormatError(observation.context.format)

    def __init__(self, observation: Observation[Any]) -> None:
        self.context = observation.context
        self.lines = cast(Iterator[E], self.make_line_iter(observation.lines))
        self.count = 1

    @classmethod
    def make_line_iter(cls, lines: Any) -> Iterator[Any]:
        return iter(lines)

    def __iter__(self) -> Iterator[T_co]:
        return self

    def __next__(self) -> T_co:
        old_count = self.count
        line = next(self.lines)
        self.count += 1
        return self.make_line_base(old_count, line)

    def timezone(self) -> ZoneInfo:
        return self.context.timezone

    @abstractmethod
    def make_line_base(self, pos: int, line: E) -> T_co: ...


def _extract_timestamp_from_pytorch_format(
    line: str, timezone: Optional[ZoneInfo] = None, job_end_time: Optional[datetime] = None
) -> Optional[datetime]:
    # Pattern 1: PyTorch error log format [EMMDD HH:MM:SS.nnnnnnnnn ...]
    # Example: [E1214 06:59:25.800744155 ProcessGroupNCCL.cpp:1923] (Dec 14)
    # Example: [E106 05:19:32.722601449 ProcessGroupNCCL.cpp:697] (Jan 6)
    pytorch_pattern = r"\[E(\d{1,2})(\d{2})\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?\s+"
    match = re.search(pytorch_pattern, line)
    if match:
        month = int(match.group(1))
        day = int(match.group(2))
        hour = int(match.group(3))
        minute = int(match.group(4))
        second = int(match.group(5))

        # Determine the correct year:
        # - If job_end_time is provided (from job's time interval), use its year
        # - If no job_end_time, fall back to current year (legacy behavior)
        if job_end_time is not None:
            dt = datetime(job_end_time.year, month, day, hour, minute, second, tzinfo=timezone)
            if dt > job_end_time:
                # If the job finished in the following year but the log entry is from the prior year, adjust to the previous year
                dt = dt.replace(year=dt.year - 1)
        else:
            dt = datetime(datetime.now().year, month, day, hour, minute, second, tzinfo=timezone)

        return dt
    return None


def _extract_timestamp_from_iso_format(line: str, timezone: Optional[ZoneInfo] = None) -> Optional[datetime]:
    # Pattern 2: Standard ISO-like format [YYYY-MM-DD HH:MM:SS]
    # Example: [2025-12-14 01:07:40]
    iso_pattern = r"\[(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\]"
    match = re.search(iso_pattern, line)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        hour = int(match.group(4))
        minute = int(match.group(5))
        second = int(match.group(6))
        dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone)
        return dt
    return None


def _extract_timestamp_from_log_line(
    line: str, timezone: Optional[ZoneInfo] = None, job_end_time: Optional[datetime] = None
) -> Optional[datetime]:
    """
    Extract timestamp from various log line formats.
    Supports:
    - PyTorch format: [E1214 06:59:25.800744155 ...] or [EMMDD HH:MM:SS...]
    - Standard ISO-like format: [2025-12-14 01:07:40]
    - NCCL format: [2025-12-14 01:07:40] (with brackets)

    Args:
        line: The log line to parse
        timezone: Optional timezone for the timestamp
        job_end_time: Optional job end time (e.g., job end time) to infer year for formats
                       that don't include year (like PyTorch format)

    Returns None gracefully if no timestamp pattern matches.
    """
    try:
        if timestamp := _extract_timestamp_from_pytorch_format(line, timezone, job_end_time):
            return timestamp

        if timestamp := _extract_timestamp_from_iso_format(line, timezone):
            return timestamp
    except (ValueError, IndexError):
        # Gracefully handle any parsing errors
        pass

    return None


class RawLineEntry(EntryBase):
    _context: Context
    _line: RawLogLine

    def __init__(self, context: Context, pos: int, line: RawLogLine):
        self._context = context
        self._pos = pos
        self._line = line
        self._timestamp: Optional[datetime] = None

    def time(self) -> Optional[datetime]:
        """Extract timestamp from log line. Returns None if no timestamp found."""
        if not self._timestamp:
            # Use the job's time_interval.end as reference for inferring year in formats
            # that don't include year (like PyTorch format [E1231 23:57:09...])
            job_end_time = None
            if self._context and hasattr(self._context, "time_interval") and self._context.time_interval:
                job_end_time = self._context.time_interval.end
            self._timestamp = _extract_timestamp_from_log_line(
                self._line,
                timezone=self._context.timezone if self._context else None,
                job_end_time=job_end_time,
            )
        return self._timestamp

    @method_cache
    def node(self, job_info: JobInfo) -> NodeStr:
        return job_info.extract_node(self.body(), self._context.source)

    def resources(self, job_info: Optional[JobInfo] = None) -> ResourceSet:
        if job_info is None:
            return ResourceSetClass.ANY
        else:
            return ResourceSetClass([NodeResource(hostname=self.node(job_info))])

    def body(self) -> str:
        return self._line


class RawLineEntrySource(EntrySourceBase[RawLogLine, RawLineEntry]):
    def make_line_base(self, pos: int, line: RawLogLine) -> RawLineEntry:
        return RawLineEntry(self.context, pos, line)


class LokiDictEntry(EntryBase):
    _context: Context
    _pos: int
    _timestamp_s: float
    _time: Optional[datetime]
    _node: Optional[NodeStr]
    _body: RawLogLine
    _replica: Optional[str]

    def __init__(
        self,
        *,
        context: Context,
        pos: int,
        timestamp_s: float,
        body: RawLogLine,
        node: Optional[NodeStr] = None,
        replica: Optional[str] = None,
    ):
        self._context = context
        self._pos = pos
        self._timestamp_s = timestamp_s
        self._node = node
        self._replica = replica
        self._body = body

    def seconds_epoch(self):
        return self._timestamp_s

    @method_cache
    def time(self) -> datetime:
        return datetime.fromtimestamp(self._timestamp_s, self._context.timezone)

    @method_cache
    def node(self, job_info: JobInfo) -> NodeStr:
        node: Optional[str]
        if (node := self._node) is not None:
            if node[0].isdigit():
                # node is in the form of ip address
                # it needs to be remapped to node namae
                node = get_hostname(node)
            return job_info.verify_node(node)
        else:
            return job_info.extract_node(self.body(), self._context.source)

    def replica(self, job_info: JobInfo) -> Optional[str]:
        return self._replica

    def body(self) -> str:
        return self._body

    def resources(self, job_info: Optional[JobInfo] = None) -> ResourceSet:
        if job_info is None:
            return ResourceSetClass.ANY
        else:
            resources: list[NodeResource | ReplicaResource] = [NodeResource(hostname=self.node(job_info))]
            if (rep := self.replica(job_info)) is not None:
                resources.append(ReplicaResource(name=rep))
            return ResourceSetClass(resources)

    @classmethod
    def get_factory(cls, source: ObservationSource, format: Format) -> "LokiDictFactoryType":
        match source:
            case ObservationSource.APPLOG:
                match format:
                    case Format.LOKI_DICT_LIST:
                        return LokiDictEntry.from_LokiDictAppEntry
                    case Format.RAW_LOKI_STREAMS:
                        return LokiDictEntry.from_LokiRawAppEntry
                    case _:
                        raise MismatchedSourceAndFormatError(source, format)
            case ObservationSource.SYSLOG:
                return LokiDictEntry.from_LokiSyslogEntry
            case ObservationSource.HEALTH_CHECK:
                match format:
                    case Format.LOKI_DICT_LIST:
                        return LokiDictEntry.from_LokiDictHealthCheckEntry
                    case Format.RAW_LOKI_STREAMS:
                        return LokiDictEntry.from_LokiRawHealthCheckEntry
                    case _:
                        raise MismatchedSourceAndFormatError(source, format)
            case _:
                raise InvalidTimestampSourceError(source)

    @classmethod
    def from_LokiDictAppEntry(cls, context: Context, pos: int, app_entry: LokiAppEntry) -> "LokiDictEntry":
        return LokiDictEntry(
            context=context,
            pos=pos,
            timestamp_s=float(app_entry["timestamp"]) * 1e-9,
            body=json.loads(app_entry["line"])["body"],
            node=None,
        )

    @classmethod
    def from_LokiRawAppEntry(cls, context: Context, pos: int, app_entry: LokiAppEntry) -> "LokiDictEntry":
        return LokiDictEntry(
            context=context,
            pos=pos,
            timestamp_s=float(app_entry["timestamp"]) * 1e-9,
            body=app_entry["body"],
            node=app_entry.get("node"),
            replica=app_entry.get("replica"),
        )

    @classmethod
    def from_LokiSyslogEntry(cls, context: Context, pos: int, syslog_entry: LokiSyslogEntry) -> "LokiDictEntry":
        return LokiDictEntry(
            context=context,
            pos=pos,
            timestamp_s=syslog_entry["timestamp"] * 1e-9,
            body=syslog_entry["body"],
            node=syslog_entry["node"],
        )

    @classmethod
    def from_LokiDictHealthCheckEntry(
        cls, context: Context, pos: int, hc_entry: LokiHealthCheckEntry
    ) -> "LokiDictEntry":
        return LokiDictEntry(
            context=context,
            pos=pos,
            timestamp_s=hc_entry["timestamp"] * 1e-3,
            body=hc_entry["body"],
            node=hc_entry["node"],
        )

    @classmethod
    def from_LokiRawHealthCheckEntry(
        cls, context: Context, pos: int, hc_entry: LokiHealthCheckEntry
    ) -> "LokiDictEntry":
        return LokiDictEntry(
            context=context,
            pos=pos,
            timestamp_s=hc_entry["timestamp"] * 1e-9,
            body=hc_entry["body"],
            node=hc_entry["node"],
        )


LokiDictFactoryType: TypeAlias = Callable[[Context, int, Any], LokiDictEntry]


class LokiDictEntrySource(EntrySourceBase[Any, LokiDictEntry]):
    factory_func: LokiDictFactoryType

    def __init__(self, observation: Observation[Any]) -> None:
        super().__init__(observation)
        self.factory_func = LokiDictEntry.get_factory(observation.context.source, observation.context.format)

    def make_line_base(self, pos: int, line: Any) -> LokiDictEntry:
        return self.factory_func(self.context, pos, line)


class RawLokiDict(TypedDict):
    timestamp: int
    body: str
    node: Optional[str]
    replica: Optional[str]


def extract_body_and_node(parsed: dict, node: Optional[str], raw_payload: str) -> tuple[str, Optional[str]]:
    """Extract body text and hostname from a parsed Loki JSON payload.

    Supports three envelope formats:
    - top-level ``body`` + ``attributes.hostname``
    - top-level ``message`` + ``hostname`` stream label
    - CloudEvents (FM logs): ``data.message`` + ``data.hostname``

    Falls back to *raw_payload* when none of the above yield a non-empty body,
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

    # CloudEvents envelope (e.g. FM logs via csp-telemetry)
    if body == "":
        data = parsed.get("data")
        if isinstance(data, dict):
            body = data.get("message") or ""
            if node is None:
                node = data.get("hostname")

    # Last resort: use raw payload so callers can still match substrings.
    if body == "":
        body = raw_payload

    return body, node


def raw_loki_unpack(timestamp: int, payload: str, node: Optional[str], replica: Optional[str]) -> Optional[RawLokiDict]:
    """Try to unpack raw loki entries. Fall back to empty body, or even skip the object if malformed."""
    body = payload
    try:
        res = json.loads(payload)
        if isinstance(res, dict):
            body, node = extract_body_and_node(res, node, payload)
    except json.JSONDecodeError:
        pass

    return RawLokiDict(
        {
            "node": node,
            "replica": replica,
            "timestamp": timestamp,
            "body": body,
        }
    )


def raw_loki_result_helper(results: Iterable[dict[str, dict[str, str]]]) -> Generator[RawLokiDict, None, None]:
    """This unpackes the contents of results, which is list list dicts. The dicts contain a possible "stream" object, and a "values" object. The "stream" object may contain information about which node/replica this data applies to.
    The "values" object maps to a list which contains tuples. The tuples are of length two and contain a string-quoted integer representing a timestamp (sometimes ms, sometimes ns), and a payload that is also sometimes ambiguous (see `raw_loki_unpack` above)."""
    for result in results:
        node = None
        replica = None
        if "stream" in result:
            # possible node/replica we will use as a base for values, below
            node = result["stream"].get("node_name", node)
            replica = result["stream"].get("pod", replica)
        for value in result["values"]:
            # If there aren't 2-tuples, skip because malformed
            if len(value) != 2:
                continue
            try:
                # try to extract timestamp, or skip this entry
                timestamp = int(value[0])
            except ValueError:
                continue

            if (entry := raw_loki_unpack(timestamp, value[1], node, replica)) is not None:
                yield entry


class RawLokiStreamsEntrySource(LokiDictEntrySource):
    @classmethod
    def make_line_iter(cls, lines: Any) -> Iterator[RawLokiDict]:
        return raw_loki_result_helper(iter(lines))


class JobKeywordEntry(EntryBase):
    _context: Context
    _item: str

    def __init__(self, context: Context, pos: int, item: str):
        self._context = context
        self._pos = pos
        self._item = item

    def time(self) -> Optional[datetime]:
        return self._context.time_interval.end

    def node(self, _job_info: JobInfo) -> NodeStr:
        raise NotImplementedError()

    def body(self) -> str:
        return self._item

    def resources(self, job_info: Optional[JobInfo] = None) -> ResourceSet:
        return ResourceSetClass.ANY


class JobKeywordEntrySource(EntrySourceBase[str, JobKeywordEntry]):
    def make_line_base(self, pos: int, line: str) -> JobKeywordEntry:
        return JobKeywordEntry(self.context, pos, line)


class UFMEntry(EntryBase):
    _context: Context
    _pos: int
    _timestamp_s: float
    _body: str
    _node: Optional[str]

    def __init__(self, context: Context, pos: int, timestamp_s: float, body: str, node: Optional[str] = None):
        self._context = context
        self._pos = pos
        self._timestamp_s = timestamp_s
        self._body = body
        self._node = node

    def time(self) -> datetime:
        return datetime.fromtimestamp(self._timestamp_s, self._context.timezone)

    @method_cache
    def node(self, job_info: JobInfo) -> NodeStr:
        if self._node is not None:
            return job_info.verify_node(self._node)
        else:
            return job_info.extract_node(self.body(), self._context.source)

    def replica(self, job_info: JobInfo) -> Optional[str]:
        return None

    def body(self) -> str:
        return self._body

    def resources(self, job_info: Optional[JobInfo] = None) -> ResourceSet:
        if job_info is None:
            return ResourceSetClass.ANY
        else:
            return ResourceSetClass([NodeResource(hostname=self.node(job_info))])

    def attribution_source_base(self, job_info: Optional[JobInfo] = None) -> Source:
        return super().attribution_source_base(job_info).derived(rule="ufmlog-event")


class UFMEntrySource(EntrySourceBase[dict[str, Any], UFMEntry]):
    @classmethod
    def make_line_iter(cls, lines: Any) -> Iterator[dict[str, Any]]:
        return iter(lines)

    def make_line_base(self, pos: int, line: dict[str, Any]) -> UFMEntry:
        timestamp_ns = line.get("timestamp", 0)
        timestamp_s = timestamp_ns * 1e-9
        body = line.get("body", "")
        node = extract_ufmlog_line_node(body)

        return UFMEntry(context=self.context, pos=pos, timestamp_s=timestamp_s, body=body, node=node)


class MetricEntry(EntryBase):
    """Represents a single metric data point"""

    _context: Context
    _pos: int
    _timestamp_s: float
    _metric_name: str
    _metric_value: int | float
    _labels: dict[str, str]
    _node: Optional[str]  # Node field
    _component: Optional[str]  # Component/device information (e.g., "mlx5_0", "mlx5_1")
    _cluster: Optional[str]  # Cluster information (e.g., "cs-001")
    _threshold: Optional[int | float]  # Threshold value for attribution

    def __init__(
        self,
        context: Context,
        pos: int,
        timestamp_s: float,
        metric_name: str,
        metric_value: int | float,
        labels: dict[str, str],
        node: Optional[str] = None,  # Node field - optional
        component: Optional[str] = None,  # Component/device field - optional
        cluster: Optional[str] = None,  # Cluster field - optional
        threshold: Optional[int | float] = None,  # Threshold field - optional
    ):
        self._context = context
        self._pos = pos
        self._timestamp_s = timestamp_s
        self._metric_name = metric_name
        self._metric_value = metric_value
        self._labels = labels
        self._node = node
        self._component = component  # Store component information
        self._cluster = cluster  # Store cluster information
        self._threshold = threshold  # Store threshold information

    def time(self) -> datetime:
        return datetime.fromtimestamp(self._timestamp_s, self._context.timezone)

    @method_cache
    def node(self, job_info: JobInfo) -> NodeStr:
        if self._node is not None:
            return job_info.verify_node(self._node)
        else:
            # For metrics, node information is essential and cannot be extracted from body
            # Unlike log entries, metric bodies only contain "metric_name=value"
            raise UnidentifiableNodeError(
                f"Node information is required for metric '{self._metric_name}' but was not provided"
            )

    def body(self) -> str:
        # Return a string representation of the metric for compatibility
        base_str = f"time: {self._timestamp_s}, metric_name: {self._metric_name}, metric_value: {self._metric_value}, threshold: {self._threshold}, cluster: {self._cluster}, node: {self._node}, component: {self._component}"

        return base_str

    def resources(self, job_info: Optional[JobInfo] = None) -> ResourceSet:
        if job_info is None:
            return ResourceSetClass.ANY
        else:
            return ResourceSetClass([NodeResource(hostname=self.node(job_info))])

    def attribution_source_base(self, job_info: Optional[JobInfo] = None) -> Source:
        return super().attribution_source_base(job_info).derived(rule="net-metrics-event")

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def metric_value(self) -> int | float:
        return self._metric_value

    @property
    def labels(self) -> dict[str, str]:
        return self._labels

    @property
    def component(self) -> Optional[str]:
        """Component/device information (e.g., 'mlx5_0', 'mlx5_1') - can be None if not available"""
        return self._component

    @property
    def cluster(self) -> Optional[str]:
        """Cluster information (e.g., 'cs-001') - can be None if not available"""
        return self._cluster

    @property
    def timestamp_s(self) -> float:
        return self._timestamp_s

    @property
    def threshold(self) -> Optional[int | float]:
        """Threshold value for attribution - can be None if not set"""
        return self._threshold


_CSP_SOURCE_EVENT_REQUIRED_FIELDS = frozenset(
    {"source", "event_id", "event_created_at", "event_region", "hostname", "fault_id"}
)
_CSP_SOURCE_EVENT_ALLOWED_FIELDS = _CSP_SOURCE_EVENT_REQUIRED_FIELDS | {"fault_component"}


def _serialize_csp_source_event(source_event: CspEventSourceEvent) -> str:
    """Serialize the compact, normalized provenance attached to a CSP event.

    Args:
        source_event: Provider event fields retained for attribution provenance.

    Returns:
        Deterministic strict JSON suitable for ``Source.datum``.

    Raises:
        ValueError: If the event is not an object, or fields are missing, unexpected, blank, or non-string.
    """
    if not isinstance(source_event, dict):
        raise ValueError("Invalid CSP source_event: expected an object")
    fields = set(source_event)
    if not _CSP_SOURCE_EVENT_REQUIRED_FIELDS <= fields or not fields <= _CSP_SOURCE_EVENT_ALLOWED_FIELDS:
        raise ValueError("Invalid CSP source_event: unexpected or missing fields")
    if any(not isinstance(value, str) or not value.strip() for value in source_event.values()):
        raise ValueError("Invalid CSP source_event: values must be non-empty strings")
    return json.dumps(source_event, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CspEventEntryObj(EntryBase):
    """Entry for a CSP event event."""

    def __init__(self, context: Context, pos: int, entry: CspEventEntry):
        self._context = context
        self._pos = pos
        self._entry = entry
        self._hostname = entry["hostname"]
        self._termination_timestamp_s = entry["termination_timestamp_s"]
        self._category = entry.get("category", "")
        self._reason = entry.get("reason", "")
        self._status = entry.get("status", "")

    def time(self) -> Optional[datetime]:
        return datetime.fromtimestamp(self._termination_timestamp_s, self._context.timezone)

    def body(self) -> str:
        body = f"csp_event: category={self._category}, reason={self._reason}, status={self._status}"
        if "source_event" not in self._entry:
            return body
        return f"{body}, source_event={_serialize_csp_source_event(self._entry['source_event'])}"

    def node(self, job_info: Optional[JobInfo] = None) -> Optional[str]:
        return self._hostname

    def resources(self, job_info: Optional[JobInfo] = None) -> ResourceSet:
        # hostname is guaranteed valid: callers must filter out entries with unknown hostnames
        # before passing them to the attribution pipeline.
        return ResourceSetClass([NodeResource(hostname=self._hostname)])


class CspEventEntrySource(EntrySourceBase[CspEventEntry, CspEventEntryObj]):
    """Process CSP event structured events."""

    @classmethod
    def make_line_iter(cls, lines: Any) -> Iterator[CspEventEntry]:
        return iter(lines)

    def make_line_base(self, pos: int, line: CspEventEntry) -> CspEventEntryObj:
        return CspEventEntryObj(context=self.context, pos=pos, entry=line)


class PrometheusMetricsEntrySource(EntrySourceBase[dict[str, Any], MetricEntry]):
    """Process Prometheus metrics data - metrics-specific implementation

    Note: This class inherits from EntrySourceBase to maintain compatibility with
    the NVIDIA Attribution pipeline, but processes metrics data differently than
    log-based sources. The 'line' concept is adapted for time-series data points.
    """

    @classmethod
    def make_line_iter(cls, lines: Any) -> Iterator[dict[str, Any]]:
        return iter(lines)

    def make_line_base(self, pos: int, line: dict[str, Any]) -> MetricEntry:
        """Convert Prometheus metric data to MetricEntry."""
        if not isinstance(line, dict):
            raise PydanticCustomError(
                "dict_type",
                "Expected dictionary for metric data, got {input_type}",
                {"input_type": type(line).__name__},
            )

        try:
            timestamp_s = line["timestamp"]
            metric_name = line["metric"]
            node = line["node"]
        except KeyError as e:
            raise PydanticCustomError(
                "missing",
                "Field required: {field}",
                {"field": e.args[0]},
            ) from e

        # Handle both integer and float metric values appropriately
        raw_value = line.get("value", 0)
        metric_value: int | float
        if isinstance(raw_value, str):
            try:
                metric_value = int(raw_value)
            except ValueError:
                metric_value = float(raw_value)
        else:
            metric_value = raw_value

        labels = line.get("labels", {})

        # Extract component information - preserve empty strings, only None if truly missing
        component = line.get("component")
        if component is None:
            component = labels.get("device")

        # Extract cluster information - preserve empty strings, only None if truly missing
        cluster = line.get("cluster")
        if cluster is None:
            cluster = labels.get("s_cluster")

        return MetricEntry(
            context=self.context,
            pos=pos,
            timestamp_s=timestamp_s,
            metric_name=metric_name,
            metric_value=metric_value,
            labels=labels,
            node=node,
            component=component,  # Pass component to MetricEntry
            cluster=cluster,  # Pass cluster to MetricEntry
        )


class TimeseriesEntryObj(EntryBase):
    """Entry for the `entry_timeseries` format.

    Nothing is parsed back out of the body. The producer declares the timestamp and the resources.
    """

    _context: Context
    _pos: int
    _ts: Optional[int]
    _body: RawLogLine
    _resources: Optional[ResourceSet]

    def __init__(self, context: Context, pos: int, entry: TimeseriesEntry):
        self._context = context
        self._pos = pos
        self._ts = entry.get("ts")  # nanoseconds
        self._body = entry["body"]
        declared = entry.get("resources")
        self._resources = validate_resource_set(declared) if declared is not None else None

    def time(self) -> Optional[datetime]:
        if self._ts is None:
            return None
        # `ts` represents "entry occurred no later than this time". A datetime holds microseconds
        # (see https://docs.python.org/3/library/datetime.html), so leftover nanoseconds round up.
        seconds, micro = divmod(-(-self._ts // 1_000), 10**6)
        return datetime.fromtimestamp(seconds, self._context.timezone) + timedelta(microseconds=micro)

    @method_cache
    def _verified_resources(self, job_info: JobInfo) -> ResourceSet:
        """The declared resources, with the declared node checked against the job.

        ANY only when nothing or `AllJobResources` was declared. Replicas ride along unread.

        Raises UnidentifiableNodeError unless exactly one node is declared.
        """
        if self._resources is None or self._resources == ResourceSetClass.ANY:
            return ResourceSetClass.ANY

        verified: list[ResourceBase] = []
        node_count = 0
        for resource in self._resources:
            if isinstance(resource, NodeResource):
                node_count += 1
                verified.append(NodeResource(hostname=job_info.verify_node(resource.hostname)))
            else:
                verified.append(resource)

        # Potentially relaxable later; for now this matches every other format's one node per entry.
        if node_count != 1:
            raise UnidentifiableNodeError(self.body())

        return ResourceSetClass(verified)

    @method_cache
    def node(self, job_info: JobInfo) -> NodeStr:
        """The one job node this entry declares.

        Callers use this as a keep-or-drop gate; attribution reads the full set from resources().
        Raises UnknownResourceError when no job node is identifiable, which
        process_observation_source treats as skip-this-entry.
        """
        verified = self._verified_resources(job_info)
        if verified == ResourceSetClass.ANY:
            raise UnidentifiableNodeError(self.body())
        return get_single_node(verified).hostname

    def body(self) -> str:
        return self._body

    def resources(self, job_info: Optional[JobInfo] = None) -> ResourceSet:
        if job_info is None:
            return ResourceSetClass.ANY
        return self._verified_resources(job_info)


class TimeseriesEntrySource(EntrySourceBase[TimeseriesEntry, TimeseriesEntryObj]):
    """Process `entry_timeseries` entries."""

    @classmethod
    def make_line_iter(cls, lines: Any) -> Iterator[TimeseriesEntry]:
        return iter(lines)

    def make_line_base(self, pos: int, line: TimeseriesEntry) -> TimeseriesEntryObj:
        return TimeseriesEntryObj(context=self.context, pos=pos, entry=line)
