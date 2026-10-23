# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    cast,
)
from zoneinfo import ZoneInfo

from attribution_lib.exceptions import (
    UnidentifiableNodeError,
    UnknownObservationFormatError,
)
from attribution_lib.job_info import JobInfo
from attribution_lib.observation import Context, Observation
from attribution_lib.observation.inbound import (
    Format,
    NodeStr,
    RawLogLine,
    TimeseriesEntry,
)
from attribution_lib.resources import get_single_node
from attribution_lib.resources.base import Base as ResourceBase
from attribution_lib.resources.node import Node as NodeResource
from attribution_lib.resources.resource_set import ResourceSet, ResourceSetClass, validate_resource_set
from attribution_lib.sources import Source
from attribution_lib.utils import method_cache


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


@dataclass(frozen=True)
class BodyFallbacks:
    """What an `entry_timeseries` entry may parse out of its body when the producer left it out.

    `parse_node` and `require_node` are orthogonal. When `resources` declares nothing:

    - neither: the entry matches every job resource (ANY)
    - `require_node` only: the entry is dropped
    - `parse_node` only: the node is read from the body if possible, else ANY
    - both: the node is read from the body, else the entry is dropped

    `parse_timestamp` and `require_timestamp` pair the same way. When `ts` is null:

    - neither: the entry is kept, untimed
    - `require_timestamp` only: the entry is dropped
    - `parse_timestamp` only: the timestamp is read from the body if possible, else untimed
    - both: the timestamp is read from the body, else the entry is dropped

    `year_fallback` only matters for year-less formats (`[E1231 23:57:09 ...]`): its year is used,
    and a parsed date after it rolls back a year. None (job end unknown) means current year.
    """

    parse_node: bool = False
    parse_timestamp: bool = False
    year_fallback: Optional[datetime] = None
    require_node: bool = False
    require_timestamp: bool = False


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
    def from_observation(
        observation: Observation[Any], *, body_fallbacks: BodyFallbacks = BodyFallbacks()
    ) -> "EntrySourceBase[Any, EntryBase]":
        """Factory that dispatches to the correct concrete subclass by format.

        Args:
            observation: The observation to convert to an entry source.
            body_fallbacks: What `entry_timeseries` may parse out of the body; other formats ignore it.
        """
        match observation.context.format:
            case Format.STRING_LIST:
                return JobKeywordEntrySource(observation)
            case Format.ENTRY_TIMESERIES:
                return TimeseriesEntrySource(observation, body_fallbacks)
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


def render_metric_body(metric: dict[str, Any]) -> str:
    """The `entry_timeseries` `body` for one Prometheus sample, as the metric rules match it.

    Args:
        metric: The sample. Requires `timestamp`, `metric` and `node`; reads `value`, `threshold`,
            `component` (else `labels["device"]`) and `cluster` (else `labels["s_cluster"]`).

    Returns:
        The rendered metric body.

    Raises:
        ValueError: If the metric is not a dictionary, or if the value is not a string.
    """
    raw_value = metric.get("value", 0)
    metric_value: int | float
    if isinstance(raw_value, str):
        try:
            metric_value = int(raw_value)
        except ValueError:
            metric_value = float(raw_value)
    else:
        metric_value = raw_value

    labels = metric.get("labels", {})
    # Extract component information - preserve empty strings, only None if truly missing
    component = metric.get("component")
    if component is None:
        component = labels.get("device")
    # Extract cluster information - preserve empty strings, only None if truly missing
    cluster = metric.get("cluster")
    if cluster is None:
        cluster = labels.get("s_cluster")

    return (
        f"time: {metric['timestamp']}, metric_name: {metric['metric']}, metric_value: {metric_value}, "
        f"threshold: {metric.get('threshold')}, cluster: {cluster}, node: {metric['node']}, component: {component}"
    )


class TimeseriesEntryObj(EntryBase):
    """Entry for the `entry_timeseries` format.

    The producer declares the timestamp and the resources. The body is parsed only for what the
    producer left out, and only where `BodyFallbacks` allows it: a missing `ts` falls back to body
    timestamp parsing, missing `resources` to body node parsing.
    """

    _context: Context
    _pos: int
    _ts: Optional[int]
    _body: RawLogLine
    _resources: Optional[ResourceSet]
    _body_fallbacks: BodyFallbacks

    def __init__(
        self,
        context: Context,
        pos: int,
        entry: TimeseriesEntry,
        body_fallbacks: BodyFallbacks = BodyFallbacks(),
    ):
        self._context = context
        self._pos = pos
        self._ts = entry.get("ts")  # nanoseconds
        self._body = entry["body"]
        declared = entry.get("resources")
        self._resources = validate_resource_set(declared) if declared is not None else None
        self._body_fallbacks = body_fallbacks

    def attribution_source_base(self, job_info: Optional[JobInfo]) -> Source:
        """Retain a verified producer-declared node separately from body fallbacks."""
        source = super().attribution_source_base(job_info)
        if (
            job_info is not None
            and self._resources is not None
            and self._resources != ResourceSetClass.ANY
            and any(isinstance(resource, NodeResource) for resource in self._resources)
        ):
            return source._replace(declared_node=get_single_node(source.resources).hostname)
        return source

    @method_cache
    def time(self) -> Optional[datetime]:
        if self._ts is None:
            if self._body_fallbacks.parse_timestamp:
                return _extract_timestamp_from_log_line(
                    self._body, timezone=self._context.timezone, job_end_time=self._body_fallbacks.year_fallback
                )
            return None
        # `ts` represents "entry occurred no later than this time". A datetime holds microseconds
        # (see https://docs.python.org/3/library/datetime.html), so leftover nanoseconds round up.
        seconds, micro = divmod(-(-self._ts // 1_000), 10**6)
        return datetime.fromtimestamp(seconds, self._context.timezone) + timedelta(microseconds=micro)

    @method_cache
    def _verified_resources(self, job_info: JobInfo) -> ResourceSet:
        """The declared resources, with the declared node checked against the job.

        ANY when `AllJobResources` was declared, or when nothing was declared, nothing could be parsed
        with `parse_node`, and `require_node` is off. Replicas ride along unread.

        Raises UnidentifiableNodeError when `require_node` finds no node, or when anything other than
        exactly one node was declared; UnknownNodeError when that node is not in the job.
        """
        if self._resources == ResourceSetClass.ANY:
            return ResourceSetClass.ANY

        verified: list[ResourceBase] = []
        node_count = 0
        for resource in self._resources or ():
            if isinstance(resource, NodeResource):
                node_count += 1
                verified.append(NodeResource(hostname=job_info.verify_node(resource.hostname)))
            else:
                verified.append(resource)

        if node_count == 0 and self._body_fallbacks.parse_node:
            try:
                verified.append(NodeResource(hostname=job_info.extract_node(self._body, self._context.source)))
                node_count = 1
            except UnidentifiableNodeError:
                # An unparseable body is the same as no declaration; `require_node` decides below.
                pass

        if len(verified) == 0:
            if self._body_fallbacks.require_node:
                raise UnidentifiableNodeError(self.body())
            return ResourceSetClass.ANY

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

    def __init__(self, observation: Observation[Any], body_fallbacks: BodyFallbacks = BodyFallbacks()) -> None:
        super().__init__(observation)
        self._body_fallbacks = body_fallbacks

    @classmethod
    def make_line_iter(cls, lines: Any) -> Iterator[TimeseriesEntry]:
        return iter(lines)

    def make_line_base(self, pos: int, line: TimeseriesEntry) -> TimeseriesEntryObj:
        return TimeseriesEntryObj(context=self.context, pos=pos, entry=line, body_fallbacks=self._body_fallbacks)
