# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime
from enum import Enum
from typing import Literal, NotRequired, TypedDict


class ExecutionContext(TypedDict):
    """Wire-format shape adapters produce and the orchestrator consumes.

    Required fields are present on every context; NotRequired fields are
    populated when the adapter has them (Slurm pulls most from sacct; the
    static adapter takes them from user-supplied config). The orchestrator
    treats unset NotRequired fields as ""/None at the attributor boundary.
    """

    id: str
    nodes: list[str]
    start_time: datetime
    end_time: datetime
    # Optional IANA zone used to interpret offsetless evidence timestamps.
    # This preserves regional/DST identity that a numeric UTC offset cannot.
    timezone: NotRequired[str]
    # Wire-format scheduler identity sent to the attribution service as
    # `workload.type`.
    workload_type: NotRequired[str]
    # BP-internal scheduler identity (the ECA component that discovered this
    # context: "slurm", "kueue", "static", ...). SQLiteOutput reads this for
    # the `scheduler_type` column independently of the service wire value.
    scheduler: NotRequired[str]
    # Per-attempt key, scheduler-agnostic. Slurm sets it = sluid; Kueue sets
    # it = the Workload's metadata.uid (a new Workload is created per
    # requeue). The sqlite output reads this as the `attempt_id` column and
    # uses it in the (job_id, attempt_id, scheduler_type) unique key, so the
    # value must be deterministic across re-runs of the same logical attempt.
    attempt_id: NotRequired[str]

    # Fields adapters write into the context dict in addition to the spine.
    # Kept here so static analysis (and humans) see the full contract.
    job_id: NotRequired[str]
    state: NotRequired[str]
    name: NotRequired[str]
    user_name: NotRequired[str]
    workload_framework: NotRequired[str]
    std_out: NotRequired[str]
    std_err: NotRequired[str]
    failed_node: NotRequired[str]
    cancelled_by: NotRequired[str]
    time_limit: NotRequired[int]
    gpu_count: NotRequired[int]
    # Per-Pod scheduler placement with source provenance. Kueue populates
    # this from live Pods and durable PlacementRecords when available.
    pod_placements: NotRequired[list[dict]]


# Wire-format enums. Values mirror the service observation API contract.
class Source(str, Enum):
    EMPTY = ""
    SLURM_INFO = "slurm_info"
    JOB_KEYWORDS = "job_keywords"
    APPLOG = "applog"
    SYSLOG = "syslog"
    HEALTH_CHECK = "health_check"
    NODE_METRICS = "node_metrics"
    NVLINK_METRICS = "nvlink_metrics"
    IB_METRICS = "ib_metrics"
    ETH_METRICS = "eth_metrics"
    NETWORK_SYSLOG = "network_syslog"
    UFMLOG = "ufmlog"
    CSP_EVENT = "csp_event"


# Wire-format enums. Values mirror the service observation API contract.
class Format(str, Enum):
    EMPTY = ""
    STRING_LIST = "string_list"
    RAW_LOG = "raw_log"
    RAW_LOKI_STREAMS = "raw_loki_streams"
    LOKI_DICT_LIST = "loki_dict_list"
    CSV_METRIC_RESPONSE = "csv_metric_response"
    UFMLOG = "ufmlog"
    PROMETHEUS_METRICS = "prometheus_metrics"
    CSP_EVENT_LIST = "csp_event_list"
    ENTRY_TIMESERIES = "entry_timeseries"


class TimeInterval(TypedDict):
    # A None bound is unbounded on that side; the service accepts it only for ENTRY_TIMESERIES.
    start: datetime | None
    end: datetime | None


class ObservationContext(TypedDict):
    time_interval: TimeInterval
    source: Source
    format: Format


class NodeResource(TypedDict):
    kind: Literal["node"]
    hostname: str


class ReplicaResource(TypedDict):
    kind: Literal["replica"]
    name: str


ResourceSet = Literal["AllJobResources"] | list[NodeResource | ReplicaResource]


class TimeseriesEntry(TypedDict):
    """One element of an ENTRY_TIMESERIES body. The service rejects any other key."""

    body: str
    # Nanoseconds since the epoch. Omit when the producer has no timestamp for the line.
    ts: NotRequired[int]
    # Omit when the producer has no resource identity for the line.
    resources: NotRequired[ResourceSet]


class Observation(TypedDict):
    context: ObservationContext
    # `body` is the connector's already-serialized payload (JSON for
    # ENTRY_TIMESERIES/LOKI_DICT_LIST/PROMETHEUS_METRICS; raw text for
    # RAW_LOG). The service-side parser branches on `format` to deserialize.
    body: str


def assert_aware(dt: datetime, field: str) -> datetime:
    """Reject naive datetimes at the boundary into the attribution service.

    The service requires AwareDatetime; the upstream attribution_lib also has
    code paths that crash on naive datetimes. Use this anywhere a connector
    pulls a datetime out of an ExecutionContext to put on the wire.
    """
    if dt.tzinfo is None:
        raise ValueError(f"{field}: datetime must be tz-aware, got naive {dt!r}")
    return dt


__all__ = [
    "ExecutionContext",
    "Format",
    "NodeResource",
    "Observation",
    "ObservationContext",
    "ReplicaResource",
    "Source",
    "TimeInterval",
    "TimeseriesEntry",
    "assert_aware",
]
