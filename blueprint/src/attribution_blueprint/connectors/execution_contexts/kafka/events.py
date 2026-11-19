# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode CloudEvents and normalize scheduler data into execution contexts.

Like the Slurm and Kueue mapping helpers, this module knows the scheduler's
payload format but has no consumer, offset, or rebalance state.

data must supply id, nodes, and timezone-aware start_time/end_time. job_id
falls back to id; attempt_id to "0"; scheduler to the configured default;
state to UNKNOWN. Other supported fields retain their ExecutionContext names.
CloudEvent id and source are retained as event_id and event_source. Kubernetes
namespace, pod_names, and pod_nodes remain extensions, as in the Kueue adapter.
failed_nodes keeps the full list and supplies the lexicographically first
failed_node, also matching the Kueue convention. Numeric allocation fields
must be non-negative integers; a time_limit of 0 means no limit and is
dropped. Invalid events raise EventError; callers decide whether to skip them
and how to account for their offsets.

A non-null data `allocations` list replaces the flat placement fields with
one entry per node. The payload, not the CloudEvent type, selects the shape,
so one consumer can accept producers on either side of a migration. nodes,
gpu_count, pod_names, pod_nodes, and a cluster shared by every allocation are
then derived from the allocations, which the context also keeps.
"""

from __future__ import annotations

import errno
import ipaddress
import json
from collections import Counter
from datetime import datetime
from typing import NotRequired, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from attribution_blueprint.orchestrator import _service_timezone_name
from attribution_blueprint.types import ExecutionContext

_OPTIONAL_STR_FIELDS = (
    "name",
    "user_name",
    "workload_framework",
    "cancelled_by",
    "std_out",
    "std_err",
    "namespace",
    "cluster",
)
_REQUIRED_DATA_KEYS = ("id", "start_time", "end_time")
# Placement fields that `allocations` replaces; sending both is ambiguous.
_FLAT_PLACEMENT_FIELDS = ("nodes", "gpu_count", "cluster", "pod_names", "pod_nodes")


class ReplicaPlacement(TypedDict):
    """One replica (for example, a Pod) placed on an allocated node."""

    name: str
    # Normalized address; absent when the producer did not report one.
    ip: NotRequired[str]


class NodeAllocation(TypedDict):
    """The GPUs a job held on one node and the replicas placed there."""

    node: str
    cluster: str
    gpus: int
    replicas: list[ReplicaPlacement]


class KafkaExecutionContext(ExecutionContext):
    """Common execution context plus Kafka correlation and Kubernetes metadata."""

    event_id: str
    event_source: str
    event_subject: NotRequired[str]
    cluster: NotRequired[str]
    failed_nodes: NotRequired[list[str]]
    namespace: NotRequired[str]
    pod_names: NotRequired[list[str]]
    pod_nodes: NotRequired[dict[str, str]]
    # Per-node placement, in event order, when the event uses that shape.
    allocations: NotRequired[list[NodeAllocation]]


class EventError(ValueError):
    """The event payload is invalid and may be skipped without retrying."""


def _read_cloudevent(
    raw: bytes | None,
    headers: list[tuple[str, bytes | str | None]] | None,
) -> tuple[str, str, dict, str | None, str, dict]:
    """Return validated attributes, data, and the complete CloudEvent envelope."""
    if raw is None:
        raise EventError("message has no value")
    try:
        body = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise EventError(f"value is not a supported JSON event: {exc}") from exc

    headers = dict(headers or [])
    if "ce_type" in headers:
        attrs = {key[3:]: _header_str(value) for key, value in headers.items() if key.startswith("ce_")}
        if "content-type" in headers:
            attrs["datacontenttype"] = _header_str(headers["content-type"])
        data = body
    else:
        if not isinstance(body, dict):
            raise EventError("structured CloudEvent must be a JSON object")
        attrs = body
        data = body.get("data")

    if attrs.get("specversion") != "1.0":
        raise EventError(f"unsupported CloudEvents specversion {attrs.get('specversion')!r}")
    event_id = attrs.get("id")
    event_type = attrs.get("type")
    if not isinstance(event_id, str) or event_id == "":
        raise EventError("CloudEvent id is missing")
    if not isinstance(event_type, str) or event_type == "":
        raise EventError(f"CloudEvent {event_id} type is missing")
    if not isinstance(data, dict):
        raise EventError(f"CloudEvent {event_id} data must be a JSON object")
    source = attrs.get("source")
    if not isinstance(source, str) or not source or any(char.isspace() for char in source):
        raise EventError(f"CloudEvent {event_id} source must be a non-empty URI-reference without whitespace")
    # Optional and only echoed on results, so an unusable value is treated as
    # absent rather than rejecting (and committing past) the completion event.
    subject = attrs.get("subject")
    if not isinstance(subject, str) or not subject:
        subject = None
    event = {**attrs, "data": data}
    return event_id, event_type, data, subject, source, event


def _header_str(value: bytes | str | None) -> str:
    """Decode CloudEvent binary headers; absent values normalize to empty text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else value


def _context_from_data(
    data: dict,
    *,
    event_id: str,
    event_source: str,
    default_scheduler: str,
    workload_type: str,
    event_subject: str | None = None,
) -> KafkaExecutionContext:
    """Validate an event `data` object and build an ExecutionContext from it.

    A non-null `allocations` key selects the per-node placement shape;
    otherwise the flat `nodes` shape applies unchanged.
    """
    has_allocations = data.get("allocations") is not None
    required = _REQUIRED_DATA_KEYS if has_allocations else (*_REQUIRED_DATA_KEYS, "nodes")
    missing = [key for key in required if data.get(key) in (None, "", [])]
    if missing:
        raise EventError(f"CloudEvent {event_id} data missing required keys: {missing}")
    allocations = _allocations(data, event_id) if has_allocations else None

    ctx_id = str(data["id"])
    ctx: KafkaExecutionContext = {
        "id": ctx_id,
        "workload_type": workload_type,
        "scheduler": _str_or_default(data.get("scheduler"), default_scheduler),
        "nodes": (
            _str_list(data["nodes"], "nodes", event_id)
            if allocations is None
            else [allocation["node"] for allocation in allocations]
        ),
        "start_time": _aware_datetime(data["start_time"], "start_time", event_id),
        "end_time": _aware_datetime(data["end_time"], "end_time", event_id),
        "job_id": _str_or_default(data.get("job_id"), ctx_id),
        "attempt_id": _str_or_default(data.get("attempt_id"), "0"),
        "state": _str_or_default(data.get("state"), "UNKNOWN"),
        "event_id": event_id,
        "event_source": event_source,
    }
    if ctx["end_time"] < ctx["start_time"]:
        raise EventError(f"CloudEvent {event_id} end_time is before start_time")
    if event_subject is not None:
        ctx["event_subject"] = event_subject

    _validate_timezone(ctx, data.get("timezone"), event_id)
    _copy_optional_fields(ctx, data, event_id)
    if allocations is not None:
        _copy_allocation_fields(ctx, allocations)
    return ctx


def _allocations(data: dict, event_id: str) -> list[NodeAllocation]:
    """Validate the per-node placement list and reject ambiguous flat fields."""
    conflicting = [key for key in _FLAT_PLACEMENT_FIELDS if data.get(key) is not None]
    if conflicting:
        raise EventError(f"CloudEvent {event_id} allocations cannot be combined with {conflicting}")
    raw = data["allocations"]
    if not isinstance(raw, list) or not raw:
        raise EventError(f"CloudEvent {event_id} allocations must be a non-empty list of objects")
    allocations = [_allocation(item, f"allocations[{index}]", event_id) for index, item in enumerate(raw)]
    _reject_duplicates([allocation["node"] for allocation in allocations], "allocation node", event_id)
    replica_names = [replica["name"] for allocation in allocations for replica in allocation["replicas"]]
    _reject_duplicates(replica_names, "replica name", event_id)
    return allocations


def _allocation(value: object, field: str, event_id: str) -> NodeAllocation:
    """Validate one node allocation; missing or null replicas mean none."""
    if not isinstance(value, dict):
        raise EventError(f"CloudEvent {event_id} {field} must be an object")
    replicas = value.get("replicas")
    if replicas is None:
        replicas = []
    if not isinstance(replicas, list):
        raise EventError(f"CloudEvent {event_id} {field}.replicas must be a list of objects")
    return {
        "node": _non_empty_str(value.get("node"), f"{field}.node", event_id),
        "cluster": _non_empty_str(value.get("cluster"), f"{field}.cluster", event_id),
        "gpus": _non_negative_int(value.get("gpus"), f"{field}.gpus", event_id),
        "replicas": [_replica(item, f"{field}.replicas[{index}]", event_id) for index, item in enumerate(replicas)],
    }


def _replica(value: object, field: str, event_id: str) -> ReplicaPlacement:
    """Validate one replica; a missing, null, or empty IP is unknown, not invalid."""
    if not isinstance(value, dict):
        raise EventError(f"CloudEvent {event_id} {field} must be an object")
    replica: ReplicaPlacement = {"name": _non_empty_str(value.get("name"), f"{field}.name", event_id)}
    ip = value.get("ip")
    if ip is None or ip == "":
        return replica
    # ip_address also accepts integers, which are not a supported encoding.
    if not isinstance(ip, str):
        raise EventError(f"CloudEvent {event_id} {field}.ip must be a string")
    try:
        replica["ip"] = str(ipaddress.ip_address(ip))
    except ValueError as exc:
        raise EventError(f"CloudEvent {event_id} {field}.ip {ip!r} is not an IP address") from exc
    return replica


def _reject_duplicates(values: list[str], field: str, event_id: str) -> None:
    """Reject a placement that names the same node or replica twice."""
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise EventError(f"CloudEvent {event_id} {field} appears more than once: {duplicates}")


def _copy_allocation_fields(ctx: KafkaExecutionContext, allocations: list[NodeAllocation]) -> None:
    """Derive the flat placement fields that collectors and outputs already read.

    gpu_count is set even when 0, so outputs do not replace it with an
    estimate from the configured GPUs per node.
    """
    ctx["allocations"] = allocations
    ctx["gpu_count"] = sum(allocation["gpus"] for allocation in allocations)
    pod_nodes = {
        replica["name"]: allocation["node"] for allocation in allocations for replica in allocation["replicas"]
    }
    if pod_nodes:
        ctx["pod_names"] = list(pod_nodes)
        ctx["pod_nodes"] = pod_nodes
    # A job spanning clusters has no single cluster to query or report.
    clusters = {allocation["cluster"] for allocation in allocations}
    if len(clusters) == 1:
        ctx["cluster"] = allocations[0]["cluster"]


def _copy_optional_fields(ctx: KafkaExecutionContext, data: dict, event_id: str) -> None:
    """Copy scheduler/Pod metadata, validating numeric and collection fields."""
    for key in _OPTIONAL_STR_FIELDS:
        value = data.get(key)
        if value is not None and value != "":
            ctx[key] = str(value)

    for key in ("time_limit", "gpu_count"):
        value = data.get(key)
        if value is not None:
            ctx[key] = _non_negative_int(value, key, event_id)
    # A zero time limit means no limit, as in the Slurm adapter. The service
    # rejects a time limit that is not positive.
    if ctx.get("time_limit") == 0:
        del ctx["time_limit"]

    failed_nodes = data.get("failed_nodes")
    if failed_nodes:
        ctx["failed_nodes"] = _str_list(failed_nodes, "failed_nodes", event_id)
        ctx["failed_node"] = min(ctx["failed_nodes"])
    elif data.get("failed_node"):
        ctx["failed_node"] = str(data["failed_node"])

    if data.get("pod_names"):
        ctx["pod_names"] = _str_list(data["pod_names"], "pod_names", event_id)
    pod_nodes = data.get("pod_nodes")
    if pod_nodes is not None:
        if not isinstance(pod_nodes, dict) or not all(
            isinstance(pod, str) and pod.strip() and isinstance(node, str) and node.strip()
            for pod, node in pod_nodes.items()
        ):
            raise EventError(f"CloudEvent {event_id} pod_nodes must map non-empty pod names to node names")
        ctx["pod_nodes"] = dict(pod_nodes)


def _validate_timezone(ctx: KafkaExecutionContext, timezone: object, event_id: str) -> None:
    """Resolve the service's timezone before an event can be committed.

    Invalid keys are payload errors; failures reading a valid zone database are
    host failures and must abort discovery without acknowledging the event.
    """
    if timezone is not None and timezone != "":
        try:
            ctx["timezone"] = ZoneInfo(str(timezone)).key
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise EventError(f"CloudEvent {event_id} timezone {timezone!r} is not an IANA key") from exc
        except OSError as exc:
            # With the tzdata package, a directory name ("America") or an
            # overlong key raises these instead of ZoneInfoNotFoundError. Any
            # other OSError is a host problem with a possibly valid key: fail
            # the run uncommitted so the event is retried, not skipped.
            if isinstance(exc, (IsADirectoryError, NotADirectoryError, FileNotFoundError)) or exc.errno in (
                errno.ENAMETOOLONG,
                errno.EINVAL,
            ):
                raise EventError(f"CloudEvent {event_id} timezone {timezone!r} is not an IANA key") from exc
            raise

    # The service needs an IANA zone; reject now what the orchestrator would
    # reject after the offset is committed (e.g. +05:30 without `timezone`).
    try:
        _service_timezone_name(ctx["start_time"], ctx["end_time"], ctx.get("timezone"))
    except ValueError as exc:
        raise EventError(f"CloudEvent {event_id}: {exc}") from exc


def _str_or_default(value: object, default: str) -> str:
    """Normalize optional scheduler identifiers using the adapter defaults."""
    return default if value is None or value == "" else str(value)


def _str_list(value: object, field: str, event_id: str) -> list[str]:
    """Copy a list of nonempty names; reject mixed or non-list payloads."""
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise EventError(f"CloudEvent {event_id} {field} must be a list of non-empty strings")
    return list(value)


def _non_empty_str(value: object, field: str, event_id: str) -> str:
    """Require a non-empty string without coercing other JSON types."""
    if not isinstance(value, str) or not value:
        raise EventError(f"CloudEvent {event_id} {field} must be a non-empty string")
    return value


def _aware_datetime(value: object, field: str, event_id: str) -> datetime:
    """Parse an ISO-8601 timestamp and require an explicit UTC offset."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise EventError(f"CloudEvent {event_id} {field}={value!r} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise EventError(f"CloudEvent {event_id} {field} must include a UTC offset")
    return parsed


def _non_negative_int(value: object, field: str, event_id: str) -> int:
    """Reject booleans, floats, and negative allocation/duration values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EventError(f"CloudEvent {event_id} {field} must be a non-negative integer")
    return value
