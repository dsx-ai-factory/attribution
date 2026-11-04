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
must be non-negative integers. Invalid events raise EventError; callers decide
whether to skip them and how to account for their offsets.
"""

from __future__ import annotations

import errno
import json
from datetime import datetime
from typing import NotRequired
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


class EventError(ValueError):
    """The event payload is invalid and may be skipped without retrying."""


def _read_cloudevent(
    raw: bytes | None,
    headers: list[tuple[str, bytes | str | None]] | None,
) -> tuple[str, str, dict, str | None, str]:
    """Return (id, type, data, subject, source) from either CloudEvent encoding."""
    if raw is None:
        raise EventError("message has no value")
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventError(f"value is not JSON: {exc}") from exc

    headers = dict(headers or [])
    if "ce_type" in headers:
        attrs = {key[3:]: _header_str(value) for key, value in headers.items() if key.startswith("ce_")}
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
    return event_id, event_type, data, subject, source


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
    """Validate an event `data` object and build an ExecutionContext from it."""
    missing = [key for key in ("id", "start_time", "end_time", "nodes") if data.get(key) in (None, "", [])]
    if missing:
        raise EventError(f"CloudEvent {event_id} data missing required keys: {missing}")

    ctx_id = str(data["id"])
    ctx: KafkaExecutionContext = {
        "id": ctx_id,
        "workload_type": workload_type,
        "scheduler": _str_or_default(data.get("scheduler"), default_scheduler),
        "nodes": _str_list(data["nodes"], "nodes", event_id),
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
    return ctx


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
