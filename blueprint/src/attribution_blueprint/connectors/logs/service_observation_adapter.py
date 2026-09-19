# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Adapt log observations to the attribution-service observation contract."""

import json
import re
from datetime import datetime, tzinfo
from typing import Any, Optional
from zoneinfo import ZoneInfo

from loguru import logger

from attribution_blueprint.types import Format, Observation, Source

_ALL_JOB_RESOURCES = "AllJobResources"

_RFC5424_HEADER = re.compile(r"^<\d+>\d+$")
# RFC3164 / journalctl default: "Sep 13 04:12:01" and space-padded "Sep  3 04:12:01".
_RFC3164_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}$")
# `dmesg -T` prefixes each line with a bracketed ctime: "[Mon Sep 13 04:12:01 2026]".
_DMESG_CTIME = re.compile(r"^\[([A-Z][a-z]{2} [A-Z][a-z]{2} [ 0-9]?\d \d{2}:\d{2}:\d{2} \d{4})\]")
_MONTH_ABBREVS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def adapt_log_observation_for_service(observation: Observation, context_timezone: Optional[str] = None) -> Observation:
    """Return a service-compatible observation without mutating the collector output.

    Args:
        observation: Log observation emitted by a connector.
        context_timezone: IANA key from the execution context, used to interpret
            offsetless line timestamps.

    Returns:
        Observation in the shape accepted by the attribution service.

    Raises:
        ZoneInfoNotFoundError: If `context_timezone` is not a known IANA key.
    """
    context = observation["context"]
    if _value(context["source"]) == Source.SYSLOG.value and _value(context["format"]) == Format.RAW_LOG.value:
        return _raw_syslog_to_loki_dict_list(observation, context_timezone)
    return observation


def _raw_syslog_to_loki_dict_list(observation: Observation, context_timezone: Optional[str]) -> Observation:
    """Convert raw syslog text to the flat Loki entry list accepted by the service.

    Lines whose timestamp cannot be parsed are dropped: stamping them with the
    context start time can make stale records pass the service-side time filter.

    Args:
        observation: Raw syslog observation emitted by a connector.
        context_timezone: IANA key from the execution context, used to interpret
            offsetless line timestamps.

    Returns:
        Service-compatible observation with a JSON-encoded Loki entry list.
    """
    context = observation["context"]
    node = _resource_node(context.get("resources"))
    window_end = _as_datetime(context["time_interval"]["end"])
    # Offsetless syslog lines are host-local, and the window may be in another zone.
    line_tz = ZoneInfo(context_timezone) if context_timezone else window_end.tzinfo
    if window_end.tzinfo is None:
        window_end = window_end.replace(tzinfo=line_tz)
    entries: list[dict[str, int | str]] = []
    dropped = 0
    for line in observation["body"].splitlines():
        if line == "":
            continue
        timestamp_ns = _line_timestamp_ns(line, window_end, line_tz)
        if timestamp_ns is None:
            dropped += 1
            continue
        entry_node = node if node is not None and node != "" else _line_node(line)
        entries.append(
            {
                "timestamp": timestamp_ns,
                "body": line,
                "node": entry_node,
            }
        )

    if dropped != 0:
        logger.warning(
            f"service_observation_adapter: dropped {dropped} of {dropped + len(entries)} "
            "syslog lines with no parseable timestamp"
        )

    adapted_context = dict(context)
    adapted_context["format"] = Format.LOKI_DICT_LIST
    adapted_context["resources"] = _ALL_JOB_RESOURCES
    return {
        "context": adapted_context,
        "body": json.dumps(entries),
    }


def _as_datetime(value: datetime | str) -> datetime:
    """Normalize an observation interval bound to a datetime.

    Args:
        value: Interval bound as a datetime or ISO-formatted string.

    Returns:
        Parsed datetime value.
    """
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _line_timestamp_ns(line: str, window_end: datetime, line_tz: Optional[tzinfo]) -> Optional[int]:
    """Return a nanosecond timestamp parsed from a syslog line, or None when undatable.

    Args:
        line: Raw syslog line.
        window_end: End of the observation time interval, used to infer a missing year.
        line_tz: Zone applied to line timestamps that carry no offset.

    Returns:
        Unix timestamp in nanoseconds, or None when the line carries no usable time.
    """
    parsed = _line_datetime(line, window_end, line_tz)
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1e9)


def _line_datetime(line: str, window_end: datetime, line_tz: Optional[tzinfo]) -> Optional[datetime]:
    """Parse the leading timestamp of a syslog line in any recognized format."""
    parts = line.split()
    if parts == []:
        return None

    iso = _iso_datetime(parts[0])
    if iso is None and _RFC5424_HEADER.match(parts[0]) is not None and len(parts) >= 2:
        iso = _iso_datetime(parts[1])
    if iso is not None:
        return iso if iso.tzinfo is not None else iso.replace(tzinfo=line_tz)

    ctime = _DMESG_CTIME.match(line)
    if ctime is not None:
        try:
            parsed = datetime.strptime(ctime.group(1), "%a %b %d %H:%M:%S %Y")
        except ValueError:
            return None
        return parsed.replace(tzinfo=line_tz)

    return _rfc3164_datetime(parts, window_end, line_tz)


def _iso_datetime(token: str) -> Optional[datetime]:
    """Parse a token as a complete ISO 8601 date-time, or None when it is not one."""
    if ":" not in token:
        return None
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None


def _has_rfc3164_prefix(parts: list[str]) -> bool:
    """Report whether a line starts with a yearless "Mon DD HH:MM:SS" triple."""
    if len(parts) < 3:
        return False
    return parts[0] in _MONTH_ABBREVS and parts[1].isdigit() and _RFC3164_TIME.match(parts[2]) is not None


def _rfc3164_datetime(parts: list[str], window_end: datetime, line_tz: Optional[tzinfo]) -> Optional[datetime]:
    """Parse a yearless "Mon DD HH:MM:SS" prefix, inferring the year from the window."""
    if _has_rfc3164_prefix(parts) is False:
        return None

    month = _MONTH_ABBREVS[parts[0]]
    day = int(parts[1])
    hour, minute, second = (int(field) for field in parts[2].split(":"))
    local_window_end = window_end.astimezone(line_tz) if line_tz is not None else window_end
    # The format carries no year, so take the latest candidate that has not run past the
    # window end: a December line read in January lands last year, not a year ahead.
    for year in (local_window_end.year, local_window_end.year - 1):
        try:
            parsed = datetime(year, month, day, hour, minute, second, tzinfo=line_tz)
        except ValueError:
            continue
        if parsed <= window_end:
            return parsed
    return None


def _line_node(line: str) -> str:
    """Infer a node name from a raw syslog line.

    Args:
        line: Raw syslog line.

    Returns:
        Node token inferred from the line, or an empty string when unavailable.
    """
    parts = line.split()
    if parts == []:
        return ""
    if _iso_datetime(parts[0]) is not None:
        return parts[1] if len(parts) >= 2 else ""
    if _RFC5424_HEADER.match(parts[0]) is not None:
        return parts[2] if len(parts) >= 3 else ""
    if _has_rfc3164_prefix(parts):
        return parts[3] if len(parts) >= 4 else ""
    if line.startswith("["):  # bracketed dmesg prefixes carry no hostname
        return ""
    return parts[0]


def _resource_node(resources: Any) -> str | None:
    """Extract a concrete node name from an observation resource value.

    Args:
        resources: Observation resource field value.

    Returns:
        Node name when the resource is node-specific, otherwise None.
    """
    if isinstance(resources, str) is False:
        return None
    if resources == "" or resources == _ALL_JOB_RESOURCES:
        return None
    return resources


def _value(value: Any) -> str:
    """Normalize enum-like and string values for observation context comparisons.

    Args:
        value: Enum-like value, string, or other comparable object.

    Returns:
        String value suitable for comparison with service enum values.
    """
    if hasattr(value, "value"):
        return str(value.value)
    if isinstance(value, str):
        return value.lower()
    return str(value)


__all__ = ["adapt_log_observation_for_service"]
