# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Build ENTRY_TIMESERIES entries out of raw syslog text.

The service keeps a syslog entry only when it can place it in time, so every
entry needs a `ts`. It is read from the line itself; a line whose timestamp
cannot be parsed is dropped here rather than stamped with a window bound,
which is how a stale record once passed the service-side time filter.
"""

import re
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger

from attribution_blueprint.types import ExecutionContext, TimeseriesEntry

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_RFC5424_HEADER = re.compile(r"^<\d+>\d+$")
# RFC 3164 / journalctl default: "Sep 13 04:12:01" and space-padded "Sep  3 04:12:01".
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


def syslog_entries_from_text(text: str, context: ExecutionContext, node: Optional[str] = None) -> list[TimeseriesEntry]:
    """One ENTRY_TIMESERIES entry per dated, non-empty syslog line.

    Args:
        text: Raw syslog text, one record per line.
        context: Execution context. `end_time` supplies the year for yearless
            lines; `timezone`, when set, is the zone of offsetless timestamps.
        node: Hostname the whole text belongs to, declared on every entry. When
            None the entry declares no resources and the service reads the
            hostname out of the line.

    Raises:
        ZoneInfoNotFoundError: If `context["timezone"]` is not a known IANA key.
    """
    window_end = context["end_time"]
    # Offsetless lines are host-local, and the window may be in another zone.
    line_tz = _line_timezone(context.get("timezone"), window_end)
    entries: list[TimeseriesEntry] = []
    dropped = 0
    for line in text.splitlines():
        if line == "":
            continue
        line_time = _line_datetime(line, window_end, line_tz)
        if line_time is None:
            dropped += 1
            continue
        entry: TimeseriesEntry = {"body": line, "ts": _unix_ns(line_time)}
        if node is not None and node != "":
            entry["resources"] = [{"kind": "node", "hostname": node}]
        entries.append(entry)

    if dropped != 0:
        logger.warning(
            f"syslog: dropped {dropped} of {dropped + len(entries)} lines with no parseable timestamp "
            f"(ctx {context.get('id')})"
        )
    return entries


def _line_timezone(context_timezone: Optional[str], window_end: datetime) -> Optional[tzinfo]:
    if context_timezone is not None and context_timezone != "":
        return ZoneInfo(context_timezone)
    return window_end.tzinfo


def _unix_ns(value: datetime) -> int:
    # Integer arithmetic: `timestamp() * 1e9` loses the last few hundred nanoseconds.
    return (value - _EPOCH) // timedelta(microseconds=1) * 1000


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
    # A bare date ("2026-01-05") parses but carries no time of day, so it does not date a line.
    if ":" not in token:
        return None
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None


def _rfc3164_datetime(parts: list[str], window_end: datetime, line_tz: Optional[tzinfo]) -> Optional[datetime]:
    """Parse a yearless "Mon DD HH:MM:SS" prefix, inferring the year from the window."""
    if len(parts) < 3 or parts[0] not in _MONTH_ABBREVS or not parts[1].isdigit():
        return None
    if _RFC3164_TIME.match(parts[2]) is None:
        return None

    month = _MONTH_ABBREVS[parts[0]]
    day = int(parts[1])
    hour, minute, second = (int(field) for field in parts[2].split(":"))
    local_window_end = window_end.astimezone(line_tz) if line_tz is not None else window_end
    # No year on the wire, so take the latest candidate that has not run past the window end:
    # a December line read in January lands last year, not a year ahead.
    for year in (local_window_end.year, local_window_end.year - 1):
        try:
            parsed = datetime(year, month, day, hour, minute, second, tzinfo=line_tz)
        except ValueError:
            continue
        if parsed <= window_end:
            return parsed
    return None


__all__ = ["syslog_entries_from_text"]
