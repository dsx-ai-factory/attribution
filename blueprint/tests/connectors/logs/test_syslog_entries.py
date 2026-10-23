# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from typing import Optional

import pytest
from loguru import logger

from attribution_blueprint.connectors.logs.syslog import syslog_entries_from_text

_WINDOW_START = datetime(2026, 1, 5, 11, 0, 0, tzinfo=timezone.utc)
"""Jan 5, 2026, 11:00:00 UTC"""
_WINDOW_END = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
"""Jan 5, 2026, 12:00:00 UTC"""


def _entries(
    text: str,
    context_timezone: Optional[str] = None,
    node: Optional[str] = "node-a",
    window_end: datetime = _WINDOW_END,
) -> list[dict]:
    context = {"id": "ctx-1", "nodes": ["node-a"], "start_time": _WINDOW_START, "end_time": window_end}
    if context_timezone is not None:
        context["timezone"] = context_timezone
    return syslog_entries_from_text(text, context, node=node)


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("2026-01-05T11:59:03Z node-a kernel: iso", id="iso"),
        pytest.param("<34>1 2026-01-05T11:59:03Z node-a app - - - rfc5424", id="rfc5424"),
        pytest.param("Jan  5 11:59:03 node-a kernel: rfc3164", id="rfc3164"),
        pytest.param("[Mon Jan  5 11:59:03 2026] nvidia-nvlink: dmesg -T", id="dmesg-ctime"),
    ],
)
def test_supported_timestamp_line_is_dated_from_the_line(line):
    entries = _entries(f"{line}\n")

    assert entries == [
        {
            "ts": 1767614343000000000,
            "body": line,
            "resources": [{"kind": "node", "hostname": "node-a"}],
        }
    ]


def test_nanosecond_iso_fraction_is_kept_to_the_microsecond():
    entries = _entries("2026-01-05T11:59:03.123456789Z node-a kernel: nano\n")

    assert entries[0]["ts"] == 1767614343123456000


def test_offsetless_line_is_dated_in_the_execution_context_timezone():
    entries = _entries("Jan  5 03:30:00 node-a kernel: oom-killer\n", context_timezone="America/Los_Angeles")

    dated = datetime.fromtimestamp(entries[0]["ts"] / 1e9, tz=timezone.utc)
    assert dated == datetime(2026, 1, 5, 11, 30, 0, tzinfo=timezone.utc)


def test_yearless_december_line_in_a_january_window_is_dated_to_the_previous_year():
    entries = _entries("Dec 31 23:59:03 node-a kernel: stale oom\n")

    assert entries[0]["ts"] == 1767225543000000000


def test_yearless_leap_day_line_is_dated_to_the_preceding_leap_year():
    entries = _entries(
        "Feb 29 12:00:00 node-a kernel: leap day\n",
        window_end=datetime(2025, 3, 1, 0, 0, 0, tzinfo=timezone.utc),
    )

    dated = datetime.fromtimestamp(entries[0]["ts"] / 1e9, tz=timezone.utc)
    assert dated == datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc)


def test_yearless_line_is_not_dated_past_a_window_end_in_another_year():
    entries = _entries(
        "Dec 31 23:00:00 node-a kernel: stale oom\n",
        context_timezone="America/New_York",
        window_end=datetime(2026, 1, 1, 2, 0, 0, tzinfo=timezone.utc),
    )

    dated = datetime.fromtimestamp(entries[0]["ts"] / 1e9, tz=timezone.utc)
    assert dated == datetime(2025, 1, 1, 4, 0, 0, tzinfo=timezone.utc)


def test_lines_without_a_parseable_timestamp_are_dropped_with_one_warning():
    warnings: list[str] = []
    sink_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
    try:
        entries = _entries(
            "[12345.678901] NVRM: Xid 43\n"
            "node-a nvrm\n"
            "2026-01-05 node-a kernel: date with no time\n"
            "2026-01-05T11:59:03Z node-a kernel: ok\n"
        )
    finally:
        logger.remove(sink_id)

    assert [entry["body"] for entry in entries] == ["2026-01-05T11:59:03Z node-a kernel: ok"]
    assert len(warnings) == 1
    assert "dropped 3 of 4" in warnings[0]


def test_known_node_is_declared_on_every_entry_and_omitted_when_unknown():
    line = "2026-01-05T11:59:03Z node-a kernel: ok\n"

    assert _entries(line)[0]["resources"] == [{"kind": "node", "hostname": "node-a"}]
    assert "resources" not in _entries(line, node=None)[0]
