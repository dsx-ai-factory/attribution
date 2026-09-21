import json
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from attribution_blueprint.connectors.logs.service_observation_adapter import adapt_log_observation_for_service
from attribution_blueprint.types import Format, Source

_WINDOW_START = datetime(2026, 1, 5, 11, 0, 0, tzinfo=timezone.utc)
_WINDOW_END = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)


def _adapted_entries(
    body: str,
    context_timezone: Optional[str] = None,
    window_end: datetime = _WINDOW_END,
) -> list[dict]:
    observation = {
        "context": {
            "source": Source.SYSLOG,
            "format": Format.RAW_LOG,
            "resources": "node-a",
            "time_interval": {"start": _WINDOW_START, "end": window_end},
        },
        "body": body,
    }
    return json.loads(adapt_log_observation_for_service(observation, context_timezone)["body"])


def test_raw_syslog_adapted_to_service_loki_dict_list():
    observation = {
        "context": {
            "source": Source.SYSLOG,
            "format": Format.RAW_LOG,
            "resources": "node-a",
            "time_interval": {
                "start": datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc),
            },
        },
        "body": "2026-05-01T00:00:03Z node-a kernel\nnode-a nvrm\n",
    }

    adapted = adapt_log_observation_for_service(observation)

    assert observation["context"]["format"] == Format.RAW_LOG
    assert adapted["context"]["format"] == Format.LOKI_DICT_LIST
    assert adapted["context"]["resources"] == "AllJobResources"
    assert json.loads(adapted["body"]) == [
        {
            "timestamp": 1777593603000000000,
            "body": "2026-05-01T00:00:03Z node-a kernel",
            "node": "node-a",
        },
    ]


def test_raw_syslog_adapter_can_infer_node_from_line():
    observation = {
        "context": {
            "source": "syslog",
            "format": "raw_log",
            "resources": "AllJobResources",
            "time_interval": {
                "start": "2026-05-01T00:00:00+00:00",
                "end": "2026-05-01T01:00:00+00:00",
            },
        },
        "body": (
            "2026-05-01T00:00:03Z node-b kernel\n"
            "<34>1 2026-05-01T00:00:04Z node-b app - - - msg\n"
            "May  1 00:00:05 node-b kernel: oom-killer\n"
        ),
    }

    adapted = adapt_log_observation_for_service(observation)

    assert [entry["node"] for entry in json.loads(adapted["body"])] == ["node-b", "node-b", "node-b"]


def test_non_raw_syslog_observation_is_unchanged():
    observation = {
        "context": {
            "source": Source.APPLOG,
            "format": Format.RAW_LOG,
            "resources": "AllJobResources",
            "time_interval": {
                "start": datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc),
            },
        },
        "body": "rank log\n",
    }

    assert adapt_log_observation_for_service(observation) is observation


def test_rfc5424_rfc3164_and_dmesg_ctime_lines_are_dated_from_the_line():
    entries = _adapted_entries(
        "<34>1 2026-01-05T11:59:03Z node-a app - - - rfc5424\n"
        "Jan  5 11:59:03 node-a kernel: rfc3164\n"
        "[Mon Jan  5 11:59:03 2026] nvidia-nvlink: dmesg -T\n"
    )

    assert [entry["timestamp"] for entry in entries] == [1767614343000000000] * 3


def test_offsetless_line_is_dated_in_the_execution_context_timezone():
    entries = _adapted_entries(
        "Jan  5 03:30:00 node-a kernel: oom-killer\n",
        context_timezone="America/Los_Angeles",
    )

    dated = datetime.fromtimestamp(entries[0]["timestamp"] / 1e9, tz=timezone.utc)
    assert dated == datetime(2026, 1, 5, 11, 30, 0, tzinfo=timezone.utc)
    assert _WINDOW_START <= dated <= _WINDOW_END


def test_yearless_december_line_in_a_january_window_is_dated_to_the_previous_year():
    entries = _adapted_entries("Dec 31 23:59:03 node-a kernel: stale oom\n")

    assert entries[0]["timestamp"] == 1767225543000000000


def test_yearless_leap_day_line_is_dated_to_the_preceding_leap_year():
    entries = _adapted_entries(
        "Feb 29 12:00:00 node-a kernel: leap day\n",
        window_end=datetime(2025, 3, 1, 0, 0, 0, tzinfo=timezone.utc),
    )

    dated = datetime.fromtimestamp(entries[0]["timestamp"] / 1e9, tz=timezone.utc)
    assert dated == datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc)


def test_yearless_line_is_not_dated_past_a_window_end_in_another_year():
    entries = _adapted_entries(
        "Dec 31 23:00:00 node-a kernel: stale oom\n",
        context_timezone="America/New_York",
        window_end=datetime(2026, 1, 1, 2, 0, 0, tzinfo=timezone.utc),
    )

    dated = datetime.fromtimestamp(entries[0]["timestamp"] / 1e9, tz=timezone.utc)
    assert dated == datetime(2025, 1, 1, 4, 0, 0, tzinfo=timezone.utc)


def test_lines_without_a_parseable_timestamp_are_dropped_with_one_warning():
    warnings: list[str] = []
    sink_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
    try:
        entries = _adapted_entries(
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


def test_observation_with_only_undatable_lines_keeps_an_empty_entry_list():
    assert _adapted_entries("[12345.678901] NVRM: Xid 43\n") == []
