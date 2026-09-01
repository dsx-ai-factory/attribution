# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timezone

from attribution_blueprint.connectors.outputs.stdout import StdoutOutput


def _make_context() -> dict:
    return {
        "id": "ctx-1",
        "nodes": ["node001", "node002"],
        "start_time": datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc),
    }


def test_write_emits_one_json_line(capsys):
    out = StdoutOutput()
    out.write(_make_context(), {"verdict": "ok"})

    captured = capsys.readouterr().out
    lines = captured.splitlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert set(parsed.keys()) == {"context", "attribution"}
    assert parsed["context"]["id"] == "ctx-1"
    assert parsed["attribution"] == {"verdict": "ok"}


def test_two_writes_two_lines(capsys):
    out = StdoutOutput()
    out.write(_make_context(), {"verdict": "a"})
    out.write(_make_context(), {"verdict": "b"})

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["attribution"] == {"verdict": "a"}
    assert json.loads(lines[1])["attribution"] == {"verdict": "b"}


def test_datetime_serializes(capsys):
    out = StdoutOutput()
    out.write(_make_context(), {"verdict": "ok"})

    captured = capsys.readouterr().out
    parsed = json.loads(captured.splitlines()[0])
    assert isinstance(parsed["context"]["start_time"], str)
    assert "2026-05-01" in parsed["context"]["start_time"]


def test_close_is_noop():
    out = StdoutOutput()
    assert out.close() is None
