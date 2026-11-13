# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from attribution_blueprint.connectors.outputs.sqlite import SQLiteOutput
from attribution_blueprint.webui import db as dbmod


def _ctx(job_id, sluid, end):
    return {
        "id": sluid,
        "sluid": sluid,
        "attempt_id": sluid,
        "workload_type": "slurm",
        "scheduler": "slurm",
        "job_id": job_id,
        "nodes": ["node-a"],
        "start_time": datetime(2026, 5, 1, 0, tzinfo=timezone.utc),
        "end_time": end,
        "state": "FAILED",
        "user_name": "alice",
    }


def _attr(cls, symptom="XID-79", rule="r"):
    return {
        "verdict": {
            "attribution_outcome": cls,
            "detail": "",
            "root_symptom": {"symptom": symptom, "source": {"rule": rule}},
        },
        "symptoms": {},
        "model_type": "AttributionResponseV2",
    }


def test_count_runs_matches_list_runs_filters(tmp_path):
    p = tmp_path / "c.db"
    out = SQLiteOutput(str(p))
    out.write(_ctx("J1", "s1", datetime(2026, 5, 1, 1, tzinfo=timezone.utc)), _attr("Fail InfraHW"))
    out.write(_ctx("J2", "s2", datetime(2026, 5, 2, 1, tzinfo=timezone.utc)), _attr("Pass"))
    out.close()
    assert dbmod.count_runs(str(p)) == 2
    assert dbmod.count_runs(str(p), outcome="Fail InfraHW") == 1
    assert dbmod.count_runs(str(p), since="2026-05-02", until="2026-05-02") == 1


@pytest.mark.parametrize("field", ["symptom", "rule"])
@pytest.mark.parametrize("value", ["XID_79%", "quote'\"\\failure", " spaced "])
def test_root_filters_match_exact_identifiers_before_pagination(tmp_path, field, value):
    p = tmp_path / "roots.db"
    matching = {field: value}
    nested = _attr("Fail InfraHW")
    nested["verdict"]["detail"] = value
    nested["verdict"]["decision_graph"] = {"nodes": [{"symptom": value, "rule": value}]}
    nested["symptoms"] = {value: [{"source": {"rule": value}}]}
    other_field = "rule" if field == "symptom" else "symptom"
    payloads = [
        _attr("Fail InfraHW", **matching),
        _attr("Fail InfraHW", **matching),
        _attr("Pass", **matching),
        nested,
        _attr("Fail InfraHW", **{other_field: value}),
        *[
            _attr("Fail InfraHW", **{field: near_match})
            for near_match in [f"prefix{value}", f"{value}suffix", value.swapcase(), f" {value} "]
        ],
    ]
    out = SQLiteOutput(str(p))
    for index, payload in enumerate(payloads):
        out.write(_ctx(f"J{index}", f"s{index}", datetime(2026, 5, index + 1, tzinfo=timezone.utc)), payload)
    out.close()

    runs, total = dbmod.list_runs_page(str(p), **matching)
    assert total == 3
    assert [run.job_id for run in runs] == ["J2", "J1", "J0"]

    runs, total = dbmod.list_runs_page(str(p), **matching, status="failed", page=1, page_size=1)
    assert total == 2
    assert [run.job_id for run in runs] == ["J0"]


def test_root_filters_combine_and_clear(tmp_path):
    p = tmp_path / "combined-roots.db"
    out = SQLiteOutput(str(p))
    for index, (symptom, rule) in enumerate([("XID-79", "r"), ("XID-79", "other"), ("other", "r")]):
        out.write(
            _ctx(f"J{index}", f"s{index}", datetime(2026, 5, index + 1, tzinfo=timezone.utc)),
            _attr("Fail InfraHW", symptom=symptom, rule=rule),
        )
    out.close()

    runs, total = dbmod.list_runs_page(str(p), symptom="XID-79", rule="r")
    assert total == 1
    assert [run.job_id for run in runs] == ["J0"]

    runs, total = dbmod.list_runs_page(str(p), symptom="", rule="")
    assert total == 3
    assert [run.job_id for run in runs] == ["J2", "J1", "J0"]


@pytest.mark.parametrize("field", ["symptom", "rule"])
def test_root_filters_ignore_invalid_json_and_non_string_fields(tmp_path, field):
    p = tmp_path / "invalid-roots.db"
    payloads = [
        json.dumps(_attr("Fail InfraHW", **{field: "{}"})),
        "{",
        "null",
        "[]",
        "{}",
        '{"verdict": null}',
        '{"verdict": {"root_symptom": null}}',
        *[json.dumps(_attr("Fail InfraHW", **{field: value})) for value in [None, {}, [], True, 1]],
    ]
    out = SQLiteOutput(str(p))
    for index in range(len(payloads)):
        out.write(
            _ctx(f"J{index}", f"s{index}", datetime(2026, 5, index + 1, tzinfo=timezone.utc)),
            _attr("Fail InfraHW"),
        )
    out.close()
    with sqlite3.connect(p) as conn:
        for index, raw_json in enumerate(payloads):
            conn.execute(
                "UPDATE attribution_run SET raw_result_json = ? WHERE execution_context_id IN "
                "(SELECT id FROM execution_context WHERE job_id = ?)",
                (raw_json, f"J{index}"),
            )

    runs, total = dbmod.list_runs_page(str(p), **{field: "{}"})
    assert total == 1
    assert [run.job_id for run in runs] == ["J0"]


def test_date_window_normalizes_end_time_offsets_to_utc(tmp_path: Path) -> None:
    """Runs ending at the same UTC instant land in the same UTC day window (#210)."""
    p = tmp_path / "offsets.db"
    pacific = timezone(timedelta(hours=-7))
    out = SQLiteOutput(str(p))
    out.write(_ctx("utc-job", "s1", datetime(2026, 8, 8, 0, 30, tzinfo=timezone.utc)), _attr("Fail InfraHW"))
    out.write(_ctx("offset-job", "s2", datetime(2026, 8, 7, 17, 30, tzinfo=pacific)), _attr("Fail InfraHW"))
    out.close()

    assert dbmod.count_runs(str(p), since="2026-08-08", until="2026-08-08") == 2
    assert dbmod.count_runs(str(p), since="2026-08-07", until="2026-08-07") == 0
    data = dbmod.dashboard_data(str(p), since="2026-08-08", until="2026-08-08")
    assert data["metrics"]["windowedRuns"] == 2
    assert [(d["date"], d["hw"]) for d in data["daily"]] == [("2026-08-08", 2)]


def test_unfiltered_dashboard_spine_uses_utc_days(tmp_path: Path) -> None:
    """Without a window, chart days and date bounds follow the UTC bucket of offset runs."""
    p = tmp_path / "offset-only.db"
    pacific = timezone(timedelta(hours=-7))
    out = SQLiteOutput(str(p))
    out.write(_ctx("offset-job", "s1", datetime(2026, 8, 7, 23, 30, tzinfo=pacific)), _attr("Fail InfraHW"))
    out.close()

    data = dbmod.dashboard_data(str(p))
    assert data["metrics"]["windowedRuns"] == 1
    assert [(d["date"], d["hw"]) for d in data["daily"]] == [("2026-08-08", 1)]
    assert data["meta"]["dateBounds"]["max"] == "2026-08-08"
    assert dbmod.date_range_bounds(str(p)) == ("2026-08-08", "2026-08-08")


def test_date_window_accepts_calendar_extremes(tmp_path: Path) -> None:
    """Windows touching year 1 or year 9999 still filter instead of overflowing the widened text bound."""
    p = tmp_path / "extremes.db"
    out = SQLiteOutput(str(p))
    out.write(_ctx("J1", "s1", datetime(2026, 8, 8, 1, tzinfo=timezone.utc)), _attr("Fail InfraHW"))
    out.close()

    assert dbmod.count_runs(str(p), since="0001-01-01") == 1
    assert dbmod.count_runs(str(p), until="9999-12-30") == 1


def test_date_window_seeks_end_time_index(tmp_path: Path) -> None:
    """The UTC window keeps an indexed end_time range so narrow windows do not scan every run."""
    p = tmp_path / "plan.db"
    out = SQLiteOutput(str(p))
    out.write(_ctx("J1", "s1", datetime(2026, 8, 8, 1, tzinfo=timezone.utc)), _attr("Fail InfraHW"))
    out.close()
    clauses, params = dbmod._filtered_clauses(since="2026-08-08", until="2026-08-08")
    sql = (
        "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM attribution_run r "
        "JOIN execution_context c ON c.id = r.execution_context_id WHERE " + " AND ".join(clauses)
    )
    with sqlite3.connect(p) as conn:
        plan = " ".join(row[-1] for row in conn.execute(sql, params).fetchall())
    assert "idx_execution_context_end_time" in plan
