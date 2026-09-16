import json
import sqlite3
from datetime import datetime, timezone

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
