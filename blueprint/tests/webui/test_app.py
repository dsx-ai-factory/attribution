# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sqlite3
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")  # NOTE: jinja2 importorskip removed

from fastapi.testclient import TestClient

from attribution_blueprint.connectors.outputs.sqlite import SQLiteOutput
from attribution_blueprint.webui import app as appmod
from attribution_blueprint.webui.app import create_app
from attribution_blueprint.webui.db import DatabaseSchemaError


@pytest.fixture(autouse=True)
def stub_spa(tmp_path, monkeypatch):
    """Provide a minimal built SPA so create_app's bringup guard passes and the
    catch-all can read index.html + serve /assets."""
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text('<html><body><div id="root"></div></body></html>', encoding="utf-8")
    (static / "assets" / "app.js").write_text("// built", encoding="utf-8")
    (static / "logo.svg").write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(appmod, "_STATIC", static)
    monkeypatch.setattr(appmod, "_INDEX", static / "index.html")


def _ctx(job_id, sluid, *, nodes=("node-a",), state="FAILED", user="alice"):
    return {
        "id": sluid,
        "sluid": sluid,
        "attempt_id": sluid,
        "workload_type": "slurm",
        "scheduler": "slurm",
        "job_id": job_id,
        "nodes": list(nodes),
        "start_time": datetime(2026, 5, 1, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 1, tzinfo=timezone.utc),
        "state": state,
        "user_name": user,
    }


def _attr(cls, symptom="XID-79", node="node-a", rule="r"):
    return {
        "verdict": {
            "attribution_outcome": cls,
            "detail": "d",
            "root_symptom": {
                "symptom": symptom,
                "source": {"rule": rule, "line": "kernel: ...", "time": "2026-05-01T00:30:00+00:00"},
            },
        },
        "symptoms": {
            symptom: [
                {
                    "type": "NodeSource",
                    "node": node,
                    "source": {"rule": rule, "line": "kernel: ...", "time": "2026-05-01T00:30:00+00:00"},
                }
            ]
        },
        "model_type": "AttributionResponseV2",
    }


@pytest.fixture
def populated_db(tmp_path):
    p = tmp_path / "test.db"
    out = SQLiteOutput(str(p))
    out.write(_ctx("100", "sluid-A"), _attr("Fail InfraHW"))
    out.write(_ctx("101", "sluid-B", state="COMPLETED"), _attr("Pass", "completes"))
    out.close()
    return str(p)


def test_index_serves_spa_with_bootstrap(populated_db):
    client = TestClient(create_app(populated_db))
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="root"' in r.text
    assert "attr-bootstrap" not in r.text
    dashboard = client.get("/api/v1/dashboard").json()
    assert dashboard["meta"]["totalRuns"] == 2


def test_dashboard_query_params_filter_server_payload(tmp_path):
    """The dashboard API filters server-side while the SPA shell stays static."""
    p = tmp_path / "dates.db"
    out = SQLiteOutput(str(p))
    # Two runs on different end-time dates.
    c1 = _ctx("300", "sluid-1")
    c1["start_time"] = datetime(2026, 5, 1, 0, tzinfo=timezone.utc)
    c1["end_time"] = datetime(2026, 5, 1, 1, tzinfo=timezone.utc)
    c2 = _ctx("301", "sluid-2")
    c2["start_time"] = datetime(2026, 5, 10, 0, tzinfo=timezone.utc)
    c2["end_time"] = datetime(2026, 5, 10, 1, tzinfo=timezone.utc)
    out.write(c1, _attr("Fail InfraHW"))
    out.write(c2, _attr("Fail InfraSW"))
    out.close()
    client = TestClient(create_app(str(p)))
    # A window that excludes both runs should return an empty dashboard payload.
    r = client.get("/?from=2099-01-01&to=2099-01-02")
    assert r.status_code == 200
    assert "attr-bootstrap" not in r.text
    dashboard = client.get("/api/v1/dashboard", params={"from": "2099-01-01", "to": "2099-01-02"}).json()
    assert dashboard["meta"]["totalRuns"] == 0


def test_empty_db_serves_empty_payload(tmp_path):
    p = tmp_path / "empty.db"
    out = SQLiteOutput(str(p))
    out._connect()
    out.close()
    client = TestClient(create_app(str(p)))
    r = client.get("/")
    assert r.status_code == 200
    assert "attr-bootstrap" not in r.text
    assert client.get("/api/v1/dashboard").json()["meta"]["totalRuns"] == 0


def test_deep_link_returns_spa_shell(populated_db):
    client = TestClient(create_app(populated_db))
    # FQDN node deep link must reach the SPA, not 404.
    r = client.get("/node/gpu-01.cluster.local")
    assert r.status_code == 200 and 'id="root"' in r.text
    r2 = client.get("/verdict/anything")
    assert r2.status_code == 200 and 'id="root"' in r2.text


def test_missing_asset_under_assets_is_404(populated_db):
    client = TestClient(create_app(populated_db))
    r = client.get("/assets/does-not-exist.js")
    assert r.status_code == 404


def test_root_file_served(populated_db):
    client = TestClient(create_app(populated_db))
    r = client.get("/logo.svg")
    assert r.status_code == 200 and "svg" in r.text


def test_xss_payload_cannot_break_out(tmp_path):
    p = tmp_path / "xss.db"
    out = SQLiteOutput(str(p))
    out.write(_ctx("200", "sluid-X"), _attr("Fail InfraHW", symptom="</script><img src=x onerror=alert(1)>"))
    out.close()
    client = TestClient(create_app(str(p)))
    body = client.get("/").text
    assert "</script><img" not in body
    assert "attr-bootstrap" not in body
    payload = client.get("/api/v1/runs", params={"pageSize": 1}).json()
    assert any("onerror=alert(1)" in s["name"] for run in payload["runs"] for s in run["symptoms"])


def test_dashboard_combined_mtbi_excludes_multicategory(tmp_path):
    p = tmp_path / "metrics.db"
    out = SQLiteOutput(str(p))
    hw = _ctx("hw", "hw-1")
    hw["gpu_count"] = 8
    hw["d_gpu_hours"] = 1000
    sw = _ctx("sw", "sw-1")
    sw["gpu_count"] = 8
    sw["d_gpu_hours"] = 1000
    mix = _ctx("mix", "mix-1")
    mix["gpu_count"] = 8
    mix["d_gpu_hours"] = 1000
    out.write(hw, _attr("Fail InfraHW"))
    out.write(sw, _attr("Fail InfraSW"))
    out.write(mix, _attr("Fail MultiCategory"))
    out.close()
    client = TestClient(create_app(str(p)))

    metrics = client.get("/api/v1/dashboard").json()["metrics"]

    assert metrics["gpuHoursRaw"] == 3000
    assert metrics["hwFailures"] == 1
    assert metrics["swFailures"] == 1
    assert metrics["mixFailures"] == 1
    assert metrics["combinedFailures"] == 2
    assert metrics["combinedMtbi"] == 1.5


def test_dashboard_failure_metrics_count_attempts_not_distinct_jobs(tmp_path):
    p = tmp_path / "attempt-metrics.db"
    out = SQLiteOutput(str(p))
    first = _ctx("same-job", "same-job-1")
    first["gpu_count"] = 8
    first["d_gpu_hours"] = 100
    second = _ctx("same-job", "same-job-2")
    second["gpu_count"] = 8
    second["d_gpu_hours"] = 100
    out.write(first, _attr("Fail InfraHW"))
    out.write(second, _attr("Fail InfraHW"))
    out.close()
    client = TestClient(create_app(str(p)))

    dashboard = client.get("/api/v1/dashboard").json()

    assert dashboard["metrics"]["totalRuns"] == 2
    assert dashboard["metrics"]["hwFailures"] == 2
    assert dashboard["daily"][0]["hw"] == 2


def test_dashboard_fallback_ledgers_use_normalized_schema(tmp_path):
    p = tmp_path / "fallback-ledgers.db"
    out = SQLiteOutput(str(p))
    hw = _ctx("hw", "hw-1", user="alice", nodes=("node-a",))
    hw["gpu_count"] = 8
    hw["d_gpu_hours"] = 1000
    sw = _ctx("sw", "sw-1", user="bob", nodes=("node-b",))
    sw["gpu_count"] = 8
    sw["d_gpu_hours"] = 500
    out.write(hw, _attr("Fail InfraHW", symptom="XID-79", node="node-a", rule="xid_rule"))
    out.write(sw, _attr("Fail InfraSW", symptom="NCCL timeout", node="node-b", rule="nccl_rule"))
    out.close()
    client = TestClient(create_app(str(p)))

    ledgers = client.get("/api/v1/dashboard").json()["ledgers"]

    assert ledgers["symptom"] == [
        {"symptom": "NCCL timeout", "count": 1, "nodeHours": 500.0, "outcomes": {"software": 1}},
        {"symptom": "XID-79", "count": 1, "nodeHours": 1000.0, "outcomes": {"hardware": 1}},
    ]
    assert ledgers["node"] == [
        {
            "node": "node-a",
            "count": 1,
            "nodeHours": 1000.0,
            "outcomes": {"hardware": 1},
            "symptoms": {"XID-79": 1},
        },
        {
            "node": "node-b",
            "count": 1,
            "nodeHours": 500.0,
            "outcomes": {"software": 1},
            "symptoms": {"NCCL timeout": 1},
        },
    ]
    assert ledgers["user"] == [
        {"user": "alice", "count": 1, "nodeHours": 1000.0, "outcomes": {"hardware": 1}},
        {"user": "bob", "count": 1, "nodeHours": 500.0, "outcomes": {"software": 1}},
    ]
    assert ledgers["rule"] == [
        {"rule": "nccl_rule", "count": 1, "nodeHours": 500.0, "outcomes": {"software": 1}},
        {"rule": "xid_rule", "count": 1, "nodeHours": 1000.0, "outcomes": {"hardware": 1}},
    ]
    assert ledgers["outcome"] == [
        {"outcome": "hardware", "count": 1, "nodeHours": 1000.0},
        {"outcome": "software", "count": 1, "nodeHours": 500.0},
    ]


def test_dashboard_surfaces_undetermined_outcome(tmp_path):
    """Fail Undetermined runs must appear in the outcome ledger and the runs
    table groups, not only in the window COUNT(*)."""
    p = tmp_path / "undetermined.db"
    out = SQLiteOutput(str(p))
    hw = _ctx("hw", "hw-1", user="alice", nodes=("node-a",))
    hw["gpu_count"] = 8
    hw["d_gpu_hours"] = 1000
    und = _ctx("und", "und-1", user="bob", nodes=("node-b",))
    und["gpu_count"] = 8
    und["d_gpu_hours"] = 500
    out.write(hw, _attr("Fail InfraHW", symptom="XID-79", node="node-a", rule="xid_rule"))
    out.write(und, _attr("Fail Undetermined", symptom="JobCrash", node="node-b", rule="crash_rule"))
    out.close()
    client = TestClient(create_app(str(p)))

    dashboard = client.get("/api/v1/dashboard").json()
    assert dashboard["meta"]["totalRuns"] == 2
    assert dashboard["ledgers"]["outcome"] == [
        {"outcome": "hardware", "count": 1, "nodeHours": 1000.0},
        {"outcome": "undetermined", "count": 1, "nodeHours": 500.0},
    ]
    assert dashboard["ledgers"]["symptom"] == [
        {"symptom": "JobCrash", "count": 1, "nodeHours": 500.0, "outcomes": {"undetermined": 1}},
        {"symptom": "XID-79", "count": 1, "nodeHours": 1000.0, "outcomes": {"hardware": 1}},
    ]

    runs = client.get("/api/v1/runs", params={"outcome": "undetermined", "status": "failed"}).json()
    assert runs["total"] == 1
    assert runs["runs"][0]["jobId"] == "und"
    assert runs["runs"][0]["outcome"] == "undetermined"
    assert runs["runs"][0]["verdictLabel"] == "Unattributed"


def test_dashboard_rollup_outcome_ledger_includes_undetermined(tmp_path):
    p = tmp_path / "rollup-undetermined.db"
    out = SQLiteOutput(str(p))
    out.write(_ctx("raw", "raw-1"), _attr("Pass"))
    out.close()
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "CREATE TABLE webui_run_fact ("
            "run_id TEXT PRIMARY KEY, "
            "end_time TEXT NOT NULL, "
            "end_day TEXT NOT NULL, "
            "job_id TEXT NOT NULL, "
            "outcome TEXT NOT NULL, "
            "gpu_hours REAL NOT NULL, "
            "gpu_count INTEGER NOT NULL, "
            "root_symptom TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO webui_run_fact"
            "(run_id, end_time, end_day, job_id, outcome, gpu_hours, gpu_count, root_symptom) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("rollup-hw", "2026-05-01T01:00:00+00:00", "2026-05-01", "hw", "Fail InfraHW", 1000, 8, "XID"),
                (
                    "rollup-und",
                    "2026-05-01T02:00:00+00:00",
                    "2026-05-01",
                    "und",
                    "Fail Undetermined",
                    500,
                    8,
                    "JobCrash",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    client = TestClient(create_app(str(p)))

    dashboard = client.get("/api/v1/dashboard").json()
    assert dashboard["metrics"]["totalRuns"] == 2
    assert dashboard["ledgers"]["outcome"] == [
        {"outcome": "hardware", "count": 1, "nodeHours": 1000.0},
        {"outcome": "undetermined", "count": 1, "nodeHours": 500.0},
    ]
    assert dashboard["ledgers"]["symptom"] == [
        {"symptom": "JobCrash", "count": 1, "nodeHours": 500.0, "outcomes": {"undetermined": 1}},
        {"symptom": "XID", "count": 1, "nodeHours": 1000.0, "outcomes": {"hardware": 1}},
    ]


def test_dashboard_rejects_malformed_dates(tmp_path):
    p = tmp_path / "invalid-dates.db"
    out = SQLiteOutput(str(p))
    out.write(_ctx("hw", "hw-1"), _attr("Fail InfraHW"))
    out.close()
    client = TestClient(create_app(str(p)))

    assert client.get("/api/v1/dashboard", params={"to": "not-a-date"}).status_code == 422
    assert client.get("/api/v1/runs", params={"to": "not-a-date"}).status_code == 422


def test_dashboard_window_gpu_count_uses_unique_nodes(tmp_path):
    p = tmp_path / "window-gpus.db"
    out = SQLiteOutput(str(p))
    first = _ctx("first", "first-1", nodes=("node-a", "node-b"))
    first["gpus_per_node"] = 4
    first["gpu_count"] = 8
    second = _ctx("second", "second-1", nodes=("node-b", "node-c"))
    second["gpus_per_node"] = 4
    second["gpu_count"] = 8
    out.write(first, _attr("Fail InfraHW"))
    out.write(second, _attr("Fail InfraSW"))
    out.close()
    client = TestClient(create_app(str(p)))

    metrics = client.get("/api/v1/dashboard").json()["metrics"]

    assert metrics["gpuCount"] == 12


def test_dashboard_rollup_window_gpu_count_uses_unique_nodes(tmp_path):
    p = tmp_path / "rollup-window-gpus.db"
    out = SQLiteOutput(str(p))
    out.write(_ctx("raw", "raw-1"), _attr("Pass"))
    out.close()
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "CREATE TABLE webui_run_fact ("
            "run_id TEXT PRIMARY KEY, "
            "end_time TEXT NOT NULL, "
            "end_day TEXT NOT NULL, "
            "job_id TEXT NOT NULL, "
            "outcome TEXT NOT NULL, "
            "gpu_hours REAL NOT NULL, "
            "gpu_count INTEGER NOT NULL, "
            "root_symptom TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO webui_run_fact"
            "(run_id, end_time, end_day, job_id, outcome, gpu_hours, gpu_count, root_symptom) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("rollup-first", "2026-05-01T01:00:00+00:00", "2026-05-01", "first", "Fail InfraHW", 100.0, 8, "XID"),
                (
                    "rollup-second",
                    "2026-05-01T02:00:00+00:00",
                    "2026-05-01",
                    "second",
                    "Fail InfraSW",
                    100.0,
                    8,
                    "NCCL",
                ),
            ],
        )
        conn.execute(
            "CREATE TABLE webui_node_day_fact ("
            "end_day TEXT NOT NULL, "
            "node_name TEXT NOT NULL, "
            "gpus_per_node INTEGER NOT NULL, "
            "PRIMARY KEY(end_day, node_name))"
        )
        conn.executemany(
            "INSERT INTO webui_node_day_fact(end_day, node_name, gpus_per_node) VALUES (?, ?, ?)",
            [
                ("2026-05-01", "node-a", 4),
                ("2026-05-01", "node-b", 4),
                ("2026-05-01", "node-c", 4),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    client = TestClient(create_app(str(p)))

    metrics = client.get("/api/v1/dashboard").json()["metrics"]

    assert metrics["totalRuns"] == 2
    assert metrics["gpuCount"] == 12


def test_dashboard_rollup_failure_metrics_count_attempts_not_distinct_jobs(tmp_path):
    p = tmp_path / "rollup-attempt-metrics.db"
    out = SQLiteOutput(str(p))
    out.write(_ctx("raw", "raw-1"), _attr("Pass"))
    out.close()
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "CREATE TABLE webui_run_fact ("
            "run_id TEXT PRIMARY KEY, "
            "end_time TEXT NOT NULL, "
            "end_day TEXT NOT NULL, "
            "job_id TEXT NOT NULL, "
            "outcome TEXT NOT NULL, "
            "gpu_hours REAL NOT NULL, "
            "gpu_count INTEGER NOT NULL, "
            "root_symptom TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO webui_run_fact"
            "(run_id, end_time, end_day, job_id, outcome, gpu_hours, gpu_count, root_symptom) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("rollup-first", "2026-05-01T01:00:00+00:00", "2026-05-01", "same-job", "Fail InfraHW", 100, 8, "XID"),
                (
                    "rollup-second",
                    "2026-05-01T02:00:00+00:00",
                    "2026-05-01",
                    "same-job",
                    "Fail InfraHW",
                    100,
                    8,
                    "XID",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    client = TestClient(create_app(str(p)))

    dashboard = client.get("/api/v1/dashboard").json()

    assert dashboard["metrics"]["totalRuns"] == 2
    assert dashboard["metrics"]["hwFailures"] == 2
    assert dashboard["daily"][0]["hw"] == 2


def test_dashboard_uses_webui_rollup_when_present(tmp_path):
    p = tmp_path / "rollup.db"
    out = SQLiteOutput(str(p))
    out.write(_ctx("raw", "raw-1"), _attr("Pass"))
    out.close()
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "CREATE TABLE webui_run_fact ("
            "run_id TEXT PRIMARY KEY, "
            "end_time TEXT NOT NULL, "
            "end_day TEXT NOT NULL, "
            "job_id TEXT NOT NULL, "
            "outcome TEXT NOT NULL, "
            "gpu_hours REAL NOT NULL, "
            "gpu_count INTEGER NOT NULL, "
            "root_symptom TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO webui_run_fact"
            "(run_id, end_time, end_day, job_id, outcome, gpu_hours, gpu_count, root_symptom) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("rollup-hw", "2026-05-01T01:00:00+00:00", "2026-05-01", "hw", "Fail InfraHW", 1000, 8, "XID"),
                ("rollup-sw", "2026-05-01T02:00:00+00:00", "2026-05-01", "sw", "Fail InfraSW", 500, 8, "NCCL"),
                (
                    "rollup-mix",
                    "2026-05-01T03:00:00+00:00",
                    "2026-05-01",
                    "mix",
                    "Fail MultiCategory",
                    250,
                    8,
                    "Mixed",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    client = TestClient(create_app(str(p)))

    dashboard = client.get("/api/v1/dashboard").json()
    metrics = dashboard["metrics"]

    assert metrics["totalRuns"] == 3
    assert metrics["gpuHoursRaw"] == 1750
    assert metrics["hwFailures"] == 1
    assert metrics["swFailures"] == 1
    assert metrics["mixFailures"] == 1
    assert metrics["combinedFailures"] == 2
    assert metrics["combinedMtbi"] == 0.875
    assert dashboard["ledgers"]["symptom"] == [
        {"symptom": "Mixed", "count": 1, "nodeHours": 250.0, "outcomes": {"multi": 1}},
        {"symptom": "NCCL", "count": 1, "nodeHours": 500.0, "outcomes": {"software": 1}},
        {"symptom": "XID", "count": 1, "nodeHours": 1000.0, "outcomes": {"hardware": 1}},
    ]
    assert dashboard["ledgers"]["outcome"] == [
        {"outcome": "hardware", "count": 1, "nodeHours": 1000.0},
        {"outcome": "multi", "count": 1, "nodeHours": 250.0},
        {"outcome": "software", "count": 1, "nodeHours": 500.0},
    ]


def test_dashboard_rollup_node_ledger_is_capped_to_top_10(tmp_path):
    p = tmp_path / "nodes.db"
    out = SQLiteOutput(str(p))
    for idx in range(12):
        node = f"node-{idx:02d}"
        out.write(_ctx(f"node-job-{idx}", f"node-sluid-{idx}", nodes=(node,)), _attr("Fail InfraHW", node=node))
    out.close()
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "CREATE TABLE webui_run_fact ("
            "run_id TEXT PRIMARY KEY, "
            "end_time TEXT NOT NULL, "
            "end_day TEXT NOT NULL, "
            "job_id TEXT NOT NULL, "
            "user_name TEXT NOT NULL, "
            "outcome TEXT NOT NULL, "
            "gpu_hours REAL NOT NULL, "
            "gpu_count INTEGER NOT NULL, "
            "root_symptom TEXT NOT NULL, "
            "root_rule TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO webui_run_fact"
            "(run_id, end_time, end_day, job_id, user_name, outcome, gpu_hours, gpu_count, root_symptom, root_rule) "
            "SELECT r.id, c.end_time, substr(c.end_time, 1, 10), c.job_id, c.user_name, r.outcome, 100.0, 8, "
            "json_extract(r.raw_result_json, '$.verdict.root_symptom.symptom'), "
            "json_extract(r.raw_result_json, '$.verdict.root_symptom.source.rule') "
            "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id"
        )
        conn.commit()
    finally:
        conn.close()
    client = TestClient(create_app(str(p)))

    node_ledger = client.get("/api/v1/dashboard").json()["ledgers"]["node"]

    assert len(node_ledger) == 10
    assert [entry["node"] for entry in node_ledger] == [f"node-{idx:02d}" for idx in range(10)]
    assert node_ledger[0]["count"] == 1
    assert node_ledger[0]["outcomes"] == {"hardware": 1}
    assert node_ledger[0]["symptoms"] == {"XID-79": 1}
    assert node_ledger[0]["nodeHours"] == 100.0


def test_dashboard_rollup_pareto_ledgers_are_capped_to_top_10(tmp_path):
    p = tmp_path / "pareto.db"
    out = SQLiteOutput(str(p))
    for idx in range(12):
        out.write(
            _ctx(f"pareto-job-{idx}", f"pareto-sluid-{idx}", user=f"user-{idx:02d}"),
            _attr("Fail InfraHW", symptom=f"symptom-{idx:02d}", rule=f"rule-{idx:02d}"),
        )
    out.close()
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "CREATE TABLE webui_run_fact ("
            "run_id TEXT PRIMARY KEY, "
            "end_time TEXT NOT NULL, "
            "end_day TEXT NOT NULL, "
            "job_id TEXT NOT NULL, "
            "user_name TEXT NOT NULL, "
            "outcome TEXT NOT NULL, "
            "gpu_hours REAL NOT NULL, "
            "gpu_count INTEGER NOT NULL, "
            "root_symptom TEXT NOT NULL, "
            "root_rule TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO webui_run_fact"
            "(run_id, end_time, end_day, job_id, user_name, outcome, gpu_hours, gpu_count, root_symptom, root_rule) "
            "SELECT r.id, c.end_time, substr(c.end_time, 1, 10), c.job_id, c.user_name, r.outcome, 100.0, 8, "
            "json_extract(r.raw_result_json, '$.verdict.root_symptom.symptom'), "
            "json_extract(r.raw_result_json, '$.verdict.root_symptom.source.rule') "
            "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id"
        )
        conn.commit()
    finally:
        conn.close()
    client = TestClient(create_app(str(p)))

    ledgers = client.get("/api/v1/dashboard").json()["ledgers"]

    assert len(ledgers["symptom"]) == 10
    assert len(ledgers["rule"]) == 10
    assert len(ledgers["user"]) == 10
    assert [entry["symptom"] for entry in ledgers["symptom"]] == [f"symptom-{idx:02d}" for idx in range(10)]
    assert [entry["rule"] for entry in ledgers["rule"]] == [f"rule-{idx:02d}" for idx in range(10)]
    assert [entry["user"] for entry in ledgers["user"]] == [f"user-{idx:02d}" for idx in range(10)]
    assert ledgers["symptom"][0]["outcomes"] == {"hardware": 1}
    assert ledgers["rule"][0]["outcomes"] == {"hardware": 1}
    assert ledgers["user"][0]["outcomes"] == {"hardware": 1}


def test_dashboard_rollup_pareto_ledgers_follow_outcome_filter(tmp_path):
    p = tmp_path / "filtered-pareto.db"
    out = SQLiteOutput(str(p))
    for idx in range(12):
        for dup in range(2):
            node = f"sw-node-{idx:02d}"
            out.write(
                _ctx(
                    f"sw-job-{idx}-{dup}",
                    f"sw-sluid-{idx}-{dup}",
                    nodes=(node,),
                    user=f"sw-user-{idx:02d}",
                ),
                _attr(
                    "Fail InfraSW",
                    symptom=f"sw-symptom-{idx:02d}",
                    node=node,
                    rule=f"sw-rule-{idx:02d}",
                ),
            )
    for idx in range(4):
        node = f"hw-node-{idx:02d}"
        out.write(
            _ctx(f"hw-job-{idx}", f"hw-sluid-{idx}", nodes=(node,), user=f"hw-user-{idx:02d}"),
            _attr("Fail InfraHW", symptom=f"hw-symptom-{idx:02d}", node=node, rule=f"hw-rule-{idx:02d}"),
        )
    out.close()
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "CREATE TABLE webui_run_fact ("
            "run_id TEXT PRIMARY KEY, "
            "end_time TEXT NOT NULL, "
            "end_day TEXT NOT NULL, "
            "job_id TEXT NOT NULL, "
            "user_name TEXT NOT NULL, "
            "outcome TEXT NOT NULL, "
            "gpu_hours REAL NOT NULL, "
            "gpu_count INTEGER NOT NULL, "
            "root_symptom TEXT NOT NULL, "
            "root_rule TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO webui_run_fact"
            "(run_id, end_time, end_day, job_id, user_name, outcome, gpu_hours, gpu_count, root_symptom, root_rule) "
            "SELECT r.id, c.end_time, substr(c.end_time, 1, 10), c.job_id, c.user_name, r.outcome, 100.0, 8, "
            "json_extract(r.raw_result_json, '$.verdict.root_symptom.symptom'), "
            "json_extract(r.raw_result_json, '$.verdict.root_symptom.source.rule') "
            "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id"
        )
        conn.commit()
    finally:
        conn.close()
    client = TestClient(create_app(str(p)))

    dashboard = client.get("/api/v1/dashboard", params={"outcome": "hardware"}).json()
    ledgers = dashboard["ledgers"]

    assert dashboard["metrics"]["totalRuns"] == 28
    assert [entry["symptom"] for entry in ledgers["symptom"]] == [f"hw-symptom-{idx:02d}" for idx in range(4)]
    assert [entry["node"] for entry in ledgers["node"]] == [f"hw-node-{idx:02d}" for idx in range(4)]
    assert [entry["rule"] for entry in ledgers["rule"]] == [f"hw-rule-{idx:02d}" for idx in range(4)]
    assert [entry["user"] for entry in ledgers["user"]] == [f"hw-user-{idx:02d}" for idx in range(4)]
    assert all(entry["outcomes"] == {"hardware": 1} for entry in ledgers["symptom"])
    assert ledgers["outcome"] == [
        {"outcome": "software", "count": 24, "nodeHours": 2400.0},
        {"outcome": "hardware", "count": 4, "nodeHours": 400.0},
    ]


def test_create_app_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_app(str(tmp_path / "nope.db"))


def test_create_app_missing_spa_build_fails_loud(tmp_path, monkeypatch):
    # Point _INDEX at a nonexistent file to simulate an editable install w/o build.
    p = tmp_path / "ok.db"
    out = SQLiteOutput(str(p))
    out._connect()
    out.close()
    monkeypatch.setattr(appmod, "_INDEX", tmp_path / "static" / "nope.html")
    with pytest.raises(appmod.WebAssetsMissingError, match="React assets missing"):
        create_app(str(p))


def test_create_app_db_missing_current_columns_errors_with_migration_hint(tmp_path):
    import sqlite3

    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE schema_version (version TEXT PRIMARY KEY, description TEXT, applied_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE attribution_run (id TEXT PRIMARY KEY, execution_context_id TEXT NOT NULL, "
            "schema_version TEXT NOT NULL, pipeline_version TEXT, service_api_version TEXT, "
            "ruleset_version TEXT, started_at TEXT NOT NULL, completed_at TEXT, outcome TEXT NOT NULL, "
            "raw_result_json TEXT NOT NULL, responsible_nodes_json TEXT NOT NULL DEFAULT '[]')"
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DatabaseSchemaError, match="v1.1 TODO"):
        create_app(str(db_path))
