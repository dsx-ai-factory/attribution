# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import sqlite3
from datetime import datetime, timezone

from attribution_blueprint.connectors.outputs.sqlite import SCHEMA_VERSION, SQLiteOutput


def _make_context(
    job_id: str = "job-1",
    sluid: str = "sluid-1",
    workload_type: str = "slurm",
    scheduler: str | None = "slurm",
) -> dict:
    """Mirror what the real Slurm connector produces: both `workload_type`
    (wire) and `scheduler` (BP-internal) are set. Tests that intentionally
    exercise the workload_type path can pass `scheduler=None`."""
    ctx: dict = {
        "id": sluid or job_id,
        "sluid": sluid,
        "attempt_id": sluid,
        "workload_type": workload_type,
        "job_id": job_id,
        "nodes": ["node001"],
        "start_time": datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc),
        "state": "COMPLETED",
        "user_name": "alice",
    }
    if scheduler is not None:
        ctx["scheduler"] = scheduler
    return ctx


def _connect(db):
    return sqlite3.connect(str(db))


def test_write_creates_all_tables_and_schema_version_row(tmp_path):
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    out.write(_make_context(), {"verdict": {"attribution_outcome": "Fail User"}})
    out.close()

    conn = _connect(db)
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"schema_version", "execution_context", "attribution_run"} <= names
        row = conn.execute("SELECT version, description FROM schema_version").fetchone()
        assert row[0] == SCHEMA_VERSION
        assert row[1]
    finally:
        conn.close()


def test_execution_context_columns_populated(tmp_path):
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    ctx = _make_context()
    ctx["workload_framework"] = "megatron"
    ctx["nodes"] = ["example-node-016", "example-node-017"]
    out.write(ctx, {"verdict": {"attribution_outcome": "Fail InfraHW"}})
    out.close()

    conn = _connect(db)
    try:
        row = conn.execute(
            "SELECT job_id, attempt_id, scheduler_type, status, workload_framework, user_name, nodes_json FROM execution_context"
        ).fetchone()
    finally:
        conn.close()

    job_id, attempt_id, sched, status, workload_framework, user_name, nodes_json = row
    assert job_id == "job-1"
    assert attempt_id == "sluid-1"
    assert sched == "slurm"
    assert status == "COMPLETED"
    assert workload_framework == "megatron"
    assert user_name == "alice"
    assert json.loads(nodes_json) == ["example-node-016", "example-node-017"]


def test_attribution_run_columns_populated(tmp_path):
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    out.write(
        _make_context(),
        {
            "verdict": {"attribution_outcome": "Fail InfraHW"},
            "model_type": "AttributionResponseV2",
            "terminal_state": "fail",
            "sub_domain": "hardware",
        },
    )
    out.close()

    conn = _connect(db)
    try:
        row = conn.execute(
            "SELECT schema_version, pipeline_version, service_api_version, outcome, terminal_state, sub_domain, "
            "started_at, completed_at, raw_result_json FROM attribution_run"
        ).fetchone()
    finally:
        conn.close()

    sv, pv, sav, outcome, terminal_state, sub_domain, started, completed, raw = row
    assert sv == SCHEMA_VERSION
    assert pv  # populated from package version
    assert sav == "AttributionResponseV2"
    assert outcome == "Fail InfraHW"
    assert terminal_state == "fail"
    assert sub_domain == "hardware"
    assert datetime.fromisoformat(started).tzinfo is not None
    assert datetime.fromisoformat(completed).tzinfo is not None
    assert json.loads(raw)["verdict"]["attribution_outcome"] == "Fail InfraHW"


def test_attribution_run_preserves_explicit_empty_taxonomy_fields(tmp_path):
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    out.write(
        _make_context(),
        {
            "verdict": {"attribution_outcome": "Fail InfraHW"},
            "terminal_state": "",
            "sub_domain": "",
        },
    )
    out.close()

    conn = _connect(db)
    try:
        row = conn.execute("SELECT terminal_state, sub_domain FROM attribution_run").fetchone()
    finally:
        conn.close()

    assert row == ("", "")


def test_re_run_adds_a_new_attribution_run_but_not_a_duplicate_context(tmp_path):
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    ctx = _make_context()
    out.write(ctx, {"verdict": {"attribution_outcome": "Fail User"}})
    out.write(ctx, {"verdict": {"attribution_outcome": "Fail InfraHW"}})
    out.close()

    conn = _connect(db)
    try:
        ec_count = conn.execute("SELECT COUNT(*) FROM execution_context").fetchone()[0]
        run_count = conn.execute("SELECT COUNT(*) FROM attribution_run").fetchone()[0]
        outcomes = [r[0] for r in conn.execute("SELECT outcome FROM attribution_run ORDER BY started_at")]
    finally:
        conn.close()

    assert ec_count == 1
    assert run_count == 2
    assert outcomes == ["Fail User", "Fail InfraHW"]


def test_distinct_attempts_are_distinct_contexts(tmp_path):
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    out.write(_make_context(job_id="j", sluid="s-A"), {"verdict": {"attribution_outcome": "Fail User"}})
    out.write(_make_context(job_id="j", sluid="s-B"), {"verdict": {"attribution_outcome": "Fail User"}})
    out.close()

    conn = _connect(db)
    try:
        n = conn.execute("SELECT COUNT(*) FROM execution_context").fetchone()[0]
    finally:
        conn.close()
    assert n == 2


def test_close_on_uninitialized(tmp_path):
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    out.close()
    assert not db.exists()


def test_close_after_writes_does_not_raise(tmp_path):
    out = SQLiteOutput(str(tmp_path / "out.db"))
    out.write(_make_context(), {"verdict": {"attribution_outcome": "Fail User"}})
    out.close()
    out.close()


def test_indexes_exist(tmp_path):
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    out.write(_make_context(), {"verdict": {"attribution_outcome": "Fail User"}})
    out.close()

    conn = _connect(db)
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    finally:
        conn.close()
    assert "idx_execution_context_job_id" in names
    assert "idx_run_outcome" in names


def test_duplicate_started_at_appends_both_verdicts(tmp_path, mocker):
    """Two writes with the same started_at must preserve both audit rows."""
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    # Pin _now() so the two writes collide on started_at.
    mocker.patch(
        "attribution_blueprint.connectors.outputs.sqlite._now",
        return_value="2026-05-01T00:00:00+00:00",
    )
    out.write(_make_context(), {"verdict": {"attribution_outcome": "Fail User"}})
    out.write(_make_context(), {"verdict": {"attribution_outcome": "Fail InfraHW"}})
    out.close()

    conn = _connect(db)
    try:
        run_count = conn.execute("SELECT COUNT(*) FROM attribution_run").fetchone()[0]
        outcomes = [r[0] for r in conn.execute("SELECT outcome FROM attribution_run ORDER BY started_at")]
    finally:
        conn.close()
    assert run_count == 2
    assert outcomes == ["Fail User", "Fail InfraHW"]


def test_responsible_nodes_collected_from_populated_symptoms(tmp_path):
    """`_responsible_nodes_from` walks `attribution["symptoms"]` (a name → site-list mapping),
    not `verdict.symptoms`. The result is the sorted union of node names across all sites.
    This locks down the path the prior review fixed in the doc but not in a test."""
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    attribution = {
        "verdict": {"attribution_outcome": "Fail InfraHW"},
        # Top-level symptoms, not verdict.symptoms - sites list under each key.
        "symptoms": {
            "XID-145-NVLINK_RLW_ERROR": [
                {"node": "n3", "ts": 1},
                {"node": "n1", "ts": 2},
            ],
            "ib_link_down": [
                {"node": "n2"},
                {"node": "n1"},  # duplicate of XID's n1 - must be deduped in the union
            ],
        },
    }
    out.write(_make_context(), attribution)
    out.close()

    conn = _connect(db)
    try:
        row = conn.execute("SELECT responsible_nodes_json FROM attribution_run").fetchone()
    finally:
        conn.close()
    assert json.loads(row[0]) == ["n1", "n2", "n3"]  # sorted union, deduped


def test_responsible_nodes_collected_from_resourceless_source_marker(tmp_path):
    """A resourceless service source can still identify a responsible node in evidence."""
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    attribution = {
        "verdict": {"attribution_outcome": "Fail InfraHW"},
        "symptoms": {
            "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)": [
                {
                    "type": "ResourcelessSource",
                    "source": {
                        "line": (
                            "NCCL WARN NET/IB: Got completion from peer with "
                            "status=IBV_WC_RETRY_EXC_ERR(12) responsible_node=node-0201"
                        ),
                    },
                    "source_channel": "applog",
                },
            ],
        },
    }
    out.write(_make_context(), attribution)
    out.close()

    conn = _connect(db)
    try:
        row = conn.execute("SELECT responsible_nodes_json FROM attribution_run").fetchone()
    finally:
        conn.close()
    assert json.loads(row[0]) == ["node-0201"]


def test_responsible_nodes_exclude_ibv_modify_qp_pair_evidence(tmp_path):
    """Pair evidence must not be reduced to one responsible endpoint."""
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    symptoms = (
        "Call to ibv_modify_qp failed",
        "Call to ibv_modify_qp failed: eno 110 timeout",
    )
    for index, symptom in enumerate(symptoms):
        attribution = {
            "verdict": {"attribution_outcome": "Fail InfraHW"},
            "symptoms": {
                symptom: [
                    {
                        "type": "NodeSource",
                        "node": f"node-{index}",
                        "source": {"line": symptom},
                        "source_channel": "applog",
                    },
                ],
            },
        }
        out.write(_make_context(job_id=f"job-{index}", sluid=f"sluid-{index}"), attribution)
    out.close()

    conn = _connect(db)
    try:
        rows = conn.execute("SELECT responsible_nodes_json FROM attribution_run ORDER BY started_at").fetchall()
    finally:
        conn.close()
    assert [json.loads(row[0]) for row in rows] == [[], []]


def test_responsible_nodes_empty_when_symptoms_absent(tmp_path):
    """No `symptoms` key → empty list (not NULL, not missing column)."""
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    out.write(_make_context(), {"verdict": {"attribution_outcome": "Fail User"}})
    out.close()

    conn = _connect(db)
    try:
        row = conn.execute("SELECT responsible_nodes_json FROM attribution_run").fetchone()
    finally:
        conn.close()
    assert json.loads(row[0]) == []


def test_kueue_context_lands_with_kueue_scheduler_via_internal_field(tmp_path):
    """The kueue ECA emits `workload_type=""` (UNDEFINED for the service)
    AND `scheduler="kueue"` (BP-internal identity). SQLite reads `scheduler`
    in preference to `workload_type`, so the `scheduler_type` column is
    `"kueue"` even though the wire payload sends an empty string. This is
    the decoupling that lets BP label its own rows without coordinating
    service-side enum changes."""
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    ctx = {
        "id": "wl-uid-1",
        "workload_type": "",  # wire-format: UNDEFINED for the service
        "scheduler": "kueue",  # BP-internal: connector identity
        "attempt_id": "wl-uid-1",
        "job_id": "team-alpha/my-workload",
        "nodes": ["node-k8s-3"],
        "start_time": datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 20, 12, 30, 0, tzinfo=timezone.utc),
        "state": "FINISHED_SUCCEEDED",
        "user_name": "alice",
    }
    out.write(ctx, {"verdict": {"attribution_outcome": "Pass"}})
    out.close()

    conn = _connect(db)
    try:
        row = conn.execute("SELECT job_id, attempt_id, scheduler_type FROM execution_context").fetchone()
    finally:
        conn.close()
    job_id, attempt_id, sched = row
    assert job_id == "team-alpha/my-workload"
    assert attempt_id == "wl-uid-1"
    assert sched == "kueue"


def test_write_prefers_scheduler_field_over_workload_type(tmp_path):
    """When both `scheduler` and `workload_type` are set, SQLite writes
    `scheduler` to the column. This locks down the precedence: the
    BP-internal identity wins over the wire field."""
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    ctx = {
        "id": "x",
        "workload_type": "some-wire-value",
        "scheduler": "static",
        "attempt_id": "x",
        "job_id": "j",
        "nodes": ["n"],
        "start_time": datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 20, 13, 0, 0, tzinfo=timezone.utc),
        "state": "DONE",
    }
    out.write(ctx, {"verdict": {"attribution_outcome": "Pass"}})
    out.close()

    conn = _connect(db)
    try:
        sched = conn.execute("SELECT scheduler_type FROM execution_context").fetchone()[0]
    finally:
        conn.close()
    assert sched == "static"


def test_write_uses_workload_type_when_scheduler_absent(tmp_path):
    """Hand-built ctx dicts without `scheduler` still produce a non-empty
    `scheduler_type` column from `workload_type`."""
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    ctx = {
        "id": "y",
        "workload_type": "slurm",
        # NOTE: no `scheduler` key set, this exercises the workload_type path.
        "attempt_id": "y",
        "job_id": "j2",
        "nodes": ["n"],
        "start_time": datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 20, 13, 0, 0, tzinfo=timezone.utc),
        "state": "COMPLETED",
    }
    out.write(ctx, {"verdict": {"attribution_outcome": "Pass"}})
    out.close()

    conn = _connect(db)
    try:
        sched = conn.execute("SELECT scheduler_type FROM execution_context").fetchone()[0]
    finally:
        conn.close()
    assert sched == "slurm"


def test_slurm_and_kueue_same_job_id_coexist_as_distinct_rows(tmp_path):
    """The unique key is (job_id, attempt_id, scheduler_type). With the v2
    schema, identical job_id/attempt_id values under different scheduler_type
    values must land as separate rows."""
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    common_attempt = "shared-id"
    common_job = "job/x"
    out.write(
        {
            "id": common_attempt,
            "workload_type": "slurm",
            "scheduler": "slurm",
            "attempt_id": common_attempt,
            "job_id": common_job,
            "nodes": ["n1"],
            "start_time": datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc),
            "end_time": datetime(2026, 5, 20, 13, 0, 0, tzinfo=timezone.utc),
            "state": "COMPLETED",
        },
        {"verdict": {"attribution_outcome": "Pass"}},
    )
    out.write(
        {
            "id": common_attempt,
            # kueue wire-format value is "" (service enum doesn't know "kueue"),
            # but the BP-internal `scheduler` is "kueue" - SQLite uses the
            # latter for the column so the row stays distinguishable.
            "workload_type": "",
            "scheduler": "kueue",
            "attempt_id": common_attempt,
            "job_id": common_job,
            "nodes": ["n2"],
            "start_time": datetime(2026, 5, 20, 14, 0, 0, tzinfo=timezone.utc),
            "end_time": datetime(2026, 5, 20, 15, 0, 0, tzinfo=timezone.utc),
            "state": "FINISHED_SUCCEEDED",
        },
        {"verdict": {"attribution_outcome": "Pass"}},
    )
    out.close()
    conn = _connect(db)
    try:
        types_present = sorted(r[0] for r in conn.execute("SELECT scheduler_type FROM execution_context"))
    finally:
        conn.close()
    # Slurm's `scheduler` is "slurm"; kueue's `scheduler` is "kueue".
    # Two distinct values → two distinct rows under the
    # (job_id, attempt_id, scheduler_type) unique key, with honest labels.
    assert types_present == ["kueue", "slurm"]


def test_workload_framework_column_populated_from_context(tmp_path):
    """When the ECA populates ctx['workload_framework'] (e.g. static ECA with
    workload_framework='megatron'), the value lands in the
    execution_context.workload_framework column."""
    db = tmp_path / "out.db"
    out = SQLiteOutput(str(db))
    ctx = _make_context()
    ctx["workload_framework"] = "megatron"
    out.write(ctx, {"verdict": {"attribution_outcome": "Fail User"}})
    out.close()

    conn = _connect(db)
    try:
        row = conn.execute("SELECT workload_framework FROM execution_context").fetchone()
    finally:
        conn.close()
    assert row[0] == "megatron"


def test_migrate_adds_taxonomy_columns_to_existing_db(tmp_path):
    from attribution_blueprint.connectors.outputs.sqlite import migrate

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "CREATE TABLE schema_version (version TEXT PRIMARY KEY, description TEXT, applied_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE attribution_run ("
            "id TEXT PRIMARY KEY, execution_context_id TEXT NOT NULL, schema_version TEXT NOT NULL, "
            "pipeline_version TEXT, service_api_version TEXT, ruleset_version TEXT, started_at TEXT NOT NULL, "
            "completed_at TEXT, outcome TEXT NOT NULL, raw_result_json TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO attribution_run"
            "(id, execution_context_id, schema_version, started_at, outcome, raw_result_json) "
            "VALUES ('run-1', 'ctx-1', '4', '2026-05-01T00:00:00+00:00', 'Fail InfraHW', '{}')"
        )
        conn.commit()
    finally:
        conn.close()

    assert migrate(str(db)) == SCHEMA_VERSION

    conn = sqlite3.connect(str(db))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(attribution_run)")}
        responsible_nodes = conn.execute("SELECT responsible_nodes_json FROM attribution_run").fetchone()[0]
    finally:
        conn.close()
    assert {"terminal_state", "sub_domain", "responsible_nodes_json"} <= columns
    assert json.loads(responsible_nodes) == []
