from datetime import datetime, timezone

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


def _attr(cls):
    return {
        "verdict": {"attribution_outcome": cls, "detail": "", "root_symptom": {"symptom": "XID-79", "source": {}}},
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
