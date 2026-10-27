# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import datetime, timezone

import pytest

from attribution_blueprint.connectors.outputs.sqlite import SQLiteOutput
from attribution_blueprint.webui import (
    bootstrap,
    db as dbmod,
)


def _seed_one(
    tmp_path,
    gpus_per_node=8,
    gpu_count=None,
    alloc_tres=None,
    catalog_id=None,
    symptom="GpuFell",
    rule="gpu_fell_off_bus",
    line="Xid 79",
    detail="GPU off bus",
    source_time="2026-05-01T00:30:00+00:00",
):
    p = tmp_path / "m.db"
    out = SQLiteOutput(str(p))
    root_symptom = {
        "symptom": symptom,
        "source": {"rule": rule, "line": line, "time": source_time},
    }
    if catalog_id is not None:
        root_symptom["catalog_id"] = catalog_id
    ctx = {
        "id": "s1",
        "sluid": "s1",
        "attempt_id": "s1",
        "workload_type": "slurm",
        "scheduler": "slurm",
        "job_id": "team/job-x",
        "nodes": ["node-a", "node-b"],
        "start_time": datetime(2026, 5, 1, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 2, tzinfo=timezone.utc),
        "state": "FAILED",
        "user_name": "alice",
    }
    if gpus_per_node is not None:
        ctx["gpus_per_node"] = gpus_per_node
    if gpu_count is not None:
        ctx["gpu_count"] = gpu_count
    if alloc_tres is not None:
        ctx["alloc_tres"] = alloc_tres
    out.write(
        ctx,
        {
            "verdict": {
                "attribution_outcome": "Fail InfraHW",
                "detail": detail,
                "root_symptom": root_symptom,
            },
            "symptoms": {
                symptom: [
                    {
                        "type": "NodeSource",
                        "node": "node-a",
                        "source": {"rule": rule, "line": line, "time": source_time},
                    }
                ]
            },
            "model_type": "AttributionResponseV2",
        },
    )
    out.close()
    return dbmod.get_run(str(p), dbmod.list_runs(str(p))[0].run_id)


def test_run_to_react_core_fields(tmp_path):
    run = _seed_one(tmp_path)
    r = bootstrap.run_to_react(run)
    assert r["jobId"] == "team/job-x"
    assert r["workload"] == "team/job-x"  # has "/" -> workload shown
    assert r["outcome"] == "hardware"
    assert r["nodes"] == 2 and r["nodeIds"] == ["node-a", "node-b"]
    assert r["elapsedH"] == 2
    assert r["gpusPerNode"] == 8
    assert r["gpuCount"] == 16
    assert r["rootSymptom"]["symptom"] == "GpuFell"
    assert r["rootSymptom"]["rule"] == "gpu_fell_off_bus"
    assert r["symptoms"] == [{"name": "GpuFell", "sites": 1}]
    assert r["errorPairs"] == ""
    # log is matched-line evidence: root symptom line present at level 'symptom'
    assert any(entry["level"] == "symptom" and "Xid 79" in entry["text"] for entry in r["log"])
    # diagnose card present (camelCase keys)
    assert set(["hasPersona", "title", "summary", "impact", "action", "supportDetails"]) <= set(r["diagnose"])
    assert "operatorDiagnose" not in r


def test_run_to_react_includes_bad_pairs(tmp_path):
    bad_pairs = "{'src_node': 'node-a', 'dst_node': 'node-b'}"
    run = _seed_one(tmp_path)
    run.raw_result["collectors"] = {"bad_pairs": bad_pairs}

    r = bootstrap.run_to_react(run)

    assert r["errorPairs"] == bad_pairs


@pytest.mark.parametrize("collectors", [None, {}, {"bad_pairs": None}, {"bad_pairs": ""}, {"bad_pairs": 0}])
def test_run_to_react_omits_invalid_bad_pairs(tmp_path, collectors):
    """Return empty error pairs for missing or malformed collector data.

    Args:
        tmp_path: Pytest temporary directory fixture.
        collectors: Collector payload being tested.
    """
    run = _seed_one(tmp_path)
    run.raw_result["collectors"] = collectors

    r = bootstrap.run_to_react(run)

    assert r["errorPairs"] == ""


def _empty_catalog(monkeypatch):
    from attribution_blueprint.webui import diagnose

    monkeypatch.setattr(diagnose, "_load_catalog", lambda: {})


def test_run_to_react_falls_back_when_catalog_is_empty(tmp_path, monkeypatch):
    _empty_catalog(monkeypatch)
    run = _seed_one(tmp_path, catalog_id="xid-79")
    r = bootstrap.run_to_react(run)
    assert r["diagnose"]["hasPersona"] is False
    assert r["diagnose"]["title"] == "GpuFell"
    assert r["diagnose"]["summary"] == "GPU off bus"
    assert r["diagnose"]["catalogId"] is None
    assert "operatorDiagnose" not in r


def test_generic_nvrx_failure_reports_unavailable_reason(tmp_path):
    """Keep the unavailable reason in the report and tenant diagnosis without operator guidance."""
    detail = "Underlying failure reason unavailable from the collected evidence."
    run = _seed_one(
        tmp_path,
        catalog_id="nvrx-detected-failure",
        symptom="NVRx detected failure",
        rule="NVRXFailureDetected",
        line="Cycle: 1 Event: failure_detected Node: node-a Rank: 0",
        detail=detail,
    )
    run.raw_result["verdict"]["attribution_outcome"] = "Fail Undetermined"

    result = bootstrap.run_to_react(run)

    assert result["verdictLabel"] == "Unattributed"
    assert result["detail"] == detail
    assert result["rootSymptom"]["symptom"] == "NVRx detected failure"
    assert result["diagnose"]["summary"] == detail
    assert "operatorDiagnose" not in result


def test_symptom_text_without_catalog_id_yields_no_persona_card(tmp_path):
    # Default seed: symptom "GpuFell" / line "Xid 79" regex-matches the xid-79
    # signal, but carries no typed catalog_id.
    run = _seed_one(tmp_path)
    r = bootstrap.run_to_react(run)
    assert r["diagnose"]["hasPersona"] is False
    assert r["diagnose"]["catalogId"] is None
    assert r["diagnose"]["title"] == "GpuFell"
    assert "operatorDiagnose" not in r


def test_run_to_react_parses_unsquashfs_timestamp_with_empty_catalog(tmp_path, monkeypatch):
    _empty_catalog(monkeypatch)
    run = _seed_one(
        tmp_path,
        symptom="unsquashfs: write failed (system path)",
        rule="UnsquashfsWriteFailed",
        line="May 01 00:30:00.123456 3218496 slurmstepd error: pyxis: couldn't start container",
        detail="Not Applicable",
        source_time="",
    )
    r = bootstrap.run_to_react(run)
    assert r["rootSymptom"]["time"] == "2026-05-01T00:30:00.123456+00:00"
    assert [line["level"] for line in r["log"]] == ["symptom"]
    assert r["diagnose"]["title"] == "unsquashfs: write failed (system path)"
    assert r["diagnose"]["summary"] == "Not Applicable"
    assert "operatorDiagnose" not in r


def test_packaged_catalog_tenant_guidance_reaches_payload(tmp_path):
    """Every packaged tenant entry retains its actions and evidence instructions."""
    from attribution_blueprint.webui.diagnose import _load_catalog

    catalog = _load_catalog()
    assert "xid-79" in catalog
    run = _seed_one(tmp_path, catalog_id="xid-79")
    checked = 0
    for entry_id, entry in catalog.items():
        tenant = entry.get("personas", {}).get("tenant_sre")
        if tenant is None:
            continue
        run.raw_result["verdict"]["root_symptom"]["catalog_id"] = entry_id
        card = bootstrap.run_to_react(run)["diagnose"]
        assert card["hasPersona"] is True, entry_id
        assert card["action"] == tenant.get("action", ""), entry_id
        support = tenant.get("support_details", "")
        expected = [support.strip()] if isinstance(support, str) and support.strip() else support or []
        assert card["supportDetails"] == expected, entry_id
        assert card["catalogId"] == entry_id
        checked += 1
    assert checked > 0


@pytest.mark.parametrize(
    ("support", "expected"),
    [
        (
            "Include the diagnosis (verdict, symptom chain); ask for help even with incomplete evidence.",
            ["Include the diagnosis (verdict, symptom chain); ask for help even with incomplete evidence."],
        ),
        ("1. Include the job ID.\n2. Include available logs.", ["1. Include the job ID.\n2. Include available logs."]),
        ([" Job ID ", "Available logs", "", None, {}], ["Job ID", "Available logs"]),
        (None, []),
        ("", []),
        (" \n ", []),
    ],
)
def test_tenant_support_is_not_split_or_replaced_by_private_guidance(tmp_path, monkeypatch, support, expected):
    from attribution_blueprint.webui import diagnose

    entry = {
        "id": "test-symptom",
        "title": "Test symptom",
        "personas": {
            "tenant_sre": {"action": "Retry only your own workload.", "support_details": support},
            "cloud_operator": {
                "summary": "PRIVATE_OPERATOR_SUMMARY",
                "action": "PRIVATE_OPERATOR_ACTION",
                "support_details": "PRIVATE_OPERATOR_EVIDENCE",
            },
        },
        "internal_notes": {"action": "PRIVATE_INTERNAL_NOTES"},
    }
    monkeypatch.setattr(diagnose, "_load_catalog", lambda: {entry["id"]: entry})
    run = _seed_one(tmp_path, catalog_id=entry["id"])
    for build in (bootstrap.run_summary_to_react, bootstrap.run_to_react):
        payload = build(run)
        assert payload["diagnose"]["action"] == "Retry only your own workload."
        assert payload["diagnose"]["supportDetails"] == expected
        assert "operatorDiagnose" not in payload
        assert "PRIVATE_" not in json.dumps(payload)


def test_missing_tenant_persona_does_not_use_operator_guidance(tmp_path, monkeypatch):
    from attribution_blueprint.webui import diagnose

    monkeypatch.setattr(
        diagnose,
        "_load_catalog",
        lambda: {
            "operator-only": {
                "id": "operator-only",
                "title": "Test symptom",
                "personas": {
                    "cloud_operator": {"action": "PRIVATE_OPERATOR_ACTION", "support_details": "PRIVATE_EVIDENCE"}
                },
                "internal_notes": "PRIVATE_INTERNAL_NOTES",
            }
        },
    )
    payload = bootstrap.run_to_react(_seed_one(tmp_path, catalog_id="operator-only"))
    assert payload["diagnose"]["hasPersona"] is False
    assert payload["diagnose"]["action"] == ""
    assert payload["diagnose"]["supportDetails"] == []
    assert "PRIVATE_" not in json.dumps(payload)


def test_run_to_react_defaults_gpus_per_node_for_historical_rows(tmp_path):
    run = _seed_one(tmp_path, gpus_per_node=None)
    r = bootstrap.run_to_react(run)
    assert r["gpusPerNode"] == 4
    assert r["gpuCount"] == 8


def test_run_to_react_uses_job_gpu_count_when_available(tmp_path):
    run = _seed_one(tmp_path, gpus_per_node=8, gpu_count=6)
    r = bootstrap.run_to_react(run)
    assert r["gpusPerNode"] == 8
    assert r["gpuCount"] == 6


def test_run_to_react_preserves_zero_gpu_count(tmp_path):
    run = _seed_one(tmp_path, gpus_per_node=8, gpu_count=0)
    r = bootstrap.run_to_react(run)
    assert r["gpuCount"] == 0


def test_run_to_react_parses_slurm_alloc_tres_gpu_count(tmp_path):
    run = _seed_one(tmp_path, gpus_per_node=8, alloc_tres="cpu=24,mem=96G,gres/gpu=3")
    r = bootstrap.run_to_react(run)
    assert r["gpuCount"] == 3


def test_run_to_react_sums_typed_slurm_alloc_tres_gpu_counts(tmp_path):
    run = _seed_one(tmp_path, gpus_per_node=8, alloc_tres="cpu=24,gres/gpu:a100=2,gres/gpu:h100=4,mem=96G")
    r = bootstrap.run_to_react(run)
    assert r["gpuCount"] == 6


def test_run_to_react_treats_cpu_only_slurm_alloc_tres_as_zero_gpus(tmp_path):
    run = _seed_one(tmp_path, gpus_per_node=8, alloc_tres="cpu=24,mem=96G")
    r = bootstrap.run_to_react(run)
    assert r["gpuCount"] == 0


def test_log_keeps_distinct_time_sites_with_same_text(tmp_path):
    """A second symptom site with the SAME line text but a DIFFERENT time is
    distinct evidence and must appear alongside the root entry - the dedup only
    suppresses EXACT (line, time) duplicates, not identical-text sites."""
    p = tmp_path / "dup.db"
    out = SQLiteOutput(str(p))
    out.write(
        {
            "id": "s2",
            "sluid": "s2",
            "attempt_id": "s2",
            "workload_type": "slurm",
            "scheduler": "slurm",
            "job_id": "team/job-y",
            "nodes": ["node-a", "node-b"],
            "start_time": datetime(2026, 5, 1, 0, tzinfo=timezone.utc),
            "end_time": datetime(2026, 5, 1, 2, tzinfo=timezone.utc),
            "state": "FAILED",
            "user_name": "alice",
        },
        {
            "verdict": {
                "attribution_outcome": "Fail InfraHW",
                "detail": "GPU off bus",
                "root_symptom": {
                    "symptom": "GpuFell",
                    "source": {"rule": "gpu_fell_off_bus", "line": "Xid 79", "time": "2026-05-01T00:30:00+00:00"},
                },
            },
            "symptoms": {
                "GpuFell": [
                    # Exact duplicate of the root site - must be suppressed.
                    {
                        "type": "NodeSource",
                        "node": "node-a",
                        "source": {"rule": "gpu_fell_off_bus", "line": "Xid 79", "time": "2026-05-01T00:30:00+00:00"},
                    },
                    # Same text, DIFFERENT time (and node) - must be kept.
                    {
                        "type": "NodeSource",
                        "node": "node-b",
                        "source": {"rule": "gpu_fell_off_bus", "line": "Xid 79", "time": "2026-05-01T01:15:00+00:00"},
                    },
                ]
            },
            "model_type": "AttributionResponseV2",
        },
    )
    out.close()
    run = dbmod.get_run(str(p), dbmod.list_runs(str(p))[0].run_id)
    r = bootstrap.run_to_react(run)
    xid_entries = [entry for entry in r["log"] if entry["text"] == "Xid 79"]
    # Two distinct entries: the root (symptom) at 00:30 and the other-time site at 01:15.
    assert len(xid_entries) == 2
    times = sorted(entry["t"] for entry in xid_entries)
    assert times == ["2026-05-01T00:30:00+00:00", "2026-05-01T01:15:00+00:00"]
    # The root entry keeps level 'symptom'; the distinct-time site is 'error'.
    levels = {entry["t"]: entry["level"] for entry in xid_entries}
    assert levels["2026-05-01T00:30:00+00:00"] == "symptom"
    assert levels["2026-05-01T01:15:00+00:00"] == "error"


def test_normalize_outcome_known_values():
    assert bootstrap.normalize_outcome("Fail InfraHW") == "hardware"
    assert bootstrap.normalize_outcome("Fail InfraSW") == "software"
    assert bootstrap.normalize_outcome("Fail MultiCategory") == "multi"
    assert bootstrap.normalize_outcome("Fail User") == "user"
    assert bootstrap.normalize_outcome("Fail Undetermined") == "undetermined"
    assert bootstrap.normalize_outcome("Pass") == "pass"


def test_display_outcome_uses_public_taxonomy_labels():
    assert bootstrap.display_outcome("Fail InfraHW") == "Hardware Fault"
    assert bootstrap.display_outcome("Fail InfraSW") == "System Software Fault"
    assert bootstrap.display_outcome("Fail User") == "Application Fault"
    assert bootstrap.display_outcome("Fail MultiCategory") == "Ambiguous Fault"
    assert bootstrap.display_outcome("Fail Undetermined") == "Unattributed"
    assert bootstrap.display_outcome("Incomplete Info") == "Unattributed"
    assert bootstrap.display_outcome("Incomplete Info - Unregistered Job") == "Unattributed"
    assert bootstrap.display_outcome("Artificial") == "Expected Failure"
    assert bootstrap.display_outcome("") == "Unattributed"
    assert bootstrap.display_outcome(None) == "Unattributed"
    assert bootstrap.display_outcome("Some New Service Outcome") == "Unattributed"


def test_normalize_outcome_unmapped_defaults_to_multi():
    # See plan deviation #2: 'multi' is the gray "other" bucket in the React palette.
    assert bootstrap.normalize_outcome("Incomplete Info") == "multi"
    assert bootstrap.normalize_outcome("") == "multi"
    assert bootstrap.normalize_outcome("Some New Service Outcome") == "multi"


def test_build_payload_shape_and_counts(tmp_path):
    _seed_one(tmp_path)  # seeds 1 row in m.db
    # _seed_one wrote to m.db - rebuild path:
    db_path = os.path.join(str(tmp_path), "m.db")
    payload = bootstrap.build_payload(db_path)
    assert isinstance(payload["runs"], list) and len(payload["runs"]) == 1
    meta = payload["meta"]
    assert meta["windowedRuns"] == 1 and meta["returnedRuns"] == 1
    assert "versions" in meta and "schema" in meta["versions"]
    assert "dateBounds" in meta
    assert meta["database"] == "m.db"


def test_build_payload_empty_db(tmp_path):
    from attribution_blueprint.connectors.outputs.sqlite import SQLiteOutput

    p = tmp_path / "empty.db"
    out = SQLiteOutput(str(p))
    out._connect()
    out.close()
    payload = bootstrap.build_payload(str(p))
    assert payload["runs"] == []
    assert payload["meta"]["windowedRuns"] == 0
    assert payload["meta"]["dateBounds"] == {"min": None, "max": None}


def test_inject_escapes_script_breakout():
    html = '<html><body><div id="root"></div></body></html>'
    payload = {"runs": [{"log": [{"text": "</script><img src=x onerror=alert(1)>"}]}], "meta": {"returnedRuns": 1}}
    out = bootstrap.inject(html, payload)
    # No literal </script> from the data may appear before the element closes.
    assert "</script><img" not in out
    assert "\\u003c/script\\u003e" in out or "\\u003C/script\\u003E".lower() in out.lower()
    # data-total-runs marker set on #root
    assert 'data-total-runs="1"' in out
    # The blob round-trips back to the original characters.
    start = out.index('id="attr-bootstrap">') + len('id="attr-bootstrap">')
    end = out.index("</script>", start)
    restored = json.loads(out[start:end].replace("\\u003c", "<").replace("\\u003e", ">").replace("\\u0026", "&"))
    assert restored["runs"][0]["log"][0]["text"] == "</script><img src=x onerror=alert(1)>"


def test_run_to_react_passes_compound_outcome_fields(tmp_path):
    p = tmp_path / "m.db"
    out = SQLiteOutput(str(p))
    ctx = {
        "id": "s1",
        "sluid": "s1",
        "attempt_id": "s1",
        "workload_type": "slurm",
        "scheduler": "slurm",
        "job_id": "job-1",
        "nodes": ["node-a"],
        "start_time": datetime(2026, 5, 1, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 1, tzinfo=timezone.utc),
        "state": "FAILED",
        "user_name": "alice",
        "gpus_per_node": 8,
    }
    out.write(
        ctx,
        {
            "verdict": {
                "attribution_outcome": "Fail InfraHW",
                "detail": "",
                "root_symptom": {"symptom": "GpuFell", "source": {"rule": "r", "line": "l", "time": "2026-05-01T00:30:00+00:00"}},
                "terminal_states": ["Fail"],
                "failure_domains": ["InfraHW"],
                "cancellation_actors": [],
                "uncertainty_type": None,
            },
            "model_type": "AttributionResponseV2",
            "connector_reports": [{"connector": "c", "source": "syslog", "status": "ok", "reason": "", "observation_count": 1}],
            "ingestion_reports": [{"connector": "c", "source": "syslog", "status": "error", "attempted": 1, "failed": 1, "reason": "HTTP 500"}],
            "evidence_status": "partial",
        },
    )
    out.close()
    run = dbmod.get_run(str(p), dbmod.list_runs(str(p))[0].run_id)
    r = bootstrap.run_to_react(run)

    assert r["terminalStates"] == ["Fail"]
    assert r["failureDomains"] == ["InfraHW"]
    assert r["cancellationActors"] == []
    assert r["uncertaintyType"] is None
    assert r["evidenceStatus"] == "partial"
    assert r["ingestionReports"] == {
        "connector_reports": [{"connector": "c", "source": "syslog", "status": "ok", "reason": "", "observation_count": 1}],
        "ingestion_reports": [{"connector": "c", "source": "syslog", "status": "error", "attempted": 1, "failed": 1, "reason": "HTTP 500"}],
    }


def test_run_to_react_defaults_compound_fields_when_absent(tmp_path):
    run = _seed_one(tmp_path)
    # Rows saved before ingestion reports were recorded still carry connector_reports.
    run.raw_result["connector_reports"] = [
        {"connector": "c", "source": "syslog", "status": "ok", "reason": "", "observation_count": 1}
    ]
    r = bootstrap.run_to_react(run)

    assert r["terminalStates"] == []
    assert r["failureDomains"] == []
    assert r["cancellationActors"] == []
    assert r["uncertaintyType"] is None
    assert r["evidenceStatus"] is None
    assert r["ingestionReports"] is None
