# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from attribution_blueprint.orchestrator import AttributionOrchestrator
from attribution_blueprint.types import Format, Source


def _make_ctx(ctx_id: str) -> dict:
    return {
        "id": ctx_id,
        "workload_type": "slurm",
        "nodes": [f"node-{ctx_id}-1", f"node-{ctx_id}-2"],
        "start_time": datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc),
        "job_id": ctx_id,
        "sluid": "",
        "parent_job_id": ctx_id,
        "state": "COMPLETED",
    }


def _make_obs(source: str = "SYSLOG") -> dict:
    return {
        "context": {
            "source": source,
            "format": "RAW_LOG_LINES",
            "resources": "",
            "time_interval": {"start": "2026-05-01T00:00:00Z", "end": "2026-05-01T01:00:00Z"},
        },
        "body": ["line"],
    }


def _make_attribution_mock():
    attr = MagicMock()
    attr.create_attributor.side_effect = lambda info: f"attr-{info['workload']['id']}"
    attr.create_or_replace_attributor.side_effect = lambda info: attr.create_attributor(info)
    attr.ingest_observation.return_value = 1
    attr.get_attribution.return_value = {"verdict": {"attribution_outcome": "Fail User"}}
    attr.delete_attributor.return_value = None
    return attr


class _ListOutput:
    def __init__(self):
        self.entries: list[tuple[dict, dict]] = []

    def write(self, ctx, attribution):
        self.entries.append((ctx, attribution))

    def close(self):
        pass


def _make_pipeline(eca, log_collectors, attribution=None, output=None, outputs=None):
    if outputs is None:
        outputs = [output] if output is not None else [_ListOutput()]
    return AttributionOrchestrator(
        execution_context_adapters=[eca],
        log_connectors=log_collectors,
        metrics_connectors=[],
        attribution=attribution or _make_attribution_mock(),
        outputs=outputs,
    )


def test_analyze_full_flow_calls_per_context_and_per_collector():
    contexts = [_make_ctx("1"), _make_ctx("2"), _make_ctx("3")]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts

    coll_a = MagicMock(source="APPLOG")
    coll_a.collect.return_value = [_make_obs("APPLOG")]
    coll_b = MagicMock(source="SYSLOG")
    coll_b.collect.return_value = [_make_obs("SYSLOG")]

    attr = _make_attribution_mock()
    output = _ListOutput()
    pipeline = _make_pipeline(eca, [coll_a, coll_b], attribution=attr, output=output)

    results = pipeline.analyze(ids=["1", "2", "3"])

    assert attr.create_attributor.call_count == 3
    assert attr.ingest_observation.call_count == 6
    assert attr.get_attribution.call_count == 3
    assert len(output.entries) == 3
    assert len(results) == 3
    assert all(r["verdict"]["attribution_outcome"] == "Fail User" for r in results)
    assert all(r["terminal_state"] == "fail" for r in results)
    assert all(r["sub_domain"] == "application" for r in results)
    assert results[0]["connector_reports"] == [
        {"connector": "MagicMock", "source": "APPLOG", "status": "ok", "reason": "", "observation_count": 1},
        {"connector": "MagicMock", "source": "SYSLOG", "status": "ok", "reason": "", "observation_count": 1},
    ]


def test_analyze_records_run_summaries():
    ctx = _make_ctx("1")
    ctx["attempt_id"] = "0"
    eca = MagicMock()
    eca.discover_by_ids.return_value = [ctx]
    attr = _make_attribution_mock()
    attr.get_attribution.return_value = {
        "verdict": {
            "attribution_outcome": "Fail InfraHW",
            "root_symptom": {"symptom": "XID-79"},
        },
        "symptoms": {"XID-79": []},
    }

    pipeline = _make_pipeline(eca, [], attribution=attr)
    pipeline.analyze(ids=["1"])

    assert pipeline.last_run_summaries == [
        {
            "context_id": "1",
            "job_id": "1",
            "attempt_id": "0",
            "terminal_state": "fail",
            "sub_domain": "hardware",
            "root_symptom": "XID-79",
            "symptoms": "XID-79",
        }
    ]


# Required keys per Attribution Service/app/v2/models.py:
#   AttributorWorkloadInfoV2 (workload) and AttributorMetadata (metadata).
# Keep these in lockstep with the service Pydantic models. A regression
# here means a 422 in production.
_REQUIRED_WORKLOAD_KEYS = {
    "id",
    "type",
    "job_start_time",
    "timezone",
    "status",
    "nodes",
    "name",
    "username",
    "framework",
}
_REQUIRED_METADATA_KEYS = {"cluster", "agent", "tenant", "ruleset"}


def test_build_attributor_info_payload_has_all_required_workload_keys():
    """A regression that drops a required field would silently 422 at the
    service. Lock the contract here so it fails in CI, not production."""
    ctx = _make_ctx("contract-1")
    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])
    info = pipeline._build_attributor_info(ctx)
    missing = _REQUIRED_WORKLOAD_KEYS - set(info["workload"])
    assert not missing, f"workload missing required keys: {missing}"


def test_build_attributor_info_payload_has_all_required_metadata_keys():
    ctx = _make_ctx("contract-2")
    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])
    info = pipeline._build_attributor_info(ctx)
    missing = _REQUIRED_METADATA_KEYS - set(info["metadata"])
    assert not missing, f"metadata missing required keys: {missing}"


def test_build_attributor_info_payload_required_fields_have_concrete_types():
    """Required fields must match the v2 service wire schema."""
    ctx = _make_ctx("contract-3")
    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])
    info = pipeline._build_attributor_info(ctx)
    w = info["workload"]
    assert isinstance(w["id"], str) and w["id"]
    assert isinstance(w["nodes"], dict)
    assert all(isinstance(node, str) for node in w["nodes"])
    assert all(isinstance(ranks, list) for ranks in w["nodes"].values())
    assert all(isinstance(rank, int) for ranks in w["nodes"].values() for rank in ranks)
    assert isinstance(w["job_start_time"], str)  # ISO; service parses to AwareDatetime
    assert isinstance(w["timezone"], str) and w["timezone"]
    # ISO must carry an offset so Pydantic's AwareDatetime accepts it.
    assert "+" in w["job_start_time"] or "Z" in w["job_start_time"]


def test_build_attributor_info_omits_legacy_straggler_parameters():
    """The service's AttributionParameters forbids unknown fields, so sending
    the legacy straggler parameter keys 422s every request. The payload must
    omit `parameters` entirely; the service defaults it."""
    ctx = _make_ctx("contract-4")
    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])
    info = pipeline._build_attributor_info(ctx)
    assert "parameters" not in info


def test_build_attributor_info_rejects_naive_start_time():
    """The service's AttributorWorkloadInfo.job_start_time is AwareDatetime;
    a naive datetime would 422 the request. Fail fast at the BP boundary so
    the caller sees the problem at construction, not at HTTP time."""
    ctx = _make_ctx("naive-1")
    ctx["start_time"] = datetime(2026, 5, 1, 0, 0, 0)  # naive

    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])
    with pytest.raises(ValueError, match="tz-aware"):
        pipeline._build_attributor_info(ctx)


def test_build_attributor_info_rejects_naive_end_time():
    ctx = _make_ctx("naive-2")
    ctx["end_time"] = datetime(2026, 5, 1, 1, 0, 0)  # naive

    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])
    with pytest.raises(ValueError, match="tz-aware"):
        pipeline._build_attributor_info(ctx)


@pytest.mark.parametrize(
    ("tz", "expected"),
    [
        (timezone.utc, "UTC"),
        (ZoneInfo("America/Los_Angeles"), "America/Los_Angeles"),
        (timezone(timedelta(hours=-7)), "Etc/GMT+7"),
        (timezone(timedelta(hours=5)), "Etc/GMT-5"),
        (timezone(timedelta(hours=14)), "Etc/GMT-14"),
    ],
)
def test_build_attributor_info_sends_service_timezone(tz, expected):
    """The payload supplies a ZoneInfo key accepted by the service."""
    ctx = _make_ctx("timezone")
    ctx["start_time"] = ctx["start_time"].replace(tzinfo=tz)
    ctx["end_time"] = ctx["end_time"].replace(tzinfo=tz)

    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])

    assert pipeline._build_attributor_info(ctx)["workload"]["timezone"] == expected


def test_build_attributor_info_uses_explicit_timezone_for_non_hour_offset():
    ctx = _make_ctx("timezone-explicit")
    fixed_offset = timezone(timedelta(hours=5, minutes=30))
    ctx["start_time"] = ctx["start_time"].replace(tzinfo=fixed_offset)
    ctx["end_time"] = ctx["end_time"].replace(tzinfo=fixed_offset)
    ctx["timezone"] = "Asia/Kolkata"

    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])

    assert pipeline._build_attributor_info(ctx)["workload"]["timezone"] == "Asia/Kolkata"


def test_build_attributor_info_rejects_non_hour_offset_without_timezone():
    ctx = _make_ctx("timezone-non-hour")
    fixed_offset = timezone(timedelta(hours=5, minutes=30))
    ctx["start_time"] = ctx["start_time"].replace(tzinfo=fixed_offset)
    ctx["end_time"] = ctx["end_time"].replace(tzinfo=fixed_offset)

    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])

    with pytest.raises(ValueError, match="cannot be represented"):
        pipeline._build_attributor_info(ctx)


def test_build_attributor_info_rejects_fixed_offset_change_without_timezone():
    ctx = _make_ctx("timezone-offset-change")
    ctx["start_time"] = ctx["start_time"].replace(tzinfo=timezone(timedelta(hours=-7)))
    ctx["end_time"] = ctx["end_time"].replace(tzinfo=timezone(timedelta(hours=-8)))

    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])

    with pytest.raises(ValueError, match="spans a UTC offset change"):
        pipeline._build_attributor_info(ctx)


def test_build_attributor_info_preserves_zoneinfo_across_dst_change():
    ctx = _make_ctx("timezone-dst")
    zone = ZoneInfo("America/Los_Angeles")
    ctx["start_time"] = datetime(2026, 3, 8, 1, 30, tzinfo=zone)
    ctx["end_time"] = datetime(2026, 3, 8, 3, 30, tzinfo=zone)

    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])

    assert pipeline._build_attributor_info(ctx)["workload"]["timezone"] == "America/Los_Angeles"


def test_build_attributor_info_rejects_invalid_explicit_timezone():
    ctx = _make_ctx("timezone-invalid")
    ctx["timezone"] = "Not/A/Zone"
    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])

    with pytest.raises(ValueError, match="valid IANA timezone key"):
        pipeline._build_attributor_info(ctx)


def test_build_attributor_info_rejects_unavailable_fixed_offset_zone():
    ctx = _make_ctx("timezone-unavailable")
    fixed_offset = timezone(timedelta(hours=-13))
    ctx["start_time"] = ctx["start_time"].replace(tzinfo=fixed_offset)
    ctx["end_time"] = ctx["end_time"].replace(tzinfo=fixed_offset)

    pipeline = _make_pipeline(eca=MagicMock(), log_collectors=[])

    with pytest.raises(ValueError, match="has no IANA fixed-offset zone"):
        pipeline._build_attributor_info(ctx)


def test_analyze_continues_when_get_attribution_raises_for_one_ctx():
    """One ExecutionContext's 500 must not abort the rest of the batch - see
    `attribution_lib` events.py datetime serialization bug for a real-world
    trigger. The bad context's attributor is still cleaned up."""
    import requests

    contexts = [_make_ctx("1"), _make_ctx("2"), _make_ctx("3")]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts

    obs = _make_obs()
    coll = MagicMock(source="APPLOG")
    coll.collect.return_value = [obs]

    attr = _make_attribution_mock()
    attr.get_attribution.side_effect = [
        {"verdict": {"attribution_outcome": "Fail User"}},
        requests.HTTPError("500"),
        {"verdict": {"attribution_outcome": "Fail User"}},
    ]

    output = _ListOutput()
    pipeline = _make_pipeline(eca, [coll], attribution=attr, output=output)
    results = pipeline.analyze(ids=["1", "2", "3"])

    assert attr.get_attribution.call_count == 3
    assert attr.delete_attributor.call_count == 3
    assert len(output.entries) == 2
    assert [r["verdict"]["attribution_outcome"] for r in results] == ["Fail User", "Fail User"]


def test_analyze_skips_failing_collector_other_contexts_unaffected():
    contexts = [_make_ctx("1"), _make_ctx("2"), _make_ctx("3")]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts

    obs = _make_obs()
    coll = MagicMock(source="APPLOG")
    coll.collect.side_effect = [[obs], Exception("boom"), [obs]]

    attr = _make_attribution_mock()
    pipeline = _make_pipeline(eca, [coll], attribution=attr)

    results = pipeline.analyze(ids=["1", "2", "3"])

    assert attr.create_attributor.call_count == 3
    assert attr.ingest_observation.call_count == 2
    assert attr.get_attribution.call_count == 3
    assert results[1]["connector_reports"] == [
        {"connector": "MagicMock", "source": "APPLOG", "status": "error", "reason": "boom", "observation_count": 0}
    ]


def test_dump_observations_writes_files(tmp_path):
    import json

    ctx = _make_ctx("ctxA")
    eca = MagicMock()
    eca.discover_by_ids.return_value = [ctx]

    obs = {
        "context": {
            "source": "syslog",
            "format": "loki_dict_list",
            "resources": "AllJobResources",
            "time_interval": {"start": ctx["start_time"], "end": ctx["end_time"]},
        },
        "body": json.dumps([{"timestamp": 1, "node": "n1", "body": "kernel: oops"}]),
    }
    coll = MagicMock(source="syslog")
    coll.collect.return_value = [obs]

    pipeline = AttributionOrchestrator(
        execution_context_adapters=[eca],
        log_connectors=[coll],
        metrics_connectors=[],
        attribution=_make_attribution_mock(),
        outputs=[_ListOutput()],
        dump_observations_dir=tmp_path,
    )
    pipeline.analyze(ids=["ctxA"])

    dumped = list(tmp_path.iterdir())
    assert len(dumped) == 1
    assert dumped[0].name == "ctxA.syslog.json"
    payload = json.loads(dumped[0].read_text())
    assert payload["context"]["source"] == "syslog"
    assert payload["body"][0]["body"] == "kernel: oops"


def test_analyze_uses_time_range_when_no_ids():
    eca = MagicMock()
    eca.discover_by_time_range.return_value = []

    pipeline = _make_pipeline(eca, [])
    since = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    until = datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc)
    nodelist = ["node1", "node2"]

    pipeline.analyze(since=since, until=until, nodelist=nodelist)

    eca.discover_by_time_range.assert_called_once_with(since, until, nodelist)
    eca.discover_by_ids.assert_not_called()


def test_analyze_uses_connector_discover_when_no_runtime_selector():
    contexts = [_make_ctx("static-1")]
    eca = MagicMock()
    eca.discover.return_value = contexts

    attr = _make_attribution_mock()
    output = _ListOutput()
    pipeline = _make_pipeline(eca, [], attribution=attr, output=output)

    results = pipeline.analyze()

    eca.discover.assert_called_once_with()
    eca.discover_by_ids.assert_not_called()
    eca.discover_by_time_range.assert_not_called()
    assert len(output.entries) == 1
    assert results[0]["verdict"]["attribution_outcome"] == "Fail User"
    assert results[0]["terminal_state"] == "fail"


def test_build_attributor_info_shape():
    ctx = _make_ctx("ctx-7")
    pipeline = _make_pipeline(MagicMock(), [])

    info = pipeline._build_attributor_info(ctx)

    assert info["workload"]["id"] == "ctx-7"
    assert info["workload"]["type"] == "slurm"
    assert info["workload"]["status"] == "COMPLETED"
    assert info["workload"]["nodes"] == {
        "node-ctx-7-1": [0, 1, 2, 3],
        "node-ctx-7-2": [4, 5, 6, 7],
    }
    assert info["workload"]["job_start_time"] == ctx["start_time"].isoformat()
    assert info["workload"]["timezone"] == "UTC"
    assert info["workload"]["job_end_time"] == ctx["end_time"].isoformat()
    assert info["metadata"]["agent"] == "attribution-blueprint"
    # Wire format requires `cluster`; we emit empty string. The blueprint
    # itself does not carry cluster as a data-model concept.
    assert info["metadata"]["cluster"] == ""


def test_build_attributor_info_uses_context_workload_type():
    ctx = _make_ctx("ctx-7")
    ctx["workload_type"] = "runai"
    pipeline = _make_pipeline(MagicMock(), [])

    info = pipeline._build_attributor_info(ctx)

    assert info["workload"]["type"] == "runai"


def test_build_attributor_info_missing_workload_type_is_undefined():
    ctx = _make_ctx("ctx-7")
    ctx.pop("workload_type")
    pipeline = _make_pipeline(MagicMock(), [])

    info = pipeline._build_attributor_info(ctx)

    assert info["workload"]["type"] == ""


def test_wire_workload_type_unaffected_by_scheduler_field():
    """The BP-internal `scheduler` field labels SQLite rows and must NOT
    leak onto the wire - the orchestrator only reads `workload_type` for
    the service payload. Kueue contexts set `workload_type=""` (UNDEFINED)
    and `scheduler="kueue"`; the outbound `workload.type` is still ""."""
    ctx = _make_ctx("ctx-7")
    ctx["workload_type"] = ""
    ctx["scheduler"] = "kueue"
    pipeline = _make_pipeline(MagicMock(), [])

    info = pipeline._build_attributor_info(ctx)

    assert info["workload"]["type"] == ""
    # And the scheduler field doesn't appear under any other key in the payload.
    assert "scheduler" not in info["workload"]


def test_build_attributor_info_honors_gpus_per_node():
    ctx = _make_ctx("ctx-7")
    pipeline = AttributionOrchestrator(
        execution_context_adapters=[MagicMock()],
        log_connectors=[],
        metrics_connectors=[],
        attribution=_make_attribution_mock(),
        outputs=[_ListOutput()],
        gpus_per_node=8,
    )

    info = pipeline._build_attributor_info(ctx)

    assert info["workload"]["nodes"] == {
        "node-ctx-7-1": [0, 1, 2, 3, 4, 5, 6, 7],
        "node-ctx-7-2": [8, 9, 10, 11, 12, 13, 14, 15],
    }


def test_build_attributor_info_rejects_non_positive_gpus_per_node():
    ctx = _make_ctx("ctx-cpu")
    pipeline = AttributionOrchestrator(
        execution_context_adapters=[MagicMock()],
        log_connectors=[],
        metrics_connectors=[],
        attribution=_make_attribution_mock(),
        outputs=[_ListOutput()],
        gpus_per_node=0,
    )

    with pytest.raises(ValueError, match="gpus_per_node"):
        pipeline._build_attributor_info(ctx)


def test_outputs_receive_workload_gpu_defaults():
    contexts = [_make_ctx("ctx-gpu")]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts
    output = _ListOutput()
    pipeline = AttributionOrchestrator(
        execution_context_adapters=[eca],
        log_connectors=[],
        metrics_connectors=[],
        attribution=_make_attribution_mock(),
        outputs=[output],
        gpus_per_node=8,
    )

    pipeline.analyze(ids=["ctx-gpu"])

    output_ctx = output.entries[0][0]
    assert output_ctx["gpus_per_node"] == 8
    assert output_ctx["gpu_count"] == 16


def test_outputs_preserve_scheduler_gpu_count():
    ctx = _make_ctx("ctx-partial")
    ctx["gpu_count"] = 6
    contexts = [ctx]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts
    output = _ListOutput()
    pipeline = AttributionOrchestrator(
        execution_context_adapters=[eca],
        log_connectors=[],
        metrics_connectors=[],
        attribution=_make_attribution_mock(),
        outputs=[output],
        gpus_per_node=8,
    )

    pipeline.analyze(ids=["ctx-partial"])

    output_ctx = output.entries[0][0]
    assert output_ctx["gpus_per_node"] == 8
    assert output_ctx["gpu_count"] == 6


def test_failing_ingest_does_not_block_other_observations():
    contexts = [_make_ctx("1")]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts

    obs = _make_obs()
    coll = MagicMock(source="APPLOG")
    coll.collect.return_value = [obs, obs, obs]

    attr = _make_attribution_mock()
    attr.ingest_observation.side_effect = [1, Exception("boom"), 3]

    pipeline = _make_pipeline(eca, [coll], attribution=attr)
    pipeline.analyze(ids=["1"])

    assert attr.ingest_observation.call_count == 3
    assert attr.get_attribution.call_count == 1


def test_raw_syslog_is_adapted_before_service_ingest():
    contexts = [_make_ctx("1")]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts

    obs = {
        "context": {
            "source": Source.SYSLOG,
            "format": Format.RAW_LOG,
            "resources": "node-1-1",
            "time_interval": {
                "start": contexts[0]["start_time"],
                "end": contexts[0]["end_time"],
            },
        },
        "body": "2026-05-01T00:00:03Z node-1-1 kernel\n",
    }
    coll = MagicMock(source="SYSLOG")
    coll.collect.return_value = [obs]

    attr = _make_attribution_mock()
    pipeline = _make_pipeline(eca, [coll], attribution=attr)
    pipeline.analyze(ids=["1"])

    ingested = attr.ingest_observation.call_args.args[1]
    assert obs["context"]["format"] == Format.RAW_LOG
    assert ingested["context"]["format"] == Format.LOKI_DICT_LIST


@pytest.mark.parametrize("ids", [["1"], ["1", "2"]])
def test_analyze_returns_results_in_context_order(ids):
    contexts = [_make_ctx(i) for i in ids]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts

    attr = _make_attribution_mock()
    attr.get_attribution.side_effect = [{"v": ctx_id} for ctx_id in ids]

    pipeline = _make_pipeline(eca, [], attribution=attr)
    results = pipeline.analyze(ids=ids)

    assert [r["v"] for r in results] == ids
    assert [r["terminal_state"] for r in results] == ["unknown" for _ in ids]


def test_build_attributor_info_username_reads_user_name_field():
    """The wire payload's workload.username must be populated from ctx['user_name'],
    not ctx['username'] - ECAs write user_name."""
    ctx = _make_ctx("ctx-7")
    ctx["user_name"] = "alice"
    pipeline = _make_pipeline(MagicMock(), [])

    info = pipeline._build_attributor_info(ctx)

    assert info["workload"]["username"] == "alice"


def test_build_attributor_info_framework_threads_through():
    """workload.framework on the wire is read from ctx['workload_framework'] when
    populated by the ECA. The wire field stays `framework` (service contract);
    the BP-internal ctx key is `workload_framework` to avoid colliding with the
    MCA connector-framework concept."""
    ctx = _make_ctx("ctx-7")
    ctx["workload_framework"] = "megatron"
    pipeline = _make_pipeline(MagicMock(), [])

    info = pipeline._build_attributor_info(ctx)

    assert info["workload"]["framework"] == "megatron"


def test_output_write_failure_does_not_abort_batch():
    """If one context's output.write raises, the other contexts still process."""
    contexts = [_make_ctx("1"), _make_ctx("2"), _make_ctx("3")]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts

    flaky_output = MagicMock()
    # Second write raises; first and third must still be attempted.
    flaky_output.write.side_effect = [None, RuntimeError("disk full"), None]

    pipeline = _make_pipeline(eca, [], output=flaky_output)
    results = pipeline.analyze(ids=["1", "2", "3"])

    assert flaky_output.write.call_count == 3
    # All three verdicts still returned (output failure logged, not propagated).
    assert len(results) == 3
    assert pipeline.last_run_stats == {"attempted": 3, "succeeded": 2}


def test_all_output_write_failures_do_not_count_as_success():
    """A verdict that reaches no output sink must not make the CLI think the
    context succeeded."""
    contexts = [_make_ctx("1"), _make_ctx("2")]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts

    failing_output = MagicMock()
    failing_output.write.side_effect = RuntimeError("disk full")

    pipeline = _make_pipeline(eca, [], output=failing_output)
    results = pipeline.analyze(ids=["1", "2"])

    assert len(results) == 2
    assert failing_output.write.call_count == 2
    assert pipeline.last_run_stats == {"attempted": 2, "succeeded": 0}


def test_one_failing_sink_does_not_skip_other_sinks_for_same_ctx():
    """Per-sink try/except: if sink B raises for ctx X, sinks A and C must still
    receive write(ctx X). This is the within-ctx counterpart to the across-ctx
    isolation tested above."""
    ctx = _make_ctx("1")
    eca = MagicMock()
    eca.discover_by_ids.return_value = [ctx]

    ok_a = MagicMock()
    raising = MagicMock()
    raising.write.side_effect = RuntimeError("boom")
    ok_c = MagicMock()

    pipeline = _make_pipeline(eca, [], outputs=[ok_a, raising, ok_c])
    results = pipeline.analyze(ids=["1"])

    assert ok_a.write.call_count == 1
    assert raising.write.call_count == 1
    assert ok_c.write.call_count == 1
    assert len(results) == 1
    assert pipeline.last_run_stats == {"attempted": 1, "succeeded": 1}


def test_analyze_continues_when_create_attributor_raises_non_409():
    """A create failure for one ctx must not abort the batch."""
    contexts = [_make_ctx("1"), _make_ctx("2"), _make_ctx("3")]
    eca = MagicMock()
    eca.discover_by_ids.return_value = contexts

    attr = _make_attribution_mock()
    # Second context's create fails hard; first and third must still process.
    attr.create_attributor.side_effect = [
        "attr-1",
        RuntimeError("boom"),
        "attr-3",
    ]

    output = _ListOutput()
    pipeline = _make_pipeline(eca, [], attribution=attr, output=output)
    results = pipeline.analyze(ids=["1", "2", "3"])

    # Two successful contexts; the failed one yields no result entry.
    assert len(results) == 2
    assert len(output.entries) == 2


def test_dump_observations_tolerates_non_json_body(tmp_path):
    """RAW_LOG bodies are plain text, not JSON; _dump_observations must round-trip them
    under a `body_raw` key rather than crashing with JSONDecodeError."""
    import json as _json

    ctx = _make_ctx("dumpme")
    eca = MagicMock()
    eca.discover_by_ids.return_value = [ctx]

    raw_text_obs = {
        "context": {
            "source": "APPLOG",
            "format": "RAW_LOG",
            "resources": "",
            "time_interval": {"start": ctx["start_time"], "end": ctx["end_time"]},
        },
        "body": "line1\nline2 not json\n",
    }
    coll = MagicMock(source="APPLOG")
    coll.collect.return_value = [raw_text_obs]

    pipeline = AttributionOrchestrator(
        execution_context_adapters=[eca],
        log_connectors=[coll],
        metrics_connectors=[],
        attribution=_make_attribution_mock(),
        outputs=[_ListOutput()],
        dump_observations_dir=tmp_path,
    )
    pipeline.analyze(ids=["dumpme"])

    dump_path = tmp_path / "dumpme.APPLOG.json"
    assert dump_path.exists()
    payload = _json.loads(dump_path.read_text())
    assert payload["body_raw"] == "line1\nline2 not json\n"
    assert "body" not in payload  # the JSON-decode key is absent for raw bodies
