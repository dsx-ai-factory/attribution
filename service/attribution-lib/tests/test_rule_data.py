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
import warnings
from datetime import datetime, timedelta

import pytest

from attribution_lib.core import process_check
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.observation.inbound import (
    Format as ObservationFormat,
    Source as ObservationSource,
)
from attribution_lib.observation.processing import render_metric_body
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.sources import Source
from test_support.helpers import (
    make_node_to_ranks,
    make_syslog_entry,
    match_bad,
    match_bad_peers,
    match_interesting,
)
from tests import rule_data_runner
from tests.rule_data_runner import RuleDataCase, iter_rule_data_cases

RULE_DATA_CASES = iter_rule_data_cases()
MIN_RULE_DATA_CASE_COUNT = 3900


@pytest.fixture(autouse=True)
def _preload_rule_data_test_rules(event_manager):
    event_manager.preload_rule_names(
        {"Synthetic MFU", "Synthetic PyTorch NCCL watchdog", "Synthetic dataloader exception"}
    )


def _make_node_metric(metric_name: str, value: str | int | float, timestamp: int, node: str, component: str) -> dict:
    return {
        "timestamp": timestamp,
        "value": str(value),
        "metric": metric_name,
        "labels": {"net_host_name": node, "device": component, "s_cluster": "test-cluster"},
        "node": node,
        "component": component,
        "cluster": "test-cluster",
    }


def _run_node_metrics_case(case_data: RuleDataCase, attribution_test_builder, simple_job_factory):
    node = case_data.case.get("node", "test-node-001")
    component = case_data.case.get("component", "mlx5_0")
    job_info = simple_job_factory(
        nodes=[node],
        cluster="test-cluster",
        source_name="test_slurm",
        start_timestamp=1744734400,
        end_timestamp=1744734500,
    )
    timestamp = case_data.case.get("timestamp", 1744734450)
    metric = _make_node_metric(case_data.case["metric"], case_data.case["value"], timestamp, node, component)

    # Under entry_timeseries the producer renders the body and stamps nanoseconds; the rules must
    # read the same verdict off either wire form.
    wire_forms = [
        (ObservationFormat.PROMETHEUS_METRICS, metric),
        (
            ObservationFormat.ENTRY_TIMESERIES,
            {
                "ts": timestamp * 1_000_000_000,
                "resources": [{"kind": "node", "hostname": node}],
                "body": render_metric_body(metric),
            },
        ),
    ]
    for format, wire in wire_forms:
        result = attribution_test_builder(
            job_info=job_info,
            data=[wire],
            source=ObservationSource.NODE_METRICS,
            format=format,
            check_fn=process_check,
        )
        node_attr = result.get_node_attributions()

        match case_data.case["expect"]:
            case "bad":
                match_bad(node_attr, node, case_data.case["expected_symptom"], case_data.case["expected_rule"])
            case "interesting":
                match_interesting(node_attr, node, case_data.case["expected_symptom"], case_data.case["expected_rule"])
            case "no_match":
                assert node not in node_attr.attributions or not node_attr.attributions[node].attributions._data
            case other:
                raise ValueError(f"Unsupported expectation {other!r} in {case_data.path}")


def _make_nvlink_metric(metric: str, value: float, timestamp: int, node: str, component: str, cluster: str) -> dict:
    return {
        "timestamp": timestamp,
        "value": str(value),
        "metric": metric,
        "labels": {"net_host_name": node, "device": component, "s_cluster": cluster},
        "node": node,
        "component": component,
        "cluster": cluster,
    }


def _run_nvlink_metrics_case(case_data: RuleDataCase, attribution_test_builder, simple_job_factory):
    node = case_data.case.get("node", "test-node-001")
    component = case_data.case.get("component", "nvswitch0")
    cluster = case_data.case.get("cluster", "host-c771bdf0")
    job_info = simple_job_factory(
        nodes=[node],
        cluster=cluster,
        source_name="test_slurm",
        start_timestamp=1744734400,
        end_timestamp=1744734500,
    )
    timestamp = case_data.case.get("timestamp", 1744734450)
    metric = _make_nvlink_metric(case_data.case["metric"], case_data.case["value"], timestamp, node, component, cluster)

    result = attribution_test_builder(
        job_info=job_info,
        data=[metric],
        source=ObservationSource.NVLINK_METRICS,
        format=ObservationFormat.PROMETHEUS_METRICS,
        check_fn=process_check,
    )
    node_attr = result.get_node_attributions()

    match case_data.case["expect"]:
        case "bad":
            match_bad(node_attr, node, case_data.case["expected_symptom"], case_data.case["expected_rule"])
        case "interesting":
            match_interesting(node_attr, node, case_data.case["expected_symptom"], case_data.case["expected_rule"])
        case "no_match":
            assert node not in node_attr.attributions or not node_attr.attributions[node].attributions._data
        case other:
            raise ValueError(f"Unsupported expectation {other!r} in {case_data.path}")


def _build_syslog_jobinfo(timezone, *, tflops_start: bool, multi_gpu: bool = True):
    start = datetime.fromtimestamp(1741860016, tz=timezone)
    end = datetime.fromtimestamp(1744734448, tz=timezone)
    tflops_time = start if tflops_start else end
    nodes = ["host-08b5fda3", "host-359412e9"] if multi_gpu else ["host-08b5fda3"]
    return (
        JobInfoBuilder(
            id="1887602",
            cluster="cluster_b722552b",
            state="CANCELLED",
            type="training",
            framework="megatron",
            node_to_ranks=make_node_to_ranks(nodes, 8 if multi_gpu else 1),
            job_start_time=start,
            source_name="artifical slurm",
            timezone=timezone,
        )
        .srun_cmd("srun test.py")
        .exit_code_signal(15)
        .job_end_time(end)
        .last_tflops_time(tflops_time)
        .time_limit(6000)
        .build()
    )


def _make_mfu_event(job_info, tflops_start: bool, obs_data):
    time = job_info.job_start_time if tflops_start else job_info.job_end_time
    obase = obs_data(job_info, [], ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES)
    src = Source(
        source_context=obase.context,
        line_num=None,
        datum="<pseudo mfu>",
        time=time,
        resources=AllJobResources,
        rule="Synthetic MFU",
    )
    return ("Application.mfu", {"datetime": time}, src)


def _make_syslog_aux_event(job_info, event_spec: dict, obs_data):
    obase = obs_data(job_info, [], ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES)
    event_type = event_spec["event_type"]
    line_num = event_spec.get("line_num", 10)
    datum = event_spec.get("datum", "<pseudo event>")
    source_rule = event_spec.get("source_rule")

    time_kind = event_spec.get("time_kind")
    time = None
    if time_kind == "job_start":
        time = job_info.job_start_time
    elif time_kind == "job_end":
        time = job_info.job_end_time
    elif time_kind == "none" or time_kind is None:
        time = None
    elif time_kind == "offset_from_job_start":
        time = job_info.job_start_time + timedelta(seconds=event_spec["offset_seconds"])
    else:
        raise ValueError(f"Unsupported syslog aux event time kind {time_kind!r}")

    match event_type:
        case "Application.mfu":
            if source_rule is None:
                source_rule = "Synthetic MFU"
            if time is None:
                raise ValueError("Application.mfu requires a time")
            payload = {"datetime": time.isoformat()}
        case "Application.pytorch.nccl_watchdog":
            if source_rule is None:
                source_rule = "Synthetic PyTorch NCCL watchdog"
            payload = {
                "raw_error_time": event_spec["raw_error_time"]
                if "raw_error_time" in event_spec
                else time.strftime("%H:%M:%S"),
                "timeout_seconds": event_spec["timeout_seconds"],
            }
        case "Application.dataloader.exception":
            if source_rule is None:
                source_rule = "Synthetic dataloader exception"
            payload = event_spec.get("payload", {})
        case other:
            raise ValueError(f"Unsupported syslog aux event type {other!r}")

    src = Source(
        source_context=obase.context,
        line_num=line_num,
        datum=datum,
        time=time,
        resources=AllJobResources,
        rule=source_rule,
    )
    return (event_type, payload, src)


def _run_syslog_single_line_mfu_case(case_data: RuleDataCase, attribution_test_builder, obs_data, timezone):
    tflops_start = case_data.case.get("tflops_start", True)
    severity = case_data.case.get("severity", "error")
    job_info = _build_syslog_jobinfo(
        timezone, tflops_start=tflops_start, multi_gpu=case_data.case.get("multi_gpu", True)
    )
    if extra_nodes := case_data.case.get("extra_nodes"):
        job_info.nodes.extend(extra_nodes)

    node = case_data.case.get("node", job_info.nodes[0])
    timestamp_s = case_data.case.get("timestamp_s", job_info.job_start_time.timestamp())
    entry = make_syslog_entry(case_data.case["line"], node, int(timestamp_s * 1e9), severity=severity)
    events_to_add = []
    if case_data.case.get("inject_mfu", True):
        events_to_add.append(_make_mfu_event(job_info, tflops_start, obs_data))

    result = attribution_test_builder(
        job_info=job_info,
        data=[entry],
        source=ObservationSource.SYSLOG,
        format=ObservationFormat.RAW_LOKI_STREAMS,
        check_fn=process_check,
        events_to_add=events_to_add,
        stream_id=case_data.description,
    )
    node_attr = result.get_node_attributions()
    expected_node = case_data.case.get("expected_node", node)

    match case_data.case["expect"]:
        case "bad":
            match_bad(node_attr, expected_node, case_data.case["expected_symptom"], case_data.case["expected_rule"])
        case "interesting":
            match_interesting(
                node_attr, expected_node, case_data.case["expected_symptom"], case_data.case["expected_rule"]
            )
        case "no_match":
            assert (
                expected_node not in node_attr.attributions
                or not node_attr.attributions[expected_node].attributions._data
            )
        case other:
            raise ValueError(f"Unsupported expectation {other!r} in {case_data.path}")


def _run_syslog_single_line_events_case(case_data: RuleDataCase, attribution_test_builder, obs_data, timezone):
    tflops_start = case_data.case.get("tflops_start", True)
    severity = case_data.case.get("severity", "error")
    job_info = _build_syslog_jobinfo(
        timezone, tflops_start=tflops_start, multi_gpu=case_data.case.get("multi_gpu", True)
    )
    node = case_data.case.get("node", job_info.nodes[0])
    timestamp_s = case_data.case.get("timestamp_s", job_info.job_start_time.timestamp())
    entry = make_syslog_entry(case_data.case["line"], node, int(timestamp_s * 1e9), severity=severity)
    events_to_add = [_make_syslog_aux_event(job_info, spec, obs_data) for spec in case_data.case.get("events", [])]

    result = attribution_test_builder(
        job_info=job_info,
        data=[entry],
        source=ObservationSource.SYSLOG,
        format=ObservationFormat.RAW_LOKI_STREAMS,
        check_fn=process_check,
        events_to_add=events_to_add,
        stream_id=case_data.description,
    )
    # Ordered event-stream parity assertion: assert the exact (event_type, rule)
    # sequence emitted, in emission order, so a reordered try_all is caught. This
    # MUST run before get_node_attributions(), which consumes the raw bad-node event.
    expected_events = case_data.case.get("expected_events")
    if expected_events is not None:
        # Filter to this case's source line (unique per case). The bad-node and the
        # tracking event share source_datum but land in different phase contexts, so
        # source_datum -- not context_id -- is the reliable per-case discriminator.
        rows = result.event_stream.query_raw(
            "SELECT t.name, r.name, e.payload FROM events e "
            "LEFT JOIN event_types t ON e.event_type = t.id "
            "LEFT JOIN rules r ON e.source_rule = r.id "
            "WHERE e.source_datum = ? "
            "ORDER BY e.id",
            [case_data.case["line"]],
        )
        actual = [(event_type, rule) for event_type, rule, _ in rows]
        expected = [(ev["event_type"], ev["rule"]) for ev in expected_events]
        assert actual == expected, f"event order mismatch in {case_data.path}: {actual!r} != {expected!r}"
        for i, (ev_spec, (_, _, payload_raw)) in enumerate(zip(expected_events, rows)):
            if expected_pl := ev_spec.get("payload"):
                actual_pl = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
                for key, val in expected_pl.items():
                    assert actual_pl.get(key) == val, (
                        f"payload[{key!r}] mismatch in {case_data.path} event {i}: "
                        f"expected {val!r}, got {actual_pl.get(key)!r}"
                    )
        return

    node_attr = result.get_node_attributions()
    expected_node = case_data.case.get("expected_node", node)

    if (expectations := case_data.case.get("expectations")) is not None:
        for expectation in expectations:
            node = expectation.get("expected_node", expected_node)
            match expectation["expect"]:
                case "bad":
                    match_bad(node_attr, node, expectation["expected_symptom"], expectation["expected_rule"])
                case "interesting":
                    match_interesting(node_attr, node, expectation["expected_symptom"], expectation["expected_rule"])
                case other:
                    raise ValueError(f"Unsupported expectation {other!r} in {case_data.path}")
        return

    match case_data.case["expect"]:
        case "bad":
            match_bad(node_attr, expected_node, case_data.case["expected_symptom"], case_data.case["expected_rule"])
        case "interesting":
            match_interesting(
                node_attr, expected_node, case_data.case["expected_symptom"], case_data.case["expected_rule"]
            )
        case other:
            raise ValueError(f"Unsupported expectation {other!r} in {case_data.path}")


def _run_syslog_fm_gpu_series_case(case_data: RuleDataCase, attribution_test_builder, timezone):
    job_info = _build_syslog_jobinfo(timezone, tflops_start=True, multi_gpu=True)
    node = case_data.case.get("node", job_info.nodes[0])
    base_ts = case_data.case.get("timestamp_s", job_info.job_start_time.timestamp())
    lines = [
        make_syslog_entry(
            (
                "Lost connection to GPU on nodeId 1, systemGuid 0xaaa, "
                f"gpuGUID 0x{gpu_guid:016x}, port LID 0x1, GPU state Error."
            ),
            node,
            int((base_ts + offset) * 1e9),
        )
        for offset, gpu_guid in enumerate(case_data.case["gpu_guids"])
    ]

    result = attribution_test_builder(
        job_info=job_info,
        data=lines,
        source=ObservationSource.SYSLOG,
        format=ObservationFormat.RAW_LOKI_STREAMS,
        check_fn=process_check,
        stream_id=case_data.description,
    )
    node_attr = result.get_node_attributions()

    for expectation in case_data.case["expectations"]:
        symptom = expectation["symptom"]
        rule = expectation["rule"]
        match expectation["expect"]:
            case "bad":
                match_bad(node_attr, node, symptom, rule)
            case "absent":
                if node in node_attr.attributions:
                    assert symptom not in node_attr.attributions[node].attributions
            case other:
                raise ValueError(f"Unsupported expectation {other!r} in {case_data.path}")


def _build_applog_jobinfo(timezone, *, tflops_start: bool, variant: str = "default"):
    start = datetime.fromtimestamp(1741860016, tz=timezone)
    end = datetime.fromtimestamp(1741862869, tz=timezone)
    tflops_time = start if tflops_start else end
    nodes_by_variant = {
        "default": ["host-9ad02e1b"],
        "peer_localhost": ["host-9ad02e1b", "localhost"],
        "named_hosts": ["host-9ad02e1b", "localhost", "testnode0-001", "testnode0-002", "testnode0-003"],
        "pushrecv_pair": ["host-9ad02e1b", "host-a222193a"],
    }
    try:
        nodes = nodes_by_variant[variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported applog job variant {variant!r}") from exc
    return (
        JobInfoBuilder(
            id="12345678",
            cluster="cluster_b722552b",
            state="CANCELLED",
            type="training",
            framework="megatron",
            node_to_ranks=make_node_to_ranks(nodes, 8),
            job_start_time=start,
            source_name="artifical slurm",
            timezone=timezone,
        )
        .job_end_time(end)
        .srun_cmd("srun test.py")
        .time_limit(600)
        .last_tflops_time(tflops_time)
        .build()
    )


def _run_applog_single_line_case(case_data: RuleDataCase, attribution_test_builder, timezone):
    tflops_start = case_data.case.get("tflops_start", True)
    job_info = _build_applog_jobinfo(
        timezone,
        tflops_start=tflops_start,
        variant=case_data.case.get("job_variant", "default"),
    )
    node = case_data.case.get("expected_node", job_info.nodes[0])

    result = attribution_test_builder(
        job_info=job_info,
        data=[case_data.case["line"]],
        source=ObservationSource.APPLOG,
        format=ObservationFormat.RAW_LOG_LINES,
        check_fn=process_check,
        stream_id=case_data.description,
    )

    expected_events = case_data.case.get("expected_events")
    if expected_events is not None:
        rows = result.event_stream.query_raw(
            "SELECT t.name, r.name, e.payload FROM events e "
            "LEFT JOIN event_types t ON e.event_type = t.id "
            "LEFT JOIN rules r ON e.source_rule = r.id "
            "WHERE e.source_datum = ? "
            "ORDER BY e.id",
            [case_data.case["line"]],
        )
        actual = [(event_type, rule) for event_type, rule, _ in rows]
        expected = [(ev["event_type"], ev["rule"]) for ev in expected_events]
        assert actual == expected, f"event order mismatch in {case_data.path}: {actual!r} != {expected!r}"
        for i, (ev_spec, (_, _, payload_raw)) in enumerate(zip(expected_events, rows)):
            if expected_pl := ev_spec.get("payload"):
                actual_pl = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
                for key, val in expected_pl.items():
                    assert actual_pl.get(key) == val, (
                        f"payload[{key!r}] mismatch in {case_data.path} event {i}: "
                        f"expected {val!r}, got {actual_pl.get(key)!r}"
                    )
        return

    node_attr = result.get_node_attributions()

    match case_data.case["expect"]:
        case "bad":
            match_bad(node_attr, node, case_data.case["expected_symptom"], case_data.case["expected_rule"])
        case "interesting":
            match_interesting(node_attr, node, case_data.case["expected_symptom"], case_data.case["expected_rule"])
        case "no_match":
            assert node not in node_attr.attributions or not node_attr.attributions[node].attributions._data
        case other:
            raise ValueError(f"Unsupported expectation {other!r} in {case_data.path}")


def _run_applog_bad_peers_case(case_data: RuleDataCase, attribution_test_builder, timezone):
    tflops_start = case_data.case.get("tflops_start", True)
    job_info = _build_applog_jobinfo(
        timezone,
        tflops_start=tflops_start,
        variant=case_data.case.get("job_variant", "default"),
    )

    result = attribution_test_builder(
        job_info=job_info,
        data=[case_data.case["line"]],
        source=ObservationSource.APPLOG,
        format=ObservationFormat.RAW_LOG_LINES,
        check_fn=process_check,
        stream_id=case_data.description,
    )
    node_attr = result.get_node_attributions()
    expected_nodes = case_data.case["expected_nodes"]

    if case_data.case["expect"] != "bad":
        raise ValueError(f"Unsupported expectation {case_data.case['expect']!r} in {case_data.path}")
    match_bad_peers(node_attr, expected_nodes, case_data.case["expected_symptom"], case_data.case["expected_rule"])


def test_rule_data_case_count_smoke():
    assert len(RULE_DATA_CASES) > MIN_RULE_DATA_CASE_COUNT


def test_rule_data_warns_on_unknown_case_keys(tmp_path, monkeypatch):
    test_file = tmp_path / "synthetic.test.toml"
    test_file.write_text(
        """
[meta]
runner = "node_metrics"

[[cases]]
description = "synthetic typo case"
body = "ECC Error"
expect = "bad"
expected_symtom = "ECC Error"
expected_rule = "ecc-error"
""".lstrip()
    )

    monkeypatch.setattr(rule_data_runner, "_bundled_rules_dir", lambda: tmp_path)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        params = rule_data_runner.iter_rule_data_cases()

    assert len(params) == 1
    assert any("expected_symtom" in str(item.message) for item in captured)


@pytest.mark.parametrize("rule_case", RULE_DATA_CASES)
def test_rule_data(rule_case: RuleDataCase, attribution_test_builder, simple_job_factory, obs_data, timezone):
    match rule_case.runner:
        case "node_metrics":
            _run_node_metrics_case(rule_case, attribution_test_builder, simple_job_factory)
        case "nvlink_metrics":
            _run_nvlink_metrics_case(rule_case, attribution_test_builder, simple_job_factory)
        case "syslog_single_line_mfu":
            _run_syslog_single_line_mfu_case(rule_case, attribution_test_builder, obs_data, timezone)
        case "syslog_single_line_events":
            _run_syslog_single_line_events_case(rule_case, attribution_test_builder, obs_data, timezone)
        case "syslog_fm_gpu_series":
            _run_syslog_fm_gpu_series_case(rule_case, attribution_test_builder, timezone)
        case "applog_single_line":
            _run_applog_single_line_case(rule_case, attribution_test_builder, timezone)
        case "applog_bad_peers":
            _run_applog_bad_peers_case(rule_case, attribution_test_builder, timezone)
        case other:
            raise ValueError(f"Unsupported runner {other!r} in {rule_case.path}")
