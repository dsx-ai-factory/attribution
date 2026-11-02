# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic regressions for local RDMA attribution and device-scoped guards."""

import ast
import json
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import timedelta

import pytest

from attribution_lib import core
from attribution_lib.attributor import Attributor
from attribution_lib.decision_trace import build_decision_graph
from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Observation,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.sources import SourceSet
from attribution_lib.utils import hostname_mapping
from test_support.helpers import make_applog_entry, make_syslog_entry


@pytest.fixture
def graphs():
    return {}


@pytest.fixture
def job(simple_job_factory):
    return simple_job_factory(state="FAILED", exit_code_signal=1, nodes=["node-a", "node-b"], ranks_per_node=1)


def modify(errno=22, local="::ffff:192.0.2.1", remote="::ffff:192.0.2.2", device="mlx5_4:1"):
    return (
        f"node-a:123:456 [0] NCCL WARN Call to ibv_modify_qp failed with {errno} error, "
        f"on dev {device}, curr state RTR, next state RTS, local GID index -1, "
        f"local GID {local}, remote GID {remote}"
    )


def verb(name, errno=11, device="mlx5_4", process="123:456"):
    suffix = f", on dev {device}:1" if device else ""
    return f"node-a:{process} [0] NCCL WARN Call to {name} failed with error Resource temporarily unavailable errno {errno}{suffix}"


@pytest.fixture
def run(event_manager, obs_data, job, request, graphs):
    count = 0

    def _run(lines, syslogs=(), mapping=None, provider=(), job_script=None, declared_nodes=True):
        current_job = job._replace(job_script_content=job_script) if job_script else job
        nonlocal count
        count += 1
        stream = event_manager.create_stream(f"{request.node.name}-{count}")
        start = job.job_start_time.timestamp()
        app = obs_data(
            job,
            [
                (
                    {"body": line, **({"ts": int((start + 20 + offset) * 1e9)} if offset is not None else {})}
                    if not declared_nodes
                    else {"body": line, "resources": [{"kind": "node", "hostname": "node-a"}]}
                    if offset is None
                    else make_applog_entry(
                        line, (start + 20 + offset) * 1e9, "node-b" if line.startswith("node-b:") else "node-a"
                    )
                )
                for line, offset in lines
            ],
            ObservationSource.APPLOG,
            insert_tflops_line=False,
        )
        sys = obs_data(
            job,
            [make_syslog_entry(line, node, (start + 20 + offset) * 1e9) for node, line, offset in syslogs],
            ObservationSource.SYSLOG,
            time_interval=TimeInterval(job.job_start_time, job.job_end_time + timedelta(minutes=3)),
        )
        stream.register_observation_context(app.context)
        stream.register_observation_context(sys.context)
        with hostname_mapping(mapping if mapping is not None else {"192.0.2.1": "node-a", "192.0.2.2": "node-b"}):
            bad = core.calculate_bad_nodes(event_stream=stream, job_info=current_job, applog_in=app, syslog_in=sys)
        # Exercise the normalized provider-evidence contract without embedding
        # provider-specific collection or inventory logic in public rules.
        for symptom, node, offset, kind in provider:
            context = replace(sys.context, source=kind)
            source = job.source("synthetic provider evidence", "test")._replace(
                source_context=context, time=job.job_start_time + timedelta(seconds=20 + offset)
            )
            bad.my_bad_nodes.add([node], SourceSet(symptom).add(source))
        recorder = DecisionRecorder(job.id)
        result = core.attribute(
            event_stream=stream,
            job_info=current_job,
            bad_node_result=bad,
            extract_job_metadata_from_logfile=False,
            run_collectors=True,
            recorder=recorder,
        )
        graphs[stream.ctx_id] = asdict(build_decision_graph(recorder, result))
        return result, stream

    return _run


def exported(result):
    bad, interesting, epilog = defaultdict(set), defaultdict(set), defaultdict(set)
    result.bad_nodes_attr.my_bad_nodes.extract_bad_interesting_epilog(bad, interesting, epilog)
    return bad, interesting


def events(stream, typ):
    return list(query_events_with_source(stream, stream.mgr.types.get_id(typ)))


def test_errno22_keeps_remote_interesting_and_pair_evidence(run):
    result, stream = run([(modify(), 0)])
    bad, interesting = exported(result)
    assert set(bad) == {"node-a"}
    assert "node-b" in interesting
    assert result.outcome_attr.value().outcome == "Fail InfraSW"
    pair = events(stream, "Application.bad_pair.complete")[0]
    assert pair[2]["local_device"] == "mlx5_4:1"
    assert pair[2]["local_dev_ip"] == "192.0.2.1"
    assert pair[2]["remote_dev_ip"] == "192.0.2.2"
    assert pair[3].datum == modify()
    reported = ast.literal_eval(result.collectors["bad_pairs"])
    assert reported["src_device"] == "mlx5_4:1"
    assert reported["dst_node"] == "node-b"


@pytest.mark.parametrize("errno,local", [(110, "::ffff:192.0.2.1"), (22, "192.0.2.1"), (11, "::ffff:192.0.2.1")])
def test_existing_resolved_and_mismatched_pair_behavior(run, errno, local):
    result, _ = run([(modify(errno, local), 0)])
    assert set(exported(result)[0]) == {"node-a", "node-b"}


def test_remote_independent_fault_is_retained(run):
    result, _ = run([(modify(), 0)], [("node-b", "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.", 1)])
    assert "node-b" in exported(result)[0]


@pytest.mark.parametrize("errno", [11, 110, 125])
@pytest.mark.parametrize("peer", ["N/A", "::ffff:198.51.100.7"])
def test_unavailable_peer_is_never_blamed(run, errno, peer):
    result, stream = run([(modify(errno, "N/A", peer, "mlx5_4:-1"), 0)])
    assert set(exported(result)[0]) == {"node-a"}
    assert any(s.startswith("Local HCA command failure") for s in exported(result)[0]["node-a"])
    assert not events(stream, "Attribution.node.bad_peers")
    pair = events(stream, "Application.bad_pair.partial")[0][2]
    assert pair["local_device"] == "mlx5_4:-1"
    source = events(stream, "Application.bad_pair.partial")[0][3]
    assert f"remote GID {peer}" in source.datum
    reported = ast.literal_eval(result.collectors["bad_pairs"])
    assert reported["src_device"] == "mlx5_4:-1"


@pytest.mark.parametrize("name", ["ibv_query_qp", "ibv_dereg_mr"])
def test_local_verbs_are_attributed_to_emitter(run, name):
    result, _ = run([(verb(name), 0)])
    assert set(exported(result)[0]) == {"node-a"}
    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    assert result.outcome_attr.value().hw_cause == "Unknown HW Cause"


def test_registration_eagain_alone_remains_interesting(run):
    result, _ = run([(verb("ibv_reg_mr"), i) for i in range(10)])
    bad, interesting = exported(result)
    assert not any("Local HCA" in symptom for values in bad.values() for symptom in values)
    assert any("ibv_reg_mr eno 11" in symptom for symptom in interesting["node-a"])


@pytest.mark.parametrize(
    "offset,device,qualified", [(299, "mlx5_4", True), (301, "mlx5_4", False), (1, "mlx5_5", False)]
)
def test_registration_guard_is_time_and_device_scoped(run, offset, device, qualified):
    result, _ = run([(verb("ibv_reg_mr"), 0), (verb("ibv_query_qp", device=device), offset)])
    bad, _ = exported(result)
    assert any("ibv_reg_mr eno 11" in symptom for symptom in bad["node-a"]) == qualified


@pytest.mark.parametrize("ambiguous", [False, True])
def test_device_less_wrapper_needs_unambiguous_same_thread_anchor(run, ambiguous):
    lines = [(modify(125, "N/A", "N/A"), 0), (verb("ibv_reg_mr", device=None), 1)]
    if ambiguous:
        lines.append((modify(125, "N/A", "N/A", device="mlx5_5:1"), 2))
    result, _ = run(lines)
    bad, _ = exported(result)
    assert any("ibv_reg_mr eno 11" in symptom for symptom in bad["node-a"]) == (not ambiguous)


@pytest.mark.parametrize(
    "device,offset,qualified", [("0000:03:00.3", 1, True), ("0000:04:00.3", 1, False), ("0000:03:00.3", 301, False)]
)
def test_irisc_corroboration_requires_same_device_and_time(run, device, offset, qualified):
    log = "mlx5_core 0000:03:00.3: Health issue observed, irisc not responding, severity(3) ERROR:"
    result, _ = run([(verb("ibv_reg_mr", device=device), 0)], [("node-a", log, offset)])
    assert any("ibv_reg_mr eno 11" in s for s in exported(result)[0]["node-a"]) == qualified


def test_health_compromised_is_not_corroboration(run):
    result, _ = run(
        [(verb("ibv_reg_mr"), 0)],
        [("node-a", "mlx5_core 0000:03:00.3: device's health compromised - reached miss count", 1)],
    )
    assert not any("ibv_reg_mr eno 11" in s for s in exported(result)[0].get("node-a", set()))


def test_nccl_error2_is_recognized(attribution_test_builder, job):
    line = "node-a:123:456 [0] RuntimeError: NCCL Error 2: unhandled system error"
    extracted = attribution_test_builder(
        job_info=job,
        data=[make_applog_entry(line, (job.job_start_time.timestamp() + 20) * 1e9, "node-a")],
        source=ObservationSource.APPLOG,
        check_fn=core.process_check,
        insert_tflops_line=False,
    )
    assert any(
        payload.get("symptom") == "ncclSystemError"
        for _, _, payload, _ in events(extracted.event_stream, "Attribution.node.bad")
    )


def test_registration_guard_does_not_combine_nodes(run):
    result, _ = run([(verb("ibv_reg_mr"), 0), (verb("ibv_query_qp").replace("node-a:", "node-b:"), 1)])
    assert not any("ibv_reg_mr eno 11" in s for s in exported(result)[0].get("node-a", set()))


def test_device_less_registration_needs_same_thread(run):
    result, _ = run([(modify(125, "N/A", "N/A"), 0), (verb("ibv_reg_mr", device=None, process="777:888"), 1)])
    assert not any("ibv_reg_mr eno 11" in s for s in exported(result)[0].get("node-a", set()))


def test_text_only_registration_exhaustion_is_not_hardware(run):
    line = "node-a:123:456 [0] NCCL WARN Call to ibv_reg_mr failed with error Resource temporarily unavailable"
    result, _ = run([(line, 0)])
    assert any("ibv_reg_mr eno 11" in s for s in exported(result)[1]["node-a"])
    assert not any("Local HCA" in s for values in exported(result)[0].values() for s in values)


def test_multiple_local_commands_preserve_each_verb_and_source(run):
    lines = [(modify(125, "N/A", "N/A", "mlx5_4:-1"), 0)]
    lines += [
        (verb(name, device=None), i + 1)
        for i, name in enumerate(["ibv_query_qp", "ibv_create_cq", "ibv_reg_mr", "ibv_dereg_mr"])
    ]
    result, _ = run(lines)
    symptoms = exported(result)[0]["node-a"]
    for name in ["ibv_modify_qp", "ibv_query_qp", "ibv_create_cq", "ibv_reg_mr", "ibv_dereg_mr"]:
        assert any(name in s for s in symptoms)
    assert "node-b" not in exported(result)[0]


def test_pci_only_syslog_does_not_guess_mlx5_device_mapping(run):
    log = "mlx5_core 0000:03:00.3: Health issue observed, irisc not responding, severity(3) ERROR:"
    result, _ = run([(verb("ibv_reg_mr"), 0)], [("node-a", log, 1)])
    assert not any("ibv_reg_mr eno 11" in s for s in exported(result)[0]["node-a"])
    assert result.outcome_attr.value().outcome == "Fail InfraSW"


@pytest.mark.parametrize("same_pci,second_offset,qualified", [(True, 2, True), (False, 2, False), (True, 121, False)])
def test_initialization_corroboration_requires_complete_same_pci_diagnosis(
    run, job, same_pci, second_offset, qualified
):
    end_offset = (job.job_end_time - job.job_start_time).total_seconds() - 20
    second_pci = "0000:03:00.3" if same_pci else "0000:04:00.3"
    first = "mlx5_core 0000:03:00.3: Firmware over 10000 MS in initializing state, aborting"
    second = f"mlx5_core {second_pci}: mlx5_init_one failed with error code -110"
    result, _ = run(
        [(verb("ibv_reg_mr", device="0000:03:00.3"), end_offset - 1)],
        [("node-a", first, end_offset + 1), ("node-a", second, end_offset + second_offset)],
    )
    assert any("ibv_reg_mr eno 11" in s for s in exported(result)[0].get("node-a", set())) == qualified


def test_single_initialization_message_is_not_corroboration(run, job):
    end_offset = (job.job_end_time - job.job_start_time).total_seconds() - 20
    first = "mlx5_core 0000:03:00.3: Firmware over 10000 MS in initializing state, aborting"
    result, _ = run([(verb("ibv_reg_mr", device="0000:03:00.3"), end_offset - 1)], [("node-a", first, end_offset + 1)])
    assert not any("ibv_reg_mr eno 11" in s for s in exported(result)[0].get("node-a", set()))


@pytest.mark.parametrize("errno", [11, 110, 125])
def test_corroborated_local_failure_has_known_nic_classification(run, errno):
    log = "mlx5_core 0000:03:00.3: Health issue observed, irisc not responding, severity(3) ERROR:"
    result, _ = run([(verb("ibv_query_qp", errno=errno, device="0000:03:00.3"), 0)], [("node-a", log, 1)])
    outcome = result.outcome_attr.value()
    assert outcome.outcome == "Fail InfraHW"
    assert outcome.hw_cause == "Known HW Cause - Node (NIC)"
    assert outcome.root_cause_symptom == "Corroborated local HCA command failure"


def test_standalone_nic_diagnosis_keeps_existing_classification(run):
    log = "mlx5_core 0000:03:00.3: Health issue observed, irisc not responding, severity(3) ERROR:"
    result, _ = run([], [("node-a", log, 1)])
    assert result.outcome_attr.value().outcome == "Fail InfraSW"
    assert "Corroborated local HCA command failure" not in exported(result)[0]["node-a"]


@pytest.mark.parametrize(
    "symptom,node,offset,kind,qualifies",
    [
        ("CSP Event NIC Fault", "node-a", 1, ObservationSource.CSP_EVENT, True),
        ("CSP Event NIC Fault", "node-b", 1, ObservationSource.CSP_EVENT, False),
        ("CSP Event NIC Fault", "node-a", 301, ObservationSource.CSP_EVENT, False),
        ("CSP Event EC2 Instance Check Failed", "node-a", 1, ObservationSource.CSP_EVENT, False),
        ("CSP Event NIC Fault", "node-a", 1, ObservationSource.APPLOG, False),
    ],
)
def test_provider_corroboration_requires_the_right_node_source_and_time(run, symptom, node, offset, kind, qualifies):
    result, _ = run([(verb("ibv_query_qp"), 0)], provider=[(symptom, node, offset, kind)])
    assert ("Corroborated local HCA command failure" in exported(result)[0]["node-a"]) == qualifies
    if qualifies:
        assert result.outcome_attr.value().root_cause_symptom == "CSP Event NIC Fault"
        assert result.outcome_attr.value().hw_cause == "Known HW Cause - Node"
        group = result.bad_nodes_attr.my_bad_nodes["node-a"].attributions
        assert group["CSP Event NIC Fault"].has_bad()


def test_registration_guard_does_not_use_unknown_times(run):
    result, _ = run([(verb("ibv_reg_mr"), None), (verb("ibv_query_qp"), None)])
    assert not any("ibv_reg_mr eno 11" in s for s in exported(result)[0]["node-a"])


def test_window_boundary_is_inclusive(run):
    result, _ = run([(verb("ibv_reg_mr"), 0), (verb("ibv_query_qp"), 300)])
    assert any("ibv_reg_mr eno 11" in s for s in exported(result)[0]["node-a"])


def test_successful_progress_excludes_stale_guard_proofs(run, job):
    timestamp = (job.job_start_time + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
    result, _ = run(
        [
            (verb("ibv_query_qp"), 0),
            (f"[{timestamp}] iteration 1/ 10 | throughput per GPU (TFLOP/s/GPU): 100.0 |", 10),
            (verb("ibv_reg_mr"), 20),
        ]
    )
    assert not any("ibv_reg_mr eno 11" in s for s in exported(result)[0].get("node-a", set()))


def test_sanitized_multi_verb_replay_keeps_one_faulty_node(run, graphs):
    calls = [(verb("ibv_query_qp", 110, device=None), i * 60) for i in range(5)]
    calls += [(modify(125, "N/A", "N/A", "mlx5_4:-1"), 241)]
    calls += [(verb("ibv_create_cq", device=None), 242 + i / 100) for i in range(56)]
    calls += [(verb("ibv_reg_mr", device=None), 243 + i / 100) for i in range(18)]
    calls += [(verb("ibv_dereg_mr", device=None), 244)]
    result, stream = run(calls)
    assert len(calls) == 81
    assert set(exported(result)[0]) == {"node-a"}
    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    graph = json.dumps(graphs[stream.ctx_id])
    assert "ibv_modify_qp" in graph and "ibv_reg_mr" in graph


def test_endpoint_details_survive_the_decision_graph(run, graphs):
    result, stream = run([(modify(), 0)])
    graph = json.dumps(graphs[stream.ctx_id])
    assert "node-a" in graph and "node-b" in graph
    assert modify() in graph
    assert "mlx5_4:1" in result.collectors["bad_pairs"]


def test_both_endpoints_keep_independent_fault_evidence(run):
    logs = [(node, "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.", 1) for node in ("node-a", "node-b")]
    result, _ = run([(modify(), 0)], logs)
    assert set(exported(result)[0]) == {"node-a", "node-b"}
    assert all(any("XID-79" in s for s in symptoms) for symptoms in exported(result)[0].values())


def test_provider_evidence_does_not_bypass_registration_guard(run):
    result, _ = run(
        [(verb("ibv_reg_mr"), 0)],
        provider=[("CSP Event NIC Fault", "node-a", 1, ObservationSource.CSP_EVENT)],
    )
    assert not any(
        "Local HCA command failure" in s or "Corroborated local HCA" in s
        for s in exported(result)[0].get("node-a", set())
    )


def test_repeated_and_incremental_attribution_recomputes_guard(event_manager, obs_data, job, request):
    stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(stream, job)
    start_ns = int((job.job_start_time.timestamp() + 20) * 1e9)
    entries = [make_applog_entry(verb("ibv_reg_mr"), start_ns, "node-a")]
    first = obs_data(job, entries, ObservationSource.APPLOG, insert_tflops_line=False)
    attributor.ingest(
        Observation(
            InboundContext(
                first.context.time_interval, first.context.source, first.context.format, first.context.timezone
            ),
            entries,
        )
    )
    result = attributor.attribute()
    assert not any("ibv_reg_mr" in s for s in exported(result)[0].get("node-a", set()))
    entries2 = [make_applog_entry(verb("ibv_query_qp"), start_ns + int(1e9), "node-a")]
    second = obs_data(job, entries2, ObservationSource.APPLOG, insert_tflops_line=False)
    attributor.ingest(
        Observation(
            InboundContext(
                second.context.time_interval, second.context.source, second.context.format, second.context.timezone
            ),
            entries2,
        )
    )
    result = attributor.attribute()
    assert any("ibv_reg_mr" in s for s in exported(result)[0]["node-a"])
    assert exported(attributor.attribute()) == exported(result)


def test_guard_proofs_do_not_leak_between_execution_contexts(run):
    run([(verb("ibv_query_qp"), 0)])
    result, _ = run([(verb("ibv_reg_mr"), 1)])
    assert not any("ibv_reg_mr" in s for s in exported(result)[0].get("node-a", set()))


def test_unknown_errno_is_not_reinterpreted_from_error_text(run):
    result, _ = run([(verb("ibv_query_qp", errno=5), 0)])
    assert not any("Local HCA" in s for values in exported(result)[0].values() for s in values)


@pytest.mark.parametrize("name", ["ibv_query_qp", "ibv_create_cq", "ibv_reg_mr", "ibv_dereg_mr"])
@pytest.mark.parametrize("errno", [11, 110, 125])
@pytest.mark.parametrize("error", ["error {errno}", "{errno}", "error 999 errno {errno}"])
def test_numeric_command_errors_retain_local_evidence(run, name, errno, error):
    """Recognize numeric spellings without bypassing the resource-limit guard."""
    line = f"node-a:123:456 [0] NCCL WARN Call to {name} failed with {error.format(errno=errno)} on dev mlx5_4"
    result, stream = run([(line, 0)])
    commands = events(stream, "Application.hca.command_failure")
    assert len(commands) == 1
    assert commands[0][2]["errno"] == errno
    assert commands[0][2]["node"] == "node-a"
    assert commands[0][3].datum == line
    bad, interesting = exported(result)
    guarded = errno == 11 and name in {"ibv_create_cq", "ibv_reg_mr"}
    selected = interesting if guarded else bad
    assert f"Local HCA command failure ({name} eno {errno})" in selected["node-a"]
    if guarded:
        assert not bad


@pytest.mark.parametrize(
    "error",
    [
        "error 999 Connection timed out",
        "error 999 Resource temporarily unavailable",
        "110 errno 999 Connection timed out",
    ],
)
def test_unsupported_numeric_errno_does_not_fall_back_to_text(run, error):
    """An explicit unsupported errno must not be replaced by a text inference."""
    line = f"node-a:123:456 [0] NCCL WARN Call to ibv_query_qp failed with {error} on dev mlx5_4"
    result, stream = run([(line, 0)])
    assert not events(stream, "Application.hca.command_failure")
    assert not exported(result)[0]


@pytest.mark.parametrize(
    "line",
    [
        "NCCL WARN Call to ibv_query_qp failed with error Connection timed out errno 110 on dev mlx5_4",
        "NCCL WARN " + modify().split("NCCL WARN ", 1)[1],
    ],
)
def test_unidentified_emitter_does_not_blame_an_implicit_rank(event_manager, obs_data, job, request, line):
    stream = event_manager.create_stream(request.node.name)
    app = obs_data(
        job,
        [{"body": line, "ts": int((job.job_start_time.timestamp() + 20) * 1e9)}],
        ObservationSource.APPLOG,
        insert_tflops_line=False,
    )
    stream.register_observation_context(app.context)
    bad = core.calculate_bad_nodes(event_stream=stream, job_info=job, applog_in=app)
    result = core.attribute(
        event_stream=stream, job_info=job, bad_node_result=bad, extract_job_metadata_from_logfile=False
    )
    assert not any("Local HCA" in s for values in exported(result)[0].values() for s in values)
    assert not exported(result)[0]
    assert not result.bad_nodes_attr.my_bad_nodes.job_level.has_bad()
    assert any(
        "Local HCA" in symptom or "ibv_modify_qp" in symptom
        for symptom, _ in result.bad_nodes_attr.my_bad_nodes.job_level.all_items()
    )


@pytest.mark.parametrize("name", ["ibv_reg_mr", "ibv_create_cq"])
def test_resource_limit_errors_cannot_corroborate_each_other(run, name):
    other = "ibv_create_cq" if name == "ibv_reg_mr" else "ibv_reg_mr"
    result, _ = run([(verb(name), 0), (verb(other), 1)])
    assert not any("Local HCA" in s for values in exported(result)[0].values() for s in values)
    assert any(name in s for s in exported(result)[1]["node-a"])


@pytest.mark.parametrize("peer", ["N/A", "::ffff:198.51.100.7", "::ffff:192.0.2.2"])
def test_overwritten_nccl_parameters_preserve_user_error(run, peer):
    result, _ = run([(modify(110, remote=peer), 0)], job_script="export NCCL_IB_TC=128")
    assert result.outcome_attr.value().outcome == "Fail User"
    assert "Overwrite NCCL parameters" in result.outcome_attr.value().root_cause_symptom
    assert set(exported(result)[0]) == ({"node-a", "node-b"} if peer == "::ffff:192.0.2.2" else {"node-a"})
    assert not any("Local HCA" in s for values in exported(result)[0].values() for s in values)


def test_unmapped_address_is_pair_evidence_not_node_identity(run):
    result, stream = run([(modify(remote="::ffff:198.51.100.7"), 0)])
    bad, interesting = exported(result)
    assert set(bad) == {"node-a"}
    assert not any("198.51" in node for node in set(bad) | set(interesting))
    assert events(stream, "Application.bad_pair.complete")[0][2]["remote_dev_ip"] == "198.51.100.7"


@pytest.mark.parametrize("errno", [22, 110])
def test_unidentified_emitter_never_blames_resolved_peer(event_manager, obs_data, job, request, errno):
    stream = event_manager.create_stream(request.node.name)
    line = "Last error: " + modify(errno, local="N/A").split("NCCL WARN ", 1)[1]
    app = obs_data(
        job,
        [{"body": line, "ts": int((job.job_start_time.timestamp() + 20) * 1e9)}],
        ObservationSource.APPLOG,
        insert_tflops_line=False,
    )
    stream.register_observation_context(app.context)
    with hostname_mapping({"192.0.2.2": "node-b"}):
        bad = core.calculate_bad_nodes(event_stream=stream, job_info=job, applog_in=app)
    result = core.attribute(
        event_stream=stream, job_info=job, bad_node_result=bad, extract_job_metadata_from_logfile=False
    )
    assert not exported(result)[0]
    assert "node-b" in exported(result)[1]


@pytest.mark.parametrize("mapped", [False, True])
def test_mapped_local_gid_identifies_unlabeled_emitter(run, mapped):
    line = "Last error: " + modify().split("NCCL WARN ", 1)[1]
    result, stream = run([(line, 0)], mapping={"192.0.2.1": "node-a"} if mapped else {}, declared_nodes=False)
    pair = events(stream, "Application.bad_pair.complete")[0]
    assert pair[3].declared_node is None
    assert pair[2]["local_node"] == ("node-a" if mapped else None)
    assert set(exported(result)[0]) == ({"node-a"} if mapped else set())


def test_logged_emitter_takes_precedence_over_declared_metadata(run):
    result, _ = run([(verb("ibv_query_qp").replace("node-a:", "node-c:"), 0)])
    assert set(exported(result)[0]) == {"node-c"}


@pytest.mark.parametrize("timed", [True, False])
def test_standard_progress_filter_demotes_local_failure(run, job, timed):
    timestamp = (job.job_start_time + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
    result, _ = run(
        [
            (verb("ibv_query_qp"), 0 if timed else None),
            (f"[{timestamp}] iteration 1/ 10 | throughput per GPU (TFLOP/s/GPU): 100.0 |", 10),
        ]
    )
    assert not any("Local HCA" in s for values in exported(result)[0].values() for s in values)
    assert any("Local HCA" in s for values in exported(result)[1].values() for s in values)


def test_existing_csp_cascade_is_preserved(run):
    result, _ = run(
        [
            (verb("ibv_query_qp"), 0),
            ("node-a:123:456 [0] pyxis: container start failed", 1),
        ],
        provider=[("CSP Event NIC Fault", "node-a", 2, ObservationSource.CSP_EVENT)],
    )
    assert result.outcome_attr.value().root_cause_symptom == "CSP Event NIC Fault"
    assert not any("pyxis" in s.lower() for s in exported(result)[0]["node-a"])


@pytest.mark.parametrize("errno", [11, 110])
def test_unresolved_qp_error_retains_original_priority(run, errno):
    import attribution_lib
    from attribution_lib.symptoms.declarations import OutcomePriority
    from attribution_lib.symptoms.registry_generation import OUTCOME_PRIORITIES

    result, _ = run([(modify(errno, remote="N/A"), 0)])
    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    assert result.outcome_attr.value().hw_cause == "Unknown HW Cause"
    match = attribution_lib.get_symptom_registry().outcome_for([result.outcome_attr.value().root_cause_symptom])
    assert match.tier_index == OUTCOME_PRIORITIES.index(OutcomePriority.MEDIUM_SW)


@pytest.mark.parametrize(
    "prefix,expected", [("0: ", "node-a"), ("1: ", "node-b"), ("[rank1] ", "node-b"), ("99: ", None), ("", None)]
)
@pytest.mark.parametrize("command", ["modify", "query"])
def test_explicit_leading_rank_uses_job_topology_only_as_fallback(run, prefix, expected, command):
    line = modify(125, local="N/A", remote="N/A") if command == "modify" else verb("ibv_query_qp", 110)
    line = prefix + line.split("NCCL WARN ", 1)[1]
    result, stream = run([(line, 0)], mapping={}, declared_nodes=False)
    event = events(stream, "Application.hca.command_failure")[0]
    assert event[3].declared_node is None
    assert event[2]["node"] == expected
    assert set(exported(result)[0]) == ({expected} if expected else set())


@pytest.mark.parametrize("prefix", ["1: ", "1: node-b:123:456 [0] NCCL WARN "])
def test_gid_or_explicit_emitter_precedes_rank_fallback(run, prefix):
    line = prefix + modify().split("NCCL WARN ", 1)[1]
    result, _ = run([(line, 0)], mapping={"192.0.2.1": "node-a"}, declared_nodes=False)
    expected = "node-b" if "node-b:" in prefix else "node-a"
    assert set(exported(result)[0]) == {expected}


def test_declared_node_precedes_explicit_rank_fallback(run):
    line = "1: " + verb("ibv_query_qp").split("NCCL WARN ", 1)[1]
    result, _ = run([(line, 0)], mapping={})
    assert set(exported(result)[0]) == {"node-a"}
