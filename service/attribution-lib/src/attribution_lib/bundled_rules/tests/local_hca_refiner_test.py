# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HCA refiners: resource guards, corroboration and evidence preservation."""

import json
from datetime import timedelta

import pytest

from attribution_lib.attributor import Attributor
from attribution_lib.bundled_rules.tests._hca_test_support import (
    exported,
    graphs as graphs,
    job as job,
    modify,
    run as run,
    verb,
)
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Observation,
    Source as ObservationSource,
)
from test_support.helpers import make_applog_entry


def test_registration_eagain_alone_remains_interesting(run):
    result, _ = run([(verb("ibv_reg_mr"), i) for i in range(10)])
    bad, interesting = exported(result)
    assert not any("Local HCA" in symptom for values in bad.values() for symptom in values)
    assert any("ibv_reg_mr eno 11" in symptom for symptom in interesting["node-a"])


@pytest.mark.parametrize(
    "offset,device,proof_node,qualified",
    [
        (299, "mlx5_4", "node-a", True),
        (300, "mlx5_4", "node-a", True),
        (301, "mlx5_4", "node-a", False),
        (1, "mlx5_5", "node-a", False),
        (1, "mlx5_4", "node-b", False),
    ],
)
def test_registration_guard_is_node_time_and_device_scoped(run, offset, device, proof_node, qualified):
    proof = verb("ibv_query_qp", device=device).replace("node-a:", f"{proof_node}:")
    result, _ = run([(verb("ibv_reg_mr"), 0), (proof, offset)])
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
    bad, interesting = exported(result)
    assert ("Corroborated local HCA command failure" in bad["node-a"]) == qualified
    assert "Local HCA command failure (ibv_reg_mr eno 11)" in interesting["node-a"]
    assert "Local HCA command failure (ibv_reg_mr eno 11)" not in bad["node-a"]


def test_health_compromised_is_not_corroboration(run):
    result, _ = run(
        [(verb("ibv_reg_mr"), 0)],
        [("node-a", "mlx5_core 0000:03:00.3: device's health compromised - reached miss count", 1)],
    )
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
    bad, interesting = exported(result)
    assert ("Corroborated local HCA command failure" in bad.get("node-a", set())) == qualified
    assert "Local HCA command failure (ibv_reg_mr eno 11)" in interesting["node-a"]
    assert "Local HCA command failure (ibv_reg_mr eno 11)" not in bad.get("node-a", set())


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


@pytest.mark.parametrize("errno", [11, 110])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "other_node,other_device,offset",
    [
        ("node-b", "0000:03:00.3", 0),
        ("node-a", "0000:04:00.3", 0),
        ("node-a", "0000:03:00.3", 302),
    ],
)
def test_compound_only_supersedes_its_qp_sources(run, graphs, errno, reverse, other_node, other_device, offset):
    """Keep independent QP evidence across nodes, devices and failure episodes."""
    matched = modify(errno, "N/A", "N/A", "0000:03:00.3")
    independent = modify(errno, "N/A", "N/A", other_device).replace("node-a:123:456", f"{other_node}:789:987")
    baseline, _ = run([(independent, offset)])
    lines = [(matched, 0), (independent, offset)]
    if reverse:
        lines.reverse()
    log = "mlx5_core 0000:03:00.3: Health issue observed, irisc not responding, severity(3) ERROR:"
    result, stream = run(lines, [("node-a", log, 1)])
    assert result.outcome_attr.value() == baseline.outcome_attr.value()
    bad, interesting = exported(result)
    symptom = f"Local HCA command failure (ibv_modify_qp eno {errno})"
    compound = "Corroborated local HCA command failure"
    assert compound in bad["node-a"]
    assert symptom in bad[other_node]
    assert symptom in interesting["node-a"]
    nodes = result.bad_nodes_attr.my_bad_nodes
    assert {source.datum for source in nodes[other_node].attributions[symptom]} == {independent}
    assert {source.datum for source in nodes["node-a"].attributions[compound]} == {matched, log}
    assert matched in {source.datum for source in nodes["node-a"].attributions[symptom].all_sources()}
    graph = json.dumps(graphs[stream.ctx_id])
    assert matched in graph and independent in graph


@pytest.mark.parametrize("errno", [11, 110])
def test_matched_qp_is_preserved_as_compound_evidence(run, graphs, errno):
    """Replace only the corroborated command's selected symptom, retaining provenance."""
    line = modify(errno, "N/A", "N/A", "0000:03:00.3")
    log = "mlx5_core 0000:03:00.3: Health issue observed, irisc not responding, severity(3) ERROR:"
    result, stream = run([(line, 0)], [("node-a", log, 1)])
    assert result.outcome_attr.value().root_cause_symptom == "Corroborated local HCA command failure"
    assert result.outcome_attr.value().hw_cause == "Known HW Cause - Node (NIC)"
    bad, interesting = exported(result)
    symptom = f"Local HCA command failure (ibv_modify_qp eno {errno})"
    assert symptom not in bad["node-a"]
    assert symptom in interesting["node-a"]
    group = result.bad_nodes_attr.my_bad_nodes["node-a"].attributions
    assert {source.datum for source in group["Corroborated local HCA command failure"]} == {line, log}
    assert line in json.dumps(graphs[stream.ctx_id])


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


@pytest.mark.parametrize("name", ["ibv_reg_mr", "ibv_create_cq"])
def test_resource_limit_errors_cannot_corroborate_each_other(run, name):
    other = "ibv_create_cq" if name == "ibv_reg_mr" else "ibv_reg_mr"
    result, _ = run([(verb(name), 0), (verb(other), 1)])
    assert not any("Local HCA" in s for values in exported(result)[0].values() for s in values)
    assert any(name in s for s in exported(result)[1]["node-a"])


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
