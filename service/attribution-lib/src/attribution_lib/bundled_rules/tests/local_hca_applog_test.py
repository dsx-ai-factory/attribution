# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-log HCA rules: parsing, emitter identity and peer evidence."""

import ast
import json

import pytest

from attribution_lib import core
from attribution_lib.bundled_rules.tests._hca_test_support import (
    events,
    exported,
    graphs as graphs,
    job as job,
    modify,
    run as run,
    verb,
)
from attribution_lib.observation.inbound import (
    Source as ObservationSource,
)
from test_support.helpers import make_applog_entry


def test_errno22_keeps_remote_interesting_and_pair_evidence(run, graphs):
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
    graph = json.dumps(graphs[stream.ctx_id])
    assert "node-a" in graph and "node-b" in graph and modify() in graph


@pytest.mark.parametrize("errno,local", [(110, "::ffff:192.0.2.1"), (22, "192.0.2.1"), (11, "::ffff:192.0.2.1")])
def test_existing_resolved_and_mismatched_pair_behavior(run, errno, local):
    result, _ = run([(modify(errno, local), 0)])
    assert set(exported(result)[0]) == {"node-a", "node-b"}


@pytest.mark.parametrize("fault_nodes", [("node-b",), ("node-a", "node-b")])
def test_independent_endpoint_faults_are_retained(run, fault_nodes):
    logs = [(node, "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.", 1) for node in fault_nodes]
    result, _ = run([(modify(), 0)], logs)
    bad, _ = exported(result)
    assert set(bad) == {"node-a", "node-b"}
    assert all(any("XID-79" in symptom for symptom in bad[node]) for node in fault_nodes)


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
        "error Resource temporarily unavailable errno 5",
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
def test_unidentified_emitter_does_not_blame_an_implicit_rank(run, line):
    result, _ = run([(line, 0)], mapping={}, declared_nodes=False)
    assert not any("Local HCA" in s for values in exported(result)[0].values() for s in values)
    assert not exported(result)[0]
    assert not result.bad_nodes_attr.my_bad_nodes.job_level.has_bad()
    assert any(
        "Local HCA" in symptom or "ibv_modify_qp" in symptom
        for symptom, _ in result.bad_nodes_attr.my_bad_nodes.job_level.all_items()
    )


@pytest.mark.parametrize("peer", ["N/A", "::ffff:198.51.100.7", "::ffff:192.0.2.2"])
def test_overwritten_nccl_parameters_preserve_user_error(run, peer):
    result, _ = run([(modify(110, remote=peer), 0)], job_script="export NCCL_IB_TC=128")
    assert result.outcome_attr.value().outcome == "Fail User"
    assert "Overwrite NCCL parameters" in result.outcome_attr.value().root_cause_symptom
    assert set(exported(result)[0]) == ({"node-a", "node-b"} if peer == "::ffff:192.0.2.2" else {"node-a"})
    assert not any("Local HCA" in s for values in exported(result)[0].values() for s in values)


@pytest.mark.parametrize("errno", [22, 110])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize(
    "local_gid,remote_gid,local_ip,remote_ip,pair_type",
    [
        ("::ffff:192.0.2.1", "::ffff:192.0.2.99", "192.0.2.1", "192.0.2.99", "complete"),
        ("N/A", "::ffff:192.0.2.99", None, "192.0.2.99", "partial"),
        ("N/A", "::ffff:c000:0263", None, "192.0.2.99", "partial"),
        ("::ffff:c000:0201", "N/A", "192.0.2.1", None, "partial"),
        ("192.0.2.1", "192.0.2.99", "192.0.2.1", "192.0.2.99", "partial"),
        ("2001:db8:1::10", "2001:db8:2::20", "2001:db8:1::10", "2001:db8:2::20", "partial"),
        ("::ffff:192.0.2.1", "N/A", "192.0.2.1", None, "partial"),
    ],
)
def test_qp_pair_preserves_addresses_separately_from_blame(
    run, graphs, errno, mapped, local_gid, remote_gid, local_ip, remote_ip, pair_type
):
    """Retain endpoint evidence even when addresses cannot identify a node."""
    mapping = {remote_ip: "node-b"} if mapped and remote_ip is not None else {}
    line = modify(errno, local=local_gid, remote=remote_gid)
    result, stream = run([(line, 0)], mapping=mapping)
    pair = events(stream, f"Application.bad_pair.{pair_type}")[0]
    expected_peer = "node-b" if mapping else remote_ip
    assert pair[2]["remote_node"] == expected_peer
    assert pair[2].get("local_dev_ip") == local_ip
    assert pair[2].get("remote_dev_ip") == remote_ip
    assert pair[3].datum == line
    report = ast.literal_eval(result.collectors["bad_pairs"])
    assert report["dst_node"] == expected_peer
    assert report.get("src_ip") == local_ip
    assert report.get("dst_ip") == remote_ip
    assert report["src_device"] == "mlx5_4:1"
    bad, interesting = exported(result)
    assert set(bad) | set(interesting) <= {"node-a", "node-b"}
    if not mapping:
        assert set(bad) == {"node-a"}
    graph = json.dumps(graphs[stream.ctx_id])
    assert line in graph


@pytest.mark.parametrize("errno", [11, 22, 110, 125])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize(
    "local_gid,remote_gid",
    [
        ("::ffff:c000:0201", "::ffff:c000:0263"),
        ("::FFFF:C000:0201", "::FFFF:C000:0263"),
        ("0:0:0:0:0:ffff:c000:0201", "0000:0000:0000:0000:0000:ffff:c000:0263"),
    ],
)
def test_equivalent_mapped_gids_keep_same_attribution(run, graphs, errno, mapped, local_gid, remote_gid):
    """Normalize equivalent GID spellings before resolving peers or building pairs."""
    mapping = {"192.0.2.99": "node-b"} if mapped else {}
    baseline, base_stream = run([(modify(errno, remote="::ffff:192.0.2.99"), 0)], mapping=mapping)
    line = modify(errno, local=local_gid, remote=remote_gid)
    result, stream = run([(line, 0)], mapping=mapping)
    assert result.outcome_attr.value() == baseline.outcome_attr.value()
    assert exported(result) == exported(baseline)
    assert result.collectors["bad_pairs"] == baseline.collectors["bad_pairs"]
    pairs = events(stream, "Application.bad_pair.complete")
    assert len(pairs) == 1
    assert pairs[0][2] == events(base_stream, "Application.bad_pair.complete")[0][2]
    assert pairs[0][3].datum == line
    assert line in json.dumps(graphs[stream.ctx_id])
    if mapped:
        assert not events(stream, "Application.hca.command_failure")


@pytest.mark.parametrize("errno", [22, 110])
def test_unidentified_emitter_never_blames_resolved_peer(run, errno):
    line = "Last error: " + modify(errno, local="N/A").split("NCCL WARN ", 1)[1]
    result, _ = run([(line, 0)], mapping={"192.0.2.2": "node-b"}, declared_nodes=False)
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
