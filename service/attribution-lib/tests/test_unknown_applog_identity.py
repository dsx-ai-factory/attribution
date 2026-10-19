# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unknown emitters must survive attribution without acquiring a job node."""

import json
from collections import defaultdict
from datetime import timedelta

import pytest

from attribution_lib.attributor import Attributor
from attribution_lib.observation.inbound import Context, Format, Observation, Source, TimeInterval
from attribution_lib.resources import AllJobResources

COMPLETION = (
    "NET/IB: Got completion from peer 192.0.2.2<1234> "
    "with status=IBV_WC_RETRY_EXC_ERR(12) opcode=IBV_WC_SEND(0) vendor_err=129"
)
SYMPTOM = "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)"


@pytest.mark.parametrize(
    ("prefix", "resources", "expected_nodes"),
    [
        ("(Worker pid=123, ip=192.0.2.1) ", None, set()),
        ("[2026-01-02 03:04:05] ", None, set()),
        ("0: ", [], set()),
        ("0: ", None, {"node-a"}),
        ("node-b:123:456 [0] ", None, {"node-b"}),
        ("(Worker pid=123, ip=192.0.2.1) ", [{"kind": "node", "hostname": "node-b"}], {"node-b"}),
        ("node-a:123:456 [0] ", [{"kind": "node", "hostname": "outside-job"}], set()),
    ],
)
def test_completion_keeps_symptom_without_inventing_emitter(
    request, event_manager, simple_job_factory, timezone, prefix, resources, expected_nodes
):
    """Completion keeps symptom without inventing emitter.

    Args:
        request: Test identity used to name the event stream.
        event_manager: Event-store fixture.
        simple_job_factory: Factory creating job metadata for the test.
        timezone: Observation timezone.
        prefix: Source prefix under test.
        resources: Explicit resources or an omitted declaration.
        expected_nodes: Nodes that the evidence is allowed to identify.

    Returns:
        None. Assertions verify the evidence and resource contract.
    """
    job = simple_job_factory(nodes=["node-a", "node-b"], ranks_per_node=1, state="FAILED", exit_code_signal=1)
    stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(stream, job)
    body = prefix + COMPLETION
    entry = {"body": body}
    if resources is not None:
        entry["resources"] = resources
    attributor.ingest(
        Observation(
            Context(TimeInterval(None, None), AllJobResources, Source.APPLOG, Format.ENTRY_TIMESERIES, timezone),
            [entry],
        )
    )
    result = attributor.attribute()
    assert SYMPTOM in {str(s) for s in result.symptoms.symptoms()}
    bad, interesting, epilog = (defaultdict(set) for _ in range(3))
    result.bad_nodes_attr.my_bad_nodes.extract_bad_interesting_epilog(bad, interesting, epilog)
    named_job_nodes = (set(bad) | set(interesting)) & set(job.nodes)
    assert named_job_nodes == expected_nodes
    records = stream.query_raw("SELECT resources FROM events WHERE source_datum = ?", [body])
    assert len(records) > 0
    if len(expected_nodes) == 0:
        assert all(row[0] == "[]" for row in records)


@pytest.mark.parametrize("resources", [None, [], "AllJobResources"])
def test_explicit_all_resources_is_distinct_from_unknown(
    request, event_manager, simple_job_factory, timezone, resources
):
    """Explicit all resources is distinct from unknown.

    Args:
        request: Test identity used to name the event stream.
        event_manager: Event-store fixture.
        simple_job_factory: Factory creating job metadata for the test.
        timezone: Observation timezone.
        resources: Explicit resources or an omitted declaration.

    Returns:
        None. Assertions verify the evidence and resource contract.
    """
    job = simple_job_factory(nodes=["node-a", "node-b"], state="FAILED", exit_code_signal=1)
    stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(stream, job)
    body = "RuntimeError: CUDA error: misaligned address"
    entry = {"body": body}
    if resources is not None:
        entry["resources"] = resources
    attributor.ingest(
        Observation(
            Context(TimeInterval(None, None), AllJobResources, Source.APPLOG, Format.ENTRY_TIMESERIES, timezone),
            [entry],
        )
    )
    result = attributor.attribute()
    assert "cudaErrorMisalignedAddress" in {str(s) for s in result.symptoms.symptoms()}
    records = stream.query_raw("SELECT resources FROM events WHERE source_datum = ?", [body])
    assert len(records) > 0
    expected = "AllJobResources" if resources == "AllJobResources" else []
    assert all(json.loads(row[0]) == expected for row in records)


@pytest.mark.parametrize("resources", [None, []])
def test_unknown_emitter_preserves_timeout_reason(request, event_manager, simple_job_factory, timezone, resources):
    """Unknown emitter preserves timeout reason.

    Args:
        request: Test identity used to name the event stream.
        event_manager: Event-store fixture.
        simple_job_factory: Factory creating job metadata for the test.
        timezone: Observation timezone.
        resources: Explicit resources or an omitted declaration.

    Returns:
        None. Assertions verify the evidence and resource contract.
    """
    job = simple_job_factory(nodes=["node-a", "node-b"], state="FAILED", exit_code_signal=1)
    stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(stream, job)
    body = "slurmstepd: error: *** JOB 42 CANCELLED DUE TO TIME LIMIT ***"
    entry = {"body": body}
    if resources is not None:
        entry["resources"] = resources
    attributor.ingest(
        Observation(
            Context(TimeInterval(None, None), AllJobResources, Source.APPLOG, Format.ENTRY_TIMESERIES, timezone),
            [entry],
        )
    )
    result = attributor.attribute()
    assert result.cancelled_by == "Timeout"
    assert "Cancelled by Timeout" in {str(s) for s in result.symptoms.symptoms()}
    attributions = result.bad_nodes_attr.my_bad_nodes
    assert attributions.num_nodes() == 0
    sources = attributions.job_level["Cancelled by Timeout"]
    assert any(source.datum == body and len(source.resources) == 0 for source in sources)


@pytest.mark.parametrize("identity", ["timestamp_prefix", "explicit_unknown", "explicit_rank"])
def test_link_down_watchdog_keeps_hardware_outcome_without_inventing_node(
    request, event_manager, simple_job_factory, timezone, identity
):
    """Exercise ingestion through refinement for timestamped port-error evidence.

    Args:
        request: Test identity used to name the event stream.
        event_manager: Event-store fixture.
        simple_job_factory: Factory creating job metadata for the test.
        timezone: Observation timezone.
        identity: Resource/identity variant under test.

    Returns:
        None. Assertions verify the evidence and resource contract.
    """
    job = simple_job_factory(nodes=["node-a", "node-b"], state="FAILED", exit_code_signal=1)
    stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(stream, job)
    port_time = job.job_start_time + timedelta(seconds=60)
    watchdog_time = port_time + timedelta(seconds=600)
    prefix = f"{port_time:%Y-%m-%dT%H%M%S%z} 0: " if identity == "timestamp_prefix" else "0: "
    port_entry = {"body": f"{prefix}[{port_time:%Y-%m-%d %H:%M:%S}] NCCL WARN NET/IB : mlx5_0 port error"}
    if identity == "explicit_unknown":
        port_entry["resources"] = []
    watchdog_entry = {
        "body": (
            f"0: [E{watchdog_time:%m%d %H:%M:%S}.000000 ProcessGroupNCCL.cpp:685] [Rank 0] "
            "Watchdog caught collective operation timeout: WorkNCCL(SeqNum=123, OpType=ALLREDUCE, "
            "Timeout(ms)=600000) ran for 600000 milliseconds before timing out."
        )
    }
    attributor.ingest(
        Observation(
            Context(
                TimeInterval(job.job_start_time, job.job_end_time),
                AllJobResources,
                Source.APPLOG,
                Format.ENTRY_TIMESERIES,
                timezone,
            ),
            [port_entry, watchdog_entry],
        )
    )

    result = attributor.attribute()

    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    assert "link down(east-west)" in result.symptoms
    assert "PyTorch NCCL watchdog" not in result.symptoms
    bad, interesting, epilog = (defaultdict(set) for _ in range(3))
    result.bad_nodes_attr.my_bad_nodes.extract_bad_interesting_epilog(bad, interesting, epilog)
    assert set(bad) & set(job.nodes) == ({"node-a"} if identity == "explicit_rank" else set())
    if identity != "explicit_rank":
        # The explicitly ranked watchdog can remain interesting on node-a;
        # the unassigned port error must not become that node's evidence.
        assert all("link down(east-west)" not in group for _, group in result.bad_nodes_attr.my_bad_nodes.items())
        assert all(len(source.resources) == 0 for source in result.symptoms["link down(east-west)"])


@pytest.mark.parametrize("declaration", ["omitted", "null", "empty", "outside", "multiple"])
@pytest.mark.parametrize("nodes", [["node-a"], ["node-a", "node-b"]])
def test_single_node_inference_requires_omitted_or_null_resources(
    request, event_manager, simple_job_factory, timezone, declaration, nodes
):
    """Preserve the single-node pin without overriding explicit resource evidence.

    Args:
        request: Test identity for the event stream.
        event_manager: Event-store fixture.
        simple_job_factory: Factory for jobs with the selected node list.
        timezone: Observation timezone.
        declaration: Missing, null, empty, or invalid explicit resource declaration.
        nodes: One or two allocated nodes.
    Returns:
        None. Checks the node and job-level attribution sets.
    """
    job = simple_job_factory(nodes=nodes, state="FAILED", exit_code_signal=1)
    stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(stream, job)
    entry = {"body": "OSError: [Errno 5] Input/output error"}
    declarations = {
        "null": None,
        "empty": [],
        "outside": [{"kind": "node", "hostname": "outside-job"}],
        "multiple": [{"kind": "node", "hostname": name} for name in ("node-a", "node-b")],
    }
    if declaration != "omitted":
        entry["resources"] = declarations[declaration]
    attributor.ingest(
        Observation(
            Context(TimeInterval(None, None), AllJobResources, Source.APPLOG, Format.ENTRY_TIMESERIES, timezone),
            [entry],
        )
    )
    result = attributor.attribute()
    groups = result.bad_nodes_attr.my_bad_nodes
    assert "IO error" in result.symptoms
    if declaration in ("omitted", "null") and len(nodes) == 1:
        assert "IO error" in groups["node-a"].symptoms()
        assert "IO error" not in groups.job_level
    else:
        assert "IO error" in groups.job_level
        assert groups.num_nodes() == 0


@pytest.mark.parametrize("prefix", ["outside-job: ", "outside-job:1:2 [0] ", "OSError:1:2 [0] "])
@pytest.mark.parametrize("declaration", ["omitted", "null"])
def test_outside_body_hostname_does_not_infer_sole_node(
    request, event_manager, simple_job_factory, timezone, prefix, declaration
):
    """Retain explicit outside-job hostnames without blaming the sole job node.

    Args:
        request: Test identity for the event stream.
        event_manager: Event-store fixture.
        simple_job_factory: Factory producing a single-node job.
        timezone: Observation timezone.
        prefix: Plain or NCCL source hostname outside the job.
        declaration: Whether entry resources are omitted or explicitly null.
    Returns:
        None. Checks that the symptom remains job-level.
    """
    job = simple_job_factory(nodes=["node-a"], state="FAILED", exit_code_signal=1)
    stream = event_manager.create_stream(request.node.name)
    attributor = Attributor.new(stream, job)
    entry = {"body": prefix + "OSError: [Errno 5] Input/output error"}
    if declaration == "null":
        entry["resources"] = None
    attributor.ingest(
        Observation(
            Context(TimeInterval(None, None), AllJobResources, Source.APPLOG, Format.ENTRY_TIMESERIES, timezone),
            [entry],
        )
    )
    result = attributor.attribute()
    assert "IO error" in result.bad_nodes_attr.my_bad_nodes.job_level
    assert result.bad_nodes_attr.my_bad_nodes.num_nodes() == 0
