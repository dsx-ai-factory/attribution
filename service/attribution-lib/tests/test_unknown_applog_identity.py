# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unknown emitters must survive attribution without acquiring a job node."""

import json
from collections import defaultdict

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
    assert records
    if not expected_nodes:
        assert all(row[0] == "[]" for row in records)


@pytest.mark.parametrize("resources", [None, [], "AllJobResources"])
def test_explicit_all_resources_is_distinct_from_unknown(
    request, event_manager, simple_job_factory, timezone, resources
):
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
    assert records
    expected = "AllJobResources" if resources == "AllJobResources" else []
    assert all(json.loads(row[0]) == expected for row in records)


@pytest.mark.parametrize("resources", [None, []])
def test_unknown_emitter_preserves_timeout_reason(request, event_manager, simple_job_factory, timezone, resources):
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
    assert any(source.datum == body and not source.resources for source in sources)
