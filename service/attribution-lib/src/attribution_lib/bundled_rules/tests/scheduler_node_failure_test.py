# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SchedulerNodeFailure with allocation NODE_FAIL evidence owned by an NVRx cycle."""

from collections import defaultdict
from dataclasses import replace

import pytest

import attribution_lib
from attribution_lib.attributor import Attributor
from attribution_lib.scheduler_failure import SchedulerFailure


@pytest.fixture
def failure(simple_job_factory):
    """Allocation NODE_FAIL evidence that names node-a.

    Args:
        simple_job_factory: Factory for JobInfo instances.

    Returns:
        The allocation's scheduler failure.
    """
    allocation = simple_job_factory(job_id="allocation", nodes=["node-a", "node-b"], state="NODE_FAIL")
    return SchedulerFailure(
        "allocation", "NODE_FAIL", "node-a", allocation.source("NODE_FAIL node-a", "slurm_failed_node_field")
    )


def _summary(result):
    """Reduce a result to its outcome, symptoms, and bad nodes.

    Args:
        result: Attribution result.

    Returns:
        A comparable summary tuple.
    """
    return (
        result.outcome_attr.value().outcome,
        sorted(result.symptoms.symptoms()),
        sorted(result.bad_nodes_attr.my_bad_nodes.attributions),
    )


def test_allocation_failure_owned_by_another_cycle_does_not_change_attribution(
    failure, simple_job_factory, event_manager
):
    """A cycle that does not own the allocation failure keeps its legacy attribution.

    Args:
        failure: Allocation NODE_FAIL evidence for node-a.
        simple_job_factory: Factory for JobInfo instances.
        event_manager: Event manager fixture.
    """
    job = simple_job_factory(job_id="cycle-0", nodes=["node-a", "node-b"], state="FAILED")
    legacy = Attributor.new(event_manager.create_stream("legacy"), job).attribute()
    other_owner = job._replace(allocation_failure=replace(failure, owner_execution_id="cycle-1"))

    supplied = Attributor.new(event_manager.create_stream("supplied"), other_owner).attribute()

    assert _summary(supplied) == _summary(legacy)


def _stream_with_sibling_cascades(event_manager, dummy_ctx, job, name):
    """Create a stream where node-b reports NVL-domain cascade XIDs.

    Args:
        event_manager: Event manager fixture.
        dummy_ctx: Observation context for the events.
        job: JobInfo that supplies the event sources.
        name: Stream name.

    Returns:
        The populated event stream.
    """
    event_manager.preload_rule_names(["sibling_xid_observation"])
    stream = event_manager.create_stream(name)
    stream.register_observation_context(dummy_ctx)
    for symptom in ("XID-145-RLW_SRC_TRACK-8", "XID-44-GR_FAULT_DURING_CTXSW"):
        stream.add_event(
            "Attribution.node.bad",
            {"node": "node-b", "symptom": symptom},
            job.source("sibling cascade", "sibling_xid_observation")._replace(source_context=dummy_ctx),
        )
    return stream


def test_owned_allocation_failure_prunes_cascades_on_same_domain_siblings(
    failure, simple_job_factory, event_manager, dummy_ctx, monkeypatch
):
    """The owning cycle's node failure prunes cascade XIDs from its NVL-domain sibling.

    Args:
        failure: Allocation NODE_FAIL evidence for node-a.
        simple_job_factory: Factory for JobInfo instances.
        event_manager: Event manager fixture.
        dummy_ctx: Observation context for the events.
        monkeypatch: Pytest fixture used to restore process-global configuration.
    """
    monkeypatch.setattr(attribution_lib, "node_domain_resolver", {"node-a": "rack-1", "node-b": "rack-1"}.get)
    job = simple_job_factory(job_id="cycle-1", nodes=["node-a", "node-b"], state="FAILED")
    legacy = Attributor.new(_stream_with_sibling_cascades(event_manager, dummy_ctx, job, "legacy"), job).attribute()
    owned = job._replace(allocation_failure=replace(failure, owner_execution_id=job.id))

    result = Attributor.new(_stream_with_sibling_cascades(event_manager, dummy_ctx, job, "owned"), owned).attribute()

    assert _bad_and_interesting(legacy)[0] == {"node-b": {"XID-145-RLW_SRC_TRACK-8", "XID-44-GR_FAULT_DURING_CTXSW"}}
    bad, interesting = _bad_and_interesting(result)
    assert bad == {"node-a": {"Node failure reported by slurm"}}
    assert interesting["node-b"] == {"XID-145-RLW_SRC_TRACK-8", "XID-44-GR_FAULT_DURING_CTXSW"}
    assert result.outcome_attr.value().outcome == "Fail InfraSW"
    sources = result.bad_nodes_attr.my_bad_nodes["node-a"].attributions["Node failure reported by slurm"]
    assert {source.rule for source in sources.all_sources()} == {"cycle_owned_allocation_failure"}


def _bad_and_interesting(result):
    """Split per-node symptoms the way the bad-node and interesting-node reports do.

    Args:
        result: Attribution result.

    Returns:
        A ``(bad, interesting)`` pair of node-to-symptom-set mappings.
    """
    bad, interesting, epilog = defaultdict(set), defaultdict(set), defaultdict(set)
    result.bad_nodes_attr.my_bad_nodes.extract_bad_interesting_epilog(bad, interesting, epilog)
    return dict(bad), dict(interesting)
