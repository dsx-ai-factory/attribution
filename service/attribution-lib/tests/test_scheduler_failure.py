# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import replace

import pytest

import attribution_lib
from attribution_lib.attributor import Attributor
from attribution_lib.core import CalcBadNodesAttributionResult, attribute
from attribution_lib.execution_cancellation import ExecutionCancellation
from attribution_lib.scheduler_failure import SchedulerFailure
from attribution_lib.sources import SourceSet


@pytest.fixture
def failure(simple_job_factory):
    allocation = simple_job_factory(job_id="allocation", nodes=["node-a", "node-b"], state="NODE_FAIL")
    return SchedulerFailure(
        "allocation", "NODE_FAIL", "node-a", allocation.source("NODE_FAIL node-a", "slurm_failed_node_field")
    )


@pytest.mark.parametrize("failed_node", ["node-b", "NODE-B.example.test"])
def test_separate_participating_node_failure_is_not_masked_by_cancellation(
    failure, simple_job_factory, event_manager, failed_node
):
    job = simple_job_factory(job_id="execution", nodes=["node-b"], state="NODE_FAIL", failed_node=failed_node)
    cancellation = ExecutionCancellation(
        job.id,
        "user-a",
        job.source("execution cancelled", "execution_cancellation")._replace(time=job.interest_end_time),
    )
    job = job._replace(allocation_failure=failure, execution_cancellation=cancellation)

    result = Attributor.new(event_manager.create_stream(job.id), job).attribute()

    assert not job.inherited_node_failure
    assert not job.suppress_inherited_node_failure
    assert result.outcome_attr.value().outcome == "Fail InfraSW"
    assert "Node failure reported by slurm" in result.bad_nodes_attr.my_bad_nodes[failed_node.casefold()]
    assert "node-a" not in result.bad_nodes_attr.my_bad_nodes
    assert job.execution_cancellation is cancellation
    assert job.allocation_failure is failure


@pytest.mark.parametrize("failed_node", ["node-a", "NODE-A.example.test"])
def test_same_allocation_node_still_suppresses_with_cancellation(
    failure, simple_job_factory, event_manager, failed_node
):
    job = simple_job_factory(job_id="execution", nodes=["node-b"], state="NODE_FAIL", failed_node=failed_node)
    job = job._replace(
        allocation_failure=failure,
        execution_cancellation=ExecutionCancellation(
            job.id,
            "user-a",
            job.source("execution cancelled", "execution_cancellation")._replace(time=job.interest_end_time),
        ),
    )

    result = Attributor.new(event_manager.create_stream(job.id), job).attribute()

    assert job.inherited_node_failure
    assert job.suppress_inherited_node_failure
    assert result.outcome_attr.value().outcome == "Cancelled by User"
    assert list(result.bad_nodes_attr.my_bad_nodes) == []
    assert job.allocation_failure is failure


@pytest.mark.parametrize("clean_exit", [False, True])
@pytest.mark.parametrize("associated_cancellation", [False, True])
def test_excluded_node_suppression_requires_associated_cancellation(
    failure, simple_job_factory, event_manager, clean_exit, associated_cancellation
):
    job = simple_job_factory(job_id="cycle-4", nodes=["node-b"], state="NODE_FAIL", failed_node="node-a")
    job = job._replace(
        allocation_failure=failure, tags=frozenset({"nvrx"}), exit_code=0 if clean_exit else 1, exit_code_signal=0
    )
    if associated_cancellation:
        job = job._replace(
            execution_cancellation=ExecutionCancellation(
                job.id,
                "user-a",
                job.source("cycle cancelled", "execution_cancellation")._replace(time=job.interest_end_time),
            )
        )
    result = Attributor.new(event_manager.create_stream(job.id), job).attribute()
    assert job.inherited_node_failure
    assert job.suppress_inherited_node_failure is associated_cancellation
    assert result.outcome_attr.value().outcome == ("Cancelled by User" if associated_cancellation else "Fail InfraSW")
    if associated_cancellation:
        assert "node-a" not in result.bad_nodes_attr.my_bad_nodes
    else:
        assert "Node failure reported by slurm" in result.bad_nodes_attr.my_bad_nodes["node-a"]
    assert job.state == "NODE_FAIL"
    assert job.failed_node == "node-a"
    assert job.allocation_failure is failure


@pytest.mark.parametrize("nodes", [["node-a"], ["node-a", "node-b"]])
@pytest.mark.parametrize("clean_exit", [False, True])
def test_in_set_allocation_failure_preserves_existing_cycle_attribution(
    failure, simple_job_factory, event_manager, nodes, clean_exit
):
    job = simple_job_factory(job_id="cycle-4", nodes=nodes, state="NODE_FAIL", failed_node="node-a")
    job = job._replace(tags=frozenset({"nvrx"}), exit_code=0 if clean_exit else 1, exit_code_signal=0)
    baseline = Attributor.new(event_manager.create_stream("baseline"), job).attribute()
    job = job._replace(allocation_failure=failure)
    result = Attributor.new(event_manager.create_stream(job.id), job).attribute()
    assert not job.inherited_node_failure
    assert result.outcome_attr.value() == baseline.outcome_attr.value()
    assert result.outcome_attr.value().outcome == "Fail InfraSW"
    assert "Node failure reported by slurm" in result.bad_nodes_attr.my_bad_nodes["node-a"]


@pytest.mark.parametrize(
    "nodes,failed_node",
    [
        ([], "node-a"),
        (["node-b"], ""),
        (["NODE-A"], "node-a"),
        (["node-a"], "node-a.example.test"),
        (["node-a.example.test"], "node-a"),
    ],
)
def test_unknown_or_matching_membership_does_not_activate_suppression(simple_job_factory, nodes, failed_node):
    job = simple_job_factory(job_id="cycle-4", nodes=nodes, state="NODE_FAIL", failed_node=failed_node)
    job = job._replace(
        allocation_failure=SchedulerFailure(
            "allocation", "NODE_FAIL", failed_node, job.source("NODE_FAIL", "allocation_failure")
        )
    )
    assert not job.inherited_node_failure


def test_allocation_evidence_alone_matches_legacy_with_missing_log(failure, simple_job_factory, event_manager):
    job = simple_job_factory(job_id="cycle-4", nodes=["node-b"], state="NODE_FAIL", failed_node="node-a")
    source = job.source("Application log format unknown", "independent_log_evidence")
    results = []
    for name, candidate in [("legacy", job), ("ownership", job._replace(allocation_failure=failure))]:
        bad_nodes = CalcBadNodesAttributionResult()
        bad_nodes.raw_keywords.add(SourceSet.new_one("log file is not recognized from slurm cmdline", source))
        results.append(
            attribute(
                event_stream=event_manager.create_stream(name),
                job_info=candidate,
                bad_node_result=bad_nodes,
                extract_job_metadata_from_logfile=False,
            )
        )
    assert results[0].outcome_attr.value() == results[1].outcome_attr.value()
    assert results[1].outcome_attr.value().outcome == "Fail InfraSW"
    assert set(results[0].outcome_attr.all_sources()) == set(results[1].outcome_attr.all_sources())


@pytest.mark.parametrize("explicit", [False, True])
def test_allocation_itself_keeps_node_failure(failure, simple_job_factory, event_manager, explicit):
    job = simple_job_factory(job_id="allocation", nodes=["node-a", "node-b"], state="NODE_FAIL", failed_node="node-a")
    job = job._replace(allocation_failure=failure if explicit else None)
    result = Attributor.new(event_manager.create_stream(job.id), job).attribute()
    assert "Node failure reported by slurm" in result.bad_nodes_attr.my_bad_nodes["node-a"]
    assert result.outcome_attr.value().outcome == "Fail InfraSW"


@pytest.mark.parametrize("state", ["FAILED", "COMPLETED"])
def test_other_cycle_states_are_unchanged(failure, simple_job_factory, event_manager, state):
    job = simple_job_factory(job_id="cycle-3", nodes=["node-b"], state=state)
    baseline = Attributor.new(event_manager.create_stream("baseline"), job).attribute()
    scoped_cycle = job._replace(allocation_failure=failure)
    result = Attributor.new(event_manager.create_stream("scoped"), scoped_cycle).attribute()
    assert result.outcome_attr.value() == baseline.outcome_attr.value()
    assert not scoped_cycle.inherited_node_failure


@pytest.mark.parametrize("symptom", ["System-out of memory", "Node failure reported by slurm"])
def test_independent_cycle_fault_is_not_suppressed(failure, simple_job_factory, event_manager, dummy_ctx, symptom):
    job = simple_job_factory(job_id="cycle-4", nodes=["node-b"], state="NODE_FAIL")
    job = job._replace(
        allocation_failure=failure,
        execution_cancellation=ExecutionCancellation(
            job.id,
            "user-a",
            job.source("cycle cancelled", "execution_cancellation")._replace(time=job.interest_end_time),
        ),
    )
    event_manager.preload_rule_names(["independent_cycle_observation"])
    stream = event_manager.create_stream(job.id)
    stream.register_observation_context(dummy_ctx)
    stream.add_event(
        "Attribution.node.bad",
        {"node": "node-b", "symptom": symptom},
        job.source("independent cycle fault", "independent_cycle_observation")._replace(source_context=dummy_ctx),
    )
    result = Attributor.new(stream, job).attribute()
    assert result.outcome_attr.value().outcome != "Incomplete Info"
    assert symptom in result.bad_nodes_attr.my_bad_nodes["node-b"]
    sources = result.bad_nodes_attr.my_bad_nodes["node-b"].attributions[symptom]
    assert {source.rule for source in sources.all_sources()} == {"independent_cycle_observation"}


def _summary(result):
    return (
        result.outcome_attr.value().outcome,
        sorted(result.symptoms.symptoms()),
        sorted(result.bad_nodes_attr.my_bad_nodes.attributions),
    )


@pytest.mark.parametrize("state", ["NODE_FAIL", "FAILED", "COMPLETED", "CANCELLED", "TIMEOUT"])
@pytest.mark.parametrize("nodes", [["node-b"], ["node-a", "node-b"]])
@pytest.mark.parametrize("failed_node", [None, "node-a"])
def test_allocation_failure_alone_does_not_change_attribution(
    failure, simple_job_factory, event_manager, state, nodes, failed_node
):
    job = simple_job_factory(job_id="execution", nodes=nodes, state=state, failed_node=failed_node)
    legacy = Attributor.new(event_manager.create_stream("legacy"), job).attribute()
    supplied = Attributor.new(
        event_manager.create_stream("supplied"), job._replace(allocation_failure=failure)
    ).attribute()

    assert _summary(supplied) == _summary(legacy)


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
