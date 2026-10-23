# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from attribution_lib.attributor import Attributor
from attribution_lib.core import CalcBadNodesAttributionResult, attribute
from attribution_lib.execution_cancellation import ExecutionCancellation
from attribution_lib.job_info import Scheduler
from attribution_lib.scheduler_failure import SchedulerFailure
from attribution_lib.sources import SourceSet


def cancellation(job, actor="user-a"):
    source = job.source("Associated execution cancellation", "execution_cancellation_evidence")._replace(
        time=job.interest_end_time
    )
    return ExecutionCancellation(job.id, actor, source)


@pytest.mark.parametrize("state", ["NODE_FAIL", "RUNNING", "COMPLETED", "PREEMPTED", "FAILED", "TIMEOUT"])
def test_cancellation_is_independent_of_containing_allocation_state(state, simple_job_factory, event_manager):
    job = simple_job_factory(job_id="execution", nodes=["node-b"], state=state, failed_node="node-a")
    failure = SchedulerFailure("allocation", "NODE_FAIL", "node-a", job.source("NODE_FAIL", "allocation_failure"))
    evidence = cancellation(job)
    job = job._replace(
        allocation_failure=failure,
        execution_cancellation=evidence,
        tags=frozenset({"nvrx"}),
        exit_code=0,
        exit_code_signal=0,
    )
    result = Attributor.new(event_manager.create_stream(job.id), job).attribute()
    assert result.outcome_attr.value().outcome == "Cancelled by User"
    assert result.cancelled_by == "user-a"
    assert result.outcome_attr.one_source() == evidence.source
    assert job.state == state and job.cancelled_by is None
    assert job.allocation_failure is failure
    assert list(result.bad_nodes_attr.my_bad_nodes) == []


@pytest.mark.parametrize(
    "actor,outcome", [("root", "Fail User"), ("site-automation-account", "Cancelled by cluster operators")]
)
def test_cancellation_actor_keeps_existing_semantics(actor, outcome, simple_job_factory, event_manager):
    job = simple_job_factory(job_id="execution", nodes=["node-b"], state="FAILED")
    job = job._replace(execution_cancellation=cancellation(job, actor))
    result = Attributor.new(event_manager.create_stream(job.id), job).attribute()
    assert result.outcome_attr.value().outcome == outcome
    assert result.cancelled_by == actor


@pytest.mark.parametrize("state", ["FAILED", "NODE_FAIL"])
def test_independent_fault_takes_precedence(simple_job_factory, event_manager, dummy_ctx, state):
    job = simple_job_factory(job_id="execution", nodes=["node-b"], state=state, failed_node="node-a")
    job = job._replace(
        execution_cancellation=cancellation(job),
        allocation_failure=SchedulerFailure(
            "allocation", "NODE_FAIL", "node-a", job.source("NODE_FAIL", "allocation_failure")
        ),
    )
    event_manager.preload_rule_names(["independent_fault"])
    stream = event_manager.create_stream(job.id)
    stream.register_observation_context(dummy_ctx)
    stream.add_event(
        "Attribution.node.bad",
        {"node": "node-b", "symptom": "XID-79"},
        job.source("Independent fault", "independent_fault")._replace(source_context=dummy_ctx),
    )
    result = Attributor.new(stream, job).attribute()
    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    assert "node-b" in result.bad_nodes_attr.my_bad_nodes


def test_cancellation_preserves_kueue_node_failure_without_a_named_node(simple_job_factory, event_manager):
    job = simple_job_factory(
        job_id="execution", nodes=["node-a", "node-b"], state="NODE_FAIL", scheduler=Scheduler.KUEUE
    )
    assert not job.failed_node
    job = job._replace(execution_cancellation=cancellation(job))
    result = Attributor.new(event_manager.create_stream(job.id), job).attribute()
    assert result.outcome_attr.value().outcome == "Fail InfraSW"
    assert "Node failure reported by slurm" in result.symptoms
    assert {source.rule for source in result.symptoms["Node failure reported by slurm"].all_sources()} == {
        "slurm state"
    }
    assert job.execution_cancellation.source in result.symptoms["Cancelled by user"].all_sources()


def test_mismatched_execution_is_rejected(simple_job_factory, event_manager):
    job = simple_job_factory(job_id="execution", nodes=["node-b"], state="FAILED")
    job = job._replace(execution_cancellation=replace(cancellation(job), execution_id="another-execution"))
    with pytest.raises(ValueError, match="different execution"):
        Attributor.new(event_manager.create_stream(job.id), job).attribute()


@pytest.mark.parametrize("actor", ["", " "])
def test_unknown_identity_is_not_a_user_cancellation(actor, simple_job_factory):
    job = simple_job_factory(job_id="execution", nodes=["node-b"], state="FAILED")
    with pytest.raises(ValueError, match="identity"):
        cancellation(job, actor)


def test_missing_timestamp_is_rejected(simple_job_factory):
    job = simple_job_factory(job_id="execution", nodes=["node-b"], state="FAILED")
    evidence = cancellation(job)
    with pytest.raises(ValueError, match="timestamp"):
        replace(evidence, source=evidence.source._replace(time=None))


@pytest.mark.parametrize(
    "keyword",
    ["log file is not recognized from slurm cmdline", "FileNotFoundError", "PermissionError", "log file too large"],
)
@pytest.mark.parametrize(
    "actor,outcome",
    [
        ("user-a", "Cancelled by User"),
        ("root", "Fail User"),
        ("site-automation-account", "Cancelled by cluster operators"),
    ],
)
def test_collection_metadata_cannot_replace_associated_cancellation(
    simple_job_factory, event_manager, keyword, actor, outcome
):
    job = simple_job_factory(job_id="execution", nodes=["node-b"], state="NODE_FAIL", failed_node="node-a")
    evidence = cancellation(job, actor)
    job = job._replace(
        execution_cancellation=evidence,
        allocation_failure=SchedulerFailure(
            "allocation", "NODE_FAIL", "node-a", job.source("NODE_FAIL", "allocation_failure")
        ),
    )
    bad_nodes = CalcBadNodesAttributionResult()
    source = job.source(keyword, "log_collection")
    bad_nodes.raw_keywords.add(SourceSet.new_one(keyword, source))
    result = attribute(
        event_stream=event_manager.create_stream(job.id),
        job_info=job,
        bad_node_result=bad_nodes,
        extract_job_metadata_from_logfile=False,
    )
    assert result.outcome_attr.value().outcome == outcome
    assert result.outcome_attr.one_source() == evidence.source
    assert keyword in bad_nodes.raw_keywords


def test_kueue_node_failure_after_root_cancellation_is_unchanged_without_cancellation_evidence(
    simple_job_factory, event_manager
):
    job = simple_job_factory(
        job_id="execution", nodes=["node-a", "node-b"], state="NODE_FAIL", scheduler=Scheduler.KUEUE
    )._replace(cancelled_by="root")

    result = Attributor.new(event_manager.create_stream(job.id), job).attribute()

    assert "Cancelled by root" in result.symptoms
    assert "Node failure reported by slurm" not in result.symptoms
