# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution of jobs whose scheduler facts arrive in a SCHEDULER observation."""

import pytest
from test_support.helpers import make_applog_entry, make_syslog_entry

from attribution_lib.attributor import Attributor
from attribution_lib.bundled_rules.common.scheduler_evidence import job_with_scheduler_evidence
from attribution_lib.observation.inbound import Context, Format, Observation, Source, TimeInterval

JOB_ID = "execution"
NVRX_TAG_ENTRY = {"body": "job tag: nvrx"}
EVICTION = (
    "LustreError: 167-0: testfs-OST001c-osc-ffff94dc5684f800: This client was evicted by testfs-OST001c; "
    "in progress operations using this service will fail."
)
CUDA_DEVICE_COUNT = (
    "RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling "
    "NumCudaDevices() that might have already set an error? Error 804: forward compatibility was attempted on "
    "non supported HW"
)


def _end_ns(job):
    """Return the interest end of the job in Unix nanoseconds.

    Args:
        job: Job under test.

    Returns:
        Timestamp for an observation entry.
    """
    return int(job.interest_end_time.timestamp()) * 10**9


def _allocation_failure(job):
    """Build the entry for a NODE_FAIL of the containing allocation on a node outside the job.

    Args:
        job: Job under test.

    Returns:
        Scheduler observation entry.
    """
    return {"body": "allocation failure: allocation=allocation state=NODE_FAIL node=node-a", "ts": _end_ns(job)}


def _cancellation(job, actor="user-a"):
    """Build the entry for a cancellation of this execution.

    Args:
        job: Job under test.
        actor: Identity that cancelled the execution.

    Returns:
        Scheduler observation entry.
    """
    return {"body": f"execution cancellation: execution={JOB_ID} cancelled_by={actor}", "ts": _end_ns(job)}


@pytest.fixture
def attribute(event_manager, timezone, request):
    """Attribute a job from observations through the Attributor and the bundled rules.

    Args:
        event_manager: Event manager with the bundled rules.
        timezone: Timezone of the observations.
        request: Test request, which names the event stream.

    Returns:
        Function that ingests the entries of each source and returns the attribution result.
    """

    def _attribute(job, **entries_by_source):
        attributor = Attributor.new(event_manager.create_stream(request.node.name), job)
        for source_name, entries in entries_by_source.items():
            context = Context(
                time_interval=TimeInterval(job.job_start_time, None),
                source=Source[source_name.upper()],
                format=Format.ENTRY_TIMESERIES,
                timezone=timezone,
            )
            attributor.ingest(Observation(context=context, lines=entries))
        return attributor.attribute()

    return _attribute


@pytest.mark.parametrize("state", ["NODE_FAIL", "RUNNING", "COMPLETED", "PREEMPTED", "FAILED", "TIMEOUT"])
def test_observed_cancellation_decides_the_execution_in_any_allocation_state(state, simple_job_factory, attribute):
    """A cancellation of this execution gives the outcome, and the inherited NODE_FAIL blames no node."""
    job = simple_job_factory(job_id=JOB_ID, nodes=["node-b"], state=state, failed_node="node-a")

    result = attribute(job, scheduler=[_allocation_failure(job), _cancellation(job)])

    assert result.outcome_attr.value().outcome == "Cancelled by User"
    assert result.cancelled_by == "user-a"
    assert result.outcome_attr.one_source().rule == "SchedulerExecutionCancellation"
    assert list(result.bad_nodes_attr.my_bad_nodes) == []


@pytest.mark.parametrize(
    ("actor", "outcome"),
    [("root", "Fail User"), ("site-automation-account", "Cancelled by cluster operators")],
    ids=["root", "cluster-operators"],
)
def test_observed_cancellation_actor_keeps_existing_semantics(actor, outcome, simple_job_factory, attribute):
    """The identity in the observation selects the same outcome as the JobInfo input."""
    job = simple_job_factory(job_id=JOB_ID, nodes=["node-b"], state="FAILED")

    result = attribute(job, scheduler=[_cancellation(job, actor)])

    assert result.outcome_attr.value().outcome == outcome
    assert result.cancelled_by == actor


@pytest.mark.parametrize(
    ("entries", "outcome", "bad_nodes"),
    [
        (["failure"], "Fail InfraSW", ["node-a"]),
        (["cancellation"], "Fail InfraSW", ["node-a"]),
    ],
    ids=["allocation-failure-alone", "cancellation-alone"],
)
def test_inherited_node_failure_needs_both_facts_to_be_suppressed(
    entries, outcome, bad_nodes, simple_job_factory, attribute
):
    """One fact alone keeps the scheduler node failure, as with the JobInfo inputs."""
    job = simple_job_factory(job_id=JOB_ID, nodes=["node-b"], state="NODE_FAIL", failed_node="node-a")
    facts = {"failure": _allocation_failure(job), "cancellation": _cancellation(job)}

    result = attribute(job, scheduler=[facts[name] for name in entries])

    assert result.outcome_attr.value().outcome == outcome
    assert list(result.bad_nodes_attr.my_bad_nodes) == bad_nodes


@pytest.mark.parametrize(
    ("state", "entries", "outcome"),
    [
        ("PREEMPTED", [NVRX_TAG_ENTRY, {"body": "job exit code: 1"}], "Fail User"),
        ("PREEMPTED", [{"body": "job exit code: 1"}], "Pass"),
        ("PREEMPTED", [NVRX_TAG_ENTRY, {"body": "job exit code: 0"}], "Pass"),
        ("TIMEOUT", [NVRX_TAG_ENTRY, {"body": "job exit code: 1"}], "Fail User"),
        ("TIMEOUT", [{"body": "job exit code: 1"}], "Pass"),
    ],
    ids=[
        "preempted-nvrx-failure",
        "preempted-untagged-failure",
        "preempted-nvrx-success",
        "timeout-nvrx-failure",
        "timeout-untagged-failure",
    ],
)
def test_nonzero_launcher_exit_blocks_pass_only_for_a_tagged_job(
    state, entries, outcome, simple_job_factory, attribute
):
    """A nonzero NVRx launcher exit code keeps a preempted or timed-out job from Pass."""
    job = simple_job_factory(job_id=JOB_ID, nodes=["node-b"], state=state)

    assert attribute(job, scheduler=entries).outcome_attr.value().outcome == outcome


@pytest.mark.parametrize(
    ("entries", "outcome"),
    [
        ([NVRX_TAG_ENTRY, {"body": "job exit code: 0"}], "Pass"),
        ([{"body": "job exit code: 0"}], "Fail Undetermined"),
        ([NVRX_TAG_ENTRY, {"body": "job exit code: 1"}], "Fail Undetermined"),
    ],
    ids=["nvrx-success", "untagged-success", "nvrx-failure"],
)
def test_clean_launcher_exit_resolves_an_ambiguous_outcome_for_a_tagged_job(
    entries, outcome, simple_job_factory, attribute
):
    """A clean NVRx launcher exit turns Fail Undetermined into Pass."""
    job = simple_job_factory(job_id=JOB_ID, nodes=["node-b"], state="FAILED")
    syslog = [make_syslog_entry(EVICTION, "node-b", _end_ns(job) - 60 * 10**9)]

    assert attribute(job, scheduler=entries, syslog=syslog).outcome_attr.value().outcome == outcome


@pytest.mark.parametrize(
    ("entries", "symptoms"),
    [
        ([{"body": "job gpu count: 0"}], ["job failed"]),
        ([{"body": "job gpu count: 8"}], ["Unexpected error from cudaGetDeviceCount()"]),
        ([], ["Unexpected error from cudaGetDeviceCount()"]),
    ],
    ids=["zero-gpus", "eight-gpus", "no-observation"],
)
def test_device_count_error_is_dropped_only_for_a_job_without_gpus(entries, symptoms, simple_job_factory, attribute):
    """The GPU count in the observation decides if the device-count error counts."""
    job = simple_job_factory(job_id=JOB_ID, nodes=["node-b"], state="FAILED")
    applog = [make_applog_entry(CUDA_DEVICE_COUNT, _end_ns(job) - 10**9, "node-b")]
    observations = {"applog": applog} if entries == [] else {"scheduler": entries, "applog": applog}

    assert sorted(attribute(job, **observations).symptoms.symptoms()) == symptoms


def test_observation_facts_replace_job_info_inputs_and_add_tags(simple_job_factory, event_manager, timezone):
    """A later entry replaces an earlier one, and tags join the tags of the job."""
    job = simple_job_factory(job_id=JOB_ID, nodes=["node-b"], state="FAILED")._replace(
        tags=frozenset({"batch"}), exit_code=7
    )
    stream = event_manager.create_stream("facts")
    context = Context(
        time_interval=TimeInterval(job.job_start_time, None),
        source=Source.SCHEDULER,
        format=Format.ENTRY_TIMESERIES,
        timezone=timezone,
    )
    entries = [NVRX_TAG_ENTRY, {"body": "job exit code: 1"}, {"body": "job exit code: 0"}, {"body": "not a fact"}]
    Attributor.new(stream, job).ingest(Observation(context=context, lines=entries))

    observed = job_with_scheduler_evidence(job, stream)

    assert observed.tags == frozenset({"batch", "nvrx"})
    assert observed.exit_code == 0
    assert observed.num_gpus == job.num_gpus
    assert observed.allocation_failure is None
    assert observed.execution_cancellation is None


def test_job_without_scheduler_observation_is_unchanged(simple_job_factory, event_manager):
    """A client that fills the JobInfo inputs directly keeps them."""
    job = simple_job_factory(job_id=JOB_ID, nodes=["node-b"], state="FAILED")._replace(exit_code=7)

    assert job_with_scheduler_evidence(job, event_manager.create_stream("empty")) is job
    assert job_with_scheduler_evidence(job, None) is job


def test_later_observation_replaces_a_fact_of_an_earlier_observation(simple_job_factory, event_manager, timezone):
    """Entries of a later observation apply after the entries of an earlier one, whatever their line numbers."""
    job = simple_job_factory(job_id=JOB_ID, nodes=["node-b"], state="FAILED")
    stream = event_manager.create_stream("two-observations")
    context = Context(
        time_interval=TimeInterval(job.job_start_time, None),
        source=Source.SCHEDULER,
        format=Format.ENTRY_TIMESERIES,
        timezone=timezone,
    )
    attributor = Attributor.new(stream, job)
    attributor.ingest(Observation(context=context, lines=[NVRX_TAG_ENTRY, {"body": "job exit code: 1"}]))
    attributor.ingest(Observation(context=context, lines=[{"body": "job exit code: 0"}]))

    assert job_with_scheduler_evidence(job, stream).exit_code == 0


@pytest.mark.parametrize(
    ("execution", "has_ts"), [("another-execution", True), (JOB_ID, False)], ids=["another-execution", "no-timestamp"]
)
def test_invalid_observed_cancellation_is_ignored(execution, has_ts, simple_job_factory, attribute, caplog):
    """A cancellation for another execution or without ``ts`` does not fail the attribution."""
    job = simple_job_factory(job_id=JOB_ID, nodes=["node-b"], state="FAILED")
    entry = {"body": f"execution cancellation: execution={execution} cancelled_by=user-a"}
    if has_ts is True:
        entry["ts"] = _end_ns(job)

    result = attribute(job, scheduler=[entry])

    assert result.outcome_attr.value().outcome != "Cancelled by User"
    assert "Ignoring a cancellation" in caplog.text
