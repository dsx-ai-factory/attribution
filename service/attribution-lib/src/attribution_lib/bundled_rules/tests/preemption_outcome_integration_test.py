# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end outcomes for Slurm jobs that end PREEMPTED.

All jobs and logs here are synthetic. The cases follow one reference chronology:
a worker fails, then the scheduler preempts the job about 90 s later and kills
it about 40 s after that.
"""

from datetime import datetime, timedelta

import pytest

from attribution_lib.attributor import Attributor
from attribution_lib.job_info import Scheduler
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Observation as InboundObservation,
    Source as ObservationSource,
    TimeInterval,
)
from test_support.helpers import make_applog_entry, make_syslog_entry, make_tflops_line

_NODES = ["node01", "node02", "node03", "node04"]
_FAILED_NODE = "node03"


def _preempted_job(simple_job_factory, timezone, scheduler=Scheduler.SLURM):
    """Build a one-hour job that ended PREEMPTED.

    Args:
        simple_job_factory: Fixture that builds a JobInfo.
        timezone: Fixture timezone.
        scheduler: Scheduler that ran the job.

    Returns:
        The job info.
    """
    job_start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone)
    job_end = job_start + timedelta(minutes=60)
    return simple_job_factory(
        job_id="test-job-preempted",
        state="PREEMPTED",
        start_timestamp=int(job_start.timestamp()),
        end_timestamp=int(job_end.timestamp()),
        nodes=_NODES,
        time_limit=120,
        exit_code_signal=0,
        scheduler=scheduler,
    )


def _observation(job_info, timezone, source, lines):
    """Wrap entries in an inbound observation over the job interval.

    Args:
        job_info: The job the entries belong to.
        timezone: Fixture timezone.
        source: Observation source.
        lines: Timeseries entries.

    Returns:
        The inbound observation.
    """
    return InboundObservation(
        context=InboundContext(
            time_interval=TimeInterval(job_info.job_start_time, job_info.job_end_time),
            source=source,
            format=ObservationFormat.ENTRY_TIMESERIES,
            timezone=timezone,
        ),
        lines=lines,
    )


def _xid_syslog(job_info, timezone, failure_time, node=_FAILED_NODE):
    """Build the XID-145 and XID-137 syslog lines of a failed NVLink.

    Args:
        job_info: The job the lines belong to.
        timezone: Fixture timezone.
        failure_time: Time of the first XID.
        node: Node that logs the XIDs.

    Returns:
        The syslog observation.
    """
    timestamp_ns = int(failure_time.timestamp() * 1_000_000_000)
    return _observation(
        job_info,
        timezone,
        ObservationSource.SYSLOG,
        [
            make_syslog_entry(
                body=(
                    "NVRM: Xid (PCI:0000:01:00): 145, pid=5678, name=python, "
                    "RLW_RXPIPE Nonfatal XC0 i0 Link 00 "
                    "(0x04080006 0x00000008 0x00000000 0x00000000 0x00000000 0x00000000)"
                ),
                node=node,
                timestamp_ns=timestamp_ns,
            ),
            make_syslog_entry(
                body=(
                    "NVRM: Xid (PCI:0000:01:00): 137, pid=5678, name=python, "
                    "RLW_RXPIPE interrupt hit on link 0 on GPU0: PRIV Error"
                ),
                node=node,
                timestamp_ns=timestamp_ns + 20_000,
            ),
        ],
    )


def _preemption_applog(job_info, timezone):
    """Build the line Slurm writes to the step log when it preempts the step.

    Slurm sends SIGTERM 43 s before the job end, as in the reference chronology.

    Args:
        job_info: The preempted job.
        timezone: Fixture timezone.

    Returns:
        The application-log observation.
    """
    sigterm_time = job_info.job_end_time - timedelta(seconds=43)
    return _observation(
        job_info,
        timezone,
        ObservationSource.APPLOG,
        [
            make_applog_entry(
                body=(
                    f"slurmstepd: error: *** STEP 424242.0 ON node01 CANCELLED AT "
                    f"{sigterm_time.strftime('%Y-%m-%dT%H:%M:%S')} DUE TO PREEMPTION ***"
                ),
                hostname="node01",
                timestamp_ns=int(sigterm_time.timestamp() * 1_000_000_000),
            )
        ],
    )


def _attribute(request, event_manager, job_info, *observations):
    """Ingest the observations and run attribution.

    Args:
        request: Pytest request, used to name the event stream.
        event_manager: Fixture that creates event streams.
        job_info: The job to attribute.
        *observations: Inbound observations to ingest.

    Returns:
        The attribution result.
    """
    attributor = Attributor.new(event_manager.create_stream(request.node.name), job_info)
    for observation in observations:
        attributor.ingest(observation)
    return attributor.attribute()


def _is_preemption_pass(result):
    """Return whether the result is the preemption Pass.

    Args:
        result: Attribution result.

    Returns:
        True when the outcome is Pass with the SLURM preempted root cause.
    """
    outcome = result.outcome_attr.value()
    return outcome.outcome == "Pass" and outcome.root_cause_symptom == "SLURM preempted"


def test_clean_preemption_is_pass(request, event_manager, simple_job_factory, timezone):
    """A preempted job with no failure evidence is Pass and shows the preemption."""
    job_info = _preempted_job(simple_job_factory, timezone)

    result = _attribute(request, event_manager, job_info)

    assert _is_preemption_pass(result)
    assert set(result.symptoms.symptoms()) == {"Job preempted"}


def test_slurm_preemption_cancel_line_is_not_a_user_cancel(request, event_manager, simple_job_factory, timezone):
    """The "CANCELLED AT ... DUE TO PREEMPTION" step-log line does not block the preemption Pass.

    Training logs progress until shortly before the preemption, so the line
    falls inside job execution, as it does in a real step log.
    """
    job_info = _preempted_job(simple_job_factory, timezone)
    progress = [
        make_applog_entry(
            body=make_tflops_line(time),
            hostname="node01",
            timestamp_ns=int(time.timestamp() * 1_000_000_000),
        )
        for time in (job_info.job_start_time + timedelta(minutes=minute) for minute in range(1, 59, 4))
    ]

    result = _attribute(
        request,
        event_manager,
        job_info,
        _observation(job_info, timezone, ObservationSource.APPLOG, progress),
        _preemption_applog(job_info, timezone),
    )

    assert _is_preemption_pass(result)
    assert "Cancelled by user" not in result.symptoms


@pytest.mark.parametrize(
    ("with_sigterm_line", "seconds_before_end", "expect_preemption_pass"),
    [
        # With the step-log preemption line, the teardown window starts at SIGTERM
        # (43 s before the job end). A failure 67 s before SIGTERM keeps its verdict.
        (True, 110, False),
        # An error 13 s after SIGTERM is teardown noise.
        (True, 30, True),
        # Without that line, the window starts a fixed 120 s before the job end.
        # The reference failure, 87 s before SIGTERM, is just outside it.
        (False, 130, False),
        # The same failure 110 s before the job end falls inside the window. The
        # engine cannot tell it from teardown noise, so the job stays Pass.
        (False, 110, True),
    ],
)
def test_failure_before_preemption_teardown_boundary(
    request, event_manager, simple_job_factory, timezone, with_sigterm_line, seconds_before_end, expect_preemption_pass
):
    """Errors before the preemption keep their verdict; teardown errors do not.

    ExternalTerminationCascade takes the termination time from the Slurm
    "CANCELLED AT ... DUE TO PREEMPTION" line. When the step log does not have
    that line, it assumes that teardown started 120 s before the job end.
    """
    job_info = _preempted_job(simple_job_factory, timezone)
    failure_time = job_info.job_end_time - timedelta(seconds=seconds_before_end)
    observations = [_xid_syslog(job_info, timezone, failure_time)]
    if with_sigterm_line is True:
        observations.append(_preemption_applog(job_info, timezone))

    result = _attribute(request, event_manager, job_info, *observations)

    assert _is_preemption_pass(result) is expect_preemption_pass
    assert result.outcome_attr.value().outcome.startswith("Fail") is not expect_preemption_pass
    assert ("Job preempted" in result.symptoms) is expect_preemption_pass
    assert "Cancelled by user" not in result.symptoms
    assert result.bad_nodes_attr.my_bad_nodes[_FAILED_NODE].attributions.has_bad() is not expect_preemption_pass


def test_pyxis_sigterm_after_preemption_is_teardown(request, event_manager, simple_job_factory, timezone):
    """The pyxis SIGTERM error that the preemption causes does not block the preemption Pass."""
    job_info = _preempted_job(simple_job_factory, timezone)
    pyxis_time = job_info.job_end_time - timedelta(seconds=42)
    progress = [
        make_applog_entry(
            body=make_tflops_line(time),
            hostname="node01",
            timestamp_ns=int(time.timestamp() * 1_000_000_000),
        )
        for time in (job_info.job_start_time + timedelta(minutes=minute) for minute in range(1, 59, 4))
    ]
    pyxis = make_applog_entry(
        body="slurmstepd: error: pyxis: could not wait for child 12345: Interrupted system call",
        hostname=_FAILED_NODE,
        timestamp_ns=int(pyxis_time.timestamp() * 1_000_000_000),
    )

    result = _attribute(
        request,
        event_manager,
        job_info,
        _observation(job_info, timezone, ObservationSource.APPLOG, [*progress, pyxis]),
        _preemption_applog(job_info, timezone),
    )

    assert _is_preemption_pass(result)
    assert "pyxis: cancelled by SIGTERM" not in result.symptoms


def test_failure_on_node_outside_allocation_is_ignored(request, event_manager, simple_job_factory, timezone):
    """Evidence from a node that the job did not run on does not block the preemption Pass."""
    job_info = _preempted_job(simple_job_factory, timezone)
    failure_time = job_info.job_end_time - timedelta(minutes=30)

    result = _attribute(request, event_manager, job_info, _xid_syslog(job_info, timezone, failure_time, node="node99"))

    assert _is_preemption_pass(result)


def test_early_failure_with_unknown_recovery_keeps_failure(request, event_manager, simple_job_factory, timezone):
    """An early failure with no recovery evidence keeps its verdict after preemption.

    The policy for unknown recovery is open. This test records the
    current behavior: unknown recovery counts as an unrecovered failure.
    """
    job_info = _preempted_job(simple_job_factory, timezone)
    failure_time = job_info.job_start_time + timedelta(minutes=10)

    result = _attribute(request, event_manager, job_info, _xid_syslog(job_info, timezone, failure_time))

    assert not _is_preemption_pass(result)
    assert result.outcome_attr.value().outcome.startswith("Fail")
    assert result.bad_nodes_attr.my_bad_nodes[_FAILED_NODE].attributions.has_bad()


@pytest.mark.parametrize(
    ("last_progress_minute", "expect_preemption_pass"),
    [
        # Training kept making progress for 44 minutes after the failure.
        (54, True),
        # Training stopped making progress before the failure.
        (9, False),
    ],
)
def test_early_failure_with_training_progress(
    request, event_manager, simple_job_factory, timezone, last_progress_minute, expect_preemption_pass
):
    """Training progress after a failure shows that the job recovered from it.

    DemoteTflopsSymptoms demotes demotable evidence logged before the last
    throughput line. When training kept going after the failure, the
    preemption Pass holds. When it stopped before the failure, the failure
    keeps its verdict.
    """
    job_info = _preempted_job(simple_job_factory, timezone)
    failure_time = job_info.job_start_time + timedelta(minutes=10)
    progress = [
        make_applog_entry(
            body=make_tflops_line(time),
            hostname="node01",
            timestamp_ns=int(time.timestamp() * 1_000_000_000),
        )
        for time in (
            job_info.job_start_time + timedelta(minutes=minute) for minute in range(1, last_progress_minute + 1, 4)
        )
    ]

    result = _attribute(
        request,
        event_manager,
        job_info,
        _xid_syslog(job_info, timezone, failure_time),
        _observation(job_info, timezone, ObservationSource.APPLOG, progress),
    )

    assert _is_preemption_pass(result) is expect_preemption_pass
    assert result.outcome_attr.value().outcome.startswith("Fail") is not expect_preemption_pass


@pytest.mark.parametrize(
    "body",
    [
        "0: omegaconf.errors.KeyValidationError: ListConfig indices must be integers or slices, not NoneType",
        "0: Segmentation fault",
        "0: ncclTimeout",
        "0: terminate called",
    ],
)
@pytest.mark.parametrize("scheduler", [Scheduler.SLURM, Scheduler.KUEUE])
def test_lone_user_failure_before_preemption_keeps_failure(
    request, event_manager, simple_job_factory, timezone, scheduler, body
):
    """A single application error before the preemption keeps its failure verdict.

    These symptoms are dropped when any other symptom is present, so the
    preemption itself must not count as another symptom.
    """
    job_info = _preempted_job(simple_job_factory, timezone, scheduler)
    error_time = job_info.job_end_time - timedelta(minutes=30)
    error = make_applog_entry(
        body=body,
        hostname=_FAILED_NODE,
        timestamp_ns=int(error_time.timestamp() * 1_000_000_000),
    )

    result = _attribute(
        request, event_manager, job_info, _observation(job_info, timezone, ObservationSource.APPLOG, [error])
    )

    assert result.outcome_attr.value().outcome.startswith("Fail")
    assert "Job preempted" not in result.symptoms


@pytest.mark.parametrize(
    ("with_slurmstepd_lines", "expect_preemption_pass"),
    [
        # The slurmstepd lines carry their own time, so the untimed lines below
        # them are placed in the teardown window.
        (True, True),
        # Without a timed line above them, the untimed lines cannot be placed in
        # time and keep their verdict.
        (False, False),
    ],
)
def test_untimed_teardown_lines_after_slurmstepd_cancel(
    request, event_manager, simple_job_factory, timezone, with_slurmstepd_lines, expect_preemption_pass
):
    """Untimed Python shutdown warnings after the step cancel line are teardown noise.

    The application log has no per-line time here, as when it is read from a
    file. Only the slurmstepd lines carry a timestamp in their text.
    """
    job_info = _preempted_job(simple_job_factory, timezone)
    sigterm_time = job_info.job_end_time - timedelta(seconds=17)
    pyxis_time = job_info.job_end_time - timedelta(seconds=7)
    shutdown_warning = (
        "43: /usr/lib/python3.12/multiprocessing/resource_tracker.py:267: UserWarning: resource_tracker: "
        "'/mp-abc123': [Errno 2] No such file or directory"
    )
    bodies = []
    if with_slurmstepd_lines is True:
        bodies += [
            f"0: [{sigterm_time.strftime('%b %d %H:%M:%S')}.123456 4321 slurmstepd   0x1234] error: "
            f"*** STEP 424242.0 ON node01 CANCELLED AT {sigterm_time.strftime('%Y-%m-%dT%H:%M:%S')} "
            "DUE to SIGNAL Terminated ***",
            f"16: [{pyxis_time.strftime('%b %d %H:%M:%S')}.123456 5678 slurmstepd   0x5678] error: "
            "pyxis: child 9876 terminated with signal 15",
        ]
    bodies.append(shutdown_warning)
    lines = [{"ts": None, "resources": None, "body": body} for body in bodies]

    result = _attribute(
        request, event_manager, job_info, _observation(job_info, timezone, ObservationSource.APPLOG, lines)
    )

    assert _is_preemption_pass(result) is expect_preemption_pass
    assert result.outcome_attr.value().outcome.startswith("Fail") is not expect_preemption_pass
    assert ("No such file or directory" in result.symptoms) is not expect_preemption_pass
