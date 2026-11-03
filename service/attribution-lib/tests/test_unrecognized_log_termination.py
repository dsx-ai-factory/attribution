# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep collection diagnostics from replacing scheduler termination evidence."""

from collections.abc import Callable
from typing import Optional

import pytest

from attribution_lib.core import AttributionResult, CalcBadNodesAttributionResult, attribute
from attribution_lib.events import EventManager
from attribution_lib.job_info import JobInfo
from attribution_lib.sources import SourceSet

_UNRECOGNIZED_LOG = "log file is not recognized from slurm cmdline"


def _attribute(
    job: JobInfo,
    event_manager: EventManager,
    stream_id: str,
    keyword: Optional[str],
    symptom: Optional[str] = None,
) -> AttributionResult:
    """Run the real refiners with synthetic collection and independent evidence.

    Args:
        job: Execution metadata supplied to attribution.
        event_manager: Fixture owning isolated event streams.
        stream_id: Unique stream name for this attribution run.
        keyword: Collection diagnostic or other raw keyword, if present.
        symptom: Independently established node symptom, if present.

    Returns:
        The attribution result, including retained raw-keyword provenance.
    """
    evidence = CalcBadNodesAttributionResult()
    if keyword is not None:
        evidence.raw_keywords.add(SourceSet.new_one(keyword, job.source(keyword, "test_rule")))
    if symptom is not None:
        evidence.my_bad_nodes.add(["node-a"], SourceSet.new_one(symptom, job.source(symptom, "test_rule")))
    return attribute(
        event_stream=event_manager.create_stream(stream_id),
        job_info=job,
        bad_node_result=evidence,
        extract_job_metadata_from_logfile=False,
    )


@pytest.mark.parametrize(
    ("state", "actor"),
    [
        ("CANCELLED", "user-a"),
        ("CANCELLED by 1001", "user-a"),
        ("CANCELLED", "root"),
        ("CANCELLED", "site-automation-account"),
        ("CANCELLED", None),
        ("TIMEOUT", "Timeout"),
        ("TIMEOUT", None),
    ],
)
def test_unrecognized_log_preserves_scheduler_termination(
    simple_job_factory: Callable[..., JobInfo],
    event_manager: EventManager,
    state: str,
    actor: Optional[str],
) -> None:
    """Retain the existing terminal verdict without discarding the diagnostic.

    Args:
        simple_job_factory: Fixture constructing synthetic execution metadata.
        event_manager: Fixture owning the baseline and diagnostic event streams.
        state: Scheduler cancellation or timeout state for this execution.
        actor: Supplied cancellation identity, or explicitly unknown identity.

    Returns:
        None. Assertions compare outcomes, symptoms, and evidence provenance.
    """
    job = simple_job_factory(
        job_id="execution", nodes=["node-a"], state=state, job_type="batch", exit_code_signal=15
    )._replace(cancelled_by=actor)

    baseline = _attribute(job, event_manager, "baseline", None)
    result = _attribute(job, event_manager, "diagnostic", _UNRECOGNIZED_LOG)

    assert result.outcome_attr.value() == baseline.outcome_attr.value()
    assert set(result.symptoms.symptoms()) == set(baseline.symptoms.symptoms())
    assert set(result.outcome_attr.all_sources()) == set(baseline.outcome_attr.all_sources())
    assert "log unrecognized" not in result.symptoms
    assert result.bad_nodes_attr.raw_keywords[_UNRECOGNIZED_LOG].one_source() == job.source(
        _UNRECOGNIZED_LOG, "test_rule"
    )
    if state == "CANCELLED" and actor == "user-a":
        assert result.outcome_attr.value().outcome == "Cancelled by User"
    if state == "TIMEOUT" and actor == "Timeout":
        assert result.outcome_attr.value().outcome == "Pass"


@pytest.mark.parametrize("actor", [None, "user-a", "Timeout"])
def test_cancellation_identity_alone_does_not_explain_failed_job(
    simple_job_factory: Callable[..., JobInfo], event_manager: EventManager, actor: Optional[str]
) -> None:
    """Keep missing-log attribution when no terminal cancellation is established.

    Args:
        simple_job_factory: Fixture constructing a failed batch execution.
        event_manager: Fixture owning the attribution event stream.
        actor: Optional cancellation identity without a matching terminal state.

    Returns:
        None. Assertions retain the existing incomplete-information verdict.
    """
    job = simple_job_factory(job_id="execution", nodes=["node-a"], state="FAILED", job_type="batch")._replace(
        cancelled_by=actor
    )

    result = _attribute(job, event_manager, "diagnostic", _UNRECOGNIZED_LOG)

    assert result.outcome_attr.value().outcome == "Incomplete Info"
    assert "log unrecognized" in result.symptoms
    assert _UNRECOGNIZED_LOG in result.bad_nodes_attr.raw_keywords


@pytest.mark.parametrize(("state", "actor"), [("FAILED", None), ("CANCELLED", "user-a"), ("TIMEOUT", "Timeout")])
def test_independent_hardware_evidence_retains_precedence(
    simple_job_factory: Callable[..., JobInfo],
    event_manager: EventManager,
    state: str,
    actor: Optional[str],
) -> None:
    """Preserve an independently established fault alongside a missing log.

    Args:
        simple_job_factory: Fixture constructing synthetic execution metadata.
        event_manager: Fixture owning the attribution event stream.
        state: Scheduler state, including cancellation and timeout.
        actor: Supplied cancellation identity, if any.

    Returns:
        None. Assertions verify the fault's outcome and node ownership.
    """
    job = simple_job_factory(job_id="execution", nodes=["node-a"], state=state, job_type="batch")._replace(
        cancelled_by=actor
    )

    result = _attribute(job, event_manager, "diagnostic", _UNRECOGNIZED_LOG, "XID-48")

    assert result.outcome_attr.value().outcome == "Fail InfraHW"
    assert "XID-48" in result.bad_nodes_attr.my_bad_nodes["node-a"]
    assert "log unrecognized" not in result.symptoms
    assert _UNRECOGNIZED_LOG in result.bad_nodes_attr.raw_keywords


def test_existing_cancellation_symptom_retains_precedence(
    simple_job_factory: Callable[..., JobInfo], event_manager: EventManager
) -> None:
    """Preserve extracted cancellation evidence even without a cancelled state.

    Args:
        simple_job_factory: Fixture constructing a failed batch execution.
        event_manager: Fixture owning baseline and diagnostic event streams.

    Returns:
        None. Assertions preserve the outcome supported by cancellation evidence.
    """
    job = simple_job_factory(job_id="execution", nodes=["node-a"], state="FAILED", job_type="batch")._replace(
        cancelled_by="user-a"
    )
    baseline = _attribute(job, event_manager, "baseline", None, "Cancelled by user")

    result = _attribute(job, event_manager, "diagnostic", _UNRECOGNIZED_LOG, "Cancelled by user")

    assert result.outcome_attr.value() == baseline.outcome_attr.value()
    assert result.outcome_attr.value().outcome == "Cancelled by User"
    assert "log unrecognized" not in result.symptoms
    assert _UNRECOGNIZED_LOG in result.bad_nodes_attr.raw_keywords


@pytest.mark.parametrize("state", ["CANCELLED", "TIMEOUT"])
def test_other_raw_keyword_promotion_is_unchanged(
    simple_job_factory: Callable[..., JobInfo], event_manager: EventManager, state: str
) -> None:
    """Keep positive completion-keyword promotion independent of this diagnostic.

    Args:
        simple_job_factory: Fixture constructing synthetic execution metadata.
        event_manager: Fixture owning the attribution event stream.
        state: Scheduler cancellation or timeout state.

    Returns:
        None. The assertion checks the existing completion promotion.
    """
    job = simple_job_factory(job_id="execution", nodes=["node-a"], state=state, job_type="batch")

    result = _attribute(job, event_manager, "completion", "Avg bus bandwidth")

    assert "the process completes" in result.symptoms
