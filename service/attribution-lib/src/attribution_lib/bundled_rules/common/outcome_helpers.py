# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional

import attribution_lib
from attribution_lib.events import EventStream

_SYSTEM_CANCEL_FLAGS = frozenset({"root", "Timeout", "PyTorch NCCL watchdog"})
from .megatron import get_pytorch_timeout_minutes
from attribution_lib.job_info import JobInfo, Scheduler
from attribution_lib.outcome import OutcomeAttribution
from attribution_lib.sources import SourceSet

if TYPE_CHECKING:
    from attribution_lib.core import AttributionResult
    from attribution_lib.symptoms.registry_generation import GeneratedOutcomeMatch

# Kubernetes queue schedulers that share rule semantics.
KUBERNETES_SCHEDULER_TYPES = frozenset({Scheduler.KUEUE, Scheduler.KAI})

# Bad symptoms a preempted job can carry without a failure: the fallback for a
# job with no evidence, and the symptom of Slurm's "CANCELLED AT" step-log line,
# which Slurm also writes for a preemption.
_PREEMPTION_ONLY_SYMPTOMS = ("job failed", "Cancelled by user")
# PostProcessSymptoms also writes "job failed" for a task-failure cancellation,
# which is failure evidence.
_TASK_FAILURE_RULE = "task failure cancellation"


def _failure_evidence(attribution_result: "AttributionResult") -> Optional["GeneratedOutcomeMatch"]:
    """Return the failure outcome that the remaining bad symptoms select.

    Args:
        attribution_result: Result after the post-outcome refiners.

    Returns:
        The outcome match of the bad symptoms, other than _PREEMPTION_ONLY_SYMPTOMS
        from the preemption itself, when it is a Fail outcome. Otherwise None.
    """
    evidence = [
        symptom
        for symptom, sources in attribution_result.symptoms.items()
        if symptom not in _PREEMPTION_ONLY_SYMPTOMS or any(source.rule == _TASK_FAILURE_RULE for source in sources)
    ]
    match = attribution_lib.get_symptom_registry().outcome_for(evidence)
    if match is None or not match.outcome.startswith("Fail"):
        return None
    return match


def apply_preemption_pass(attribution_result: "AttributionResult", job_info: JobInfo) -> bool:
    """Reclassify a preempted job to Pass when its evidence gives no failure verdict.

    Preemption must not mask evidence-backed failures. Teardown noise near the
    job end is demoted earlier by ExternalTerminationCascade, so a failure
    verdict here comes from evidence before the preemption. The decision uses
    the remaining bad symptoms, not the outcome, because refiners after the
    outcome calculation can remove the symptom that chose it.

    Args:
        attribution_result: Result whose outcome and symptoms are updated in place.
        job_info: Job to check.

    Returns:
        True when the job was reclassified to Pass.
    """
    if job_info.attributed_cancellation is not None or job_info.state is None or job_info.state.upper() != "PREEMPTED":
        return False
    failure = _failure_evidence(attribution_result)
    if failure is not None:
        # The outcome can come from a symptom that post-processing removed, such
        # as "Cancelled by user" next to a sigkill_handler failure in the same tier.
        if not attribution_result.outcome_attr.value().outcome.startswith("Fail"):
            restored = SourceSet(
                OutcomeAttribution(
                    outcome=failure.outcome,
                    hw_cause=failure.hw_cause,
                    root_cause_symptom=failure.symptom,
                    doc=failure.doc,
                )
            )
            source = attribution_result.symptoms[failure.symptom].one_source()
            if source is not None:
                restored.add(source)
            attribution_result.outcome_attr = restored
        return False

    previous_source = attribution_result.outcome_attr.one_source()
    if previous_source is None:
        previous_source = job_info.source(job_info.state, "slurm state")

    attribution_result.outcome_attr = SourceSet(
        OutcomeAttribution(outcome="Pass", hw_cause="Not Applicable", root_cause_symptom="SLURM preempted")
    ).add(previous_source.derived(rule="reclassified: slurm preempted"))
    # Show the preemption as the reason for the job end.
    for symptom in _PREEMPTION_ONLY_SYMPTOMS:
        attribution_result.symptoms.discard(symptom)
    attribution_result.symptoms.add(job_info.source_value("Job preempted", job_info.state, "state preempted"))
    return True


def apply_user_cancellation_reclassification(
    attribution_result: "AttributionResult",
    job_info: JobInfo,
) -> bool:
    """Reclassify to 'Cancelled by User' when job was cancelled by a regular user.

    Only reclassifies if the current outcome is "Fail User" - infrastructure errors
    (InfraHW/InfraSW) should take priority as the user likely cancelled because of
    the underlying error.

    Returns:
        True if reclassification was applied, False otherwise.
    """

    current_outcome = attribution_result.outcome_attr.value().outcome
    if job_info.attributed_cancelled_by is None or current_outcome != "Fail User":
        return False

    cancelled_by = job_info.attributed_cancelled_by
    if cancelled_by in _SYSTEM_CANCEL_FLAGS or cancelled_by.startswith("svc-"):
        return False

    cancellation = job_info.attributed_cancellation
    previous_source = (
        cancellation.source
        if cancellation is not None
        else attribution_result.outcome_attr.one_source() or job_info.source(cancelled_by, "slurmdb cancelled_by")
    )

    new_outcome = OutcomeAttribution(
        outcome="Cancelled by User",
        hw_cause="Not Applicable",
        root_cause_symptom=f"Cancelled by {cancelled_by}",
    )

    attribution_result.outcome_attr = SourceSet(new_outcome).add(
        previous_source.derived(rule="reclassified: user cancellation")
    )
    return True


def apply_outcome_reclassification_rules(
    attribution_result: "AttributionResult",
    event_stream: EventStream,
    job_info: JobInfo,
) -> None:
    """Apply post-processing outcome reclassification rules."""

    if apply_preemption_pass(attribution_result, job_info):
        return

    # Conditions:
    # - Cancelled by Timeout (i.e., scheduler timeout) and not user-cancel
    # - No other error symptoms
    # - distributed timeout (from applog runtime_args) < job_info.time_limit
    # Reclassify TIMEOUT to Pass when it's the only symptom (no other errors found)
    if job_info.time_limit is not None:
        normalized_symptoms = [s.lower() for s in attribution_result.symptoms.symptoms()]
        is_scheduler_timeout = (
            attribution_result.cancelled_by is not None and attribution_result.cancelled_by.lower() == "timeout"
        )
        has_user_cancel_symptom = any("cancelled by user" in s for s in normalized_symptoms)
        has_only_timeout_symptoms = (
            all(s.startswith("cancelled by timeout") for s in normalized_symptoms) or not normalized_symptoms
        )

        if is_scheduler_timeout and not has_user_cancel_symptom and has_only_timeout_symptoms:
            # Determine root_cause_symptom and derived rule based on pytorch timeout check
            pytorch_timeout_minutes = get_pytorch_timeout_minutes(event_stream)
            if pytorch_timeout_minutes is not None and pytorch_timeout_minutes.value() < float(job_info.time_limit):
                root_cause_symptom = "SLURM timeout with shorter user distributed timeout"
                derived_rule = "reclassified: pytorch timeout < slurm limit"
                fallback_source = pytorch_timeout_minutes.one_source()
            else:
                root_cause_symptom = "SLURM timeout (no other errors)"
                derived_rule = "reclassified: timeout only symptom"
                fallback_source = job_info.source(job_info.state, "slurm state")

            # Reclassify to Pass
            previous_source = attribution_result.outcome_attr.one_source()

            new_outcome = OutcomeAttribution(
                outcome="Pass",
                hw_cause="Not Applicable",
                root_cause_symptom=root_cause_symptom,
            )

            base_source = previous_source if previous_source is not None else fallback_source
            assert base_source is not None
            attribution_result.outcome_attr = SourceSet(new_outcome).add(base_source.derived(rule=derived_rule))
            return

    # Reclassify to "Cancelled by User" when job was cancelled by a regular user
    apply_user_cancellation_reclassification(attribution_result, job_info)
