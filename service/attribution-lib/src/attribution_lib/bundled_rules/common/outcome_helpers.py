# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import TYPE_CHECKING

from attribution_lib.events import EventStream

_SYSTEM_CANCEL_FLAGS = frozenset({"root", "Timeout", "PyTorch NCCL watchdog"})
from .megatron import get_pytorch_timeout_minutes
from attribution_lib.job_info import JobInfo
from attribution_lib.outcome import OutcomeAttribution
from attribution_lib.sources import SourceSet

if TYPE_CHECKING:
    from attribution_lib.core import AttributionResult


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
    if job_info.cancelled_by is None or current_outcome != "Fail User":
        return False

    cancelled_by = job_info.cancelled_by
    if cancelled_by in _SYSTEM_CANCEL_FLAGS or cancelled_by.startswith("svc-"):
        return False

    previous_source = attribution_result.outcome_attr.one_source() or job_info.source(
        cancelled_by, "slurmdb cancelled_by"
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

    # Reclassify PREEMPTED jobs to Pass
    if job_info.state is not None and job_info.state.upper() == "PREEMPTED":
        previous_source = attribution_result.outcome_attr.one_source()

        if previous_source is None:
            previous_source = job_info.source(job_info.state, "slurm state")

        new_outcome = OutcomeAttribution(
            outcome="Pass",
            hw_cause="Not Applicable",
            root_cause_symptom="SLURM preempted",
        )

        attribution_result.outcome_attr = SourceSet(new_outcome).add(
            previous_source.derived(rule="reclassified: slurm preempted")
        )
        return

    # Conditions:
    # - Cancelled by Timeout (i.e., SLURM timeout) and not user-cancel
    # - No other error symptoms
    # - distributed timeout (from applog runtime_args) < job_info.time_limit
    # Reclassify TIMEOUT to Pass when it's the only symptom (no other errors found)
    if job_info.time_limit is not None:
        normalized_symptoms = [s.lower() for s in attribution_result.symptoms.symptoms()]
        is_slurm_timeout = (
            attribution_result.cancelled_by is not None and attribution_result.cancelled_by.lower() == "timeout"
        )
        has_user_cancel_symptom = any("cancelled by user" in s for s in normalized_symptoms)
        has_only_timeout_symptoms = (
            all(s.startswith("cancelled by timeout") for s in normalized_symptoms) or not normalized_symptoms
        )

        if is_slurm_timeout and not has_user_cancel_symptom and has_only_timeout_symptoms:
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
