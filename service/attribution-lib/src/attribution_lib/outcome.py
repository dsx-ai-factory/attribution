# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

import attribution_lib
from attribution_lib.attr_sub_class import AttributionSubClassEnum
from attribution_lib.decision_trace_recorder import Action, DecisionRecorder, NodeType
from attribution_lib.events import EventStream
from attribution_lib.filter import get_pytorch_timeout_minutes
from attribution_lib.job_info import JobInfo
from attribution_lib.sources import (
    SourceSet,
    SymptomSet,
)

if TYPE_CHECKING:
    from attribution_lib.core import AttributionResult, CalcBadNodesAttributionResult


# Symptoms that should be preserved for reporting (s_symptoms) but excluded from
# root-cause outcome selection.
_OUTCOME_IGNORED_SYMPTOM_PREFIXES = ("plr_tx_bw_loss_high",)


def is_outcome_ignored_symptom(symptom: str) -> bool:
    """Return True when symptom should not participate in outcome selection.

    To ignore additional symptom families, append their stable prefixes to
    _OUTCOME_IGNORED_SYMPTOM_PREFIXES.
    """
    return any(symptom.startswith(prefix) for prefix in _OUTCOME_IGNORED_SYMPTOM_PREFIXES)


class ClassifiedSymptom(Protocol):
    errClass: AttributionSubClassEnum
    suggestion: str
    isProgramError: bool


@dataclass(kw_only=True)
class OutcomeAttribution:
    """Class for the output of calculate_outcome. Has sensible defaults for not-found situations."""

    root_cause_symptom: str = ""
    hw_cause: str = "Not Applicable"
    outcome: str = "Incomplete Info"
    sub_class: AttributionSubClassEnum = AttributionSubClassEnum.UNDETERMINED

    @classmethod
    def from_lepton_symptom(cls, symptom: ClassifiedSymptom) -> "OutcomeAttribution":
        if symptom.isProgramError:
            outcome = "Fail User"
        else:
            outcome = "Fail InfraSW"
        return cls(
            outcome=outcome,
            hw_cause=symptom.suggestion,
            root_cause_symptom=symptom.errClass.to_short_string(),
            sub_class=symptom.errClass,
        )


def calculate_outcome(
    symptoms: SymptomSet[str],
    recorder: Optional[DecisionRecorder] = None,
) -> SourceSet[OutcomeAttribution]:
    registry = attribution_lib.get_symptom_registry()
    match = registry.outcome_for(
        symptoms.symptoms(),
        ignored_prefixes=_OUTCOME_IGNORED_SYMPTOM_PREFIXES,
    )

    if match is not None:
        outcome = OutcomeAttribution(
            outcome=match.outcome,
            hw_cause=match.hw_cause,
            root_cause_symptom=match.symptom,
        )
        sset = SourceSet(outcome)
        source = symptoms[match.symptom].one_source()
        if source is not None:
            sset.add(source)
        if recorder is not None:
            recorder.record(
                node_type=NodeType.OUTCOME,
                rule_name="symptom_to_outcome",
                action=Action.MATCHED,
                symptom=match.symptom,
                reason=f"{match.symptom} matched tier {match.tier_index + 1} → {match.outcome}",
            )

        # After outcome calculation: Clean up XID-145 dynamic symptoms
        symptoms_to_remove = []
        for symptom in symptoms.symptoms():
            if (
                symptom == "XID-145-InfraHW"
                or symptom == "XID-145-InfraSW"
                or symptom == "XID-145-MultiCategory"
                or symptom == "XID-145-Undetermined"
            ):
                symptoms_to_remove.append(symptom)

        for symptom in symptoms_to_remove:
            symptoms.discard(symptom)

        return sset

    # Clean up dynamic symptoms even if no outcome found
    symptoms_to_remove = []
    for symptom in symptoms.symptoms():
        if (
            symptom == "XID-145-InfraHW"
            or symptom == "XID-145-InfraSW"
            or symptom == "XID-145-MultiCategory"
            or symptom == "XID-145-Undetermined"
        ):
            symptoms_to_remove.append(symptom)

    for symptom in symptoms_to_remove:
        symptoms.discard(symptom)

    return SourceSet(OutcomeAttribution())


def apply_user_cancellation_reclassification(
    attribution_result: "AttributionResult",
    job_info: JobInfo,
) -> bool:
    """Reclassify to 'Cancelled by User' when job was cancelled by a regular user.

    Only reclassifies if the current outcome is "Fail User" - infrastructure errors
    (InfraHW/InfraSW) should take priority as the user likely cancelled because of
    the underlying error.

    System/service accounts should NOT trigger this reclassification.

    Returns:
        True if reclassification was applied, False otherwise.
    """
    SYSTEM_CANCEL_ACCOUNTS = {"root", "Timeout", "PyTorch NCCL watchdog"}
    current_outcome = attribution_result.outcome_attr.value().outcome
    if job_info.cancelled_by is None or current_outcome != "Fail User":
        return False

    cancelled_by = job_info.cancelled_by
    is_system_account = cancelled_by in SYSTEM_CANCEL_ACCOUNTS or cancelled_by.startswith("svc-")
    if is_system_account:
        return False

    previous_source = attribution_result.outcome_attr.one_source() or job_info.source(
        cancelled_by, "slurmdb cancelled_by"
    )

    new_outcome = OutcomeAttribution(
        outcome="Cancelled by User",
        hw_cause="Not Applicable",
        root_cause_symptom=f"Cancelled by {cancelled_by}",
    )

    updated_source_set = SourceSet(new_outcome).add(previous_source.derived(rule="reclassified: user cancellation"))

    attribution_result.outcome_attr = updated_source_set
    return True


def apply_outcome_reclassification_rules(
    attribution_result: "AttributionResult",
    event_stream: EventStream,
    job_info: JobInfo,
    calc_bad_nodes_result: "CalcBadNodesAttributionResult",
) -> None:
    """Apply post-processing outcome reclassification rules."""

    # Reclassify PREEMPTED jobs to Pass
    if job_info.state.upper() == "PREEMPTED":
        previous_source = attribution_result.outcome_attr.one_source()

        if previous_source is None:
            previous_source = job_info.source(job_info.state, "slurm state")

        new_outcome = OutcomeAttribution(
            outcome="Pass",
            hw_cause="Not Applicable",
            root_cause_symptom="SLURM preempted",
        )

        updated_source_set = SourceSet(new_outcome).add(previous_source.derived(rule="reclassified: slurm preempted"))

        attribution_result.outcome_attr = updated_source_set
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
            updated_source_set = SourceSet(new_outcome).add(base_source.derived(rule=derived_rule))

            attribution_result.outcome_attr = updated_source_set
            return

    # Reclassify to "Cancelled by User" when job was cancelled by a regular user
    apply_user_cancellation_reclassification(attribution_result, job_info)
