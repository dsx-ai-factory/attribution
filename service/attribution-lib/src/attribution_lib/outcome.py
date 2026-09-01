# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import attribution_lib
from attribution_lib.decision_trace_recorder import Action, DecisionRecorder, NodeType
from attribution_lib.sources import (
    SourceSet,
    SymptomSet,
)

# Symptoms that should be preserved for reporting (s_symptoms) but excluded from
# root-cause outcome selection.
_OUTCOME_IGNORED_SYMPTOM_PREFIXES = ("plr_tx_bw_loss_high",)


def is_outcome_ignored_symptom(symptom: str) -> bool:
    """Return True when symptom should not participate in outcome selection.

    To ignore additional symptom families, append their stable prefixes to
    _OUTCOME_IGNORED_SYMPTOM_PREFIXES.
    """
    return any(symptom.startswith(prefix) for prefix in _OUTCOME_IGNORED_SYMPTOM_PREFIXES)


@dataclass(kw_only=True)
class OutcomeAttribution:
    """Class for the output of calculate_outcome. Has sensible defaults for not-found situations."""

    root_cause_symptom: str = ""
    hw_cause: str = "Not Applicable"
    outcome: str = "Incomplete Info"
    doc: Optional[str] = None


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
            doc=match.doc,
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

        return sset

    return SourceSet(OutcomeAttribution())
