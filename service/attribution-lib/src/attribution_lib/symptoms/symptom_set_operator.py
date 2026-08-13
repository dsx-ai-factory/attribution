# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Abstracted operators over SymptomSet using metadata-driven cascades.
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeVar

from loguru import logger

import attribution_lib
from attribution_lib.decision_trace_recorder import Action, DecisionRecorder, NodeType
from attribution_lib.job_info import JobInfo
from attribution_lib.sources import SymptomSet
from attribution_lib.symptoms.base import BaseSymptom

if TYPE_CHECKING:
    from attribution_lib.symptoms.registry_generation import GeneratedSymptomRegistry

CascadeSymptom = TypeVar("CascadeSymptom", str, BaseSymptom)


def _evaluate_runtime_condition(
    condition: str,
    *,
    registry: "GeneratedSymptomRegistry",
    job_info: Optional[JobInfo],
    symptoms: SymptomSet[CascadeSymptom],
) -> bool:
    """Evaluate conditional discard predicate by name via the registry."""
    predicate = registry.predicate_for(condition)
    if predicate is None:
        return False

    return bool(predicate(symptoms, job_info))


# Topological sort of discard targets
def _collect_discard_targets(
    start_id: str,
    registry: "GeneratedSymptomRegistry",
    symptoms: SymptomSet[CascadeSymptom],
    job_info: Optional[JobInfo],
) -> set[str]:
    visited: set[str] = set()
    discard_targets: set[str] = set()

    stack: list[str] = [start_id]
    while stack:
        symptom_id = stack.pop()
        if symptom_id in visited:
            continue
        visited.add(symptom_id)

        symptom_metadata = registry.get(symptom_id)
        if symptom_metadata is not None:
            for discard_id in symptom_metadata.cascade_discards:
                discard_targets.add(discard_id)
                if registry.get(discard_id) is not None:
                    stack.append(discard_id)
            # Evaluate conditional discards from registry
            for target_id, cond in symptom_metadata.conditional_discards:
                if _evaluate_runtime_condition(cond, registry=registry, job_info=job_info, symptoms=symptoms):
                    discard_targets.add(target_id)
                    if registry.get(target_id) is not None:
                        stack.append(target_id)

    return discard_targets


def cascade_prune(
    symptoms: SymptomSet[CascadeSymptom],
    primary_symptom: str,
    *,
    job_info: Optional[JobInfo] = None,
    recorder: Optional[DecisionRecorder] = None,
    hostname: Optional[str] = None,
) -> None:
    """Apply DAG-based cascade pruning for a primary symptom over a SymptomSet[str] in place.

    Traverses the registry graph starting at the primary symptom, computes the
    transitive closure of discard patterns, and prunes matching symptoms while
    preserving provenance in SymptomSet.
    """
    start_id = primary_symptom

    registry = attribution_lib.get_symptom_registry()
    discard_targets = _collect_discard_targets(start_id, registry, symptoms, job_info)

    current = list(symptoms.symptoms())
    to_remove: list[CascadeSymptom] = []
    for current_symptom in current:
        current_symptom_str = str(current_symptom)
        for discard_target in discard_targets:
            if discard_target == current_symptom_str or discard_target in current_symptom_str:
                logger.info(f"DAG PRUNING: Primary symptom '{start_id}' triggered discard of '{current_symptom}'")
                to_remove.append(current_symptom)
                if recorder is not None:
                    recorder.record(
                        node_type=NodeType.CASCADE_PRUNE,
                        rule_name="DagCascadePruning",
                        action=Action.PRUNED,
                        symptom=current_symptom_str,
                        hostname=hostname,
                        reason=f"cascade child of {start_id}",
                    )
                break

    if len(to_remove) > 0:
        symptoms.discard_seq(to_remove, purge=False)
