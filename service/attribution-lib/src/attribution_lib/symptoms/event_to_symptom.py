# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import Optional

from loguru import logger

import attribution_lib
from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.exceptions import UnknownSymptomError
from attribution_lib.job_info import JobInfo
from attribution_lib.sources import SourceSet, SymptomSet
from attribution_lib.symptoms.base import BaseSymptom
from attribution_lib.symptoms.symptom_enums import RegisteredSymptoms
from attribution_lib.symptoms.symptom_set_operator import cascade_prune

UNKNOWN_SYMPTOM_PREFIX = "unknown-symptom-"


def translate_node_attributions_to_symptoms(
    job_attributions: JobNodeAttributions,
) -> tuple[dict[str, SymptomSet[BaseSymptom]], set[str]]:
    """Translate per-node string attributions into `SymptomSet[BaseSymptom]`.

    Symptoms that resolve to a ``RegisteredSymptoms`` member (exact or prefix
    match) keep their original string as ``symptom_id``.  Unresolvable symptoms
    are tagged with :data:`UNKNOWN_SYMPTOM_PREFIX` and passed through so they
    can surface as ``Fail Undetermined`` in outcome resolution.

    Returns:
        A tuple of:
        - per-node `SymptomSet[BaseSymptom]` mapping
        - a set of symptom ids that were skipped (currently unused; kept for
          API compatibility)
    """
    registry = attribution_lib.get_symptom_registry()
    node_symptoms: dict[str, SymptomSet[BaseSymptom]] = {}
    skipped: set[str] = set()

    for node, group in job_attributions.attributions.items():
        per_node_set: SymptomSet[BaseSymptom] = SymptomSet()

        for symptom_str, src_set in group.attributions.items():
            # Resolution: exact value → prefix fallback → unknown tagging.
            # Non-fatal demotions are encoded as a dedicated symptom id so DAG
            # rules can treat them uniformly.
            try:
                if "non-fatal" in symptom_str:
                    resolved = RegisteredSymptoms.NON_FATAL.value
                else:
                    resolved = registry.resolve_symptom(symptom_str)

                # preserve the original symptom_str as symptom_id for
                # prefix-matched symptoms (e.g. "XID-149.21-...", "NV_ERR_...")
                symptom_metadata = registry.get(resolved)
                base_symptom = BaseSymptom(
                    symptom_id=symptom_str,
                    sources=None,
                    attributes=None,
                    cascade_discards=(list(symptom_metadata.cascade_discards) if symptom_metadata is not None else []),
                )
            except UnknownSymptomError:
                logger.warning("Unregistered symptom id tagged as unknown: {}", symptom_str)
                base_symptom = BaseSymptom(
                    symptom_id=f"{UNKNOWN_SYMPTOM_PREFIX}{symptom_str}",
                    sources=None,
                    attributes=None,
                    cascade_discards=[],
                )

            translated_sources = SourceSet(base_symptom)
            for source in src_set:
                translated_sources.add(source)
                base_symptom.add_source(source)

            per_node_set.add(translated_sources)

        if not per_node_set.is_void():
            node_symptoms[node] = per_node_set

    return node_symptoms, skipped


def translate_and_prune_node_attributions(
    job_attributions: JobNodeAttributions,
    job_info: JobInfo,
    recorder: Optional[DecisionRecorder] = None,
) -> tuple[dict[str, SymptomSet[BaseSymptom]], set[str]]:
    """Translate JobNodeAttributions to SymptomSet[BaseSymptom] and apply DAG pruning"""

    registry = attribution_lib.get_symptom_registry()
    node_symptoms, skipped = translate_node_attributions_to_symptoms(job_attributions)
    pruned: dict[str, SymptomSet[BaseSymptom]] = {}

    for node, per_node_set in node_symptoms.items():
        current_symptoms = list(per_node_set.symptoms())

        for base_symptom in current_symptoms:
            if base_symptom.symptom_id.startswith(UNKNOWN_SYMPTOM_PREFIX):
                continue
            resolved = registry.resolve_symptom(base_symptom.symptom_id)
            cascade_prune(per_node_set, resolved, job_info=job_info, recorder=recorder, hostname=node)

        # Keep only nodes that still have at least one "bad" symptom after pruning.
        if len(per_node_set) > 0:
            pruned[node] = per_node_set

    return pruned, skipped


def project_node_symptoms_to_bad_nodes(
    original_bad_nodes: JobNodeAttributions,
    node_symptoms_base: dict[str, SymptomSet[BaseSymptom]],
) -> JobNodeAttributions:
    """Project DAG-pruned node BaseSymptom view back into a JobNodeAttributions view,
    and restore DAG decisions with the original bad/interesting demotions.
    - Symptoms kept by the DAG remain marked as bad.
    - Symptoms discarded by the DAG are demoted from bad -> interesting
    """
    pruned_bad_nodes = JobNodeAttributions()

    for node, per_node_set in node_symptoms_base.items():
        for base_symptom, src_set in per_node_set.items():
            string_sources = src_set.value_mapped_copy(lambda s: str(s))
            pruned_bad_nodes.add([node], string_sources)

    pruned_by_node: dict[str, set[str]] = {
        node: set(group.symptoms()) for node, group in pruned_bad_nodes.attributions.items()
    }

    for node, group in list(original_bad_nodes.attributions.items()):
        kept = pruned_by_node.get(node, set())
        for symptom in list(group.symptoms()):
            if symptom in kept:
                continue

            # Preserve legacy behavior for:
            # - EAP_AUTHENTICATION_RECOVERY is purged entirely instead of demotion
            # - CUDA_ERROR_ECC_UNCORRECTABLE remains as a bad symptom at node level
            # - CUDA_ERROR_ILLEGAL_ADDRESS remains as a bad symptom at node level
            if symptom == RegisteredSymptoms.EAP_AUTHENTICATION_RECOVERY.value:
                original_bad_nodes.purge_from_node(node, symptom)
                continue

            if symptom == RegisteredSymptoms.CUDA_ERROR_ECC_UNCORRECTABLE.value:
                continue

            if symptom == RegisteredSymptoms.CUDA_ERROR_ILLEGAL_ADDRESS.value:
                continue

            group.discard(symptom, purge=False)

    return original_bad_nodes
