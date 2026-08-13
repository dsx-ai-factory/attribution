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

from collections import defaultdict
from fnmatch import fnmatchcase
from typing import Optional

from loguru import logger

import attribution_lib
from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.decision_trace_recorder import Action, DecisionRecorder, NodeType
from attribution_lib.exceptions import UnknownSymptomError
from attribution_lib.job_info import JobInfo
from attribution_lib.sources import SourceSet, SymptomSet
from attribution_lib.symptoms.base import BaseSymptom
from attribution_lib.symptoms.declarations import CascadeScope, DagDiscardBehavior
from attribution_lib.symptoms.symptom_set_operator import cascade_prune

UNKNOWN_SYMPTOM_PREFIX = "unknown-symptom-"


def _apply_nvl_domain_cascade_pruning(
    node_symptoms: dict[str, SymptomSet[BaseSymptom]],
    recorder: Optional[DecisionRecorder] = None,
) -> None:
    """Prune allowlisted cascade targets when a *sibling* node reports a primary.

    A target is pruned on a node only when a different node in the same NVL
    domain reports an NVL-domain primary. A lone node carrying both a primary
    and a target keeps the more-specific target.

    Args:
        node_symptoms: Per-node symptoms after same-node DAG pruning.
        recorder: Optional decision recorder for cascade-prune trace events.
    """
    domain_resolver = attribution_lib.node_domain_resolver
    if domain_resolver is None:
        return

    registry = attribution_lib.get_symptom_registry()
    nodes_by_nvl_domain: dict[str, list[str]] = defaultdict(list)
    discards_by_nvl_domain_and_node: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for node, per_node_set in node_symptoms.items():
        nvl_domain = domain_resolver(node)
        if nvl_domain is None:
            continue

        nodes_by_nvl_domain[nvl_domain].append(node)
        for symptom in per_node_set.symptoms():
            if symptom.symptom_id.startswith(UNKNOWN_SYMPTOM_PREFIX):
                continue
            discards_by_nvl_domain_and_node[nvl_domain][node].extend(
                registry.cascade_discards_for(symptom.symptom_id, scope=CascadeScope.NVL_DOMAIN)
            )

    for nvl_domain, nodes in nodes_by_nvl_domain.items():
        for node in nodes:
            sibling_targets = {
                target
                for primary_node, targets in discards_by_nvl_domain_and_node[nvl_domain].items()
                if primary_node != node
                for target in targets
            }
            if len(sibling_targets) == 0:
                continue

            per_node_set = node_symptoms[node]
            to_remove = [
                symptom
                for symptom in list(per_node_set.symptoms())
                if any(fnmatchcase(symptom.symptom_id, target) for target in sibling_targets)
            ]
            if len(to_remove) == 0:
                continue

            reason = f"cascade child of a primary on a sibling node in NVL domain {nvl_domain}"
            if recorder is not None:
                for symptom in to_remove:
                    recorder.record(
                        node_type=NodeType.CASCADE_PRUNE,
                        rule_name="DagCascadePruning",
                        action=Action.PRUNED,
                        symptom=symptom.symptom_id,
                        hostname=node,
                        reason=reason,
                    )
            per_node_set.discard_seq(to_remove, purge=False)


def translate_node_attributions_to_symptoms(
    job_attributions: JobNodeAttributions,
) -> tuple[dict[str, SymptomSet[BaseSymptom]], set[str]]:
    """Translate per-node string attributions into `SymptomSet[BaseSymptom]`.

    Symptoms that resolve through the registry (substring alias, exact, or prefix
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
            # Resolution: substring alias (e.g. non-fatal) → exact value → prefix
            # fallback → unknown tagging.
            try:
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
    """Translate node attributions and apply same-node and configured domain pruning."""

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

    _apply_nvl_domain_cascade_pruning(node_symptoms, recorder=recorder)

    for node, per_node_set in node_symptoms.items():
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

    registry = attribution_lib.get_symptom_registry()
    for node, group in list(original_bad_nodes.attributions.items()):
        kept = pruned_by_node.get(node, set())
        for symptom in list(group.symptoms()):
            if symptom in kept:
                continue

            # How a DAG-discarded symptom projects back is declared per-symptom via
            # dag_discard (default DEMOTE: bad -> interesting). KEEP leaves it
            # bad; PURGE removes it from the node.
            behavior = registry.dag_discard_for(symptom)
            if behavior is DagDiscardBehavior.PURGE:
                original_bad_nodes.purge_from_node(node, symptom)
            elif behavior is DagDiscardBehavior.KEEP:
                continue
            else:
                group.discard(symptom, purge=False)

    return original_bad_nodes
