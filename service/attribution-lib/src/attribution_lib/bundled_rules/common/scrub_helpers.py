# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re
from typing import Optional

from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.sources import SymptomSet


def clean_bad_nodes_of_cascaded_symptoms(my_bad_nodes: JobNodeAttributions, symptoms_to_keep: list[str]):
    """Given a symptom_to_keep, we believe all other symptoms are cascaded, so discard anything but this symptom"""
    for k in my_bad_nodes:
        for symptom in my_bad_nodes[k].symptoms():
            if symptom not in symptoms_to_keep:
                my_bad_nodes[k].discard(symptom)


def remove_cascaded_network_symptoms(symptoms: SymptomSet[str], my_bad_nodes: JobNodeAttributions):
    to_discard = [
        "Connection reset by peer",
        "Connection closed",
        "PyTorch NCCL watchdog",
        "Software caused connection abort",
        "cudaErrorDevicesUnavailable",
    ]
    for symptom in to_discard:
        symptoms.discard(symptom)
    for _node, errors in my_bad_nodes.items():
        for symptom in to_discard:
            errors.discard(symptom)


# Symptoms that must not be pinned to a specific node even when they surface on
# one — fabric/collective/catastrophic signals whose blame does not belong to the
# reporting node. Stripped from per-node attributions by get_clean_bad_nodes and
# skipped by refiners that would otherwise attribute them to a node (e.g.
# SingleNodeInfraBackfill).
NON_NODE_ATTRIBUTABLE_SYMPTOM_MARKERS = (
    "IB Error",
    "local catastrophic error",
    "CE deadlock DETECTED",
    "PyTorch NCCL watchdog",
    "pushrecv",
    "NVRx detected failure",
)

_NON_NODE_ATTRIBUTABLE_RE = re.compile("|".join(re.escape(m) for m in NON_NODE_ATTRIBUTABLE_SYMPTOM_MARKERS))


def is_non_node_attributable_symptom(symptom: str) -> bool:
    return bool(_NON_NODE_ATTRIBUTABLE_RE.search(symptom))


def get_clean_bad_nodes(my_bad_nodes: JobNodeAttributions, cluster: str) -> JobNodeAttributions:
    # NOTE (rule-credit migration): these two are placeholder credited-rule names,
    # intentionally NOT renamed in the credited-rule-provenance change because this
    # manual match depends on them. Retire them here when scrub is made rule-based —
    # key the demotion off the symptom (as the block below already does) and
    # reconcile these findings' source.resources to AllJobResources. Until then,
    # the rules that emit them keep crediting these names on purpose.
    non_node_compat = frozenset(["the process completes", "alias else"])
    nodes = list(my_bad_nodes.attributions.keys())  # a copy so we can delete while iterating
    for node in nodes:
        my_bad_nodes.attributions[node].attributions.discard_by_rule(non_node_compat)
        symptoms = list(my_bad_nodes.attributions[node].symptoms())
        for symptom in symptoms:
            if is_non_node_attributable_symptom(symptom):
                my_bad_nodes.attributions[node].discard(symptom)
    return my_bad_nodes


def filter_xid_145_cascade_symptoms(
    symptoms: SymptomSet[str], my_bad_nodes: Optional[JobNodeAttributions] = None
) -> None:
    """Filter out XID-145 specific cascade symptoms.

    When XID-145 errors occur, they can cause cascading failures that show up as
    secondary symptoms. This function removes known XID-145 cascade symptoms
    from both job-level symptoms and per-node attributions.

    Args:
        symptoms: SymptomSet to filter (modified in place)
        my_bad_nodes: Per-node attributions to filter (modified in place)
    """
    cascade_symptoms_to_filter = [
        "cudaErrorLaunchFailure",
        "cudaErrorNvlinkUncorrectable",
        "XID-45-PREEMPTIVE_REMOVAL",
        "NV_ERR_INVALID_STATE",
    ]
    # RXPIPE-8 is treated as the primary symptom; SRC_TRACK variants are cascaded noise.
    if "XID-145-RLW_RXPIPE-8" in symptoms:
        cascade_symptoms_to_filter.extend(
            [
                "XID-145-RLW_SRC_TRACK-4",
                "XID-145-RLW_SRC_TRACK-8",
            ]
        )

    # Find symptoms to discard
    symptoms_to_discard = []
    for symptom in symptoms.symptoms():
        if any(cascade in symptom for cascade in cascade_symptoms_to_filter):
            symptoms_to_discard.append(symptom)

    # Discard cascade symptoms (using discard_seq for efficiency)
    if symptoms_to_discard:
        symptoms.discard_seq(symptoms_to_discard, purge=False)

    if my_bad_nodes is not None:
        my_bad_nodes.discard_matching_from_all_nodes(cascade_symptoms_to_filter)


def filter_cascade_symptoms_symptomset(
    symptoms: SymptomSet[str], my_bad_nodes: Optional[JobNodeAttributions] = None
) -> None:
    """Generic cascade symptom filter that dispatches to specific filtering functions.

    This function checks for various primary symptoms that can cause cascading failures
    and calls the appropriate specific filtering function for each type detected.

    Args:
        symptoms: SymptomSet to filter (modified in place)
        my_bad_nodes: Per-node attributions to filter (modified in place)
    """
    # Check for XID-145 and apply its specific cascade filtering
    has_xid_145 = any("XID-145" in symptom for symptom in symptoms.symptoms())
    if has_xid_145:
        filter_xid_145_cascade_symptoms(symptoms, my_bad_nodes)
