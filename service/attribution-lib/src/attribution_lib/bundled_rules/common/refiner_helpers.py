# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Helper functions specific to refiner rules.
"""

from collections import defaultdict
from typing import Optional

from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.job_info import JobInfo
from attribution_lib.sources import SourceSet


def _get_earliest_watchdog_source(symptoms):
    """Get the source with the earliest timestamp from PyTorch NCCL watchdog symptoms."""
    watchdog_sources = symptoms["PyTorch NCCL watchdog"]
    earliest_source = None
    earliest_time = None
    for source in watchdog_sources:
        if source.time:
            if earliest_time is None or source.time < earliest_time:
                earliest_time = source.time
                earliest_source = source
    return earliest_source


def _add_symptom_from_source(symptoms, source, job_info, symptom_name: str, rule: str):
    if source:
        new_source = source.derived(rule=rule)
        symptoms.add(SourceSet(symptom_name).add(new_source))
    else:
        symptoms.add(job_info.source_value(symptom_name, None, rule))


def _is_it_nvswitch_reboot_for_gb200_only(
    my_bad_nodes: JobNodeAttributions, job_info: JobInfo
) -> Optional[list[SourceSet[str]]]:
    """Check if multiple nodes on the same rack have XID-149 symptoms, indicating potential nvswitch reboot.

    Returns:
        List of SourceSet[str] with "potential nvswitch reboot" symptom if condition is met, None otherwise.
    """
    # xid_149_nodes_by_rack: {rack_name: {node_name: [(error, sources)]}}

    xid_149_nodes_by_rack: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    # Iterate all nodes in my_bad_nodes, for nodes with "XID-149.4" in symptoms
    for node_name, node_group in my_bad_nodes.items():
        for error in node_group.symptoms():
            if "XID-149.4-" in error and "-" in node_name:
                # Extract the rack name (prefix before dash, for example "rack001")
                rack_name = node_name.split("-")[0]
                # Get the SourceSet for this error from the node's attributions
                error_sources = node_group.attributions[error]
                xid_149_nodes_by_rack[rack_name][node_name].append((error, error_sources))

    # If a rack name has more than 1 node with "XID-149.4" in symptoms, we need to add a new symptom
    sourceset_list = []
    for rack_name, node_data in xid_149_nodes_by_rack.items():
        number_of_nodes_used_from_rack = len([node for node in job_info.nodes if node.startswith(rack_name)])
        # If the number of nodes has seen XID-149 more than 2 times and the number of nodes from the rack is the same as the number of nodes has seen XID-149, we can add the symptom
        if len(node_data) >= 2 and len(node_data) == number_of_nodes_used_from_rack:
            # Create a SourceSet for each node with XID-149 on this rack
            for node_name, error_sources_list in node_data.items():
                for error, sources in error_sources_list:
                    sourceset_list.append(SourceSet.new_one("LIKELY-NVSWITCH-REBOOT", sources.one_source()))

    return sourceset_list if sourceset_list else None
