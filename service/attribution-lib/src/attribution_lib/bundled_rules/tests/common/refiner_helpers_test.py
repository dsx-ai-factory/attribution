# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.bundled_rules.common.refiner_helpers import _is_it_nvswitch_reboot_for_gb200_only
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.observation.inbound import (
    Context as ObservationContext,
)
from attribution_lib.sources import SourceSet

from test_support.helpers import make_node_to_ranks


@pytest.fixture
def mock_context():
    """Create a mock observation context"""
    return Mock(spec=ObservationContext)


@pytest.fixture
def job_info(mock_context):
    now = datetime.now(ZoneInfo("UTC"))
    builder = JobInfoBuilder(
        id="12345",
        type="training",
        state="FAILED",
        cluster="test-cluster",
        framework="pytorch",
        source_name="test",
        timezone=ZoneInfo("UTC"),
        job_start_time=now,
        node_to_ranks=make_node_to_ranks(
            ["nswitch1-001", "nswitch1-002", "nswitch1-003", "nswitch2-001", "nswitch2-002"], 8
        ),
    )
    builder.srun_cmd("srun python train.py")
    builder.job_script_content("")
    builder.interest_end_time(now)
    builder.num_gpus(8)
    return builder.build()


def test_nvswitch_reboot_single_node_no_reboot(job_info):
    """Test that single node with XID-149 doesn't trigger nvswitch reboot detection"""
    my_bad_nodes = JobNodeAttributions()

    # Add XID-149 to a single node
    source = job_info.source("XID-149", "test_rule")
    source_set = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source)
    my_bad_nodes.add(["nswitch1-001"], source_set)

    result = _is_it_nvswitch_reboot_for_gb200_only(my_bad_nodes, job_info)

    # Should return None since only one node has XID-149
    assert result is None, f"Expected None, got {result}"


def test_nvswitch_reboot_multiple_nodes_same_rack(job_info):
    """Test that multiple nodes on same rack with XID-149 triggers nvswitch reboot detection"""
    my_bad_nodes = JobNodeAttributions()

    # Add XID-149 to all nodes on the same rack (nswitch1 has 3 nodes in job_info)
    source1 = job_info.source("XID-149", "test_rule1")
    source2 = job_info.source("XID-149", "test_rule2")
    source3 = job_info.source("XID-149", "test_rule3")

    source_set1 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source1)
    source_set2 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source2)
    source_set3 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source3)

    my_bad_nodes.add(["nswitch1-001"], source_set1)
    my_bad_nodes.add(["nswitch1-002"], source_set2)
    my_bad_nodes.add(["nswitch1-003"], source_set3)

    result = _is_it_nvswitch_reboot_for_gb200_only(my_bad_nodes, job_info)

    # Should return a list with SourceSets for "LIKELY-NVSWITCH-REBOOT"
    assert result is not None, "Result should not be None"
    assert len(result) == 3, f"Expected 3 results (one per node), got {len(result)}"

    # All should be "LIKELY-NVSWITCH-REBOOT"
    for i, symptom_source_set in enumerate(result):
        assert symptom_source_set.value() == "LIKELY-NVSWITCH-REBOOT", (
            f"Expected 'LIKELY-NVSWITCH-REBOOT', got '{symptom_source_set.value()}' for result {i}"
        )
        assert len(symptom_source_set) == 1, f"Expected 1 source, got {len(symptom_source_set)} for result {i}"

        # Verify the source has the correct rule
        source = next(iter(symptom_source_set))
        assert source.rule is not None, f"Source rule should not be None for result {i}"
        assert "test_rule" in source.rule, (
            f"Expected source rule to contain 'test_rule', got '{source.rule}' for result {i}"
        )


def test_nvswitch_reboot_multiple_nodes_different_racks(job_info):
    """Test that multiple nodes on different racks with XID-149 doesn't trigger nvswitch reboot detection"""
    my_bad_nodes = JobNodeAttributions()

    # Add XID-149 to nodes on different racks
    source1 = job_info.source("XID-149", "test_rule1")
    source2 = job_info.source("XID-149", "test_rule2")

    source_set1 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source1)
    source_set2 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source2)

    my_bad_nodes.add(["nswitch1-001"], source_set1)
    my_bad_nodes.add(["nswitch2-001"], source_set2)

    result = _is_it_nvswitch_reboot_for_gb200_only(my_bad_nodes, job_info)

    # Should return None since nodes are on different racks
    assert result is None, f"Expected None, got {result}"


def test_nvswitch_reboot_no_xid_149(job_info):
    """Test that nodes without XID-149 don't trigger nvswitch reboot detection"""
    my_bad_nodes = JobNodeAttributions()

    # Add different errors to nodes
    source1 = job_info.source("XID-63", "test_rule1")
    source2 = job_info.source("XID-92", "test_rule2")

    source_set1 = SourceSet.new_one("XID-63-INFOROM_DRAM_RETIREMENT_EVENT", source1)
    source_set2 = SourceSet.new_one("XID-92-EXCESSIVE_SBE_INTERRUPTS", source2)

    my_bad_nodes.add(["nswitch1-001"], source_set1)
    my_bad_nodes.add(["nswitch1-002"], source_set2)

    result = _is_it_nvswitch_reboot_for_gb200_only(my_bad_nodes, job_info)

    # Should return None since no XID-149 errors
    assert result is None, f"Expected None, got {result}"


def test_nvswitch_reboot_node_without_dash(job_info):
    """Test that nodes without dash in name are ignored"""
    my_bad_nodes = JobNodeAttributions()

    # Add XID-149 to a node without dash in name
    source = job_info.source("XID-149", "test_rule")
    source_set = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source)
    my_bad_nodes.add(["nswitch1001"], source_set)

    result = _is_it_nvswitch_reboot_for_gb200_only(my_bad_nodes, job_info)

    # Should return None since node name doesn't contain dash
    assert result is None, f"Expected None, got {result}"


def test_nvswitch_reboot_multiple_racks_with_xid_149(job_info):
    """Test that multiple racks with XID-149 each get their own nvswitch reboot detection"""
    my_bad_nodes = JobNodeAttributions()

    # Add XID-149 to all nodes on different racks
    source1 = job_info.source("XID-149", "test_rule1")
    source2 = job_info.source("XID-149", "test_rule2")
    source3 = job_info.source("XID-149", "test_rule3")
    source4 = job_info.source("XID-149", "test_rule4")
    source5 = job_info.source("XID-149", "test_rule5")

    source_set1 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source1)
    source_set2 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source2)
    source_set3 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source3)
    source_set4 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source4)
    source_set5 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source5)

    # All three nodes on rack nswitch1
    my_bad_nodes.add(["nswitch1-001"], source_set1)
    my_bad_nodes.add(["nswitch1-002"], source_set2)
    my_bad_nodes.add(["nswitch1-003"], source_set3)

    # Both nodes on rack nswitch2
    my_bad_nodes.add(["nswitch2-001"], source_set4)
    my_bad_nodes.add(["nswitch2-002"], source_set5)

    result = _is_it_nvswitch_reboot_for_gb200_only(my_bad_nodes, job_info)

    # Should return a list with five SourceSets (one for each node)
    assert result is not None, "Result should not be None"
    assert len(result) == 5, f"Expected 5 results (one per node), got {len(result)}"

    # All should be "LIKELY-NVSWITCH-REBOOT"
    for i, symptom_source_set in enumerate(result):
        assert symptom_source_set.value() == "LIKELY-NVSWITCH-REBOOT", (
            f"Expected 'LIKELY-NVSWITCH-REBOOT', got '{symptom_source_set.value()}' for result {i}"
        )
        assert len(symptom_source_set) == 1, f"Expected 1 source, got {len(symptom_source_set)} for result {i}"

        # Verify the source has the correct rule
        source = next(iter(symptom_source_set))
        assert source.rule is not None, f"Source rule should not be None for result {i}"
        assert "test_rule" in source.rule, (
            f"Expected source rule to contain 'test_rule', got '{source.rule}' for result {i}"
        )


def test_nvswitch_reboot_three_nodes_same_rack(job_info):
    """Test that three nodes on same rack with XID-149 triggers nvswitch reboot detection for all three nodes"""
    my_bad_nodes = JobNodeAttributions()

    # Add XID-149 to three nodes on the same rack
    source1 = job_info.source("XID-149", "test_rule1")
    source2 = job_info.source("XID-149", "test_rule2")
    source3 = job_info.source("XID-149", "test_rule3")

    source_set1 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source1)
    source_set2 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source2)
    source_set3 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source3)

    my_bad_nodes.add(["nswitch1-001"], source_set1)
    my_bad_nodes.add(["nswitch1-002"], source_set2)
    my_bad_nodes.add(["nswitch1-003"], source_set3)

    result = _is_it_nvswitch_reboot_for_gb200_only(my_bad_nodes, job_info)

    # Should return a list with three SourceSets (one for each node)
    assert result is not None, "Result should not be None"
    assert len(result) == 3, f"Expected 3 results (one per node), got {len(result)}"

    # All should be "LIKELY-NVSWITCH-REBOOT"
    for i, symptom_source_set in enumerate(result):
        assert symptom_source_set.value() == "LIKELY-NVSWITCH-REBOOT", (
            f"Expected 'LIKELY-NVSWITCH-REBOOT', got '{symptom_source_set.value()}' for result {i}"
        )
        assert len(symptom_source_set) == 1, f"Expected 1 source, got {len(symptom_source_set)} for result {i}"

        # Verify the source has the correct rule
        source = next(iter(symptom_source_set))
        assert source.rule is not None, f"Source rule should not be None for result {i}"
        assert "test_rule" in source.rule, (
            f"Expected source rule to contain 'test_rule', got '{source.rule}' for result {i}"
        )


def test_nvswitch_reboot_exception_handling(job_info):
    """Test that the function handles exceptions gracefully"""
    my_bad_nodes = JobNodeAttributions()

    # Add XID-149 to all nodes on the same rack (nswitch1 has 3 nodes in job_info)
    source1 = job_info.source("XID-149", "test_rule1")
    source2 = job_info.source("XID-149", "test_rule2")
    source3 = job_info.source("XID-149", "test_rule3")

    source_set1 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source1)
    source_set2 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source2)
    source_set3 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source3)

    my_bad_nodes.add(["nswitch1-001"], source_set1)
    my_bad_nodes.add(["nswitch1-002"], source_set2)
    my_bad_nodes.add(["nswitch1-003"], source_set3)

    # The function should handle exceptions and return None if there's an error
    result = _is_it_nvswitch_reboot_for_gb200_only(my_bad_nodes, job_info)

    # Should return a list with SourceSets for "LIKELY-NVSWITCH-REBOOT"
    assert result is not None, "Result should not be None"
    assert len(result) == 3, f"Expected 3 results (one per node), got {len(result)}"

    # All should be "LIKELY-NVSWITCH-REBOOT"
    for i, symptom_source_set in enumerate(result):
        assert symptom_source_set.value() == "LIKELY-NVSWITCH-REBOOT", (
            f"Expected 'LIKELY-NVSWITCH-REBOOT', got '{symptom_source_set.value()}' for result {i}"
        )
        assert len(symptom_source_set) == 1, f"Expected 1 source, got {len(symptom_source_set)} for result {i}"


def test_nvswitch_reboot_same_node_multiple_times(job_info):
    """Test that same node appearing multiple times with XID-149 is treated as single node"""
    my_bad_nodes = JobNodeAttributions()

    # Add XID-149 to the same node multiple times
    source1 = job_info.source("XID-149", "test_rule1")
    source2 = job_info.source("XID-149", "test_rule2")
    source3 = job_info.source("XID-149", "test_rule3")

    source_set1 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source1)
    source_set2 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source2)
    source_set3 = SourceSet.new_one("XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0", source3)

    # Add the same node multiple times with different XID-149 sources
    my_bad_nodes.add(["nswitch1-001"], source_set1)
    my_bad_nodes.add(["nswitch1-001"], source_set2)
    my_bad_nodes.add(["nswitch1-001"], source_set3)

    result = _is_it_nvswitch_reboot_for_gb200_only(my_bad_nodes, job_info)

    # Should return None since only one unique node has XID-149
    assert result is None, f"Expected None since same node appears multiple times, got {result}"
