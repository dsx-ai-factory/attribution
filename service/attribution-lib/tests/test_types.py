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
from zoneinfo import ZoneInfo

import pytest

from attribution_lib.attr_types import (
    BadPairInfo,
    JobNodeAttributions,
    NodeAttributionGroup,
)
from attribution_lib.exceptions import (
    InvalidRankError,
    MismatchedNodeAttributionGroupUpdateError,
    UnknownNodeError,
)
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.sources import Source, SourceSet
from test_support.helpers import make_node_to_ranks


class TestBadPairInfo:
    def test_creation_and_repr(self):
        pair = BadPairInfo(
            local_node="node1",
            remote_node="node2",
            local_dev_ip="192.168.1.1",
            local_device="gpu0",
            remote_dev_ip="192.168.1.2",
            remote_device="gpu1",
        )

        assert pair.local_node == "node1"
        assert pair.remote_node == "node2"
        assert pair.local_dev_ip == "192.168.1.1"
        assert pair.local_device == "gpu0"
        assert pair.remote_dev_ip == "192.168.1.2"
        assert pair.remote_device == "gpu1"

    def test_equality_and_hash(self):
        pair1 = BadPairInfo("node1", "node2", "192.168.1.1", "gpu0")
        pair2 = BadPairInfo("node2", "node1", "192.168.1.2", "gpu1")
        pair3 = BadPairInfo("node1", "node3", "192.168.1.2")

        # Test symmetrical equality
        assert pair1 == pair2
        assert pair2 == pair1
        assert pair1 != pair3
        assert pair1 != "not_a_pair"

        # Test hash functionality
        pairs_set = {pair1, pair2, pair3}
        assert len(pairs_set) == 3

        assert hash(pair1) != hash(pair2)
        assert hash(pair2) == hash(pair3)

    def test_update_bidirectional(self):
        # Test updating in both directions
        pair1 = BadPairInfo("node1", "node2")
        pair2 = BadPairInfo("node1", "node2", local_device="gpu0", remote_device="gpu1")
        pair3 = BadPairInfo("node2", "node1", local_device="gpu1", remote_device="gpu0")

        # Test forward update
        pair1.update(pair2)
        assert pair1.local_device == "gpu0"
        assert pair1.remote_device == "gpu1"

        # Test reverse update
        pair1 = BadPairInfo("node1", "node2")  # Reset
        pair1.update(pair3)
        assert pair1.local_device == "gpu0"  # Should match despite reversed order
        assert pair1.remote_device == "gpu1"

    def test_update_with_none_values(self):
        """Test that update works correctly with None values"""
        pair1 = BadPairInfo("node1", "node2", None, None, None, None)
        pair2 = BadPairInfo("node1", "node2", "192.168.1.1", "gpu0", None, None)
        pair3 = BadPairInfo("node2", "node1", None, None, "192.168.1.2", "gpu1")

        # Test forward update
        pair1.update(pair2)
        assert pair1.local_dev_ip == "192.168.1.1"
        assert pair1.local_device == "gpu0"
        assert pair1.remote_dev_ip is None
        assert pair1.remote_device is None

        # Test reverse update
        pair1 = BadPairInfo("node1", "node2", None, None, None, None)  # Reset
        pair1.update(pair3)
        assert pair1.local_dev_ip == "192.168.1.2"
        assert pair1.local_device == "gpu1"
        assert pair1.remote_dev_ip is None
        assert pair1.remote_device is None


class TestSourceSet:
    def test_creation(self, full_context):
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        source_set = SourceSet("test_symptom").add(base)

        assert source_set.has_bad()
        assert source_set.value() == "test_symptom"

    def test_interesting_sources(self, full_context):
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        source_set = SourceSet("test_symptom").add_interesting(base)

        assert not source_set.has_bad()
        assert len(source_set._interesting) == 1


class TestNodeAttributionGroup:
    def test_add_attribution_multiple(self, full_context):
        group = NodeAttributionGroup("test_node")
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        source_set1 = SourceSet("symptom1").add(base)
        source_set2 = SourceSet("symptom2").add_interesting(base)

        group.add(source_set1)
        group.add(source_set2)

        assert len(group) == 1
        assert "symptom1" in group
        assert "symptom2" in group.attributions._data

    def test_update_mismatched_hostname(self):
        group1 = NodeAttributionGroup("node1")
        group2 = NodeAttributionGroup("node2")

        with pytest.raises(MismatchedNodeAttributionGroupUpdateError):
            group1.update(group2)

    def test_update_preserves_attributions(self, full_context):
        group1 = NodeAttributionGroup("node1")
        group2 = NodeAttributionGroup("node1")
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        source_set1 = SourceSet("symptom1").add(base)
        source_set2 = SourceSet("symptom1").add(base.derived(rule="other_rule"))

        group1.add(source_set1)
        group2.add(source_set2)

        group1.update(group2)
        assert len(group1.attributions["symptom1"]) == 2

    def test_discard(self, full_context):
        group = NodeAttributionGroup("node1")
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        source_set1 = SourceSet("symptom1").add(base)
        source_set2 = SourceSet("symptom1").add(base.derived(rule="other_rule"))
        source_set3 = SourceSet("symptom2").add(base)

        group.add(source_set1)
        group.add(source_set2)
        group.add(source_set3)

        group.discard("symptom1")
        assert set(group.symptoms()) == {"symptom2"}
        group.discard("symptom2")
        assert set(group.symptoms()) == set()

        group.add(source_set1)
        group.add(source_set2)
        group.add(source_set3)
        group.discard("symptom2")
        assert set(group.symptoms()) == {"symptom1"}
        assert set((att.rule for att in group.attributions["symptom1"])) == {"test_rule", "other_rule"}

    def test_duplicate_attribution(self, full_context):
        """Test adding duplicate attributions"""
        group = NodeAttributionGroup("test_node")
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        source_set1 = SourceSet("symptom1").add(base)
        source_set2 = SourceSet("symptom1").add(base)

        group.add(source_set1)
        group.add(source_set2)

        assert len(group.attributions["symptom1"]) == 1


class TestJobNodeAttributions:
    def test_attribute_and_update(self, full_context):
        job_attrs = JobNodeAttributions()
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        source_set1 = SourceSet("symptom1").add(base)
        source_set2 = SourceSet("symptom2").add_interesting(base)

        job_attrs.add(["node1", "node2"], source_set1)
        job_attrs.add(["node2", "node3"], source_set2)

        assert len(job_attrs.attributions) == 3
        assert len(job_attrs.attributions["node2"].attributions) == 1

    def test_discard(self, full_context):
        job_attrs = JobNodeAttributions()
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        source_set1 = SourceSet("test_symptom").add(base)
        source_set2 = SourceSet("test_symptom2").add(base)

        job_attrs.add(["node1", "node2"], source_set1)
        job_attrs.add(["node1", "node2"], source_set2)

        job_attrs.discard(["node1"], ["test_symptom"])

        assert "test_symptom" not in job_attrs.attributions["node1"].attributions
        assert "test_symptom" in job_attrs.attributions["node2"].attributions

        job_attrs.discard(["node1"], ["test_symptom2"])
        assert not job_attrs.attributions["node1"].has_bad()

    def test_update(self, full_context):
        job_attrs1 = JobNodeAttributions()
        job_attrs2 = JobNodeAttributions()
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        # Create different attributions
        source_set1 = SourceSet("symptom1").add(base)
        source_set2 = SourceSet("symptom2").add_interesting(base)
        source_set3 = SourceSet("symptom1").add(base.derived(rule="other_rule"))

        # Add attributions to both objects with some overlap
        job_attrs1.add(["node1", "node2"], source_set1)
        job_attrs2.add(["node2", "node3"], source_set2)
        job_attrs2.add(["node1"], source_set3)  # Same symptom as source_set1, different rule

        # Update and verify
        job_attrs1.update(job_attrs2)

        # Verify correct number of nodes
        assert len(job_attrs1.attributions) == 3
        assert set(job_attrs1.attributions.keys()) == {"node1", "node2", "node3"}

        # Verify node1 has both rules for symptom1
        node1_symptom1_rules = {att.rule for att in job_attrs1.attributions["node1"].attributions["symptom1"]}
        assert node1_symptom1_rules == {"test_rule", "other_rule"}

        # Verify node2 has both symptoms
        assert set(job_attrs1.attributions["node2"].attributions.symptoms()) == {"symptom1"}

        # Verify node3 has only symptom2
        assert set(job_attrs1.attributions["node3"].attributions.symptoms()) == set()
        assert len(job_attrs1.attributions["node3"].attributions["symptom2"]) == 0

    def test_attribute_empty_node_list(self, full_context):
        """Test attributing to empty node list"""
        job_attrs = JobNodeAttributions()
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        source_set = SourceSet("symptom1").add(base)

        job_attrs.add([], source_set)  # Empty node list
        assert len(job_attrs.attributions) == 0

    def test_discard_nonexistent_symptoms(self, full_context):
        """Test discarding symptoms that don't exist"""
        job_attrs = JobNodeAttributions()
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        source_set = SourceSet("symptom1").add(base)

        job_attrs.add(["node1"], source_set)
        job_attrs.discard(["node1"], ["nonexistent_symptom"])

        assert "node1" in job_attrs.attributions
        assert "symptom1" in job_attrs.attributions["node1"].attributions

    def test_update_with_empty_job(self, full_context):
        """Test updating with empty JobNodeAttributions"""
        job_attrs1 = JobNodeAttributions()
        job_attrs2 = JobNodeAttributions()

        # Update empty with empty
        job_attrs1.update(job_attrs2)
        assert len(job_attrs1.attributions) == 0

        # Update non-empty with empty
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        source_set = SourceSet("symptom1").add(base)
        job_attrs1.add(["node1"], source_set)

        job_attrs1.update(job_attrs2)
        assert len(job_attrs1.attributions) == 1

    def test_discard_matching_from_all_nodes(self, full_context):
        """Test discarding symptoms by substring match from all nodes."""
        job_attrs = JobNodeAttributions()
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        job_attrs.add(["node1"], SourceSet("XID-45-PREEMPTIVE_REMOVAL").add(base))
        job_attrs.add(["node1"], SourceSet("cudaErrorLaunchFailure").add(base))
        job_attrs.add(["node1"], SourceSet("XID-145-RLW_SRC_TRACK-8").add(base))
        job_attrs.add(["node2"], SourceSet("cudaErrorNvlinkUncorrectable").add(base))
        job_attrs.add(["node2"], SourceSet("real_symptom").add(base))

        job_attrs.discard_matching_from_all_nodes(["cudaError", "XID-45"])

        # Matched symptoms removed
        assert "XID-45-PREEMPTIVE_REMOVAL" not in job_attrs["node1"]
        assert "cudaErrorLaunchFailure" not in job_attrs["node1"]
        assert "cudaErrorNvlinkUncorrectable" not in job_attrs["node2"]

        # Non-matching symptoms preserved
        assert "XID-145-RLW_SRC_TRACK-8" in job_attrs["node1"]
        assert "real_symptom" in job_attrs["node2"]

    def test_discard_matching_from_all_nodes_no_match(self, full_context):
        """When no symptoms match substrings, nothing is discarded."""
        job_attrs = JobNodeAttributions()
        now = datetime.now()
        base = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        job_attrs.add(["node1"], SourceSet("symptom_A").add(base))
        job_attrs.discard_matching_from_all_nodes(["no_match_pattern"])

        assert "symptom_A" in job_attrs["node1"]


def test_extract_node_exceptions():
    tz = ZoneInfo("America/Los_Angeles")
    job_info = (
        JobInfoBuilder(
            id="test_job_123",
            type="training",
            state="COMPLETED",
            job_start_time=datetime.now(tz=tz),
            node_to_ranks=make_node_to_ranks(["node1", "node2"], 4),
            framework="pytorch",
            cluster="test-cluster",
            source_name="artificial slurm",
            timezone=tz,
        )
        .srun_cmd("srun test.py")
        .exit_code_signal(0)
        .job_end_time(datetime.now(tz=tz))
        .time_limit(600)
        .build()
    )

    # Test unknown node
    with pytest.raises(UnknownNodeError):
        job_info.verify_node("fancy")

    # Test invalid rank
    with pytest.raises(InvalidRankError):
        job_info.get_name_from_rank(100)


if __name__ == "__main__":
    pytest.main([__file__])
