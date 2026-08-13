# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import datetime
from zoneinfo import ZoneInfo

import pytest

from attribution_lib.attr_types import JobNodeAttributions, NodeAttributionGroup
from attribution_lib.core import process_check
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.observation.inbound import (
    Format as ObservationFormat,
    Source as ObservationSource,
)
from attribution_lib.sources import SourceSet
from test_support.helpers import make_node_to_ranks

LINK_DOWN_EAST_WEST = "link down(east-west)"
NODE_DOWN_UFM = "node down reported by ufm"


@pytest.fixture(scope="class")
def ufm_timezone():
    """UFM tests require UTC timezone to match the test data."""
    return ZoneInfo("UTC")


@pytest.fixture
def job_info(ufm_timezone):
    """Fixture providing job parameters for testing"""
    return (
        JobInfoBuilder(
            id="test_job_123",
            job_start_time=datetime.datetime(2025, 6, 18, 17, 46, 0, tzinfo=ufm_timezone),
            type="test",
            state="COMPLETED",
            cluster="test_cluster",
            node_to_ranks=make_node_to_ranks(["host-c8e09d93", "host-f77b60a9", "host-59805cc2"], 1),
            framework="test_framework",
            source_name="testing",
            timezone=ufm_timezone,
        )
        .interest_start_time(datetime.datetime(2025, 6, 18, 17, 47, 0, tzinfo=ufm_timezone))
        .interest_end_time(datetime.datetime(2025, 6, 18, 17, 48, 0, tzinfo=ufm_timezone))
        .job_end_time(datetime.datetime(2025, 6, 18, 17, 52, 0, tzinfo=ufm_timezone))
        .time_limit(6000)
        .last_tflops_time(datetime.datetime(2025, 6, 18, 17, 47, 0, tzinfo=ufm_timezone))
        .exit_code_signal(0)
        .num_gpus(8)
        .build()
    )


@pytest.fixture
def job_info_2(ufm_timezone):
    """Fixture providing job parameters for testing"""
    return (
        JobInfoBuilder(
            id="test_job_123",
            job_start_time=datetime.datetime(2025, 6, 19, 10, 47, 0, tzinfo=ufm_timezone),
            type="test",
            state="COMPLETED",
            cluster="test_cluster",
            node_to_ranks=make_node_to_ranks(["host-c8e09d93", "host-f77b60a9", "host-59805cc2"], 1),
            framework="test_framework",
            source_name="testing",
            timezone=ufm_timezone,
        )
        .interest_start_time(datetime.datetime(2025, 6, 19, 17, 47, 0, tzinfo=ufm_timezone))
        .interest_end_time(datetime.datetime(2025, 6, 19, 17, 48, 0, tzinfo=ufm_timezone))
        .job_end_time(datetime.datetime(2025, 6, 18, 17, 49, 0, tzinfo=ufm_timezone))
        .time_limit(6000)
        .last_tflops_time(datetime.datetime(2025, 6, 18, 17, 47, 0, tzinfo=ufm_timezone))
        .exit_code_signal(0)
        .num_gpus(8)
        .build()
    )


@pytest.fixture
def host_to_switch_link_down_log_entries():
    """Fixture providing sample log entries for testing"""
    return [
        {
            "body": "2025-06-18 17:47:26.839 [13406653] [329] WARNING Site [TEST1-FAB1] "
            "[Fabric_Topology] Link [Source 0200000000000001_1  TO Dest: "
            "0200000000000003_7]: Link went down: "
            "(Computer:host-c8e09d93 (0200000000000002)) "
            "0200000000000001:1 - (Switch:SW1.1-DH1:7) 0200000000000003:7",
            "node": "test-fab1-ufm01",
            "timestamp": 1750268847276617503,
        },
        {
            "body": " 2025-06-18 17:47:58.779 [13406663] [329] WARNING Site [TEST1-FAB1] "
            "[Fabric_Topology] Link [Source 0200000000000004_1  TO Dest: "
            "0200000000000005_28]: Link went down: "
            "(Computer:host-f77b60a9 (0200000000000004)) "
            "0200000000000004:1 - (Switch:SW2.1-DH2:28) 0200000000000005:28, "
            "cable S/N: TESTSN00000001",
            "node": "test-fab1-ufm01",
            "timestamp": 1750268879256741095,
        },
    ]


@pytest.fixture
def host_to_switch_fatal_link_down_log_entries():
    """Fixture providing log entries 1 minute before job_end_time for link down(east-west) testing"""
    return [
        {
            "body": "2025-06-18 17:51:00.000 [13406653] [329] WARNING Site [TEST1-FAB1] "
            "[Fabric_Topology] Link [Source 0200000000000001_1  TO Dest: "
            "0200000000000003_7]: Link went down: "
            "(Computer:host-c8e09d93 (0200000000000002)) "
            "0200000000000001:1 - (Switch:SW1.1-DH1:7) 0200000000000003:7",
            "node": "test-fab1-ufm01",
            "timestamp": 1750269060000000000,  # 2025-06-18 17:51:00 - 60 seconds before job_end_time (17:52:00)
        },
        {
            "body": " 2025-06-18 17:51:30.000 [13406663] [329] WARNING Site [TEST1-FAB1] "
            "[Fabric_Topology] Link [Source 0200000000000004_1  TO Dest: "
            "0200000000000005_28]: Link went down: "
            "(Computer:host-f77b60a9 (0200000000000004)) "
            "0200000000000004:1 - (Switch:SW2.1-DH2:28) 0200000000000005:28, "
            "cable S/N: TESTSN00000001",
            "node": "test-fab1-ufm01",
            "timestamp": 1750269090000000000,  # 2025-06-18 17:51:30 - 30 seconds before job_end_time (17:52:00)
        },
    ]


@pytest.fixture
def switch_to_host_link_down_log_entries():
    """Fixture providing sample log entries for testing"""
    return [
        {
            "body": " 2025-06-18 17:47:58.779 [13289852] [329] WARNING Site [TEST1-FAB1] [Fabric_Topology] Link [Source 0200000000000006_23  TO Dest: 0200000000000007_1]: Link went down: (Switch:SW3.1-DH3:1/12/1) 0200000000000006:23 - (Computer:host-59805cc2 (0200000000000008)) 0200000000000007:1, cable S/N: TESTSN00000002",
            "node": "test-fab1-ufm01",
            "timestamp": 1750268847276617503,
        },
    ]


@pytest.fixture
def node_down_log_entries():
    """Fixture providing sample 'node is Down' log entries for testing"""
    return [
        {
            "body": "2025-06-18 17:47:26.839 [13273115] [331] WARNING Site [TEST1-FAB1] "
            "[Fabric_Topology] Site [default(1234) / NA / NA / NA]: "
            "Site configuration changes: 0200000000000002 (host-c8e09d93) node is Down",
            "node": "test-fab1-ufm01",
            "timestamp": 1750268847276617503,
        },
        {
            "body": "2025-06-18 17:47:58.779 [13273116] [331] WARNING Site [TEST1-FAB1] "
            "[Fabric_Topology] Site [default(1234) / NA / NA / NA]: "
            "Site configuration changes: 0200000000000004 (host-f77b60a9) node is Down",
            "node": "test-fab1-ufm01",
            "timestamp": 1750268879256741095,
        },
    ]


@pytest.fixture
def node_down_fatal_log_entries():
    """Fixture providing 'node is Down' log entries 1 minute before job_end_time"""
    return [
        {
            "body": "2025-06-18 17:51:00.000 [13273115] [331] WARNING Site [TEST1-FAB1] "
            "[Fabric_Topology] Site [default(1234) / NA / NA / NA]: "
            "Site configuration changes: 0200000000000002 (host-c8e09d93) node is Down",
            "node": "test-fab1-ufm01",
            "timestamp": 1750269060000000000,  # 2025-06-18 17:51:00 - 60 seconds before job_end_time (17:52:00)
        },
        {
            "body": "2025-06-18 17:51:30.000 [13273116] [331] WARNING Site [TEST1-FAB1] "
            "[Fabric_Topology] Site [default(1234) / NA / NA / NA]: "
            "Site configuration changes: 0200000000000004 (host-f77b60a9) node is Down",
            "node": "test-fab1-ufm01",
            "timestamp": 1750269090000000000,  # 2025-06-18 17:51:30 - 30 seconds before job_end_time (17:52:00)
        },
    ]


def test_host_to_switch_link_down_attribution(host_to_switch_link_down_log_entries, job_info, attribution_test_builder):
    """Test parsing of link down log entries"""
    result = attribution_test_builder(
        job_info,
        host_to_switch_link_down_log_entries,
        ObservationSource.UFMLOG,
        ObservationFormat.UFMLOG,
        process_check,
    )
    observation_source = result.observation_source
    nodes = result.get_node_attributions()
    # Verify results
    assert isinstance(nodes, JobNodeAttributions)
    assert len(nodes.attributions) == 2

    node_name = job_info.nodes[0]
    node_attribution_group = nodes[node_name]
    assert isinstance(node_attribution_group, NodeAttributionGroup)
    assert node_attribution_group.has_bad()  # Should not be marked as bad

    # Access the source_set directly (bypassing has_bad filter)
    # For interesting attributions, link down(east-west) won't be in the filtered symptom set
    source_set = node_attribution_group.attributions[LINK_DOWN_EAST_WEST]
    assert isinstance(source_set, SourceSet)
    assert source_set.has_bad()
    assert not source_set.has_interesting()
    assert len(source_set) == 1  # Should have one source

    # verify source (get from _interesting set since it's not in _bad)
    source = next(iter(source_set))
    assert source.source_context is not None
    assert source.source_context.source == ObservationSource.UFMLOG
    assert source.source_context.format == ObservationFormat.UFMLOG
    assert source.source_context.timezone == observation_source.context.timezone
    assert source.source_context.time_interval == observation_source.context.time_interval
    assert source.line_num == 1
    assert "Link went down: (Computer:host-c8e09d93 (0200000000000002))" in source.datum

    node_name = job_info.nodes[1]
    assert node_name in nodes
    node_attribution_group = nodes[node_name]
    assert len(node_attribution_group) == 1  # Should have one source
    assert node_attribution_group.has_bad()

    # Verify the source information for the second node
    link_down_sources = node_attribution_group.attributions[LINK_DOWN_EAST_WEST]
    assert link_down_sources.has_bad()

    # Get the first (and only) source from _interesting
    source = next(iter(link_down_sources._bad))

    # Verify the source context is properly set (not None)
    assert source.source_context is not None
    assert source.source_context.source == ObservationSource.UFMLOG
    assert source.source_context.format == ObservationFormat.UFMLOG
    assert source.source_context.timezone == observation_source.context.timezone
    assert source.source_context.time_interval == observation_source.context.time_interval

    # Verify the line number is properly set (should be 2 for second entry)
    assert source.line_num == 2

    # Verify the datum contains the log line
    assert "Link went down" in source.datum
    assert "host-f77b60a9" in source.datum


def test_switch_to_host_link_down_attribution(switch_to_host_link_down_log_entries, job_info, attribution_test_builder):
    result = attribution_test_builder(
        job_info,
        switch_to_host_link_down_log_entries,
        ObservationSource.UFMLOG,
        ObservationFormat.UFMLOG,
        process_check,
    )
    observation_source = result.observation_source
    nodes = result.get_node_attributions()
    # Verify results
    assert isinstance(nodes, JobNodeAttributions)
    assert len(nodes.attributions) == 1

    node_name = job_info.nodes[2]
    node_attribution_group = nodes[node_name]
    assert isinstance(node_attribution_group, NodeAttributionGroup)
    assert node_attribution_group.has_bad()  # Should not be marked as bad

    source_set = node_attribution_group.attributions[LINK_DOWN_EAST_WEST]

    # verify source (get from _interesting set since it's not in _bad)
    source = next(iter(source_set._bad))
    assert source.source_context is not None
    assert source.source_context.source == ObservationSource.UFMLOG
    assert source.source_context.format == ObservationFormat.UFMLOG
    assert source.source_context.timezone == observation_source.context.timezone
    assert source.source_context.time_interval == observation_source.context.time_interval
    assert source.line_num == 1
    assert "Link went down: (Switch:SW3.1-DH3:1/12/1) 0200000000000006:23 - (Computer:host-59805cc2" in source.datum


def test_non_matching_host_to_switch_link_down_attribution(
    host_to_switch_link_down_log_entries, job_info_2, attribution_test_builder
):
    """Test parsing of link down log entries"""
    result = attribution_test_builder(
        job_info_2,
        host_to_switch_link_down_log_entries,
        ObservationSource.UFMLOG,
        ObservationFormat.UFMLOG,
        process_check,
    )
    nodes = result.get_node_attributions()

    # Verify results
    assert isinstance(nodes, JobNodeAttributions)
    assert len(nodes.attributions) == 0


def test_host_to_switch_fatal_link_down_attribution(
    host_to_switch_fatal_link_down_log_entries, job_info, attribution_test_builder
):
    """Test parsing of link down log entries"""
    result = attribution_test_builder(
        job_info,
        host_to_switch_fatal_link_down_log_entries,
        ObservationSource.UFMLOG,
        ObservationFormat.UFMLOG,
        process_check,
    )
    observation_source = result.observation_source
    nodes = result.get_node_attributions()
    # Verify results
    assert isinstance(nodes, JobNodeAttributions)
    assert len(nodes.attributions) == 2

    node_name = job_info.nodes[0]
    node_attribution_group = nodes[node_name]
    assert isinstance(node_attribution_group, NodeAttributionGroup)
    assert node_attribution_group.has_bad()
    assert LINK_DOWN_EAST_WEST in node_attribution_group

    # verify symptoms set is properly set
    symptom_set = node_attribution_group.symptoms()
    assert LINK_DOWN_EAST_WEST in symptom_set

    # verify source_set is properly set
    source_set = node_attribution_group.attributions[LINK_DOWN_EAST_WEST]
    assert isinstance(source_set, SourceSet)
    assert source_set.has_bad()
    assert not source_set.has_interesting()
    assert len(source_set) == 1  # Should have one source

    # verify source
    source = next(iter(source_set))
    assert source.source_context is not None
    assert source.source_context.source == ObservationSource.UFMLOG
    assert source.source_context.format == ObservationFormat.UFMLOG
    assert source.source_context.timezone == observation_source.context.timezone
    assert source.source_context.time_interval == observation_source.context.time_interval
    assert source.line_num == 1
    assert "Link went down: (Computer:host-c8e09d93 (0200000000000002))" in source.datum

    node_name = job_info.nodes[1]
    assert node_name in nodes
    node_attribution_group = nodes[node_name]
    assert len(node_attribution_group) == 1
    assert LINK_DOWN_EAST_WEST in node_attribution_group.symptoms()
    assert node_attribution_group.has_bad()

    # Verify the source information for the second node
    # Get the source set for the LINK_DOWN symptom
    link_down_sources = node_attribution_group.attributions[LINK_DOWN_EAST_WEST]
    assert len(link_down_sources) == 1  # Should have one source

    # Get the first (and only) source
    source = next(iter(link_down_sources))

    # Verify the source context is properly set (not None)
    assert source.source_context is not None
    assert source.source_context.source == ObservationSource.UFMLOG
    assert source.source_context.format == ObservationFormat.UFMLOG
    assert source.source_context.timezone == observation_source.context.timezone
    assert source.source_context.time_interval == observation_source.context.time_interval

    # Verify the line number is properly set (should be 2 for second entry)
    assert source.line_num == 2

    # Verify the datum contains the log line
    assert "Link went down" in source.datum
    assert "host-f77b60a9" in source.datum


def test_node_down_attribution(node_down_log_entries, job_info, attribution_test_builder):
    """Test parsing of node down log entries"""
    result = attribution_test_builder(
        job_info, node_down_log_entries, ObservationSource.UFMLOG, ObservationFormat.UFMLOG, process_check
    )
    observation_source = result.observation_source
    nodes = result.get_node_attributions()
    # Verify results
    assert isinstance(nodes, JobNodeAttributions)
    assert len(nodes.attributions) == 2

    node_name = job_info.nodes[0]
    node_attribution_group = nodes[node_name]
    assert isinstance(node_attribution_group, NodeAttributionGroup)
    assert not node_attribution_group.has_bad()

    # Access the source_set directly - UFM node-down is interesting, not bad
    source_set = node_attribution_group.attributions[NODE_DOWN_UFM]
    assert isinstance(source_set, SourceSet)
    assert not source_set.has_bad()
    assert source_set.has_interesting()
    assert len(source_set._interesting) == 1  # Should have one interesting source

    # verify source
    source = next(iter(source_set._interesting))
    assert source.source_context is not None
    assert source.source_context.source == ObservationSource.UFMLOG
    assert source.source_context.format == ObservationFormat.UFMLOG
    assert source.source_context.timezone == observation_source.context.timezone
    assert source.source_context.time_interval == observation_source.context.time_interval
    assert source.line_num == 1
    assert "(host-c8e09d93) node is Down" in source.datum

    node_name = job_info.nodes[1]
    assert node_name in nodes
    node_attribution_group = nodes[node_name]
    assert not node_attribution_group.has_bad()

    # Verify the source information for the second node
    node_down_sources = node_attribution_group.attributions[NODE_DOWN_UFM]
    assert node_down_sources.has_interesting()

    # Get the first (and only) source from _interesting
    source = next(iter(node_down_sources._interesting))

    # Verify the source context is properly set (not None)
    assert source.source_context is not None
    assert source.source_context.source == ObservationSource.UFMLOG
    assert source.source_context.format == ObservationFormat.UFMLOG
    assert source.source_context.timezone == observation_source.context.timezone
    assert source.source_context.time_interval == observation_source.context.time_interval

    # Verify the line number is properly set (should be 2 for second entry)
    assert source.line_num == 2

    # Verify the datum contains the log line
    assert "node is Down" in source.datum
    assert "host-f77b60a9" in source.datum


def test_non_matching_node_down_attribution(node_down_log_entries, job_info_2, attribution_test_builder):
    """Test parsing of node down log entries outside job time boundaries"""
    result = attribution_test_builder(
        job_info_2, node_down_log_entries, ObservationSource.UFMLOG, ObservationFormat.UFMLOG, process_check
    )
    nodes = result.get_node_attributions()

    # Verify results
    assert isinstance(nodes, JobNodeAttributions)
    assert len(nodes.attributions) == 0


def test_node_down_fatal_attribution(node_down_fatal_log_entries, job_info, attribution_test_builder):
    """Test parsing of node down log entries near job end time"""
    result = attribution_test_builder(
        job_info, node_down_fatal_log_entries, ObservationSource.UFMLOG, ObservationFormat.UFMLOG, process_check
    )
    observation_source = result.observation_source
    nodes = result.get_node_attributions()
    # Verify results
    assert isinstance(nodes, JobNodeAttributions)
    assert len(nodes.attributions) == 2

    node_name = job_info.nodes[0]
    node_attribution_group = nodes[node_name]
    assert isinstance(node_attribution_group, NodeAttributionGroup)
    assert not node_attribution_group.has_bad()

    # UFM node-down is interesting, not bad - verify it exists as interesting
    assert node_attribution_group.attributions.has_interesting()

    # verify source_set is properly set
    source_set = node_attribution_group.attributions[NODE_DOWN_UFM]
    assert isinstance(source_set, SourceSet)
    assert not source_set.has_bad()
    assert source_set.has_interesting()
    assert len(source_set._interesting) == 1  # Should have one interesting source

    # verify source
    source = next(iter(source_set._interesting))
    assert source.source_context is not None
    assert source.source_context.source == ObservationSource.UFMLOG
    assert source.source_context.format == ObservationFormat.UFMLOG
    assert source.source_context.timezone == observation_source.context.timezone
    assert source.source_context.time_interval == observation_source.context.time_interval
    assert source.line_num == 1
    assert "(host-c8e09d93) node is Down" in source.datum

    node_name = job_info.nodes[1]
    assert node_name in nodes
    node_attribution_group = nodes[node_name]
    assert not node_attribution_group.has_bad()
    assert node_attribution_group.attributions.has_interesting()

    # Verify the source information for the second node
    # Get the source set for the NODE_DOWN_UFM symptom
    node_down_sources = node_attribution_group.attributions[NODE_DOWN_UFM]
    assert len(node_down_sources._interesting) == 1  # Should have one interesting source

    # Get the first (and only) source
    source = next(iter(node_down_sources._interesting))

    # Verify the source context is properly set (not None)
    assert source.source_context is not None
    assert source.source_context.source == ObservationSource.UFMLOG
    assert source.source_context.format == ObservationFormat.UFMLOG
    assert source.source_context.timezone == observation_source.context.timezone
    assert source.source_context.time_interval == observation_source.context.time_interval

    # Verify the line number is properly set (should be 2 for second entry)
    assert source.line_num == 2

    # Verify the datum contains the log line
    assert "node is Down" in source.datum
    assert "host-f77b60a9" in source.datum
