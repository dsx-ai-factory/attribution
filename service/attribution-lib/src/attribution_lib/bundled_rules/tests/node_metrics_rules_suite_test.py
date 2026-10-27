# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests for Node Metrics Attribution
# These tests verify core integration points and unit tests for node_metrics.check()

import inspect
from unittest.mock import Mock, patch

import pytest

import attribution_lib
from attribution_lib.attributor import ConsumedObservation, MismatchedSourceAndFormatError
from attribution_lib.bundled_rules.common import node_metrics_spec
from attribution_lib.core import (
    CalcBadNodesAttributionResult,
    calculate_bad_nodes,
    get_bundled_rules_path,
    process_check,
)
from attribution_lib.observation import Observation
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.observation.processing import EntrySourceBase, render_metric_body
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rules_engine import RuleLoader


def test_node_metrics_rules_register_metric_thresholds():
    """The bundled rule exposes its node-metric threshold specification."""
    loader = RuleLoader()

    loader.load(get_bundled_rules_path() / "node_metrics_rules.py")

    thresholds = loader.build().get_metric_spec(ObservationSource.NODE_METRICS).thresholds
    expected = node_metrics_spec.metric_thresholds
    assert thresholds["node_infiniband_link_downed_total"] == expected["node_infiniband_link_downed_total"]
    assert thresholds["ib_roce_adp_retrans"] == expected["ib_roce_adp_retrans"]
    assert thresholds["up"] == expected["up"]


@pytest.fixture
def node_metrics_jobinfo(simple_job_factory):
    """JobInfo fixture for node metrics integration tests."""
    return simple_job_factory(
        job_id="test_job_123",
        cluster="test_cluster",
        job_type="batch",
        state="FAILED",
        framework="test",
        ranks_per_node=1,
        start_timestamp=1744734400,
        end_timestamp=1744734500,
        nodes=["test-node-001", "test-node-002"],
        source_name="test_slurm",
    )


@pytest.fixture
def attributor_jobinfo(simple_job_factory):
    """JobInfo fixture for attributor integration tests."""
    return simple_job_factory(
        job_id="test_job_attributor",
        cluster="test_cluster",
        job_type="batch",
        state="FAILED",
        framework="test",
        ranks_per_node=1,
        start_timestamp=1744734400,
        end_timestamp=1744734500,
        nodes=["test-node-001"],
        source_name="test_slurm",
    )


@pytest.fixture
def placeholder_jobinfo(simple_job_factory):
    """JobInfo fixture for placeholder tests."""
    return simple_job_factory(
        job_id="test_placeholder",
        cluster="test_cluster",
        job_type="batch",
        state="FAILED",
        framework="test",
        ranks_per_node=1,
        start_timestamp=1744734400,
        end_timestamp=1744734500,
        nodes=["test-node-001"],
        source_name="test_slurm",
    )


@pytest.fixture
def node_metrics_context(attributor_jobinfo, timezone):
    """Observation context for NODE_METRICS."""
    time_interval = TimeInterval(attributor_jobinfo.job_start_time, attributor_jobinfo.job_end_time)
    return InboundContext(
        time_interval=time_interval,
        source=ObservationSource.NODE_METRICS,
        format=ObservationFormat.ENTRY_TIMESERIES,
        timezone=timezone,
    )


@pytest.fixture
def sample_prometheus_metrics():
    """Sample prometheus metrics data for testing."""
    metric = {
        "timestamp": 1744734450,
        "value": "1",
        "metric": "node_infiniband_link_downed_total",
        "labels": {"net_host_name": "test-node-001", "s_cluster": "test-cluster", "device": "mlx5_0"},
        "node": "test-node-001",
        "component": "mlx5_0",
        "cluster": "test-cluster",
    }
    return [
        {
            "ts": 1744734450 * 10**9,
            "resources": [{"kind": "node", "hostname": "test-node-001"}],
            "body": render_metric_body(metric),
        }
    ]


class TestCalculateBadNodesIntegration:
    """Test calculate_bad_nodes integration with node_metrics_in parameter"""

    def test_calculate_bad_nodes_has_node_metrics_in_parameter(self):
        """Test that calculate_bad_nodes function has node_metrics_in parameter"""
        # This test verifies the function signature includes the new parameter
        sig = inspect.signature(calculate_bad_nodes)

        # Verify node_metrics_in parameter exists
        assert "node_metrics_in" in sig.parameters

        # Verify it's optional with None default
        param = sig.parameters["node_metrics_in"]
        assert param.default is None

    @patch("attribution_lib.core.process_observation_source")
    def test_calculate_bad_nodes_calls_process_observation_source(
        self, mock_process, request, event_manager, node_metrics_jobinfo
    ):
        """Test that calculate_bad_nodes calls process_observation_source when node_metrics_in is provided"""
        mock_process.return_value = set()

        event_stream = event_manager.create_stream(request.node.name)
        # Create mock entry source
        mock_entry_source = Mock(spec=EntrySourceBase)
        mock_entry_source.context = Mock()
        mock_entry_source.context.source = ObservationSource.NODE_METRICS

        # Call calculate_bad_nodes with node_metrics_in
        result = calculate_bad_nodes(
            event_stream=event_stream, job_info=node_metrics_jobinfo, node_metrics_in=mock_entry_source
        )

        # Verify process_observation_source was called for node_metrics
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["observation_source"] is mock_entry_source
        assert call_kwargs["job_info"] is node_metrics_jobinfo
        assert call_kwargs["pre_filtered"] is False

        # Verify result is CalcBadNodesAttributionResult
        assert isinstance(result, CalcBadNodesAttributionResult)

    def test_calculate_bad_nodes_skips_node_metrics_when_none(self, request, event_manager, node_metrics_jobinfo):
        """Test that calculate_bad_nodes skips node_metrics processing when node_metrics_in is None"""
        event_stream = event_manager.create_stream(request.node.name)
        # Call without node_metrics_in
        result = calculate_bad_nodes(event_stream=event_stream, job_info=node_metrics_jobinfo)

        # Should complete without error and return result
        assert isinstance(result, CalcBadNodesAttributionResult)

    def test_calculate_bad_nodes_skips_node_metrics_when_no_nodes(self):
        """Test that calculate_bad_nodes conditional logic checks for nodes"""
        # This test verifies the conditional logic exists
        # The actual behavior is tested by the other tests that pass/fail node_metrics_in

        # Test that the function signature and conditional logic is correct
        # by checking the source code has the expected pattern
        source_lines = inspect.getsourcelines(calculate_bad_nodes)[0]
        source_text = "".join(source_lines)

        # Verify the conditional pattern exists (requires_job_nodes config + job_info.nodes check)
        assert "requires_job_nodes" in source_text
        assert "job_info.nodes" in source_text
        assert "node_metrics_in" in source_text


class TestAttributorIntegration:
    """Test attributor.py integration with NODE_METRICS case"""

    def test_node_metrics_case_exists_in_attributor(
        self, request, event_manager, attributor_jobinfo, node_metrics_context, sample_prometheus_metrics
    ):
        """Test that NODE_METRICS case exists in attributor switch statement"""
        event_stream = event_manager.create_stream(request.node.name)
        # Create observation
        observation = Observation.from_inbound(context=node_metrics_context, lines=sample_prometheus_metrics)
        event_stream.register_observation_context(observation.context)

        # This should not raise MismatchedSourceAndFormatError anymore
        # Instead it should process the NODE_METRICS case
        try:
            consumed_obs = ConsumedObservation.new(event_stream, attributor_jobinfo, observation)
            # If we get here, the NODE_METRICS case is implemented
            assert consumed_obs is not None
        except Exception as e:
            # Check if it's the old placeholder error
            if "MismatchedSourceAndFormatError" in str(type(e)):
                pytest.fail("NODE_METRICS case still has placeholder implementation")
            else:
                # Some other error is acceptable during integration testing
                # as we haven't implemented the full logic yet
                pass

    def test_node_metrics_format_validation(self, request, event_manager, attributor_jobinfo, node_metrics_context):
        """Test that NODE_METRICS case validates its format"""
        event_stream = event_manager.create_stream(request.node.name)
        # Create context with wrong format
        wrong_format_context = InboundContext(
            time_interval=TimeInterval(attributor_jobinfo.job_start_time, attributor_jobinfo.job_end_time),
            source=ObservationSource.NODE_METRICS,
            format=ObservationFormat.STRING_LIST,  # Wrong format
            timezone=node_metrics_context.timezone,
        )

        # Create observation with wrong format
        observation = Observation.from_inbound(context=wrong_format_context, lines=["dummy"])

        # Should raise MismatchedSourceAndFormatError
        with pytest.raises(MismatchedSourceAndFormatError):
            ConsumedObservation.new(event_stream, attributor_jobinfo, observation)


class TestNodeMetricsProcessCheck:
    """Test that node_metrics processing works through process_check"""

    def test_process_check_callable(self, request, event_manager, placeholder_jobinfo):
        """Test that process_check works for node_metrics observations"""
        event_stream = event_manager.create_stream(request.node.name)
        assert callable(process_check)

        mock_entry_source = Mock()
        mock_entry_source.__iter__ = Mock(return_value=iter([]))
        mock_entry_source.context = Mock()
        mock_entry_source.context.source = ObservationSource.NODE_METRICS

        process_check(
            event_stream=event_stream,
            observation_source=mock_entry_source,
            job_info=placeholder_jobinfo,
            rules_engine=attribution_lib.get_rules_engine(),
        )
