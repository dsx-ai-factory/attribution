# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest
from pydantic_core import PydanticCustomError

from attribution_lib.exceptions import (
    MissingRequiredFieldError,
    UnhandledObservationFormatError,
    UnknownObservationFormatError,
    UnknownObservationSourceError,
    UnknownResourceError,
)
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.observation import (
    Context,
    Observation,
)
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Id,
    LokiAppEntry,
    LokiHealthCheckEntry,
    LokiSyslogEntry,
    Observation as InboundObservation,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.observation.processing import (
    BodyFallbacks,
    EntrySourceBase,
    LokiDictEntry,
    LokiDictEntrySource,
    PrometheusMetricsEntrySource,
    RawLineEntry,
    RawLineEntrySource,
    RawLokiStreamsEntrySource,
    TimeseriesEntrySource,
    _extract_timestamp_from_iso_format,
    _extract_timestamp_from_log_line,
    _extract_timestamp_from_pytorch_format,
    extract_body_and_node,
    raw_loki_result_helper,
    raw_loki_unpack,
)
from attribution_lib.resources import Node, Replica
from attribution_lib.resources.resource_set import ResourceSetClass
from attribution_lib.rules_engine import process_observation_source
from test_support.helpers import make_node_to_ranks, observation_base

JOB_NODE = "fake-node-a"
OTHER_JOB_NODE = "fake-node-b"
NODE_RESOURCE = [{"kind": "node", "hostname": JOB_NODE}]


@pytest.fixture
def job_params(timezone):
    return (
        JobInfoBuilder(
            id="test_job",
            node_to_ranks=make_node_to_ranks(["node1", "node2"], 2),
            framework="test",
            type="test",
            state="COMPLETED",
            job_start_time=datetime.now(tz=timezone),
            cluster="test_cluster",
            source_name="artifical slurm",
            timezone=timezone,
        )
        .exit_code_signal(0)
        .job_end_time(datetime.now(tz=timezone) + timedelta(minutes=5))
        .time_limit(1000)
        .build()
    )


@pytest.fixture
def applog_context(time_interval, job_resources, timezone):
    """Standard APPLOG context fixture."""
    return InboundContext(
        time_interval, job_resources, ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES, timezone
    )


@pytest.fixture
def syslog_context(time_interval, job_resources, timezone):
    """Standard SYSLOG context fixture."""
    return InboundContext(
        time_interval, job_resources, ObservationSource.SYSLOG, ObservationFormat.RAW_LOKI_STREAMS, timezone
    )


@pytest.fixture
def healthcheck_context(time_interval, job_resources, timezone):
    """Standard HEALTH_CHECK context fixture."""
    return InboundContext(
        time_interval, job_resources, ObservationSource.HEALTH_CHECK, ObservationFormat.PROMETHEUS_METRICS, timezone
    )


@pytest.fixture
def loki_dict_context(time_interval, job_resources, timezone):
    """Standard LOKI_DICT_LIST context fixture."""
    return InboundContext(
        time_interval, job_resources, ObservationSource.APPLOG, ObservationFormat.LOKI_DICT_LIST, timezone
    )


@pytest.fixture
def processing_context(applog_context):
    """Standard processing context (Context wrapper around InboundContext)."""
    return Context(**vars(applog_context), id=Id(12345))


@pytest.fixture
def syslog_loki_dict_context(time_interval, job_resources, timezone):
    """Standard SYSLOG LOKI_DICT_LIST context fixture."""
    return InboundContext(
        time_interval, job_resources, ObservationSource.SYSLOG, ObservationFormat.LOKI_DICT_LIST, timezone
    )


@pytest.fixture
def loki_processing_context(loki_dict_context):
    """Processing context for LOKI_DICT_LIST format."""
    return Context(**vars(loki_dict_context), id=Id(12345))


@pytest.fixture
def sample_lines():
    """Sample log lines for testing."""
    return ["line1", "line2", "line3"]


@pytest.fixture
def sample_metric_data():
    """Sample metric data for testing."""
    return {
        "timestamp": 1642608000,
        "metric": "cpu_usage",
        "node": "node1",
        "value": "75.5",
        "labels": {"instance": "node1:9100"},
    }


@pytest.fixture
def metric_data_with_fields():
    """Metric data with explicit component and cluster fields."""
    return {
        "timestamp": 1642608000,
        "metric": "node_infiniband_link_downed_total",
        "node": "test-node-001",
        "value": "1",
        "component": "mlx5_0",
        "cluster": "host-c771bdf0",
        "labels": {"net_host_name": "test-node-001", "s_cluster": "host-c771bdf0", "device": "mlx5_0"},
    }


@pytest.fixture
def metric_data_from_labels():
    """Metric data with component and cluster only in labels."""
    return {
        "timestamp": 1642608000,
        "metric": "node_infiniband_link_downed_total",
        "node": "test-node-002",
        "value": "1",
        "labels": {
            "net_host_name": "test-node-002",
            "s_cluster": "host-c771bdf0",
            "device": "mlx5_1",
            "instance": "test-node-002:9100",
        },
    }


@pytest.fixture
def timeseries_context(job_resources, timezone):
    """The format declares its own window, so these tests pin it open and let each entry carry its own ts."""
    return InboundContext(
        TimeInterval(None, None),
        job_resources,
        ObservationSource.SYSLOG,
        ObservationFormat.ENTRY_TIMESERIES,
        timezone,
    )


@pytest.fixture
def syslog_job_info(simple_job_factory):
    """A two-node job, so a test can name a second one and still be inside the workload."""
    return simple_job_factory(nodes=[JOB_NODE, OTHER_JOB_NODE])


def entry(body, *, ts=None, resources=None):
    """A wire entry, omitting the optional keys the caller left out rather than nulling them."""
    made = {"body": body}
    if ts is not None:
        made["ts"] = ts
    if resources is not None:
        made["resources"] = resources
    return made


def ns(moment):
    return int(moment.timestamp() * 1e9)


def only_entry(context, resources):
    """The single entry of a one-entry observation declaring `resources`."""
    return next(iter(TimeseriesEntrySource(Observation(context, [entry("a body", resources=resources)]))))


@pytest.fixture
def extract(timezone):
    """The bodies that survive time filtering, read straight off the rules engine seam."""

    def _extract(job_info, source, format, lines, *, time_interval=None):
        engine = Mock()
        process_observation_source(
            event_stream=None,
            observation_source=observation_base(job_info, timezone, source, format, lines, time_interval=time_interval),
            job_info=job_info,
            rules_engine=engine,
            pre_filtered=True,
        )
        processed = engine.observation_processor.return_value.process_line
        return [call.args[0].datum for call in processed.call_args_list]

    return _extract


class TestTimeInterval:
    def test_equality(self, time_interval):
        interval2 = TimeInterval(time_interval.start, time_interval.end)
        assert time_interval == interval2

        different_interval = TimeInterval(time_interval.start, time_interval.start + timedelta(minutes=10))
        assert time_interval != different_interval

    def test_hash(self, time_interval):
        interval2 = TimeInterval(time_interval.start, time_interval.end)
        assert hash(time_interval) == hash(interval2)


class TestInboundContext:
    def test_equality(self, applog_context):
        context1 = applog_context
        context2 = InboundContext(
            applog_context.time_interval,
            applog_context.resources,
            applog_context.source,
            applog_context.format,
            applog_context.timezone,
        )
        assert context1 == context2

        different_context = InboundContext(
            applog_context.time_interval,
            applog_context.resources,
            ObservationSource.SYSLOG,  # Different source
            applog_context.format,
            applog_context.timezone,
        )
        assert context1 != different_context

    def test_hash(self, applog_context):
        context1 = applog_context
        context2 = InboundContext(
            applog_context.time_interval,
            applog_context.resources,
            applog_context.source,
            applog_context.format,
            applog_context.timezone,
        )
        assert hash(context1) == hash(context2)


class TestObservation:
    def test_raw_line_source_creation(self, applog_context, sample_lines):
        observation = Observation.from_inbound(applog_context, sample_lines)

        source = EntrySourceBase.from_observation(observation)
        assert isinstance(source, RawLineEntrySource)

        # Test iteration
        line_objects = list(source)
        assert len(line_objects) == 3
        assert all(isinstance(line, RawLineEntry) for line in line_objects)
        assert [line.body() for line in line_objects] == sample_lines

    def test_syslog_raw_loki(self, syslog_context):
        logroot = Path(__file__).parent / "app_logs"
        path = logroot / "test-raw-loki.json"
        result = json.loads(path.read_bytes())
        observation = Observation.from_inbound(syslog_context, result)

        source = EntrySourceBase.from_observation(observation)
        assert isinstance(source, RawLokiStreamsEntrySource)

        line_objects = list(source)
        assert all(isinstance(line, LokiDictEntry) for line in line_objects)

    def test_invalid_comparison(self, applog_context):
        """Test comparing Observation with invalid type"""
        observation = InboundObservation(applog_context, ["line1"])
        with pytest.raises(NotImplementedError):
            observation == "not an observation"

    def test_duplicate_hash(self, applog_context):
        """Test hashing observation with undefined format raises exception"""
        observation1 = Observation.from_inbound(applog_context, ["line1"])
        observation2 = Observation.from_inbound(applog_context, ["line1"])

        assert observation1.context.id == observation2.context.id

    def test_unique_hash(self, applog_context):
        """Test hashing observation with undefined format raises exception"""
        observation1 = Observation.from_inbound(applog_context, ["line1"])
        observation2 = Observation.from_inbound(applog_context, ["line1", "line2"])

        assert observation1.context.id != observation2.context.id

        observation2 = Observation.from_inbound(applog_context, ["line1"])
        assert observation1.context.id == observation2.context.id

    def test_hash_with_undefined_format(self, time_interval, job_resources, timezone):
        """Test hashing observation with undefined format raises exception"""
        context = InboundContext(
            time_interval, job_resources, ObservationSource.APPLOG, ObservationFormat.UNDEFINED, timezone
        )
        with pytest.raises(UnknownObservationSourceError):
            _ = Observation.from_inbound(context, ["line1"])

    def test_make_source_unsupported_formats(self, time_interval, job_resources, timezone):
        """Test make_source with unsupported formats"""
        formats = [
            ObservationFormat.CSV_METRIC_RESPONSE,
        ]
        for format in formats:
            context = InboundContext(time_interval, job_resources, ObservationSource.APPLOG, format, timezone)
            observation = Observation.from_inbound(context, ["line1"])

            with pytest.raises(UnhandledObservationFormatError):
                EntrySourceBase.from_observation(observation)

    def test_make_source_unknown_formats(self, time_interval, job_resources, timezone):
        """Test make_source with unknown formats"""
        formats: list[str] = [
            "INVALID_FORMAT",
        ]

        for format in formats:
            typed_format: ObservationFormat = format  # type: ignore
            context = InboundContext(time_interval, job_resources, ObservationSource.APPLOG, typed_format, timezone)
            observation = Observation.from_inbound(context, ["line1"])

            with pytest.raises(UnknownObservationFormatError):
                EntrySourceBase.from_observation(observation)


class TestRawLineEntry:
    def test_line_processing(self, processing_context, job_params):
        line = RawLineEntry(processing_context, 1, "Test log message from node1")

        assert line.time() is None
        assert line.body() == "Test log message from node1"

        # Test attribution source creation
        source = line.attribution_source_base(job_params)
        assert source.source_context.id == Id(12345)
        assert source.line_num == 1
        assert source.datum == "Test log message from node1"
        assert source.time is None

    def test_node_extraction(self, processing_context, job_params):
        """Test node extraction from line"""
        line = RawLineEntry(processing_context, 1, "[node1] Test message")
        assert line.node(job_params) == "node1"


class TestRawLineEntrySource:
    def test_iteration(self, applog_context, sample_lines):
        """Test iteration behavior of RawLineEntrySource"""
        observation = Observation(applog_context, sample_lines)

        source = RawLineEntrySource(observation)

        # Test iterator protocol
        iterator = iter(source)
        assert iterator is source

        # Test line counting
        first_line = next(source)
        assert source.count == 2
        assert first_line._pos == 1

        second_line = next(source)
        assert source.count == 3
        assert second_line._pos == 2


class TestLokiDictEntry:
    def test_node_extraction_from_ip(self, loki_processing_context, job_params):
        """Test node extraction when node is provided as IP address"""
        entry = LokiDictEntry(
            context=loki_processing_context, pos=1, timestamp_s=1742808956.553062, body="Test message", node="node2"
        )

        assert entry.node(job_params) == "node2"

    def test_node_extraction_from_body(self, loki_processing_context, job_params):
        """Test node extraction from message body when node field is not present"""
        entry = LokiDictEntry(
            context=loki_processing_context, pos=1, timestamp_s=1742808956.553062, body="[node1] Test message"
        )

        assert entry.node(job_params) == "node1"

    def test_timestamp_handling(self, processing_context, timezone):
        """Test timestamp conversion"""
        timestamp_s = 1742808956.553062
        entry = LokiDictEntry(context=processing_context, pos=1, timestamp_s=timestamp_s, body="Test message")

        expected_timestamp = datetime.fromtimestamp(timestamp_s, tz=timezone)
        assert entry.time() == expected_timestamp
        assert entry.seconds_epoch() == timestamp_s


class TestLokiDictEntrySource:
    def test_iteration(self, syslog_loki_dict_context, job_params, timezone):
        """Test iteration behavior of LokiDictEntrySource"""
        context = Context(**vars(syslog_loki_dict_context), id=Id(12345))

        entry_dict = {
            "timestamp": 1742808956553062000,
            "node": "node1",
            "body": "[rank 1]: LustreError: 11-0: testfs-OST0001-osc-ff1e1315d7554800: operation ost_connect to node 192.0.2.10@tcp failed: rc = -16",
        }

        observation = Observation(context, [entry_dict])
        source = LokiDictEntrySource(observation)

        # Test iterator protocol
        iterator = iter(source)
        assert iterator is source

        # Test line counting and parsing
        first_line = next(source)
        assert first_line.node(job_params) == "node1"
        assert first_line.time() == datetime.fromtimestamp(1742808956553062000 * 1e-9, tz=timezone)
        assert first_line.body() == entry_dict["body"]
        assert first_line._pos == 1

    def test_applog_factory_methods(self, loki_dict_context, timezone):
        """Test different factory methods for LokiDictEntry"""
        # Test app entry factory
        app_entry = {
            "timestamp": "1742808956553062000",
            "line": json.dumps({"body": "Test message"}),
        }
        app_result = LokiDictEntry.from_LokiDictAppEntry(Context(**vars(loki_dict_context), id=Id(12345)), 1, app_entry)
        assert app_result.seconds_epoch() == 1742808956.553062
        assert app_result.body() == "Test message"

        """Test different factory methods for LokiDictEntry"""
        context = InboundContext(
            loki_dict_context.time_interval,
            loki_dict_context.resources,
            ObservationSource.APPLOG,
            ObservationFormat.RAW_LOKI_STREAMS,
            timezone,
        )

        # Test app entry factory
        app_entry = LokiAppEntry(
            timestamp="1742808956553062000",
            body="Test message",
        )
        app_result = LokiDictEntry.from_LokiRawAppEntry(Context(**vars(context), id=Id(12345)), 1, app_entry)
        assert app_result.seconds_epoch() == 1742808956.553062
        assert app_result.body() == "Test message"

    def test_syslog_factory_methods(self, syslog_loki_dict_context):
        # Test syslog entry factory
        syslog_entry = LokiSyslogEntry(
            timestamp=1742808956553062000,
            node="node1",
            body="Test message",
        )
        syslog_result = LokiDictEntry.from_LokiSyslogEntry(
            Context(**vars(syslog_loki_dict_context), id=Id(12345)), 1, syslog_entry
        )
        assert syslog_result.seconds_epoch() == 1742808956.553062
        assert syslog_result._node == "node1"
        assert syslog_result.body() == "Test message"

    def test_healthcheck_factory_methods(self, time_interval, job_resources, timezone):
        context = InboundContext(
            time_interval,
            job_resources,
            ObservationSource.HEALTH_CHECK,
            ObservationFormat.LOKI_DICT_LIST,
            timezone,
        )

        # Test health check entry factory
        hc_entry = LokiHealthCheckEntry(
            timestamp=1742808956553,  # milliseconds
            node="node1",
            body="Test message",
        )
        hc_result = LokiDictEntry.from_LokiDictHealthCheckEntry(Context(**vars(context), id=Id(12345)), 1, hc_entry)
        assert hc_result.seconds_epoch() == 1742808956.553
        assert hc_result._node == "node1"
        assert hc_result.body() == "Test message"

        context = InboundContext(
            time_interval, job_resources, ObservationSource.HEALTH_CHECK, ObservationFormat.RAW_LOKI_STREAMS, timezone
        )

        # Test health check entry factory
        hc_entry = LokiHealthCheckEntry(
            timestamp=1742808956553000000,  # nanoseconds
            node="node1",
            body="Test message",
        )
        hc_result = LokiDictEntry.from_LokiRawHealthCheckEntry(Context(**vars(context), id=Id(12345)), 1, hc_entry)
        assert hc_result.seconds_epoch() == 1742808956.553
        assert hc_result._node == "node1"
        assert hc_result.body() == "Test message"


class TestMetricEntry:
    """Test MetricEntry and ValidationError for missing fields"""

    def test_valid_metric_data(self, healthcheck_context, sample_metric_data):
        """Test MetricEntry with valid metric data"""
        source = PrometheusMetricsEntrySource(Observation(healthcheck_context, [sample_metric_data]))
        entry = source.make_line_base(0, sample_metric_data)

        assert entry._timestamp_s == 1642608000
        assert entry._node == "node1"
        assert entry._metric_name == "cpu_usage"
        assert entry._metric_value == 75.5
        assert entry._labels == {"instance": "node1:9100"}
        assert entry._component is None  # Should be None when not available
        assert entry._cluster is None  # Should be None when not available

    def test_missing_timestamp_raises_exception(self, healthcheck_context):
        """Test that missing timestamp raises PydanticCustomError"""
        invalid_metric_data = {"metric": "cpu_usage", "node": "node1", "value": "75.5"}

        source = PrometheusMetricsEntrySource(Observation(healthcheck_context, [invalid_metric_data]))

        with pytest.raises(PydanticCustomError) as exc_info:
            source.make_line_base(0, invalid_metric_data)

        assert exc_info.value.type == "missing"
        assert "Field required: timestamp" in str(exc_info.value)

    def test_missing_metric_name_raises_exception(self, healthcheck_context):
        """Test that missing metric name raises PydanticCustomError"""
        invalid_metric_data = {"timestamp": 1642608000, "node": "node1", "value": "75.5"}

        source = PrometheusMetricsEntrySource(Observation(healthcheck_context, [invalid_metric_data]))

        with pytest.raises(PydanticCustomError) as exc_info:
            source.make_line_base(0, invalid_metric_data)

        assert exc_info.value.type == "missing"
        assert "Field required: metric" in str(exc_info.value)

    def test_missing_node_raises_exception(self, healthcheck_context):
        """Test that missing node raises PydanticCustomError"""
        invalid_metric_data = {"timestamp": 1642608000, "metric": "cpu_usage", "value": "75.5"}

        source = PrometheusMetricsEntrySource(Observation(healthcheck_context, [invalid_metric_data]))

        with pytest.raises(PydanticCustomError) as exc_info:
            source.make_line_base(0, invalid_metric_data)

        assert exc_info.value.type == "missing"
        assert "Field required: node" in str(exc_info.value)

    def test_missing_required_field_error_properties(self):
        """Test MissingRequiredFieldError properties directly"""
        test_data = {"existing_field": "value"}
        error = MissingRequiredFieldError("missing_field", test_data)

        assert error.missing_field == "missing_field"
        assert error.data == test_data
        assert "Missing required field 'missing_field'" in str(error)
        assert "{'existing_field': 'value'}" in str(error)

    def test_component_and_cluster_fields(self, healthcheck_context, metric_data_with_fields):
        """Test that component and cluster fields are properly extracted and accessible"""
        source = PrometheusMetricsEntrySource(Observation(healthcheck_context, [metric_data_with_fields]))
        entry = source.make_line_base(0, metric_data_with_fields)

        assert entry.component == "mlx5_0"
        assert entry.cluster == "host-c771bdf0"
        assert entry._component == "mlx5_0"
        assert entry._cluster == "host-c771bdf0"

    def test_component_and_cluster_from_labels(self, healthcheck_context, metric_data_from_labels):
        """Test that component and cluster are extracted from labels when direct fields are missing"""
        source = PrometheusMetricsEntrySource(Observation(healthcheck_context, [metric_data_from_labels]))
        entry = source.make_line_base(0, metric_data_from_labels)

        assert entry.component == "mlx5_1"  # Extracted from labels["device"]
        assert entry.cluster == "host-c771bdf0"  # Extracted from labels["s_cluster"]

    def test_component_and_cluster_precedence(self, healthcheck_context):
        """Test that direct fields take precedence over labels"""
        # Test precedence: direct fields should override labels
        metric_data_precedence = {
            "timestamp": 1642608000,
            "metric": "node_infiniband_link_downed_total",
            "node": "test-node-003",
            "value": "1",
            "component": "direct_component",  # This should take precedence
            "cluster": "direct_cluster",  # This should take precedence
            "labels": {
                "device": "labels_component",  # This should be ignored
                "s_cluster": "labels_cluster",  # This should be ignored
            },
        }

        source = PrometheusMetricsEntrySource(Observation(healthcheck_context, [metric_data_precedence]))
        entry = source.make_line_base(0, metric_data_precedence)

        assert entry.component == "direct_component"  # Direct field takes precedence
        assert entry.cluster == "direct_cluster"  # Direct field takes precedence

    def test_component_and_cluster_none_values(self, healthcheck_context):
        """Test that component and cluster can be None when not available"""
        # Test with no component or cluster information available
        metric_data_no_fields = {
            "timestamp": 1642608000,
            "metric": "node_infiniband_link_downed_total",
            "node": "test-node-003",
            "value": "1",
            "labels": {"net_host_name": "test-node-003"},  # No device or s_cluster
        }

        source = PrometheusMetricsEntrySource(Observation(healthcheck_context, [metric_data_no_fields]))
        entry = source.make_line_base(0, metric_data_no_fields)

        assert entry.component is None  # Should be None when not available
        assert entry.cluster is None  # Should be None when not available
        assert entry._component is None  # Internal field should also be None
        assert entry._cluster is None  # Internal field should also be None

    def test_component_and_cluster_empty_string_preserved(self, healthcheck_context):
        """Test that empty strings are preserved as empty strings, not converted to None"""
        # Test with empty string component and cluster fields
        metric_data_empty_strings = {
            "timestamp": 1642608000,
            "metric": "node_infiniband_link_downed_total",
            "node": "test-node-003",
            "value": "1",
            "component": "",  # Empty string should remain empty string
            "cluster": "",  # Empty string should remain empty string
            "labels": {"device": "label_device", "s_cluster": "label_cluster"},  # These should be ignored
        }

        source = PrometheusMetricsEntrySource(Observation(healthcheck_context, [metric_data_empty_strings]))
        entry = source.make_line_base(0, metric_data_empty_strings)

        assert entry.component == ""  # Empty string should be preserved
        assert entry.cluster == ""  # Empty string should be preserved
        assert entry._component == ""  # Internal field should also be empty string
        assert entry._cluster == ""  # Internal field should also be empty string

    def test_component_and_cluster_fallback_to_labels(self, healthcheck_context):
        """Test that when direct fields are missing (None), we fall back to labels"""
        # Test with no direct component/cluster fields, only in labels
        metric_data_fallback = {
            "timestamp": 1642608000,
            "metric": "node_infiniband_link_downed_total",
            "node": "test-node-003",
            "value": "1",
            # No "component" or "cluster" keys - should fallback to labels
            "labels": {"device": "mlx5_2", "s_cluster": "test-cluster"},
        }

        source = PrometheusMetricsEntrySource(Observation(healthcheck_context, [metric_data_fallback]))
        entry = source.make_line_base(0, metric_data_fallback)

        assert entry.component == "mlx5_2"  # Should get value from labels["device"]
        assert entry.cluster == "test-cluster"  # Should get value from labels["s_cluster"]


class TestTimeseriesEntry:
    """The entry_timeseries format: explicit timestamps, explicit resources, a declared filter window."""

    def test_an_entry_decodes_its_timestamp_declared_resources_and_verbatim_body(
        self, timeseries_context, syslog_job_info, timezone
    ):
        entries = [
            entry(
                "declared",
                ts=1744734437347712782,
                resources=[{"kind": "node", "hostname": JOB_NODE.upper()}, {"kind": "replica", "name": "worker-0"}],
            ),
            entry(
                f"{JOB_NODE}: [E0105 06:59:25.800744155 ProcessGroupNCCL.cpp:1923] undeclared",
                ts=None,
                resources=None,
            ),
        ]

        declared, undeclared = TimeseriesEntrySource(Observation(timeseries_context, entries))

        # ts only promises the entry is no later than it, so the trailing 782ns rounds the microsecond up.
        assert declared.time() == datetime.fromtimestamp(1744734437, timezone) + timedelta(microseconds=347713)
        assert declared.node(syslog_job_info) == JOB_NODE
        # For now, replicas ride along unread, as they do on loki_dict_list.
        assert declared.resources(syslog_job_info) == ResourceSetClass(
            [Node(hostname=JOB_NODE), Replica(name="worker-0")]
        )
        # Without BodyFallbacks the scrapable prefix and dated body stay unread.
        assert undeclared.body().endswith("undeclared")
        assert undeclared.resources(syslog_job_info) is ResourceSetClass.ANY
        assert undeclared.time() is None

    def test_an_absent_null_or_zero_timestamp(self, timeseries_context, timezone):
        entries = [{"body": "no ts key"}, {"body": "explicit null", "ts": None}, {"body": "epoch", "ts": 0}]

        source = TimeseriesEntrySource(Observation(timeseries_context, entries))

        assert [item.time() for item in source] == [None, None, datetime.fromtimestamp(0, timezone)]

    @pytest.mark.parametrize(
        "declared",
        [
            pytest.param(
                [{"kind": "node", "hostname": OTHER_JOB_NODE}, {"kind": "node", "hostname": JOB_NODE}],
                id="two_nodes",
            ),
            pytest.param([], id="empty_array"),
            pytest.param([{"kind": "replica", "name": "worker-0"}], id="replica_without_a_node"),
        ],
    )
    def test_an_entry_naming_anything_but_one_job_node_raises_unknown_resource_error(
        self, timeseries_context, syslog_job_info, declared
    ):
        entry_obj = only_entry(timeseries_context, declared)

        with pytest.raises(UnknownResourceError):
            entry_obj.node(syslog_job_info)

    # Syslog-shaped for the context's source, with a yearless framework timestamp for the body parser.
    UNDECLARED = f"Jan  5 06:59:25 {JOB_NODE} torchrun: [E0105 06:59:25.800744155 ProcessGroupNCCL.cpp:1923] undeclared"
    REPLICA = [{"kind": "replica", "name": "worker-0"}]
    OTHER_NODE = [{"kind": "node", "hostname": OTHER_JOB_NODE}]

    @pytest.mark.parametrize(
        ("fallbacks", "wire", "expected_resources", "expected_time"),
        [
            pytest.param(
                {"parse_node": True},
                entry(UNDECLARED),
                ResourceSetClass([Node(hostname=JOB_NODE)]),
                None,
                id="node",
            ),
            pytest.param(
                {"parse_timestamp": True},
                entry(UNDECLARED),
                ResourceSetClass.ANY,
                datetime(2024, 1, 5, 6, 59, 25, tzinfo=ZoneInfo("America/Los_Angeles")),
                id="timestamp",
            ),
            pytest.param(
                {"parse_node": True},
                entry(UNDECLARED, resources=REPLICA),
                ResourceSetClass([Node(hostname=JOB_NODE), Replica(name="worker-0")]),
                None,
                id="replica_without_a_node",
            ),
            pytest.param(
                {"parse_node": True, "parse_timestamp": True},
                entry(UNDECLARED, ts=10**9, resources=OTHER_NODE),
                ResourceSetClass([Node(hostname=OTHER_JOB_NODE)]),
                datetime.fromtimestamp(1, ZoneInfo("America/Los_Angeles")),
                id="declared_fields_win",
            ),
        ],
    )
    def test_body_fallbacks_decide_what_is_parsed_out_of_the_body(
        self, timeseries_context, syslog_job_info, timezone, fallbacks, wire, expected_resources, expected_time
    ):
        body_fallbacks = BodyFallbacks(**fallbacks, year_fallback=datetime(2024, 1, 6, tzinfo=timezone))

        (made,) = TimeseriesEntrySource(Observation(timeseries_context, [wire]), body_fallbacks)

        assert made.resources(syslog_job_info) == expected_resources
        assert made.time() == expected_time


class TestTimeseriesExtraction:
    def test_the_declared_interval_is_the_only_time_filter(self, extract, syslog_job_info):
        before_job = syslog_job_info.job_start_time - timedelta(hours=2)
        in_declared = "inside the declared window"
        in_job_window = "inside the job window"
        untimed = "no timestamp at all"
        entries = [
            entry(in_declared, ts=ns(before_job + timedelta(minutes=30)), resources=NODE_RESOURCE),
            entry(
                in_job_window,
                ts=ns(syslog_job_info.job_start_time + timedelta(minutes=30)),
                resources=NODE_RESOURCE,
            ),
            entry(untimed, resources=NODE_RESOURCE),
        ]

        def surviving(time_interval):
            return extract(
                syslog_job_info,
                ObservationSource.SYSLOG,
                ObservationFormat.ENTRY_TIMESERIES,
                entries,
                time_interval=time_interval,
            )

        # An untimed entry cannot be outside a window, so it survives every one of them.
        assert surviving(TimeInterval(before_job, before_job + timedelta(hours=1))) == [in_declared, untimed]
        assert surviving(TimeInterval(None, None)) == [in_declared, in_job_window, untimed]
        # A half-null interval still filters on the bound it has, which two nulls would never exercise.
        assert surviving(TimeInterval(syslog_job_info.job_start_time, None)) == [in_job_window, untimed]


# TODO: Legacy tests should be removed once validated in production.
class TestLegacyBehaviorIsPreserved:
    """The registered applog fallbacks resolve exactly what the legacy applog formats did; retire with them."""

    @pytest.fixture
    def applog_job(self, simple_job_factory):
        """Two nodes, two ranks each, ending 2024-01-06 so a December line without a year is 2023."""
        return simple_job_factory(
            nodes=[JOB_NODE, OTHER_JOB_NODE], ranks_per_node=2, start_timestamp=1704000000, end_timestamp=1704560000
        )

    @pytest.mark.parametrize(
        "line",
        [
            "[rank3]: NCCL watchdog timed out",
            "terminate called after throwing an instance of 'c10::DistBackendError'",
            f"{OTHER_JOB_NODE}: a direct node name",
            "[E1231 23:57:09.800744155 ProcessGroupNCCL.cpp:1923] dated the year before the job ended",
            "[E0105 06:59:25.800744155 ProcessGroupNCCL.cpp:1923] dated the year the job ended",
        ],
    )
    def test_applog_x_rawlog_is_preserved(self, applog_job, timezone, line):
        def applog(fmt, lines, **kwargs):
            return observation_base(applog_job, timezone, ObservationSource.APPLOG, fmt, lines, **kwargs)

        # raw_log_lines reads its year off the interval end, so it keeps the job window it always had.
        (legacy,) = applog(ObservationFormat.RAW_LOG_LINES, [line])
        (undeclared,) = applog(
            ObservationFormat.ENTRY_TIMESERIES, [entry(line)], time_interval=TimeInterval(None, None)
        )

        assert undeclared.node(applog_job) == legacy.node(applog_job)
        assert undeclared.time() == legacy.time()

    LOKI_STREAMS = [
        {"stream": {"node_name": JOB_NODE, "pod": "worker-0"}, "values": [["1704000001000000001", "from labels"]]},
        {"stream": {"pod": "worker-1"}, "values": [["1704000003000000000", "[rank3]: pod label, node from body"]]},
        {
            "stream": {},
            "values": [
                ["1704000004000000000", json.dumps({"body": "payload", "attributes": {"hostname": OTHER_JOB_NODE}})],
                ["1704000005000000000", json.dumps({"data": {"message": "cloud event", "hostname": JOB_NODE}})],
                ["1704000006000000000", "not json, kept verbatim"],
            ],
        },
    ]

    def test_applog_x_rawlokistreams_is_preserved(self, applog_job, timezone):
        def applog(fmt, lines):
            return observation_base(
                applog_job, timezone, ObservationSource.APPLOG, fmt, lines, time_interval=TimeInterval(None, None)
            )

        def as_entry(record):
            """One decoded record as an entry: the stream's node and pod labels, ts, body."""
            resources = []
            if record["node"] is not None:
                resources.append({"kind": "node", "hostname": record["node"]})
            if record["replica"] is not None:
                resources.append({"kind": "replica", "name": record["replica"]})
            return entry(record["body"], ts=record["timestamp"], resources=resources or None)

        legacy = list(applog(ObservationFormat.RAW_LOKI_STREAMS, self.LOKI_STREAMS))
        records = raw_loki_result_helper(iter(self.LOKI_STREAMS))
        new = list(applog(ObservationFormat.ENTRY_TIMESERIES, [as_entry(r) for r in records]))

        for made, old in zip(new, legacy, strict=True):
            assert (made.body(), made.resources(applog_job)) == (old.body(), old.resources(applog_job))
            # entry_timeseries rounds ns up to the microsecond; the legacy float path rounded to nearest.
            assert abs(made.time() - old.time()) <= timedelta(microseconds=1)


class TestTimestampExtraction:
    """Tests for timestamp extraction from various log formats."""

    def test_pytorch_format_standard_mmdd(self, timezone):
        """Test PyTorch timestamp extraction with standard 4-digit MMDD format (e.g., Dec 14)."""

        line = "[E1214 06:59:25.800744155 ProcessGroupNCCL.cpp:1923] Error message"
        result = _extract_timestamp_from_pytorch_format(line, timezone)

        assert result is not None
        assert result.month == 12
        assert result.day == 14
        assert result.hour == 6
        assert result.minute == 59
        assert result.second == 25

    def test_pytorch_format_single_digit_month(self, timezone):
        """Test PyTorch timestamp extraction with single-digit month MDD format (e.g., Jan 6)."""

        line = "1703: [rank1703]:[E106 05:19:32.722601449 ProcessGroupNCCL.cpp:697] [Rank 26] Watchdog caught collective operation timeout"
        result = _extract_timestamp_from_pytorch_format(line, timezone)

        assert result is not None
        assert result.month == 1
        assert result.day == 6
        assert result.hour == 5
        assert result.minute == 19
        assert result.second == 32

    def test_pytorch_format_single_digit_month_various(self, timezone):
        """Test PyTorch timestamp extraction with various single-digit month formats."""

        test_cases = [
            # (line, expected_month, expected_day)
            ("[E115 12:00:00.000000000 file.cpp:1] msg", 1, 15),  # Jan 15
            ("[E215 12:00:00.000000000 file.cpp:1] msg", 2, 15),  # Feb 15
            ("[E901 12:00:00.000000000 file.cpp:1] msg", 9, 1),  # Sep 1
            ("[E930 12:00:00.000000000 file.cpp:1] msg", 9, 30),  # Sep 30
        ]

        for line, expected_month, expected_day in test_cases:
            result = _extract_timestamp_from_pytorch_format(line, timezone)
            assert result is not None, f"Failed to parse: {line}"
            assert result.month == expected_month, f"Month mismatch for {line}"
            assert result.day == expected_day, f"Day mismatch for {line}"

    def test_pytorch_format_double_digit_month(self, timezone):
        """Test PyTorch timestamp extraction with double-digit months (Oct, Nov, Dec)."""

        test_cases = [
            # (line, expected_month, expected_day)
            ("[E1001 12:00:00.000000000 file.cpp:1] msg", 10, 1),  # Oct 1
            ("[E1115 12:00:00.000000000 file.cpp:1] msg", 11, 15),  # Nov 15
            ("[E1231 12:00:00.000000000 file.cpp:1] msg", 12, 31),  # Dec 31
        ]

        for line, expected_month, expected_day in test_cases:
            result = _extract_timestamp_from_pytorch_format(line, timezone)
            assert result is not None, f"Failed to parse: {line}"
            assert result.month == expected_month, f"Month mismatch for {line}"
            assert result.day == expected_day, f"Day mismatch for {line}"

    def test_pytorch_format_no_match(self, timezone):
        """Test that non-matching lines return None."""

        lines = [
            "Regular log message without timestamp",
            "[INFO] Some info message",
            "2025-01-06 12:00:00 ISO format only",
        ]

        for line in lines:
            result = _extract_timestamp_from_pytorch_format(line, timezone)
            assert result is None, f"Should not match: {line}"

    def test_extract_timestamp_from_log_line_pytorch_single_digit_month(self, timezone):
        """Test the main extraction function with PyTorch single-digit month format."""

        line = "[E106 05:19:32.722601449 ProcessGroupNCCL.cpp:697] Error"
        result = _extract_timestamp_from_log_line(line, timezone)

        assert result is not None
        assert result.month == 1
        assert result.day == 6
        assert result.hour == 5
        assert result.minute == 19
        assert result.second == 32

    def test_rawlineentry_with_pytorch_single_digit_month(self, processing_context):
        """Test RawLineEntry.time() with PyTorch single-digit month format."""
        line = RawLineEntry(
            processing_context,
            1,
            "1703: [rank1703]:[E106 05:19:32.722601449 ProcessGroupNCCL.cpp:697] Watchdog timeout",
        )

        timestamp = line.time()
        assert timestamp is not None
        assert timestamp.month == 1
        assert timestamp.day == 6
        assert timestamp.hour == 5
        assert timestamp.minute == 19
        assert timestamp.second == 32

    def test_iso_format_standard(self, timezone):
        """Test ISO timestamp extraction with standard format [YYYY-MM-DD HH:MM:SS]."""

        line = "[2025-12-14 01:07:40] Some log message"
        result = _extract_timestamp_from_iso_format(line, timezone)

        assert result is not None
        assert result.year == 2025
        assert result.month == 12
        assert result.day == 14
        assert result.hour == 1
        assert result.minute == 7
        assert result.second == 40

    def test_iso_format_various_dates(self, timezone):
        """Test ISO timestamp extraction with various valid dates."""

        test_cases = [
            # (line, year, month, day, hour, minute, second)
            ("[2024-01-01 00:00:00] New Year", 2024, 1, 1, 0, 0, 0),
            ("[2025-06-15 12:30:45] Midday", 2025, 6, 15, 12, 30, 45),
            ("[2023-12-31 23:59:59] End of year", 2023, 12, 31, 23, 59, 59),
            ("prefix [2025-03-20 08:15:30] suffix", 2025, 3, 20, 8, 15, 30),
        ]

        for line, year, month, day, hour, minute, second in test_cases:
            result = _extract_timestamp_from_iso_format(line, timezone)
            assert result is not None, f"Failed to parse: {line}"
            assert result.year == year, f"Year mismatch for {line}"
            assert result.month == month, f"Month mismatch for {line}"
            assert result.day == day, f"Day mismatch for {line}"
            assert result.hour == hour, f"Hour mismatch for {line}"
            assert result.minute == minute, f"Minute mismatch for {line}"
            assert result.second == second, f"Second mismatch for {line}"

    def test_iso_format_no_match(self, timezone):
        """Test that non-matching lines return None for ISO format."""

        lines = [
            "Regular log message without timestamp",
            "[E1214 06:59:25.800744155 ProcessGroupNCCL.cpp:1923]",  # PyTorch format, not ISO
            "2025-12-14 01:07:40",  # ISO date but without brackets
            "[2025/12/14 01:07:40]",  # Wrong separator (slashes instead of dashes)
            "[2025-12-14T01:07:40]",  # T separator instead of space
        ]

        for line in lines:
            result = _extract_timestamp_from_iso_format(line, timezone)
            assert result is None, f"Should not match ISO format: {line}"

    def test_extract_timestamp_from_log_line_iso_format(self, timezone):
        """Test the main extraction function with ISO format."""

        line = "[2025-01-06 14:30:00] NCCL error occurred"
        result = _extract_timestamp_from_log_line(line, timezone)

        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 6
        assert result.hour == 14
        assert result.minute == 30
        assert result.second == 0


class TestExtractBodyAndNode:
    """Tests for extract_body_and_node covering all three envelope formats."""

    def test_raw_envelope_body_and_attributes(self):
        """Raw log envelope format: top-level 'body' + attributes.hostname."""
        parsed = {"body": "some syslog line", "attributes": {"hostname": "node-01"}}
        body, node = extract_body_and_node(parsed, None, '{"body":"..."}')
        assert body == "some syslog line"
        assert node == "node-01"

    def test_lepton_message(self):
        """Lepton format: top-level 'message', no attributes."""
        parsed = {"message": "lepton log line"}
        body, node = extract_body_and_node(parsed, None, '{"message":"..."}')
        assert body == "lepton log line"
        assert node is None

    def test_cloudevents_fm_log(self):
        """CloudEvents envelope: data.message + data.hostname (FM logs via csp-telemetry)."""
        raw = '{"data":{"hostname":"nvswitch-01","message":"nmxc-fm: Lost connection to GPU gpuGUID 0xabc"}}'
        parsed = json.loads(raw)
        body, node = extract_body_and_node(parsed, None, raw)
        assert "Lost connection to GPU" in body
        assert "gpuGUID 0xabc" in body
        assert node == "nvswitch-01"

    def test_cloudevents_does_not_overwrite_existing_node(self):
        """When node is already known, data.hostname should not overwrite it."""
        parsed = {"data": {"hostname": "nvswitch-01", "message": "fm log"}}
        body, node = extract_body_and_node(parsed, "already-set", "{}")
        assert body == "fm log"
        assert node == "already-set"

    def test_attributes_does_not_overwrite_existing_node(self):
        """When node is already known, attributes.hostname should not overwrite it."""
        parsed = {"body": "log line", "attributes": {"hostname": "attr-node"}}
        body, node = extract_body_and_node(parsed, "stream-node", "{}")
        assert body == "log line"
        assert node == "stream-node"

    def test_unknown_format_falls_back_to_raw_payload(self):
        """Unknown JSON structure falls back to the raw payload string."""
        raw = '{"specversion":"1.0","type":"event","data":{"severity":"INFO"}}'
        parsed = json.loads(raw)
        body, node = extract_body_and_node(parsed, None, raw)
        assert body == raw
        assert node is None

    def test_data_not_dict_falls_back_to_raw(self):
        """data field is a string, not a dict - skip CloudEvents path, fall back to raw."""
        raw = '{"data":"just a string"}'
        parsed = json.loads(raw)
        body, node = extract_body_and_node(parsed, None, raw)
        assert body == raw

    def test_raw_loki_unpack_cloudevents(self):
        """Integration: raw_loki_unpack correctly unpacks CloudEvents FM log."""
        raw = json.dumps(
            {
                "data": {
                    "hostname": "nvswitch-02",
                    "message": "nmxc-fm: Lost connection to GPU gpuGUID 0xdef",
                }
            }
        )
        result = raw_loki_unpack(1000000000, raw, None, None)
        assert result is not None
        assert "Lost connection to GPU" in result["body"]
        assert result["node"] == "nvswitch-02"

    def test_raw_loki_unpack_preserves_existing_node(self):
        """raw_loki_unpack should not clobber node from stream labels."""
        raw = json.dumps({"body": "some log", "attributes": {"hostname": "attr-node"}})
        result = raw_loki_unpack(1000000000, raw, "stream-node", None)
        assert result is not None
        assert result["node"] == "stream-node"
