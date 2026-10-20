# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from attribution_lib.exceptions import (
    InvalidTimeIntervalError,
    UnknownObservationFormatError,
    UnknownObservationSourceError,
    UnknownResourceError,
)
from attribution_lib.observation import (
    Observation,
)
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Observation as InboundObservation,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.observation.processing import (
    BodyFallbacks,
    EntrySourceBase,
    TimeseriesEntrySource,
    _extract_timestamp_from_iso_format,
    _extract_timestamp_from_log_line,
    _extract_timestamp_from_pytorch_format,
    render_metric_body,
)
from attribution_lib.resources import Node, Replica
from attribution_lib.resources.resource_set import ResourceSetClass
from attribution_lib.rules_engine import process_observation_source
from test_support.helpers import observation_base

JOB_NODE = "fake-node-a"
"""A node that IS part of the test job."""
OTHER_JOB_NODE = "fake-node-b"
"""Another node that IS part of the test job."""
OUT_OF_JOB_NODE = "node-not-in-test-job"
"""A node that is NOT part of the test job."""
NODE_RESOURCE = [{"kind": "node", "hostname": JOB_NODE}]
"""A resource that declares the test job node."""
LINE_NAMING_JOB_NODE = (
    f"Jan  5 06:59:25 {JOB_NODE} torchrun: [E0105 06:59:25.800744155 ProcessGroupNCCL.cpp:1923] undeclared"
)
"""A syslog-shaped line that names the test job node in its body."""


@pytest.fixture
def applog_context(time_interval, timezone):
    """Standard APPLOG context fixture."""
    return InboundContext(time_interval, ObservationSource.APPLOG, ObservationFormat.ENTRY_TIMESERIES, timezone)


@pytest.fixture
def timeseries_context(timezone):
    """The format declares its own window, so these tests pin it open and let each entry carry its own ts."""
    return InboundContext(
        TimeInterval(None, None),
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

    def _extract(job_info, source, lines, *, time_interval=None):
        engine = Mock()
        process_observation_source(
            event_stream=None,
            observation_source=observation_base(job_info, timezone, source, lines, time_interval=time_interval),
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

    def test_start_after_end_rejected(self, time_interval):
        with pytest.raises(InvalidTimeIntervalError):
            TimeInterval(time_interval.end, time_interval.start)

    def test_equal_or_null_bounds_accepted(self, time_interval):
        TimeInterval(time_interval.start, time_interval.start)
        TimeInterval(None, time_interval.end)
        TimeInterval(time_interval.start, None)


class TestInboundContext:
    def test_equality(self, applog_context):
        context1 = applog_context
        context2 = InboundContext(
            applog_context.time_interval,
            applog_context.source,
            applog_context.format,
            applog_context.timezone,
        )
        assert context1 == context2

        different_context = InboundContext(
            applog_context.time_interval,
            ObservationSource.SYSLOG,  # Different source
            applog_context.format,
            applog_context.timezone,
        )
        assert context1 != different_context

    def test_hash(self, applog_context):
        context1 = applog_context
        context2 = InboundContext(
            applog_context.time_interval,
            applog_context.source,
            applog_context.format,
            applog_context.timezone,
        )
        assert hash(context1) == hash(context2)


class TestObservation:
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

    def test_hash_with_undefined_format(self, time_interval, timezone):
        """Test hashing observation with undefined format raises exception"""
        context = InboundContext(time_interval, ObservationSource.APPLOG, ObservationFormat.UNDEFINED, timezone)
        with pytest.raises(UnknownObservationSourceError):
            _ = Observation.from_inbound(context, ["line1"])

    def test_make_source_unknown_formats(self, time_interval, timezone):
        """Test make_source with unknown formats"""
        formats: list[str] = [
            "INVALID_FORMAT",
        ]

        for format in formats:
            typed_format: ObservationFormat = format  # type: ignore
            context = InboundContext(time_interval, ObservationSource.APPLOG, typed_format, timezone)
            observation = Observation.from_inbound(context, ["line1"])

            with pytest.raises(UnknownObservationFormatError):
                EntrySourceBase.from_observation(observation)


class TestRenderMetricBody:
    @pytest.mark.parametrize(
        ("metric", "expected_body"),
        [
            pytest.param(
                {
                    "timestamp": 1744734450,
                    "value": 0,
                    "metric": "up",
                    "labels": {},
                    "node": "n1",
                    "component": "",
                    "cluster": "",
                },
                "time: 1744734450, metric_name: up, metric_value: 0, threshold: None, cluster: , node: n1, component: ",
                id="int_value_and_empty_strings_kept",
            ),
            pytest.param(
                {
                    "timestamp": 1744734450,
                    "value": "60",
                    "metric": "ib_roce_adp_retrans",
                    "labels": {"device": "mlx5_0"},
                    "node": "n1",
                },
                "time: 1744734450, metric_name: ib_roce_adp_retrans, metric_value: 60, threshold: None, cluster: None, node: n1, component: mlx5_0",
                id="str_int_value_component_from_labels",
            ),
            pytest.param(
                {
                    "timestamp": 1744734450,
                    "value": "60.5",
                    "metric": "ib_roce_adp_retrans",
                    "labels": {"s_cluster": "c1"},
                    "node": "n1",
                    "component": "mlx5_1",
                },
                "time: 1744734450, metric_name: ib_roce_adp_retrans, metric_value: 60.5, threshold: None, cluster: c1, node: n1, component: mlx5_1",
                id="str_float_value_cluster_from_labels",
            ),
            pytest.param(
                {
                    "timestamp": 1744734450,
                    "value": "1",
                    "metric": "node_infiniband_link_downed_total",
                    "labels": {"device": "mlx5_9", "s_cluster": "from-labels"},
                    "node": "n1",
                    "component": "mlx5_0",
                    "cluster": "direct",
                },
                "time: 1744734450, metric_name: node_infiniband_link_downed_total, metric_value: 1, threshold: None, cluster: direct, node: n1, component: mlx5_0",
                id="direct_fields_win_over_labels",
            ),
            pytest.param(
                {
                    "timestamp": 1744734450,
                    "value": "1",
                    "metric": "node_infiniband_link_downed_total",
                    "labels": {"device": "mlx5_2", "s_cluster": "c2"},
                    "node": "n1",
                    "component": None,
                    "cluster": None,
                },
                "time: 1744734450, metric_name: node_infiniband_link_downed_total, metric_value: 1, threshold: None, cluster: c2, node: n1, component: mlx5_2",
                id="explicit_none_falls_back_to_labels",
            ),
            pytest.param(
                {"timestamp": 1744734450, "value": 6253.417646976346, "metric": "ib_roce_adp_retrans", "node": "n1"},
                "time: 1744734450, metric_name: ib_roce_adp_retrans, metric_value: 6253.417646976346, threshold: None, cluster: None, node: n1, component: None",
                id="float_value_no_labels",
            ),
        ],
    )
    def test_render_metric_body_renders_the_line_the_metric_rules_match(self, metric, expected_body):
        assert render_metric_body(metric) == expected_body


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

    REPLICA = [{"kind": "replica", "name": "worker-0"}]
    OTHER_NODE = [{"kind": "node", "hostname": OTHER_JOB_NODE}]

    @pytest.mark.parametrize(
        ("fallbacks", "wire", "expected_resources", "expected_time"),
        [
            pytest.param(
                {"parse_node": True},
                entry(LINE_NAMING_JOB_NODE),
                ResourceSetClass([Node(hostname=JOB_NODE)]),
                None,
                id="node",
            ),
            pytest.param(
                {"parse_timestamp": True},
                entry(LINE_NAMING_JOB_NODE),
                ResourceSetClass.ANY,
                datetime(2024, 1, 5, 6, 59, 25, tzinfo=ZoneInfo("America/Los_Angeles")),
                id="timestamp",
            ),
            pytest.param(
                {"parse_node": True},
                entry(LINE_NAMING_JOB_NODE, resources=REPLICA),
                ResourceSetClass([Node(hostname=JOB_NODE), Replica(name="worker-0")]),
                None,
                id="replica_without_a_node",
            ),
            pytest.param(
                {"parse_node": True},
                entry("Jan  5 06:59:25 kernel: no hostname field on this line"),
                ResourceSetClass.ANY,
                None,
                id="unparseable_body_without_require_node",
            ),
            pytest.param(
                {"parse_node": True, "parse_timestamp": True},
                entry(LINE_NAMING_JOB_NODE, ts=10**9, resources=OTHER_NODE),
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

    # The two line shapes the fabric log node parser understands, plus one it does not.
    UFM_LINK_DOWN = (
        "2025-06-18 17:47:26.839 [13406653] [329] WARNING Site [FAB1] [Fabric_Topology] Link went down: "
        f"(Computer:{JOB_NODE} (0200000000000002)) 0200000000000001:1 - (Switch:SW1.1-DH1:7) 0200000000000003:7"
    )
    UFM_NODE_DOWN = f"Site configuration changes: 1234567890 ({OTHER_JOB_NODE}) node is Down"
    UFM_UNPARSED = "2025-06-18 17:47:26.839 [13406653] [329] INFO Site [FAB1] something else"

    def test_a_fabric_log_entry_declaring_no_resources_resolves_its_node_from_the_line(self, syslog_job_info, timezone):
        """The producer's node is the fabric manager host, so the job node has to come out of the line."""
        lines = [(self.UFM_LINK_DOWN, 1750268847276617503), (self.UFM_NODE_DOWN, 1750268879256741095)]
        new = observation_base(
            syslog_job_info,
            timezone,
            ObservationSource.UFMLOG,
            [entry(body, ts=ts) for body, ts in lines] + [entry(self.UFM_UNPARSED, ts=lines[0][1])],
        )

        link_down, node_down, unparsed = new
        assert [link_down.node(syslog_job_info), node_down.node(syslog_job_info)] == [JOB_NODE, OTHER_JOB_NODE]
        with pytest.raises(UnknownResourceError):
            unparsed.node(syslog_job_info)


METRIC_BODY = "time: 1718452800, metric_name: up, metric_value: 0, threshold: None, cluster: c1, node: x, component: "
TIMESERIES_SOURCES = [
    ObservationSource.SYSLOG,
    ObservationSource.NODE_METRICS,
    ObservationSource.NVLINK_METRICS,
    ObservationSource.HEALTH_CHECK,
    ObservationSource.CSP_EVENT,
]
OPEN_WINDOW = TimeInterval(None, None)


@pytest.mark.parametrize("source", TIMESERIES_SOURCES)
class TestTimeWindowFilter:
    """The declared interval is the only time filter, and it reads the same for every source."""

    def test_a_timed_entry_is_kept_only_inside_the_declared_window(self, extract, syslog_job_info, source):
        before_job = syslog_job_info.job_start_time - timedelta(hours=2)
        in_declared = "inside the declared window"
        in_job_window = "inside the job window"
        entries = [
            entry(in_declared, ts=ns(before_job + timedelta(minutes=30)), resources=NODE_RESOURCE),
            entry(
                in_job_window, ts=ns(syslog_job_info.job_start_time + timedelta(minutes=30)), resources=NODE_RESOURCE
            ),
        ]

        def surviving(time_interval):
            return extract(syslog_job_info, source, entries, time_interval=time_interval)

        assert surviving(TimeInterval(before_job, before_job + timedelta(hours=1))) == [in_declared]
        assert surviving(OPEN_WINDOW) == [in_declared, in_job_window]
        # A half-null interval still filters on the bound it has, which two nulls would never exercise.
        assert surviving(TimeInterval(syslog_job_info.job_start_time, None)) == [in_job_window]


class TestSyslogFallbacks:
    """What a syslog line needs to carry, or have parsed out of it, to be kept."""

    @pytest.mark.parametrize(
        ("resources", "body", "kept"),
        [
            pytest.param(NODE_RESOURCE, "kernel: declared job node", True, id="declares_a_job_node"),
            pytest.param(
                [{"kind": "node", "hostname": OUT_OF_JOB_NODE}],
                "kernel: declared host outside the job",
                False,
                id="declares_a_host_outside_the_job",
            ),
            pytest.param(
                [{"kind": "node", "hostname": JOB_NODE}, {"kind": "node", "hostname": OTHER_JOB_NODE}],
                "kernel: two declared nodes",
                False,
                id="declares_two_nodes",
            ),
            pytest.param("AllJobResources", LINE_NAMING_JOB_NODE, False, id="declares_all_job_resources"),
            pytest.param(None, LINE_NAMING_JOB_NODE, True, id="line_names_a_job_node"),
            pytest.param(None, "Jan  5 06:59:25 kernel: no hostname field on this line", False, id="no_node_anywhere"),
        ],
    )
    def test_resources(self, extract, syslog_job_info, resources, body, kept):
        """A declared job node wins; failing that the line's hostname field; failing that the line is dropped."""
        ts = ns(syslog_job_info.job_start_time + timedelta(minutes=30))

        surviving = extract(
            syslog_job_info,
            ObservationSource.SYSLOG,
            [entry(body, ts=ts, resources=resources)],
            time_interval=OPEN_WINDOW,
        )

        assert surviving == ([body] if kept else [])

    @pytest.mark.parametrize(
        ("ts", "kept"),
        [
            pytest.param(1704466765800744155, True, id="ts_present"),
            pytest.param(0, True, id="ts_zero_is_the_epoch"),
            pytest.param(None, False, id="ts_null"),
        ],
    )
    def test_timestamp(self, extract, syslog_job_info, ts, kept):
        """The source requires a timestamp, so an untimed line is dropped even by a window that excludes nothing."""
        body = "kernel: a line"

        surviving = extract(
            syslog_job_info,
            ObservationSource.SYSLOG,
            [entry(body, ts=ts, resources=NODE_RESOURCE)],
            time_interval=OPEN_WINDOW,
        )

        assert surviving == ([body] if kept else [])


@pytest.mark.parametrize("source", [ObservationSource.NODE_METRICS, ObservationSource.NVLINK_METRICS])
class TestMetricFallbacks:
    """What a metric sample needs to declare to be kept; nothing is parsed out of the body."""

    @pytest.mark.parametrize(
        ("resources", "kept"),
        [
            pytest.param(NODE_RESOURCE, True, id="declares_a_job_node"),
            pytest.param([{"kind": "node", "hostname": OUT_OF_JOB_NODE}], False, id="declares_a_host_outside_the_job"),
            pytest.param(None, False, id="declares_no_node"),
        ],
    )
    def test_resources(self, extract, syslog_job_info, source, resources, kept):
        """A sample naming no job node cannot be charged to the job, so it is dropped rather than resolved to ANY."""
        surviving = extract(
            syslog_job_info,
            source,
            [entry(METRIC_BODY, ts=1718452800_000_000_000, resources=resources)],
            time_interval=OPEN_WINDOW,
        )

        assert surviving == ([METRIC_BODY] if kept else [])

    @pytest.mark.parametrize(
        ("ts", "kept"),
        [
            pytest.param(1718452800_000_000_000, True, id="ts_present"),
            pytest.param(None, True, id="ts_null_is_kept_untimed"),
        ],
    )
    def test_timestamp(self, extract, syslog_job_info, source, ts, kept):
        surviving = extract(
            syslog_job_info,
            source,
            [entry(METRIC_BODY, ts=ts, resources=NODE_RESOURCE)],
            time_interval=OPEN_WINDOW,
        )

        assert surviving == ([METRIC_BODY] if kept else [])


class TestHealthCheckFallbacks:
    """What a health check record needs to carry, or have parsed out of it, to be kept."""

    BODY = "Not responding"

    @pytest.mark.parametrize(
        ("resources", "body", "kept"),
        [
            pytest.param(NODE_RESOURCE, BODY, True, id="declares_a_job_node"),
            pytest.param(
                [{"kind": "node", "hostname": OUT_OF_JOB_NODE}], BODY, False, id="declares_a_host_outside_the_job"
            ),
            pytest.param(None, LINE_NAMING_JOB_NODE, True, id="line_names_a_job_node"),
            pytest.param(None, BODY, False, id="no_node_anywhere"),
        ],
    )
    def test_resources(self, extract, syslog_job_info, resources, body, kept):
        """A declared job node wins; failing that a syslog hostname field in the body; failing that it is dropped."""
        surviving = extract(
            syslog_job_info,
            ObservationSource.HEALTH_CHECK,
            [entry(body, ts=1718452800_000_000_000, resources=resources)],
            time_interval=OPEN_WINDOW,
        )

        assert surviving == ([body] if kept else [])

    @pytest.mark.parametrize(
        ("ts", "kept"),
        [
            pytest.param(1718452800_000_000_000, True, id="ts_present"),
            pytest.param(None, False, id="ts_null"),
        ],
    )
    def test_timestamp(self, extract, syslog_job_info, ts, kept):
        """Every legacy record carried a timestamp, so an untimed one is dropped rather than kept untimed."""
        surviving = extract(
            syslog_job_info,
            ObservationSource.HEALTH_CHECK,
            [entry(self.BODY, ts=ts, resources=NODE_RESOURCE)],
            time_interval=OPEN_WINDOW,
        )

        assert surviving == ([self.BODY] if kept else [])


class TestCspEventFallbacks:
    """A provider event names its host and moment outside the body, so both must be declared."""

    BODY = "csp_event: category=HOST_FAILURE, reason=BAD_GPU, status="

    @pytest.mark.parametrize(
        ("resources", "kept"),
        [
            pytest.param(NODE_RESOURCE, True, id="declares_a_job_node"),
            pytest.param([{"kind": "node", "hostname": OUT_OF_JOB_NODE}], False, id="declares_a_host_outside_the_job"),
            pytest.param(None, False, id="declares_no_node"),
        ],
    )
    def test_resources(self, extract, syslog_job_info, resources, kept):
        """Only a declared job node keeps the entry; the body is never parsed for one."""
        surviving = extract(
            syslog_job_info,
            ObservationSource.CSP_EVENT,
            [entry(self.BODY, ts=1718452800_000_000_000, resources=resources)],
            time_interval=OPEN_WINDOW,
        )

        assert surviving == ([self.BODY] if kept else [])

    @pytest.mark.parametrize(
        ("ts", "kept"),
        [
            pytest.param(1718452800_000_000_000, True, id="ts_present"),
            pytest.param(None, False, id="ts_null"),
        ],
    )
    def test_timestamp(self, extract, syslog_job_info, ts, kept):
        """Every legacy event carried a timestamp, so an untimed one is dropped rather than kept untimed."""
        surviving = extract(
            syslog_job_info,
            ObservationSource.CSP_EVENT,
            [entry(self.BODY, ts=ts, resources=NODE_RESOURCE)],
            time_interval=OPEN_WINDOW,
        )

        assert surviving == ([self.BODY] if kept else [])


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
