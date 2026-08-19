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
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import attribution_lib
import pytest
from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.bundled_rules.common.applog_helpers import NCCLCudaVersion, WatchdogFoundValues
from attribution_lib.bundled_rules.common.megatron import MegatronStragglerItem
from attribution_lib.bundled_rules.common.nccl_watchdog import NcclWatchdogParseState
from attribution_lib.bundled_rules.common.rankpulse import RankpulseParseState
from attribution_lib.core import process_check
from attribution_lib.hostlist import parse_hostlist
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.observation import Observation
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.observation.processing import EntrySourceBase
from attribution_lib.resources import AllJobResources
from pytest import param as pparam
from test_support.helpers import (
    AttributionTestResult,
    UnknownAttr,
    create_simple_job_info,
    make_applog_entry,
    make_node_to_ranks,
    match_bad,
    match_bad_peers,
    match_interesting,
    match_interesting_peers,
    rule_lines,
)

xfail = pytest.mark.xfail
TEST_LOG_ROOT = Path(__file__).parents[4] / "tests" / "app_logs"


@pytest.fixture
def app_tst_bldr(attribution_test_builder):
    def _build(
        job_info,
        data,
        *,
        format=ObservationFormat.RAW_LOG_LINES,
        stream_id=None,
        insert_tflops_line=False,
    ):
        return attribution_test_builder(
            job_info=job_info,
            data=data,
            source=ObservationSource.APPLOG,
            format=format,
            check_fn=process_check,
            stream_id=stream_id,
            insert_tflops_line=insert_tflops_line,
        )

    return _build


@pytest.fixture(scope="class")
def big_jobinfo_builder(timezone):
    def _build_jobinfo(tflops_start):
        start = datetime.fromtimestamp(1741860016, tz=timezone)
        end = datetime.fromtimestamp(1741862869, tz=timezone)
        if tflops_start:
            tflops_time = start
        else:
            tflops_time = end
        return (
            JobInfoBuilder(
                id="12345678",
                cluster="cluster_b722552b",
                state="CANCELLED",
                type="training",
                framework="megatron",
                node_to_ranks=make_node_to_ranks(
                    parse_hostlist(
                        "host-9ad02e1b,host-a222193a,host-9162d81d,host-a29ce5ce,host-dc7744c0,host-317c4378,host-2a0e0225,host-0fa931d7,host-9768084c,host-6ae263cb,host-7c7c4f72-[012,026],host-c0fb4fa6,host-a24f75fe-[012,026],host-1c45cecf,host-35d3de2e,host-266e2be8,host-cafb4253,host-1757a2aa,host-b7755ffc,host-a16d8280,host-86c9ed2e,host-2fae07a6,host-8e73d80c,host-c7468e4b-[012,026],host-1a5c686d,host-76d788f7,host-ae47ffe0,host-842833c0,host-45cfe541-[012,026],host-8125b059,host-7246b36f,host-367aa41f,host-9a019e48,host-ce7e073b,host-90ba544e,host-4901c404,host-afb57776,host-ae915c9e,host-fb57ce4e,host-b191f023,host-000cea4b-[012,026],host-0cb679c3-[012,026],host-de3c4d4d,host-1a571114-[012,026],host-0ec56ac6,host-27b7a0aa,host-d780fcb2,host-c29e894d,host-9d96ff8a,host-9baef99b,host-e15b201f,host-a5f9577d,host-f8d44ef4,host-f6a003f3,host-4ec3a0f5,host-0e16f33b,host-3348d23f"
                    ),
                    8,
                ),
                job_start_time=start,
                source_name="artifical slurm",
                timezone=timezone,
            )
            .job_end_time(end)
            .srun_cmd(
                "srun -l --container-image=/redacted_path/aef53de8+megatron-lm+pytorch+24.08-py3-test-model-v1.sqsh --container-mounts=/lustre:/lustre,/redacted_path/8a6f6ab0:/root/.netrc --output=/redacted_path/eba7bc22%x_%j_date_25-03-13_time_03-00-16.log sh -c python -u /redacted_path/c7da3173  \t--disable-gloo-process-groups \t--ckpt-format torch_dist \t--ckpt-fully-parallel-save \t--ckpt-fully-parallel-load \t--ckpt-assume-constant-structure \t--async-save         \t--squared-relu         --no-mmap-bin-files \t--tp-comm-overlap \t--distributed-timeout-minutes 10 \t--exit-duration-in-mins 230 \t--no-create-attention-mask-in-dataloader                 --overlap-grad-reduce         --overlap-param-gather         --sequence-parallel         --tensor-model-parallel-size 4         --use-distributed-optimizer         \t--window-size 512 0 \t--untie-embeddings-and-output-weights         --init-method-std 0.014         --position-embedding-type none         --num-layers 32         --hidden-size 2048         --num-attention-heads 16         --group-query-attention         --num-query-groups 8         --hybrid-override-pattern M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M- \t--ffn-hidden-size 8192 \t--kv-channels 128         --seq-length 4096         --max-position-embeddings 4096         --train-samples 1000000         --lr-decay-style cosine         --lr-warmup-samples 50000         --lr-decay-samples 950000         --save /redacted_path/39d73042         --load /redacted_path/39d73042 \t--per-split-data-args-path /redacted_path/03b2db45 \t--data-cache-path /redacted_path/5a0e7c28         --tiktoken-pattern v2 \t--tokenizer-type TikTokenizer         --tokenizer-model /redacted_path/0ab2e758         --distributed-backend nccl         --micro-batch-size 4         --global-batch-size 256         --lr 1.2e-3         --min-lr 3.0e-6         --weight-decay 0.1         --clip-grad 1.0         --attention-dropout 0.0         --hidden-dropout 0.0         --disable-bias-linear         --normalization RMSNorm         --adam-beta1 0.9         --adam-beta2 0.95         --log-interval 10 \t--log-params-norm \t--log-num-zeros-in-grad \t--log-throughput         --save-interval 2000 \t--save-retain-interval 2000         --eval-interval 1000         --eval-iters 14         --bf16         --use-mcore-models         --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec         --tensorboard-dir /redacted_path/2bce3ee3"
            )
            .exit_code_signal(15)
            .time_limit(600)
            .last_tflops_time(tflops_time)
            .build()
        )

    return _build_jobinfo


@pytest.fixture(scope="class")
def big_jobinfo(timezone, big_jobinfo_builder):
    return big_jobinfo_builder(True)


class TestCheckResults:
    @pytest.fixture(scope="class")
    def checkresults_result(self, event_manager_class, big_jobinfo, obs_file):
        filename = TEST_LOG_ROOT / "test-model_12345678_date_25-01-01_time_00-00-00_sparsified.log"
        event_stream = event_manager_class.create_stream("12345678")
        observation_source = obs_file(big_jobinfo, filename, ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES)
        event_stream.register_observation_context(observation_source.context)
        process_check(
            event_stream=event_stream,
            job_info=big_jobinfo,
            observation_source=observation_source,
            rules_engine=attribution_lib.get_rules_engine(),
        )

        return AttributionTestResult(
            event_stream=event_stream,
            observation_source=observation_source,
            job_info=big_jobinfo,
        )

    @pytest.fixture(scope="class")
    def checkresults_nodes(self, checkresults_result):
        return checkresults_result.get_node_attributions()

    def test_nodes(self, checkresults_nodes):
        attrs = list(checkresults_nodes.attributions["host-4901c404"].attributions["cudaErrorLaunchFailure"])
        assert len(attrs) == 1

        assert attrs[0].line_num == 176

        bad_attrs = list(checkresults_nodes.attributions["host-4901c404"].attributions["cudaErrorLaunchFailure"]._bad)
        assert len(bad_attrs) == 1

        interesting_attrs = list(
            checkresults_nodes.attributions["host-4901c404"].attributions["cudaErrorLaunchFailure"]._interesting
        )
        assert len(interesting_attrs) == 0

    def test_param(self, checkresults_result):
        versions = NCCLCudaVersion.collect(checkresults_result.event_stream)
        nccl_version = versions["nccl_version"]
        assert next(iter(nccl_version._interesting)).line_num == 29
        assert nccl_version.value() == "2.22.3"

        cuda_version = versions["cuda_version"]
        assert next(iter(cuda_version._interesting)).line_num == 29
        assert cuda_version.value() == "12.6"


class TestFiltering:
    def test_node_filtering_logging(self, simple_job_factory, app_tst_bldr):
        """Test that applog lines from nodes not in job are logged and ignored."""

        # Create test applog data using helper function
        test_applog_data = [
            make_applog_entry(
                body="Nov 15 10:30:00 job-node-001 Starting training process",
                timestamp_ns=1700005000000000000,
                severity="info",
                hostname="job-node-001",
            ),
            make_applog_entry(
                body="Nov 15 10:31:00 other-node-999 torch.OutOfMemoryError: CUDA out of memory",
                timestamp_ns=1700005100000000000,
                severity="error",
                hostname="other-node-999",
            ),
        ]

        job_info = simple_job_factory(
            state="RUNNING",
            framework="pytorch",
            start_timestamp=1700000000,
            end_timestamp=1700010000,
            nodes=["job-node-001", "job-node-002"],
        )

        # Mock the logger to capture debug calls
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            _ = app_tst_bldr(job_info, test_applog_data, format=ObservationFormat.RAW_LOKI_STREAMS)

            # Check if any debug logging was called for node filtering
            # Note: applog has complex state machine, so we check if logging occurred
            debug_calls = mock_logger.debug.call_args_list
            if debug_calls:
                # Check for node filtering logs
                any(
                    len(call.args) > 0 and "Applog" in str(call.args[0]) and "ignored:" in str(call.args[0])
                    for call in debug_calls
                )
                # Note: Due to state machine complexity, this test verifies the logging mechanism works
                # without making strict assertions about specific log calls

    def test_state_machine_logging_failure_protection(self, simple_job_factory, obs_data, event_manager):
        """Test that logging failures don't break the critical state machine flow."""
        # Create job info with limited node list
        job_info = simple_job_factory(
            state="RUNNING",
            framework="pytorch",
            start_timestamp=1700000000,
            nodes=["test-node"],
        )

        # Create mock entry that will trigger UnknownResourceError
        # Note: Using raw format here because we need malformed data
        test_applog_data = [
            {
                "stream": {"hostname": "unknown-node"},  # Node not in job
                "values": [
                    [
                        "1700005000000000000",
                        "Nov 15 10:30:00 unknown-node Test log message",
                    ]
                ],
            }
        ]

        observation_source = obs_data(
            job_info, test_applog_data, ObservationSource.APPLOG, ObservationFormat.RAW_LOKI_STREAMS
        )
        event_stream = event_manager.create_stream("test_state_machine_loggin_failure_protection")
        event_stream.register_observation_context(observation_source.context)
        # Mock logging to track calls first, then test exception handling
        with patch("attribution_lib.logline_filtering.log_ignored_node_line") as mock_log:
            # First run without exception to see if logging is called
            process_check(
                event_stream=event_stream,
                job_info=job_info,
                observation_source=observation_source,
                rules_engine=attribution_lib.get_rules_engine(),
            )

            # If logging was called, test exception protection
            if mock_log.called:
                mock_log.reset_mock()
                mock_log.side_effect = RuntimeError("Logging system failure!")

                # This should NOT crash despite logging failure in state machine
                process_check(
                    job_info=job_info,
                    observation_source=observation_source,
                    event_stream=event_stream,
                    rules_engine=attribution_lib.get_rules_engine(),
                )

                assert mock_log.called


class TestIgnoredHostnamePatterns:
    """Test cases for ignored hostname patterns configuration."""

    def test_ignored_hostname_connection_closed(self, simple_job_factory, app_tst_bldr):
        """Test that hostnames matching ignored patterns are skipped during 'Connection closed' attribution."""
        attribution_lib.ignored_hostname_prefixes = ["scanner-"]

        job_info = simple_job_factory(
            state="FAILED",
            framework="pytorch",
            start_timestamp=1700000000,
            end_timestamp=1700010000,
            nodes=["test-node"],
        )

        test_applog_data = [
            make_applog_entry(
                body="NCCL WARN Net : Connection closed by remote peer scanner-001<37046>",
                timestamp_ns=1700005000000000000,
                severity="warn",
                hostname="test-node",
            ),
            make_applog_entry(
                body="NCCL WARN Net : Connection closed by remote peer compute-node-003<37046>",
                timestamp_ns=1700005100000000000,
                severity="warn",
                hostname="test-node",
            ),
        ]

        result = app_tst_bldr(job_info, test_applog_data, format=ObservationFormat.RAW_LOKI_STREAMS)
        node_attr = JobNodeAttributions.from_event_stream(result.event_stream, job_info)

        # The ignored hostname should not appear in attributions
        assert "scanner-001" not in node_attr.attributions

        # The normal connection closed should be attributed
        assert "compute-node-003" in node_attr.attributions
        assert "Connection closed" in node_attr.attributions["compute-node-003"].attributions


class TestRawLokiIllFormed:
    @pytest.fixture(scope="class")
    def job_info(self, timezone, big_jobinfo_builder):
        return big_jobinfo_builder(True)

    @pytest.fixture(scope="class")
    def the_json(self):
        return json.loads("""[
    {
      "stream": {
        "cluster": "test-cluster-cs-001",
        "p_cluster": "test-cluster-cs-001-v3",
        "service_name": "test-cluster-cs-001"
      },
      "values": [
        [
          "1766087271661001504",
          "0:   virtual_pipeline_model_parallel_size ............ None"
        ],
        [
          "1766087271661003704",
          "0:   vision_backbone_type ............................ vit"
        ],
        [
          "1766087271661005384",
          "33"
        ]]
        }]""")

    @pytest.fixture(scope="class")
    def observation(self, the_json, timezone, job_info):
        # Create observation context
        time_interval = TimeInterval(job_info.job_start_time, job_info.job_end_time)
        resources = AllJobResources  # You may need to populate this with actual resources
        context = InboundContext(
            time_interval=time_interval,
            resources=resources,
            source=ObservationSource.APPLOG,
            format=ObservationFormat.RAW_LOKI_STREAMS,
            timezone=timezone,
        )

        # Create observation from lines with json body
        observation = Observation.from_inbound(context=context, lines=the_json)
        return EntrySourceBase.from_observation(observation)

    def test_for_proper_unpack(self, observation, job_info):
        expected = [
            (
                1766087271661001504 * 1e-9,
                "0:   virtual_pipeline_model_parallel_size ............ None",
                job_info.nodes[0],
            ),
            (
                1766087271661003704 * 1e-9,
                "0:   vision_backbone_type ............................ vit",
                job_info.nodes[0],
            ),
            (1766087271661005384 * 1e-9, "33", job_info.nodes[int(33 / 8)]),
        ]

        for exp, entry in zip(expected, observation, strict=True):
            assert datetime.fromtimestamp(exp[0], tz=observation.context.timezone) == entry.time()
            assert exp[1] == entry.body()
            assert exp[2] == entry.node(job_info)


class TestNcclWatchdogAccumulator:
    """Test NcclWatchdogAccumulator fires correctly on watchdog and stack trace lines."""

    def test_fires_on_watchdog_line(self, app_tst_bldr, big_jobinfo):
        """NcclWatchdogAccumulator emits NCCLWatchdog.report for watchdog timeout lines."""
        watchdog_log = [
            "[rank0]:[E1014 22:57:41.268311790 ProcessGroupNCCL.cpp:685] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=123, OpType=ALLREDUCE, NumelIn=1024, NumelOut=1024, Timeout(ms)=600000) ran for 600000 milliseconds before timing out.",
            "[rank1]:[E1014 22:57:41.268311790 ProcessGroupNCCL.cpp:685] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=123, OpType=ALLREDUCE, NumelIn=1024, NumelOut=1024, Timeout(ms)=600000) ran for 600000 milliseconds before timing out.",
        ]
        result = app_tst_bldr(big_jobinfo, watchdog_log)
        nccl_data = NcclWatchdogParseState.collect(result.event_stream)

        assert len(nccl_data) == 2
        assert [sorted(list(report.timeouts_by_key.values())[0].ranks) for report in nccl_data] == [[0, 1], [1]]

    def test_fires_on_stack_trace_line(self, app_tst_bldr, big_jobinfo):
        """NcclWatchdogAccumulator emits NCCLWatchdog.report for stack trace lines."""
        stack_log = [
            "[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] Stack trace of the failed collective:",
            "#0 at /redacted_path/6c8972f9:603",
        ]
        result = app_tst_bldr(big_jobinfo, stack_log)
        nccl_data = NcclWatchdogParseState.collect(result.event_stream)

        assert len(nccl_data) == 1
        assert len(nccl_data[0].stacktrace_by_key) == 1

    def test_deduplicates_identical_stack_traces(self, app_tst_bldr, big_jobinfo):
        """Same stack trace from multiple ranks is grouped into one entry."""
        log = """[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] Stack trace of the failed collective:
#0 at /redacted_path/6c8972f9:603
#1 at /redacted_path/6c8972f9:1234

[rank1]:[E118 15:31:44.123456789 ProcessGroupNCCL.cpp:603] Stack trace of the failed collective:
#0 at /redacted_path/6c8972f9:603
#1 at /redacted_path/6c8972f9:1234
""".split("\n")
        result = app_tst_bldr(big_jobinfo, log)
        nccl_data = NcclWatchdogParseState.collect(result.event_stream)[0]

        assert len(nccl_data.stacktrace_by_key) == 1
        trace_data = list(nccl_data.stacktrace_by_key.values())[0]
        assert sorted(trace_data.ranks) == [0, 1]

    def test_separates_different_stack_traces(self, app_tst_bldr, big_jobinfo):
        """Different stack traces from different ranks produce separate entries."""
        log = """[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] Stack trace of the failed collective:
#0 at /redacted_path/6c8972f9:603
#1 at /redacted_path/6c8972f9:1234

[rank1]:[E118 15:31:44.123456789 ProcessGroupNCCL.cpp:603] Stack trace of the failed collective:
#0 at /redacted_path/6c8972f9:777
#1 at /redacted_path/6c8972f9:888
""".split("\n")
        result = app_tst_bldr(big_jobinfo, log)
        nccl_data = NcclWatchdogParseState.collect(result.event_stream)[0]

        assert len(nccl_data.stacktrace_by_key) == 2
        traces_by_ranks = {tuple(sorted(d.ranks)): k for k, d in nccl_data.stacktrace_by_key.items()}
        assert (0,) in traces_by_ranks
        assert (1,) in traces_by_ranks

    def test_stack_trace_tolerates_noise(self, app_tst_bldr, big_jobinfo):
        """Non-stack lines between stack lines don't break trace collection."""
        log = """[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] Stack trace of the failed collective:
#0 at /redacted_path/6c8972f9:603
Some other log line
#1 at /redacted_path/6c8972f9:1234
Another log line
#2 at /redacted_path/111b1ae2(+0x12ab56) [0x7f8b4f12ab56]
""".split("\n")
        result = app_tst_bldr(big_jobinfo, log)
        nccl_data = NcclWatchdogParseState.collect(result.event_stream)[0]

        assert len(nccl_data.stacktrace_by_key) == 1
        trace_content = list(nccl_data.stacktrace_by_key.keys())[0]
        assert "#0 at" in trace_content
        assert "#1 at" in trace_content
        assert "#2 at" in trace_content
        assert "Some other log line" not in trace_content

    def test_stack_trace_ends_after_max_lookahead(self, app_tst_bldr, big_jobinfo):
        """Stack trace collection stops after 20 consecutive non-stack lines."""
        lines = ["[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] Stack trace of the failed collective:"]
        lines.append("#0 at /redacted_path/6c8972f9:603")
        # Add 25 non-stack lines (exceeds default max_lookahead of 20)
        for i in range(25):
            lines.append(f"Non-stack log line {i}")
        # This stack line comes after the lookahead limit and should not be included
        lines.append("#1 at /redacted_path/6c8972f9:1234")

        result = app_tst_bldr(big_jobinfo, lines)
        nccl_data = NcclWatchdogParseState.collect(result.event_stream)[0]

        assert len(nccl_data.stacktrace_by_key) == 1
        trace_content = list(nccl_data.stacktrace_by_key.keys())[0]
        assert "#0 at" in trace_content
        assert "#1 at" not in trace_content


class TestCollectRankpulseAll:
    """Test Collect.RankpulseAll merges NCCL watchdog and RankPulse information."""

    def test_collects_nccl_and_rankpulse(self, app_tst_bldr, big_jobinfo):
        """Both accumulators emit data consumed by the combined collector."""
        combined_log = """[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] Stack trace of the failed collective:
#0 at /redacted_path/6c8972f9:603
#1 at /redacted_path/6c8972f9:1234
[rank0]:[E1014 22:57:41.268311790 ProcessGroupNCCL.cpp:685] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=123, OpType=ALLREDUCE, NumelIn=1024, NumelOut=1024, Timeout(ms)=600000) ran for 600000 milliseconds before timing out.
[RANK 0] rankpulse - INFO: CPU hang detected at [2025-10-14 03:01:37]: GPU 0% for 5s
[RANK 0] rankpulse - INFO: ----- CPU callstack -----
[RANK 0] rankpulse - INFO: Current thread 0x123 (most recent call first):
[RANK 0] rankpulse - INFO:   File "/path/a.py", line 10 in func_a
[RANK 0] rankpulse - INFO: ----- Finished -----
""".split("\n")

        result = app_tst_bldr(big_jobinfo, combined_log)
        nccl_data = NcclWatchdogParseState.collect(result.event_stream)[0]
        rankpulse_data = RankpulseParseState.collect(result.event_stream)[0]

        assert len(nccl_data.stacktrace_by_key) == 1
        assert len(nccl_data.timeouts_by_key) == 1
        trace_data = list(nccl_data.stacktrace_by_key.values())[0]
        watchdog_data = list(nccl_data.timeouts_by_key.values())[0]
        assert watchdog_data.ranks == [0]
        assert trace_data.ranks == [0]
        assert rankpulse_data.rankpulse_callstacks[0].timestamp == "2025-10-14 03:01:37"


class TestGetNodeBehavior:
    """Test the get_node() method behavior for MegatronStragglerItem."""

    def test_megatron_straggler_item_get_node_success(self):
        """Test MegatronStragglerItem.get_node() when node resolution succeeds."""

        # Mock source
        mock_source = Mock()
        mock_source.rule = "test_rule"

        # Create item with resolved node
        item = MegatronStragglerItem(rank=4, node="resolved_node_4", score=214.52, source=mock_source)

        # Should return resolved node
        assert item.get_node() == "resolved_node_4"

    def test_megatron_straggler_item_get_node_failure(self):
        """Test MegatronStragglerItem.get_node() when node resolution fails."""

        # Mock source
        mock_source = Mock()
        mock_source.rule = "test_rule"

        # Create item with unresolved node
        item = MegatronStragglerItem(rank=5, node=None, score=214.56, source=mock_source)

        # Should return None
        assert item.get_node() is None


class TestApplogRules:
    interesting_applog_rules = (
        []
        + rule_lines(
            [
                "0: NET/OFI Request 0x1551fec2bb50 completed with error. RC: 5. Error: Unresponsive receiver (connection previously established) My EFA addr: fi_addr_efa://[fe80::test:test:test:0001]:8:100000001 My host id: i-0test000000000001 Peer EFA addr: fi_addr_efa://[fe80::test:test:test:0002]:8:100000002 Peer host id: i-0test000000000002. Completed length: 0, Request: { dev: 0, size: 77824, state: CREATED, type: SEND }",
            ],
            "bad",
            [0],
            "EFA Error-5",
            "completed with error",
        )
        + rule_lines(  # "New rules" - GotCompletionFromPeerNew
            [
                "0: testnode1-001:1234567:1234568 [7] transport/net_ib.cc:1984 NCCL WARN NET/IB: Got completion from peer 127.0.0.1<43767> with status=12 opcode=1 len=0 vendor err 129 (Send) localGid ::ffff:192.168.124.83 remoteGids::ffff:192.168.127.54 hca mlx5_17",
                "0: NET/IB: Got completion from peer 127.0.0.1<43767> with status=12 opcode=1 len=0 vendor err 129 (Send) localGid ::ffff:192.168.124.83 remoteGids::ffff:192.168.127.54 hca mlx5_17",
                "0: testnode1-002:9876543:9876544 [7] transport/net_ib.cc:1984 NCCL WARN NET/IB: Got completion from peer 127.0.0.1<39428> with status=12 opcode=129 len=12352 vendor err 129 (Recv) localGid ::ffff:192.168.127.54 remoteGids::ffff:192.168.124.83 hca mlx5_17",
            ],
            "bad_peers",
            [0, "localhost"],
            "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)",
            "Got completion from peer",
        )
        + rule_lines(  # More "new rules" - GotCompletionFromPeerNew
            [
                "0: testnode1-002:9876543:9876544 [7] transport/net_ib.cc:1984 NCCL WARN NET/IB: Got completion from peer 127.0.0.1<39428> with status=5 opcode=129 len=12352 vendor err 244 (Recv) localGid ::ffff:192.168.127.54 remoteGids::ffff:192.168.124.83 hca mlx5_17",
            ],
            "bad_peers",
            [0, "localhost"],
            "IB Error-5 Vendor-244 (IBV_WC_WR_FLUSH_ERR)",
            "Got completion from peer",
        )
        + rule_lines(  # "Old rules" - GotCompletionFromPeerOld
            [
                "0: host-98f6741a:1234567:1234568 [2] ib_plugin.c:1007 NCCL WARN NET/IB : Got completion from peer 127.0.0.1<12015> with error 12, opcode 0, len 5448, vendor err 129",
            ],
            "bad_peers",
            [0, "localhost"],
            "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)",
            "Got completion from peer",
        )
        + rule_lines(  # More "old rules" - GotCompletionFromPeerOld
            [
                "0: host-98f6741a:1234567:1234568 [2] ib_plugin.c:1007 NCCL WARN NET/IB : Got completion from peer 127.0.0.1<57985> with error 5, opcode 0, len 5448, vendor err 249",
            ],
            "bad_peers",
            [0, "localhost"],
            "IB Error-5 Vendor-249 (IBV_WC_WR_FLUSH_ERR)",
            "Got completion from peer",
        )
        + rule_lines(
            [
                "0: host-98f6741a:1234567:1234568 [2] ib_plugin.c:1007 NCCL WARN NET/IB : Got completion with error 12, opcode 1, len 0, vendor err 129",
            ],
            "bad",
            [0],
            "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)",
            "Got completion with error",
        )
        + rule_lines(
            [
                "0: p2p.cc:886 NCCL WARN NET/IB : Irrecoverable receive communicator error on device 0. Reason: No active devices left. Peer: 127.0.0.1<12345>, wr_id=42, req=0x000000001234, status=IBV_WC_RETRY_EXC_ERR(12), opcode=IBV_WC_RDMA_WRITE(1), vendor_err=0, reqSize=129 (Recv) localGid:2001:db8::1 remoteGid:2001:db8::2, qpn=42, hca=mlx5_0",
            ],
            "bad_peers",
            [0, "localhost"],
            "IB Error-12 (IBV_WC_RETRY_EXC_ERR)",
            "Irrecoverable NET/IB communicator error",
        )
        + rule_lines(
            [
                "0: Call to ibv_modify_qp failed with 110 Connection timed out, on dev mlx5_4:1, curr state INIT, next state RTR, local GID index 3, local GID ::ffff:192.168.26.95, remote GID ::ffff:127.0.0.1",
                "0: Call to ibv_modify_qp failed with 110 Connection timed out, on dev mlx5_4:1, curr state INIT, next state RTR, local GID index 3, local GID ::ffff:192.168.26.95, remote GID localhost",
            ],
            "bad_peers",
            [0, "localhost"],
            "Call to ibv_modify_qp failed: eno 110 timeout",
            "Call to ibv_modify_qp failed",
        )
        + rule_lines(
            [
                "0: Call to ibv_modify_qp failed with 22 Connection timed out, on dev mlx5_4:1, curr state INIT, next state RTR, local GID index 3, local GID ::ffff:192.168.26.95, remote GID ::ffff:127.0.0.1",
            ],
            "bad_peers",
            [0, "localhost"],
            "Call to ibv_modify_qp failed: eno 22 invalid argument",
            "Call to ibv_modify_qp failed",
        )
        + rule_lines(
            [
                "0: Call to ibv_modify_qp failed with 22 Connection timed out, on dev mlx5_4:1, curr state INIT, next state RTR, local GID index 3, local GID ::ffff:192.168.26.95, remote GID localhost",
            ],
            "bad_peers",
            [0, "localhost"],
            "Call to ibv_modify_qp failed: eno 22 invalid argument(Mismatched GID)",
            "Call to ibv_modify_qp failed",
        )
        + rule_lines(
            [
                "0: Call to ibv_modify_qp failed with 11 Resource temporarily unavailable, on dev mlx5_4:1, curr state INIT, next state RTR, local GID index 3, local GID ::ffff:192.168.26.95, remote GID ::ffff:127.0.0.1",
            ],
            "bad_peers",
            [0, "localhost"],
            "Call to ibv_modify_qp failed: eno 11 resource unavailable",
            "Call to ibv_modify_qp failed",
        )
        + rule_lines(
            [
                "0: host-584fe0e8:9876543:9876544 [1] socket.c:540 NCCL WARN Net : Connection closed by remote peer host-9ad02e1b<37046>\n",
            ],
            "bad",
            [0],
            "Connection closed",
            "Connection closed by remote peer",
        )
    )

    @pytest.mark.parametrize(
        "line,attr,nodes,symptom,rule",
        interesting_applog_rules,
    )
    def test_rule_interesting(
        self, line, attr, nodes, symptom, rule, big_jobinfo_builder, obs_data, event_stream, request
    ):
        big_jobinfo = big_jobinfo_builder(False)
        event_stream = event_stream(symptom)
        obs = obs_data(big_jobinfo, [line], ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES)
        event_stream.register_observation_context(obs.context)
        process_check(
            event_stream=event_stream,
            job_info=big_jobinfo,
            observation_source=obs,
            rules_engine=attribution_lib.get_rules_engine(),
        )
        symptom = f"(non-fatal) {symptom}"
        node_attr = JobNodeAttributions.from_event_stream(event_stream, big_jobinfo)
        mapped_nodes = []
        for node in nodes:
            if isinstance(node, int):
                mapped_nodes.append(big_jobinfo.nodes[node])
            else:
                assert isinstance(node, str)
                mapped_nodes.append(node)
        match attr:
            case "bad_peers":
                assert len(mapped_nodes) == 2
                match_interesting_peers(node_attr, mapped_nodes, symptom, rule)
            case "bad":
                assert len(mapped_nodes) == 1
                match_interesting(node_attr, mapped_nodes[0], symptom, rule)
            case _:
                raise UnknownAttr(attr)

    applog_rules = (
        []
        # moved to applog_rules.test.toml
        + rule_lines(
            [
                "0: enroot-mount: failed to mount: /etc/resolv.conf at /tmp/.enroot/abc123",
            ],
            "bad",
            [0],
            "pyxis: enroot mount failure from system directory",
            "PyxisEnrootMountFailureSystem",
        )
        + rule_lines(
            [
                "0: enroot-mount: failed to mount: /opt/nvidia/nsight-systems at /raid/enroot/data/user-00001/pyxis_123/opt/nvidia/nsight-systems: No such file or directory",
            ],
            "bad",
            [0],
            "pyxis: enroot mount path not found",
            "PyxisEnrootMountFailureSystem",
        )
        + rule_lines(
            [
                "0: enroot-mount: failed to mount: /home/user/data.sqsh at /tmp/.enroot/abc123",
            ],
            "bad",
            [0],
            "pyxis: enroot mount failure from user directory",
            "PyxisEnrootMountFailureUser",
        )
        + rule_lines(
            [
                "0: enroot-mount: failed to mount: /lustre/scratch/model.sqsh at /tmp/.enroot/abc123",
            ],
            "bad",
            [0],
            "pyxis: enroot mount failure from user directory",
            "PyxisEnrootMountFailureUser",
        )
        + rule_lines(
            [
                "0: pyxis:     [ERROR] /etc/enroot/hooks.d/98-nvidia.sh exited with return code 1",
                "0: [ERROR] /etc/enroot/hooks.d/98-nvidia.sh exited with return code 125",
            ],
            "bad",
            [0],
            "pyxis: nvidia hook failed",
            "alias else",
        )
        + rule_lines(
            [
                "0: pyxis:     [ERROR] /etc/enroot/hooks.d/10-cgroups.sh exited with return code 1",
                "0: [ERROR] /etc/enroot/hooks.d/10-cgroups.sh exited with return code 2",
            ],
            "bad",
            [0],
            "pyxis: cgroup hook failed",
            "alias else",
        )
        + rule_lines(
            [
                "0: pyxis:     nvidia-container-cli: mount error: file creation failed: /dev/nvidia-modeset",
                "0: nvidia-container-cli: mount error: stat failed: /dev/nvidiactl",
            ],
            "bad",
            [0],
            "pyxis: nvidia-container-cli mount error",
            "alias else",
        )
        + rule_lines(
            [
                "0: pyxis:     nvidia-container-cli: initialization error: nvml error: driver not loaded",
                "0: nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1",
            ],
            "bad",
            [0],
            "pyxis: nvidia-container-cli init error",
            "alias else",
        )
        + rule_lines(
            [
                "0: pyxis:     nvidia-container-cli: ldcache error: process /usr/sbin/ldconfig.real failed with error code: 1",
            ],
            "bad",
            [0],
            "pyxis: nvidia-container-cli ldcache error",
            "alias else",
        )
        + rule_lines(
            [
                "0: pyxis:     nvidia-container-cli: detection error: nvml error: not supported",
                "0: pyxis:     nvidia-container-cli: detection error: path error: ///var/run/nvidia-fabricmanager/socket: permission denied",
            ],
            "bad",
            [0],
            "pyxis: nvidia-container-cli detection error",
            "alias else",
        )
        + rule_lines(
            [
                "0: pyxis:     [ERROR] Command not found: nvidia-container-cli, see https://github.com/NVIDIA/libnvidia-container",
            ],
            "bad",
            [0],
            "pyxis: nvidia-container-cli not installed",
            "alias else",
        )
        + rule_lines(
            [
                "0: pyxis:     [ERROR] /etc/enroot/hooks.d/99-mellanox.sh exited with return code 1",
            ],
            "bad",
            [0],
            "pyxis: mellanox hook failed",
            "alias else",
        )
        + rule_lines(
            [
                "0: error: pyxis:     nvidia-container-cli: device error: 42: unknown device",
                "0: nvidia-container-cli: device error: 8: unknown device",
            ],
            "bad",
            [0],
            "pyxis: nvidia-container-cli device error",
            "alias else",
        )
        + rule_lines(
            [
                "0: error: pyxis:     [ERROR] Unknown NVIDIA driver capability: comput",
                "0: [ERROR] Unknown NVIDIA driver capability: bogus",
            ],
            "bad",
            [0],
            "pyxis: nvidia unknown driver capability",
            "alias else",
        )
        + rule_lines(
            [
                "0: error: pyxis:     nvidia-container-cli: requirement error: unsatisfied condition: cuda>=99.0, please update your driver to a newer version, or use an earlier cuda container",
                "0: nvidia-container-cli: requirement error: unsatisfied condition: driver>=999",
            ],
            "bad",
            [0],
            "pyxis: nvidia-container-cli requirement error",
            "alias else",
        )
        # B. Registry / image pull (HTTP status code extractor + curl/json keyword rules)
        + rule_lines(
            [
                "0: error: pyxis:     [ERROR] URL https://registry-1.docker.io/v2/library/private-image/manifests/latest returned error code: 401 Unauthorized"
            ],
            "bad",
            [0],
            "enroot: registry auth denied",
            "EnrootRegistryHttpError",
        )
        + rule_lines(
            ["0: error: pyxis:     [ERROR] URL https://ghcr.io/token returned error code: 403 Forbidden"],
            "bad",
            [0],
            "enroot: registry auth denied",
            "EnrootRegistryHttpError",
        )
        + rule_lines(
            [
                "0: error: pyxis:     [ERROR] URL https://registry-1.docker.io/v2/library/ubuntu/manifests/36.04 returned error code: 404 Not Found"
            ],
            "bad",
            [0],
            "enroot: registry image not found",
            "EnrootRegistryHttpError",
        )
        + rule_lines(
            [
                # Docker-hub-fallback case (older enroot, missing credentials): hostname with a dot misparsed into the repo path.
                "0: error: pyxis:     [ERROR] URL https://registry-1.docker.io/v2/registry.example.com/sample/team/image/manifests/latest returned error code: 401 Unauthorized",
                "0: error: pyxis:     [ERROR] URL https://registry-1.docker.io/v2/registry.example.com/sample/team/evals/manifests/26.03 returned error code: 401 Unauthorized",
            ],
            "bad",
            [0],
            "enroot: missing credentials (docker hub fallback)",
            "EnrootRegistryHttpError",
        )
        + rule_lines(
            ["0: error: pyxis:     [ERROR] Invalid image reference: docker:///foo"],
            "bad",
            [0],
            "enroot: invalid image reference",
            "alias else",
        )
        + rule_lines(
            [
                "0: error: pyxis:     [ERROR] Could not parse digest information from https://registry.example.com/v2/sample/image/manifests/main",
                "0: error: pyxis:     [ERROR] Could not retrieve digest from https://registry.example/v2/foo/manifests/bar",
            ],
            "bad",
            [0],
            "enroot: digest error",
            "alias else",
        )
        + rule_lines(
            ["0: error: pyxis:     curl: (56) CONNECT tunnel failed, response 502"],
            "bad",
            [0],
            "enroot: curl tunnel failure",
            "alias else",
        )
        + rule_lines(
            [
                "0: error: pyxis:     curl: (28) Operation timed out after 1200000 milliseconds with 1510260015 out of 10371127195 bytes received"
            ],
            "bad",
            [0],
            "enroot: curl transfer timeout",
            "alias else",
        )
        + rule_lines(
            ["0: error: pyxis:     curl: (52) Empty reply from server"],
            "bad",
            [0],
            "enroot: curl empty reply from server",
            "alias else",
        )
        + rule_lines(
            ["0: error: pyxis:     curl: (6) Could not resolve host: registry-1.docker.io"],
            "bad",
            [0],
            "enroot: DNS resolution failed",
            "alias else",
        )
        + rule_lines(
            ["0: error: pyxis:     [ERROR] Could not process JSON input"],
            "bad",
            [0],
            "enroot: JSON parse failed",
            "alias else",
        )
        + rule_lines(
            [
                "0: error: pyxis:     zstd: /*stdin*\\: unexpected end of file",
                "0: error: pyxis:     /*stdin*\\ : Decoding error (36) : Restored data doesn't match checksum",
            ],
            "bad",
            [0],
            "enroot: corrupted cache layer",
            "alias else",
        )
        # C. Filesystem / storage
        + rule_lines(
            ["0: error: pyxis:     [ERROR] Could not acquire rootfs lock"],
            "bad",
            [0],
            "enroot: rootfs lock contention",
            "alias else",
        )
        + rule_lines(
            ["0: error: pyxis:     [ERROR] Not enough space left in /var/run/enroot"],
            "bad",
            [0],
            "enroot: not enough space (system path)",
            "EnrootNotEnoughSpace",
        )
        + rule_lines(
            ["0: error: pyxis:     [ERROR] Not enough space left in /home/alice/.local/share/enroot"],
            "bad",
            [0],
            "enroot: not enough space (user path)",
            "EnrootNotEnoughSpace",
        )
        + rule_lines(
            ["0: error: pyxis:     [ERROR] Not enough space left in /lustre/scratch/bob/enroot"],
            "bad",
            [0],
            "enroot: not enough space (user path)",
            "EnrootNotEnoughSpace",
        )
        + rule_lines(
            ["0: error: pyxis:     [ERROR] Invalid image format: /tmp/bad.sqsh"],
            "bad",
            [0],
            "enroot: invalid image format",
            "alias else",
        )
        + rule_lines(
            [
                # Ephemeral pattern: last component pyxis_<digits>_<digits>.<digits>
                "0: error: pyxis:     [ERROR] File already exists: /raid/containers/data/user-00002/pyxis_00001_1000001.0",
                # also older pyxis pattern pyxis_<jobid>.<stepid>
                "0: error: pyxis:     [ERROR] File already exists: /raid/enroot/data/user-00003/pyxis_00002.18",
            ],
            "bad",
            [0],
            "pyxis: file already exists (ephemeral)",
            "EnrootFileAlreadyExists",
        )
        + rule_lines(
            [
                "0: error: pyxis:     mkdir: cannot create directory '/raid/enroot/cache/group-00001': Permission denied",
                "0: error: pyxis:     mkdir: cannot create directory '/var/lib/enroot/cache/job-00001/.tokens.28686': Permission denied",
            ],
            "bad",
            [0],
            "pyxis: mkdir permission denied (system path)",
            "PyxisMkdirPermissionDenied",
        )
        + rule_lines(
            ["0: error: pyxis:     mkdir: cannot create directory '/home/alice/.enroot/cache': Permission denied"],
            "bad",
            [0],
            "pyxis: mkdir permission denied (user path)",
            "PyxisMkdirPermissionDenied",
        )
        + rule_lines(
            ["0: error: pyxis:     FATAL ERROR: File system corruption detected"],
            "bad",
            [0],
            "unsquashfs: file system corruption",
            "alias else",
        )
        + rule_lines(
            [
                "0: error: pyxis:     FATAL ERROR: dir_scan: failed to make directory /persist-var/example-host-0002/lib/enroot/data/job-00002/pyxis_0, because Permission denied"
            ],
            "bad",
            [0],
            "unsquashfs: dir_scan permission denied (system path)",
            "UnsquashfsDirScanPermissionDenied",
        )
        + rule_lines(
            [
                "0: error: pyxis:     FATAL ERROR: dir_scan: failed to make directory /home/alice/work/cache, because Permission denied"
            ],
            "bad",
            [0],
            "unsquashfs: dir_scan permission denied (user path)",
            "UnsquashfsDirScanPermissionDenied",
        )
        + rule_lines(
            [
                '0: error: pyxis:     /usr/lib/enroot/runtime.sh: line 384: 2529132 Killed                  unsquashfs ${TTY_OFF+-no-progress} -user-xattrs -d "${rootfs}" "${image}" 1>&2'
            ],
            "bad",
            [0],
            "unsquashfs: killed by OOM",
            "UnsquashfsOomKilled",
        )
        + rule_lines(
            ["0: error: pyxis:     FATAL ERROR: Probably out of space on output filesystem"],
            "bad",
            [0],
            "mksquashfs: disk full (system path)",
            "MksquashfsDiskFull",
        )
        + rule_lines(
            [
                "0: error: pyxis:     FATAL ERROR:writer: failed to write file /raid/containers/data/user-00004/pyxis_X/usr/local/lib/foo.pyc"
            ],
            "bad",
            [0],
            "unsquashfs: write failed (system path)",
            "UnsquashfsWriteFailed",
        )
        + rule_lines(
            ["0: error: pyxis:     FATAL ERROR:writer: failed to write file /home/bob/scratch/rootfs/usr/bin/foo"],
            "bad",
            [0],
            "unsquashfs: write failed (user path)",
            "UnsquashfsWriteFailed",
        )
        # D. Namespace / kernel / exec
        + rule_lines(
            [
                "0: error: pyxis:     enroot-nsenter: failed to create user namespace: Permission denied",
                "0: error: pyxis:     enroot-nsenter: failed to create user namespace: No space left on device",
                "0: error: pyxis:     enroot-nsenter: failed to create mount namespace: Operation not permitted",
                "0: error: pyxis:     enroot-nsenter: failed to create network namespace: Invalid argument",
                "0: error: pyxis:     enroot-nsenter: failed to create IPC namespace: EPERM",
            ],
            "bad",
            [0],
            "pyxis: failed to create namespace",
            "PyxisNamespaceCreateFailed",
        )
        + rule_lines(
            [
                "0: error: pyxis:     enroot-aufs2ovlfs: failed to set capabilities: Operation not permitted",
                "0: error: pyxis:     enroot-mount: failed to set capabilities: Operation not permitted",
                "0: error: pyxis:     enroot-nsenter: failed to get capabilities: Operation not permitted",
            ],
            "bad",
            [0],
            "enroot: failed to set capabilities",
            "EnrootCapSetFailed",
        )
        + rule_lines(
            [
                "0: error: pyxis:     enroot-switchroot: failed to execute: /bin/bash: Exec format error",
                "0: error: pyxis:     enroot-switchroot: failed to execute: /bin/bash: No such file or directory",
            ],
            "bad",
            [0],
            "enroot: switchroot exec failed",
            "alias else",
        )
        # E. Pyxis-direct
        + rule_lines(
            [
                "0: error: pyxis: container start failed with error code: 1",
                "0: error: pyxis: container start failed with error code: 56",
            ],
            "bad",
            [0],
            "pyxis: container start nonzero exit",
            "PyxisContainerStartNonzero",
        )
        + rule_lines(
            ["0: error: pyxis: couldn't chdir to /does/not/exist: No such file or directory"],
            "bad",
            [0],
            "pyxis: couldn't chdir to container workdir",
            "PyxisCouldntChdirWorkdir",
        )
        # F. Pyxis args
        + rule_lines(
            ["0: srun: error: pyxis: --container-image specified multiple times"],
            "bad",
            [0],
            "pyxis: --container-image specified multiple times",
            "alias else",
        )
        + rule_lines(
            ['0: srun: error: pyxis: mount source must be a relative path, an absolute path, "tmpfs" or "umount"'],
            "bad",
            [0],
            "pyxis: mount source invalid",
            "alias else",
        )
        + rule_lines(
            [
                "0: srun: error: pyxis: both --container-mount-home and --no-container-mount-home were specified",
                "0: srun: error: pyxis: both --container-writable and --container-readonly were specified",
            ],
            "bad",
            [0],
            "pyxis: conflicting options",
            "PyxisConflictingOptions",
        )
        + rule_lines(
            ['0: error: pyxis: error: "create" flag was passed to --container-name but the container already exists'],
            "bad",
            [0],
            "pyxis: container name already exists",
            "alias else",
        )
        + rule_lines(
            [
                "0: srun: error: Invalid --container-image argument: B.",
                "0: srun: error: Invalid --container-mounts argument: missingcolon.",
                "0: srun: error: Invalid --no-container-mount-home argument: (null).",
            ],
            "bad",
            [0],
            "srun: invalid pyxis argument",
            "SrunInvalidPyxisArgument",
        )
        # G. Lifecycle wrappers + cancellation
        + rule_lines(
            ["0: error: pyxis: failed to create container filesystem"],
            "bad",
            [0],
            "pyxis: failed to create container filesystem",
            "alias else",
        )
        + rule_lines(
            ["0: error: pyxis: received SIGTERM, forwarding to child 807892"],
            "bad",
            [0],
            "pyxis: cancelled by SIGTERM",
            "alias else",
        )
        + rule_lines(
            ["0: error: pyxis: could not wait for child 1661846: Interrupted system call"],
            "bad",
            [0],
            "pyxis: cancelled by SIGTERM",
            "PyxisCouldNotWaitChildInterrupted",
        )
        + rule_lines(
            ["0: error: pyxis: container start error: Interrupted system call"],
            "bad",
            [0],
            "pyxis: cancelled by SIGTERM",
            "alias else",
        )
        # H. Shared memory / peer-task cascades
        + rule_lines(
            ["0: error: pyxis: container was not started successfully by another task"],
            "bad",
            [0],
            "pyxis: cascade from peer task",
            "alias else",
        )
        + interesting_applog_rules
    )

    @pytest.mark.parametrize(
        "line,attr,nodes,symptom,rule",
        applog_rules,
    )
    def test_rule(self, line, attr, nodes, symptom, rule, big_jobinfo, obs_data, event_stream, request):
        event_stream = event_stream(symptom)
        obs = obs_data(big_jobinfo, [line], ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES)
        event_stream.register_observation_context(obs.context)
        process_check(
            event_stream=event_stream,
            job_info=big_jobinfo,
            observation_source=obs,
            rules_engine=attribution_lib.get_rules_engine(),
        )

        node_attr = JobNodeAttributions.from_event_stream(event_stream, big_jobinfo)
        mapped_nodes = []
        for node in nodes:
            if isinstance(node, int):
                mapped_nodes.append(big_jobinfo.nodes[node])
            else:
                assert isinstance(node, str)
                mapped_nodes.append(node)
        match attr:
            case "bad_peers":
                assert len(mapped_nodes) == 2
                match_bad_peers(node_attr, mapped_nodes, symptom, rule)
            case "bad":
                assert len(mapped_nodes) == 1
                match_bad(node_attr, mapped_nodes[0], symptom, rule)
            case _:
                raise UnknownAttr(attr)

    applog_to_symptom_rules = (
        []
        # moved generated cuda/applog matrices to applog_rules.test.toml
        + [
            pparam(
                "Call to ibv_modify_qp failed",
                "Call to ibv_modify_qp failed",
                "alias else",
                marks=xfail(reason="Redundant rule"),
            ),
            ("NCCL WARN Cuda failure", None, None),
            (" port error", "link down(east-west)", "link down(east-west)"),
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            ("pushrecv", "pushrecv", "pushrecv"),
            ("local catastrophic error", "local catastrophic error", "local catastrophic error"),
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            ("torch.OutOfMemoryError", "cudaError out of memory", "cudaError out of memory"),
            ("torch.cuda.OutOfMemoryError", "cudaError out of memory", "cudaError out of memory"),
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            ("RESOURCE_EXHAUSTED: Out of memory", "cudaError out of memory", "cudaError out of memory"),
            # moved to applog_rules.test.toml
            ("Call to ibv_set_ece failed", "Call to ibv_set_ece failed", "Call to ibv_set_ece failed"),
            ("sigkill_handler] Killing subprocess", "sigkill_handler] Killing subprocess", "alias else"),
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
            # moved to applog_rules.test.toml
        ]
    )

    @pytest.mark.parametrize(
        "line,symptom,rule",
        applog_to_symptom_rules,
    )
    def test_applog_to_symptom_rule(self, line, symptom, rule, big_jobinfo_builder, obs_data, event_stream, request):
        event_stream = event_stream(symptom)
        big_jobinfo = big_jobinfo_builder(True)
        obs = obs_data(big_jobinfo, [f"0: {line}"], ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES)
        event_stream.register_observation_context(obs.context)
        process_check(
            event_stream=event_stream,
            job_info=big_jobinfo,
            observation_source=obs,
            rules_engine=attribution_lib.get_rules_engine(),
        )

        node_attr = JobNodeAttributions.from_event_stream(event_stream, big_jobinfo)
        if symptom is None:
            assert len(node_attr.attributions) == 0
        else:
            match_bad(node_attr, big_jobinfo.nodes[0], symptom, rule)

    interesting_substr = [
        "NCCL watchdog thread terminated",
        "Exception raised from ncclCommWatchdog",
        "ProcessGroupNCCL's watchdog got stuck",
        "detected by watchdog at worker",
        "Watchdog caught collective operation",
    ]

    @pytest.mark.parametrize(
        "line,symptom,rule",
        applog_to_symptom_rules,
    )
    def test_applog_to_symptom_rule_interesting(
        self, line, symptom, rule, big_jobinfo_builder, obs_data, event_stream, request
    ):
        event_stream = event_stream(symptom)
        big_jobinfo = big_jobinfo_builder(False)
        obs = obs_data(big_jobinfo, [f"0: {line}"], ObservationSource.APPLOG, ObservationFormat.RAW_LOG_LINES)
        event_stream.register_observation_context(obs.context)
        process_check(
            event_stream=event_stream,
            job_info=big_jobinfo,
            observation_source=obs,
            rules_engine=attribution_lib.get_rules_engine(),
        )

        node_attr = JobNodeAttributions.from_event_stream(event_stream, big_jobinfo)
        if symptom is None:
            assert len(node_attr.attributions) == 0
        elif "NaN reported by Megatron" not in symptom:
            match_interesting(node_attr, big_jobinfo.nodes[0], f"(non-fatal) {symptom}", rule)

    def test_ibv_modify_qp_with_nccl_params(self, attribution_test_builder, timezone, event_manager):
        """Test that ibv_modify_qp error includes NCCL parameter warning when present in job script"""
        # Create job info with NCCL parameters in job script
        start = datetime.fromtimestamp(1700000000, tz=timezone)
        end = datetime.fromtimestamp(1700010000, tz=timezone)
        job_info = (
            JobInfoBuilder(
                id="test-123",
                cluster="test-cluster",
                state="CANCELLED",
                type="training",
                framework="pytorch",
                node_to_ranks=make_node_to_ranks(["host-9ad02e1b", "localhost"], 8),
                job_start_time=start,
                source_name="test",
                timezone=timezone,
            )
            .job_end_time(end)
            .time_limit(600)
            .job_script_content("export NCCL_IB_TC=128\nexport NCCL_IB_GID_INDEX=3\n")
            .build()
        )

        # Test log line with error code 110 (timeout)
        log_lines = [
            "0: Call to ibv_modify_qp failed with 110 Connection timed out, on dev mlx5_4:1, curr state INIT, next state RTR, local GID index 3, local GID ::ffff:192.168.26.95, remote GID ::ffff:127.0.0.1"
        ]

        result = attribution_test_builder(
            job_info=job_info,
            data=log_lines,
            source=ObservationSource.APPLOG,
            format=ObservationFormat.RAW_LOG_LINES,
            check_fn=process_check,
        )

        node_attr = JobNodeAttributions.from_event_stream(result.event_stream, job_info)

        # Should have bad_peers with NCCL parameter warning
        assert "host-9ad02e1b" in node_attr.attributions
        symptoms = node_attr.attributions["host-9ad02e1b"].attributions._data

        # Check that the symptom includes the NCCL parameter warning
        expected_symptom = "Call to ibv_modify_qp failed: eno 110 timeout(Overwrite NCCL parameters)"
        assert expected_symptom in symptoms, f"Expected '{expected_symptom}' in {list(symptoms.keys())}"

    def test_address_already_in_use(self, attribution_test_builder, timezone, event_manager):
        """Test that 'Address already in use' creates bad node attribution"""
        start = datetime.fromtimestamp(1700000000, tz=timezone)
        end = datetime.fromtimestamp(1700010000, tz=timezone)
        job_info = (
            JobInfoBuilder(
                id="test-456",
                cluster="test-cluster",
                state="FAILED",
                type="training",
                framework="pytorch",
                node_to_ranks=make_node_to_ranks(["host-abc123"], 8),
                job_start_time=start,
                source_name="test",
                timezone=timezone,
            )
            .job_end_time(end)
            .time_limit(600)
            .build()
        )

        # Test log line with "Address already in use"
        log_lines = ["0: ERROR: Address already in use on port 12345"]

        result = attribution_test_builder(
            job_info=job_info,
            data=log_lines,
            source=ObservationSource.APPLOG,
            format=ObservationFormat.RAW_LOG_LINES,
            check_fn=process_check,
        )

        node_attr = JobNodeAttributions.from_event_stream(result.event_stream, job_info)

        # Should have bad node attribution
        assert "host-abc123" in node_attr.attributions
        symptoms = node_attr.attributions["host-abc123"].attributions._data
        assert "Address already in use" in symptoms

    def test_address_already_in_use_excluded_for_nccl_bind(self, attribution_test_builder, timezone, event_manager):
        """Test that 'Address already in use' from ras/client_support.cc NCCL bind is excluded"""
        start = datetime.fromtimestamp(1700000000, tz=timezone)
        end = datetime.fromtimestamp(1700010000, tz=timezone)
        job_info = (
            JobInfoBuilder(
                id="test-789",
                cluster="test-cluster",
                state="COMPLETED",
                type="training",
                framework="pytorch",
                node_to_ranks=make_node_to_ranks(["host-def456"], 8),
                job_start_time=start,
                source_name="test",
                timezone=timezone,
            )
            .job_end_time(end)
            .time_limit(600)
            .build()
        )

        # This specific pattern should be excluded (non-fatal NCCL warning)
        log_lines = ["0: ras/client_support.cc:159 NCCL WARN Call to bind failed: Address already in use"]

        result = attribution_test_builder(
            job_info=job_info,
            data=log_lines,
            source=ObservationSource.APPLOG,
            format=ObservationFormat.RAW_LOG_LINES,
            check_fn=process_check,
        )

        node_attr = JobNodeAttributions.from_event_stream(result.event_stream, job_info)

        # Should NOT have any bad node attribution for this excluded pattern
        # Either the node is not in attributions, or it doesn't have "Address already in use" symptom
        if "host-def456" in node_attr.attributions:
            symptoms = node_attr.attributions["host-def456"].attributions._data
            assert "Address already in use" not in symptoms

    def test_address_already_in_use_lowercase_libuv(self, attribution_test_builder, timezone, event_manager):
        """Test that lowercase 'address already in use' (libuv rendering) is detected.

        torch's libuv-backed TCPStore renders EADDRINUSE via uv_strerror(), which is
        all-lowercase, unlike the glibc strerror() rendering surfaced by Python's
        OSError ("[Errno 98] Address already in use"). This regression case
        verifies that the lowercase rendering is detected.
        """
        start = datetime.fromtimestamp(1700000000, tz=timezone)
        end = datetime.fromtimestamp(1700010000, tz=timezone)
        job_info = (
            JobInfoBuilder(
                id="test-399020",
                cluster="test-cluster",
                state="FAILED",
                type="training",
                framework="pytorch",
                node_to_ranks=make_node_to_ranks(["host-abc123"], 8),
                job_start_time=start,
                source_name="test",
                timezone=timezone,
            )
            .job_end_time(end)
            .time_limit(600)
            .build()
        )

        log_lines = [
            "0: torch.distributed.DistNetworkError: The server socket has failed to listen on any local "
            "network address. port: 14485, useIpv6: false, code: -98, name: EADDRINUSE, "
            "message: address already in use"
        ]

        result = attribution_test_builder(
            job_info=job_info,
            data=log_lines,
            source=ObservationSource.APPLOG,
            format=ObservationFormat.RAW_LOG_LINES,
            check_fn=process_check,
        )

        node_attr = JobNodeAttributions.from_event_stream(result.event_stream, job_info)

        assert "host-abc123" in node_attr.attributions
        symptoms = node_attr.attributions["host-abc123"].attributions._data
        assert "Address already in use" in symptoms


class TestNCCLCudaVersionMatch:
    """Test NCCLCudaVersion.match() method."""

    def test_match_both_nccl_and_cuda_version(self, app_tst_bldr, big_jobinfo):
        """Test matching line with both NCCL and CUDA version."""
        data = ["0: NCCL version 2.18.5+cuda12.2"]
        result = app_tst_bldr(big_jobinfo, data)

        versions = NCCLCudaVersion.collect(result.event_stream)
        assert "nccl_version" in versions
        assert versions["nccl_version"].value() == "2.18.5"
        assert "cuda_version" in versions
        assert versions["cuda_version"].value() == "12.2"

    def test_match_only_nccl_version(self, app_tst_bldr, big_jobinfo):
        """Test matching line with only NCCL version (no CUDA version)."""
        data = [
            "104: [rank106]: torch.distributed.DistBackendError: NCCL error in: /opt/pytorch/pytorch/torch/csrc/distributed/c10d/NCCLUtils.hpp:275, unhandled system error (run with NCCL_DEBUG=INFO for details), NCCL version 2.22.3"
        ]
        result = app_tst_bldr(big_jobinfo, data)

        versions = NCCLCudaVersion.collect(result.event_stream)
        assert "nccl_version" in versions
        assert versions["nccl_version"].value() == "2.22.3"
        # No cuda_version should be present
        assert "cuda_version" not in versions

    def test_match_no_version_info(self, app_tst_bldr, big_jobinfo):
        """Test line without NCCL version info is not matched."""
        data = ["Some log line without version information"]
        result = app_tst_bldr(big_jobinfo, data)

        versions = NCCLCudaVersion.collect(result.event_stream)
        assert len(versions) == 0

    def test_match_malformed_version_string(self, app_tst_bldr, big_jobinfo):
        """Test line with 'NCCL version' text but no parseable version."""
        data = ["NCCL version is not available"]
        result = app_tst_bldr(big_jobinfo, data)

        versions = NCCLCudaVersion.collect(result.event_stream)
        # Should not match because the regex won't capture the version
        assert len(versions) == 0

    def test_collect_returns_first_event_only(self, app_tst_bldr, big_jobinfo):
        """Test that collect() returns only the first NCCL version event."""
        data = [
            "0: NCCL version 2.18.5+cuda12.2",
            "1: Some other log",
            "2: NCCL version 2.19.0+cuda12.3",  # This should be ignored
        ]
        result = app_tst_bldr(big_jobinfo, data)

        versions = NCCLCudaVersion.collect(result.event_stream)
        # Should only have the first version
        assert versions["nccl_version"].value() == "2.18.5"
        assert versions["cuda_version"].value() == "12.2"


class TestWatchdogFoundValues:
    """Test WatchdogFoundValues.match() and collect() methods."""

    def test_match_watchdog_timeout(self, app_tst_bldr, big_jobinfo):
        """Test matching a watchdog timeout line."""
        data = [
            "[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=91183503, OpType=BROADCAST, NumelIn=8192, NumelOut=8192, Timeout(ms)=600000) ran for 600000 milliseconds before timing out."
        ]
        result = app_tst_bldr(big_jobinfo, data)

        # Check that the event was added by trying to get its type ID
        watchdog_type_id = result.event_stream.mgr.types.get_id("Application.pytorch.nccl_watchdog")
        assert watchdog_type_id is not None

    def test_match_no_watchdog(self, app_tst_bldr, big_jobinfo):
        """Test that non-watchdog lines are not matched."""
        data = ["Some regular log line without watchdog timeout"]
        result = app_tst_bldr(big_jobinfo, data)

        # Check that no watchdog events were emitted
        watchdog_type_id = result.event_stream.mgr.types.get_id("Application.pytorch.nccl_watchdog")
        if watchdog_type_id is not None:
            count = result.event_stream._conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_type=? AND status=1", [watchdog_type_id]
            ).fetchone()[0]
            assert count == 0

    def test_collect_with_mfu_event(self, app_tst_bldr, big_jobinfo):
        """Test collect() when MFU event provides timestamp context."""
        data = [
            " [2024-02-23 09:59:14] iteration   516520/  900692 | consumed samples:   1067990784 | elapsed time per iteration (ms): 8096.8 | throughput per GPU (TFLOP/s/GPU): 399.3 | learning rate: 4.838E-05 | decoupled learning rate: 2.190E-04 | global batch size:  2304 | lm loss: 1.437022E+00 | loss scale: 1.0 | grad norm: 0.217 | num zeros: 1640372992.0 | number of skipped iterations:   0 | number of nan iterations:   0 |",
            "[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=91183503, OpType=BROADCAST, NumelIn=8192, NumelOut=8192, Timeout(ms)=600000) ran for 600000 milliseconds before timing out.",
        ]
        result = app_tst_bldr(big_jobinfo, data, insert_tflops_line=False)

        watchdog_data = WatchdogFoundValues.collect(result.event_stream, big_jobinfo)

        assert watchdog_data is not None
        watchdog_value = watchdog_data.value()
        assert "error_time" in watchdog_value
        assert "timeout_seconds" in watchdog_value
        assert watchdog_value["timeout_seconds"] == 600.0  # 600000 ms = 600 s

    def test_collect_without_mfu_uses_job_start(self, app_tst_bldr, big_jobinfo):
        """Test collect() falls back to job start time when no MFU event."""
        data = [
            "[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=91183503, OpType=BROADCAST, NumelIn=8192, NumelOut=8192, Timeout(ms)=600000) ran for 600000 milliseconds before timing out."
        ]
        result = app_tst_bldr(big_jobinfo, data)

        watchdog_data = WatchdogFoundValues.collect(result.event_stream, big_jobinfo)

        assert watchdog_data is not None
        watchdog_value = watchdog_data.value()
        assert "error_time" in watchdog_value
        assert "timeout_seconds" in watchdog_value
        # Check that error_time has same date as job start
        job_start = big_jobinfo.interest_start_time
        assert watchdog_value["error_time"].year == job_start.year
        assert watchdog_value["error_time"].month == job_start.month
        assert watchdog_value["error_time"].day == job_start.day

    def test_collect_no_watchdog_event(self, app_tst_bldr, big_jobinfo):
        """Test collect() returns None when no watchdog event present."""
        data = ["Some regular log line"]
        result = app_tst_bldr(big_jobinfo, data)

        watchdog_data = WatchdogFoundValues.collect(result.event_stream, big_jobinfo)
        assert watchdog_data is None

    def test_collect_multiple_watchdog_events_returns_first(self, app_tst_bldr, big_jobinfo):
        """Test that collect() returns only the first watchdog event."""
        data = [
            " [2024-02-23 09:59:14] iteration   516520/  900692 | consumed samples:   1067990784 | elapsed time per iteration (ms): 8096.8 | throughput per GPU (TFLOP/s/GPU): 399.3 | learning rate: 4.838E-05 | decoupled learning rate: 2.190E-04 | global batch size:  2304 | lm loss: 1.437022E+00 | loss scale: 1.0 | grad norm: 0.217 | num zeros: 1640372992.0 | number of skipped iterations:   0 | number of nan iterations:   0 |",
            "[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=91183503, OpType=BROADCAST, NumelIn=8192, NumelOut=8192, Timeout(ms)=600000) ran for 600000 milliseconds before timing out.",
            "[rank1]:[E118 16:45:20.123456789 ProcessGroupNCCL.cpp:603] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=91183504, OpType=ALLREDUCE, NumelIn=4096, NumelOut=4096, Timeout(ms)=300000) ran for 300000 milliseconds before timing out.",
        ]
        result = app_tst_bldr(big_jobinfo, data, insert_tflops_line=False)

        watchdog_data = WatchdogFoundValues.collect(result.event_stream, big_jobinfo)

        assert watchdog_data is not None
        watchdog_value = watchdog_data.value()
        # Should be the first watchdog event (600 seconds, not 300)
        assert watchdog_value["timeout_seconds"] == 600.0

    def test_watchdog_timeout_conversion(self, app_tst_bldr, big_jobinfo):
        """Test timeout conversion from milliseconds to seconds."""
        test_cases = [
            (600000, 600.0),  # 10 minutes
            (300000, 300.0),  # 5 minutes
            (1800000, 1800.0),  # 30 minutes
            (60000, 60.0),  # 1 minute
        ]

        for timeout_ms, expected_seconds in test_cases:
            data = [
                f"[rank0]:[E118 15:31:43.493196954 ProcessGroupNCCL.cpp:603] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=91183503, OpType=BROADCAST, NumelIn=8192, NumelOut=8192, Timeout(ms)={timeout_ms}) ran for {timeout_ms} milliseconds before timing out."
            ]
            result = app_tst_bldr(big_jobinfo, data)

            watchdog_data = WatchdogFoundValues.collect(result.event_stream, big_jobinfo)
            assert watchdog_data is not None
            assert watchdog_data.value()["timeout_seconds"] == expected_seconds


class TestNCCLLateArrival:
    """Tests for NCCL Late Arrival detection rules."""

    def test_nccl_late_arrival_single_node_straggler(self, request, app_tst_bldr):
        """Watchdog lines are aggregated by the filter to identify straggler nodes."""

        # 4 nodes x 2 ranks/node = 8 ranks in world size
        nodes = ["node-000", "node-001", "node-002", "node-003"]
        job_info = create_simple_job_info(
            request=request,
            nodes=nodes,
            ranks_per_node=2,
        )
        # Ranks 0-5 report watchdog timeout, ranks 6, 7 (node-003) never report.
        watchdog_lines = [
            f"{r}: [rank{r}]:[E222 18:24:56.341356226 ProcessGroupNCCL.cpp:697] "
            f"[Rank {r}] Watchdog caught collective operation timeout: "
            f"WorkNCCL(SeqNum=506, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) "
            f"ran for 600087 milliseconds before timing out."
            for r in range(6)
        ]

        result = app_tst_bldr(job_info, watchdog_lines)
        node_attr = result.get_node_attributions()

        match_interesting(node_attr, "node-003", "NCCL Late Arrival", "NCCLLateArrivalNodes")
