# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Megatron straggler report lines produce per-node reports in ``Collect.MegatronStraggler``.

Megatron-Core's ``StragglerDetector`` logs a "Bottom" line and a "Top" line for each report.
Each line lists ``<estimated throughput in TFLOPs>/<rank>/<node>`` entries. Some Megatron-Core
versions omit the node name.
"""

from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

from attribution_lib.attributor import Attributor
from attribution_lib.bundled_rules.common.straggler_types import StragglerNodeReport
from attribution_lib.ingest_summary import DropReason, IngestSummary
from attribution_lib.job_info import JobInfo
from attribution_lib.observation.inbound import (
    Context as ObservationContext,
    Format as ObservationFormat,
    Observation as InboundObservation,
    Source as ObservationSource,
    TimeInterval,
)
from test_support.helpers import create_simple_job_info, make_applog_entry

UTC = ZoneInfo("UTC")

# Rank 5 reaches about three quarters of the throughput of the fastest rank.
BOTTOM_WITH_SLOW_RANK = "^^^^ Bottom 2 Ranks with lowest  Etpt(TF): 213.46/5/node-b, 287.92/1/node-a,"
# Every rank is within 5 % of the fastest rank, which is the default threshold.
BOTTOM_WITHIN_THRESHOLD = "^^^^ Bottom 2 Ranks with lowest  Etpt(TF): 287.50/5/node-b, 287.92/1/node-a,"
TOP = "^^^^ Top    2 Ranks with highest Etpt(TF): 289.21/6/node-b, 289.31/2/node-a,"
# Python's default log format puts the level and the logger name in front of each message.
# The leading "0: " is the rank label that the launcher adds to each output line.
MEGATRON_LOGGER_PREFIX = "0: INFO:megatron.core.utils:"
OTHER_MEGATRON_INFO_LINE = "0: INFO:megatron.core.timers:(min, max) time across ranks (ms):"


@pytest.fixture
def job_info() -> JobInfo:
    """Two nodes with four ranks each: ranks 0-3 run on node-a and ranks 4-7 on node-b."""
    return create_simple_job_info(timezone=UTC, nodes=["node-a", "node-b"], ranks_per_node=4)


def _observation(job_info: JobInfo, lines: list[str], *, minutes_after_start: int) -> InboundObservation:
    """Build one observation of application log lines. The first line is ``minutes_after_start`` after the job start."""
    context = ObservationContext(
        time_interval=TimeInterval(start=job_info.job_start_time, end=job_info.job_end_time),
        source=ObservationSource.APPLOG,
        format=ObservationFormat.ENTRY_TIMESERIES,
        timezone=UTC,
    )
    first_line_time = job_info.job_start_time + timedelta(minutes=minutes_after_start)
    entries = [
        make_applog_entry(line, (first_line_time + timedelta(seconds=offset)).timestamp() * 1e9)
        for offset, line in enumerate(lines)
    ]
    return InboundObservation(context, entries)


def _collect(
    event_manager, job_info: JobInfo, stream_name: str, observations: list[InboundObservation]
) -> tuple[dict[str, StragglerNodeReport], list[IngestSummary]]:
    """Ingest the observations in the given order into the stream ``stream_name``.

    Return the ``megatron_straggler`` digest and the ingest summary of each observation.
    """
    attributor = Attributor.new(event_manager.create_stream(stream_name), job_info)
    observation_ids = [attributor.ingest(observation) for observation in observations]
    digest = attributor.attribute(run_collectors=True).collectors["megatron_straggler"]
    return digest, [attributor.ingested_observations[observation_id].summary for observation_id in observation_ids]


def _ingest_and_collect(
    request: pytest.FixtureRequest, event_manager, job_info: JobInfo, *line_batches: list[str]
) -> tuple[dict[str, StragglerNodeReport], list[IngestSummary]]:
    """Ingest each batch of application log lines as one observation. The batches are ten minutes apart.

    Return the ``megatron_straggler`` digest and the ingest summary of each observation.
    """
    observations = [
        _observation(job_info, lines, minutes_after_start=10 * batch_number)
        for batch_number, lines in enumerate(line_batches, start=1)
    ]
    return _collect(event_manager, job_info, request.node.name, observations)


def test_rank_below_the_inferred_threshold_is_reported_on_its_node(request, event_manager, job_info):
    digest, _ = _ingest_and_collect(request, event_manager, job_info, [BOTTOM_WITH_SLOW_RANK, TOP])

    assert list(digest) == ["node-b"]
    report = digest["node-b"]
    assert [(rank_report.rank, rank_report.score) for rank_report in report.rank_scores] == [(5, 213.46)]
    assert BOTTOM_WITH_SLOW_RANK in {source.datum for source in report.sources}


def test_ranks_within_the_inferred_threshold_are_not_reported(request, event_manager, job_info):
    digest, _ = _ingest_and_collect(request, event_manager, job_info, [BOTTOM_WITHIN_THRESHOLD, TOP])

    assert digest == {}


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param([BOTTOM_WITH_SLOW_RANK], id="bottom-line-without-top-line"),
        pytest.param(["^^^^ Bottom 2 Ranks with lowest  Etpt(TF): not available,", TOP], id="no-rank-entries"),
        pytest.param(["^^^^ Bottom 1 Ranks with lowest  Etpt(TF): 213.46/99/node-z,", TOP], id="rank-outside-the-job"),
    ],
)
def test_incomplete_or_unusable_report_is_not_reported(request, event_manager, job_info, lines):
    digest, _ = _ingest_and_collect(request, event_manager, job_info, lines)

    assert digest == {}


def test_report_in_python_default_log_format_passes_the_prefilter(request, event_manager, job_info):
    lines = [
        OTHER_MEGATRON_INFO_LINE,
        MEGATRON_LOGGER_PREFIX + BOTTOM_WITH_SLOW_RANK,
        MEGATRON_LOGGER_PREFIX + TOP,
    ]

    digest, (summary,) = _ingest_and_collect(request, event_manager, job_info, lines)

    assert list(digest) == ["node-b"]
    # The prefilter still drops the other "INFO:megatron" line.
    assert summary.dropped[DropReason.PRE_FILTERED] == 1


def test_reports_in_separate_observations_pair_within_their_own_observation(request, event_manager, job_info):
    """Line numbers restart in each observation, so a Bottom group must not pair with another report's Top group."""
    first_report = [
        "^^^^ Bottom 1 Ranks with lowest  Etpt(TF): 100.00/1/node-a,",
        "^^^^ Top    1 Ranks with highest Etpt(TF): 110.00/2/node-a,",
    ]
    second_report = [BOTTOM_WITH_SLOW_RANK, TOP]

    digest, _ = _ingest_and_collect(request, event_manager, job_info, first_report, second_report)

    flagged_ranks = {node: [rank_report.rank for rank_report in report.rank_scores] for node, report in digest.items()}
    assert flagged_ranks == {"node-a": [1], "node-b": [5]}


@pytest.mark.parametrize(
    "line_batches",
    [
        pytest.param([[BOTTOM_WITH_SLOW_RANK], [TOP]], id="bottom-line-in-the-first-observation"),
        pytest.param([[TOP], [BOTTOM_WITH_SLOW_RANK]], id="bottom-line-in-the-second-observation"),
    ],
)
def test_bottom_group_does_not_pair_with_top_group_of_another_observation(
    request, event_manager, job_info, line_batches
):
    digest, _ = _ingest_and_collect(request, event_manager, job_info, *line_batches)

    assert digest == {}


def test_rank_scores_follow_the_order_of_ingest(request, event_manager, job_info):
    """An observation id is a salted hash. An order by that id gives the same digest for both orders of ingest."""
    slow = _observation(job_info, [BOTTOM_WITH_SLOW_RANK, TOP], minutes_after_start=10)
    slower_lines = ["^^^^ Bottom 1 Ranks with lowest  Etpt(TF): 200.00/5/node-b,", TOP]
    slower = _observation(job_info, slower_lines, minutes_after_start=20)

    slow_first, _ = _collect(event_manager, job_info, f"{request.node.name}-slow-first", [slow, slower])
    slower_first, _ = _collect(event_manager, job_info, f"{request.node.name}-slower-first", [slower, slow])

    assert [rank_report.score for rank_report in slow_first["node-b"].rank_scores] == [213.46, 200.0]
    assert [rank_report.score for rank_report in slower_first["node-b"].rank_scores] == [200.0, 213.46]
