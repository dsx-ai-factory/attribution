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


def _ingest_and_collect(
    request: pytest.FixtureRequest, event_manager, job_info: JobInfo, lines: list[str]
) -> tuple[dict[str, StragglerNodeReport], IngestSummary]:
    """Ingest application log lines. Return the ``megatron_straggler`` digest and the ingest summary."""
    first_line_time = job_info.job_start_time + timedelta(minutes=10)
    entries = [
        make_applog_entry(line, (first_line_time + timedelta(seconds=offset)).timestamp() * 1e9)
        for offset, line in enumerate(lines)
    ]
    context = ObservationContext(
        time_interval=TimeInterval(start=job_info.job_start_time, end=job_info.job_end_time),
        source=ObservationSource.APPLOG,
        format=ObservationFormat.ENTRY_TIMESERIES,
        timezone=UTC,
    )
    attributor = Attributor.new(event_manager.create_stream(request.node.name), job_info)
    observation_id = attributor.ingest(InboundObservation(context, entries))
    digest = attributor.attribute(run_collectors=True).collectors["megatron_straggler"]
    return digest, attributor.ingested_observations[observation_id].summary


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

    digest, summary = _ingest_and_collect(request, event_manager, job_info, lines)

    assert list(digest) == ["node-b"]
    # The prefilter still drops the other "INFO:megatron" line.
    assert summary.dropped[DropReason.PRE_FILTERED] == 1
