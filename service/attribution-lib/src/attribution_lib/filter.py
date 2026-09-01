# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import attribution_lib
from attribution_lib.decision_trace_recorder import DecisionRecorder
from attribution_lib.events import EventStream
from attribution_lib.job_info import JobInfo
from attribution_lib.rule_types.definitions import FilterContext, FilterPriority, RefinerContext

# Constants for bad-to-interesting type mappings
_BAD_TYPES = ["Attribution.node.bad", "Attribution.node.bad_peers"]
_INTERESTING_TYPES = ["Attribution.node.interesting", "Attribution.node.interesting_peers"]


def filter_events(
    event_stream: EventStream,
    job_info: JobInfo,
    *,
    phase: Optional[int] = None,
):
    """Run event-stream filter rules (EARLY / MID / STANDARD / LATE)."""
    attribution_lib.get_rules_engine().execute(FilterContext(job_info=job_info, event_stream=event_stream, phase=phase))


def _run_refiners(context: RefinerContext) -> RefinerContext:
    attribution_lib.get_rules_engine().execute(context)
    return context


def run_scrub_phase_refiners(
    event_stream: EventStream,
    job_info: JobInfo,
    *,
    my_bad_nodes: Any,
    recorder: Optional[DecisionRecorder] = None,
) -> None:
    """Run SCRUB-phase refiners only (per-node cascade cleanup before aggregation)."""
    _run_refiners(
        RefinerContext(
            event_stream=event_stream,
            job_info=job_info,
            symptoms=None,
            my_bad_nodes=my_bad_nodes,
            phase_window=(FilterPriority.SCRUB, FilterPriority.FINAL),
            recorder=recorder,
        )
    )


def run_pre_outcome_refiners(
    event_stream: EventStream,
    job_info: JobInfo,
    *,
    symptoms: Any,
    my_bad_nodes: Any,
    raw_keywords: Any = None,
    extract_job_metadata_from_logfile: bool = False,
    recorder: Optional[DecisionRecorder] = None,
) -> tuple[Optional[dict], Optional[set]]:
    """Run pre-outcome refiner rules on materialized symptoms and my_bad_nodes."""
    ctx = _run_refiners(
        RefinerContext(
            event_stream=event_stream,
            job_info=job_info,
            symptoms=symptoms,
            my_bad_nodes=my_bad_nodes,
            raw_keywords=raw_keywords,
            extract_job_metadata_from_logfile=extract_job_metadata_from_logfile,
            phase_window=(FilterPriority.FINAL, FilterPriority.POST_OUTCOME),
            recorder=recorder,
        )
    )
    return ctx.node_symptoms_base, ctx.skipped_symptom_ids


def run_post_outcome_refiners(
    event_stream: EventStream,
    job_info: JobInfo,
    *,
    symptoms: Any,
    my_bad_nodes: Any,
    outcome: Any,
    cancelled_by: Any = None,
    attribution_result: Any = None,
    bad_node_result: Any = None,
    recorder: Optional[DecisionRecorder] = None,
) -> "RefinerContext":
    """Run post-outcome refiner rules (POST_OUTCOME phase) after calculate_outcome."""
    return _run_refiners(
        RefinerContext(
            event_stream=event_stream,
            job_info=job_info,
            symptoms=symptoms,
            my_bad_nodes=my_bad_nodes,
            outcome=outcome,
            cancelled_by=cancelled_by,
            attribution_result=attribution_result,
            bad_node_result=bad_node_result,
            phase_window=(FilterPriority.POST_OUTCOME, None),
            recorder=recorder,
        )
    )
