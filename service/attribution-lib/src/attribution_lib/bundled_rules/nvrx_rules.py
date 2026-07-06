# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
NVRX Straggler Detector rules.

These rules parse per-line Straggler Detector output from NVRX clusters.
Events are reconstructed into groups at collection time by StragglerParseState.collect().
"""

from loguru import logger

from attribution_lib.bundled_rules.common.nvrx import (
    NVRXStragglerParams,
    StragglerParseState,
    filter_nvrx_stragglers,
)
from attribution_lib.exceptions import UnknownNodeError, UnknownResourceError
from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rule_helpers import _bad, _interesting
from attribution_lib.rules_engine import (
    EventTriple,
    RuleContext,
    collector_def,
    declare_event_types,
    declare_rule_names,
    extractor_def,
)
from attribution_lib.sources import Source

rule_author = "FACT Team"
rule_sources = {ObsSource.APPLOG}

declare_event_types("Attribution.node.bad", "Attribution.node.interesting")

# NVRX derived rules are produced by StragglerParseState.collect() in nvrx.py,
# not by any specific extractor, so they must be declared here.
declare_rule_names(
    "NVRX Straggler Score Type",
    "NVRX Straggler Qualifier Type",
    "NVRX Straggler Body",
    "NVRX Straggler Unbounded Body",
)

NVRX_SYNTHETIC_FAULT_INJECTION = "NVRx synthetic fault injection"


@extractor_def(
    name="NVRXInjectedException",
    match_re=r"nvidia_resiliency_ext\.shared_utils\.inject_fault\.InjectedException",
)
def nvrx_injected_exception(groups: dict[str, str], node: str, source: Source, ctx: RuleContext) -> list[EventTriple]:
    """Surface an NVRx injected exception as an artificial failure.

    Args:
        groups: Empty, the match regex contains no capture groups.
        node: Source node supplied by the observation reader.
        source: Source attribution for the matching log line.
        ctx: Rule context with optional job metadata.

    Returns:
        A bad-node attribution for the node that logged the injected exception.
    """
    return [_bad(node, NVRX_SYNTHETIC_FAULT_INJECTION, source)]


@extractor_def(
    name="NVRXRankSectionTimeout",
    match_re=(
        r"(?:\[[A-Z]+\]\s+\[(?P<reported_node>[A-Za-z0-9_.-]+)\]\s+)?"
        r"(?:\[[^\]]+\]\s+)*"
        r"rank_monitor_server\.py:\d+\s+\[Cycle (?:\d+)\]\s+"
        r"Section 'step' has been open for (?P<duration_seconds>\d+(?:\.\d+)?) seconds\. "
        r"Timeout is (?P<timeout_seconds>\d+(?:\.\d+)?) seconds\."
    ),
)
def nvrx_rank_section_timeout(groups: dict[str, str], node: str, source: Source, ctx: RuleContext) -> list[EventTriple]:
    """Surface NVRx rank-monitor step timeouts as interesting evidence.

    Args:
        groups: Regex captures from the matched rank-monitor timeout line.
        node: Source node supplied by the observation reader.
        source: Source attribution for the matching log line.
        ctx: Rule context with optional job metadata.

    Returns:
        An interesting attribution for the reported node, or for the source node
        when the line does not include a reported node.
    """
    duration_seconds = float(groups["duration_seconds"])
    timeout_seconds = float(groups["timeout_seconds"])
    if duration_seconds <= timeout_seconds:
        return []

    reported_node = groups.get("reported_node")
    attributed_node = node
    if reported_node is not None and reported_node != "":
        attributed_node = reported_node.lower()
        if ctx.job_info is not None:
            try:
                attributed_node = ctx.job_info.verify_node(reported_node)
            except UnknownResourceError:
                attributed_node = reported_node.lower()
    return [_interesting(attributed_node, "NVRx step section timeout", source)]


@extractor_def(
    name="StragglerDetectorHeader",
    match_re=r"GPU (?P<class>individual|relative) performance:\s*$",
    derived_event_types=["Application.NVRXStraggler.header"],
)
def straggler_header(groups: dict[str, str], node: str, source: Source, ctx: RuleContext) -> list[EventTriple]:
    return [
        EventTriple(
            type="Application.NVRXStraggler.header",
            payload={"class": groups["class"]},
            source=source,
        )
    ]


@extractor_def(
    name="StragglerDetectorQualifier",
    match_re=r"(?P<class>Best|Worst) performing (?P<count>\d+)/(?P<max_count>\d+) ranks:\s*$",
    derived_event_types=["Application.NVRXStraggler.qualifier"],
)
def straggler_qualifier(groups: dict[str, str], node: str, source: Source, ctx: RuleContext) -> list[EventTriple]:
    return [
        EventTriple(
            type="Application.NVRXStraggler.qualifier",
            payload={"class": groups["class"], "count": groups["count"], "max_count": groups["max_count"]},
            source=source,
        )
    ]


@extractor_def(
    name="StragglerDetectorRank",
    match_re=r"Rank=(?P<rank>[^ ]+) Node=(?P<node>[^ ]+) Score=(?P<score>\d\.\d\d)\s*$",
    derived_event_types=["Application.NVRXStraggler.rank"],
)
def straggler_rank(groups: dict[str, str], node: str, source: Source, ctx: RuleContext) -> list[EventTriple]:
    rank = int(groups["rank"])
    reported_node = groups["node"]
    score = float(groups["score"])
    resolved_node = None
    if ctx.job_info is not None:
        try:
            resolved_node = ctx.job_info.verify_node(reported_node)
        except UnknownNodeError:
            if ctx.job_info.rank_exists(rank):
                resolved_node = ctx.job_info.get_name_from_rank(rank)
            else:
                logger.warning(
                    "NVRX straggler line references node {!r} and rank {} but neither is in "
                    "job {} topology; emitting event with node=None",
                    reported_node,
                    rank,
                    ctx.job_info.id,
                )
    return [
        EventTriple(
            type="Application.NVRXStraggler.rank",
            payload={"rank": rank, "reported_node": reported_node, "node": resolved_node, "score": score},
            source=source,
        )
    ]


@collector_def(
    name="Collect.NVRXStraggler",
    formatter=lambda raw: {"nvrx_straggler": raw},
    params_model=NVRXStragglerParams,
)
def _collect_nvrx_straggler(event_stream, _job_info, params):
    groups = StragglerParseState.collect(event_stream)
    return filter_nvrx_stragglers(groups, params)
