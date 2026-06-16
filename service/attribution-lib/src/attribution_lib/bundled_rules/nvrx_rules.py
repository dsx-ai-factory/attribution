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

from attribution_lib.bundled_rules.common.nvrx import (
    NVRXStragglerParams,
    StragglerParseState,
    filter_nvrx_stragglers,
)
from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rules_engine import EventTriple, RuleContext, collector_def, declare_rule_names, extractor_def
from attribution_lib.sources import Source

rule_author = "FACT Team"
rule_sources = {ObsSource.APPLOG}

# NVRX derived rules are produced by StragglerParseState.collect() in nvrx.py,
# not by any specific extractor, so they must be declared here.
declare_rule_names(
    "NVRX Straggler Score Type",
    "NVRX Straggler Qualifier Type",
    "NVRX Straggler Body",
    "NVRX Straggler Unbounded Body",
)


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
        resolved_node = ctx.job_info.find_node(reported_node, rank)
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
