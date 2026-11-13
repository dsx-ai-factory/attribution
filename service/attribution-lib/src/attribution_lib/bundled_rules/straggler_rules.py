# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Megatron straggler detection rules.

Megatron rules parse ranks inline and emit completed group events.
"""

import re

from attribution_lib.bundled_rules.common.megatron import (
    MEGATRON_STRAGGLER_BOTTOM_RANKS,
    MEGATRON_STRAGGLER_QUALIFIER_TYPE,
    MEGATRON_STRAGGLER_SCORE_TYPE,
    MEGATRON_STRAGGLER_TOP_RANKS,
    MegatronStragglerGroup,
    MegatronStragglerItem,
    MegatronStragglerParams,
    MegatronStragglerParseState,
    filter_megatron_stragglers,
)
from attribution_lib.bundled_rules.common.straggler_types import StragglerQualifier, StragglerScoreType
from loguru import logger

from attribution_lib.exceptions import UnknownResourceError
from attribution_lib.rules import EventTriple, RuleContext, collector_def, rule_processor
from attribution_lib.sources import Source, SourceSet

rule_author = "NVIDIA Attribution team"

RANK_SCORE_RE = re.compile(r"(?P<score>\d+\.\d+)/(?P<rank>\d+)/?")


def _parse_megatron_ranks(
    ranks_str: str, source: Source, ctx: RuleContext, rule_name: str
) -> list[MegatronStragglerItem]:
    """Parse a Megatron ranks string into MegatronStragglerItem list."""
    items: list[MegatronStragglerItem] = []
    if ctx.job_info is None:
        return items

    rank_pairs = ranks_str.split(",")
    for pair in rank_pairs:
        pair = pair.strip()
        if match := RANK_SCORE_RE.match(pair):
            score = float(match.group("score"))
            rank = int(match.group("rank"))
            try:
                node = ctx.job_info.get_name_from_rank(rank)
                items.append(MegatronStragglerItem(rank, node, score, source.derived(rule=rule_name)))
            except UnknownResourceError:
                logger.debug(f"Skipping Megatron straggler rank {rank}: unknown resource (score={score})")
                continue
    return items


def _megatron_group_event(
    ranks_str: str,
    source: Source,
    ctx: RuleContext,
    *,
    rule_name: str,
    qualifier: StragglerQualifier,
    event_type: str,
) -> list[EventTriple]:
    """Emit one Megatron straggler group event for a "Bottom" or "Top" report line."""
    ranks = _parse_megatron_ranks(ranks_str, source, ctx, rule_name)
    if not ranks:
        return []

    score_type_set = SourceSet(StragglerScoreType.INDIVIDUAL)
    score_type_set.add(source.derived(rule=MEGATRON_STRAGGLER_SCORE_TYPE))

    qual_set = SourceSet(qualifier)
    qual_set.add(source.derived(rule=MEGATRON_STRAGGLER_QUALIFIER_TYPE))

    group = MegatronStragglerGroup(ranks, qual_set, score_type_set)
    return [
        EventTriple(
            type=event_type,
            payload={"data": group},
            source=ranks[0].source,
        )
    ]


@rule_processor("straggler:MegatronStragglerBottom")
def megatron_bottom(groups: dict[str, str], node: str, source: Source, ctx: RuleContext) -> list[EventTriple]:
    """Emit the lowest-throughput ranks. ``Collect.MegatronStraggler`` flags straggler ranks from this group."""
    return _megatron_group_event(
        groups["ranks"],
        source,
        ctx,
        rule_name=MEGATRON_STRAGGLER_BOTTOM_RANKS,
        qualifier=StragglerQualifier.WORST,
        event_type="Application.MegatronStraggler.bottom",
    )


@rule_processor("straggler:MegatronStragglerTop")
def megatron_top(groups: dict[str, str], node: str, source: Source, ctx: RuleContext) -> list[EventTriple]:
    """Emit the highest-throughput ranks. ``Collect.MegatronStraggler`` infers its threshold from this group."""
    return _megatron_group_event(
        groups["ranks"],
        source,
        ctx,
        rule_name=MEGATRON_STRAGGLER_TOP_RANKS,
        qualifier=StragglerQualifier.BEST,
        event_type="Application.MegatronStraggler.top",
    )


@collector_def(
    name="Collect.MegatronStraggler",
    formatter=lambda raw: {"megatron_straggler": raw},
    params_model=MegatronStragglerParams,
)
def _collect_megatron_straggler(event_stream, _job_info, params):
    data = MegatronStragglerParseState.collect(event_stream)
    return filter_megatron_stragglers(data, params)
