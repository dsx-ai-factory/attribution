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
Megatron straggler detection rules.

Megatron rules parse ranks inline and emit completed group events.
"""

import re

from attribution_lib.bundled_rules.common.megatron import (
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
from attribution_lib.rules_engine import EventTriple, RuleContext, collector_def, rule_processor
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


@rule_processor("straggler:MegatronStragglerTop")
def megatron_top(groups: dict[str, str], node: str, source: Source, ctx: RuleContext) -> list[EventTriple]:
    ranks = _parse_megatron_ranks(groups["ranks"], source, ctx, MEGATRON_STRAGGLER_TOP_RANKS)
    if not ranks:
        return []

    score_type_set = SourceSet(StragglerScoreType.INDIVIDUAL)
    score_type_set.add(source.derived(rule=MEGATRON_STRAGGLER_SCORE_TYPE))

    qual_set = SourceSet(StragglerQualifier.BEST)
    qual_set.add(source.derived(rule=MEGATRON_STRAGGLER_QUALIFIER_TYPE))

    group = MegatronStragglerGroup(ranks, qual_set, score_type_set)
    return [
        EventTriple(
            type="Application.MegatronStraggler.top",
            payload={"data": group},
            source=ranks[0].source,
        )
    ]


@collector_def(
    name="Collect.MegatronStraggler",
    formatter=lambda raw: {"megatron_straggler": raw},
    params_model=MegatronStragglerParams,
)
def _collect_megatron_straggler(event_stream, _job_info, params):
    data = MegatronStragglerParseState.collect(event_stream)
    return filter_megatron_stragglers(data, params)
