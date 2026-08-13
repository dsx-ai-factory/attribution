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
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Self

from loguru import logger
from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import EventStream, sql_placeholders
from attribution_lib.exceptions import UndeclaredEventTypeError
from attribution_lib.rules_engine import RuleParamsBase
from attribution_lib.sources import (
    Source,
    SourceSet,
)

from .straggler_types import (
    StragglerNodeReport,
    StragglerQualifier,
    StragglerRankReport,
    StragglerScoreType,
)

# Constants for Megatron Straggler Detector string literals
MEGATRON_STRAGGLER_QUALIFIER_TYPE = "Megatron Straggler Qualifier Type"
MEGATRON_STRAGGLER_SCORE_TYPE = "Megatron Straggler Score Type"
MEGATRON_STRAGGLER_BOTTOM_RANKS = "Megatron Straggler Bottom Ranks"
MEGATRON_STRAGGLER_TOP_RANKS = "Megatron Straggler Top Ranks"


class MegatronStragglerParams(RuleParamsBase):
    """Parameters to guide Megatron stragglers"""

    straggler_threshold: float = 0.95
    score_types: list[StragglerScoreType] = [StragglerScoreType.INDIVIDUAL]
    infer_threshold_from_data: bool = True


@dataclass
class MegatronStragglerData:
    """Consolidated data structure for Megatron Straggler Detector state."""

    reports: list["MegatronStragglerGroup"] = field(default_factory=list)
    top_scores: list[float] = field(default_factory=list)
    bottom_scores: list[float] = field(default_factory=list)

    def update(self, other: "MegatronStragglerData") -> None:
        """Update this data structure with data from another."""
        self.reports.extend(other.reports)
        self.top_scores.extend(other.top_scores)
        self.bottom_scores.extend(other.bottom_scores)


class MegatronStragglerItem(NamedTuple):
    """Class to represent nodes reported by Megatron Straggler Detector."""

    rank: int
    node: Optional[str]
    score: float
    source: Source

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Reconstruct MegatronStragglerItem from dict."""
        return cls(
            rank=data["rank"],
            node=data.get("node"),
            score=data["score"],
            source=Source.from_dict(data["source"]),
        )

    def get_node(self) -> str | None:
        """Get the node identifier, returning None if node cannot be resolved."""
        if self.node is not None:
            return self.node
        logger.warning(f"Could not resolve node for Megatron rank {self.rank}")
        return None


class MegatronStragglerGroup(NamedTuple):
    """A collection of MegatronStragglerItems logically grouped in Megatron Straggler Detector reports."""

    ranks: list[MegatronStragglerItem]
    qualifier: SourceSet[StragglerQualifier]
    score_type: SourceSet[StragglerScoreType]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Reconstruct MegatronStragglerGroup from dict."""
        ranks = [MegatronStragglerItem.from_dict(r) for r in data["ranks"]]
        qualifier: SourceSet[StragglerQualifier] = SourceSet.from_dict(data["qualifier"], StragglerQualifier)
        score_type: SourceSet[StragglerScoreType] = SourceSet.from_dict(data["score_type"], StragglerScoreType)

        return cls(
            ranks=ranks,
            qualifier=qualifier,
            score_type=score_type,
        )


class MegatronStragglerParseState:
    """Collection-time reconstruction of Megatron Straggler Detector groups from events."""

    @classmethod
    def collect(cls, event_stream: EventStream) -> MegatronStragglerData:
        top_typ = event_stream.mgr.types.get_id("Application.MegatronStraggler.top")
        try:
            bottom_typ = event_stream.mgr.types.get_id("Application.MegatronStraggler.bottom")
        except UndeclaredEventTypeError:
            bottom_typ = None

        type_ids = [top_typ]
        if bottom_typ is not None:
            type_ids.append(bottom_typ)

        type_ph = sql_placeholders(len(type_ids))
        rows = event_stream.query_raw(
            f"SELECT event_type, payload FROM events WHERE context_id=? AND event_type IN ({type_ph}) AND status=1 ORDER BY line_num ASC",
            [event_stream.ctx_id] + type_ids,
        )

        res = MegatronStragglerData()
        for evt_type, payload_json in rows:
            if payload_json is None:
                continue
            payload = json.loads(payload_json)

            # Reconstruct typed object from dict
            data = MegatronStragglerGroup.from_dict(payload["data"])
            scores = [i.score for i in data.ranks]

            if evt_type == top_typ:
                res.top_scores.extend(scores)
            elif bottom_typ is not None and evt_type == bottom_typ:
                res.bottom_scores.extend(scores)

            res.reports.append(data)
        return res


def filter_megatron_stragglers(
    data: MegatronStragglerData,
    params: MegatronStragglerParams,
) -> dict[str, StragglerNodeReport]:
    """Filter Megatron groups into per-node straggler reports.

    Megatron emits Bottom/Top group pairs in sequence. For each Bottom group:
    find the next group (must be Top with matching score_type) and use it to
    compute the per-event threshold when ``infer_threshold_from_data`` is set.
    A rank in the Bottom group is a straggler iff its score is strictly less
    than the event threshold.
    """
    res: dict[str, StragglerNodeReport] = {}
    groups = data.reports
    i = 0
    while i < len(groups):
        group = groups[i]
        if not _is_bottom_in_scope(group, params):
            i += 1
            continue
        top_group = _paired_top_group(groups, i, params)
        if top_group is None and params.infer_threshold_from_data:
            i += 1
            continue
        event_threshold = _megatron_event_threshold(top_group, params)
        _accumulate_megatron_node_reports(group, event_threshold, res)
        i += 1
    return res


def _is_bottom_in_scope(group: MegatronStragglerGroup, params: MegatronStragglerParams) -> bool:
    """True iff the group's score_type is configured AND its qualifier is WORST."""
    if group.score_type.value() not in params.score_types:
        return False
    return group.qualifier.value() == StragglerQualifier.WORST


def _paired_top_group(
    groups: list[MegatronStragglerGroup], index: int, params: MegatronStragglerParams
) -> Optional[MegatronStragglerGroup]:
    """Return the BEST-qualifier group paired with the WORST at ``index``, else None.

    The candidate must share the same ``score_type`` as the bottom group; otherwise
    inferring the threshold would mix unrelated score scales.
    """
    if index + 1 >= len(groups):
        return None
    bottom = groups[index]
    nxt = groups[index + 1]
    if (
        nxt.score_type.value() == bottom.score_type.value()
        and nxt.score_type.value() in params.score_types
        and nxt.qualifier.value() == StragglerQualifier.BEST
    ):
        return nxt
    return None


def _megatron_event_threshold(top_group: Optional[MegatronStragglerGroup], params: MegatronStragglerParams) -> float:
    """Per-event threshold: either inferred from top scores, or the static configured threshold."""
    if top_group is None or not params.infer_threshold_from_data:
        return params.straggler_threshold
    top_scores = [item.score for item in top_group.ranks]
    if not top_scores:
        return params.straggler_threshold
    return max(top_scores) * params.straggler_threshold


def _accumulate_megatron_node_reports(
    group: MegatronStragglerGroup,
    event_threshold: float,
    res: dict[str, StragglerNodeReport],
) -> None:
    """Accumulate per-node straggler reports for ranks scoring below ``event_threshold``."""
    group_sources = list(group.score_type) + list(group.qualifier)
    for item in group.ranks:
        if item.score >= event_threshold:
            continue
        node = item.get_node()
        if node is None:
            continue
        report = res.setdefault(node, StragglerNodeReport(pattern="Megatron Pattern"))
        report.rank_scores.append(StragglerRankReport(rank=item.rank, score=item.score))
        report.sources.append(item.source)
        report.sources.extend(group_sources)


def get_pytorch_timeout_minutes(event_stream: EventStream) -> Optional[SourceSet[float]]:
    typ_id = event_stream.mgr.types.get_id("Application.megatron-lm.runtime_args")

    for _eid, _evt_type, payload, source in query_events_with_source(event_stream, typ_id):
        if not payload:
            continue

        possible_payload_keys = ["pytorch_timeout_minutes", "distributed_timeout_minutes"]
        found_key = None
        for k in possible_payload_keys:
            if k in payload:
                found_key = k
                break

        if found_key is None:
            continue

        return SourceSet(float(payload[found_key])).add_interesting(source)

    return None
