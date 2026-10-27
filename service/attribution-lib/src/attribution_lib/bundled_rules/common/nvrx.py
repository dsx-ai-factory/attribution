# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple, Optional, Self

from loguru import logger
from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import EventStream
from attribution_lib.exceptions import UndeclaredEventTypeError, UnknownResourceError
from attribution_lib.rules import RuleContext, RuleParamsBase
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


def resolve_reported_node(groups: dict[str, str], node: str, ctx: RuleContext) -> str:
    """Resolve an NVRx ``reported_node`` capture to a canonical node name.

    NVRx log lines may carry a bracketed reported node (e.g. ``[node042]``).
    When present it is verified against the job's node set, falling back to its
    lowercased form for an unknown node; when absent, the source node is used.

    Args:
        groups: Regex captures; an optional ``reported_node`` group is honored.
        node: Source node supplied by the observation reader.
        ctx: Rule context with optional job metadata.

    Returns:
        The resolved node name.
    """
    reported_node = groups.get("reported_node")
    if reported_node is None or reported_node == "":
        return node
    if ctx.job_info is not None:
        try:
            return ctx.job_info.verify_node(reported_node)
        except UnknownResourceError:
            return reported_node.lower()
    return reported_node.lower()


class NVRXStragglerParams(RuleParamsBase):
    """Per-request configuration for the ``Collect.NVRXStraggler`` collector."""

    straggler_threshold: float = 0.7
    score_types: list[StragglerScoreType] = [StragglerScoreType.RELATIVE]


class StragglerItem(NamedTuple):
    """Class to represent nodes reported by Straggler Detector."""

    rank: int
    reported_node: str
    node: Optional[str]
    score: float
    source: Source

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            rank=data["rank"],
            reported_node=data["reported_node"],
            node=data.get("node"),
            score=data["score"],
            source=Source.from_dict(data["source"]),
        )

    def get_node(self) -> str | None:
        """Get the node identifier, preferring resolved node over reported node."""
        if self.node is not None:
            return self.node
        if self.reported_node:
            logger.debug(f"NVRX rank {self.rank}: Node resolution failed, using reported_node '{self.reported_node}'")
            return self.reported_node
        logger.warning(f"Could not resolve node for NVRX rank {self.rank} - both node and reported_node are empty")
        return None


class StragglerGroup(NamedTuple):
    """A collection of StragglerItems (1 per node) logically grouped in Straggler Detector reports."""

    ranks: list[StragglerItem]
    total_ranks: int
    these_ranks: int
    qualifier: SourceSet[StragglerQualifier]
    score_type: SourceSet[StragglerScoreType]

    def is_complete(self) -> bool:
        """Return whether this group has received all of the ranks expected in the report."""
        return len(self.ranks) == self.these_ranks

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        ranks = [StragglerItem.from_dict(r) for r in data["ranks"]]
        qualifier: SourceSet[StragglerQualifier] = SourceSet.from_dict(data["qualifier"], StragglerQualifier)
        score_type: SourceSet[StragglerScoreType] = SourceSet.from_dict(data["score_type"], StragglerScoreType)

        return cls(
            ranks=ranks,
            total_ranks=data["total_ranks"],
            these_ranks=data["these_ranks"],
            qualifier=qualifier,
            score_type=score_type,
        )


class StragglerUnboundedGroup(NamedTuple):
    """A collection of StragglerItems (1 per node) logically grouped in Straggler Detector reports.
    This is for the corner case where the worst/best is not well-posed"""

    ranks: list[StragglerItem]
    score_type: SourceSet[StragglerScoreType]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        ranks = [StragglerItem.from_dict(r) for r in data["ranks"]]
        score_type: SourceSet[StragglerScoreType] = SourceSet.from_dict(data["score_type"], StragglerScoreType)

        return cls(ranks=ranks, score_type=score_type)


class StragglerParseState:
    """Collection-time reconstruction of NVRX Straggler Detector groups from events."""

    @classmethod
    def collect(cls, event_stream: EventStream) -> list[StragglerGroup | StragglerUnboundedGroup]:
        """Reconstruct straggler groups from raw per-line events in the event stream.

        Events can be emitted by an extension rule bundle:
          - Application.NVRXStraggler.header  (score type: individual/relative)
          - Application.NVRXStraggler.qualifier  (best/worst, count, max_count)
          - Application.NVRXStraggler.rank  (rank, reported_node, node, score)

        This method reconstructs the original group structure by walking events
        in line_num order, using the same state transitions as the original
        state machine but at collection time.
        """

        try:
            header_typ = event_stream.mgr.types.get_id("Application.NVRXStraggler.header")
            qualifier_typ = event_stream.mgr.types.get_id("Application.NVRXStraggler.qualifier")
            rank_typ = event_stream.mgr.types.get_id("Application.NVRXStraggler.rank")
        except UndeclaredEventTypeError:
            return []

        type_ids = [header_typ, qualifier_typ, rank_typ]

        results: list[StragglerGroup | StragglerUnboundedGroup] = []

        # State for reconstruction
        current_score_type: Optional[SourceSet[StragglerScoreType]] = None
        seen_score_types: set[StragglerScoreType] = set()
        seen_qualifiers: set[StragglerQualifier] = set()
        current_group: Optional[StragglerGroup | StragglerUnboundedGroup] = None
        last_rank_line: int = -1

        def _finalize_current():
            nonlocal current_group
            if current_group is None:
                return
            if isinstance(current_group, StragglerUnboundedGroup):
                if current_group.ranks:
                    results.append(current_group)
            elif isinstance(current_group, StragglerGroup):
                if current_group.is_complete():
                    results.append(current_group)
            current_group = None

        for _eid, evt_type, payload, source in query_events_with_source(
            event_stream, type_ids, order_by="e.line_num ASC"
        ):
            if source is None:
                continue

            if evt_type == header_typ:
                _finalize_current()
                score_type_val = StragglerScoreType.from_str(payload["class"])
                if score_type_val in seen_score_types:
                    current_score_type = None
                    continue
                score_type_set: SourceSet[StragglerScoreType] = SourceSet(score_type_val)
                score_type_set.add(source.derived(rule="NVRX Straggler Score Type"))
                current_score_type = score_type_set
                seen_score_types.add(score_type_val)
                seen_qualifiers.clear()
                last_rank_line = -1

            elif evt_type == qualifier_typ:
                if current_score_type is None:
                    continue
                _finalize_current()
                qualifier_val = StragglerQualifier.from_str(payload["class"])
                if len(seen_qualifiers) >= 2 or qualifier_val in seen_qualifiers:
                    current_score_type = None
                    continue
                seen_qualifiers.add(qualifier_val)
                qual_set: SourceSet[StragglerQualifier] = SourceSet(qualifier_val)
                qual_set.add(source.derived(rule="NVRX Straggler Qualifier Type"))
                current_group = StragglerGroup(
                    ranks=[],
                    total_ranks=int(payload["max_count"]),
                    these_ranks=int(payload["count"]),
                    qualifier=qual_set,
                    score_type=current_score_type,
                )
                last_rank_line = -1

            elif evt_type == rank_typ:
                if current_score_type is None:
                    continue

                line_num = source.line_num
                assert line_num is not None

                # For unbounded groups, check line adjacency to detect group end
                if isinstance(current_group, StragglerUnboundedGroup):
                    if last_rank_line >= 0 and line_num > last_rank_line + 1:
                        _finalize_current()
                        current_score_type = None
                        continue  # Orphan rank after gap

                if current_group is None:
                    # No qualifier seen after header -> unbounded body
                    current_group = StragglerUnboundedGroup(
                        ranks=[],
                        score_type=current_score_type,
                    )

                rule_name = (
                    "NVRX Straggler Unbounded Body"
                    if isinstance(current_group, StragglerUnboundedGroup)
                    else "NVRX Straggler Body"
                )
                item = StragglerItem(
                    rank=payload["rank"],
                    reported_node=payload["reported_node"],
                    node=payload.get("node"),
                    score=payload["score"],
                    source=source.derived(rule=rule_name),
                )
                current_group.ranks.append(item)
                last_rank_line = line_num

                if isinstance(current_group, StragglerGroup) and current_group.is_complete():
                    results.append(current_group)
                    current_group = None
                    if len(seen_qualifiers) == 2 and len(seen_score_types) == 2:
                        current_score_type = None

        _finalize_current()
        return results


def filter_nvrx_stragglers(
    groups: list[StragglerGroup | StragglerUnboundedGroup],
    params: NVRXStragglerParams,
) -> dict[str, StragglerNodeReport]:
    """Filter raw NVRX groups into per-node straggler reports.

    Only bounded ``StragglerGroup`` contributes; unbounded groups are ill-posed
    for thresholding and are dropped. A rank is a straggler iff its score is
    ``<=`` ``params.straggler_threshold`` and the group's score_type appears in
    ``params.score_types``.
    """
    res: dict[str, StragglerNodeReport] = {}
    for group in groups:
        if not isinstance(group, StragglerGroup):
            continue
        if group.score_type.value() not in params.score_types:
            continue

        group_sources = list(group.score_type) + list(group.qualifier)
        for item in group.ranks:
            if item.score > params.straggler_threshold:
                continue
            node = item.get_node()
            if node is None:
                continue
            report = res.setdefault(node, StragglerNodeReport(pattern="NVRX Pattern"))
            report.rank_scores.append(StragglerRankReport(rank=item.rank, score=item.score))
            report.sources.append(item.source)
            report.sources.extend(group_sources)
    return res
