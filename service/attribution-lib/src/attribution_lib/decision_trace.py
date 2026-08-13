# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Per-node, per-symptom decision trace extracted from AttributionResult."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from attribution_lib.core import AttributionResult
from attribution_lib.decision_trace_recorder import Action, DecisionRecorder, NodeType, SourceEvidence
from attribution_lib.exceptions import DecisionGraphError
from attribution_lib.sources import Source, SourceSet
from attribution_lib.symptoms.base import BaseSymptom


@dataclass(frozen=True)
class SourceDetail:
    """Single piece of evidence backing a symptom attribution."""

    observation_source: Optional[str]
    rule: Optional[str]
    timestamp: Optional[str]
    line_num: Optional[int]
    datum: Optional[str]

    @property
    def dedup_key(self) -> tuple:
        return (self.observation_source, self.rule, self.timestamp, self.line_num)


@dataclass
class SymptomDetail:
    """Provenance and cascade information for one symptom on one node."""

    status: str
    sources: list[SourceDetail]
    cascade_discards: list[str]

    def merge(self, other: "SymptomDetail") -> None:
        """Merge another SymptomDetail into this one, deduplicating sources."""
        seen = {s.dedup_key for s in self.sources}
        for s in other.sources:
            if s.dedup_key not in seen:
                seen.add(s.dedup_key)
                self.sources.append(s)
        self.sources.sort(key=lambda s: (s.timestamp or "", s.line_num or 0))
        if other.status == "bad":
            self.status = "bad"
        if other.cascade_discards and not self.cascade_discards:
            self.cascade_discards = other.cascade_discards


def get_short_node_name(name: str) -> str:
    """Strip FQDN suffixes to get short display name.

    Args:
        name: Full node hostname (e.g. "node-042.example.com").

    Returns:
        Short node name (e.g. "node-042").
    """
    return name.split(".", 1)[0]


def _to_source_detail(source: Source) -> SourceDetail:
    """Convert a Source NamedTuple to a SourceDetail."""
    return SourceDetail(
        observation_source=source.source_context.source.value if source.source_context else None,
        rule=source.rule,
        timestamp=source.time.isoformat() if source.time else None,
        line_num=source.line_num,
        datum=source.datum,
    )


def _get_cascade_discards(
    node: str,
    symptom: str,
    node_symptoms_base: dict[str, Any],
) -> list[str]:
    """Look up cascade_discards for a symptom from the DAG-pruned BaseSymptom view.

    Args:
        node: Node hostname (raw, pre-canonicalization).
        symptom: Symptom string to look up (e.g. "XID-48").
        node_symptoms_base: Per-node SymptomSet[BaseSymptom] from CalcBadNodesAttributionResult.

    Returns:
        List of symptom patterns this symptom suppresses, or empty list.
    """
    per_node_set = node_symptoms_base.get(node)
    if per_node_set is None:
        return []

    for base_symptom, _source_set in per_node_set.all_items():
        if isinstance(base_symptom, BaseSymptom) and base_symptom.symptom_id == symptom:
            return list(base_symptom.cascade_discards)

    return []


def _determine_status(source_set: SourceSet) -> str:
    """Determine symptom status from SourceSet bad/interesting state.

    Args:
        source_set: SourceSet containing bad and/or interesting sources.

    Returns:
        One of "bad", "interesting", or "inactive".
    """
    if source_set.has_bad():
        return "bad"
    if source_set.has_interesting():
        return "interesting"
    return "inactive"


def _collect_sources(source_set: SourceSet) -> list[SourceDetail]:
    """Collect, deduplicate, and sort all sources from a SourceSet.

    Args:
        source_set: SourceSet containing active and demoted symptom sources.

    Returns:
        List of SourceDetail sorted by (timestamp, line_num).
    """
    seen: set[tuple] = set()
    sources: list[SourceDetail] = []

    for source in source_set.all_sources():
        detail = _to_source_detail(source)
        if detail.dedup_key not in seen:
            seen.add(detail.dedup_key)
            sources.append(detail)

    sources.sort(key=lambda s: (s.timestamp or "", s.line_num or 0))
    return sources


def build_symptom_details(
    att_res: AttributionResult,
    canonical_from_normalized: Optional[Mapping[str, str]] = None,
) -> dict[str, dict[str, SymptomDetail]]:
    """Extract per-node, per-symptom provenance and cascade discard info.

    Args:
        att_res: Completed attribution result from the pipeline.
        canonical_from_normalized: Optional mapping from normalized to canonical node names.

    Returns:
        Mapping of short node name -> symptom string -> SymptomDetail.
    """
    cmap = canonical_from_normalized if canonical_from_normalized is not None else {}
    node_symptoms_base = att_res.bad_nodes_attr.node_symptoms_base

    details: dict[str, dict[str, SymptomDetail]] = {}

    for node, attr_group in att_res.bad_nodes_attr.my_bad_nodes.attributions.items():
        canonical_node = cmap.get(node, node)
        short_node = get_short_node_name(canonical_node)

        for symptom, source_set in attr_group.attributions.all_items():
            if source_set.is_void():
                continue

            detail = SymptomDetail(
                status=_determine_status(source_set),
                sources=_collect_sources(source_set),
                cascade_discards=_get_cascade_discards(node, symptom, node_symptoms_base),
            )

            node_dict = details.setdefault(short_node, {})
            if symptom in node_dict:
                node_dict[symptom].merge(detail)
            else:
                node_dict[symptom] = detail

    return details


def symptom_details_to_dict(
    details: dict[str, dict[str, SymptomDetail]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Serialize symptom details to a plain dict for JSON output."""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for node, symptoms in details.items():
        result[node] = {}
        for sym, detail in symptoms.items():
            result[node][sym] = asdict(detail)
    return result


@dataclass(frozen=True)
class DecisionNode:
    """A single node in the attribution decision graph."""

    node_id: str
    node_type: NodeType
    hostname: Optional[str]
    symptom: str
    rule_name: Optional[str]
    action: Action
    reason: Optional[str] = None
    datum: Optional[str] = None
    line_num: Optional[int] = None
    timestamp: Optional[str] = None
    obs_source: Optional[str] = None


@dataclass(frozen=True)
class DecisionEdge:
    """A directed edge between two DecisionNodes."""

    parent_id: str
    child_id: str
    edge_type: str


@dataclass(frozen=True)
class DecisionDAG:
    """Complete decision DAG (directed acyclic graph) for one job attribution.

    Edges always flow from earlier pipeline stages to later ones (extraction →
    filter/scrub/refiner → outcome) with no back-edges, guaranteeing acyclicity.
    This enables topological traversal, reachability queries, and partial
    re-evaluation when rules change.
    """

    nodes: list[DecisionNode]
    edges: list[DecisionEdge]


@dataclass(frozen=True)
class ExtractionKey:
    """Stable identity for deduplicating extraction evidence."""

    symptom: str
    hostname: Optional[str]
    obs_source: Optional[str]
    rule_name: Optional[str]
    timestamp: Optional[str]
    line_num: Optional[int]
    datum: Optional[str]

    @classmethod
    def from_source(cls, symptom: str, source: Source, hostname: Optional[str]) -> "ExtractionKey":
        evidence = SourceEvidence.from_source(source, hostname=hostname)
        return cls.from_evidence(symptom, evidence)

    @classmethod
    def from_evidence(cls, symptom: str, evidence: SourceEvidence) -> "ExtractionKey":
        return cls(
            symptom=symptom,
            hostname=evidence.hostname,
            obs_source=evidence.obs_source,
            rule_name=evidence.rule_name,
            timestamp=evidence.timestamp,
            line_num=evidence.line_num,
            datum=evidence.datum,
        )


def build_decision_graph(
    recorder: DecisionRecorder,
    att_res: AttributionResult,
    canonical_from_normalized: Optional[Mapping[str, str]] = None,
) -> DecisionDAG:
    """Assemble flat DecisionEvents into a directed acyclic graph (DAG).

    The algorithm has three phases:

    Phase 1 — Build extraction nodes from the final AttributionResult.
      Walk my_bad_nodes and node_symptoms_base to create one EXTRACTION node
      per (symptom, source) pair.  ``extraction_ids`` maps each symptom string
      to its node IDs so later phases can wire edges back to the right sources.

    Phase 2 — Convert recorder events into decision nodes.
      Outcome recorder events can be superseded by later reclassification, so
      they are used only to preserve extraction provenance.

    Phase 3 — Synthesize the authoritative final outcome from att_res.
      This keeps the graph consistent with post-outcome refiners.

    Returns domain dataclasses — callers handle presentation.
    """
    cmap = canonical_from_normalized if canonical_from_normalized is not None else {}
    nodes: list[DecisionNode] = []
    edges: list[DecisionEdge] = []
    node_counter = 0
    extraction_ids: dict[str, list[str]] = {}
    extraction_hosts: dict[str, Optional[str]] = {}
    _seen_extraction_sources: set[ExtractionKey] = set()

    def _add_extraction_node_from_evidence(
        *,
        symptom: str,
        evidence: SourceEvidence,
    ) -> None:
        nonlocal node_counter
        nid = f"ext_{node_counter}"
        node_counter += 1
        nodes.append(
            DecisionNode(
                node_id=nid,
                node_type=NodeType.EXTRACTION,
                hostname=evidence.hostname,
                symptom=symptom,
                rule_name=evidence.rule_name,
                action=Action.CREATED,
                datum=evidence.datum,
                line_num=evidence.line_num,
                timestamp=evidence.timestamp,
                obs_source=evidence.obs_source,
            )
        )
        extraction_ids.setdefault(symptom, []).append(nid)
        extraction_hosts[nid] = evidence.hostname

    def _add_extraction_node(symptom: str, source: Source, hostname: Optional[str]) -> None:
        evidence = SourceEvidence.from_source(source, hostname=hostname)
        source_key = ExtractionKey.from_source(symptom, source, hostname)
        if source_key in _seen_extraction_sources:
            return
        _seen_extraction_sources.add(source_key)
        _add_extraction_node_from_evidence(symptom=symptom, evidence=evidence)

    def _add_extraction_from_evidence(symptom: str, evidence: dict[str, Any], hostname: Optional[str]) -> None:
        source_evidence = SourceEvidence.from_payload(evidence, hostname=hostname)
        source_key = ExtractionKey.from_evidence(symptom, source_evidence)
        if source_key in _seen_extraction_sources:
            return
        _seen_extraction_sources.add(source_key)
        _add_extraction_node_from_evidence(symptom=symptom, evidence=source_evidence)

    def _matching_extraction_ids(symptom: str, hostname: Optional[str]) -> list[str]:
        ids = extraction_ids.get(symptom, [])
        if hostname is None:
            return ids
        matching = [ext_id for ext_id in ids if extraction_hosts.get(ext_id) in (hostname, None)]
        return matching

    def _ensure_extraction_for_event(
        symptom: str, hostname: Optional[str], event_evidence: Optional[dict[str, Any]]
    ) -> None:
        if not symptom:
            return
        if event_evidence is None:
            event_evidence = {}

        sources = event_evidence.get("sources", [])
        if not isinstance(sources, list):
            raise DecisionGraphError(f"Decision evidence sources for {symptom!r} must be a list")
        for source_evidence in sources:
            _add_extraction_from_evidence(symptom, source_evidence, hostname)
        if _matching_extraction_ids(symptom, hostname):
            return
        _add_extraction_from_evidence(symptom, event_evidence, hostname)

    # Phase 1a: extractions from my_bad_nodes (post-refinement symptom sources)
    for raw_node, attr_group in att_res.bad_nodes_attr.my_bad_nodes.attributions.items():
        canonical_node = cmap.get(raw_node, raw_node)
        for symptom, source_set in attr_group.attributions.all_items():
            if source_set.is_void():
                continue
            for source in source_set.all_sources():
                _add_extraction_node(symptom, source, canonical_node)

    # Phase 1b: extractions from node_symptoms_base (BaseSymptom sources not
    # already covered above — e.g. symptoms only in the pre-refinement view)
    for raw_node, per_node_set in att_res.bad_nodes_attr.node_symptoms_base.items():
        canonical_node = cmap.get(raw_node, raw_node)
        for base_symptom, base_source_set in per_node_set.all_items():
            if not isinstance(base_symptom, BaseSymptom):
                continue
            sid = base_symptom.symptom_id
            for source in base_source_set.all_sources():
                _add_extraction_node(sid, source, canonical_node)

    # Phase 1c: job-level symptoms are not always represented in per-node
    # provenance (for example raw keywords, SLURM metadata, or app-level
    # symptoms). Preserve their source path so rule/outcome nodes do not
    # become orphaned in the decision graph.
    for symptom, source_set in att_res.symptoms.all_items():
        if source_set.is_void():
            continue
        for source in source_set.all_sources():
            _add_extraction_node(symptom, source, None)

    # Phase 2: convert recorder events into decision/outcome nodes and
    # wire edges from extraction nodes to the decisions that reference them.
    for event in recorder.events:
        if event.node_type == NodeType.OUTCOME:
            # Outcome events can be superseded by post-outcome reclassification
            # refiners. The final OUTCOME node is always built from
            # att_res.outcome_attr below.
            _ensure_extraction_for_event(event.symptom, event.hostname, event.evidence)
            continue

        _ensure_extraction_for_event(event.symptom, event.hostname, event.evidence)
        nid = f"dec_{node_counter}"
        node_counter += 1
        nodes.append(
            DecisionNode(
                node_id=nid,
                node_type=event.node_type,
                hostname=event.hostname,
                symptom=event.symptom,
                rule_name=event.rule_name,
                action=event.action,
                reason=event.reason,
                datum=event.evidence.get("datum") if event.evidence else None,
                line_num=event.evidence.get("line_num") if event.evidence else None,
                timestamp=event.evidence.get("timestamp") if event.evidence else None,
                obs_source=event.evidence.get("obs_source") if event.evidence else None,
            )
        )
        edge_type = {Action.PRUNED: "PRUNES", Action.CREATED: "CREATES", Action.MATCHED: "MATCHES"}.get(
            event.action, "DEMOTES"
        )
        for ext_id in _matching_extraction_ids(event.symptom, event.hostname):
            edges.append(DecisionEdge(parent_id=ext_id, child_id=nid, edge_type=edge_type))

    # Phase 3: synthesize the authoritative outcome from the final
    # AttributionResult. Recorder outcome events may be intermediate because
    # post-outcome refiners can reclassify the attribution.
    outcome_val = att_res.outcome_attr.value()
    nid = f"outcome_{recorder.job_id}"
    nodes.append(
        DecisionNode(
            node_id=nid,
            node_type=NodeType.OUTCOME,
            hostname=None,
            symptom=outcome_val.root_cause_symptom or "",
            rule_name="symptom_to_outcome",
            action=Action.MATCHED,
            reason=f"{outcome_val.root_cause_symptom} → {outcome_val.outcome}",
        )
    )
    winning_symptom = outcome_val.root_cause_symptom
    if winning_symptom:
        if not _matching_extraction_ids(winning_symptom, None):
            for source in att_res.outcome_attr.all_sources():
                _add_extraction_node(winning_symptom, source, None)
        for ext_id in _matching_extraction_ids(winning_symptom, None):
            edges.append(DecisionEdge(parent_id=ext_id, child_id=nid, edge_type="MATCHES"))

    return DecisionDAG(nodes=nodes, edges=edges)
