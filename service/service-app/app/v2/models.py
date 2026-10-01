# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from enum import Enum, StrEnum
from typing import Annotated, Any, Literal, Optional
from zoneinfo import ZoneInfo

from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.compound_outcome import (
    CancellationActor,
    FailureDomain,
    TerminalState,
    UncertaintyType,
    compound_outcome_for,
)
from attribution_lib.core import AttributionResult
from attribution_lib.decision_trace import (
    DecisionDAG as LibDecisionDAG,
    DecisionEdge as LibDecisionEdge,
    DecisionNode as LibDecisionNode,
)
from attribution_lib.decision_trace_recorder import (
    Action as LibAction,
    NodeType as LibNodeType,
)
from attribution_lib.job_info import JobInfo, JobInfoBuilder, Scheduler
from attribution_lib.observation.inbound import Id as ObservationId
from attribution_lib.sources import (
    Source as LibSource,
    SourceSet,
)
from fastapi.encoders import jsonable_encoder
from loguru import logger
from pydantic import AwareDatetime, BaseModel, Field

from app.attribution import AttributionSource, Symptom
from app.attributor import (
    AttributionParameters,
    AttributorId,
    AttributorMetadata,
)
from app.exceptions import DAG_SHAPE_ERRORS, ResponseModel
from app.util import HashableModel


class AttributorWorkloadInfoV2(HashableModel):
    """V2 wire shape: ``nodes`` is the explicit node->ranks mapping."""

    id: str
    type: str
    # Explicit scheduler identity (e.g. "slurm", "kueue").
    scheduler: Optional[Scheduler] = None
    srun_cmd: Optional[str] = "dummy"
    job_start_time: AwareDatetime
    timezone: ZoneInfo
    job_end_time: Optional[AwareDatetime] = None
    interest_start_time: Optional[AwareDatetime] = None
    interest_end_time: Optional[AwareDatetime] = None
    exit_code_signal: Optional[int] = None
    cancelled_by: Optional[str] = None
    time_limit: Optional[int] = None
    failed_node: Optional[str] = None
    status: str
    nodes: dict[str, list[int]]
    name: str
    username: str
    last_tflops_time: Optional[datetime] = None
    framework: str


class AttributorInfoV2(HashableModel):
    """V2 wire shape for the /attributor POST body."""

    workload: AttributorWorkloadInfoV2
    metadata: AttributorMetadata
    parameters: AttributionParameters = Field(default_factory=AttributionParameters)

    def to_lib_jobinfo(self) -> JobInfo:
        """Translate the V2 wire payload into a lib ``JobInfo`` for attribution."""
        w = self.workload
        builder = JobInfoBuilder(
            id=w.id,
            framework=w.framework,
            type=w.type,
            cluster=self.metadata.cluster,
            state=w.status,
            job_start_time=w.job_start_time,
            source_name="attributor service translation",
            timezone=w.timezone,
            node_to_ranks=w.nodes,
        )
        if w.scheduler is not None:
            builder.scheduler(w.scheduler)
        if w.interest_start_time:
            builder.interest_start_time(w.interest_start_time)
        if w.interest_end_time:
            builder.interest_end_time(w.interest_end_time)
        if w.srun_cmd is not None:
            builder.srun_cmd(w.srun_cmd)
        if w.cancelled_by is not None:
            builder.cancelled_by(w.cancelled_by)
        if w.last_tflops_time is not None:
            builder.last_tflops_time(w.last_tflops_time)
        if w.job_end_time is not None:
            builder.job_end_time(w.job_end_time)
        if w.exit_code_signal is not None:
            builder.exit_code_signal(w.exit_code_signal)
        if w.time_limit is not None:
            builder.time_limit(w.time_limit)
        if w.failed_node is not None:
            builder.failed_node(w.failed_node)
        return builder.build()


class NewResponseV2(ResponseModel):
    """Response to creating a new attribution attributor."""

    attributor_id: AttributorId


class ObservationResponseV2(ResponseModel):
    """Response to performing an ingestion."""

    observation_id: ObservationId


def _source_sort_key(s: LibSource) -> tuple:
    """Sort sources by (obs_id, line_num, rule, datum) for deterministic ordering."""
    return (
        s.source_context.id,
        s.line_num if s.line_num is not None else -1,
        s.rule or "",
        s.datum or "",
    )


class SymptomStatus(StrEnum):
    """Whether a source still drives attribution, or was demoted to supporting evidence."""

    BAD = "bad"
    INTERESTING = "interesting"


class ResourcelessSource(BaseModel):
    """A source not bound to any specific node (e.g. a job-level signal)."""

    type: Literal["ResourcelessSource"] = Field(default="ResourcelessSource")
    status: SymptomStatus
    source: AttributionSource


class NodeSource(BaseModel):
    """A source bound to a specific node (e.g. a per-node GPU or NIC symptom)."""

    type: Literal["NodeSource"] = Field(default="NodeSource")
    node: str
    status: SymptomStatus
    source: AttributionSource


SymptomSite = Annotated[ResourcelessSource | NodeSource, Field(discriminator="type")]


class SourcedSymptom(BaseModel):
    """A symptom paired with the specific source that surfaced it."""

    symptom: Symptom
    source: AttributionSource
    # Symptoms with no doc slug (Pass/Cancelled) carry None.
    catalog_id: Optional[str] = Field(
        default=None,
        description=(
            "Documentation slug identifying the failure-catalog entry this symptom maps to. "
            "A reference into that catalog, not a guaranteed-unique key per symptom."
        ),
    )


class DecisionNodeType(str, Enum):
    """Attribution pipeline stage a decision node was produced by."""

    UNKNOWN = "UNKNOWN"
    EXTRACTION = "EXTRACTION"
    FILTER = "FILTER"
    SCRUB = "SCRUB"
    REFINER = "REFINER"
    CASCADE_PRUNE = "CASCADE_PRUNE"
    OUTCOME = "OUTCOME"

    @classmethod
    def from_lib(cls, node_type: LibNodeType) -> "DecisionNodeType":
        """Map a lib node type onto the V2 wire enum.

        Args:
            node_type: Library node type.

        Returns:
            Matching V2 node type, or UNKNOWN for a stage this schema predates.
        """
        try:
            return cls(node_type.value)
        except ValueError:
            return cls.UNKNOWN


class DecisionAction(str, Enum):
    """Effect a decision node had on the symptom it references."""

    UNKNOWN = "UNKNOWN"
    CREATED = "CREATED"
    MATCHED = "MATCHED"
    REMOVED = "REMOVED"
    PRUNED = "PRUNED"

    @classmethod
    def from_lib(cls, action: LibAction) -> "DecisionAction":
        """Map a lib action onto the V2 wire enum.

        Args:
            action: Library action.

        Returns:
            Matching V2 action, or UNKNOWN for an action this schema predates.
        """
        try:
            return cls(action.value)
        except ValueError:
            return cls.UNKNOWN


class DecisionNode(BaseModel):
    """One evidence, rule-action, or outcome node in a runtime Decision DAG."""

    node_id: str
    node_type: DecisionNodeType
    hostname: Optional[str]
    symptom: str
    rule_name: Optional[str]
    action: DecisionAction
    reason: Optional[str] = None
    datum: Optional[str] = None
    line_num: Optional[int] = None
    timestamp: Optional[str] = None
    obs_source: Optional[str] = None

    @classmethod
    def from_lib(cls, node: LibDecisionNode) -> "DecisionNode":
        """Convert an attribution-lib decision node to its wire model.

        Args:
            node: Library decision node.

        Returns:
            V2 decision node.
        """
        return cls(
            node_id=node.node_id,
            node_type=DecisionNodeType.from_lib(node.node_type),
            hostname=node.hostname,
            symptom=node.symptom,
            rule_name=node.rule_name,
            action=DecisionAction.from_lib(node.action),
            reason=node.reason,
            datum=node.datum,
            line_num=node.line_num,
            timestamp=node.timestamp,
            obs_source=node.obs_source,
        )


class DecisionEdge(BaseModel):
    """One directed relationship in a runtime Decision DAG."""

    parent_id: str
    child_id: str
    edge_type: str

    @classmethod
    def from_lib(cls, edge: LibDecisionEdge) -> "DecisionEdge":
        """Convert an attribution-lib decision edge to its wire model.

        Args:
            edge: Library decision edge.

        Returns:
            V2 decision edge.
        """
        return cls(
            parent_id=edge.parent_id,
            child_id=edge.child_id,
            edge_type=edge.edge_type,
        )


class DecisionGraph(BaseModel):
    """Per-attribution runtime Decision DAG."""

    nodes: list[DecisionNode]
    edges: list[DecisionEdge]

    @classmethod
    def from_lib(cls, graph: LibDecisionDAG) -> "DecisionGraph":
        """Convert an attribution-lib Decision DAG to its wire model.

        Args:
            graph: Library Decision DAG.

        Returns:
            V2 decision graph.
        """
        return cls(
            nodes=[DecisionNode.from_lib(node) for node in graph.nodes],
            edges=[DecisionEdge.from_lib(edge) for edge in graph.edges],
        )


def _wire_decision_graph(graph: Optional[LibDecisionDAG]) -> Optional[DecisionGraph]:
    """Convert a lib Decision DAG for the wire without risking the verdict.

    Conversion reads every node and edge of a lib-owned structure, so a lib-side
    shape change can raise even though the verdict is sound. The graph is a
    diagnostic add-on and is dropped rather than failing the response.

    Args:
        graph: Library Decision DAG, or None when no graph was requested.

    Returns:
        The V2 decision graph, or None when unrequested or not convertible.
    """
    if graph is None:
        return None
    try:
        return DecisionGraph.from_lib(graph)
    except DAG_SHAPE_ERRORS as e:
        logger.warning(f"Failed to convert decision graph, omitting it from the verdict: {e}")
        return None


class _UnsetType:
    """Marker for a response field the service leaves off the wire entirely."""


_UNSET = _UnsetType()


class Verdict(BaseModel):
    """Top-level attribution outcome: raw lib outcome, free-form detail, and the driving root-cause symptom if any."""

    root_symptom: SourcedSymptom | None
    attribution_outcome: str
    detail: str
    # Additive compound-outcome fields: a typed decomposition of
    # attribution_outcome into terminal state / failure domain / cancellation actor /
    # uncertainty reason. attribution_outcome remains the source of truth and is
    # unchanged; these are a static translation of it. Field order is not semantically
    # meaningful; lists are for wire transport, not to imply ranking.
    terminal_states: list[TerminalState] = Field(
        min_length=1,
        description=(
            "Plausible terminal states in NVIDIA Attribution's assessment. Values are alternatives, with no ranking "
            "or probabilities; omitted states are ruled out by that assessment. A singleton does not "
            "guarantee correctness, and the number of values is not a confidence score."
        ),
    )
    failure_domains: list[FailureDomain] = Field(
        default=[],
        description=(
            "Plausible failure domains, conditional on Fail. Values express alternatives, not simultaneous "
            "causes or equal probabilities. Empty means failure is absent or its domain is unspecified; "
            "consult terminal_states to distinguish these cases."
        ),
    )
    cancellation_actors: list[CancellationActor] = Field(
        default=[],
        description=(
            "Plausible initiators, conditional on Cancelled. Values express alternatives without ranking "
            "or probabilities. Empty means cancellation is absent or its actor is unspecified; "
            "consult terminal_states to distinguish these cases."
        ),
    )
    uncertainty_type: Optional[UncertaintyType] = Field(
        default=None,
        description=(
            "Qualitative source of uncertainty using NVIDIA Attribution's operational definitions: Aleatoric means "
            "ambiguous or insufficient job evidence; Epistemic means limitations of NVIDIA Attribution, such as rule "
            "gaps or bugs. This is not a probability, confidence score, or error rate. Null means no "
            "source is reported, not guaranteed correctness. The source is not inferred from set size."
        ),
    )
    # Defaults are not validated, so the sentinel survives as the field's value and is
    # dropped by exclude_if, leaving an explicit None to mean "requested but unavailable".
    decision_graph: Optional[DecisionGraph] = Field(
        default=_UNSET,  # type: ignore[assignment]
        description=(
            "Runtime Decision DAG: the graph when requested and available, null when requested "
            "but unavailable, absent when not requested."
        ),
        exclude_if=lambda graph: graph is _UNSET,
    )


def _split_sources(source_set: SourceSet[str]) -> tuple[set[LibSource], set[LibSource]]:
    """Split a source set into its bad and its interesting-only sources."""
    bad = set(source_set)
    return bad, set(source_set.all_sources()) - bad


def _index_nodes_by_source(
    my_bad_nodes: JobNodeAttributions,
    include_interesting: bool,
) -> dict[tuple[Symptom, LibSource], list[tuple[str, SymptomStatus]]]:
    """Invert the per-node attributions map into a (symptom, source) -> [(node, status)] lookup."""
    idx: defaultdict[tuple[Symptom, LibSource], list[tuple[str, SymptomStatus]]] = defaultdict(list)
    for node, node_group in my_bad_nodes.items():
        attributions = node_group.attributions
        symptom_items = attributions.all_items() if include_interesting else attributions.items()
        for symptom, source_set in symptom_items:
            bad, interesting = _split_sources(source_set)
            for lib_source in bad:
                idx[(symptom, lib_source)].append((node, SymptomStatus.BAD))
            if include_interesting:
                for lib_source in interesting:
                    idx[(symptom, lib_source)].append((node, SymptomStatus.INTERESTING))
    return idx


def _sites_for_symptom(
    symptom: Symptom,
    source_set: SourceSet[str],
    nodes_by_key: Mapping[tuple[Symptom, LibSource], list[tuple[str, SymptomStatus]]],
    include_interesting: bool,
) -> list[SymptomSite]:
    """Build the list of SymptomSite entries for one symptom, emitting NodeSource per
    (node, source) binding or ResourcelessSource when a source has no node attribution."""
    sites: list[SymptomSite] = []
    bad, interesting = _split_sources(source_set)
    candidates = bad | interesting if include_interesting else bad
    for lib_source in sorted(candidates, key=_source_sort_key):
        attr_source = AttributionSource.from_lib(lib_source)
        bindings = nodes_by_key.get((symptom, lib_source), [])
        has_bad_node_binding = any(status is SymptomStatus.BAD for _node, status in bindings)
        # Job's verdict, kept even if the only node that reported it demoted its copy.
        if lib_source in bad and not has_bad_node_binding:
            sites.append(ResourcelessSource(status=SymptomStatus.BAD, source=attr_source))
        elif len(bindings) == 0:
            sites.append(ResourcelessSource(status=SymptomStatus.INTERESTING, source=attr_source))
        for node, node_status in sorted(bindings):
            sites.append(NodeSource(node=node, status=node_status, source=attr_source))
    return sites


_LIB_TYPES_ENCODER: dict[type, Any] = {
    LibSource: lambda v: AttributionSource.from_lib(v).model_dump(mode="json"),
}


class AttributionResponseV2(ResponseModel):
    """V2 response to GET /v2/attributor/{att_id}/attribution."""

    verdict: Verdict
    symptoms: dict[Symptom, list[SymptomSite]]
    observation_context: list[ObservationId]
    collectors: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_lib_attribution(
        cls,
        ingested_observations: set[ObservationId],
        result: AttributionResult,
        decision_graph: Optional[LibDecisionDAG] = None,
        decision_graph_requested: bool = False,
        include_interesting: bool = False,
    ) -> "AttributionResponseV2":
        """Build a V2 response from the lib's AttributionResult."""
        return cls(
            verdict=cls._build_verdict(result, decision_graph, decision_graph_requested),
            symptoms=cls._build_symptoms(result, include_interesting),
            observation_context=sorted(ingested_observations),
            collectors=jsonable_encoder(result.collectors, custom_encoder=_LIB_TYPES_ENCODER),
        )

    @staticmethod
    def _build_verdict(
        result: AttributionResult,
        decision_graph: Optional[LibDecisionDAG] = None,
        decision_graph_requested: bool = False,
    ) -> Verdict:
        """Construct the verdict from the lib's AttributionResult."""
        outcome = result.outcome_attr.value()
        # Despite its name, `hw_cause` is the lib's universal outcome-detail
        # field, populated by rules across every outcome class (User, SW,
        # MultiCategory, Pass, etc.), not just HW.
        detail = outcome.hw_cause

        driver_source = result.outcome_attr.one_source()
        root_symptom: SourcedSymptom | None = None
        if outcome.root_cause_symptom != "" and driver_source is not None:
            root_symptom = SourcedSymptom(
                symptom=outcome.root_cause_symptom,
                source=AttributionSource.from_lib(driver_source),
                catalog_id=outcome.doc,
            )

        compound = compound_outcome_for(outcome.outcome)

        # Only pass decision_graph when the caller asked for it: an unset field is omitted
        # from the response, while an explicit None reports a graph we could not supply.
        fields: dict[str, Any] = {
            "root_symptom": root_symptom,
            "attribution_outcome": outcome.outcome,
            "detail": detail,
            "terminal_states": sorted(compound.terminal_states),
            "failure_domains": sorted(compound.failure_domains),
            "cancellation_actors": sorted(compound.cancellation_actors),
            "uncertainty_type": compound.uncertainty,
        }
        if decision_graph_requested:
            fields["decision_graph"] = _wire_decision_graph(decision_graph)
        return Verdict(**fields)

    @staticmethod
    def _build_symptoms(
        result: AttributionResult,
        include_interesting: bool = False,
    ) -> dict[Symptom, list[SymptomSite]]:
        """Construct the symptoms map from the lib's AttributionResult."""
        nodes_by_key = _index_nodes_by_source(result.bad_nodes_attr.my_bad_nodes, include_interesting)
        out: dict[Symptom, list[SymptomSite]] = {}
        symptom_items = result.symptoms.all_items() if include_interesting else result.symptoms.items()
        for symptom, source_set in sorted(symptom_items):
            sites = _sites_for_symptom(symptom, source_set, nodes_by_key, include_interesting)
            if len(sites) > 0:
                out[symptom] = sites
        return out
