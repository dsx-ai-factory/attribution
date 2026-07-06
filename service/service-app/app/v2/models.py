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
from typing import Annotated, Any, Literal, Optional

from attribution_lib.attr_types import JobNodeAttributions
from attribution_lib.core import AttributionResult
from attribution_lib.job_info import JobInfo, JobInfoBuilder
from attribution_lib.sources import (
    Source as LibSource,
    SourceSet,
)
from fastapi.encoders import jsonable_encoder
from pydantic import AwareDatetime, BaseModel, Field

from app.attribution import AttributionSource, Symptom
from app.attributor import (
    AttributionParameters,
    AttributorId,
    AttributorMetadata,
)
from app.exceptions import ResponseModel
from app.observation import ObservationId
from app.util import HashableModel, to_zoneinfo


class AttributorWorkloadInfoV2(HashableModel):
    """V2 wire shape: ``nodes`` is the explicit node->ranks mapping."""

    id: str
    type: str
    srun_cmd: Optional[str] = "dummy"
    job_start_time: AwareDatetime
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
            timezone=to_zoneinfo(w.job_start_time),
            node_to_ranks=w.nodes,
        )
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


class ResourcelessSource(BaseModel):
    """A source not bound to any specific node (e.g. a job-level signal)."""

    type: Literal["ResourcelessSource"] = Field(default="ResourcelessSource")
    source: AttributionSource


class NodeSource(BaseModel):
    """A source bound to a specific node (e.g. a per-node GPU or NIC symptom)."""

    type: Literal["NodeSource"] = Field(default="NodeSource")
    node: str
    source: AttributionSource


SymptomSite = Annotated[ResourcelessSource | NodeSource, Field(discriminator="type")]


class SourcedSymptom(BaseModel):
    """A symptom paired with the specific source that surfaced it."""

    symptom: Symptom
    source: AttributionSource


class Verdict(BaseModel):
    """Top-level attribution outcome: raw lib outcome, free-form detail, and the driving root-cause symptom if any."""

    root_symptom: SourcedSymptom | None
    attribution_outcome: str
    detail: str


def _index_nodes_by_source(my_bad_nodes: JobNodeAttributions) -> dict[tuple[Symptom, LibSource], list[str]]:
    """Invert the per-node attributions map into a (symptom, source) -> [nodes] lookup."""
    idx: defaultdict[tuple[Symptom, LibSource], list[str]] = defaultdict(list)
    for node, node_group in my_bad_nodes.items():
        for symptom, source_set in node_group.attributions.items():
            for lib_source in source_set:
                idx[(symptom, lib_source)].append(node)
    return idx


def _sites_for_symptom(
    symptom: Symptom,
    source_set: SourceSet[str],
    nodes_by_key: Mapping[tuple[Symptom, LibSource], list[str]],
) -> list[SymptomSite]:
    """Build the list of SymptomSite entries for one symptom, emitting NodeSource per
    (node, source) binding or ResourcelessSource when a source has no node attribution."""
    sites: list[SymptomSite] = []
    for lib_source in sorted(source_set, key=_source_sort_key):
        attr_source = AttributionSource.from_lib(lib_source)
        nodes = nodes_by_key.get((symptom, lib_source))
        if nodes is None:
            sites.append(ResourcelessSource(source=attr_source))
            continue
        for node in sorted(nodes):
            sites.append(NodeSource(node=node, source=attr_source))
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
    ) -> "AttributionResponseV2":
        """Build a V2 response from the lib's AttributionResult."""
        return cls(
            verdict=cls._build_verdict(result),
            symptoms=cls._build_symptoms(result),
            observation_context=sorted(ingested_observations),
            collectors=jsonable_encoder(result.collectors, custom_encoder=_LIB_TYPES_ENCODER),
        )

    @staticmethod
    def _build_verdict(result: AttributionResult) -> Verdict:
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
            )

        return Verdict(
            root_symptom=root_symptom,
            attribution_outcome=outcome.outcome,
            detail=detail,
        )

    @staticmethod
    def _build_symptoms(result: AttributionResult) -> dict[Symptom, list[SymptomSite]]:
        """Construct the symptoms map from the lib's AttributionResult."""
        nodes_by_key = _index_nodes_by_source(result.bad_nodes_attr.my_bad_nodes)
        out: dict[Symptom, list[SymptomSite]] = {}
        for symptom, source_set in sorted(result.symptoms.items()):
            sites = _sites_for_symptom(symptom, source_set, nodes_by_key)
            if len(sites) > 0:
                out[symptom] = sites
        return out
