# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime
from typing import Annotated, Literal, NamedTuple, Optional, Union

from attribution_lib.attr_types import NodeAttributionGroup
from attribution_lib.core import AttributionResult
from attribution_lib.job_info import JobInfo, JobInfoBuilder, Scheduler
from attribution_lib.observation.inbound import Id as ObservationId
from attribution_lib.outcome import calculate_outcome
from attribution_lib.sources import SourceSet, SymptomSet
from loguru import logger
from pydantic import AwareDatetime, BaseModel, Field

from app.attribution import (
    AttributionClassEnum,
    AttributionSource,
    RecommendationEnum,
    SingleAttribution,
    SingleAttributionBase,
    Symptom,
)
from app.attributor import (
    AttributionParameters,
    AttributorId,
    AttributorMetadata,
    WorkloadTypeEnum,
)
from app.exceptions import ResponseModel
from app.util import HashableModel, to_zoneinfo

# Legacy V1 `workload.type` values that name a scheduler. 
_V1_TYPE_TO_SCHEDULER: dict[WorkloadTypeEnum, Scheduler] = {
    WorkloadTypeEnum.SLURM: Scheduler.SLURM,
    WorkloadTypeEnum.KUEUE: Scheduler.KUEUE,
}


class AttributorWorkloadInfoV1(HashableModel):
    """V1 wire shape: legacy positional node list + ranks-per-node fan-out, or an explicit
    node-to-ranks mapping. If both are sent, the explicit mapping wins (see to_lib_jobinfo)."""

    id: str
    type: WorkloadTypeEnum
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
    nranks: Optional[int] = None
    ranks_per_node: Optional[int] = None
    nodes: Optional[list[str]] = None
    node_to_ranks: Optional[dict[str, list[int]]] = None
    name: str
    username: str
    last_tflops_time: Optional[datetime] = None
    framework: str


class AttributorInfoV1(HashableModel):
    """V1 wire shape for the /attributor POST body."""

    workload: AttributorWorkloadInfoV1
    metadata: AttributorMetadata
    parameters: AttributionParameters = Field(default_factory=AttributionParameters)

    def to_lib_jobinfo(self) -> JobInfo:
        """Translate the V1 wire payload into a lib ``JobInfo`` for attribution."""
        w = self.workload
        # Explicit node_to_ranks (the v1 optional field) takes precedence; otherwise synthesize the
        # uniform mapping from positional nodes + ranks_per_node. The lib has a single node_to_ranks
        # input now, so there is no "both supplied" conflict to surface (explicit simply wins).
        if w.node_to_ranks is not None:
            node_to_ranks = w.node_to_ranks
        elif w.nodes is not None and w.ranks_per_node is not None:
            node_to_ranks = JobInfoBuilder.uniform_node_to_ranks(w.nodes, w.ranks_per_node)
        elif w.nodes is not None:
            # nodes without ranks_per_node: keep as empty-rank members, don't drop the list.
            node_to_ranks = {n: [] for n in w.nodes}
        else:
            node_to_ranks = {}
        builder = JobInfoBuilder(
            id=w.id,
            framework=w.framework,
            type=w.type,
            cluster=self.metadata.cluster,
            state=w.status,
            job_start_time=w.job_start_time,
            source_name="attributor service translation",
            timezone=to_zoneinfo(w.job_start_time),
            node_to_ranks=node_to_ranks,
        )
        # V1 has no `workload.scheduler` wire field so derive the scheduler from the 
        # legacy `type` here.
        scheduler = _V1_TYPE_TO_SCHEDULER.get(w.type)
        if scheduler is not None:
            builder.scheduler(scheduler)
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


# Straggler/workload/node/storage/switch/global shapes kept out of shared models
# so V2 (and any future version) can stay rule-agnostic.


class WorkloadRecord(BaseModel):
    """A class for non-failure related observations"""

    type: Literal["WorkloadRecord"] = Field(default="WorkloadRecord")
    key: str
    val: str
    sources: list[AttributionSource]


class RankScore(NamedTuple):
    """A rank and its associated score."""

    rank: int
    score: float


class StragglerAttribution(SingleAttributionBase):
    """A node-specific attribution"""

    type: Literal["StragglerAttribution"] = Field(default="StragglerAttribution")
    symptom: Literal["Straggler"] = Field(default="Straggler")
    description: str = Field(default="Straggler Pattern")  # Dynamic description based on source
    attribution_class: AttributionClassEnum = AttributionClassEnum.UNDETERMINED
    recommendation: RecommendationEnum = RecommendationEnum.AMBIGUOUS
    rank_scores: list[RankScore] = []


SingleAttributionTypes = Annotated[
    Union[SingleAttribution, StragglerAttribution],
    Field(discriminator="type"),
]


class NodeAttribution(BaseModel):
    """A node-specific attribution"""

    type: Literal["NodeAttribution"] = Field(default="NodeAttribution")
    attributions: list[SingleAttributionTypes]
    node: str

    @classmethod
    def from_lib_node_group(
        cls,
        node_attr: NodeAttributionGroup,
    ) -> "NodeAttribution":
        attributions = []
        for _symptom, source_set in node_attr.attributions._data.items():
            if source_set.has_bad():
                tmp: SymptomSet[Symptom] = SymptomSet()
                tmp.add(source_set)
                outcome = calculate_outcome(tmp)
                att_class = AttributionClassEnum.from_lib_outcomes(outcome.value().outcome)
                if (
                    att := SingleAttribution.from_lib_source_set_bad(
                        source_set,
                        outcome.value().hw_cause,
                        att_class,
                        RecommendationEnum.UNKNOWN,
                    )
                ) is not None:
                    attributions.append(att)
            elif source_set.has_interesting():
                if (att := SingleAttribution.from_lib_source_set_other(source_set)) is not None:
                    attributions.append(att)
            else:
                continue
        if not attributions:
            raise ValueError(f"NodeAttributionGroup for {node_attr.hostname} has no bad or interesting attributions")
        return cls(
            node=node_attr.hostname,
            attributions=attributions,
        )


class StorageAttribution(BaseModel):
    """Storage-specific attribution; details TBD"""

    type: Literal["StorageAttribution"] = Field(default="StorageAttribution")
    attributions: list[SingleAttributionTypes]


class SwitchAttribution(BaseModel):
    """Switch-specific attribution; details TBD"""

    type: Literal["SwitchAttribution"] = Field(default="SwitchAttribution")
    attributions: list[SingleAttributionTypes]


class GlobalAttribution(BaseModel):
    """Non-specific attribution (i.e. job-level)"""

    type: Literal["GlobalAttribution"] = Field(default="GlobalAttribution")
    attributions: list[SingleAttributionTypes]


AttributionTypes = Annotated[
    Union[WorkloadRecord, NodeAttribution, StorageAttribution, SwitchAttribution, GlobalAttribution],
    Field(discriminator="type"),
]


class AttributionResponseV1(ResponseModel):
    """Response to attribution requests"""

    observation_context: list[ObservationId]
    attributions: list[AttributionTypes]

    @classmethod
    def from_lib_attribution(
        cls,
        ingested_observations: set[ObservationId],
        result: AttributionResult,
    ) -> "AttributionResponseV1":
        attributions: list[AttributionTypes] = []

        # V1-only debt: hardcodes straggler digest keys. V2 stays rule-agnostic
        # (dumps result.collectors raw). Don't repeat this pattern elsewhere.
        node_straggler_results: dict[str, StragglerAttribution] = {}
        for digest_key in ("nvrx_straggler", "megatron_straggler"):
            for node, report in result.collectors.get(digest_key, {}).items():
                node_straggler_results[node] = StragglerAttribution(
                    description=report.pattern,
                    rank_scores=[RankScore(rank=r.rank, score=r.score) for r in report.rank_scores],
                    sources=[AttributionSource.from_lib(s) for s in report.sources],
                )

        bad_nodes = set(result.bad_nodes_attr.my_bad_nodes.keys())
        bad_nodes.update(node_straggler_results.keys())
        sorted_nodes = sorted(list(bad_nodes))
        for node in sorted_nodes:
            if node in result.bad_nodes_attr.my_bad_nodes:
                node_symptoms = result.bad_nodes_attr.my_bad_nodes[node]
                node_attr = NodeAttribution.from_lib_node_group(node_symptoms)
            else:
                node_attr = NodeAttribution(node=node, attributions=list())

            if node in node_straggler_results:
                node_attr.attributions.append(node_straggler_results[node])
            attributions.append(node_attr)

        if (outcome_attr := result.outcome_attr) is not None:
            att_class = AttributionClassEnum.from_lib_outcomes(outcome_attr.value().outcome)
            description = outcome_attr.value().hw_cause
        else:
            att_class = AttributionClassEnum.UNDETERMINED
            description = ""

        recommendation = RecommendationEnum.UNKNOWN

        sorted_items: "list[tuple[Symptom, SourceSet[Symptom]]]" = sorted(
            list(result.symptoms._data.items()), key=lambda x: x[0]
        )
        for _symptom, source_set in sorted_items:
            l_attributions = []
            if source_set.has_bad():
                if (
                    att := SingleAttribution.from_lib_source_set_bad(source_set, description, att_class, recommendation)
                ) is not None:
                    l_attributions.append(att)
            elif source_set.has_interesting():
                if (att := SingleAttribution.from_lib_source_set_other(source_set)) is not None:
                    l_attributions.append(att)
            if len(l_attributions) > 0:
                attributions.append(GlobalAttribution(attributions=l_attributions))

        response = AttributionResponseV1(
            observation_context=sorted(list(ingested_observations)), attributions=attributions
        )
        logger.info(
            f"Built AttributionResponseV1 with {len(response.attributions)} attributions "
            f"for {len(response.observation_context)} observations"
        )
        return response


class NewResponseV1(ResponseModel):
    """Response to creating a new attribution attributor."""

    attributor_id: AttributorId


class ObservationResponseV1(ResponseModel):
    """Response to performing an ingestion."""

    observation_id: ObservationId
