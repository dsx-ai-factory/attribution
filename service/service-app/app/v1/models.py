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
from typing import Optional

from attribution_lib.bundled_rules.common.megatron import MegatronStragglerParseState
from attribution_lib.bundled_rules.common.nvrx import StragglerGroup, StragglerParseState
from attribution_lib.core import AttributionResult
from attribution_lib.job_info import JobInfo, JobInfoBuilder
from attribution_lib.sources import SourceSet
from loguru import logger
from pydantic import AwareDatetime, Field

from app.attribution import (
    AttributionClassEnum,
    AttributionTypes,
    GlobalAttribution,
    NodeAttribution,
    RecommendationEnum,
    SingleAttribution,
    StragglerAttribution,
    Symptom,
)
from app.attributor import (
    AttributionParameters,
    AttributorId,
    AttributorMetadata,
    WorkloadTypeEnum,
)
from app.exceptions import ResponseModel
from app.observation import ObservationId
from app.util import HashableModel, to_zoneinfo


class AttributorWorkloadInfoV1(HashableModel):
    """V1 wire shape: legacy positional node list + ranks-per-node fan-out, or explicit
    node-to-ranks mapping. Mutually exclusive at the lib layer."""

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
        w = self.workload
        builder_kwargs: dict = dict(
            id=w.id,
            framework=w.framework,
            type=w.type,
            cluster=self.metadata.cluster,
            state=w.status,
            job_start_time=w.job_start_time,
            source_name="attributor service translation",
            timezone=to_zoneinfo(w.job_start_time),
        )
        if w.ranks_per_node is not None:
            builder_kwargs["RANKS_PER_NODE"] = w.ranks_per_node
        builder = JobInfoBuilder(**builder_kwargs)
        if w.nodes is not None:
            builder = builder.nodes(w.nodes)
        if w.node_to_ranks is not None:
            builder = builder.node_to_ranks(w.node_to_ranks)
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


class AttributionResponseV1(ResponseModel):
    """Response to attribution requests"""

    observation_context: list[ObservationId]
    attributions: list[AttributionTypes]

    @classmethod
    def from_lib_attribution(
        cls,
        parameters: AttributionParameters,
        ingested_observations: set[ObservationId],
        result: AttributionResult,
    ) -> "AttributionResponseV1":
        attributions: list[AttributionTypes] = []

        nvrx_results = StragglerParseState.collect(result.event_stream)
        nvrx_groups = [g for g in nvrx_results if isinstance(g, StragglerGroup)]
        node_straggler_results = StragglerAttribution.unpack_lib_groups(
            parameters.nvrx_straggler_parameters, nvrx_groups
        )

        megatron_data = MegatronStragglerParseState.collect(result.event_stream)
        if megatron_data.reports:
            megatron_straggler_results = StragglerAttribution.unpack_megatron_lib_groups(
                parameters.megatron_straggler_parameters,
                megatron_data.reports,
                megatron_data.top_scores,
            )
            node_straggler_results.update(megatron_straggler_results)

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
