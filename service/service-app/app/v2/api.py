# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Annotated

from fastapi import APIRouter, Query, Request, Response

from app.attributor import AttributionContext, AttributorId
from app.exceptions import (
    DuplicateAttributorError,
    DuplicateObservationError,
    ObservationIngestionErrorResponse,
    UnknownAttributorIdError,
    ValidationErrorResponse,
)
from app.observation import Observation
from app.v2.models import (
    AttributionResponseV2,
    AttributorInfoV2,
    NewResponseV2,
    ObservationResponseV2,
)

router = APIRouter()


@router.get("/healthz")
def health():
    return None


@router.post(
    "/attributor",
    responses={
        201: {"model": NewResponseV2},
        **DuplicateAttributorError.response(),
        **ValidationErrorResponse.response(),
    },
)
def new_attribution(att: AttributorInfoV2, request: Request, response: Response) -> NewResponseV2:
    """Create a new attribution attributor based on provided metadata and return an id with which to identify it. Providing identical metadata to an existing attributor is an error."""
    attribution = request.app.state.aim.add_attributor(att)
    response.status_code = 201
    return NewResponseV2(attributor_id=attribution.id)


@router.post(
    "/attributor/{att_id}/observation",
    responses={
        201: {"model": ObservationResponseV2},
        **UnknownAttributorIdError.response(),
        **DuplicateObservationError.response(),
        **ObservationIngestionErrorResponse.response(),
        **ValidationErrorResponse.response(),
    },
)
def ingest(att_id: AttributorId, obs_data: Observation, request: Request, response: Response) -> ObservationResponseV2:
    """Pass observation to specified attributor and return an id that identifies it. It is an error to provide a duplicate observation. When the id is returned, the observation has been successfully ingested."""
    ingested = request.app.state.aim.get_attributor(att_id).ingest(obs_data)
    response.status_code = 201
    return ObservationResponseV2(observation_id=ingested)


@router.get(
    "/attributor/{att_id}/attribution",
    responses={
        200: {"model": AttributionResponseV2},
        **UnknownAttributorIdError.response(),
        **ValidationErrorResponse.response(),
    },
)
def attribute(
    att_id: AttributorId,
    request: Request,
    response: Response,
    include_decision_graph: Annotated[
        bool,
        Query(description="Include the runtime Decision DAG in the verdict."),
    ] = False,
) -> AttributionResponseV2:
    """Perform attribution on specified attributor based on all previous observation ingested on this attributor."""
    ctx: AttributionContext = request.app.state.aim.get_attributor(att_id).attribute(
        include_decision_graph=include_decision_graph
    )
    response.status_code = 200
    return AttributionResponseV2.from_lib_attribution(
        ctx.observation_ids,
        ctx.result,
        ctx.decision_graph,
        ctx.decision_graph_requested,
    )


@router.delete(
    "/attributor/{att_id}",
    responses={
        204: {},
        **UnknownAttributorIdError.response(),
        **ValidationErrorResponse.response(),
    },
)
def delete_attribution(att_id: AttributorId, request: Request, response: Response) -> None:
    """Destroy attribution attributor indicated by id. It is an error if the id is not present in the service."""
    request.app.state.aim.delete_attributor(att_id)
    response.status_code = 204
