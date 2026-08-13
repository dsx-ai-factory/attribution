# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from fastapi import APIRouter, Request, Response

from app.attributor import (
    AttributionContext,
    AttributorId,
)
from app.exceptions import (
    DuplicateAttributorError,
    DuplicateObservationError,
    UnknownAttributorIdError,
)
from app.observation import Observation
from app.v1.models import (
    AttributionResponseV1,
    AttributorInfoV1,
    NewResponseV1,
    ObservationResponseV1,
)

router = APIRouter()


@router.get("/healthz")
def health():
    return None


@router.post(
    "/attributor",
    responses={
        201: {"model": NewResponseV1},
        **DuplicateAttributorError.response(),
    },
)
def new_attribution(att: AttributorInfoV1, request: Request, response: Response) -> NewResponseV1:
    """Create a new attribution attributor based on provided metadata and return an id with which to identify it. Providing identical metadata to an existing attributor is an error."""
    attribution = request.app.state.aim.add_attributor(att)
    response.status_code = 201
    return NewResponseV1(attributor_id=attribution.id)


@router.post(
    "/attributor/{att_id}/observation",
    responses={
        201: {"model": ObservationResponseV1},
        **UnknownAttributorIdError.response(),
        **DuplicateObservationError.response(),
    },
)
def ingest(att_id: AttributorId, obs_data: Observation, request: Request, response: Response) -> ObservationResponseV1:
    """Pass observation to specified attributor and return an id that identifies it. It is an error to provide a duplicate observation. When the id is returned, the observation has been successfully ingested."""
    ingested = request.app.state.aim.get_attributor(att_id).ingest(obs_data)
    response.status_code = 201
    return ObservationResponseV1(observation_id=ingested)


@router.get(
    "/attributor/{att_id}/attribution",
    responses={
        200: {"model": AttributionResponseV1},
        **UnknownAttributorIdError.response(),
    },
)
def attribute(att_id: AttributorId, request: Request, response: Response) -> AttributionResponseV1:
    """Perform attribution on specified attributor based on all previous observation ingested on this attributor."""
    ctx: AttributionContext = request.app.state.aim.get_attributor(att_id).attribute()
    response.status_code = 200
    return AttributionResponseV1.from_lib_attribution(ctx.observation_ids, ctx.result)


@router.delete(
    "/attributor/{att_id}",
    responses={
        204: {},
        **UnknownAttributorIdError.response(),
    },
)
def delete_attribution(att_id: AttributorId, request: Request, response: Response):
    """Destroy attribution attributor indicated by id. It is an error if the id is not present in the service."""
    request.app.state.aim.delete_attributor(att_id)
    response.status_code = 204
