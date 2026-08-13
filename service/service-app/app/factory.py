# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
from contextlib import asynccontextmanager
from typing import Union

from attribution_lib.core import setup_rules
from attribution_lib.exceptions import (
    AttributionError as LibAttributionError,
    InvalidRuleParams,
    JobInfoError,
    RuleAcceptsNoParamsError,
)
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

import app.attributor_manager as attributor_manager
import app.v1.api as v1
import app.v2.api as v2
from app import __version__
from app.exceptions import AttributionHTTPException
from app.logging_config import setup_logging
from app.metrics import VERSION_INFO

_DESCRIPTION = """
NVIDIA Attribution Service helps interpret workload information (primarily failures).

It produces 'attributions', typically node-based, based on 'observations'.

The model is that POSTing to the '/attributor' endpoint with some workload information plus agent metadata creates a handle that is returned to the user.

The handle can then be used as follows:

* POSTing 'observations' to '/attributor/{handle}/observation' feeds logs/metrics/etc to the service and associates them with the handle. Each such POST will produce an 'observation_id' that can be used to track the ingestion of the observation.

* GETing '/attributor/{handle}/attribution' will produce an attribution that tries to explain how the workload failed (if it did) and explains how the service reached that conclusion.

* DELETEing '/attributor/{handle}' will clean up that handle.

It is possible to interleave observation-posting and attribution-getting; the more observations that have been ingested, the better the quality of the attribution, generally speaking.
"""


def create_app() -> FastAPI:
    """Create the service application, initializing logging and attribution rules."""
    otlp_provider = setup_logging()
    setup_rules()
    global_aim = attributor_manager.AttributorManager(expiration_period_hours=24)
    instrumentator = Instrumentator()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        instrumentator.expose(app)
        # Keep a strong ref to avoid garbage collection of the task.
        app.state.cleanup_task = asyncio.create_task(_cleanup_expired_attributors(global_aim))
        yield
        if otlp_provider is not None:
            otlp_provider.force_flush(timeout_millis=5000)
            otlp_provider.shutdown()

    lapp = FastAPI(
        title="NVIDIA Attribution Service",
        description=_DESCRIPTION,
        summary="Service for producing failure attribution/workload information from metrics & logs",
        contact={"name": "NVIDIA Corporation"},
        lifespan=lifespan,
        # TBD: License, URL, etc
    )

    lapp.state.aim = global_aim
    lapp.include_router(v1.router, prefix="/v1")
    lapp.include_router(v2.router, prefix="/v2")
    lapp.include_router(v2.router, prefix="/latest")
    instrumentator.instrument(lapp)

    @lapp.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder(
                {
                    "errors": exc.errors(),
                    "body": exc.body,
                    "message": "Validation failed. Please check the errors.",
                }
            ),
        )

    @lapp.exception_handler(AttributionHTTPException)
    async def attribution_http_exception_handler(_request: Request, exception: AttributionHTTPException):
        return exception.to_json_response()

    @lapp.exception_handler(JobInfoError)
    async def jobinfo_error_handler(_request: Request, exception: JobInfoError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "errors": str(exception),
                "body": repr(exception),
                "message": "Workload validation failed.",
            },
        )

    @lapp.exception_handler(RuleAcceptsNoParamsError)
    @lapp.exception_handler(InvalidRuleParams)
    async def rule_params_error_handler(
        _request: Request, exception: Union[RuleAcceptsNoParamsError, InvalidRuleParams]
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "errors": [str(exception)],
                "body": repr(exception),
                "message": "Rule parameter validation failed.",
            },
        )

    @lapp.exception_handler(LibAttributionError)
    async def unhandled_lib_error(_request: Request, exception: LibAttributionError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "errors": str(exception),
                "body": repr(exception),
                "message": "Unhandled exception from attribution library.",
            },
        )

    @lapp.middleware("http")
    async def log_http(request: Request, call_next):
        logger.debug(f"REQUEST: {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            logger.debug(f"RESPONSE: {request.method} {request.url.path} Status: {response.status_code}")
            return response
        except Exception as e:
            logger.exception(f"Error handling request: {e}")
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    VERSION_INFO.labels(version=__version__).set(1)
    logger.info(f"Current Version: {__version__}")

    return lapp


async def _cleanup_expired_attributors(aim: attributor_manager.AttributorManager):
    """Background task to periodically clean up expired attributors."""
    while True:
        aim.cleanup_expired_attributors()
        # Run cleanup every 1 hour
        await asyncio.sleep(60 * 60 * 1)
