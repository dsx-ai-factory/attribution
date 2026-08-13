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
from typing import Any
from zoneinfo import ZoneInfo

from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Id as ObservationId,  # noqa: F401
    Observation as LibObservation,
    Source as ObservationSource,
    TimeInterval as LibTimeInterval,
)
from attribution_lib.resources import AllJobResources
from pydantic import BaseModel, RootModel, ValidationError

from app.exceptions import InvalidObservationBodyError, UnknownObservationFormatError
from app.util import HashableModel, TimeInterval


class ObservationContext(HashableModel):
    """Describes the scope of an observation."""

    time_interval: TimeInterval
    resources: str = "AllJobResources"
    source: ObservationSource
    format: ObservationFormat
    is_prefiltered: bool = False

    def to_lib_context(self, timezone: ZoneInfo) -> InboundContext:
        """Convert the observation context to a lib context."""
        return InboundContext(
            LibTimeInterval(start=self.time_interval.start, end=self.time_interval.end),
            AllJobResources,
            self.source,
            self.format,
            timezone,
            pre_filtered=self.is_prefiltered,
        )


class RawLokiStreamPair(RootModel):
    root: tuple[str, str]


class RawLokiStream(BaseModel):
    stream: dict[str, Any]
    values: list[RawLokiStreamPair]


class RawLokiStreams(RootModel):
    root: list[RawLokiStream]


class Observation(HashableModel):
    """Describes association of some observation data with a context."""

    context: ObservationContext
    body: str

    def to_lib_observation(self, timezone: ZoneInfo) -> LibObservation:
        """Convert the observation to a lib observation."""
        context = self.context.to_lib_context(timezone)
        body: str | list[str] | dict[str, Any]
        match context.format:
            case ObservationFormat.STRING_LIST:
                body = self.body.split("\n")
            case ObservationFormat.RAW_LOG_LINES | ObservationFormat.CSV_METRIC_RESPONSE:
                body = self.body.split("\n")
            case (
                ObservationFormat.LOKI_DICT_LIST
                | ObservationFormat.UFMLOG
                | ObservationFormat.PROMETHEUS_METRICS
                | ObservationFormat.CSP_EVENT_LIST
            ):
                try:
                    body = json.loads(self.body)
                except json.JSONDecodeError as e:
                    raise InvalidObservationBodyError.new(context.format.value).to_http_exception() from e
            case ObservationFormat.RAW_LOKI_STREAMS:
                try:
                    body = RawLokiStreams.model_validate_json(self.body).model_dump()
                except ValidationError as e:
                    raise InvalidObservationBodyError.new(context.format.value).to_http_exception() from e
            case _:
                raise UnknownObservationFormatError.new(context.format.value).to_http_exception()

        return LibObservation(context, body)
