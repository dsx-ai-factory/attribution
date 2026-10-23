# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
from typing import Literal, Optional, TypeAlias

from attribution_lib.observation.inbound import (
    Id as ObservationId,
    Source as ObservationSource,
)
from attribution_lib.sources import (
    Source as LibSource,
    SourceSet,
    SymptomSet,
)
from pydantic import BaseModel, Field


# TODO: move under app/v1/.
class AttributionClassEnum(str, Enum):
    """The types of attributions/errors we can find. V1-only."""

    UNDEFINED = ""
    NO_ERROR = "no_error"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    USER = "user"
    ARTIFICIAL = "artificial"
    UNDETERMINED = "undetermined"

    @classmethod
    def from_lib_outcomes(cls, outcome: str) -> "AttributionClassEnum":
        match outcome:
            case "Incomplete Info":
                return cls.UNDETERMINED
            case "Fail Undetermined":
                return cls.UNDETERMINED
            case "Pass":
                return cls.NO_ERROR
            case "Fail InfraSW":
                return cls.SOFTWARE
            case "Fail User":
                return cls.USER
            case "Fail InfraHW":
                return cls.HARDWARE
            case "Artificial":
                return cls.ARTIFICIAL
        return cls.UNDEFINED


class RecommendationEnum(str, Enum):
    """The types of attributions/errors we can find."""

    UNDEFINED = ""
    DRAIN = "drain"
    NO_ERROR = "no_error"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"

    @classmethod
    def from_lib_recovery_actions(cls, recovery_actions: SymptomSet[str]) -> "RecommendationEnum":
        ## This obivously needs to be updated.
        return cls.UNKNOWN


Symptom: TypeAlias = str


class AttributionSource(BaseModel):
    """Description of provenance of attribution."""

    observation_id: ObservationId
    source: ObservationSource
    pos: Optional[int]
    line: Optional[str]
    rule: Optional[str]
    time: Optional[datetime]

    @classmethod
    def from_lib(cls, source: LibSource) -> "AttributionSource":
        return cls(
            observation_id=source.source_context.id,
            source=source.source_context.source,
            pos=source.line_num,
            line=source.datum,
            time=source.time,
            rule=source.rule,
        )


class SingleAttributionBase(BaseModel):
    """A base for the different types of attributions we can produce"""

    symptom: Symptom
    description: str
    attribution_class: AttributionClassEnum
    recommendation: RecommendationEnum
    sources: list[AttributionSource] = []


class SingleAttribution(SingleAttributionBase):
    type: Literal["SingleAttribution"] = Field(default="SingleAttribution")

    @classmethod
    def from_lib_source_set_bad(
        cls,
        source_set: SourceSet[str],
        description: str,
        attribution_class: AttributionClassEnum,
        recommendation: RecommendationEnum,
    ) -> Optional["SingleAttribution"]:
        if not source_set.has_bad():
            return None
        return SingleAttribution(
            symptom=source_set.value(),
            description=description,
            attribution_class=attribution_class,
            recommendation=recommendation,
            sources=[AttributionSource.from_lib(attr_source) for attr_source in source_set._bad],
        )

    @classmethod
    def from_lib_source_set_other(
        cls,
        source_set: SourceSet[str],
    ) -> Optional["SingleAttribution"]:
        if not source_set.has_interesting():
            return None
        return SingleAttribution(
            symptom=source_set.value(),
            description="",
            attribution_class=AttributionClassEnum.UNDETERMINED,
            recommendation=RecommendationEnum.AMBIGUOUS,
            sources=[AttributionSource.from_lib(attr_source) for attr_source in source_set._interesting],
        )
