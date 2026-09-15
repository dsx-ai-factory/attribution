# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Annotated, Any, ClassVar, Literal, Self, Union

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, RootModel
from pydantic.fields import FieldInfo

# Errors an attribution_lib change to the attribution result or Decision DAG shape can surface
# while the graph is walked: missing/renamed attributes and keys, changed types, shortened
# sequences, and unparseable enum values (pydantic's ValidationError is a ValueError).
DAG_SHAPE_ERRORS = (AttributeError, TypeError, KeyError, IndexError, ValueError)


class ResponseModel(BaseModel):
    """Base model for responses selected by their model_type discriminator."""

    _registry: ClassVar[dict[str, type["ResponseModel"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

        # Subclasses that declare model_type statically (needed for discriminated
        # unions such as the observation-ingestion error union) opt out of dynamic
        # injection. All others get model_type injected for backward compatibility,
        # keeping their published schema (e.g. the v1/v2 responses) unchanged.
        if "model_type" not in cls.__annotations__:
            literal_type = Literal[cls.__name__]  # ty: ignore[invalid-type-form]
            cls.model_fields["model_type"] = FieldInfo(annotation=literal_type, default=cls.__name__, frozen=True)
            cls.__annotations__["model_type"] = literal_type

    def __init__(self, **data):
        # Automatically set model_type during initialization
        data["model_type"] = self.__class__.__name__
        super().__init__(**data)

    @classmethod
    def get_union(cls) -> Any:
        """Generate a runtime discriminated union of registered response models."""
        return Annotated[
            Union[tuple(cls._registry.values())],  # type: ignore
            Field(discriminator="model_type"),
        ]

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "ResponseModel":
        model_type = obj.get("model_type")
        if not isinstance(model_type, str):
            raise ValueError("response model_type must be a string")
        model = cls._registry.get(model_type)
        if model is None:
            raise ValueError(f"unknown response model_type: {model_type}")
        return model.model_validate(obj)


class ValidationErrorResponse(BaseModel):
    """Published schema for the validation envelope returned by the application handlers."""

    errors: str | list[Any]
    body: Any
    message: str

    @classmethod
    def response(cls) -> dict[int | str, dict[str, Any]]:
        return {422: {"description": "Validation failed.", "model": cls}}


class AttributionErrorModel(ResponseModel):
    code: ClassVar[int] = 500
    msg: ClassVar[str] = "Unknown"

    def detail(self) -> str:
        return self.msg

    @classmethod
    def response(cls) -> dict[int | str, dict[str, Any]]:
        return {cls.code: {"description": cls.msg, "model": cls}}

    def to_http_exception(self) -> "AttributionHTTPException":
        return AttributionHTTPException(model=self)


class DuplicateObservationError(AttributionErrorModel):
    """Error for the case where repeated Observations are fed to an attributor."""

    model_type: Literal["DuplicateObservationError"] = Field(frozen=True)
    code: ClassVar[int] = 409
    msg: ClassVar[str] = "Duplicate observation {obs_id} rejected."
    obs_id: int

    @classmethod
    def new(cls, obs_id: int) -> Self:
        return cls(model_type="DuplicateObservationError", obs_id=obs_id)

    def detail(self) -> str:
        return self.msg.format(obs_id=self.obs_id)


class UnknownAttributorIdError(AttributionErrorModel):
    """Error for when an attributor id is not known to the attribution service."""

    model_type: Literal["UnknownAttributorIdError"] = Field(frozen=True)
    code: ClassVar[int] = 404
    msg: ClassVar[str] = "Attribution id {att_id} unknown."
    att_id: str

    @classmethod
    def new(cls, att_id: str) -> Self:
        return cls(model_type="UnknownAttributorIdError", att_id=att_id)

    def detail(self) -> str:
        return self.msg.format(att_id=self.att_id)


class DuplicateAttributorError(AttributionErrorModel):
    """Error for when a new attributor appears to be identical to an extant one."""

    model_type: Literal["DuplicateAttributorError"] = Field(frozen=True)
    code: ClassVar[int] = 409
    msg: ClassVar[str] = "Duplicate attribution {att_id} rejected."
    att_id: str

    @classmethod
    def new(cls, att_id: str) -> Self:
        return cls(model_type="DuplicateAttributorError", att_id=att_id)

    def detail(self) -> str:
        return self.msg.format(att_id=self.att_id)


class UnknownObservationFormatError(AttributionErrorModel):
    """Error for when we receive an unknown observation format"""

    model_type: Literal["UnknownObservationFormatError"] = Field(
        frozen=True,
    )
    code: ClassVar[int] = 400
    msg: ClassVar[str] = "Unknown observation format {format}."
    format: str

    @classmethod
    def new(cls, format: str) -> Self:
        return cls(model_type="UnknownObservationFormatError", format=format)

    def detail(self) -> str:
        return self.msg.format(format=self.format)


class InvalidObservationBodyError(AttributionErrorModel):
    """Error for a structured observation body that cannot be decoded."""

    model_type: Literal["InvalidObservationBodyError"] = Field(frozen=True)
    code: ClassVar[int] = 400
    msg: ClassVar[str] = "Invalid observation body for format {format}."
    format: str

    @classmethod
    def new(cls, format: str) -> Self:
        return cls(model_type="InvalidObservationBodyError", format=format)

    def detail(self) -> str:
        return self.msg.format(format=self.format)


class IncompatibleObservationSourceFormatError(AttributionErrorModel):
    model_type: Literal["IncompatibleObservationSourceFormatError"] = Field(
        frozen=True,
    )
    code: ClassVar[int] = 400
    msg: ClassVar[str] = "Mismatched observation source {source} and format {format}."
    source: str
    format: str

    @classmethod
    def new(cls, source: str, format: str) -> Self:
        return cls(
            model_type="IncompatibleObservationSourceFormatError",
            source=source,
            format=format,
        )

    def detail(self) -> str:
        return self.msg.format(source=self.source, format=self.format)


class EventStorageOverflowError(AttributionErrorModel):
    """Error for when an attributor's event storage exceeds its memory budget.

    Raised instead of a generic 500 when ATTRIBUTOR_ALLOW_DISK_SPILL=false
    (the default for containerized deployments with no writable disk) and an
    attributor's DuckDB connection exceeds ATTRIBUTOR_DUCKDB_MEMORY_LIMIT.
    Isolated to the one attributor/request that triggered it.
    """

    model_type: Literal["EventStorageOverflowError"] = Field(frozen=True)
    code: ClassVar[int] = 507
    msg: ClassVar[str] = "Attributor {att_id} exceeded its memory budget."
    att_id: str

    @classmethod
    def new(cls, att_id: str) -> Self:
        return cls(model_type="EventStorageOverflowError", att_id=att_id)

    def detail(self) -> str:
        return self.msg.format(att_id=self.att_id)


ObservationIngestionError = Annotated[
    UnknownObservationFormatError | InvalidObservationBodyError | IncompatibleObservationSourceFormatError,
    Field(discriminator="model_type"),
]


class ObservationIngestionErrorResponse(RootModel[ObservationIngestionError]):
    """Published union of client-fixable observation ingestion errors."""

    @classmethod
    def response(cls) -> dict[int | str, dict[str, Any]]:
        return {
            400: {
                "description": "Observation ingestion failed.",
                "model": cls,
                "content": {
                    "application/json": {
                        "examples": {
                            "UnknownObservationFormatError": {
                                "summary": "Unsupported recognized observation format",
                                "value": UnknownObservationFormatError.new("future_format").model_dump(),
                            },
                            "InvalidObservationBodyError": {
                                "summary": "Malformed structured observation body",
                                "value": InvalidObservationBodyError.new("raw_loki_streams").model_dump(),
                            },
                            "IncompatibleObservationSourceFormatError": {
                                "summary": "Incompatible observation source and format",
                                "value": IncompatibleObservationSourceFormatError.new(
                                    "applog", "string_list"
                                ).model_dump(),
                            },
                        }
                    }
                },
            }
        }


class AttributionHTTPException(HTTPException):
    def __init__(self, model: AttributionErrorModel):
        super().__init__(status_code=model.code, detail=model.model_dump())

    def to_json_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content=self.detail,
        )
