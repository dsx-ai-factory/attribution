# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Annotated, Any, ClassVar, Literal, Union

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo


class ResponseModel(BaseModel):
    """Base model that auto-generates a discriminated union of its subclasses"""

    _registry: ClassVar[dict[str, type]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

        # Properly inject model_type with default value
        literal_type = Literal[cls.__name__]
        cls.model_fields["model_type"] = FieldInfo(annotation=literal_type, default=cls.__name__, frozen=True)
        cls.__annotations__["model_type"] = literal_type

    def __init__(self, **data):
        # Automatically set model_type during initialization
        data["model_type"] = self.__class__.__name__
        super().__init__(**data)

    @classmethod
    def get_union(cls) -> type[BaseModel]:
        """Generate discriminated union of all subclasses"""
        return Annotated[
            Union[tuple(cls._registry.values())],  # type: ignore
            Field(discriminator="model_type"),
        ]

    @classmethod
    def from_dict(cls, obj) -> "ResponseModel":
        # Dynamically create a container model
        class UnionContainer(BaseModel):
            data: cls.get_union()

        return UnionContainer.model_validate({"data": obj}).data


class FactErrorModel(ResponseModel):
    code: ClassVar[int] = 500
    msg: ClassVar[str] = "Unknown"

    def detail(self) -> str:
        return self.msg

    @classmethod
    def response(cls) -> dict[int, dict[str, Any]]:
        return {cls.code: {"description": cls.msg, "model": cls}}

    def to_http_exception(self) -> "FactHTTPException":
        return FactHTTPException(model=self)


class DuplicateObservationError(FactErrorModel):
    """Error for the case where repeated Observations are fed to an attributor."""

    code: ClassVar[int] = 409
    msg: ClassVar[str] = "Duplicate observation {obs_id} rejected."
    obs_id: int

    @classmethod
    def new(cls, obs_id: int):
        return cls(obs_id=obs_id)

    def detail(self) -> str:
        return self.msg.format(obs_id=self.obs_id)


class UnknownAttributorIdError(FactErrorModel):
    """Error for when an attributor id is not known to the attribution service."""

    code: ClassVar[int] = 404
    msg: ClassVar[str] = "Attribution id {att_id} unknown."
    att_id: str

    @classmethod
    def new(cls, att_id: str):
        return cls(att_id=att_id)

    def detail(self) -> str:
        return self.msg.format(att_id=self.att_id)


class DuplicateAttributorError(FactErrorModel):
    """Error for when a new attributor appears to be identical to an extant one."""

    code: ClassVar[int] = 409
    msg: ClassVar[str] = "Duplicate attribution {att_id} rejected."
    att_id: str

    @classmethod
    def new(cls, att_id: str):
        return cls(att_id=att_id)

    def detail(self) -> str:
        return self.msg.format(att_id=self.att_id)


class UnknownObservationFormatError(FactErrorModel):
    """Error for when we receive an unknown observation format"""

    code: ClassVar[int] = 400
    msg: ClassVar[str] = "Unknown observation format {format}."
    format: str

    @classmethod
    def new(cls, format: str):
        return cls(format=format)

    def detail(self) -> str:
        return self.msg.format(format=self.format)


class IncompatibleObservationSourceFormatError(FactErrorModel):
    code: ClassVar[int] = 400
    msg: ClassVar[str] = "Mismatched observation source {source} and format {format}."
    source: str
    format: str

    @classmethod
    def new(cls, source: str, format: str):
        return cls(source=source, format=format)

    def detail(self) -> str:
        return self.msg.format(source=self.source, format=self.format)


class FactHTTPException(HTTPException):
    def __init__(self, model: FactErrorModel):
        super().__init__(status_code=model.code, detail=model.model_dump())

    def to_json_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content=self.detail,
        )
