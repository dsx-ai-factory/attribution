# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator
from typing import Any, ClassVar, Dict, List
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class EventStorageOverflowError(BaseModel):
    """
    Error for when an attributor's event storage exceeds its memory budget.  Raised instead of a generic 500 when ATTRIBUTOR_ALLOW_DISK_SPILL=false (the default for containerized deployments with no writable disk) and an attributor's DuckDB connection exceeds ATTRIBUTOR_DUCKDB_MEMORY_LIMIT. Isolated to the one attributor/request that triggered it.
    """ # noqa: E501
    model_type: StrictStr
    att_id: StrictStr
    __properties: ClassVar[List[str]] = ["model_type", "att_id"]

    @field_validator('model_type')
    def model_type_validate_enum(cls, value):
        """Validates the enum"""
        if value not in set(['EventStorageOverflowError']):
            raise ValueError("must be one of enum values ('EventStorageOverflowError')")
        return value

    model_config = ConfigDict(
        validate_by_name=True,
        validate_by_alias=True,
        validate_assignment=True,
        protected_namespaces=(),
    )


    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.model_dump(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(to_jsonable_python(self.to_dict()))

    @classmethod
    def from_json(cls, json_str: str) -> Optional[Self]:
        """Create an instance of EventStorageOverflowError from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        """Return the dictionary representation of the model using alias.

        This has the following differences from calling pydantic's
        `self.model_dump(by_alias=True)`:

        * `None` is only added to the output dict for nullable fields that
          were set at model initialization. Other fields with value `None`
          are ignored.
        """
        excluded_fields: Set[str] = set([
        ])

        _dict = self.model_dump(
            by_alias=True,
            exclude=excluded_fields,
            exclude_none=True,
        )
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of EventStorageOverflowError from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "model_type": obj.get("model_type"),
            "att_id": obj.get("att_id")
        })
        return _obj


