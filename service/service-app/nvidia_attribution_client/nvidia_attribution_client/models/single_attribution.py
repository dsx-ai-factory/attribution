# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator
from typing import Any, ClassVar, Dict, List, Optional
from nvidia_attribution_client.models.attribution_class_enum import AttributionClassEnum
from nvidia_attribution_client.models.attribution_source import AttributionSource
from nvidia_attribution_client.models.recommendation_enum import RecommendationEnum
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class SingleAttribution(BaseModel):
    """
    SingleAttribution
    """ # noqa: E501
    symptom: StrictStr
    description: StrictStr
    attribution_class: AttributionClassEnum
    recommendation: RecommendationEnum
    sources: Optional[List[AttributionSource]] = None
    type: Optional[StrictStr] = 'SingleAttribution'
    __properties: ClassVar[List[str]] = ["symptom", "description", "attribution_class", "recommendation", "sources", "type"]

    @field_validator('type')
    def type_validate_enum(cls, value):
        """Validates the enum"""
        if value is None:
            return value

        if value not in set(['SingleAttribution']):
            raise ValueError("must be one of enum values ('SingleAttribution')")
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
        """Create an instance of SingleAttribution from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of each item in sources (list)
        _items = []
        if self.sources:
            for _item_sources in self.sources:
                if _item_sources:
                    _items.append(_item_sources.to_dict())
            _dict['sources'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of SingleAttribution from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "symptom": obj.get("symptom"),
            "description": obj.get("description"),
            "attribution_class": obj.get("attribution_class"),
            "recommendation": obj.get("recommendation"),
            "sources": [AttributionSource.from_dict(_item) for _item in obj["sources"]] if obj.get("sources") is not None else None,
            "type": obj.get("type") if obj.get("type") is not None else 'SingleAttribution'
        })
        return _obj


