# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, field_validator
from typing import Any, ClassVar, Dict, List
from nvidia_attribution_client.models.attributions_inner import AttributionsInner
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class AttributionResponseV1(BaseModel):
    """
    Response to attribution requests
    """ # noqa: E501
    observation_context: List[StrictInt]
    attributions: List[AttributionsInner]
    model_type: StrictStr
    __properties: ClassVar[List[str]] = ["observation_context", "attributions", "model_type"]

    @field_validator('model_type')
    def model_type_validate_enum(cls, value):
        """Validates the enum"""
        if value not in set(['AttributionResponseV1']):
            raise ValueError("must be one of enum values ('AttributionResponseV1')")
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
        """Create an instance of AttributionResponseV1 from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of each item in attributions (list)
        _items = []
        if self.attributions:
            for _item_attributions in self.attributions:
                if _item_attributions:
                    _items.append(_item_attributions.to_dict())
            _dict['attributions'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of AttributionResponseV1 from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "observation_context": obj.get("observation_context"),
            "attributions": [AttributionsInner.from_dict(_item) for _item in obj["attributions"]] if obj.get("attributions") is not None else None,
            "model_type": obj.get("model_type")
        })
        return _obj


