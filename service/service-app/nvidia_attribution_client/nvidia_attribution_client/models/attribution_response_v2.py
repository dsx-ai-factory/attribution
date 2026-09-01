# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, field_validator
from typing import Any, ClassVar, Dict, List, Optional
from nvidia_attribution_client.models.symptoms_value_inner import SymptomsValueInner
from nvidia_attribution_client.models.verdict import Verdict
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class AttributionResponseV2(BaseModel):
    """
    V2 response to GET /v2/attributor/{att_id}/attribution.
    """ # noqa: E501
    verdict: Verdict
    symptoms: Dict[str, List[SymptomsValueInner]]
    observation_context: List[StrictInt]
    collectors: Optional[Dict[str, Any]] = None
    model_type: StrictStr
    __properties: ClassVar[List[str]] = ["verdict", "symptoms", "observation_context", "collectors", "model_type"]

    @field_validator('model_type')
    def model_type_validate_enum(cls, value):
        """Validates the enum"""
        if value not in set(['AttributionResponseV2']):
            raise ValueError("must be one of enum values ('AttributionResponseV2')")
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
        """Create an instance of AttributionResponseV2 from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of verdict
        if self.verdict:
            _dict['verdict'] = self.verdict.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each value in symptoms (dict of array)
        _field_dict_of_array = {}
        if self.symptoms:
            for _key_symptoms in self.symptoms:
                if self.symptoms[_key_symptoms] is not None:
                    _field_dict_of_array[_key_symptoms] = [
                        _item.to_dict() for _item in self.symptoms[_key_symptoms]
                    ]
            _dict['symptoms'] = _field_dict_of_array
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of AttributionResponseV2 from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "verdict": Verdict.from_dict(obj["verdict"]) if obj.get("verdict") is not None else None,
            "symptoms": dict(
                (_k,
                        [SymptomsValueInner.from_dict(_item) for _item in _v]
                        if _v is not None
                        else None
                )
                for _k, _v in obj.get("symptoms", {}).items()
            ),
            "observation_context": obj.get("observation_context"),
            "collectors": obj.get("collectors"),
            "model_type": obj.get("model_type")
        })
        return _obj


