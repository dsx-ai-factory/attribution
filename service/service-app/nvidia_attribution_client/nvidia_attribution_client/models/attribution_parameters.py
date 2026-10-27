# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict
from typing import Any, ClassVar, Dict, List, Optional
from nvidia_attribution_client.models.rule_parameters import RuleParameters
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class AttributionParameters(BaseModel):
    """
    Parameters to guide attribution.
    """ # noqa: E501
    rule_params: Optional[Dict[str, RuleParameters]] = None
    __properties: ClassVar[List[str]] = ["rule_params"]

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
        """Create an instance of AttributionParameters from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of each value in rule_params (dict)
        _field_dict = {}
        if self.rule_params:
            for _key_rule_params in self.rule_params:
                if self.rule_params[_key_rule_params]:
                    _field_dict[_key_rule_params] = self.rule_params[_key_rule_params].to_dict()
            _dict['rule_params'] = _field_dict
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of AttributionParameters from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "rule_params": dict(
                (_k, RuleParameters.from_dict(_v))
                for _k, _v in obj["rule_params"].items()
            )
            if obj.get("rule_params") is not None
            else None
        })
        return _obj


