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

from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt
from typing import Any, ClassVar, Dict, List, Optional, Union
from fact_client.models.straggler_score_type import StragglerScoreType
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class MegatronStragglerParameters(BaseModel):
    """
    Parameters to guide Megatron stragglers
    """ # noqa: E501
    straggler_threshold: Optional[Union[StrictFloat, StrictInt]] = 0.95
    score_types: Optional[List[StragglerScoreType]] = None
    infer_threshold_from_data: Optional[StrictBool] = True
    __properties: ClassVar[List[str]] = ["straggler_threshold", "score_types", "infer_threshold_from_data"]

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
        """Create an instance of MegatronStragglerParameters from a JSON string"""
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
        """Create an instance of MegatronStragglerParameters from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "straggler_threshold": obj.get("straggler_threshold") if obj.get("straggler_threshold") is not None else 0.95,
            "score_types": obj.get("score_types"),
            "infer_threshold_from_data": obj.get("infer_threshold_from_data") if obj.get("infer_threshold_from_data") is not None else True
        })
        return _obj


