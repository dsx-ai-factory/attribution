# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict
from typing import Any, ClassVar, Dict, List, Optional
from nvidia_attribution_client.models.attribution_parameters import AttributionParameters
from nvidia_attribution_client.models.attributor_metadata import AttributorMetadata
from nvidia_attribution_client.models.attributor_workload_info_v1 import AttributorWorkloadInfoV1
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class AttributorInfoV1(BaseModel):
    """
    V1 wire shape for the /attributor POST body.
    """ # noqa: E501
    workload: AttributorWorkloadInfoV1
    metadata: AttributorMetadata
    parameters: Optional[AttributionParameters] = None
    __properties: ClassVar[List[str]] = ["workload", "metadata", "parameters"]

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
        """Create an instance of AttributorInfoV1 from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of workload
        if self.workload:
            _dict['workload'] = self.workload.to_dict()
        # override the default output from pydantic by calling `to_dict()` of metadata
        if self.metadata:
            _dict['metadata'] = self.metadata.to_dict()
        # override the default output from pydantic by calling `to_dict()` of parameters
        if self.parameters:
            _dict['parameters'] = self.parameters.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of AttributorInfoV1 from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "workload": AttributorWorkloadInfoV1.from_dict(obj["workload"]) if obj.get("workload") is not None else None,
            "metadata": AttributorMetadata.from_dict(obj["metadata"]) if obj.get("metadata") is not None else None,
            "parameters": AttributionParameters.from_dict(obj["parameters"]) if obj.get("parameters") is not None else None
        })
        return _obj


