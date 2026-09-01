# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr
from typing import Any, ClassVar, Dict, List, Optional
from nvidia_attribution_client.models.decision_action import DecisionAction
from nvidia_attribution_client.models.decision_node_type import DecisionNodeType
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class DecisionNode(BaseModel):
    """
    One evidence, rule-action, or outcome node in a runtime Decision DAG.
    """ # noqa: E501
    node_id: StrictStr
    node_type: DecisionNodeType
    hostname: Optional[StrictStr]
    symptom: StrictStr
    rule_name: Optional[StrictStr]
    action: DecisionAction
    reason: Optional[StrictStr] = None
    datum: Optional[StrictStr] = None
    line_num: Optional[StrictInt] = None
    timestamp: Optional[StrictStr] = None
    obs_source: Optional[StrictStr] = None
    __properties: ClassVar[List[str]] = ["node_id", "node_type", "hostname", "symptom", "rule_name", "action", "reason", "datum", "line_num", "timestamp", "obs_source"]

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
        """Create an instance of DecisionNode from a JSON string"""
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
        # set to None if hostname (nullable) is None
        # and model_fields_set contains the field
        if self.hostname is None and "hostname" in self.model_fields_set:
            _dict['hostname'] = None

        # set to None if rule_name (nullable) is None
        # and model_fields_set contains the field
        if self.rule_name is None and "rule_name" in self.model_fields_set:
            _dict['rule_name'] = None

        # set to None if reason (nullable) is None
        # and model_fields_set contains the field
        if self.reason is None and "reason" in self.model_fields_set:
            _dict['reason'] = None

        # set to None if datum (nullable) is None
        # and model_fields_set contains the field
        if self.datum is None and "datum" in self.model_fields_set:
            _dict['datum'] = None

        # set to None if line_num (nullable) is None
        # and model_fields_set contains the field
        if self.line_num is None and "line_num" in self.model_fields_set:
            _dict['line_num'] = None

        # set to None if timestamp (nullable) is None
        # and model_fields_set contains the field
        if self.timestamp is None and "timestamp" in self.model_fields_set:
            _dict['timestamp'] = None

        # set to None if obs_source (nullable) is None
        # and model_fields_set contains the field
        if self.obs_source is None and "obs_source" in self.model_fields_set:
            _dict['obs_source'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of DecisionNode from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "node_id": obj.get("node_id"),
            "node_type": obj.get("node_type"),
            "hostname": obj.get("hostname"),
            "symptom": obj.get("symptom"),
            "rule_name": obj.get("rule_name"),
            "action": obj.get("action"),
            "reason": obj.get("reason"),
            "datum": obj.get("datum"),
            "line_num": obj.get("line_num"),
            "timestamp": obj.get("timestamp"),
            "obs_source": obj.get("obs_source")
        })
        return _obj


