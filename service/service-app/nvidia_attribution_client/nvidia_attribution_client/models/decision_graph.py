# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict
from typing import Any, ClassVar, Dict, List
from nvidia_attribution_client.models.decision_edge import DecisionEdge
from nvidia_attribution_client.models.decision_node import DecisionNode
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class DecisionGraph(BaseModel):
    """
    Per-attribution runtime Decision DAG.
    """ # noqa: E501
    nodes: List[DecisionNode]
    edges: List[DecisionEdge]
    __properties: ClassVar[List[str]] = ["nodes", "edges"]

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
        """Create an instance of DecisionGraph from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of each item in nodes (list)
        _items = []
        if self.nodes:
            for _item_nodes in self.nodes:
                if _item_nodes:
                    _items.append(_item_nodes.to_dict())
            _dict['nodes'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in edges (list)
        _items = []
        if self.edges:
            for _item_edges in self.edges:
                if _item_edges:
                    _items.append(_item_edges.to_dict())
            _dict['edges'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of DecisionGraph from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "nodes": [DecisionNode.from_dict(_item) for _item in obj["nodes"]] if obj.get("nodes") is not None else None,
            "edges": [DecisionEdge.from_dict(_item) for _item in obj["edges"]] if obj.get("edges") is not None else None
        })
        return _obj


