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

from pydantic import BaseModel, ConfigDict, Field, StrictStr
from typing import Any, ClassVar, Dict, List, Optional
from typing_extensions import Annotated
from nvidia_attribution_client.models.cancellation_actor import CancellationActor
from nvidia_attribution_client.models.decision_graph import DecisionGraph
from nvidia_attribution_client.models.failure_domain import FailureDomain
from nvidia_attribution_client.models.sourced_symptom import SourcedSymptom
from nvidia_attribution_client.models.terminal_state import TerminalState
from nvidia_attribution_client.models.uncertainty_type import UncertaintyType
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class Verdict(BaseModel):
    """
    Top-level attribution outcome: raw lib outcome, free-form detail, and the driving root-cause symptom if any.
    """ # noqa: E501
    root_symptom: Optional[SourcedSymptom]
    attribution_outcome: StrictStr
    detail: StrictStr
    terminal_states: Annotated[List[TerminalState], Field(min_length=1)] = Field(description="Plausible terminal states in NVIDIA Attribution's assessment. Values are alternatives, with no ranking or probabilities; omitted states are ruled out by that assessment. A singleton does not guarantee correctness, and the number of values is not a confidence score.")
    failure_domains: Optional[List[FailureDomain]] = Field(default=None, description="Plausible failure domains, conditional on Fail. Values express alternatives, not simultaneous causes or equal probabilities. Empty means failure is absent or its domain is unspecified; consult terminal_states to distinguish these cases.")
    cancellation_actors: Optional[List[CancellationActor]] = Field(default=None, description="Plausible initiators, conditional on Cancelled. Values express alternatives without ranking or probabilities. Empty means cancellation is absent or its actor is unspecified; consult terminal_states to distinguish these cases.")
    uncertainty_type: Optional[UncertaintyType] = None
    decision_graph: Optional[DecisionGraph] = None
    __properties: ClassVar[List[str]] = ["root_symptom", "attribution_outcome", "detail", "terminal_states", "failure_domains", "cancellation_actors", "uncertainty_type", "decision_graph"]

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
        """Create an instance of Verdict from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of root_symptom
        if self.root_symptom:
            _dict['root_symptom'] = self.root_symptom.to_dict()
        # override the default output from pydantic by calling `to_dict()` of decision_graph
        if self.decision_graph:
            _dict['decision_graph'] = self.decision_graph.to_dict()
        # set to None if root_symptom (nullable) is None
        # and model_fields_set contains the field
        if self.root_symptom is None and "root_symptom" in self.model_fields_set:
            _dict['root_symptom'] = None

        # set to None if uncertainty_type (nullable) is None
        # and model_fields_set contains the field
        if self.uncertainty_type is None and "uncertainty_type" in self.model_fields_set:
            _dict['uncertainty_type'] = None

        # set to None if decision_graph (nullable) is None
        # and model_fields_set contains the field
        if self.decision_graph is None and "decision_graph" in self.model_fields_set:
            _dict['decision_graph'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of Verdict from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "root_symptom": SourcedSymptom.from_dict(obj["root_symptom"]) if obj.get("root_symptom") is not None else None,
            "attribution_outcome": obj.get("attribution_outcome"),
            "detail": obj.get("detail"),
            "terminal_states": obj.get("terminal_states"),
            "failure_domains": obj.get("failure_domains"),
            "cancellation_actors": obj.get("cancellation_actors"),
            "uncertainty_type": obj.get("uncertainty_type"),
            "decision_graph": DecisionGraph.from_dict(obj["decision_graph"]) if obj.get("decision_graph") is not None else None
        })
        return _obj


