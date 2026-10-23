# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import json
import pprint
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, field_validator
from typing import Any, List, Optional
from nvidia_attribution_client.models.node_source import NodeSource
from nvidia_attribution_client.models.resourceless_source import ResourcelessSource
from pydantic import StrictStr, Field
from typing import Union, List, Set, Optional, Dict
from typing_extensions import Literal, Self

SYMPTOMSVALUEINNER_ONE_OF_SCHEMAS = ["NodeSource", "ResourcelessSource"]

class SymptomsValueInner(BaseModel):
    """
    SymptomsValueInner
    """
    # data type: ResourcelessSource
    oneof_schema_1_validator: Optional[ResourcelessSource] = None
    # data type: NodeSource
    oneof_schema_2_validator: Optional[NodeSource] = None
    actual_instance: Optional[Union[NodeSource, ResourcelessSource]] = None
    one_of_schemas: Set[str] = { "NodeSource", "ResourcelessSource" }

    model_config = ConfigDict(
        validate_assignment=True,
        protected_namespaces=(),
    )


    discriminator_value_class_map: Dict[str, str] = {
    }

    def __init__(self, *args, **kwargs) -> None:
        if args:
            if len(args) > 1:
                raise ValueError("If a position argument is used, only 1 is allowed to set `actual_instance`")
            if kwargs:
                raise ValueError("If a position argument is used, keyword arguments cannot be used.")
            super().__init__(actual_instance=args[0])
        else:
            super().__init__(**kwargs)

    @field_validator('actual_instance')
    def actual_instance_must_validate_oneof(cls, v):
        instance = SymptomsValueInner.model_construct()
        error_messages = []
        match = 0
        # validate data type: ResourcelessSource
        if not isinstance(v, ResourcelessSource):
            error_messages.append(f"Error! Input type `{type(v)}` is not `ResourcelessSource`")
        else:
            match += 1
        # validate data type: NodeSource
        if not isinstance(v, NodeSource):
            error_messages.append(f"Error! Input type `{type(v)}` is not `NodeSource`")
        else:
            match += 1
        if match > 1:
            # more than 1 match
            raise ValueError("Multiple matches found when setting `actual_instance` in SymptomsValueInner with oneOf schemas: NodeSource, ResourcelessSource. Details: " + ", ".join(error_messages))
        elif match == 0:
            # no match
            raise ValueError("No match found when setting `actual_instance` in SymptomsValueInner with oneOf schemas: NodeSource, ResourcelessSource. Details: " + ", ".join(error_messages))
        else:
            return v

    @classmethod
    def from_dict(cls, obj: Union[str, Dict[str, Any]]) -> Self:
        return cls.from_json(json.dumps(obj))

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Returns the object represented by the json string"""
        instance = cls.model_construct()
        error_messages = []
        match = 0

        # deserialize data into ResourcelessSource
        try:
            instance.actual_instance = ResourcelessSource.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into NodeSource
        try:
            instance.actual_instance = NodeSource.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))

        if match > 1:
            # more than 1 match
            raise ValueError("Multiple matches found when deserializing the JSON string into SymptomsValueInner with oneOf schemas: NodeSource, ResourcelessSource. Details: " + ", ".join(error_messages))
        elif match == 0:
            # no match
            raise ValueError("No match found when deserializing the JSON string into SymptomsValueInner with oneOf schemas: NodeSource, ResourcelessSource. Details: " + ", ".join(error_messages))
        else:
            return instance

    def to_json(self) -> str:
        """Returns the JSON representation of the actual instance"""
        if self.actual_instance is None:
            return "null"

        if hasattr(self.actual_instance, "to_json") and callable(self.actual_instance.to_json):
            return self.actual_instance.to_json()
        else:
            return json.dumps(self.actual_instance)

    def to_dict(self) -> Optional[Union[Dict[str, Any], NodeSource, ResourcelessSource]]:
        """Returns the dict representation of the actual instance"""
        if self.actual_instance is None:
            return None

        if hasattr(self.actual_instance, "to_dict") and callable(self.actual_instance.to_dict):
            return self.actual_instance.to_dict()
        else:
            # primitive type
            return self.actual_instance

    def to_str(self) -> str:
        """Returns the string representation of the actual instance"""
        return pprint.pformat(self.model_dump())


