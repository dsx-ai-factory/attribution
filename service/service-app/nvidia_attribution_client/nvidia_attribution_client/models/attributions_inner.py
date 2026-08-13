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
import json
import pprint
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, field_validator
from typing import Any, List, Optional
from nvidia_attribution_client.models.global_attribution import GlobalAttribution
from nvidia_attribution_client.models.node_attribution import NodeAttribution
from nvidia_attribution_client.models.storage_attribution import StorageAttribution
from nvidia_attribution_client.models.switch_attribution import SwitchAttribution
from nvidia_attribution_client.models.workload_record import WorkloadRecord
from pydantic import StrictStr, Field
from typing import Union, List, Set, Optional, Dict
from typing_extensions import Literal, Self

ATTRIBUTIONSINNER_ONE_OF_SCHEMAS = ["GlobalAttribution", "NodeAttribution", "StorageAttribution", "SwitchAttribution", "WorkloadRecord"]

class AttributionsInner(BaseModel):
    """
    AttributionsInner
    """
    # data type: WorkloadRecord
    oneof_schema_1_validator: Optional[WorkloadRecord] = None
    # data type: NodeAttribution
    oneof_schema_2_validator: Optional[NodeAttribution] = None
    # data type: StorageAttribution
    oneof_schema_3_validator: Optional[StorageAttribution] = None
    # data type: SwitchAttribution
    oneof_schema_4_validator: Optional[SwitchAttribution] = None
    # data type: GlobalAttribution
    oneof_schema_5_validator: Optional[GlobalAttribution] = None
    actual_instance: Optional[Union[GlobalAttribution, NodeAttribution, StorageAttribution, SwitchAttribution, WorkloadRecord]] = None
    one_of_schemas: Set[str] = { "GlobalAttribution", "NodeAttribution", "StorageAttribution", "SwitchAttribution", "WorkloadRecord" }

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
        instance = AttributionsInner.model_construct()
        error_messages = []
        match = 0
        # validate data type: WorkloadRecord
        if not isinstance(v, WorkloadRecord):
            error_messages.append(f"Error! Input type `{type(v)}` is not `WorkloadRecord`")
        else:
            match += 1
        # validate data type: NodeAttribution
        if not isinstance(v, NodeAttribution):
            error_messages.append(f"Error! Input type `{type(v)}` is not `NodeAttribution`")
        else:
            match += 1
        # validate data type: StorageAttribution
        if not isinstance(v, StorageAttribution):
            error_messages.append(f"Error! Input type `{type(v)}` is not `StorageAttribution`")
        else:
            match += 1
        # validate data type: SwitchAttribution
        if not isinstance(v, SwitchAttribution):
            error_messages.append(f"Error! Input type `{type(v)}` is not `SwitchAttribution`")
        else:
            match += 1
        # validate data type: GlobalAttribution
        if not isinstance(v, GlobalAttribution):
            error_messages.append(f"Error! Input type `{type(v)}` is not `GlobalAttribution`")
        else:
            match += 1
        if match > 1:
            # more than 1 match
            raise ValueError("Multiple matches found when setting `actual_instance` in AttributionsInner with oneOf schemas: GlobalAttribution, NodeAttribution, StorageAttribution, SwitchAttribution, WorkloadRecord. Details: " + ", ".join(error_messages))
        elif match == 0:
            # no match
            raise ValueError("No match found when setting `actual_instance` in AttributionsInner with oneOf schemas: GlobalAttribution, NodeAttribution, StorageAttribution, SwitchAttribution, WorkloadRecord. Details: " + ", ".join(error_messages))
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

        # deserialize data into WorkloadRecord
        try:
            instance.actual_instance = WorkloadRecord.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into NodeAttribution
        try:
            instance.actual_instance = NodeAttribution.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into StorageAttribution
        try:
            instance.actual_instance = StorageAttribution.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into SwitchAttribution
        try:
            instance.actual_instance = SwitchAttribution.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into GlobalAttribution
        try:
            instance.actual_instance = GlobalAttribution.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))

        if match > 1:
            # more than 1 match
            raise ValueError("Multiple matches found when deserializing the JSON string into AttributionsInner with oneOf schemas: GlobalAttribution, NodeAttribution, StorageAttribution, SwitchAttribution, WorkloadRecord. Details: " + ", ".join(error_messages))
        elif match == 0:
            # no match
            raise ValueError("No match found when deserializing the JSON string into AttributionsInner with oneOf schemas: GlobalAttribution, NodeAttribution, StorageAttribution, SwitchAttribution, WorkloadRecord. Details: " + ", ".join(error_messages))
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

    def to_dict(self) -> Optional[Union[Dict[str, Any], GlobalAttribution, NodeAttribution, StorageAttribution, SwitchAttribution, WorkloadRecord]]:
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


