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

from datetime import datetime
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr
from typing import Any, ClassVar, Dict, List, Optional
from nvidia_attribution_client.models.scheduler import Scheduler
from typing import Optional, Set
from typing_extensions import Self
from pydantic_core import to_jsonable_python

class AttributorWorkloadInfoV2(BaseModel):
    """
    V2 wire shape: ``nodes`` is the explicit node->ranks mapping.
    """ # noqa: E501
    id: StrictStr
    type: StrictStr
    scheduler: Optional[Scheduler] = None
    srun_cmd: Optional[StrictStr] = None
    job_start_time: datetime
    timezone: StrictStr
    job_end_time: Optional[datetime] = None
    interest_start_time: Optional[datetime] = None
    interest_end_time: Optional[datetime] = None
    exit_code_signal: Optional[StrictInt] = None
    cancelled_by: Optional[StrictStr] = None
    time_limit: Optional[StrictInt] = None
    failed_node: Optional[StrictStr] = None
    status: StrictStr
    nodes: Dict[str, List[StrictInt]]
    name: StrictStr
    username: StrictStr
    last_tflops_time: Optional[datetime] = None
    framework: StrictStr
    __properties: ClassVar[List[str]] = ["id", "type", "scheduler", "srun_cmd", "job_start_time", "timezone", "job_end_time", "interest_start_time", "interest_end_time", "exit_code_signal", "cancelled_by", "time_limit", "failed_node", "status", "nodes", "name", "username", "last_tflops_time", "framework"]

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
        """Create an instance of AttributorWorkloadInfoV2 from a JSON string"""
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
        # set to None if scheduler (nullable) is None
        # and model_fields_set contains the field
        if self.scheduler is None and "scheduler" in self.model_fields_set:
            _dict['scheduler'] = None

        # set to None if srun_cmd (nullable) is None
        # and model_fields_set contains the field
        if self.srun_cmd is None and "srun_cmd" in self.model_fields_set:
            _dict['srun_cmd'] = None

        # set to None if job_end_time (nullable) is None
        # and model_fields_set contains the field
        if self.job_end_time is None and "job_end_time" in self.model_fields_set:
            _dict['job_end_time'] = None

        # set to None if interest_start_time (nullable) is None
        # and model_fields_set contains the field
        if self.interest_start_time is None and "interest_start_time" in self.model_fields_set:
            _dict['interest_start_time'] = None

        # set to None if interest_end_time (nullable) is None
        # and model_fields_set contains the field
        if self.interest_end_time is None and "interest_end_time" in self.model_fields_set:
            _dict['interest_end_time'] = None

        # set to None if exit_code_signal (nullable) is None
        # and model_fields_set contains the field
        if self.exit_code_signal is None and "exit_code_signal" in self.model_fields_set:
            _dict['exit_code_signal'] = None

        # set to None if cancelled_by (nullable) is None
        # and model_fields_set contains the field
        if self.cancelled_by is None and "cancelled_by" in self.model_fields_set:
            _dict['cancelled_by'] = None

        # set to None if time_limit (nullable) is None
        # and model_fields_set contains the field
        if self.time_limit is None and "time_limit" in self.model_fields_set:
            _dict['time_limit'] = None

        # set to None if failed_node (nullable) is None
        # and model_fields_set contains the field
        if self.failed_node is None and "failed_node" in self.model_fields_set:
            _dict['failed_node'] = None

        # set to None if last_tflops_time (nullable) is None
        # and model_fields_set contains the field
        if self.last_tflops_time is None and "last_tflops_time" in self.model_fields_set:
            _dict['last_tflops_time'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of AttributorWorkloadInfoV2 from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "id": obj.get("id"),
            "type": obj.get("type"),
            "scheduler": obj.get("scheduler"),
            "srun_cmd": obj.get("srun_cmd"),
            "job_start_time": obj.get("job_start_time"),
            "timezone": obj.get("timezone"),
            "job_end_time": obj.get("job_end_time"),
            "interest_start_time": obj.get("interest_start_time"),
            "interest_end_time": obj.get("interest_end_time"),
            "exit_code_signal": obj.get("exit_code_signal"),
            "cancelled_by": obj.get("cancelled_by"),
            "time_limit": obj.get("time_limit"),
            "failed_node": obj.get("failed_node"),
            "status": obj.get("status"),
            "nodes": obj.get("nodes"),
            "name": obj.get("name"),
            "username": obj.get("username"),
            "last_tflops_time": obj.get("last_tflops_time"),
            "framework": obj.get("framework")
        })
        return _obj


