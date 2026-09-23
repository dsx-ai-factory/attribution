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
from enum import Enum
from typing_extensions import Self


class FailureDomain(str, Enum):
    """
    Possible domains of responsibility carried by `Fail`.
    """

    """
    allowed enum values
    """
    USER = 'User'
    INFRAHW = 'InfraHW'
    INFRASW = 'InfraSW'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of FailureDomain from a JSON string"""
        return cls(json.loads(json_str))


