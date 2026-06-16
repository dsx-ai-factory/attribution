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


class StragglerScoreType(int, Enum):
    """
    Type of scoring Straggler Detector supports.
    """

    """
    allowed enum values
    """
    NUMBER_0 = 0
    NUMBER_1 = 1

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of StragglerScoreType from a JSON string"""
        return cls(json.loads(json_str))


