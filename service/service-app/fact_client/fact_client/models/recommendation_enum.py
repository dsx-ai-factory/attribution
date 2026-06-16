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


class RecommendationEnum(str, Enum):
    """
    The types of attributions/errors we can find.
    """

    """
    allowed enum values
    """
    EMPTY = ''
    DRAIN = 'drain'
    NO_ERROR = 'no_error'
    AMBIGUOUS = 'ambiguous'
    UNKNOWN = 'unknown'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of RecommendationEnum from a JSON string"""
        return cls(json.loads(json_str))


