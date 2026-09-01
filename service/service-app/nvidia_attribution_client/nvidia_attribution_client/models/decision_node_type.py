# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class DecisionNodeType(str, Enum):
    """
    Attribution pipeline stage a decision node was produced by.
    """

    """
    allowed enum values
    """
    UNKNOWN = 'UNKNOWN'
    EXTRACTION = 'EXTRACTION'
    FILTER = 'FILTER'
    SCRUB = 'SCRUB'
    REFINER = 'REFINER'
    CASCADE_PRUNE = 'CASCADE_PRUNE'
    OUTCOME = 'OUTCOME'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of DecisionNodeType from a JSON string"""
        return cls(json.loads(json_str))


