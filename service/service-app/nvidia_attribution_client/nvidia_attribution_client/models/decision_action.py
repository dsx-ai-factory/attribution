# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class DecisionAction(str, Enum):
    """
    Effect a decision node had on the symptom it references.
    """

    """
    allowed enum values
    """
    UNKNOWN = 'UNKNOWN'
    CREATED = 'CREATED'
    MATCHED = 'MATCHED'
    REMOVED = 'REMOVED'
    PRUNED = 'PRUNED'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of DecisionAction from a JSON string"""
        return cls(json.loads(json_str))


