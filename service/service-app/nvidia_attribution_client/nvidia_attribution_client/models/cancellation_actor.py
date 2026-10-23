# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class CancellationActor(str, Enum):
    """
    Possible initiators carried by `Cancelled`.
    """

    """
    allowed enum values
    """
    USER = 'User'
    OPERATOR = 'Operator'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of CancellationActor from a JSON string"""
        return cls(json.loads(json_str))


