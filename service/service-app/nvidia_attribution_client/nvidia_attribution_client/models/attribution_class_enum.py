# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class AttributionClassEnum(str, Enum):
    """
    The types of attributions/errors we can find. V1-only.
    """

    """
    allowed enum values
    """
    EMPTY = ''
    NO_ERROR = 'no_error'
    HARDWARE = 'hardware'
    SOFTWARE = 'software'
    USER = 'user'
    ARTIFICIAL = 'artificial'
    UNDETERMINED = 'undetermined'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of AttributionClassEnum from a JSON string"""
        return cls(json.loads(json_str))


