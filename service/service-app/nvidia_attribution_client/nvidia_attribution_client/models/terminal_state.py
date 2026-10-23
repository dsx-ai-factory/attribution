# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class TerminalState(str, Enum):
    """
    What happened to the job. Always present; every outcome maps to at least one of these (possibly more than one, when NVIDIA Attribution can't narrow it down further -- see \"Incomplete Info\" in `_OUTCOME_TABLE`).
    """

    """
    allowed enum values
    """
    PASS = 'Pass'
    FAIL = 'Fail'
    EXPECTEDFAIL = 'ExpectedFail'
    CANCELLED = 'Cancelled'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of TerminalState from a JSON string"""
        return cls(json.loads(json_str))


