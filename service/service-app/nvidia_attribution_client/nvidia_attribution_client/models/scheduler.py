# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class Scheduler(str, Enum):
    """
    Recognized values for `JobInfo.scheduler`.  Unlike `JobInfo.type`, which is a free-form legacy wire token, every `JobInfo.scheduler` value is one of these members. `UNKNOWN` means \"no recognized scheduler identity\": callers must not use it to build a rendered ID and should fall back to generic wording. Existing Slurm symptom strings are unchanged.
    """

    """
    allowed enum values
    """
    EMPTY = ''
    SLURM = 'slurm'
    KUEUE = 'kueue'
    KAI = 'kai'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of Scheduler from a JSON string"""
        return cls(json.loads(json_str))


