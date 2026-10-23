# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class WorkloadTypeEnum(str, Enum):
    """
    The types of workload identifiers we can find.
    """

    """
    allowed enum values
    """
    EMPTY = ''
    SLURM = 'slurm'
    KUEUE = 'kueue'
    KAI = 'kai'
    RUNAI = 'runai'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of WorkloadTypeEnum from a JSON string"""
        return cls(json.loads(json_str))


