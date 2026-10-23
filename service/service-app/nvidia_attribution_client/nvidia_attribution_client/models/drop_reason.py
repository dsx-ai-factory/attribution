# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class DropReason(str, Enum):
    """
    Why an entry was discarded before the extraction rules.
    """

    """
    allowed enum values
    """
    OUTSIDE_WINDOW = 'outside_window'
    NO_TIMESTAMP = 'no_timestamp'
    NO_NODE = 'no_node'
    UNKNOWN_NODE = 'unknown_node'
    SOURCE_SKIPPED_NO_JOB_NODES = 'source_skipped_no_job_nodes'
    PRE_FILTERED = 'pre_filtered'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of DropReason from a JSON string"""
        return cls(json.loads(json_str))


