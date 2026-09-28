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


class UncertaintyType(str, Enum):
    """
    NVIDIA Attribution's qualitative classification of the source of uncertainty.  ALEATORIC describes ambiguity in the available job evidence, including insufficient data (\"Incomplete Info\"). It does not assert that the job behaved randomly or that better observations could never resolve it. EPISTEMIC describes limitations of NVIDIA Attribution itself, such as missing rules or bugs (\"Fail Undetermined\"). These are the operational definitions from the compound-outcome RFC, not a general statistical uncertainty model.  Neither value measures likelihood, confidence, severity, or an error rate. None means no uncertainty source is reported, not guaranteed correctness. The source is supplied explicitly, not inferred from the possibility count.
    """

    """
    allowed enum values
    """
    ALEATORIC = 'Aleatoric'
    EPISTEMIC = 'Epistemic'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of UncertaintyType from a JSON string"""
        return cls(json.loads(json_str))


