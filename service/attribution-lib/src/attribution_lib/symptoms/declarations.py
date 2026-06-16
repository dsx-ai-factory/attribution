# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from attribution_lib.rules_engine import RuleLocation


class OutcomePriority(IntEnum):
    """Outcome selection priority tiers.

    Lower values win. The 16 values map 1:1 to the ordered outcome tiers that the
    generated registry consumes for outcome selection.
    """

    SENTINEL = 0
    CRITICAL = 100
    CSP = 200
    HIGH = 300
    HIGH_HW = 400
    HIGH_XID145 = 500
    MEDIUM = 600
    MEDIUM_XID = 700
    MEDIUM_BROAD = 800
    MEDIUM_SW = 900
    LOW = 1000
    LOW_GENERIC = 1100
    LOW_NETWORK = 1200
    LOW_USER = 1300
    INCOMPLETE = 1400
    FALLBACK = 1500


@dataclass(frozen=True, slots=True)
class SymptomDeclaration:
    """Declarative metadata for a symptom prefix.

    Outcome declarations populate outcome, priority, and doc. Cascade-only
    declarations leave those fields empty and only describe pruning edges.
    """

    prefix: str
    outcome: Optional[str] = None
    hw_cause: str = "Not Applicable"
    priority: Optional[OutcomePriority] = None
    doc: Optional[str] = None
    cascade_discards: tuple[str, ...] = ()
    conditional_discards: tuple[tuple[str, str], ...] = ()
    cascade_noop: bool = False
    # When True, this prefix participates in dynamic symptom resolution as a
    # generic fallback (e.g. "XID-", "NV_ERR_"): raw symptom ids that start with
    # this prefix resolve to it instead of being treated as unknown.
    fallback_prefix: bool = False
    location: Optional[RuleLocation] = field(default=None, compare=False)
