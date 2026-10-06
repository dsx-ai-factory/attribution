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
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Optional

from attribution_lib.symptoms.family import SymptomFamily

if TYPE_CHECKING:
    from attribution_lib.rule_types.definitions import RuleLocation


class OutcomePriority(IntEnum):
    """Outcome selection priority tiers.

    Lower values win. The values map 1:1 to the ordered outcome tiers that the
    generated registry consumes for outcome selection.
    """

    SENTINEL = 0
    CRITICAL = 100
    CSP = 200
    HIGH = 300
    HIGH_HW = 400
    HIGH_KERNEL_PANIC = 450
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


class CascadeScope(str, Enum):
    """Resource scope where cascade discards apply."""

    NODE = "node"
    NVL_DOMAIN = "nvl_domain"


class SymptomCategory(str, Enum):
    """Semantic category assigned to related symptoms."""

    CSP_EVENT = "csp_event"


class DagDiscardBehavior(str, Enum):
    """What happens to a bad symptom when the DAG discards it during projection.

    DEMOTE (default): the symptom is demoted bad -> interesting. KEEP: it stays a
    bad symptom. PURGE: it is removed from the node entirely.
    """

    DEMOTE = "demote"
    KEEP = "keep"
    PURGE = "purge"


@dataclass(frozen=True, slots=True)
class SymptomDeclaration:
    """Declarative metadata for a symptom prefix.

    Outcome declarations populate outcome, priority, and doc. Cascade-only
    declarations leave those fields empty and describe pruning edges for one
    resource scope. Any declaration may assign the symptom to a semantic
    category for registry queries and structured family identity.
    """

    prefix: str
    outcome: Optional[str] = None
    hw_cause: str = "Not Applicable"
    priority: Optional[OutcomePriority] = None
    doc: Optional[str] = None
    cascade_discards: tuple[str, ...] = ()
    conditional_discards: tuple[tuple[str, str], ...] = ()
    cascade_noop: bool = False
    scope: CascadeScope = CascadeScope.NODE
    category: Optional[SymptomCategory] = None
    # When True, this prefix participates in dynamic symptom resolution as a
    # generic fallback (e.g. "XID-", "NV_ERR_"): raw symptom ids that start with
    # this prefix resolve to it instead of being treated as unknown.
    fallback_prefix: bool = False
    # Behavior when the DAG discards this symptom while projecting back to bad nodes.
    dag_discard: DagDiscardBehavior = DagDiscardBehavior.DEMOTE
    # Substring alias: any raw symptom id CONTAINING this substring resolves to this
    # prefix (e.g. NON_FATAL declares "non-fatal" so "(non-fatal) X" resolves to it).
    resolve_contains: Optional[str] = None
    location: Optional[RuleLocation] = field(default=None, compare=False)
    # Appended to preserve positional compatibility for existing constructors.
    family: Optional[SymptomFamily] = None
