# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Semantic symptom selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from attribution_lib.symptoms.base import BaseSymptom

_MISSING_PROPERTY = object()


@dataclass(frozen=True, slots=True)
class SymptomSelector:
    """Select runtime symptoms by exact ID, family, and family properties.

    Supplied criteria are conjunctive. A ``frozenset`` property criterion means
    that any value in the set is accepted; other values use equality.

    Attributes:
        symptom_id: Optional exact runtime symptom ID.
        family_name: Optional symptom-family name.
        family_properties: Required family property values.
    """

    symptom_id: Optional[str] = None
    family_name: Optional[str] = None
    family_properties: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate selector criteria and freeze caller-owned mappings.

        Raises:
            ValueError: If required criteria are absent or empty.
            TypeError: If a criterion has an invalid type or is not hashable.
        """
        if self.symptom_id is not None and not isinstance(self.symptom_id, str):
            raise TypeError("SymptomSelector symptom_id must be a string when supplied")
        if self.symptom_id is not None and self.symptom_id.strip() == "":
            raise ValueError("SymptomSelector symptom_id must be non-empty when supplied")
        if self.family_name is not None and not isinstance(self.family_name, str):
            raise TypeError("SymptomSelector family_name must be a string when supplied")
        if self.family_name is not None and self.family_name.strip() == "":
            raise ValueError("SymptomSelector family_name must be non-empty when supplied")
        if self.symptom_id is None and self.family_name is None:
            raise ValueError("SymptomSelector requires symptom_id or family_name")
        if len(self.family_properties) > 0 and self.family_name is None:
            raise ValueError("SymptomSelector family_properties require family_name")

        copied_properties = dict(self.family_properties)
        for name, expected in copied_properties.items():
            if not isinstance(name, str):
                raise TypeError("SymptomSelector family property names must be strings")
            if name.strip() == "":
                raise ValueError("SymptomSelector family property names must be non-empty")
            try:
                hash(expected)
            except TypeError as exc:
                raise TypeError(f"SymptomSelector property {name!r} must be hashable") from exc
        object.__setattr__(self, "family_properties", MappingProxyType(copied_properties))

    def member(self, symptom: BaseSymptom) -> bool:
        """Return whether a runtime symptom satisfies every selector criterion.

        Args:
            symptom: Runtime symptom to evaluate.

        Returns:
            True when all supplied identity criteria match.
        """
        if self.symptom_id is not None and symptom.symptom_id != self.symptom_id:
            return False
        if self.family_name is None:
            return True

        family = symptom.family
        if family is None or family.name != self.family_name:
            return False
        for name, expected in self.family_properties.items():
            actual = family.properties.get(name, _MISSING_PROPERTY)
            if isinstance(expected, frozenset):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def __hash__(self) -> int:
        """Return a hash independent of property mapping order.

        Returns:
            Hash derived from every selector criterion.
        """
        return hash(
            (
                self.symptom_id,
                self.family_name,
                tuple(sorted(self.family_properties.items())),
            )
        )
