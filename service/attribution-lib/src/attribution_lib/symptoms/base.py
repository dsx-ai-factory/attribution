# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Symptom interfaces for attribution."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Optional


class BaseSymptom:
    """Runtime symptom instance - carries metadata through the system.
    Base contract:
    - symptom_id: The actual symptom identifier string (e.g., "XID-13-", "IB Error-12")
    - sources: Where this symptom was detected (log lines, metrics, etc.)
    - attributes: Additional metadata about this symptom occurrence
    - cascade_discards: List of symptom patterns to discard when this symptom is present
    """

    def __init__(
        self,
        *,
        symptom_id: str,
        sources: Optional[Sequence[Any]] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        cascade_discards: Optional[Sequence[str]] = None,
    ):
        self._symptom_id = symptom_id
        self._sources: list[Any] = list(sources or [])
        self._attributes: dict[str, Any] = dict(attributes or {})
        self._cascade_discards: list[str] = list(cascade_discards or [])

    @property
    def symptom_id(self) -> str:
        """The symptom identifier string (e.g., 'XID-13-', 'IB Error-12')."""
        return self._symptom_id

    @property
    def sources(self) -> list[Any]:
        """Sources where this symptom was detected (mutable list)."""
        return self._sources

    @property
    def attributes(self) -> MutableMapping[str, Any]:
        """Additional metadata about this symptom."""
        return self._attributes

    @property
    def cascade_discards(self) -> list[str]:
        """List of symptom patterns to discard when this symptom is present."""
        return self._cascade_discards

    def add_source(self, source: Any) -> None:
        """Add a source to this symptom."""
        self._sources.append(source)

    def add_sources(self, sources: Sequence[Any]) -> None:
        """Add multiple sources to this symptom."""
        self._sources.extend(sources)

    def update(self, updates: Mapping[str, Any]) -> None:
        """Update symptom attributes."""
        self._attributes.update(updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symptom_id": self._symptom_id,
            "sources": self._sources,
            "attributes": dict(self._attributes),
            "cascade_discards": list(self._cascade_discards),
        }

    def matches(self, pattern: str) -> bool:
        """Check if this symptom matches a given pattern.
        Matches against symptom_id (exact or substring match).
        """
        return pattern in self._symptom_id

    def __eq__(self, other: Any) -> bool:
        """Enable equality comparison for set operations using string form."""
        return str(self) == str(other)

    def __hash__(self) -> int:
        # Keep hash consistent with __eq__ (both use string form)
        return hash(str(self))

    def __str__(self) -> str:
        return self._symptom_id
