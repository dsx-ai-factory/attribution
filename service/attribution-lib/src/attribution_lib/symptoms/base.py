# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symptom interfaces for attribution."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Optional

from attribution_lib.symptoms.family import SymptomFamily


class BaseSymptom:
    """Runtime symptom instance - carries metadata through the system.
    Base contract:
    - symptom_id: The actual symptom identifier string (e.g., "XID-13-", "IB Error-12")
    - family: Optional structured family identity
    - sources: Where this symptom was detected (log lines, metrics, etc.)
    - attributes: Additional metadata about this symptom occurrence
    - cascade_discards: List of symptom patterns to discard when this symptom is present
    """

    def __init__(
        self,
        *,
        symptom_id: str,
        family: Optional[SymptomFamily] = None,
        sources: Optional[Sequence[Any]] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        cascade_discards: Optional[Sequence[str]] = None,
    ):
        if family is not None and len(family.property_types) > 0:
            raise ValueError("BaseSymptom family must contain member properties, not property_types")
        self._symptom_id = symptom_id
        self._family = family
        self._sources: list[Any] = list(sources or [])
        self._attributes: dict[str, Any] = dict(attributes or {})
        self._cascade_discards: list[str] = list(cascade_discards or [])

    @property
    def symptom_id(self) -> str:
        """The symptom identifier string (e.g., 'XID-13-', 'IB Error-12')."""
        return self._symptom_id

    @property
    def family(self) -> Optional[SymptomFamily]:
        """Return structured family identity supplied by declaration or emission."""
        return self._family

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
        """Serialize the symptom, including structured identity when present.

        Returns:
            Symptom data in its compatible dictionary representation.
        """
        result: dict[str, Any] = {
            "symptom_id": self._symptom_id,
            "sources": self._sources,
            "attributes": dict(self._attributes),
            "cascade_discards": list(self._cascade_discards),
        }
        if self._family is not None:
            result["family"] = {
                "name": self._family.name,
                "properties": dict(self._family.properties),
            }
        return result

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
