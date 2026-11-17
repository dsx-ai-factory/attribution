# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import attribution_lib
from attribution_lib.symptoms.base import BaseSymptom


class SymptomFactory:
    """Factory for creating symptom objects."""

    @staticmethod
    def from_enum(
        symptom_enum,
        *,
        sources: Optional[list[str]],
        attributes: Optional[dict[str, Any]],
    ) -> BaseSymptom:
        """Create a symptom from a registered symptom enum.

        If the enum id does not resolve to metadata, we still construct a
        BaseSymptom with no cascade_discards.
        """
        registry = attribution_lib.get_symptom_registry()
        symptom_id = symptom_enum.value
        symptom_metadata = registry.metadata_for(symptom_id)
        return BaseSymptom(
            symptom_id=symptom_id,
            sources=sources,
            attributes=attributes,
            cascade_discards=(list(symptom_metadata.cascade_discards) if symptom_metadata is not None else []),
        )

    @staticmethod
    def from_string(
        symptom_id: str,
        *,
        sources: Optional[list[str]],
        attributes: Optional[dict[str, Any]],
    ) -> BaseSymptom:
        """Create a symptom from a string (for backward compatibility with legacy code).

        Raises:
            ValueError: If the symptom ID does not resolve to registered metadata.
        """
        registry = attribution_lib.get_symptom_registry()
        symptom_metadata = registry.metadata_for(symptom_id)
        # Enforce that free-form strings are only allowed if declared exactly
        # or covered by an explicit fallback prefix.
        if symptom_metadata is None:
            raise ValueError(f"Unregistered symptom id: '{symptom_id}'")
        return BaseSymptom(
            symptom_id=symptom_id,
            sources=sources,
            attributes=attributes,
            cascade_discards=(list(symptom_metadata.cascade_discards) if symptom_metadata is not None else []),
        )
