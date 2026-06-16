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

        If the enum id is not present in the metadata registry, we still
        construct a BaseSymptom with no cascade_discards.
        """
        registry = attribution_lib.get_symptom_registry()
        symptom_id = symptom_enum.value
        symptom_metadata = registry.get(symptom_id)
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
            ValueError: if the provided symptom_id is not registered in the metadata registry.
        """
        registry = attribution_lib.get_symptom_registry()
        # Enforce that free-form strings are only allowed if present in registry
        if not registry.has_metadata(symptom_id):
            raise ValueError(f"Unregistered symptom id: '{symptom_id}'")
        symptom_metadata = registry.get(symptom_id)
        return BaseSymptom(
            symptom_id=symptom_id,
            sources=sources,
            attributes=attributes,
            cascade_discards=(list(symptom_metadata.cascade_discards) if symptom_metadata is not None else []),
        )
