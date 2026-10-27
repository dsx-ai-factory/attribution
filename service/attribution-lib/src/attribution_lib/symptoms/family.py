# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured identity for related symptoms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, UnionType
from typing import TypeAlias, TypeVar, get_args

FamilyPropertyType: TypeAlias = type | UnionType
_Value = TypeVar("_Value")


def _immutable_mapping(values: Mapping[str, _Value], *, field_name: str) -> Mapping[str, _Value]:
    """Validate and defensively freeze a family mapping.

    Args:
        values: Property names and their type specifications or resolved values.
        field_name: Public field name used in validation errors.

    Returns:
        An immutable copy of ``values``.

    Raises:
        ValueError: If a property name is empty.
        TypeError: If a property name is not a string or a value is not hashable.
    """
    copied = dict(values)
    for name, value in copied.items():
        if not isinstance(name, str):
            raise TypeError(f"SymptomFamily {field_name} names must be strings")
        if name.strip() == "":
            raise ValueError(f"SymptomFamily {field_name} names must be non-empty")
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"SymptomFamily {field_name}[{name!r}] must be hashable") from exc
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class SymptomFamily:
    """Describe either a symptom-family schema or one family member.

    A declaration may supply ``property_types`` and a concrete member may supply
    ``properties``; both can be empty for name-only membership. The mutually
    exclusive forms let declarations, emitted events, and runtime symptoms share one representation.

    Attributes:
        name: Rule-defined family name.
        property_types: Property schema supplied by a family declaration.
        properties: Resolved property values supplied by a family member.
    """

    name: str
    property_types: Mapping[str, FamilyPropertyType] = field(default_factory=dict)
    properties: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the family form and freeze caller-owned mappings.

        Raises:
            ValueError: If the name is empty or both family forms are supplied.
            TypeError: If a name, type, or property value is invalid.
        """
        if not isinstance(self.name, str):
            raise TypeError("SymptomFamily name must be a string")
        if self.name.strip() == "":
            raise ValueError("SymptomFamily name must be non-empty")
        if len(self.property_types) > 0 and len(self.properties) > 0:
            raise ValueError("SymptomFamily cannot define both property_types and properties")
        for name, property_type in self.property_types.items():
            union_members = get_args(property_type) if isinstance(property_type, UnionType) else ()
            members = union_members if len(union_members) > 0 else (property_type,)
            try:
                for member in members:
                    _ = isinstance(None, member)
            except TypeError as exc:
                raise TypeError(f"SymptomFamily property_types[{name!r}] must be a runtime-checkable type") from exc
        object.__setattr__(
            self,
            "property_types",
            _immutable_mapping(self.property_types, field_name="property_types"),
        )
        object.__setattr__(
            self,
            "properties",
            _immutable_mapping(self.properties, field_name="properties"),
        )

    def __hash__(self) -> int:
        """Return a hash independent of input mapping order.

        Returns:
            Hash derived from the complete structured family value.
        """
        return hash(
            (
                self.name,
                tuple(sorted(self.property_types.items())),
                tuple(sorted(self.properties.items())),
            )
        )
