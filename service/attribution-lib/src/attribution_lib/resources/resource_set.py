# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sys
from collections.abc import (
    Iterable,
    Iterator,
    Set as _Set,
)
from typing import Annotated, Any

from pydantic import ValidationError
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import PlainValidator

from . import Base, Resource


class _ResourceSet(_Set):
    """
    Immutable, hashable container for ResourceBase objects.
    """

    __slots__ = ("_items", "_universal")

    # ---------- Construction -------------------------------------------------
    def __init__(self, resources: Iterable[Base] = (), *, universal: bool = False) -> None:
        if universal and resources:
            raise ValueError("Universal _ResourceSet cannot hold concrete items")
        self._universal: bool = universal
        self._items: frozenset[Base] = frozenset(resources)

    # ---------- Class-level sentinel ----------------------------------------
    #: Singleton representing "any and all resources".
    ANY: "_ResourceSet"  # defined after class body

    # ---------- Set interface -----------------------------------------------
    def __contains__(self, item: object) -> bool:  # type: ignore[override]
        return True if self._universal else item in self._items

    def __iter__(self) -> Iterator[Base]:
        return iter(()) if self._universal else iter(self._items)

    def __len__(self) -> int:
        return sys.maxsize if self._universal else len(self._items)

    # ---------- Hash / equality ---------------------------------------------
    def __hash__(self) -> int:
        return hash(("ANY",)) if self._universal else hash(self._items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ResourceSet):
            return NotImplemented
        if self._universal or other._universal:
            return self._universal and other._universal
        return self._items == other._items

    # ---------- Algebra ------------------------------------------------------
    def union(self, *others: "_ResourceSet") -> "_ResourceSet":
        if self._universal or any(o._universal for o in others):
            return _ResourceSet.ANY
        combined = self._items.union(*(o._items for o in others))
        return _ResourceSet(combined)

    def intersection(self, *others: "_ResourceSet") -> "_ResourceSet":
        if self._universal:
            return others[0] if others else self
        if any(o._universal for o in others):
            return self
        intersected = self._items.intersection(*(o._items for o in others))
        return _ResourceSet(intersected)

    # ---------- Nice repr ----------------------------------------------------
    def __repr__(self) -> str:
        return "_ResourceSet(ANY)" if self._universal else f"_ResourceSet({list(self._items)})"

    def to_dict(self) -> str:
        return serialize_resource_set(self)


# Establish singleton AFTER class definition
_ResourceSet.ANY = _ResourceSet(universal=True)
AllJobResources = _ResourceSet.ANY


def validate_resource_set(value: Any) -> _ResourceSet:
    """
    Validator function for _ResourceSet.
    Accepts either:
    - String "AllJobResources" -> creates _ResourceSet with universal=True
    - List/iterable -> validates each item as Resource and creates _ResourceSet
    - Existing _ResourceSet -> validates and returns as-is
    """

    # Case 1: Already a _ResourceSet instance
    if isinstance(value, _ResourceSet):
        return value

    # Case 2: String "AllJobResources" -> universal set
    elif isinstance(value, str):
        if value == "AllJobResources":
            return _ResourceSet(universal=True)
        else:
            raise ValueError(f"String value must be 'AllJobResources', got '{value}'")

    # Case 3: List/iterable -> validate each item and create _ResourceSet
    elif hasattr(value, "__iter__"):
        try:
            # Convert to list first to validate it's iterable
            value_list = list(value)

            # Validate each item as a Base
            validated_resources = []
            for i, item in enumerate(value_list):
                try:
                    validated_resource = Resource.validate_python(item)
                    validated_resources.append(validated_resource)
                except ValidationError as e:
                    raise ValueError(f"Item at index {i} is not a valid Resource: {e}")

            return _ResourceSet(validated_resources)

        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot create _ResourceSet from iterable: {e}")

    else:
        raise ValueError(
            f"Cannot create _ResourceSet from {type(value).__name__}. "
            f"Expected 'AllJobResources' string, list/iterable, or _ResourceSet instance."
        )


def serialize_resource_set(value: _ResourceSet) -> Any:
    """
    Serializer function for _ResourceSet.
    Returns:
    - "AllJobResources" if universal=True
    - List of items otherwise
    """

    if value._universal:
        return "AllJobResources"
    else:
        # Convert frozenset to list for serialization
        # Note: frozenset is not ordered, so we sort for consistent output
        items_list = sorted(list(value._items), key=lambda x: (x.kind, str(x)))
        return items_list


# Public alias for the concrete class — use this to construct instances.
# ``ResourceSet`` below is an Annotated wrapper for Pydantic; it is a type
# alias, not a callable.
ResourceSetClass = _ResourceSet

# Create the Annotated type that combines validation and serialization
ResourceSet = Annotated[
    _ResourceSet,
    PlainValidator(validate_resource_set),
    PlainSerializer(serialize_resource_set, return_type=str | list, when_used="always"),
]
