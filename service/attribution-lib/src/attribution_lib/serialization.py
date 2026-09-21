# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Serialization utilities for event payloads.

serialize() converts Python objects to JSON-compatible types.
deserialize() is identity - it returns JSON data unchanged.

This means complex objects (Source, StragglerGroup, etc.) are stored as dicts
and read back as dicts. Consuming code should work with dicts, not expect
reconstructed objects.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


def serialize(obj: Any) -> Any:
    """Convert an object to JSON-compatible types.

    Handles:
    - Primitives (str, int, float, bool, None) → pass through
    - datetime → ISO 8601 string
    - Enum → its value
    - mappings → recursively serialize values
    - list/tuple → recursively serialize items
    - dataclass/namedtuple/object → serialize to dict representation
    """
    if obj is None:
        return None

    # Primitives
    if isinstance(obj, (bool, int, float, str)):
        return obj

    # Datetime → ISO string
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Enum → its value
    if isinstance(obj, Enum):
        return obj.value

    # Set/frozenset → list
    if isinstance(obj, (set, frozenset)):
        return [serialize(v) for v in obj]

    # Mappings
    if isinstance(obj, Mapping):
        return {str(k): serialize(v) for k, v in obj.items()}

    # NamedTuple (check BEFORE tuple - namedtuples are tuples)
    if hasattr(obj, "_fields") and hasattr(obj, "_asdict") and not isinstance(obj, type):
        return {k: serialize(v) for k, v in obj._asdict().items()}

    # Dataclass (check BEFORE generic object)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: serialize(getattr(obj, f.name)) for f in fields(obj)}

    # List/tuple (after namedtuple check)
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]

    # ResourceSet (special handling)
    if type(obj).__name__ == "_ResourceSet":
        if obj._universal:
            return "AllJobResources"
        else:
            return [serialize(item) for item in obj]

    # Pydantic models
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return serialize(obj.model_dump())

    # Objects with __dict__
    if hasattr(obj, "__dict__"):
        return {k: serialize(v) for k, v in obj.__dict__.items()}

    # Fallback to string
    return str(obj)


def deserialize(data: Any) -> Any:
    """Pass through JSON data unchanged.

    This is identity - JSON data comes in, same data goes out.
    Complex types are returned as dicts, not reconstructed as objects.
    """
    return data
