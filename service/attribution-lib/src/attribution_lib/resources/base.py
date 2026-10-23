# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


def set_key(*, key: str) -> Callable[[type["Base"]], type["Base"]]:
    """
    Parameters
    ----------
    key : str
        Name of the field whose *value* becomes the resource's unique `key`.
    """

    def decorator(cls: type["Base"]) -> type["Base"]:
        # ------- 1. Validation -----------------------
        if key == "key":
            raise ValueError("`key` field cannot be its own key source")
        if key not in cls.__annotations__:
            raise AttributeError(f"{cls.__name__} has no field '{key}'")

        cls.__key_field__ = key  # opt-in introspection hook
        return cls

    return decorator


class Base(BaseModel):
    """
    Root model: immutable + hashable on (kind, key).
    """

    __key_field__: ClassVar[str]

    kind: str = Field(..., description="Discriminator")
    key: str = Field(default="", description="Unique identifier — auto-populated by __init__ from @set_key field")

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __init__(self, **data: Any) -> None:
        """
        Auto-populate ``key`` from the configured source field.

        The algorithm:

        1. If a caller already supplied ``key``, respect it.
        2. Otherwise, derive the key from ``self.__key_field__``.
        3. Delegate to Pydantic for full validation.
        """

        # 1. Bail out early if user already provided a key -------------------
        if "key" in data:
            super().__init__(**data)
            return  # ← **VALID** early exit

        # 2. Attempt automatic derivation -----------------------------------
        key_source = getattr(self.__class__, "__key_field__", None)
        if key_source is None:
            raise ValueError(
                f"{self.__class__.__name__} lacks '__key_field__' "
                "metadata—did you forget @register_resource(key='field')?"
            )

        if key_source in data:
            data["key"] = data[key_source]
        else:
            # Handle default-value case
            field_info = self.model_fields.get(key_source)
            default_val = getattr(field_info, "default", None)
            if default_val is not None:
                data["key"] = default_val
            else:
                raise ValueError(f"Cannot derive key: field '{key_source}' not provided and has no default.")

        # 3. Standard Pydantic validation -----------------------------------
        super().__init__(**data)

    def __hash__(self) -> int:
        return hash((self.kind, self.key))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Base) and (self.kind, self.key) == (other.kind, other.key)
