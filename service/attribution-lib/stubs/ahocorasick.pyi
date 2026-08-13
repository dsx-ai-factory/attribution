# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Iterator
from typing import Any, Generic, TypeVar

_Value = TypeVar("_Value")

class Automaton(Generic[_Value]):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def add_word(self, word: str, value: _Value) -> bool: ...
    def make_automaton(self) -> None: ...
    def iter(self, string: str) -> Iterator[tuple[int, _Value]]: ...
    def items(self) -> Iterator[tuple[str, _Value]]: ...
