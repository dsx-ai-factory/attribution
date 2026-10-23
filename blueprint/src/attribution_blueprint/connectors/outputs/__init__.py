# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.types import ExecutionContext


class OutputConnector(Protocol):
    def write(self, context: ExecutionContext, attribution: dict) -> None: ...
    def close(self) -> None: ...


if "output" not in _registry._FRAMEWORKS:
    _registry.register_framework(name="output", instances=True)


__all__ = ["OutputConnector"]
