# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Protocol

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.types import ExecutionContext


class ExecutionContextAdapter(Protocol):
    def discover(self) -> list[ExecutionContext]: ...

    def discover_by_ids(self, ids: list[str]) -> list[ExecutionContext]: ...

    def discover_by_time_range(
        self,
        start: datetime,
        end: datetime,
        nodelist: list[str] | None = None,
    ) -> list[ExecutionContext]: ...


if "eca" not in _registry._FRAMEWORKS:
    _registry.register_framework(name="eca", aliases=("execution_context",), has_selector=True)


__all__ = ["ExecutionContextAdapter"]
