# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.types import ExecutionContext, Observation, Source


class LogConnector(Protocol):
    @property
    def source(self) -> Source: ...

    def collect(self, context: ExecutionContext) -> list[Observation]: ...


if "log" not in _registry._FRAMEWORKS:
    _registry.register_framework(name="log", aliases=("logs",), sources=("applog", "syslog"))


__all__ = ["LogConnector"]
