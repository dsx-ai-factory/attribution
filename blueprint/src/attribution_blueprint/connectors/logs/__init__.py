# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
