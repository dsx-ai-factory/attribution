# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol

from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import (
    PlacementQuery,
    PodPlacement,
)


class PlacementResolver(Protocol):
    def placements_for_many(self, queries: list[PlacementQuery]) -> dict[str, list[PodPlacement]]: ...
