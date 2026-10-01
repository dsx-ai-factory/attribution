# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from enum import Enum
from typing import Protocol

from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import (
    PlacementQuery,
    PlacementRecord,
    PodPlacement,
)


class CreateResult(str, Enum):
    CREATED = "created"
    DUPLICATE = "duplicate"
    # An existing record's mutable `node_failure_reason` was patched. Unlike
    # DUPLICATE, this means the store's copy changed.
    UPDATED = "updated"


class PlacementConflictError(RuntimeError):
    pass


class PlacementReader(Protocol):
    def list_for_workload(self, namespace: str, workload_uid: str) -> list[PlacementRecord]: ...


class PlacementStore(PlacementReader, Protocol):
    def get_by_pod_uid(self, pod_uid: str) -> PlacementRecord | None: ...

    def create(self, record: PlacementRecord) -> CreateResult: ...


class PlacementResolver(Protocol):
    def placements_for_many(self, queries: list[PlacementQuery]) -> dict[str, list[PodPlacement]]: ...
