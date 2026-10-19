# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Protocol

from attribution_blueprint.kubernetes.daemon.models import PlacementRecord


class CreateResult(str, Enum):
    CREATED = "created"
    DUPLICATE = "duplicate"
    UPDATED = "updated"


class PlacementConflictError(RuntimeError):
    pass


class PlacementReader(Protocol):
    def list_for_pod(self, namespace: str, pod_uid: str) -> list[PlacementRecord]: ...

    def list_for_resource(self, namespace: str, resource_uid: str) -> list[PlacementRecord]: ...

    def list_for_owner(self, namespace: str, owner_uid: str) -> list[PlacementRecord]: ...


class PlacementStore(PlacementReader, Protocol):
    def get_by_pod_uid(self, pod_uid: str) -> PlacementRecord | None: ...

    def create(self, record: PlacementRecord) -> CreateResult: ...

    def mark_deleted(self, pod_uid: str) -> None: ...

    def reconcile_active_pods(
        self,
        namespace: str,
        active_pod_uids: set[str],
    ) -> None: ...
