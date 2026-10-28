# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from attribution_blueprint.kubernetes.daemon.models import PlacementRecord, ResourceObservation


class CreateResult(str, Enum):
    CREATED = "created"
    DUPLICATE = "duplicate"
    UPDATED = "updated"


class PlacementConflictError(RuntimeError):
    pass


class ObservationAppendResult(str, Enum):
    APPENDED = "appended"
    DUPLICATE = "duplicate"
    UPDATED = "updated"


class ObservationConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptureResult:
    observation: ObservationAppendResult
    placement: CreateResult | None


class ObservationJournal(Protocol):
    def append_observation(self, observation: ResourceObservation) -> ObservationAppendResult: ...

    def list_observations_for_uid(
        self,
        cluster_id: str,
        namespace: str,
        uid: str,
    ) -> list[ResourceObservation]: ...

    def list_observations_for_owner(
        self,
        cluster_id: str,
        namespace: str,
        owner_uid: str,
    ) -> list[ResourceObservation]: ...


class PlacementReader(Protocol):
    def list_for_resource(self, namespace: str, resource_uid: str) -> list[PlacementRecord]: ...


class PlacementStore(PlacementReader, ObservationJournal, Protocol):
    def capture_observation(
        self,
        observation: ResourceObservation,
        *,
        placement: PlacementRecord | None,
        mark_deleted: bool = False,
    ) -> CaptureResult: ...

    def get_by_pod_uid(self, pod_uid: str) -> PlacementRecord | None: ...

    def create(self, record: PlacementRecord) -> CreateResult: ...

    def mark_deleted(self, pod_uid: str) -> None: ...

    def reconcile_active_pods(
        self,
        namespace: str,
        active_pod_uids: set[str],
    ) -> None: ...
