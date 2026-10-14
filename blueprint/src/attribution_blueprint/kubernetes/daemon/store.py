# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from attribution_blueprint.kubernetes.daemon.models import (
    CoverageGap,
    CoverageGapReason,
    PlacementRecord,
    ResourceObservation,
    WatchCoverage,
)


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
    def capture_observation(
        self,
        observation: ResourceObservation,
        *,
        source_name: str,
        placement: PlacementRecord | None,
        advance_checkpoint: bool,
        mark_deleted: bool = False,
    ) -> CaptureResult: ...

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

    def get_watch_coverage(
        self,
        cluster_id: str,
        namespace: str,
        source_name: str,
    ) -> WatchCoverage | None: ...

    def begin_coverage_gap(
        self,
        *,
        cluster_id: str,
        namespace: str,
        source_name: str,
        api_version: str,
        kind: str,
        reason: CoverageGapReason,
        start_resource_version: str,
        detected_at: str,
    ) -> int: ...

    def complete_watch_list(
        self,
        *,
        cluster_id: str,
        namespace: str,
        source_name: str,
        api_version: str,
        kind: str,
        resource_version: str,
        completed_at: str,
        gap_id: int | None,
    ) -> None: ...

    def mark_watch_checkpoint(
        self,
        *,
        cluster_id: str,
        namespace: str,
        source_name: str,
        resource_version: str,
        checkpointed_at: str,
    ) -> None: ...

    def list_coverage_gaps(
        self,
        cluster_id: str,
        namespace: str,
    ) -> list[CoverageGap]: ...


class PlacementReader(Protocol):
    def list_for_resource(self, namespace: str, resource_uid: str) -> list[PlacementRecord]: ...


class PlacementStore(PlacementReader, ObservationJournal, Protocol):
    def get_by_pod_uid(self, pod_uid: str) -> PlacementRecord | None: ...

    def create(self, record: PlacementRecord) -> CreateResult: ...

    def mark_deleted(self, pod_uid: str) -> None: ...

    def reconcile_active_pods(
        self,
        namespace: str,
        active_pod_uids: set[str],
    ) -> None: ...
