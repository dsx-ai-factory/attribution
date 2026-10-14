# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kueue-specific Prometheus placement query model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from attribution_blueprint.kubernetes.daemon.models import PodPlacement

__all__ = ["OwnerIdentity", "PlacementQuery", "PodPlacement", "WorkloadIdentity"]


@dataclass(frozen=True)
class WorkloadIdentity:
    """Identity of a Kueue Workload."""

    name: str
    uid: str


@dataclass(frozen=True)
class OwnerIdentity:
    """Identity of a resource directly owned by a Kueue Workload."""

    api_version: str
    kind: str
    name: str
    uid: str


@dataclass(frozen=True)
class PlacementQuery:
    """One Workload's inputs to a shared historical-placement query."""

    workload_uid: str
    namespace: str
    owner_job_uids: frozenset[str]
    start_time: datetime
    end_time: datetime
