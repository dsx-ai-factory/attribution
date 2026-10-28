# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kueue-specific Prometheus placement query model."""

from __future__ import annotations

from dataclasses import dataclass

from attribution_blueprint.connectors.execution_contexts.kueue.prometheus_placement import (
    PlacementQuery,
)
from attribution_blueprint.kubernetes.daemon.models import PodPlacement

__all__ = ["OwnerIdentity", "PlacementQuery", "PodPlacement", "WorkloadIdentity"]


@dataclass(frozen=True)
class WorkloadIdentity:
    """Compatibility identity for callers of the former placement package."""

    name: str
    uid: str


@dataclass(frozen=True)
class OwnerIdentity:
    """Compatibility owner identity for the former placement package."""

    api_version: str
    kind: str
    name: str
    uid: str
