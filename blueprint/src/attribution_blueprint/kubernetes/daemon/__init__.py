# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable, scheduler-neutral Kubernetes evidence collection."""

from attribution_blueprint.kubernetes.daemon.client import KubernetesDaemonClient
from attribution_blueprint.kubernetes.daemon.models import (
    PlacementRecord,
    PodPlacement,
    ResourceIdentity,
)
from attribution_blueprint.kubernetes.daemon.runtime import KubernetesDaemon
from attribution_blueprint.kubernetes.daemon.store import (
    CreateResult,
    PlacementStore,
)

__all__ = [
    "CreateResult",
    "KubernetesDaemon",
    "KubernetesDaemonClient",
    "PlacementRecord",
    "PlacementStore",
    "PodPlacement",
    "ResourceIdentity",
]
