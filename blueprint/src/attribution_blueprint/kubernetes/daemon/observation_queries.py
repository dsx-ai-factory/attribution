# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared read helpers for retained Kubernetes observations."""

from __future__ import annotations

from attribution_blueprint.kubernetes.daemon.models import (
    OwnerReference,
    ResourceIdentity,
    ResourceObservation,
)
from attribution_blueprint.kubernetes.daemon.store import ObservationJournal


def latest_observation_for_uid(
    journal: ObservationJournal,
    cluster_id: str,
    namespace: str,
    uid: str,
) -> ResourceObservation | None:
    """Return the newest retained version of one resource UID."""
    observations = journal.list_observations_for_uid(cluster_id, namespace, uid)
    return observations[-1] if observations else None


def latest_observations_for_kind(
    journal: ObservationJournal,
    cluster_id: str,
    namespace: str,
    kind: str,
    *,
    name: str | None = None,
) -> list[ResourceObservation]:
    """Return latest retained resources of a kind, optionally filtered by name."""
    observations = journal.list_latest_observations_for_kind(
        cluster_id,
        namespace,
        kind,
    )
    if name is None:
        return observations
    return [observation for observation in observations if observation.name == name]


def controller_owner_reference(
    observation: ResourceObservation,
    kind: str | None = None,
) -> OwnerReference | None:
    """Return the sole retained controller, optionally restricted by kind."""
    owners = [
        owner
        for owner in observation.owner_references
        if owner.controller and (kind is None or owner.kind == kind)
    ]
    return owners[0] if len(owners) == 1 else None


def observation_identity(observation: ResourceObservation) -> ResourceIdentity:
    """Project a retained observation to its stable Kubernetes identity."""
    return ResourceIdentity(
        api_version=observation.api_version,
        kind=observation.kind,
        namespace=observation.namespace,
        name=observation.name,
        uid=observation.uid,
    )
