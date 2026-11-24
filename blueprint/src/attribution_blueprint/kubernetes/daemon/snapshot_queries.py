# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared read helpers for retained Kubernetes snapshots."""

from __future__ import annotations

from attribution_blueprint.kubernetes.daemon.models import (
    OwnerReference,
    ResourceIdentity,
    ResourceSnapshot,
)
from attribution_blueprint.kubernetes.daemon.store import SnapshotJournal


def latest_snapshot_for_uid(
    journal: SnapshotJournal,
    cluster_id: str,
    namespace: str,
    uid: str,
) -> ResourceSnapshot | None:
    """Return the newest retained version of one resource UID."""
    snapshots = journal.list_snapshots_for_uid(cluster_id, namespace, uid)
    return snapshots[-1] if snapshots else None


def latest_snapshots_for_kind(
    journal: SnapshotJournal,
    cluster_id: str,
    namespace: str,
    kind: str,
    *,
    name: str | None = None,
) -> list[ResourceSnapshot]:
    """Return latest retained resources of a kind, optionally filtered by name."""
    snapshots = journal.list_latest_snapshots_for_kind(
        cluster_id,
        namespace,
        kind,
    )
    if name is None:
        return snapshots
    return [snapshot for snapshot in snapshots if snapshot.name == name]


def controller_owner_reference(
    snapshot: ResourceSnapshot,
    kind: str | None = None,
) -> OwnerReference | None:
    """Return the sole retained controller, optionally restricted by kind."""
    owners = [
        owner
        for owner in snapshot.owner_references
        if owner.controller and (kind is None or owner.kind == kind)
    ]
    return owners[0] if len(owners) == 1 else None


def snapshot_identity(snapshot: ResourceSnapshot) -> ResourceIdentity:
    """Project a retained snapshot to its stable Kubernetes identity."""
    return ResourceIdentity(
        api_version=snapshot.api_version,
        kind=snapshot.kind,
        namespace=snapshot.namespace,
        name=snapshot.name,
        uid=snapshot.uid,
    )
