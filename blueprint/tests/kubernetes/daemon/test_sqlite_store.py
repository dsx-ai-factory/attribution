# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from dataclasses import replace

import pytest

from attribution_blueprint.kubernetes.daemon.models import (
    PlacementRecord,
    ResourceIdentity,
)
from attribution_blueprint.kubernetes.daemon.sqlite_store import SQLitePlacementStore
from attribution_blueprint.kubernetes.daemon.store import CreateResult, PlacementConflictError


def _record(*, associated: bool) -> PlacementRecord:
    associated_resources = ()
    if associated:
        associated_resources = (
            ResourceIdentity(
                "scheduling.example.io/v1",
                "Admission",
                "team-a",
                "trainer-admission",
                "admission-uid-1",
            ),
        )
    return PlacementRecord(
        namespace="team-a",
        pod_name="trainer-0",
        pod_uid="pod-uid-1",
        node_name="node-a",
        observed_at="2026-10-13T20:00:00Z",
        associated_resources=associated_resources,
    )


def test_unresolved_placement_can_be_enriched_later(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))

    assert store.create(_record(associated=False)) == CreateResult.CREATED
    assert store.list_for_resource("team-a", "admission-uid-1") == []

    assert store.create(_record(associated=True)) == CreateResult.UPDATED
    assert store.list_for_resource("team-a", "admission-uid-1") == [_record(associated=True)]


def test_verified_owner_refreshes_pod_owner_references(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    job = ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")
    replacement_job = ResourceIdentity("batch/v1", "Job", "team-a", "trainer-2", "job-uid-2")
    jobset = ResourceIdentity(
        "jobset.x-k8s.io/v1alpha2",
        "JobSet",
        "team-a",
        "training",
        "jobset-uid-1",
    )
    initial = replace(_record(associated=True), owner_references=(job,), owner=job)
    reassigned = replace(
        initial,
        owner_references=(replacement_job,),
        owner=jobset,
        associated_resources=(),
    )

    store.create(initial)
    assert store.create(reassigned) == CreateResult.UPDATED

    stored = store.get_by_pod_uid(initial.pod_uid)
    assert stored.owner == jobset
    assert stored.owner_references == (replacement_job,)
    assert stored.associated_resources == ()
    assert store.list_for_resource("team-a", "admission-uid-1") == []


def test_active_placement_does_not_expire(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path), retention_days=1)
    record = _record(associated=False)
    store.create(record)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE placement_record SET updated_at = datetime('now', '-2 days')"
        )

    assert store.get_by_pod_uid(record.pod_uid) == record


def test_deleted_placement_expires_after_retention_window(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path), retention_days=1)
    record = _record(associated=False)
    store.create(record)
    store.mark_deleted(record.pod_uid)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE placement_deletion SET deleted_at = datetime('now', '-2 days')"
        )

    assert store.get_by_pod_uid(record.pod_uid) is None


def test_reconcile_marks_absent_pods_without_resetting_deletion_time(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    active = _record(associated=False)
    absent = replace(
        active,
        pod_name="trainer-1",
        pod_uid="pod-uid-2",
    )
    store.create(active)
    store.create(absent)

    store.reconcile_active_pods("team-a", {active.pod_uid})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE placement_deletion SET deleted_at = datetime('now', '-1 day')"
        )
        first_deleted_at = connection.execute(
            "SELECT deleted_at FROM placement_deletion WHERE pod_uid = ?",
            (absent.pod_uid,),
        ).fetchone()[0]

    store.reconcile_active_pods("team-a", {active.pod_uid})
    with sqlite3.connect(path) as connection:
        markers = connection.execute(
            "SELECT pod_uid, deleted_at FROM placement_deletion ORDER BY pod_uid"
        ).fetchall()

    assert markers == [(absent.pod_uid, first_deleted_at)]


def test_store_closes_connection_after_operation(tmp_path, monkeypatch):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))

    class TrackingConnection(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    connection = sqlite3.connect(store.path, factory=TrackingConnection)
    connection.row_factory = sqlite3.Row
    monkeypatch.setattr(store, "_connect", lambda: connection)

    assert store.get_by_pod_uid("missing") is None
    assert connection.closed


def test_multiple_associated_resources_are_queryable(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    record = _record(associated=True).enriched(
        associated_resources=(
            ResourceIdentity(
                "scheduling.example.net/v1",
                "ResourceGroup",
                "team-a",
                "group-a",
                "group-uid-1",
            ),
        )
    )

    store.create(record)

    assert store.list_for_resource("team-a", "admission-uid-1") == [record]
    assert store.list_for_resource("team-a", "group-uid-1") == [record]


def test_later_scheduling_timestamp_and_failure_evidence_enrich_placement(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    initial = _record(associated=False)
    updated = replace(
        initial,
        scheduled_at="2026-10-13T20:00:01Z",
        node_failure_reason="NodeLost",
    )

    assert store.create(initial) == CreateResult.CREATED
    assert store.create(updated) == CreateResult.UPDATED
    assert store.get_by_pod_uid(initial.pod_uid) == updated


def test_two_nonempty_scheduling_timestamps_still_conflict(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    initial = replace(_record(associated=False), scheduled_at="2026-10-13T20:00:01Z")
    conflicting = replace(initial, scheduled_at="2026-10-13T20:00:02Z")

    store.create(initial)

    with pytest.raises(PlacementConflictError, match="conflicting scheduling timestamps"):
        store.create(conflicting)


def test_resource_uid_replaces_stale_resource_metadata(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    initial = _record(associated=True)
    updated_resource = ResourceIdentity(
        "scheduling.example.io/v2",
        "Admission",
        "team-a",
        "trainer-admission",
        "admission-uid-1",
    )
    updated = replace(initial, associated_resources=(updated_resource,))

    store.create(initial)

    assert store.create(updated) == CreateResult.UPDATED
    [stored] = store.list_for_resource("team-a", "admission-uid-1")
    assert stored.associated_resources == (updated_resource,)
