# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from dataclasses import replace

import pytest

from attribution_blueprint.kubernetes.daemon.models import (
    ObservationEventType,
    OwnerReference,
    PlacementRecord,
    ResourceIdentity,
    ResourceObservation,
)
from attribution_blueprint.kubernetes.daemon.sqlite_store import SQLitePlacementStore
from attribution_blueprint.kubernetes.daemon.store import (
    CreateResult,
    ObservationAppendResult,
    ObservationConflictError,
    PlacementConflictError,
)


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


def _observation(
    *,
    resource_version: str = "1",
    event_type: ObservationEventType = "SNAPSHOT",
    selected_metadata: dict | None = None,
    status: dict | None = None,
) -> ResourceObservation:
    return ResourceObservation(
        cluster_id="cluster-a",
        api_version="v1",
        kind="Pod",
        namespace="team-a",
        name="trainer-0",
        uid="pod-uid-1",
        resource_version=resource_version,
        event_type=event_type,
        observed_at="2026-10-14T00:00:00Z",
        source_timestamp="2026-10-13T23:59:59Z",
        owner_references=(
            OwnerReference(
                api_version="batch/v1",
                kind="Job",
                name="trainer",
                uid="job-uid-1",
                controller=True,
            ),
        ),
        selected_metadata=selected_metadata,
        status=status,
    )


def test_resource_observation_preserves_controller_owner_reference():
    observation = ResourceObservation.from_resource(
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "namespace": "team-a",
                "name": "trainer",
                "uid": "job-uid-1",
                "resourceVersion": "12",
                "ownerReferences": [
                    {
                        "apiVersion": "jobset.x-k8s.io/v1alpha2",
                        "kind": "JobSet",
                        "name": "training-set",
                        "uid": "jobset-uid-1",
                        "controller": True,
                    }
                ],
            },
        },
        cluster_id="cluster-a",
        event_type="SNAPSHOT",
        observed_at="2026-10-14T00:00:00Z",
        status={"active": 1},
    )

    assert observation.owner_references == (
        OwnerReference(
            api_version="jobset.x-k8s.io/v1alpha2",
            kind="JobSet",
            name="training-set",
            uid="jobset-uid-1",
            controller=True,
        ),
    )
    assert observation.status == {"active": 1}


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


def test_observation_journal_is_append_only_and_survives_reopen(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    initial = _observation(
        selected_metadata={
            "annotations": {"batch.kubernetes.io/job-completion-index": "0"}
        },
        status={"phase": "Pending"},
    )
    updated = _observation(
        resource_version="2",
        event_type="MODIFIED",
        status={"phase": "Running"},
    )

    assert store.append_observation(initial) == ObservationAppendResult.APPENDED
    assert store.append_observation(updated) == ObservationAppendResult.APPENDED

    reopened = SQLitePlacementStore(str(path))
    assert reopened.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1") == [
        initial,
        updated,
    ]


def test_observation_schema_adds_selected_metadata_to_existing_journal(tmp_path):
    path = tmp_path / "placements.db"
    SQLitePlacementStore(str(path))
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE resource_observation DROP COLUMN selected_metadata_json")
        connection.execute("PRAGMA user_version = 3")

    store = SQLitePlacementStore(str(path))
    observation = _observation(
        selected_metadata={
            "labels": {"jobset.sigs.k8s.io/restart-attempt": "1"},
        }
    )

    assert store.append_observation(observation) == ObservationAppendResult.APPENDED
    assert store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1") == [observation]
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4


def test_replayed_resource_version_is_deduplicated(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    initial = _observation(status={"phase": "Pending"})
    replayed = replace(
        initial,
        event_type="ADDED",
        observed_at="2026-10-14T00:01:00Z",
    )

    assert store.append_observation(initial) == ObservationAppendResult.APPENDED
    assert store.append_observation(replayed) == ObservationAppendResult.DUPLICATE
    assert store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1") == [initial]


def test_changed_payload_for_same_resource_version_is_rejected(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    initial = _observation(status={"phase": "Pending"})
    conflicting = replace(initial, status={"phase": "Running"})

    store.append_observation(initial)

    with pytest.raises(ObservationConflictError, match="changed during replay"):
        store.append_observation(conflicting)


def test_observations_are_queryable_by_controller_owner(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    observation = _observation(status={"phase": "Running"})

    store.append_observation(observation)

    assert store.list_observations_for_owner("cluster-a", "team-a", "job-uid-1") == [
        observation
    ]


def test_observation_journal_does_not_change_placement_view(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))

    store.append_observation(_observation(status={"phase": "Running"}))

    assert store.get_by_pod_uid("pod-uid-1") is None
