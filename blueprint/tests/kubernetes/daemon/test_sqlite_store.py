# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

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


def _history_observation(
    *,
    kind: str,
    name: str,
    uid: str,
    observed_at: str,
    owners: tuple[OwnerReference, ...] = (),
    status: dict | None = None,
    event_type: ObservationEventType = "MODIFIED",
) -> ResourceObservation:
    return ResourceObservation(
        cluster_id="cluster-a",
        api_version="v1",
        kind=kind,
        namespace="team-a",
        name=name,
        uid=uid,
        resource_version="1",
        event_type=event_type,
        observed_at=observed_at,
        owner_references=owners,
        status=status,
    )


def _capture(store: SQLitePlacementStore, observation: ResourceObservation) -> ObservationAppendResult:
    return store.capture_observation(
        observation,
        source_name="pods",
        placement=None,
        advance_checkpoint=False,
    ).observation


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

    reopened = SQLitePlacementStore(str(path), retention_days=1)
    assert reopened.get_by_pod_uid(record.pod_uid) is None


def test_deleted_capture_rolls_back_observation_and_placement_with_marker(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    observation = _observation(event_type="DELETED")
    record = _record(associated=False)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_deletion_marker
            BEFORE INSERT ON placement_deletion
            BEGIN
                SELECT RAISE(ABORT, 'synthetic deletion marker failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="synthetic deletion marker failure"):
        store.capture_observation(
            observation,
            source_name="pods",
            placement=record,
            advance_checkpoint=True,
            mark_deleted=True,
        )

    assert store.list_observations_for_uid("cluster-a", "team-a", record.pod_uid) == []
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

    assert _capture(store, initial) == ObservationAppendResult.APPENDED
    assert _capture(store, updated) == ObservationAppendResult.APPENDED

    reopened = SQLitePlacementStore(str(path))
    assert reopened.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1") == [
        initial,
        updated,
    ]


def test_expired_workload_prunes_its_complete_controller_history(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path), retention_days=30)
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    jobset_owner = OwnerReference(
        api_version="jobset.x-k8s.io/v1alpha2",
        kind="JobSet",
        name="training",
        uid="jobset-uid",
        controller=True,
    )
    job_owner = OwnerReference(
        api_version="batch/v1",
        kind="Job",
        name="worker",
        uid="job-uid",
        controller=True,
    )
    observations = [
        _history_observation(
            kind="JobSet",
            name="training",
            uid="jobset-uid",
            observed_at=old,
        ),
        _history_observation(
            kind="Job",
            name="worker",
            uid="job-uid",
            observed_at=old,
            owners=(jobset_owner,),
        ),
        _history_observation(
            kind="Pod",
            name="worker-0",
            uid="pod-uid",
            observed_at=old,
            owners=(job_owner,),
        ),
        _history_observation(
            kind="Workload",
            name="training-workload",
            uid="workload-uid",
            observed_at=old,
            owners=(jobset_owner,),
            status={
                "conditions": [
                    {
                        "type": "Finished",
                        "status": "True",
                        "lastTransitionTime": old,
                    }
                ]
            },
        ),
    ]
    for observation in observations:
        _capture(store, observation)

    SQLitePlacementStore(str(path), retention_days=1)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM resource_observation"
        ).fetchone()[0] == 0


def test_old_active_workload_history_is_retained(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path), retention_days=1)
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    workload = _history_observation(
        kind="Workload",
        name="training-workload",
        uid="workload-uid",
        observed_at=old,
        status={"conditions": [{"type": "Admitted", "status": "True"}]},
    )

    _capture(store, workload)
    store.prune_expired()

    assert store.list_observations_for_uid(
        "cluster-a",
        "team-a",
        "workload-uid",
    ) == [workload]


def test_newer_active_workload_protects_owner_group_from_late_old_terminal(
    tmp_path,
):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path), retention_days=30)
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    current = datetime.now(timezone.utc).isoformat()
    jobset_owner = OwnerReference(
        api_version="jobset.x-k8s.io/v1alpha2",
        kind="JobSet",
        name="training",
        uid="jobset-uid",
        controller=True,
    )
    active = _history_observation(
        kind="Workload",
        name="training-workload",
        uid="workload-new",
        observed_at=current,
        owners=(jobset_owner,),
        status={
            "conditions": [
                {"type": "Admitted", "status": "True"},
            ]
        },
    )
    expired = _history_observation(
        kind="Workload",
        name="training-workload",
        uid="workload-old",
        observed_at=old,
        owners=(jobset_owner,),
        status={
            "conditions": [
                {
                    "type": "Finished",
                    "status": "True",
                    "lastTransitionTime": old,
                }
            ]
        },
    )

    # Simulate delayed delivery: the expired incarnation is inserted last.
    _capture(store, active)
    _capture(store, expired)

    reopened = SQLitePlacementStore(str(path), retention_days=1)

    assert reopened.list_observations_for_uid(
        "cluster-a",
        "team-a",
        "workload-new",
    ) == [active]
    assert reopened.list_observations_for_uid(
        "cluster-a",
        "team-a",
        "workload-old",
    ) == [expired]


def test_expired_non_workload_histories_are_pruned(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path), retention_days=30)
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    kai_job = OwnerReference(
        api_version="batch/v1",
        kind="Job",
        name="kai-job",
        uid="kai-job-uid",
        controller=True,
    )
    missing_job = OwnerReference(
        api_version="batch/v1",
        kind="Job",
        name="missing-job",
        uid="missing-job-uid",
        controller=True,
    )
    observations = [
        _history_observation(
            kind="Job",
            name="standalone-job",
            uid="standalone-job-uid",
            observed_at=old,
            event_type="DELETED",
        ),
        _history_observation(
            kind="Pod",
            name="orphaned-pod",
            uid="orphaned-pod-uid",
            observed_at=old,
            owners=(missing_job,),
            event_type="DELETED",
        ),
        _history_observation(
            kind="Job",
            name="kai-job",
            uid="kai-job-uid",
            observed_at=old,
        ),
        _history_observation(
            kind="PodGroup",
            name="kai-group",
            uid="podgroup-uid",
            observed_at=old,
            owners=(kai_job,),
            event_type="DELETED",
        ),
    ]
    for observation in observations:
        _capture(store, observation)

    SQLitePlacementStore(str(path), retention_days=1)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM resource_observation"
        ).fetchone()[0] == 0


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

    assert _capture(store, observation) == ObservationAppendResult.APPENDED
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

    assert _capture(store, initial) == ObservationAppendResult.APPENDED
    assert _capture(store, replayed) == ObservationAppendResult.DUPLICATE
    assert store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1") == [initial]


def test_replayed_resource_version_accepts_additive_projection(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    initial = _observation(
        selected_metadata={"labels": {"existing": "value"}},
        status={"phase": "Running"},
    )
    expanded = replace(
        initial,
        event_type="SNAPSHOT",
        observed_at="2026-10-14T00:01:00Z",
        source_timestamp="2026-10-14T00:00:30Z",
        selected_metadata={
            "labels": {
                "existing": "value",
                "jobset.sigs.k8s.io/restart-attempt": "1",
            }
        },
        status={
            "phase": "Running",
            "startTime": "2026-10-14T00:00:30Z",
        },
    )

    assert _capture(store, initial) == ObservationAppendResult.APPENDED
    with sqlite3.connect(path) as connection:
        original = connection.execute(
            "SELECT id, observed_at FROM resource_observation"
        ).fetchone()

    assert _capture(store, expanded) == ObservationAppendResult.UPDATED
    # Replaying an older projection after upgrade keeps the richer durable row.
    assert _capture(store, initial) == ObservationAppendResult.DUPLICATE

    [stored] = store.list_observations_for_uid(
        "cluster-a",
        "team-a",
        "pod-uid-1",
    )
    assert stored.event_type == initial.event_type
    assert stored.observed_at == initial.observed_at
    assert stored.source_timestamp == expanded.source_timestamp
    assert stored.selected_metadata == expanded.selected_metadata
    assert stored.status == expanded.status
    with sqlite3.connect(path) as connection:
        replayed = connection.execute(
            "SELECT id, observed_at FROM resource_observation"
        ).fetchone()
    assert replayed == original


def test_changed_payload_for_same_resource_version_is_rejected(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    initial = _observation(status={"phase": "Pending"})
    conflicting = replace(initial, status={"phase": "Running"})

    _capture(store, initial)

    with pytest.raises(ObservationConflictError, match="changed during replay"):
        _capture(store, conflicting)


def test_changed_owners_for_same_resource_version_are_rejected(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    initial = _observation(status={"phase": "Running"})
    conflicting = replace(
        initial,
        owner_references=(
            *initial.owner_references,
            OwnerReference(
                api_version="batch/v1",
                kind="Job",
                name="replacement",
                uid="job-uid-2",
                controller=True,
            ),
        ),
    )

    _capture(store, initial)

    with pytest.raises(ObservationConflictError, match="changed during replay"):
        _capture(store, conflicting)


def test_observations_are_queryable_by_controller_owner(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    observation = _observation(status={"phase": "Running"})

    _capture(store, observation)

    assert store.list_observations_for_owner("cluster-a", "team-a", "job-uid-1") == [
        observation
    ]


def test_latest_observations_are_queryable_by_resource_kind(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    initial = _observation(status={"phase": "Pending"})
    updated = _observation(
        resource_version="2",
        event_type="MODIFIED",
        status={"phase": "Running"},
    )
    _capture(store, initial)
    _capture(store, updated)

    assert store.list_latest_observations_for_kind(
        "cluster-a",
        "team-a",
        "Pod",
    ) == [updated]


def test_controller_history_recursively_includes_descendant_resources(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    jobset = replace(
        _observation(),
        api_version="jobset.x-k8s.io/v1alpha2",
        kind="JobSet",
        name="training",
        uid="jobset-uid-1",
        owner_references=(),
    )
    job = replace(
        _observation(resource_version="2"),
        api_version="batch/v1",
        kind="Job",
        name="training-workers-0",
        uid="job-uid-1",
        owner_references=(
            OwnerReference(
                api_version="jobset.x-k8s.io/v1alpha2",
                kind="JobSet",
                name="training",
                uid="jobset-uid-1",
                controller=True,
            ),
        ),
    )
    pod = replace(
        _observation(resource_version="3"),
        name="training-workers-0-abcde",
        uid="pod-uid-1",
    )
    unrelated = replace(
        _observation(resource_version="4"),
        name="other-pod",
        uid="other-pod-uid",
        owner_references=(
            OwnerReference(
                api_version="batch/v1",
                kind="Job",
                name="other-job",
                uid="other-job-uid",
                controller=True,
            ),
        ),
    )
    for observation in (jobset, job, pod, unrelated):
        _capture(store, observation)

    assert store.list_observations_for_controller(
        "cluster-a",
        "team-a",
        "jobset-uid-1",
    ) == [jobset, job, pod]


def test_observation_journal_does_not_change_placement_view(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))

    _capture(store, _observation(status={"phase": "Running"}))

    assert store.get_by_pod_uid("pod-uid-1") is None


def test_watch_coverage_and_gaps_survive_reopen(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    store.complete_watch_list(
        cluster_id="cluster-a",
        namespace="team-a",
        source_name="pods",
        api_version="v1",
        kind="Pod",
        resource_version="10",
        completed_at="2026-10-14T00:00:00Z",
        gap_id=None,
    )
    gap_id = store.begin_coverage_gap(
        cluster_id="cluster-a",
        namespace="team-a",
        source_name="pods",
        api_version="v1",
        kind="Pod",
        reason="WATCH_SOURCE_CHANGED",
        start_resource_version="10",
        started_at="2026-10-13T23:59:30Z",
        detected_at="2026-10-14T00:00:00Z",
    )
    store.complete_watch_list(
        cluster_id="cluster-a",
        namespace="team-a",
        source_name="pods",
        api_version="v1",
        kind="Pod",
        resource_version="20",
        completed_at="2026-10-14T00:01:00Z",
        gap_id=gap_id,
    )

    reopened = SQLitePlacementStore(str(path))
    coverage = reopened.get_watch_coverage("cluster-a", "team-a", "pods")
    assert coverage is not None
    assert coverage.coverage_started_at == "2026-10-14T00:00:00Z"
    assert coverage.resource_version == "20"
    assert coverage.checkpointed_at == "2026-10-14T00:01:00Z"
    [gap] = reopened.list_coverage_gaps("cluster-a", "team-a")
    assert gap.reason == "WATCH_SOURCE_CHANGED"
    assert gap.start_resource_version == "10"
    assert gap.started_at == "2026-10-13T23:59:30Z"
    assert gap.detected_at == "2026-10-14T00:00:00Z"
    assert gap.end_resource_version == "20"
    assert gap.recovered_at == "2026-10-14T00:01:00Z"


def test_history_pruning_advances_coverage_boundary(tmp_path):
    store = SQLitePlacementStore(
        str(tmp_path / "placements.db"),
        retention_days=1,
    )
    old = datetime.now(timezone.utc) - timedelta(days=2)
    store.complete_watch_list(
        cluster_id="cluster-a",
        namespace="team-a",
        source_name="pods",
        api_version="v1",
        kind="Pod",
        resource_version="10",
        completed_at=old.isoformat(),
        gap_id=None,
    )

    # Pruning is now periodic, not a side effect of ordinary reads/writes.
    store.prune_expired()

    coverage = store.get_watch_coverage("cluster-a", "team-a", "pods")
    assert coverage is not None
    boundary = datetime.fromisoformat(coverage.coverage_started_at).replace(
        tzinfo=timezone.utc
    )
    assert boundary >= datetime.now(timezone.utc) - timedelta(days=1, seconds=5)


def test_coverage_gap_migration_uses_detection_time_for_existing_rows(tmp_path):
    path = tmp_path / "placements.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE coverage_gap (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                source_name TEXT NOT NULL,
                api_version TEXT NOT NULL,
                kind TEXT NOT NULL,
                reason TEXT NOT NULL,
                start_resource_version TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                end_resource_version TEXT,
                recovered_at TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO coverage_gap(
                cluster_id, namespace, source_name, api_version, kind,
                reason, start_resource_version, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cluster-a",
                "team-a",
                "pods",
                "v1",
                "Pod",
                "RESOURCE_VERSION_EXPIRED",
                "10",
                "2026-10-14T00:00:00Z",
            ),
        )

    store = SQLitePlacementStore(str(path))

    [gap] = store.list_coverage_gaps("cluster-a", "team-a")
    assert gap.started_at == gap.detected_at
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4


def test_atomic_capture_does_not_advance_checkpoint_when_placement_conflicts(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.complete_watch_list(
        cluster_id="cluster-a",
        namespace="team-a",
        source_name="pods",
        api_version="v1",
        kind="Pod",
        resource_version="10",
        completed_at="2026-10-14T00:00:00Z",
        gap_id=None,
    )
    store.create(_record(associated=False))
    observation = _observation(resource_version="11", event_type="MODIFIED")
    conflicting_placement = replace(_record(associated=False), node_name="node-b")

    with pytest.raises(PlacementConflictError):
        store.capture_observation(
            observation,
            source_name="pods",
            placement=conflicting_placement,
            advance_checkpoint=True,
        )

    assert store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1") == []
    coverage = store.get_watch_coverage("cluster-a", "team-a", "pods")
    assert coverage is not None
    assert coverage.resource_version == "10"
