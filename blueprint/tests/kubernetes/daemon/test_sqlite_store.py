# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def test_changed_payload_for_same_resource_version_is_rejected(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    initial = _observation(status={"phase": "Pending"})
    conflicting = replace(initial, status={"phase": "Running"})

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
    assert gap.detected_at == "2026-10-14T00:00:00Z"
    assert gap.end_resource_version == "20"
    assert gap.recovered_at == "2026-10-14T00:01:00Z"


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
