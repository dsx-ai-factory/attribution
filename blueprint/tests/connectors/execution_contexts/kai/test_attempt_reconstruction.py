# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest

from attribution_blueprint.connectors.execution_contexts.kai.attempts import (
    PodGroupAttempt,
    attempts_from_podgroup_history,
    observations_from_live_resources,
)
from attribution_blueprint.connectors.execution_contexts.kai.connector import (
    KaiExecutionContextAdapter,
    _AttemptEvidence,
    _context_from_attempt_evidence,
    _WorkloadEvidence,
)
from attribution_blueprint.kubernetes.daemon.client import KubernetesDaemonClient
from attribution_blueprint.kubernetes.daemon.health import HealthState
from attribution_blueprint.kubernetes.daemon.models import (
    OwnerReference,
    PlacementRecord,
    ResourceIdentity,
    ResourceObservation,
)
from attribution_blueprint.kubernetes.daemon.server import serve_api
from attribution_blueprint.kubernetes.daemon.sqlite_store import SQLitePlacementStore

_RESTART_LABEL = "jobset.sigs.k8s.io/restart-attempt"


def _owner(kind: str, name: str, uid: str) -> OwnerReference:
    return OwnerReference(
        api_version=("jobset.x-k8s.io/v1alpha2" if kind == "JobSet" else "batch/v1"),
        kind=kind,
        name=name,
        uid=uid,
        controller=True,
    )


def _observation(
    *,
    kind: str,
    name: str,
    uid: str,
    resource_version: str,
    observed_at: str,
    owners: tuple[OwnerReference, ...] = (),
    status: dict | None = None,
    selected_metadata: dict | None = None,
) -> ResourceObservation:
    return ResourceObservation(
        cluster_id="cluster-a",
        api_version={
            "JobSet": "jobset.x-k8s.io/v1alpha2",
            "Job": "batch/v1",
            "Pod": "v1",
            "PodGroup": "scheduling.run.ai/v2alpha2",
        }[kind],
        kind=kind,
        namespace="team-a",
        name=name,
        uid=uid,
        resource_version=resource_version,
        event_type="MODIFIED",
        observed_at=observed_at,
        owner_references=owners,
        selected_metadata=selected_metadata,
        status=status,
    )


def _podgroup_observation(
    owner: dict, observed_at: str, *, uid: str = "podgroup-uid", **fields
) -> dict:
    """Keep lifecycle signals explicit while sharing PodGroup identity boilerplate."""
    return {
        "kind": "PodGroup",
        "metadata": {
            "namespace": "team-a",
            "name": "pg-training",
            "uid": uid,
            "ownerReferences": [owner],
        },
        "observedAt": observed_at,
        **fields,
    }


def _condition(
    condition_type: str,
    transition: str,
    *,
    reason: str | None = None,
) -> dict:
    condition = {
        "type": condition_type,
        "status": "True",
        "lastTransitionTime": transition,
    }
    if reason:
        condition["reason"] = reason
    return condition


def _capture(store: SQLitePlacementStore, observation: ResourceObservation) -> None:
    store.capture_observation(
        observation,
        source_name=f"{observation.kind.lower()}s",
        placement=None,
        advance_checkpoint=False,
    )


def _seed_kai_restart_history(store: SQLitePlacementStore) -> None:
    jobset_owner = _owner("JobSet", "training", "jobset-uid")
    podgroup_owner = OwnerReference(
        api_version="jobset.x-k8s.io/v1alpha2",
        kind="JobSet",
        name="training",
        uid="jobset-uid",
        controller=False,
    )
    job0_owner = _owner("Job", "training-worker-0", "job-uid-0")
    job1_owner = _owner("Job", "training-worker-1", "job-uid-1")
    top_owner = {
        "annotations": {
            "kai.scheduler/top-owner-metadata": (
                "kind: JobSet\n"
                "name: training\n"
                "uid: jobset-uid\n"
                "group: jobset.x-k8s.io\n"
            )
        }
    }
    observations = [
        _observation(
            kind="JobSet",
            name="training",
            uid="jobset-uid",
            resource_version="1",
            observed_at="2026-10-14T09:59:00Z",
        ),
        _observation(
            kind="PodGroup",
            name="pg-training",
            uid="podgroup-uid",
            resource_version="2",
            observed_at="2026-10-14T10:00:00Z",
            owners=(podgroup_owner,),
            selected_metadata=top_owner,
            status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
        ),
        _observation(
            kind="Job",
            name="training-worker-0",
            uid="job-uid-0",
            resource_version="3",
            observed_at="2026-10-14T10:00:00Z",
            owners=(jobset_owner,),
            selected_metadata={"labels": {_RESTART_LABEL: "0"}},
            status={"startTime": "2026-10-14T10:00:00Z"},
        ),
        _observation(
            kind="Pod",
            name="training-worker-0-pod",
            uid="pod-uid-0",
            resource_version="4",
            observed_at="2026-10-14T10:01:00Z",
            owners=(job0_owner,),
            selected_metadata={
                "annotations": {"pod-group-name": "pg-training"},
                "labels": {_RESTART_LABEL: "0"},
            },
            status={"startTime": "2026-10-14T10:01:00Z"},
        ),
        _observation(
            kind="JobSet",
            name="training",
            uid="jobset-uid",
            resource_version="5",
            observed_at="2026-10-14T10:31:00Z",
            status={
                "restarts": 1,
                "conditions": [
                    _condition(
                        "RestartingJobSet",
                        "2026-10-14T10:31:00Z",
                        reason="DefaultFailurePolicy",
                    )
                ],
            },
        ),
        _observation(
            kind="Job",
            name="training-worker-1",
            uid="job-uid-1",
            resource_version="6",
            observed_at="2026-10-14T10:31:00Z",
            owners=(jobset_owner,),
            selected_metadata={"labels": {_RESTART_LABEL: "1"}},
            status={"startTime": "2026-10-14T10:31:00Z"},
        ),
        _observation(
            kind="Pod",
            name="training-worker-1-pod",
            uid="pod-uid-1",
            resource_version="7",
            observed_at="2026-10-14T10:32:00Z",
            owners=(job1_owner,),
            selected_metadata={
                "annotations": {"pod-group-name": "pg-training"},
                "labels": {_RESTART_LABEL: "1"},
            },
            status={"startTime": "2026-10-14T10:32:00Z"},
        ),
        _observation(
            kind="JobSet",
            name="training",
            uid="jobset-uid",
            resource_version="8",
            observed_at="2026-10-14T11:00:00Z",
            status={
                "restarts": 1,
                "terminalState": "Completed",
                "conditions": [_condition("Completed", "2026-10-14T11:00:00Z")],
            },
        ),
    ]
    for observation in observations:
        _capture(store, observation)

    podgroup = ResourceIdentity(
        "scheduling.run.ai/v2alpha2",
        "PodGroup",
        "team-a",
        "pg-training",
        "podgroup-uid",
    )
    for pod_name, pod_uid, node_name, observed_at in (
        (
            "training-worker-0-pod",
            "pod-uid-0",
            "node-a",
            "2026-10-14T10:01:00Z",
        ),
        (
            "training-worker-1-pod",
            "pod-uid-1",
            "node-b",
            "2026-10-14T10:32:00Z",
        ),
    ):
        store.create(
            PlacementRecord(
                namespace="team-a",
                pod_name=pod_name,
                pod_uid=pod_uid,
                node_name=node_name,
                observed_at=observed_at,
                scheduled_at=observed_at,
                associated_resources=(podgroup,),
            )
        )


def test_retained_kai_jobset_retries_are_distinct_attempts(tmp_path, mocker):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    _seed_kai_restart_history(store)
    server = serve_api(
        HealthState(),
        store,
        0,
        "team-a",
        cluster_id="cluster-a",
    )
    try:
        client = KubernetesDaemonClient(f"http://127.0.0.1:{server.server_port}")
        adapter = KaiExecutionContextAdapter(
            namespace="team-a",
            placement_records_enabled=True,
            placement_reader=client,
            history_reader=client,
        )
        mocker.patch.object(
            adapter._kubernetes,
            "get_one",
            return_value=None,
        )
        contexts = adapter.discover_by_ids(["pg-training"])
        replayed = adapter.discover_by_ids(["pg-training"])
        mocker.patch.object(
            adapter._kubernetes,
            "list",
            return_value=[],
        )
        retained_by_time = adapter.discover_by_time_range(
            datetime(2026, 10, 14, 10, 35, tzinfo=timezone.utc),
            datetime(2026, 10, 14, 10, 45, tzinfo=timezone.utc),
        )
        retained_on_node = adapter.discover_by_time_range(
            datetime(2026, 10, 14, 10, 35, tzinfo=timezone.utc),
            datetime(2026, 10, 14, 10, 45, tzinfo=timezone.utc),
            ["node-b"],
        )
    finally:
        server.shutdown()
        server.server_close()

    assert len(contexts) == 2
    assert [item["attempt_id"] for item in replayed] == [
        item["attempt_id"] for item in contexts
    ]
    assert contexts[0]["attempt_id"].startswith("kai:v1:podgroup-uid:")
    assert contexts[0]["state"] == "FAILED"
    assert contexts[0]["nodes"] == ["node-a"]
    assert contexts[1]["state"] == "COMPLETED"
    assert contexts[1]["nodes"] == ["node-b"]
    assert all(item["history_complete"] is False for item in contexts)
    assert [item["attempt_id"] for item in retained_by_time] == [
        contexts[1]["attempt_id"]
    ]
    assert [item["attempt_id"] for item in retained_on_node] == [
        contexts[1]["attempt_id"]
    ]


def test_history_rejects_unsupported_owner_kinds_and_groups():
    unsupported_owners = [
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "name": "trainer",
            "uid": "deployment-uid",
            "controller": True,
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Job",
            "name": "trainer",
            "uid": "job-uid",
            "controller": True,
        },
        {
            "apiVersion": "batch/v1",
            "kind": "JobSet",
            "name": "trainer",
            "uid": "jobset-uid",
            "controller": True,
        },
    ]

    for owner in unsupported_owners:
        history = {
            "observations": [
                _podgroup_observation(
                    owner,
                    "2026-10-14T10:00:00Z",
                    status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
                )
            ]
        }

        assert (
            attempts_from_podgroup_history(
                history,
                pod_group_name="pg-training",
                now=datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
            )
            == []
        )


def test_active_restart_carries_across_scheduling_boundaries():
    owner = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "name": "training",
        "uid": "jobset-uid",
        "controller": True,
    }
    history = {
        "observations": [
            _podgroup_observation(
                owner,
                "2026-10-14T10:00:00Z",
                status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
            ),
            {
                "kind": "Pod",
                "metadata": {"uid": "pod-old", "name": "old-pod"},
                "selectedMetadata": {
                    "annotations": {"pod-group-name": "pg-training"},
                    "labels": {_RESTART_LABEL: "0"},
                },
                "observedAt": "2026-10-14T10:01:00Z",
                "status": {"startTime": "2026-10-14T10:01:00Z"},
            },
            _podgroup_observation(
                owner,
                "2026-10-14T11:00:00Z",
                status={"lastStartTimestamp": "2026-10-14T11:00:00Z"},
            ),
            {
                "kind": "Job",
                "metadata": {"uid": "job-new", "name": "new-job"},
                "selectedMetadata": {
                    "labels": {_RESTART_LABEL: "1"}
                },
                "observedAt": "2026-10-14T11:00:00Z",
                "status": {"startTime": "2026-10-14T11:00:00Z"},
            },
            {
                "kind": "Pod",
                "metadata": {"uid": "pod-new", "name": "new-pod"},
                "selectedMetadata": {
                    "annotations": {"pod-group-name": "pg-training"},
                    "labels": {_RESTART_LABEL: "1"},
                },
                "observedAt": "2026-10-14T11:01:00Z",
                "status": {"startTime": "2026-10-14T11:01:00Z"},
            },
            _podgroup_observation(
                owner,
                "2026-10-14T12:00:00Z",
                status={"lastStartTimestamp": "2026-10-14T12:00:00Z"},
            ),
            {
                "kind": "Pod",
                "metadata": {"uid": "pod-latest", "name": "latest-pod"},
                "selectedMetadata": {
                    "annotations": {"pod-group-name": "pg-training"},
                    "labels": {_RESTART_LABEL: "1"},
                },
                "observedAt": "2026-10-14T12:01:00Z",
                "status": {"startTime": "2026-10-14T12:01:00Z"},
            },
        ]
    }

    attempts = attempts_from_podgroup_history(
        history,
        pod_group_name="pg-training",
        now=datetime(2026, 10, 14, 13, 0, tzinfo=timezone.utc),
    )

    assert len(attempts) == 3
    assert [attempt.restart_attempt for attempt in attempts] == [
        "0",
        "1",
        "1",
    ]
    assert attempts[0].pod_uids == frozenset({"pod-old"})
    assert attempts[1].pod_uids == frozenset({"pod-new"})
    assert attempts[2].pod_uids == frozenset({"pod-latest"})
    assert len({attempt.attempt_id for attempt in attempts}) == 3
    assert attempts[0].end_time == attempts[1].start_time
    assert attempts[1].end_time == attempts[2].start_time


def test_restart_boundaries_are_scoped_to_podgroup_incarnation():
    owner = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "name": "training",
        "uid": "jobset-uid",
        "controller": True,
    }
    history = {
        "observations": [
            _podgroup_observation(
                owner,
                "2026-10-14T10:00:00Z",
                uid="podgroup-old",
                status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
            ),
            {
                "kind": "Pod",
                "metadata": {"uid": "pod-old", "name": "old-pod"},
                "selectedMetadata": {
                    "annotations": {"pod-group-name": "pg-training"},
                    "labels": {_RESTART_LABEL: "0"},
                },
                "observedAt": "2026-10-14T10:01:00Z",
                "status": {"startTime": "2026-10-14T10:01:00Z"},
            },
            _podgroup_observation(
                owner,
                "2026-10-14T10:35:00Z",
                uid="podgroup-old",
                selectedMetadata={"deletionTimestamp": "2026-10-14T10:30:00Z"},
                eventType="DELETED",
            ),
            _podgroup_observation(
                owner,
                "2026-10-14T10:35:00Z",
                uid="podgroup-new",
                status={"lastStartTimestamp": "2026-10-14T10:30:00Z"},
            ),
            {
                "kind": "Pod",
                "metadata": {"uid": "pod-new", "name": "new-pod"},
                "selectedMetadata": {
                    "annotations": {"pod-group-name": "pg-training"},
                    "labels": {_RESTART_LABEL: "1"},
                },
                "observedAt": "2026-10-14T10:40:00Z",
                "status": {"startTime": "2026-10-14T10:40:00Z"},
            },
        ]
    }

    attempts = attempts_from_podgroup_history(
        history,
        pod_group_name="pg-training",
        now=datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
    )

    assert len(attempts) == 2
    old, new = attempts
    assert old.pod_group_uid == "podgroup-old"
    assert old.restart_attempt == "0"
    assert old.pod_uids == frozenset({"pod-old"})
    assert new.pod_group_uid == "podgroup-new"
    assert new.restart_attempt == "1"
    assert new.pod_uids == frozenset({"pod-new"})
    assert all(attempt.end_time >= attempt.start_time for attempt in attempts)


@pytest.mark.parametrize(
    (
        "restart_time",
        "pod_time",
        "initial_pod_time",
        "expected_starts",
        "expected_members",
    ),
    [
        (
            "2026-10-14T09:55:00Z",
            "2026-10-14T09:56:00Z",
            None,
            ["10:00"],
            [frozenset({"pod-current"})],
        ),
        (
            "2026-10-14T10:30:00Z",
            "2026-10-14T10:31:00Z",
            None,
            ["10:30"],
            [frozenset({"pod-current"})],
        ),
        (
            "2026-10-14T10:30:00Z",
            "2026-10-14T10:31:00Z",
            "2026-10-14T10:05:00Z",
            ["10:00", "10:30"],
            [
                frozenset({"pod-initial"}),
                frozenset({"pod-current"}),
            ],
        ),
        (
            "2026-10-14T10:30:00Z",
            "2026-10-14T10:31:00Z",
            "2026-10-14T09:59:00Z",
            ["10:00", "10:30"],
            [
                frozenset({"pod-initial"}),
                frozenset({"pod-current"}),
            ],
        ),
    ],
    ids=[
        "active-before-boundary",
        "first-seen-after-boundary",
        "unlabeled-execution-before-restart",
        "predating-unlabeled-execution",
    ],
)
def test_initial_boundary_preserves_nonzero_restart_evidence_time(
    restart_time,
    pod_time,
    initial_pod_time,
    expected_starts,
    expected_members,
):
    owner = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "name": "training",
        "uid": "jobset-uid",
        "controller": True,
    }
    initial_pods = (
        [
            {
                "kind": "Pod",
                "metadata": {
                    "uid": "pod-initial",
                    "name": "initial-pod",
                },
                "selectedMetadata": {
                    "annotations": {
                        "pod-group-name": "pg-training"
                    }
                },
                "observedAt": initial_pod_time,
                "status": {"startTime": initial_pod_time},
            }
        ]
        if initial_pod_time
        else []
    )
    history = {
        "observations": [
            *initial_pods,
            {
                "kind": "Job",
                "metadata": {"uid": "job-current", "name": "current-child"},
                "selectedMetadata": {"labels": {_RESTART_LABEL: "2"}},
                "observedAt": restart_time,
                "status": {"startTime": restart_time},
            },
            {
                "kind": "Pod",
                "metadata": {"uid": "pod-current", "name": "current-pod"},
                "selectedMetadata": {
                    "annotations": {"pod-group-name": "pg-training"},
                    "labels": {_RESTART_LABEL: "2"},
                },
                "observedAt": pod_time,
                "status": {"startTime": pod_time},
            },
            _podgroup_observation(
                owner,
                "2026-10-14T10:00:00Z",
                status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
            ),
        ]
    }

    attempts = attempts_from_podgroup_history(
        history,
        pod_group_name="pg-training",
        now=datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
    )

    assert [attempt.start_time.strftime("%H:%M") for attempt in attempts] == (
        expected_starts
    )
    assert [attempt.restart_attempt for attempt in attempts] == (
        ["0", "2"] if initial_pod_time else ["2"]
    )
    assert [attempt.pod_uids for attempt in attempts] == expected_members


def test_single_podgroup_includes_pod_that_predates_start():
    owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": "trainer",
        "uid": "job-uid",
        "controller": True,
    }
    history = {
        "observations": [
            {
                "kind": "Pod",
                "metadata": {
                    "uid": "pod-uid",
                    "name": "trainer-pod",
                },
                "selectedMetadata": {"annotations": {"pod-group-name": "pg-training"}},
                "observedAt": "2026-10-14T10:06:00Z",
                "status": {"startTime": "2026-10-14T10:00:00Z"},
            },
            _podgroup_observation(
                owner,
                "2026-10-14T10:06:00Z",
                status={"lastStartTimestamp": "2026-10-14T10:05:00Z"},
            ),
        ]
    }

    attempts = attempts_from_podgroup_history(
        history,
        pod_group_name="pg-training",
        now=datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
    )

    assert len(attempts) == 1
    assert attempts[0].pod_uids == frozenset({"pod-uid"})
    assert attempts[0].membership_resolved is True


@pytest.mark.parametrize(
    "deletion_timestamp",
    [None, "2026-10-14T10:10:00Z"],
    ids=["without-source-timestamp", "with-source-timestamp"],
)
def test_deleted_podgroup_retains_later_owner_terminal_and_pods(
    deletion_timestamp,
):
    owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": "trainer",
        "uid": "job-uid",
        "controller": True,
    }
    history = {
        "observations": [
            _podgroup_observation(
                owner,
                "2026-10-14T10:00:00Z",
                status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
            ),
            {
                "kind": "Pod",
                "metadata": {
                    "uid": "pod-uid",
                    "name": "trainer-pod",
                },
                "selectedMetadata": {"annotations": {"pod-group-name": "pg-training"}},
                "observedAt": "2026-10-14T10:01:00Z",
                "status": {"startTime": "2026-10-14T10:01:00Z"},
            },
            _podgroup_observation(
                owner,
                "2026-10-14T10:15:00Z",
                eventType="DELETED",
                selectedMetadata=(
                    {"deletionTimestamp": deletion_timestamp}
                    if deletion_timestamp
                    else None
                ),
            ),
            {
                "kind": "Job",
                "metadata": {"uid": "job-uid", "name": "trainer"},
                "observedAt": "2026-10-14T10:20:00Z",
                "status": {
                    "conditions": [
                        _condition(
                            "Complete",
                            "2026-10-14T10:20:00Z",
                        )
                    ]
                },
            },
        ]
    }

    attempts = attempts_from_podgroup_history(
        history,
        pod_group_name="pg-training",
        now=datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
    )

    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.boundary_resolved is True
    assert attempt.open is False
    assert attempt.terminal is not None
    assert attempt.terminal["type"] == "Complete"
    assert attempt.pod_uids == frozenset({"pod-uid"})


def test_recreated_podgroups_with_same_restart_isolate_pods_and_terminal():
    owner = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "name": "training",
        "uid": "jobset-uid",
        "controller": True,
    }
    history = {
        "observations": [
            _podgroup_observation(
                owner,
                "2026-10-14T10:00:00Z",
                uid="podgroup-old",
                status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
            ),
            {
                "kind": "Pod",
                "metadata": {
                    "uid": "pod-old",
                    "name": "trainer-old",
                },
                "selectedMetadata": {
                    "annotations": {"pod-group-name": "pg-training"},
                    "labels": {_RESTART_LABEL: "0"},
                },
                "observedAt": "2026-10-14T10:01:00Z",
                "status": {"startTime": "2026-10-14T10:01:00Z"},
            },
            _podgroup_observation(
                owner,
                "2026-10-14T10:15:00Z",
                uid="podgroup-old",
                selectedMetadata={"deletionTimestamp": "2026-10-14T10:10:00Z"},
                eventType="DELETED",
            ),
            _podgroup_observation(
                owner,
                "2026-10-14T10:16:00Z",
                uid="podgroup-new",
                status={"lastStartTimestamp": "2026-10-14T10:10:00Z"},
            ),
            {
                "kind": "Pod",
                "metadata": {
                    "uid": "pod-new",
                    "name": "trainer-new",
                },
                "selectedMetadata": {
                    "annotations": {"pod-group-name": "pg-training"},
                    "labels": {_RESTART_LABEL: "0"},
                },
                "observedAt": "2026-10-14T10:16:00Z",
                "status": {"startTime": "2026-10-14T10:11:00Z"},
            },
            {
                "kind": "JobSet",
                "metadata": {
                    "uid": "jobset-uid",
                    "name": "training",
                },
                "observedAt": "2026-10-14T10:20:00Z",
                "status": {
                    "conditions": [
                        _condition(
                            "Completed",
                            "2026-10-14T10:20:00Z",
                        )
                    ]
                },
            },
        ]
    }

    attempts = attempts_from_podgroup_history(
        history,
        pod_group_name="pg-training",
        now=datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
    )

    assert len(attempts) == 2
    old, new = attempts
    assert old.pod_group_uid == "podgroup-old"
    assert old.end_time == datetime(2026, 10, 14, 10, 10, tzinfo=timezone.utc)
    assert old.terminal is None
    assert old.pod_uids == frozenset({"pod-old"})
    assert new.pod_group_uid == "podgroup-new"
    assert new.terminal is not None
    assert new.terminal["type"] == "Completed"
    assert new.pod_uids == frozenset({"pod-new"})
    assert new.membership_resolved is False

    mixed_history = {
        "observations": history["observations"][:-1],
        "coverage": {},
    }
    [retained] = KaiExecutionContextAdapter._evidence_from_histories(
        [mixed_history],
        pod_group_name="pg-training",
    )
    live_podgroup = {
        "apiVersion": "scheduling.run.ai/v2alpha2",
        "kind": "PodGroup",
        "metadata": {
            "namespace": "team-a",
            "name": "pg-training",
            "uid": "podgroup-new",
            "ownerReferences": [owner],
        },
        "status": {"lastStartTimestamp": "2026-10-14T10:10:00Z"},
    }
    live_owner = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "namespace": "team-a",
            "name": "training",
            "uid": "jobset-uid",
        },
        "status": {},
    }
    live_pods = [
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "namespace": "team-a",
                "name": "trainer-old",
                "uid": "pod-old",
                "annotations": {"pod-group-name": "pg-training"},
                "labels": {_RESTART_LABEL: "0"},
            },
            "spec": {"nodeName": "old-node"},
            "status": {
                "startTime": "2026-10-14T10:01:00Z",
                "reason": "NodeLost",
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "namespace": "team-a",
                "name": "trainer-new",
                "uid": "pod-new",
                "annotations": {"pod-group-name": "pg-training"},
                "labels": {_RESTART_LABEL: "0"},
            },
            "spec": {"nodeName": "new-node"},
            "status": {"startTime": "2026-10-14T10:11:00Z"},
        },
    ]
    now = datetime(2026, 10, 14, 11, tzinfo=timezone.utc)
    live_observations = observations_from_live_resources(
        live_podgroup,
        live_pods,
        [],
        live_owner,
        now=now,
    )
    combined_items = KaiExecutionContextAdapter._combine_attempt_evidence(
        [
            retained,
            _WorkloadEvidence(
                namespace="team-a",
                pod_group_name="pg-training",
                observed_until=now,
                podgroup=live_podgroup,
                owner_object=live_owner,
                pods={pod["metadata"]["uid"]: pod for pod in live_pods},
                observations=live_observations,
            ),
        ]
    )
    combined = next(
        evidence
        for evidence in combined_items
        if evidence.attempt.pod_group_uid == "podgroup-new"
    )
    old_combined = next(
        evidence
        for evidence in combined_items
        if evidence.attempt.pod_group_uid == "podgroup-old"
    )
    context = _context_from_attempt_evidence(combined)

    assert combined.attempt.pod_uids == frozenset({"pod-new"})
    assert old_combined.owner_object is None
    assert _context_from_attempt_evidence(old_combined)["state"] == "NODE_FAIL"
    assert context["state"] == "RUNNING"
    assert context["nodes"] == ["new-node"]


def test_unstarted_replacement_keeps_closed_old_execution_recoverable():
    owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": "trainer",
        "uid": "job-uid",
        "controller": True,
    }
    history = {
        "observations": [
            {
                "kind": "Pod",
                "metadata": {
                    "uid": "pod-old",
                    "name": "trainer-old",
                },
                "selectedMetadata": {"annotations": {"pod-group-name": "pg-training"}},
                "observedAt": "2026-10-14T10:01:00Z",
                "status": {"startTime": "2026-10-14T10:01:00Z"},
            },
            _podgroup_observation(
                owner, "2026-10-14T10:14:00Z", uid="podgroup-old", eventType="DELETED"
            ),
            _podgroup_observation(
                owner,
                "2026-10-14T10:00:00Z",
                uid="podgroup-old",
                status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
            ),
            _podgroup_observation(owner, "2026-10-14T10:15:00Z", uid="podgroup-new"),
            {
                "kind": "Job",
                "metadata": {"uid": "job-uid", "name": "trainer"},
                "observedAt": "2026-10-14T10:20:00Z",
                "status": {
                    "conditions": [
                        _condition(
                            "Complete",
                            "2026-10-14T10:20:00Z",
                        )
                    ]
                },
            },
        ]
    }

    attempts = attempts_from_podgroup_history(
        history,
        pod_group_name="pg-training",
        now=datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
    )

    assert len(attempts) == 1
    assert attempts[0].boundary_resolved is True
    assert attempts[0].terminal is not None
    assert attempts[0].terminal["type"] == "Complete"
    assert attempts[0].open is False
    assert attempts[0].pod_uids == frozenset({"pod-old"})


def test_next_scheduling_start_preserves_earlier_membership_after_gc():
    owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": "trainer",
        "uid": "job-uid",
        "controller": True,
    }
    history = {
        "observations": [
            _podgroup_observation(
                owner,
                "2026-10-14T10:00:00Z",
                uid="podgroup-old",
                status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
            ),
            {
                "kind": "Pod",
                "metadata": {"uid": "pod-first", "name": "first-pod"},
                "selectedMetadata": {
                    "annotations": {"pod-group-name": "pg-training"},
                    "labels": {_RESTART_LABEL: "0"},
                },
                "observedAt": "2026-10-14T10:01:00Z",
                "status": {"startTime": "2026-10-14T10:01:00Z"},
            },
            _podgroup_observation(
                owner,
                "2026-10-14T11:00:00Z",
                uid="podgroup-old",
                status={"lastStartTimestamp": "2026-10-14T11:00:00Z"},
            ),
            {
                "kind": "Pod",
                "metadata": {"uid": "pod-second", "name": "second-pod"},
                "selectedMetadata": {
                    "annotations": {"pod-group-name": "pg-training"},
                    "labels": {_RESTART_LABEL: "0"},
                },
                "observedAt": "2026-10-14T11:01:00Z",
                "status": {"startTime": "2026-10-14T11:01:00Z"},
            },
            _podgroup_observation(
                owner, "2026-10-14T11:50:00Z", uid="podgroup-old", eventType="DELETED"
            ),
            _podgroup_observation(
                owner, "2026-10-14T12:00:00Z", uid="podgroup-unscheduled"
            ),
            {
                "kind": "Job",
                "metadata": {"uid": "job-uid", "name": "trainer"},
                "observedAt": "2026-10-14T12:10:00Z",
                "status": {
                    "conditions": [
                        _condition(
                            "Complete",
                            "2026-10-14T12:10:00Z",
                        )
                    ]
                },
            },
        ]
    }

    attempts = attempts_from_podgroup_history(
        history,
        pod_group_name="pg-training",
        now=datetime(2026, 10, 14, 13, 0, tzinfo=timezone.utc),
    )

    assert len(attempts) == 2
    first, second = attempts
    assert first.boundary_resolved is True
    assert first.terminal is None
    assert first.pod_uids == frozenset({"pod-first"})
    assert second.boundary_resolved is True
    assert second.terminal is not None
    assert second.pod_uids == frozenset({"pod-second"})


def test_unstarted_replacement_without_terminal_remains_uncertain():
    owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": "trainer",
        "uid": "job-uid",
        "controller": True,
    }
    history = {
        "observations": [
            _podgroup_observation(
                owner,
                "2026-10-14T10:00:00Z",
                uid="podgroup-old",
                status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
            ),
            _podgroup_observation(owner, "2026-10-14T10:15:00Z", uid="podgroup-new"),
        ]
    }

    attempts = attempts_from_podgroup_history(
        history,
        pod_group_name="pg-training",
        now=datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
    )

    assert len(attempts) == 1
    assert attempts[0].boundary_resolved is False
    assert attempts[0].terminal is None
    assert attempts[0].open is False


@pytest.mark.parametrize(
    "completed, expected_state", [(False, "NODE_FAIL"), (True, "COMPLETED")]
)
def test_retained_member_pod_node_loss_does_not_require_placement(
    completed, expected_state
):
    owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": "trainer",
        "uid": "job-uid",
        "controller": True,
    }
    history = {
        "observations": [
            _podgroup_observation(
                owner,
                "2026-10-14T10:00:00Z",
                status={"lastStartTimestamp": "2026-10-14T10:00:00Z"},
            ),
            {
                "kind": "Pod",
                "metadata": {"uid": "lost-pod", "name": "lost-pod"},
                "selectedMetadata": {"annotations": {"pod-group-name": "pg-training"}},
                "observedAt": "2026-10-14T10:01:00Z",
                "status": {
                    "startTime": "2026-10-14T10:01:00Z",
                    "reason": "NodeLost",
                },
            },
        ]
    }
    if completed:
        history["observations"].append(
            {
                "kind": "Job",
                "metadata": {"uid": "job-uid", "name": "trainer"},
                "observedAt": "2026-10-14T11:00:00Z",
                "status": {
                    "conditions": [_condition("Complete", "2026-10-14T11:00:00Z")]
                },
            }
        )
    [workload_evidence] = KaiExecutionContextAdapter._evidence_from_histories(
        [history],
        pod_group_name="pg-training",
    )
    [evidence] = KaiExecutionContextAdapter._combine_attempt_evidence(
        [workload_evidence]
    )
    context = _context_from_attempt_evidence(evidence)

    assert context["state"] == expected_state
    assert context["nodes"] == []
    assert "failed_node" not in context


def test_reconstructed_attempt_rejects_time_only_placement_membership():
    attempt = PodGroupAttempt(
        pod_group_uid="podgroup-uid",
        namespace="team-a",
        pod_group_name="pg-training",
        owner_kind="JobSet",
        owner_name="training",
        owner_uid="jobset-uid",
        start_key="2026-10-14T10:00:00Z",
        restart_attempt="1",
        start_time=datetime(2026, 10, 14, 10, tzinfo=timezone.utc),
        end_time=datetime(2026, 10, 14, 11, tzinfo=timezone.utc),
        terminal=None,
        open=True,
        boundary_resolved=True,
        membership_resolved=True,
        pod_uids=frozenset({"current-pod"}),
    )
    context = _context_from_attempt_evidence(
        _AttemptEvidence(
            attempt=attempt,
            placements={
                "current-pod": PlacementRecord(
                    namespace="team-a",
                    pod_name="current-pod",
                    pod_uid="current-pod",
                    node_name="node-current",
                    observed_at="2026-10-14T10:05:00Z",
                    scheduled_at="2026-10-14T10:05:00Z",
                ),
                "excluded-pod": PlacementRecord(
                    namespace="team-a",
                    pod_name="excluded-pod",
                    pod_uid="excluded-pod",
                    node_name="node-excluded",
                    observed_at="2026-10-14T10:06:00Z",
                    scheduled_at="2026-10-14T10:06:00Z",
                    node_failure_reason="NodeLost",
                ),
            },
        )
    )

    assert context["state"] == "RUNNING"
    assert context["nodes"] == ["node-current"]
    assert "failed_node" not in context
