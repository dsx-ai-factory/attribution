# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import random
from datetime import datetime, timedelta, timezone

import pytest

from attribution_blueprint.connectors.execution_contexts.kai.attempts import (
    PodGroupAttempt,
    _normalize_snapshot,
    attempts_from_podgroup_history,
    snapshots_from_live_resources,
)
from attribution_blueprint.connectors.execution_contexts.kai.connector import (
    KaiExecutionContextAdapter,
    _AttemptEvidence,
    _context_from_attempt_evidence,
    _latest_resources_by_uid,
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


def _dt(seconds):
    return datetime(2026, 10, 14, 10, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def _ts(seconds):
    return _dt(seconds).isoformat().replace("+00:00", "Z")


def _snapshot(kind, uid, observed_at, *, metadata=None, **fields):
    """Share identity boilerplate; lifecycle evidence stays explicit in each test."""
    return {
        "kind": kind,
        "metadata": {"uid": uid, **(metadata or {})},
        "observedAt": observed_at,
        **fields,
    }


def _reconstruct(history):
    return attempts_from_podgroup_history(
        history, pod_group_name="pg-training", now=datetime(2026, 10, 14, 11, tzinfo=timezone.utc)
    )


def _live_resource(snapshot, **fields):
    """Build a live API resource from explicit test snapshot fields."""
    return {
        "apiVersion": {
            "Pod": "v1",
            "JobSet": "jobset.x-k8s.io/v1alpha2",
            "PodGroup": "scheduling.run.ai/v2alpha2",
        }[snapshot["kind"]],
        "kind": snapshot["kind"],
        "metadata": {
            "namespace": "team-a",
            **snapshot["metadata"],
            **snapshot.get("selectedMetadata", {}),
        },
        "status": copy.deepcopy(snapshot.get("status", {})),
        **fields,
    }


def _placement(uid, node, seconds, **fields):
    return PlacementRecord(
        namespace="team-a",
        pod_name=uid,
        pod_uid=uid,
        node_name=node,
        observed_at=_ts(seconds),
        scheduled_at=_ts(seconds),
        **fields,
    )


def _retained_attempts(history):
    evidence = KaiExecutionContextAdapter._evidence_from_histories(
        [history], pod_group_name="pg-training"
    )
    return KaiExecutionContextAdapter._combine_attempt_evidence(evidence)


def _owner_reference(kind: str, name: str, uid: str) -> OwnerReference:
    return OwnerReference(
        api_version=("jobset.x-k8s.io/v1alpha2" if kind == "JobSet" else "batch/v1"),
        kind=kind,
        name=name,
        uid=uid,
        controller=True,
    )


def _owner(kind: str, name: str, uid: str) -> dict:
    return {
        "apiVersion": ("jobset.x-k8s.io/v1alpha2" if kind == "JobSet" else "batch/v1"),
        "kind": kind,
        "name": name,
        "uid": uid,
        "controller": True,
    }


def _pod(
    uid: str,
    name: str,
    observed_at: str,
    *,
    start_time: str,
    restart: str | None = None,
    pod_group_name: str = "pg-training",
    node: str | None = None,
    reason: str | None = None,
    job_uid: str = "job-uid",
) -> dict:
    selected_metadata = {"annotations": {"pod-group-name": pod_group_name}}
    if restart is not None:
        selected_metadata["labels"] = {_RESTART_LABEL: restart}
    status = {"startTime": start_time}
    if reason is not None:
        status["reason"] = reason
    observation = _snapshot(
        "Pod",
        uid,
        observed_at,
        metadata={
            "name": name,
            "ownerReferences": [{"kind": "Job", "uid": job_uid, "controller": True}],
        },
        selectedMetadata=selected_metadata,
        status=status,
    )
    if node is not None:
        observation["spec"] = {"nodeName": node}
    return observation


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


def _podgroup(owner: dict, observed_at: str, *, uid: str = "podgroup-uid", **fields) -> dict:
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


def _condition(condition_type: str, transition: str, *, reason: str | None = None) -> dict:
    condition = {"type": condition_type, "status": "True", "lastTransitionTime": transition}
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
    jobset_owner = _owner_reference("JobSet", "training", "jobset-uid")
    podgroup_owner = OwnerReference(
        api_version="jobset.x-k8s.io/v1alpha2",
        kind="JobSet",
        name="training",
        uid="jobset-uid",
        controller=False,
    )
    top_owner = {
        "annotations": {
            "kai.scheduler/top-owner-metadata": (
                "kind: JobSet\nname: training\nuid: jobset-uid\ngroup: jobset.x-k8s.io\n"
            )
        }
    }
    observations = [
        _observation(
            kind="JobSet",
            name="training",
            uid="jobset-uid",
            resource_version="1",
            observed_at=_ts(-60),
        ),
        _observation(
            kind="PodGroup",
            name="pg-training",
            uid="podgroup-uid",
            resource_version="2",
            observed_at=_ts(0),
            owners=(podgroup_owner,),
            selected_metadata=top_owner,
            status={"lastStartTimestamp": _ts(0)},
        ),
        _observation(
            kind="JobSet",
            name="training",
            uid="jobset-uid",
            resource_version="5",
            observed_at=_ts(1860),
            status={
                "restarts": 1,
                "conditions": [
                    _condition("RestartingJobSet", _ts(1860), reason="DefaultFailurePolicy")
                ],
            },
        ),
        _observation(
            kind="JobSet",
            name="training",
            uid="jobset-uid",
            resource_version="8",
            observed_at=_ts(3600),
            status={
                "restarts": 1,
                "terminalState": "Completed",
                "conditions": [_condition("Completed", _ts(3600))],
            },
        ),
    ]
    for generation, job_start, pod_start, job_version, pod_version in (
        ("0", 0, 60, "3", "4"),
        ("1", 1860, 1920, "6", "7"),
    ):
        job_name, job_uid = f"training-worker-{generation}", f"job-uid-{generation}"
        labels = {_RESTART_LABEL: generation}
        observations.extend(
            [
                _observation(
                    kind="Job",
                    name=job_name,
                    uid=job_uid,
                    resource_version=job_version,
                    observed_at=_ts(job_start),
                    owners=(jobset_owner,),
                    selected_metadata={"labels": labels},
                    status={"startTime": _ts(job_start)},
                ),
                _observation(
                    kind="Pod",
                    name=f"{job_name}-pod",
                    uid=f"pod-uid-{generation}",
                    resource_version=pod_version,
                    observed_at=_ts(pod_start),
                    owners=(_owner_reference("Job", job_name, job_uid),),
                    selected_metadata={
                        "annotations": {"pod-group-name": "pg-training"},
                        "labels": labels,
                    },
                    status={"startTime": _ts(pod_start)},
                ),
            ]
        )
    observations.sort(key=lambda snapshot: int(snapshot.resource_version))
    for observation in observations:
        _capture(store, observation)

    podgroup = ResourceIdentity(
        "scheduling.run.ai/v2alpha2", "PodGroup", "team-a", "pg-training", "podgroup-uid"
    )
    for pod_name, pod_uid, node_name, observed_at in (
        ("training-worker-0-pod", "pod-uid-0", "node-a", _ts(60)),
        ("training-worker-1-pod", "pod-uid-1", "node-b", _ts(1920)),
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
    server = serve_api(HealthState(), store, 0, "team-a", cluster_id="cluster-a")
    try:
        client = KubernetesDaemonClient(f"http://127.0.0.1:{server.server_port}")
        adapter = KaiExecutionContextAdapter(
            namespace="team-a",
            placement_records_enabled=True,
            placement_reader=client,
            history_reader=client,
        )
        mocker.patch.object(adapter._kubernetes, "get_one", return_value=None)
        contexts = adapter.discover_by_ids(["pg-training"])
        replayed = adapter.discover_by_ids(["pg-training"])
        mocker.patch.object(adapter._kubernetes, "list", return_value=[])
        retained_by_time = adapter.discover_by_time_range(_dt(2100), _dt(2700))
        retained_on_node = adapter.discover_by_time_range(_dt(2100), _dt(2700), ["node-b"])
    finally:
        server.shutdown()
        server.server_close()

    assert len(contexts) == 2
    assert [item["attempt_id"] for item in replayed] == [item["attempt_id"] for item in contexts]
    assert contexts[0]["attempt_id"].startswith("kai:v1:podgroup-uid:")
    assert contexts[0]["state"] == "FAILED"
    assert contexts[0]["nodes"] == ["node-a"]
    assert contexts[1]["state"] == "COMPLETED"
    assert contexts[1]["nodes"] == ["node-b"]
    assert all(item["history_complete"] is False for item in contexts)
    assert [item["attempt_id"] for item in retained_by_time] == [contexts[1]["attempt_id"]]
    assert [item["attempt_id"] for item in retained_on_node] == [contexts[1]["attempt_id"]]


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
            "observations": [_podgroup(owner, _ts(0), status={"lastStartTimestamp": _ts(0)})]
        }

        assert _reconstruct(history) == []


def test_active_restart_carries_across_scheduling_boundaries():
    owner = _owner("JobSet", "training", "jobset-uid")
    history = {
        "observations": [
            _podgroup(owner, _ts(0), status={"lastStartTimestamp": _ts(0)}),
            _pod("pod-old", "old-pod", _ts(60), start_time=_ts(60), restart="0", job_uid="job-old"),
            _podgroup(owner, _ts(3600), status={"lastStartTimestamp": _ts(3600)}),
            _snapshot(
                "Job",
                "job-new",
                _ts(3600),
                metadata={"name": "new-job", "ownerReferences": [owner]},
                selectedMetadata={"labels": {_RESTART_LABEL: "1"}},
                status={"startTime": _ts(3600)},
            ),
            _pod(
                "pod-new",
                "new-pod",
                _ts(3660),
                start_time=_ts(3660),
                restart="1",
                job_uid="job-new",
            ),
            _podgroup(owner, _ts(7200), status={"lastStartTimestamp": _ts(7200)}),
            _pod(
                "pod-latest",
                "latest-pod",
                _ts(7260),
                start_time=_ts(7260),
                restart="1",
                job_uid="job-new",
            ),
        ]
    }

    history["observations"].append(
        _snapshot(
            "Job",
            "job-old",
            _ts(60),
            metadata={"ownerReferences": [owner]},
            selectedMetadata={"labels": {_RESTART_LABEL: "0"}},
        )
    )
    attempts = attempts_from_podgroup_history(history, pod_group_name="pg-training", now=_dt(10800))

    assert len(attempts) == 3
    assert [attempt.restart_attempt for attempt in attempts] == ["0", "1", "1"]
    assert attempts[0].pod_uids == frozenset({"pod-old"})
    assert attempts[1].pod_uids == frozenset({"pod-new"})
    assert attempts[2].pod_uids == frozenset({"pod-latest"})
    assert len({attempt.attempt_id for attempt in attempts}) == 3
    assert attempts[0].end_time == attempts[1].start_time
    assert attempts[1].end_time == attempts[2].start_time


def test_restart_boundaries_are_scoped_to_podgroup_incarnation():
    owner = _owner("JobSet", "training", "jobset-uid")
    history = {
        "observations": [
            _podgroup(owner, _ts(0), uid="podgroup-old", status={"lastStartTimestamp": _ts(0)}),
            _pod(
                "pod-old", "old-pod", _ts(60), start_time=_ts(60), restart="1", job_uid="worker-job"
            ),
            _podgroup(
                owner,
                _ts(2100),
                uid="podgroup-old",
                selectedMetadata={"deletionTimestamp": _ts(1800)},
                eventType="DELETED",
            ),
            _podgroup(
                owner, _ts(2100), uid="podgroup-new", status={"lastStartTimestamp": _ts(1800)}
            ),
            _pod(
                "pod-new",
                "new-pod",
                _ts(2400),
                start_time=_ts(2400),
                restart="1",
                job_uid="worker-job",
            ),
        ]
    }

    history["observations"].append(
        _snapshot(
            "Job",
            "worker-job",
            _ts(60),
            metadata={"ownerReferences": [owner]},
            selectedMetadata={"labels": {_RESTART_LABEL: "1"}},
        )
    )
    attempts = _reconstruct(history)

    assert len(attempts) == 2
    old, new = attempts
    assert old.pod_group_uid == "podgroup-old"
    assert old.restart_attempt == "1"
    assert old.pod_uids == frozenset({"pod-old"})
    assert new.pod_group_uid == "podgroup-new"
    assert new.restart_attempt == "1"
    assert new.pod_uids == frozenset({"pod-new"})
    assert all(attempt.end_time >= attempt.start_time for attempt in attempts)


@pytest.mark.parametrize(
    ("restart_time", "pod_time", "initial_pod_time", "expected_starts", "expected_members"),
    [
        (_ts(-300), _ts(-240), None, ["10:00"], [frozenset({"pod-current"})]),
        (_ts(1800), _ts(1860), None, ["10:30"], [frozenset({"pod-current"})]),
        (
            _ts(1800),
            _ts(1860),
            _ts(300),
            ["10:00", "10:30"],
            [frozenset({"pod-initial"}), frozenset({"pod-current"})],
        ),
        (
            _ts(1800),
            _ts(1860),
            _ts(-60),
            ["10:00", "10:30"],
            [frozenset({"pod-initial"}), frozenset({"pod-current"})],
        ),
    ],
    ids=[
        "active-before-boundary",
        "first-seen-after-boundary",
        "unlabeled-execution-before-restart",
        "predating-unlabeled-execution",
    ],
)
@pytest.mark.parametrize("owner_kind", ["JobSet", "Job"])
def test_initial_boundary_preserves_nonzero_restart_evidence_time(
    restart_time, pod_time, initial_pod_time, expected_starts, expected_members, owner_kind
):
    owner = _owner(owner_kind, "training", "owner-uid")
    initial_pods = (
        [
            _pod(
                "pod-initial",
                "initial-pod",
                initial_pod_time,
                start_time=initial_pod_time,
                job_uid=("owner-uid" if owner_kind == "Job" else "job-initial"),
            )
        ]
        if initial_pod_time
        else []
    )
    job_metadata = {
        "uid": "owner-uid" if owner_kind == "Job" else "job-current",
        "name": "current-child",
    }
    if owner_kind == "JobSet":
        job_metadata["ownerReferences"] = [owner]
    history = {
        "observations": [
            *initial_pods,
            {
                "kind": "Job",
                "metadata": job_metadata,
                "selectedMetadata": {"labels": {_RESTART_LABEL: "2"}},
                "observedAt": restart_time,
                "status": {"startTime": restart_time},
            },
            _pod(
                "pod-current",
                "current-pod",
                pod_time,
                start_time=pod_time,
                restart="2",
                job_uid=("owner-uid" if owner_kind == "Job" else "job-current"),
            ),
            _podgroup(owner, _ts(0), status={"lastStartTimestamp": _ts(0)}),
        ]
    }

    if initial_pod_time and owner_kind == "JobSet":
        history["observations"].append(
            _snapshot("Job", "job-initial", initial_pod_time, metadata={"ownerReferences": [owner]})
        )
    attempts = _reconstruct(history)

    assert [attempt.start_time.strftime("%H:%M") for attempt in attempts] == (expected_starts)
    assert [attempt.restart_attempt for attempt in attempts] == (
        ["0", "2"] if initial_pod_time else ["2"]
    )
    assert [attempt.pod_uids for attempt in attempts] == expected_members
    assert all(attempt.membership_resolved for attempt in attempts)


@pytest.mark.parametrize(
    "deletion_timestamp",
    [None, _ts(600)],
    ids=["without-source-timestamp", "with-source-timestamp"],
)
@pytest.mark.parametrize(
    "completion_status",
    [{"conditions": [_condition("Complete", _ts(1200))]}, {"completionTime": _ts(1200)}],
    ids=["complete-condition", "completion-time"],
)
def test_deleted_podgroup_retains_later_owner_terminal_and_pods(
    deletion_timestamp, completion_status
):
    owner = _owner("Job", "trainer", "job-uid")
    history = {
        "observations": [
            _podgroup(owner, _ts(0), status={"lastStartTimestamp": _ts(0)}),
            _pod("pod-uid", "trainer-pod", _ts(60), start_time=_ts(60)),
            _podgroup(
                owner,
                _ts(900),
                eventType="DELETED",
                selectedMetadata=(
                    {"deletionTimestamp": deletion_timestamp} if deletion_timestamp else None
                ),
            ),
            _snapshot(
                "Job", "job-uid", _ts(1200), metadata={"name": "trainer"}, status=completion_status
            ),
        ]
    }

    attempts = _reconstruct(history)

    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.boundary_resolved is True
    assert attempt.open is False
    assert attempt.terminal is not None
    assert attempt.terminal["type"] == "Complete"
    assert attempt.pod_uids == frozenset({"pod-uid"})


def test_recreated_podgroups_with_same_restart_isolate_pods_and_terminal():
    owner = _owner("JobSet", "training", "jobset-uid")
    history = {
        "observations": [
            _podgroup(owner, _ts(0), uid="podgroup-old", status={"lastStartTimestamp": _ts(0)}),
            _pod(
                "pod-old",
                "trainer-old",
                _ts(60),
                start_time=_ts(60),
                restart="0",
                job_uid="worker-job",
            ),
            _podgroup(
                owner,
                _ts(900),
                uid="podgroup-old",
                selectedMetadata={"deletionTimestamp": _ts(600)},
                eventType="DELETED",
            ),
            _podgroup(owner, _ts(960), uid="podgroup-new", status={"lastStartTimestamp": _ts(600)}),
            _pod(
                "pod-new",
                "trainer-new",
                _ts(960),
                start_time=_ts(660),
                restart="0",
                job_uid="worker-job",
            ),
            _snapshot(
                "JobSet",
                "jobset-uid",
                _ts(1200),
                metadata={"name": "training"},
                status={"conditions": [_condition("Completed", _ts(1200))]},
            ),
        ]
    }

    history["observations"].append(
        _snapshot(
            "Job",
            "worker-job",
            _ts(60),
            metadata={"ownerReferences": [owner]},
            selectedMetadata={"labels": {_RESTART_LABEL: "0"}},
        )
    )
    attempts = _reconstruct(history)

    assert len(attempts) == 2
    old, new = attempts
    assert old.pod_group_uid == "podgroup-old"
    assert old.end_time == _dt(600)
    assert old.terminal is None
    assert old.pod_uids == frozenset({"pod-old"})
    assert new.pod_group_uid == "podgroup-new"
    assert new.terminal is not None
    assert new.terminal["type"] == "Completed"
    assert new.pod_uids == frozenset({"pod-new"})
    assert new.membership_resolved is False

    mixed_history = {
        "observations": history["observations"][:5] + history["observations"][6:],
        "coverage": {},
    }
    [retained] = KaiExecutionContextAdapter._evidence_from_histories(
        [mixed_history], pod_group_name="pg-training"
    )
    records = history["observations"]
    live_podgroup = _live_resource(records[3])
    live_owner = _live_resource(records[5], status={})
    live_pods = [
        _live_resource(records[1], spec={"nodeName": "old-node"}),
        _live_resource(records[4], spec={"nodeName": "new-node"}),
    ]
    live_pods[0]["status"]["reason"] = "NodeLost"
    for pod in live_pods:
        pod["metadata"]["resourceVersion"] = "2"
    now = datetime(2026, 10, 14, 11, tzinfo=timezone.utc)
    live_observations = snapshots_from_live_resources(
        live_podgroup, live_pods, [], live_owner, now=now
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
                observations=live_observations,
            ),
        ]
    )
    combined = next(
        evidence for evidence in combined_items if evidence.attempt.pod_group_uid == "podgroup-new"
    )
    old_combined = next(
        evidence for evidence in combined_items if evidence.attempt.pod_group_uid == "podgroup-old"
    )
    context = _context_from_attempt_evidence(combined)

    assert combined.attempt.pod_uids == frozenset({"pod-new"})
    assert old_combined.owner_object is None
    assert _context_from_attempt_evidence(old_combined)["state"] == "NODE_FAIL"
    assert context["state"] == "RUNNING"
    assert context["nodes"] == ["new-node"]


def test_recreated_podgroup_excludes_foreign_owner_job_failure():
    old_owner = _owner("JobSet", "old-training", "old-owner")
    new_owner = _owner("JobSet", "new-training", "new-owner")
    start = _ts(660)
    finish = _ts(1200)

    def child_job(uid, name, owner, reason):
        return _snapshot(
            "Job",
            uid,
            start,
            metadata={"name": name, "ownerReferences": [owner]},
            selectedMetadata={"labels": {_RESTART_LABEL: "1"}},
            status={
                "startTime": start,
                "conditions": [_condition("Failed", finish, reason=reason)],
            },
        )

    foreign_pod = _pod(
        "foreign-pod",
        "old-pod",
        _ts(720),
        start_time=_ts(720),
        restart="1",
        node="old-owner-node",
        reason="NodeLost",
        job_uid="new-job",
    )
    foreign_pod["metadata"]["ownerReferences"] = [
        {"kind": "Job", "uid": "foreign-old-job", "controller": True}
    ]
    history = {
        "observations": [
            _podgroup(
                old_owner,
                _ts(540),
                uid="podgroup-old",
                eventType="DELETED",
                selectedMetadata={"deletionTimestamp": _ts(540)},
                status={"lastStartTimestamp": _ts(0)},
            ),
            _podgroup(
                new_owner, start, uid="podgroup-new", status={"lastStartTimestamp": _ts(600)}
            ),
            child_job("foreign-old-job", "old-worker", old_owner, "DeadlineExceeded"),
            foreign_pod,
            child_job("new-job", "new-worker", new_owner, "BackoffLimitExceeded"),
            _pod(
                "pod-new",
                "new-pod",
                _ts(720),
                start_time=_ts(720),
                restart="1",
                node="new-owner-node",
                job_uid="new-job",
            ),
            _snapshot(
                "JobSet",
                "new-owner",
                finish,
                metadata={"name": "new-training"},
                status={"conditions": [_condition("Failed", finish)]},
            ),
        ]
    }

    new_evidence = next(
        evidence
        for evidence in _retained_attempts(history)
        if evidence.attempt.pod_group_uid == "podgroup-new"
    )
    assert new_evidence.attempt.job_uids == frozenset({"new-job"})
    assert new_evidence.attempt.pod_uids == frozenset({"pod-new"})
    context = _context_from_attempt_evidence(new_evidence)
    assert context["state"] == "FAILED"
    assert context["owner_state"] == "Failed"
    assert context["nodes"] == ["new-owner-node"]
    assert "failed_node" not in context


@pytest.mark.parametrize("reallocated", [False, True])
def test_next_scheduling_start_preserves_earlier_membership_after_gc(reallocated):
    owner = _owner("Job", "trainer", "job-uid")
    history = {
        "observations": [
            _podgroup(owner, _ts(0), uid="podgroup-old", status={"lastStartTimestamp": _ts(0)}),
            _pod("pod-first", "first-pod", _ts(60), start_time=_ts(60), restart="0"),
            _podgroup(
                owner, _ts(3600), uid="podgroup-old", status={"lastStartTimestamp": _ts(3600)}
            ),
            _pod("pod-second", "second-pod", _ts(3660), start_time=_ts(3660), restart="0"),
            _podgroup(owner, _ts(6600), uid="podgroup-old", eventType="DELETED"),
            _podgroup(owner, _ts(7200), uid="podgroup-unscheduled"),
            _snapshot(
                "Job",
                "job-uid",
                _ts(7800),
                metadata={"name": "trainer"},
                status={"conditions": [_condition("Complete", _ts(7800))]},
            ),
        ]
    }

    if not reallocated:
        del history["observations"][2:4]
    attempts = attempts_from_podgroup_history(history, pod_group_name="pg-training", now=_dt(10800))

    assert len(attempts) == (2 if reallocated else 1)
    last = attempts[-1]
    assert last.boundary_resolved is True
    assert last.terminal["type"] == "Complete"
    assert last.open is False
    assert last.pod_uids == frozenset({"pod-second" if reallocated else "pod-first"})
    if reallocated:
        assert attempts[0].boundary_resolved is True
        assert attempts[0].terminal is None
        assert attempts[0].pod_uids == frozenset({"pod-first"})


def test_unstarted_replacement_without_terminal_remains_uncertain():
    owner = _owner("Job", "trainer", "job-uid")
    history = {
        "observations": [
            _podgroup(owner, _ts(0), uid="podgroup-old", status={"lastStartTimestamp": _ts(0)}),
            _podgroup(owner, _ts(900), uid="podgroup-new"),
        ]
    }

    attempts = _reconstruct(history)

    assert len(attempts) == 1
    assert attempts[0].boundary_resolved is False
    assert attempts[0].terminal is None
    assert attempts[0].open is False


@pytest.mark.parametrize("node", [None, "node-a"])
@pytest.mark.parametrize("completed, expected_state", [(False, "NODE_FAIL"), (True, "COMPLETED")])
def test_retained_member_pod_node_loss_does_not_require_placement(completed, expected_state, node):
    owner = _owner("Job", "trainer", "job-uid")
    history = {
        "observations": [
            _podgroup(owner, _ts(0), status={"lastStartTimestamp": _ts(0)}),
            _pod("lost-pod", "lost-pod", _ts(60), start_time=_ts(60), reason="NodeLost", node=node),
        ]
    }
    if completed:
        history["observations"].append(
            _snapshot(
                "Job",
                "job-uid",
                _ts(3600),
                metadata={"name": "trainer"},
                status={"conditions": [_condition("Complete", _ts(3600))]},
            )
        )
    [evidence] = _retained_attempts(history)
    context = _context_from_attempt_evidence(evidence)

    assert context["state"] == expected_state
    assert context["nodes"] == ([node] if node else [])
    assert context.get("failed_node") == (node if not completed else None)


def test_latest_pod_deterministically_controls_context_state():

    owner = _owner("Job", "trainer", "job-uid")
    healthy = _pod("pod-uid", "trainer-pod", _ts(60), start_time=_ts(60), node="node-a")
    lost = _pod(
        "pod-uid", "trainer-pod", _ts(120), start_time=_ts(60), node="node-a", reason="NodeLost"
    )
    observations = [_podgroup(owner, _ts(0), status={"lastStartTimestamp": _ts(0)}), healthy, lost]

    for seed in range(10):
        replay = copy.deepcopy(observations + observations)
        random.Random(seed).shuffle(replay)
        [evidence] = KaiExecutionContextAdapter._combine_attempt_evidence(
            [
                _WorkloadEvidence(
                    namespace="team-a",
                    pod_group_name="pg-training",
                    observed_until=datetime(2026, 10, 14, 11, tzinfo=timezone.utc),
                    observations=replay,
                )
            ]
        )
        context = _context_from_attempt_evidence(evidence)
        assert context["state"] == "NODE_FAIL"
        assert context["failed_node"] == "node-a"
        assert context["nodes"] == ["node-a"]


def test_reconstructed_attempt_rejects_time_only_placement_membership():
    attempt = PodGroupAttempt(
        pod_group_uid="podgroup-uid",
        namespace="team-a",
        pod_group_name="pg-training",
        owner_kind="JobSet",
        owner_name="training",
        owner_uid="jobset-uid",
        start_key=_ts(0),
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
                "current-pod": _placement("current-pod", "node-current", 300),
                "excluded-pod": _placement(
                    "excluded-pod", "node-excluded", 360, node_failure_reason="NodeLost"
                ),
            },
        )
    )

    assert context["state"] == "RUNNING"
    assert context["nodes"] == ["node-current"]
    assert "failed_node" not in context


def _retry_history(delay=5, owner_kind="JobSet"):
    owner = _owner(owner_kind, "training", "new-job" if owner_kind == "Job" else "owner-uid")
    boundary = 100 + delay
    job = _snapshot(
        "Job",
        "new-job",
        _ts(100),
        metadata={"name": "new-job", "ownerReferences": [owner] if owner_kind == "JobSet" else []},
        selectedMetadata={"labels": {_RESTART_LABEL: "1"}},
        status={"startTime": _ts(100)},
    )
    pending = _pod("new", "new", _ts(100), start_time=_ts(100), restart="1", job_uid="new-job")
    pending["status"] = {"phase": "Pending"}
    running = copy.deepcopy(pending)
    running.update(
        observedAt=_ts(boundary + 1),
        status={
            "phase": "Running",
            "startTime": _ts(boundary),
            "conditions": [_condition("PodScheduled", _ts(boundary))],
        },
    )
    return {
        "observations": [
            _podgroup(owner, _ts(0), status={"lastStartTimestamp": _ts(0)}),
            _pod(
                "old",
                "old",
                _ts(1),
                start_time=_ts(1),
                restart="0",
                job_uid=("old-job" if owner_kind == "JobSet" else "new-job"),
            ),
            job,
            pending,
            _snapshot(
                "Job",
                "old-job",
                _ts(100),
                metadata={"name": "old-job", "ownerReferences": [owner]},
                selectedMetadata={"labels": {_RESTART_LABEL: "0"}},
                status={"conditions": [_condition("Failed", _ts(100))]},
            ),
            running,
            _podgroup(owner, _ts(boundary), status={"lastStartTimestamp": _ts(boundary)}),
        ],
        "coverage": {
            "sources": [
                {
                    "sourceName": source,
                    "coverageStartedAt": _ts(-1),
                    "checkpointedAt": _ts(boundary + 100),
                }
                for source in ["pods", "jobs", "jobsets", "kai-podgroups"]
            ],
            "gaps": [],
        },
    }


@pytest.fixture
def retry_history():
    """Give each test an independent generation-0 failure and pending generation-1 retry."""
    return _retry_history()


def _retry_attempts(history):
    return attempts_from_podgroup_history(
        history,
        pod_group_name="pg-training",
        now=datetime.fromisoformat(history["observations"][-1]["observedAt"])
        + timedelta(seconds=100),
    )


@pytest.mark.parametrize("missing", ["gap", "late-watch", "missing-coverage", "missing-scheduled", "unowned-pods"])
def test_initial_retry_allocation_requires_continuous_evidence(retry_history, missing):
    coverage = retry_history["coverage"]
    if missing == "gap":
        coverage["gaps"] = [{"sourceName": "pods", "startedAt": _ts(99), "recoveredAt": _ts(105)}]
    elif missing == "late-watch":
        coverage["sources"][0]["coverageStartedAt"] = _ts(101)
    elif missing == "missing-coverage":
        retry_history["coverage"] = {}
    elif missing == "unowned-pods":
        for pod in (retry_history["observations"][3], retry_history["observations"][-2]):
            pod["metadata"]["ownerReferences"] = []
    else:
        retry_history["observations"][-2]["status"]["conditions"] = []
    assert len(_retry_attempts(retry_history)) == 3


@pytest.mark.parametrize("delay", [1, 3600], ids=["short", "long"])
def test_initial_retry_allocation_has_no_time_threshold(delay):
    history = _retry_history(delay)
    before = _retry_attempts({**history, "observations": history["observations"][:-2]})
    after = _retry_attempts(history)
    assert len(after) == 2
    assert [a.attempt_id for a in before] == [a.attempt_id for a in after]
    assert after[-1].job_uids == frozenset({"new-job"})
    assert after[-1].pod_uids == frozenset({"new"})


def test_combined_retry_evidence_matches_reconstruction(retry_history):
    combined = KaiExecutionContextAdapter._combine_attempt_evidence(
        [
            _WorkloadEvidence(
                namespace="default",
                pod_group_name="pg-training",
                observed_until=_dt(205),
                observations=retry_history["observations"],
                retained_coverage=[retry_history["coverage"]],
            )
        ]
    )
    assert [item.attempt for item in combined] == _retry_attempts(retry_history)


@pytest.mark.parametrize("earlier", ["allocation", "terminal"])
def test_earlier_execution_blocks_retry_merge(retry_history, earlier):
    if earlier == "allocation":
        record = copy.deepcopy(retry_history["observations"][-2])
        record["metadata"]["uid"] = "earlier"
        record.update(
            observedAt=_ts(100),
            status={"startTime": _ts(100), "conditions": [_condition("PodScheduled", _ts(100))]},
        )
    else:
        record = _snapshot(
            "Job",
            "failed-job",
            _ts(105),
            selectedMetadata={"labels": {_RESTART_LABEL: "1"}},
            status={"conditions": [_condition("Failed", _ts(105))]},
        )
    retry_history["observations"].insert(-2, record)
    assert len(_retry_attempts(retry_history)) == 3


@pytest.mark.parametrize("generations, expected_count", [(["0"], 2), (["0", "1"], 3)])
def test_unlabeled_terminal_uses_owned_pod_generations(retry_history, generations, expected_count):
    for generation in generations:
        pod = _pod(
            f"failed-{generation}",
            f"failed-{generation}",
            _ts(100),
            start_time=_ts(100),
            restart=generation,
        )
        pod["metadata"]["ownerReferences"] = [
            {"kind": "Job", "uid": "failed-job", "controller": True}
        ]
        retry_history["observations"].insert(-2, pod)
    retry_history["observations"].insert(
        -2,
        _snapshot(
            "Job", "failed-job", _ts(105), status={"conditions": [_condition("Failed", _ts(105))]}
        ),
    )
    assert len(_retry_attempts(retry_history)) == expected_count


def test_unlabeled_job_start_uses_owned_pod_generation(retry_history):
    job = retry_history["observations"][2]
    job.pop("selectedMetadata")
    job.update(observedAt=_ts(98), status={"startTime": _ts(98)})
    for pod in (retry_history["observations"][3], retry_history["observations"][-2]):
        pod["metadata"]["ownerReferences"] = [{"kind": "Job", "uid": "new-job", "controller": True}]
    attempts = _retry_attempts(retry_history)
    assert len(attempts) == 2
    assert attempts[-1].start_time == _dt(98)
    assert attempts[-1].job_uids == frozenset({"new-job"})
    assert attempts[-1].pod_uids == frozenset({"new"})


def test_boundary_terminal_closes_only_previous_attempt(retry_history):
    for reason in ["InitialReason", "UpdatedReason"]:
        retry_history["observations"].insert(
            -2,
            _snapshot(
                "JobSet",
                "owner-uid",
                _ts(105),
                status={"conditions": [_condition("Failed", _ts(105), reason=reason)]},
            ),
        )
    expected = _retry_attempts(retry_history)
    assert len(expected) == 3
    assert expected[-2].terminal["reason"] in {"InitialReason", "UpdatedReason"}
    assert expected[-1].terminal is None
    for seed in range(10):
        replay = copy.deepcopy(retry_history)
        replay["observations"] *= 2
        random.Random(seed).shuffle(replay["observations"])
        # Replay observation ordering must not change the evaluation clock.
        actual = attempts_from_podgroup_history(replay, pod_group_name="pg-training", now=_dt(205))
        assert actual == expected


def test_job_owner_does_not_coalesce_retry_allocation():
    assert len(_retry_attempts(_retry_history(owner_kind="Job"))) == 3


@pytest.mark.parametrize("owner_kind", ["JobSet", "Job"])
def test_later_allocation_remains_separate(owner_kind):
    history = _retry_history(owner_kind=owner_kind)
    before = _retry_attempts(history)
    allocation = copy.deepcopy(history["observations"][-1])
    allocation.update(observedAt=_ts(155), status={"lastStartTimestamp": _ts(155)})
    history["observations"].append(allocation)
    after = _retry_attempts(history)
    assert len(after) == len(before) + 1
    assert after[-1].attempt_id != before[-1].attempt_id


@pytest.mark.parametrize(
    "kind, spec", [("Pod", {"nodeName": "node-a"}), ("Job", {"activeDeadlineSeconds": 600})]
)
@pytest.mark.parametrize("retained_version", ["4", "5"])
@pytest.mark.parametrize("reverse", [False, True])
def test_live_spec_survives_equal_version_history(kind, spec, retained_version, reverse):
    live = _snapshot(
        kind,
        "resource-uid",
        _ts(0),
        metadata={"resourceVersion": "4"},
        spec=spec,
        status={"phase": "Running"},
    )
    # Actual daemon responses retain status and metadata, but omit spec.
    retained = _snapshot(
        kind,
        "resource-uid",
        _ts(120),
        metadata={"resourceVersion": retained_version},
        status={"phase": "Failed"},
    )
    snapshots = [retained, live] if reverse else [live, retained]
    result = _latest_resources_by_uid(snapshots)["resource-uid"]
    if retained_version == "4":
        assert result["spec"] == spec
        assert result["status"] == {"phase": "Running"}
    else:
        assert result["spec"] == spec
        assert result["status"] == {"phase": "Failed"}


@pytest.mark.parametrize(
    "containers",
    [
        {},
        {"metadata": None, "selectedMetadata": None, "status": None},
        {"metadata": {}, "selectedMetadata": {}, "status": {}},
        {
            "metadata": {"ownerReferences": None},
            "selectedMetadata": {"labels": None, "annotations": None},
            "status": {"conditions": None},
        },
    ],
)
@pytest.mark.parametrize("spec", [{}, {"spec": {"nodeName": "node-a"}}])
def test_snapshot_normalization_preserves_unknown_evidence_and_input(containers, spec):

    original = {"sourceTimestamp": None, **containers, **spec}
    before = copy.deepcopy(original)
    normalized = _normalize_snapshot(original)

    assert normalized["metadata"] == {"ownerReferences": []}
    assert normalized["selectedMetadata"] == {"labels": {}, "annotations": {}}
    assert normalized["status"] == {"conditions": []}
    assert "startTime" not in normalized["status"]
    assert normalized["sourceTimestamp"] is None
    assert ("spec" in normalized) == ("spec" in original)
    assert normalized.get("spec") == original.get("spec")
    normalized["status"]["startTime"] = "changed"
    normalized["metadata"]["uid"] = "changed"
    normalized["selectedMetadata"]["labels"] = {"changed": "1"}
    assert original == before


@pytest.mark.parametrize("nullable", [False, True])
def test_retry_reconstruction_accepts_missing_or_null_optional_containers(nullable):

    history = _retry_history()
    expected = _retry_attempts(history)
    replay = copy.deepcopy(history["observations"])
    for snapshot in replay:
        for container, fields in (
            ("metadata", ("ownerReferences",)),
            ("selectedMetadata", ("labels", "annotations")),
            ("status", ("conditions",)),
        ):
            mapping = snapshot.get(container) or {}
            for field in fields:
                if not mapping.get(field):
                    if nullable:
                        mapping[field] = None
                    else:
                        mapping.pop(field, None)
            if nullable and not mapping:
                snapshot[container] = None
    before = copy.deepcopy(replay)
    assert _retry_attempts({**history, "observations": replay}) == expected
    assert replay == before


@pytest.mark.parametrize("retirement", ["deleted", "unscheduled-successor"])
def test_retired_attempt_without_terminal_has_bounded_uncertain_end(retirement):
    owner = _owner("Job", "trainer", "job-uid")
    retired = (
        _podgroup(owner, _ts(1800), eventType="DELETED")
        if retirement == "deleted"
        else _podgroup(owner, _ts(1800), uid="successor")
    )
    records = [_podgroup(owner, _ts(0), status={"lastStartTimestamp": _ts(0)}), retired]
    history = {"observations": records}
    for now in (_dt(7200), _dt(14400)):
        [attempt] = attempts_from_podgroup_history(history, pod_group_name="pg-training", now=now)
        assert attempt.end_time == _dt(1800)
        assert not attempt.open
        assert not attempt.membership_resolved


def test_pod_scheduled_before_boundary_stays_in_earlier_attempt():
    owner = _owner("Job", "trainer", "job-uid")
    pod = _pod("delayed", "delayed", _ts(3602), start_time=_ts(3602))
    pod["status"]["conditions"] = [_condition("PodScheduled", _ts(3599))]
    attempts = _reconstruct(
        {
            "observations": [
                _podgroup(owner, _ts(0), status={"lastStartTimestamp": _ts(0)}),
                _podgroup(owner, _ts(3600), status={"lastStartTimestamp": _ts(3600)}),
                pod,
            ]
        }
    )
    assert [a.pod_uids for a in attempts] == [frozenset({"delayed"}), frozenset()]


@pytest.mark.parametrize(
    "references",
    [
        [],
        [{"kind": "Job", "uid": "job-uid", "controller": False}],
        [{"kind": "Job", "uid": "other-job", "controller": True}],
    ],
)
def test_historical_pods_require_matching_job_controller(references):
    owner = _owner("Job", "trainer", "job-uid")
    pod = _pod(
        "foreign", "foreign", _ts(60), start_time=_ts(60), node="foreign-node", reason="NodeLost"
    )
    pod["metadata"]["ownerReferences"] = references
    [evidence] = _retained_attempts(
        {"observations": [_podgroup(owner, _ts(0), status={"lastStartTimestamp": _ts(0)}), pod]}
    )
    assert not evidence.attempt.pod_uids
    context = _context_from_attempt_evidence(evidence)
    assert context["nodes"] == []
    assert context["state"] != "NODE_FAIL"
    assert "failed_node" not in context
