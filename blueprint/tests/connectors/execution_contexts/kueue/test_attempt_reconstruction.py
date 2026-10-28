# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from attribution_blueprint.connectors.execution_contexts.kueue.attempts import (
    WorkloadAttempt,
    attempts_from_workload_history,
    live_attempt_identity,
)
from attribution_blueprint.connectors.execution_contexts.kueue.connector import (
    KueueExecutionContextAdapter,
    _parse_workload,
)
from attribution_blueprint.connectors.outputs.sqlite import SQLiteOutput
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
from attribution_blueprint.orchestrator import AttributionOrchestrator

_RESTART_LABEL = "jobset.sigs.k8s.io/restart-attempt"


def test_attempt_id_preserves_existing_canonical_hash():
    timestamp = datetime(2026, 10, 14, tzinfo=timezone.utc)
    attempt = WorkloadAttempt(
        workload_uid="workload-uid",
        namespace="team-a",
        workload_name="training",
        owner_kind="Job",
        admission_key="admission-1",
        restart_attempt="2",
        start_time=timestamp,
        end_time=timestamp,
        terminal=None,
        open=False,
        boundary_resolved=True,
        pod_uids=frozenset(),
    )

    assert attempt.attempt_id == "kueue:v1:workload-uid:249ca67a73b47645"


def _owner(kind: str, name: str, uid: str) -> OwnerReference:
    return OwnerReference(
        api_version=(
            "jobset.x-k8s.io/v1alpha2"
            if kind == "JobSet"
            else "batch/v1"
        ),
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
    restart_attempt: str | None = None,
    event_type: str = "MODIFIED",
) -> ResourceObservation:
    selected_metadata = (
        {"labels": {_RESTART_LABEL: restart_attempt}}
        if restart_attempt is not None
        else None
    )
    return ResourceObservation(
        cluster_id="cluster-a",
        api_version={
            "JobSet": "jobset.x-k8s.io/v1alpha2",
            "Job": "batch/v1",
            "Pod": "v1",
            "Workload": "kueue.x-k8s.io/v1beta1",
        }[kind],
        kind=kind,
        namespace="team-a",
        name=name,
        uid=uid,
        resource_version=resource_version,
        event_type=event_type,
        observed_at=observed_at,
        owner_references=owners,
        selected_metadata=selected_metadata,
        status=status,
    )


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


def _seed_jobset_retry_history(store: SQLitePlacementStore) -> None:
    jobset_owner = _owner("JobSet", "training", "jobset-uid")
    job0_owner = _owner("Job", "training-workers-0", "job-uid-0")
    job1_owner = _owner("Job", "training-workers-1", "job-uid-1")
    observations = [
        _observation(
            kind="JobSet",
            name="training",
            uid="jobset-uid",
            resource_version="1",
            observed_at="2026-10-14T09:59:00Z",
        ),
        _observation(
            kind="Workload",
            name="training-workload",
            uid="workload-uid",
            resource_version="2",
            observed_at="2026-10-14T10:00:00Z",
            owners=(jobset_owner,),
            status={
                "conditions": [
                    _condition("Admitted", "2026-10-14T10:00:00Z")
                ]
            },
        ),
        _observation(
            kind="Job",
            name="training-workers-0",
            uid="job-uid-0",
            resource_version="3",
            observed_at="2026-10-14T10:00:00Z",
            owners=(jobset_owner,),
            restart_attempt="0",
            status={
                "startTime": "2026-10-14T10:00:00Z",
                "conditions": [
                    _condition(
                        "Failed",
                        "2026-10-14T10:31:00Z",
                        reason="BackoffLimitExceeded",
                    )
                ],
            },
        ),
        _observation(
            kind="Pod",
            name="training-workers-0-pod",
            uid="pod-uid-0",
            resource_version="4",
            observed_at="2026-10-14T10:01:00Z",
            owners=(job0_owner,),
            restart_attempt="0",
            status={"startTime": "2026-10-14T10:01:00Z"},
        ),
        _observation(
            kind="JobSet",
            name="training",
            uid="jobset-uid",
            resource_version="4",
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
            name="training-workers-1",
            uid="job-uid-1",
            resource_version="5",
            observed_at="2026-10-14T10:31:00Z",
            owners=(jobset_owner,),
            restart_attempt="1",
            status={
                "startTime": "2026-10-14T10:31:00Z",
                "conditions": [
                    _condition("Complete", "2026-10-14T11:00:00Z")
                ],
            },
        ),
        _observation(
            kind="Pod",
            name="training-workers-1-pod",
            uid="pod-uid-1",
            resource_version="6",
            observed_at="2026-10-14T10:32:00Z",
            owners=(job1_owner,),
            restart_attempt="1",
            status={"startTime": "2026-10-14T10:32:00Z"},
        ),
        _observation(
            kind="Workload",
            name="training-workload",
            uid="workload-uid",
            resource_version="7",
            observed_at="2026-10-14T11:00:00Z",
            owners=(jobset_owner,),
            status={
                "conditions": [
                    _condition(
                        "Finished",
                        "2026-10-14T11:00:00Z",
                        reason="Succeeded",
                    )
                ]
            },
            event_type="DELETED",
        ),
    ]
    for observation in observations:
        _capture(store, observation)

    workload = ResourceIdentity(
        "kueue.x-k8s.io/v1beta1",
        "Workload",
        "team-a",
        "training-workload",
        "workload-uid",
    )
    for pod_name, pod_uid, node_name, observed_at in (
        (
            "training-workers-0-pod",
            "pod-uid-0",
            "node-a",
            "2026-10-14T10:01:00Z",
        ),
        (
            "training-workers-1-pod",
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
                associated_resources=(workload,),
            )
        )


def test_retained_jobset_retries_flow_as_distinct_execution_contexts(
    tmp_path,
    mocker,
):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    _seed_jobset_retry_history(store)
    # Reopen the same database to exercise daemon restart durability.
    store = SQLitePlacementStore(str(path))
    server = serve_api(
        HealthState(),
        store,
        0,
        "team-a",
        cluster_id="cluster-a",
    )
    try:
        client = KubernetesDaemonClient(
            f"http://127.0.0.1:{server.server_port}"
        )
        adapter = KueueExecutionContextAdapter(
            namespace="team-a",
            placement_records_enabled=True,
            placement_reader=client,
        )
        live_lookup = mocker.patch.object(
            adapter._kubernetes,
            "get_one",
            return_value=None,
        )
        live_list = mocker.patch.object(
            adapter._kubernetes,
            "list",
            return_value=[],
        )
        contexts = adapter.discover_by_ids(["training-workload"])
        replayed = adapter.discover_by_ids(["training-workload"])
        assert live_lookup.call_count == 2
        ranged = adapter.discover_by_time_range(
            datetime(2026, 10, 14, 9, 59, tzinfo=timezone.utc),
            datetime(2026, 10, 14, 11, 1, tzinfo=timezone.utc),
        )
        node_a_only = adapter.discover_by_time_range(
            datetime(2026, 10, 14, 9, 59, tzinfo=timezone.utc),
            datetime(2026, 10, 14, 11, 1, tzinfo=timezone.utc),
            nodelist=["node-a"],
        )
        outside = adapter.discover_by_time_range(
            datetime(2026, 10, 15, tzinfo=timezone.utc),
            datetime(2026, 10, 16, tzinfo=timezone.utc),
        )
        assert live_list.call_count == 3
        history = client.find_histories(
            "team-a",
            "Workload",
            name="training-workload",
        )[0]
        late_history = deepcopy(history)
        late_history["observations"].reverse()
        late_reconstruction = attempts_from_workload_history(
            late_history,
            now=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
            workload_name="training-workload",
        )
    finally:
        server.shutdown()
        server.server_close()

    assert len(contexts) == 2
    assert [context["attempt_id"] for context in ranged] == [
        context["attempt_id"]
        for context in contexts
    ]
    assert [context["attempt_id"] for context in node_a_only] == [
        contexts[0]["attempt_id"]
    ]
    assert outside == []
    assert [context["attempt_id"] for context in replayed] == [
        context["attempt_id"]
        for context in contexts
    ]
    assert [attempt.attempt_id for attempt in late_reconstruction] == [
        context["attempt_id"]
        for context in contexts
    ]
    assert contexts[0]["workload_uid"] == "workload-uid"
    assert contexts[0]["attempt_id"].startswith(
        "kueue:v1:workload-uid:"
    )
    assert contexts[0]["start_time"] == datetime(
        2026,
        10,
        14,
        10,
        0,
        tzinfo=timezone.utc,
    )
    assert contexts[0]["end_time"] == datetime(
        2026,
        10,
        14,
        10,
        31,
        tzinfo=timezone.utc,
    )
    assert contexts[0]["state"] == "FAILED"
    assert contexts[0]["nodes"] == ["node-a"]
    assert contexts[1]["start_time"] == datetime(
        2026,
        10,
        14,
        10,
        31,
        tzinfo=timezone.utc,
    )
    assert contexts[1]["end_time"] == datetime(
        2026,
        10,
        14,
        11,
        0,
        tzinfo=timezone.utc,
    )
    assert contexts[1]["state"] == "COMPLETED"
    assert contexts[1]["nodes"] == ["node-b"]
    assert all(context["history_complete"] is False for context in contexts)

    attribution = MagicMock()
    attribution.create_attributor.side_effect = (
        lambda info: f"attr-{info.workload.id}"
    )
    attribution.get_attribution.return_value = {
        "verdict": {"attribution_outcome": "Fail User"}
    }
    attribution.delete_attributor.return_value = None
    output = MagicMock()
    output.close.return_value = None
    pipeline = AttributionOrchestrator(
        execution_context_adapters=[adapter],
        log_connectors=[],
        metrics_connectors=[],
        attribution=attribution,
        outputs=[output],
    )
    mocker.patch.object(
        adapter,
        "discover_by_ids",
        return_value=contexts,
    )

    results = pipeline.analyze(ids=["training-workload"])

    assert len(results) == 2
    assert attribution.create_attributor.call_count == 2
    assert output.write.call_count == 2


def test_scheduler_readmission_creates_a_new_attempt():
    owner_reference = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "name": "training",
        "uid": "jobset-uid",
        "controller": True,
    }

    def workload_observation(
        resource_version: str,
        observed_at: str,
        conditions: list[dict],
    ) -> dict:
        return {
            "apiVersion": "kueue.x-k8s.io/v1beta1",
            "kind": "Workload",
            "metadata": {
                "namespace": "team-a",
                "name": "training-workload",
                "uid": "workload-uid",
                "resourceVersion": resource_version,
                "ownerReferences": [owner_reference],
            },
            "eventType": "MODIFIED",
            "observedAt": observed_at,
            "status": {"conditions": conditions},
        }

    history = {
        "namespace": "team-a",
        "observations": [
            workload_observation(
                "1",
                "2026-10-14T10:00:00Z",
                [_condition("Admitted", "2026-10-14T10:00:00Z")],
            ),
            workload_observation(
                "2",
                "2026-10-14T10:30:00Z",
                [
                    _condition(
                        "Evicted",
                        "2026-10-14T10:30:00Z",
                        reason="Preempted",
                    )
                ],
            ),
            workload_observation(
                "3",
                "2026-10-14T11:00:00Z",
                [_condition("Admitted", "2026-10-14T11:00:00Z")],
            ),
            workload_observation(
                "4",
                "2026-10-14T12:00:00Z",
                [
                    _condition(
                        "Finished",
                        "2026-10-14T12:00:00Z",
                        reason="Succeeded",
                    )
                ],
            ),
        ],
    }

    attempts = attempts_from_workload_history(
        history,
        now=datetime(2026, 10, 14, 13, 0, tzinfo=timezone.utc),
        workload_name="training-workload",
    )
    contexts = [
        KueueExecutionContextAdapter._context_from_history_attempt(
            attempt,
            [],
            {},
        )
        for attempt in attempts
    ]

    assert len(contexts) == 2
    assert contexts[0]["attempt_id"] != contexts[1]["attempt_id"]
    assert contexts[0]["state"] == "PREEMPTED"
    assert contexts[0]["start_time"] == datetime(
        2026,
        10,
        14,
        10,
        0,
        tzinfo=timezone.utc,
    )
    assert contexts[0]["end_time"] == datetime(
        2026,
        10,
        14,
        10,
        30,
        tzinfo=timezone.utc,
    )
    assert contexts[1]["state"] == "COMPLETED"

    assert contexts[1]["start_time"] == datetime(
        2026,
        10,
        14,
        11,
        0,
        tzinfo=timezone.utc,
    )

    coverage = {
        "sources": [
            {
                "sourceName": source_name,
                "coverageStartedAt": "2026-10-14T09:00:00Z",
                "checkpointedAt": "2026-10-14T13:00:00Z",
            }
            for source_name in (
                "kueue-workloads",
                "jobsets",
                "jobs",
                "pods",
            )
        ],
        "gaps": [],
        "disabledWatches": {},
    }
    complete_context = (
        KueueExecutionContextAdapter._context_from_history_attempt(
            attempts[0],
            [],
            coverage,
        )
    )
    assert complete_context["history_complete"] is True

    coverage["gaps"] = [
        {
            "sourceName": "pods",
            "startedAt": "2026-10-14T10:10:00Z",
            "recoveredAt": "2026-10-14T10:20:00Z",
        }
    ]
    partial_context = (
        KueueExecutionContextAdapter._context_from_history_attempt(
            attempts[0],
            [],
            coverage,
        )
    )
    assert partial_context["history_complete"] is False


def test_jobset_restart_generation_carries_across_readmission():
    owner_reference = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "name": "training",
        "uid": "jobset-uid",
        "controller": True,
    }

    def workload_observation(
        resource_version: str,
        observed_at: str,
        conditions: list[dict],
    ) -> dict:
        return {
            "kind": "Workload",
            "metadata": {
                "namespace": "team-a",
                "name": "training-workload",
                "uid": "workload-uid",
                "ownerReferences": [owner_reference],
            },
            "resourceVersion": resource_version,
            "observedAt": observed_at,
            "status": {"conditions": conditions},
        }

    history = {
        "observations": [
            workload_observation(
                "1",
                "2026-10-14T10:00:00Z",
                [_condition("Admitted", "2026-10-14T10:00:00Z")],
            ),
            {
                "kind": "Job",
                "metadata": {"uid": "job-uid-0"},
                "selectedMetadata": {
                    "labels": {_RESTART_LABEL: "0"},
                },
                "observedAt": "2026-10-14T10:01:00Z",
                "status": {"startTime": "2026-10-14T10:01:00Z"},
            },
            {
                "kind": "Pod",
                "metadata": {"uid": "restart-zero-pod"},
                "selectedMetadata": {
                    "labels": {_RESTART_LABEL: "0"},
                },
                "observedAt": "2026-10-14T10:02:00Z",
                "status": {"startTime": "2026-10-14T10:02:00Z"},
            },
            {
                "kind": "Job",
                "metadata": {"uid": "job-uid-1"},
                "selectedMetadata": {
                    "labels": {_RESTART_LABEL: "1"},
                },
                "observedAt": "2026-10-14T10:30:00Z",
                "status": {"startTime": "2026-10-14T10:30:00Z"},
            },
            {
                "kind": "Pod",
                "metadata": {"uid": "old-pod"},
                "selectedMetadata": {
                    "labels": {_RESTART_LABEL: "1"},
                },
                "observedAt": "2026-10-14T10:31:00Z",
                "status": {"startTime": "2026-10-14T10:31:00Z"},
            },
            workload_observation(
                "2",
                "2026-10-14T11:00:00Z",
                [_condition("Admitted", "2026-10-14T11:00:00Z")],
            ),
            {
                "kind": "Pod",
                "metadata": {"uid": "new-pod"},
                "selectedMetadata": {
                    "labels": {_RESTART_LABEL: "1"},
                },
                "observedAt": "2026-10-14T11:01:00Z",
                "status": {"startTime": "2026-10-14T11:01:00Z"},
            },
        ]
    }

    attempts = attempts_from_workload_history(
        history,
        now=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        workload_name="training-workload",
    )
    by_identity = {
        (attempt.admission_key, attempt.restart_attempt): attempt
        for attempt in attempts
    }

    assert len({attempt.attempt_id for attempt in attempts}) == len(attempts)
    assert set(by_identity) == {
        ("2026-10-14T10:00:00Z", "0"),
        ("2026-10-14T10:00:00Z", "1"),
        ("2026-10-14T11:00:00Z", "1"),
    }
    assert by_identity[
        ("2026-10-14T10:00:00Z", "0")
    ].pod_uids == frozenset({"restart-zero-pod"})
    assert by_identity[
        ("2026-10-14T10:00:00Z", "1")
    ].pod_uids == frozenset({"old-pod"})
    assert by_identity[
        ("2026-10-14T11:00:00Z", "1")
    ].pod_uids == frozenset({"new-pod"})


def test_recreated_workloads_bound_pods_and_owner_terminal_by_incarnation():
    job_owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": "trainer",
        "uid": "job-uid",
        "controller": True,
    }
    pod_owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": "trainer",
        "uid": "job-uid",
        "controller": True,
    }
    history = {
        "observations": [
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "training-workload",
                    "uid": "workload-old",
                    "ownerReferences": [job_owner],
                },
                "eventType": "ADDED",
                "observedAt": "2026-10-14T10:00:00Z",
                "status": {
                    "conditions": [
                        _condition("Admitted", "2026-10-14T10:00:00Z")
                    ]
                },
            },
            {
                "kind": "Pod",
                "metadata": {
                    "uid": "pod-old",
                    "ownerReferences": [pod_owner],
                },
                "eventType": "ADDED",
                "observedAt": "2026-10-14T10:01:00Z",
                "status": {"startTime": "2026-10-14T10:01:00Z"},
            },
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "training-workload",
                    "uid": "workload-old",
                    "ownerReferences": [job_owner],
                },
                "eventType": "DELETED",
                "observedAt": "2026-10-14T10:15:00Z",
                "sourceTimestamp": "2026-10-14T10:00:00Z",
                "selectedMetadata": {
                    "deletionTimestamp": "2026-10-14T10:15:00Z"
                },
                "status": {},
            },
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "training-workload",
                    "uid": "workload-new",
                    "ownerReferences": [job_owner],
                },
                "eventType": "ADDED",
                "observedAt": "2026-10-14T10:16:00Z",
                "status": {
                    "conditions": [
                        _condition("Admitted", "2026-10-14T10:10:00Z")
                    ]
                },
            },
            {
                "kind": "Pod",
                "metadata": {
                    "uid": "pod-new",
                    "ownerReferences": [pod_owner],
                },
                "eventType": "ADDED",
                "observedAt": "2026-10-14T10:16:00Z",
                "status": {"startTime": "2026-10-14T10:11:00Z"},
            },
            {
                "kind": "Job",
                "metadata": {"uid": "job-uid"},
                "eventType": "MODIFIED",
                "observedAt": "2026-10-14T12:00:00Z",
                "status": {
                    "conditions": [
                        _condition("Complete", "2026-10-14T12:00:00Z")
                    ]
                },
            },
        ]
    }

    attempts = attempts_from_workload_history(
        history,
        now=datetime(2026, 10, 14, 13, 0, tzinfo=timezone.utc),
        workload_name="training-workload",
    )

    assert len(attempts) == 2
    old, new = attempts
    assert old.workload_uid == "workload-old"
    assert old.end_time == datetime(2026, 10, 14, 10, 10, tzinfo=timezone.utc)
    assert old.terminal is None
    assert old.open is False
    assert old.pod_uids == frozenset({"pod-old"})
    assert new.workload_uid == "workload-new"
    assert new.terminal["kind"] == "Job"
    assert new.pod_uids == frozenset({"pod-new"})


def test_deletion_request_timestamp_does_not_reuse_generic_source_timestamp():
    history = {
        "observations": [
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "training-workload",
                    "uid": "workload-uid",
                },
                "eventType": "ADDED",
                "observedAt": "2026-10-14T12:00:00Z",
                "status": {
                    "conditions": [
                        _condition("Admitted", "2026-10-14T12:00:00Z")
                    ]
                },
            },
            {
                "kind": "Pod",
                "metadata": {"uid": "pod-uid"},
                "eventType": "ADDED",
                "observedAt": "2026-10-14T12:01:00Z",
                "status": {"startTime": "2026-10-14T12:01:00Z"},
            },
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "training-workload",
                    "uid": "workload-uid",
                },
                "eventType": "DELETED",
                "observedAt": "2026-10-14T12:10:00Z",
                # Generic sourceTimestamp can point at an earlier status
                # transition; only explicit deletionTimestamp is authoritative.
                "sourceTimestamp": "2026-10-14T12:00:00Z",
                "selectedMetadata": {
                    "deletionTimestamp": "2026-10-14T12:10:00Z"
                },
                "status": {},
            },
        ]
    }

    [attempt] = attempts_from_workload_history(
        history,
        now=datetime(2026, 10, 14, 13, 0, tzinfo=timezone.utc),
        workload_name="training-workload",
    )

    assert attempt.end_time == datetime(
        2026, 10, 14, 12, 10, tzinfo=timezone.utc
    )
    assert attempt.pod_uids == frozenset({"pod-uid"})


def test_replacement_without_source_boundary_preserves_uncertainty():
    now = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
    history = {
        "observations": [
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "training-workload",
                    "uid": "workload-old",
                    "ownerReferences": [
                        {
                            "kind": "Job",
                            "name": "trainer",
                            "uid": "job-uid",
                            "controller": True,
                        }
                    ],
                },
                "eventType": "ADDED",
                "observedAt": "2026-10-14T10:00:00Z",
                "status": {
                    "conditions": [
                        _condition("Admitted", "2026-10-14T10:00:00Z")
                    ]
                },
            },
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "training-workload",
                    "uid": "workload-new",
                },
                "eventType": "ADDED",
                "observedAt": "2026-10-14T10:15:00Z",
                "status": {"conditions": []},
            },
            {
                "kind": "Pod",
                "metadata": {"uid": "ambiguous-pod"},
                "eventType": "ADDED",
                "observedAt": "2026-10-14T10:16:00Z",
                "status": {"startTime": "2026-10-14T10:16:00Z"},
            },
            {
                "kind": "Job",
                "metadata": {"uid": "job-uid"},
                "eventType": "MODIFIED",
                "observedAt": "2026-10-14T10:20:00Z",
                "status": {
                    "conditions": [
                        _condition("Complete", "2026-10-14T10:20:00Z")
                    ]
                },
            },
        ]
    }

    [attempt] = attempts_from_workload_history(
        history,
        now=now,
        workload_name="training-workload",
    )

    assert attempt.end_time == now
    assert attempt.open is False
    assert attempt.boundary_resolved is False
    assert attempt.terminal is None
    assert attempt.pod_uids == frozenset()


def test_deadline_exceeded_outranks_generic_workload_failure():
    history = {
        "observations": [
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "training-workload",
                    "uid": "workload-uid",
                    "ownerReferences": [
                        {
                            "kind": "Job",
                            "name": "trainer",
                            "uid": "job-uid",
                            "controller": True,
                        }
                    ],
                },
                "observedAt": "2026-10-14T10:00:00Z",
                "status": {
                    "conditions": [
                        _condition("Admitted", "2026-10-14T10:00:00Z"),
                        _condition(
                            "Finished",
                            "2026-10-14T11:00:00Z",
                            reason="Failed",
                        ),
                    ]
                },
            },
            {
                "kind": "Job",
                "metadata": {"uid": "job-uid"},
                "observedAt": "2026-10-14T11:00:00Z",
                "status": {
                    "conditions": [
                        _condition(
                            "Failed",
                            "2026-10-14T11:00:00Z",
                            reason="DeadlineExceeded",
                        )
                    ]
                },
            },
        ]
    }

    [attempt] = attempts_from_workload_history(
        history,
        now=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        workload_name="training-workload",
    )
    context = KueueExecutionContextAdapter._context_from_history_attempt(
        attempt,
        [],
        {},
    )

    assert context["state"] == "TIMEOUT"


def test_child_job_completion_does_not_finish_jobset_attempt():
    history = {
        "observations": [
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "training-workload",
                    "uid": "workload-uid",
                    "ownerReferences": [
                        {
                            "kind": "JobSet",
                            "name": "training",
                            "uid": "jobset-uid",
                            "controller": True,
                        }
                    ],
                },
                "observedAt": "2026-10-14T10:00:00Z",
                "status": {
                    "conditions": [
                        _condition("Admitted", "2026-10-14T10:00:00Z")
                    ]
                },
            },
            {
                "kind": "Job",
                "metadata": {"uid": "child-job-uid"},
                "observedAt": "2026-10-14T10:10:00Z",
                "status": {
                    "conditions": [
                        _condition("Complete", "2026-10-14T10:10:00Z")
                    ]
                },
            },
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "training-workload",
                    "uid": "workload-uid",
                },
                "observedAt": "2026-10-14T11:00:00Z",
                "status": {
                    "conditions": [
                        _condition(
                            "Finished",
                            "2026-10-14T11:00:00Z",
                            reason="Succeeded",
                        )
                    ]
                },
            },
        ]
    }

    [attempt] = attempts_from_workload_history(
        history,
        now=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        workload_name="training-workload",
    )

    assert attempt.end_time == datetime(
        2026,
        10,
        14,
        11,
        0,
        tzinfo=timezone.utc,
    )
    assert attempt.terminal["kind"] == "Workload"


def test_never_admitted_history_does_not_invent_running_attempt():
    history = {
        "observations": [
            {
                "kind": "Workload",
                "metadata": {
                    "namespace": "team-a",
                    "name": "pending-workload",
                    "uid": "workload-uid",
                },
                "observedAt": "2026-10-14T10:00:00Z",
                "status": {"conditions": []},
            }
        ]
    }

    assert attempts_from_workload_history(
        history,
        now=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        workload_name="pending-workload",
    ) == []


def test_live_attempt_without_admission_boundary_uses_explicit_pre_execution_identity():
    workload = {
        "metadata": {
            "name": "pending-workload",
            "namespace": "team-a",
            "uid": "workload-uid",
            "creationTimestamp": "2026-10-14T10:00:00Z",
        },
        "status": {"conditions": []},
    }

    identity = live_attempt_identity(workload, [])

    assert identity is not None
    assert identity.attempt_id == "kueue:pre:v1:workload-uid"
    assert identity.boundary_resolved is False


def test_live_jobset_attempt_without_restart_resources_does_not_invent_zero():
    admitted = "2026-10-14T10:00:00Z"
    owner = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "name": "training",
        "uid": "jobset-uid",
        "controller": True,
    }
    workload = {
        "metadata": {
            "name": "training-workload",
            "namespace": "team-a",
            "uid": "workload-uid",
            "creationTimestamp": "2026-10-14T09:59:00Z",
            "ownerReferences": [owner],
        },
        "status": {
            "conditions": [
                _condition("Admitted", admitted),
            ]
        },
    }
    history = {
        "observations": [
            {
                "kind": "Workload",
                "metadata": workload["metadata"],
                "observedAt": admitted,
                "status": workload["status"],
            },
            {
                "kind": "Job",
                "metadata": {
                    "uid": "job-uid-1",
                    "ownerReferences": [owner],
                },
                "selectedMetadata": {
                    "labels": {_RESTART_LABEL: "1"},
                },
                "observedAt": "2026-10-14T10:30:00Z",
                "status": {"startTime": "2026-10-14T10:30:00Z"},
            },
        ]
    }

    live_with_resources = live_attempt_identity(
        workload,
        [
            {
                "metadata": {
                    "labels": {_RESTART_LABEL: "1"},
                },
                "status": {"startTime": "2026-10-14T10:30:00Z"},
            }
        ],
    )
    assert live_with_resources is not None
    assert live_attempt_identity(workload, []) is None
    retained = next(
        attempt
        for attempt in attempts_from_workload_history(
            history,
            workload_name="training-workload",
            now=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        )
        if attempt.restart_attempt == "1"
    )
    assert retained.restart_attempt == "1"
    assert live_with_resources.attempt_id == retained.attempt_id


def test_live_then_retained_discovery_writes_one_execution_context(tmp_path):
    admitted = "2026-10-14T10:00:00Z"
    finished = "2026-10-14T11:00:00Z"
    workload = {
        "metadata": {
            "name": "training-workload",
            "namespace": "team-a",
            "uid": "workload-uid",
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": "training",
                    "uid": "job-uid",
                    "controller": True,
                }
            ],
        },
        "status": {
            "conditions": [
                _condition("Admitted", admitted),
                _condition("Finished", finished, reason="Succeeded"),
            ]
        },
    }
    pod = {
        "metadata": {
            "name": "training-pod",
            "namespace": "team-a",
            "uid": "pod-uid",
            "creationTimestamp": "2026-10-14T10:01:00Z",
            "labels": {
                "batch.kubernetes.io/controller-uid": "job-uid",
            },
        },
        "spec": {"nodeName": "node-a"},
        "status": {"startTime": "2026-10-14T10:01:00Z"},
    }
    live = _parse_workload(
        workload,
        [pod],
        {},
        datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
    )
    assert live is not None

    history = {
        "observations": [
            {
                "kind": "Workload",
                "metadata": workload["metadata"],
                "observedAt": admitted,
                "status": workload["status"],
            }
        ]
    }
    [attempt] = attempts_from_workload_history(
        history,
        workload_name="training-workload",
        now=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
    )
    retained = KueueExecutionContextAdapter._context_from_history_attempt(
        attempt,
        [],
        {},
    )
    assert live["attempt_id"] == retained["attempt_id"]

    db = tmp_path / "attributions.db"
    result = {"verdict": {"attribution_outcome": "Pass"}}
    live_output = SQLiteOutput(str(db))
    live_output.write(live, result)
    live_output.close()
    retained_output = SQLiteOutput(str(db))
    retained_output.write(retained, result)
    retained_output.close()

    with sqlite3.connect(db) as connection:
        context_count = connection.execute(
            "SELECT COUNT(*) FROM execution_context"
        ).fetchone()[0]
        run_count = connection.execute(
            "SELECT COUNT(*) FROM attribution_run"
        ).fetchone()[0]

    assert context_count == 1
    assert run_count == 2


def test_live_state_replaces_open_retained_attempt():
    start = datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)
    retained = {
        "workload_uid": "workload-uid",
        "state": "RUNNING",
        "id": "retained-attempt",
        "attempt_id": "retained-attempt",
        "start_time": start,
        "end_time": start,
        "kueue_state": "RUNNING",
        "nodes": ["node-b"],
        "pod_names": ["retry-pod"],
        "history_complete": False,
    }
    live = {
        "workload_uid": "workload-uid",
        "state": "COMPLETED",
        "id": "workload-uid",
        "attempt_id": "workload-uid",
        "start_time": start,
        "end_time": datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
        "kueue_state": "FINISHED_SUCCEEDED",
        "nodes": ["node-a", "node-b"],
        "pod_names": ["first-pod", "retry-pod"],
    }

    [reconciled] = KueueExecutionContextAdapter._reconcile_live_contexts(
        [retained],
        [live],
    )

    assert reconciled["state"] == "COMPLETED"
    assert reconciled["kueue_state"] == "FINISHED_SUCCEEDED"
    assert reconciled["attempt_id"] == "retained-attempt"
    assert reconciled["nodes"] == ["node-b"]
    assert reconciled["pod_names"] == ["retry-pod"]
    assert reconciled["history_complete"] is False


def test_retained_time_range_requires_namespace():
    reader = MagicMock()
    reader.find_histories = MagicMock(return_value=[])
    adapter = KueueExecutionContextAdapter(
        placement_records_enabled=True,
        placement_reader=reader,
    )

    with pytest.raises(
        ValueError,
        match="retained-history time-range discovery requires",
    ):
        adapter.discover_by_time_range(
            datetime(2026, 10, 14, tzinfo=timezone.utc),
            datetime(2026, 10, 15, tzinfo=timezone.utc),
        )
