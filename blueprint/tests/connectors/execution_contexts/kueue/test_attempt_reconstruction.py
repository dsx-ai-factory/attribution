# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import MagicMock

from attribution_blueprint.connectors.execution_contexts.kueue.attempts import (
    attempts_from_workload_history,
)
from attribution_blueprint.connectors.execution_contexts.kueue.connector import (
    KueueExecutionContextAdapter,
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
from attribution_blueprint.orchestrator import AttributionOrchestrator

_RESTART_LABEL = "jobset.sigs.k8s.io/restart-attempt"


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
        contexts = adapter.discover_by_ids(["training-workload"])
        replayed = adapter.discover_by_ids(["training-workload"])
        assert live_lookup.call_count == 2
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
