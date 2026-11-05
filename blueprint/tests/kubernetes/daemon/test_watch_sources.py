# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

from kubernetes import client

from attribution_blueprint.kubernetes.daemon.resolvers import normalize_resource
from attribution_blueprint.kubernetes.daemon.watch_sources import (
    selected_metadata,
    selected_status,
    source_timestamp,
)


def test_pod_status_retains_restart_and_container_execution_history():
    status = selected_status(
        "Pod",
        {
            "status": {
                "phase": "Running",
                "containerStatuses": [
                    {
                        "name": "trainer",
                        "restartCount": 1,
                        "state": {"running": {"started_at": "2026-10-14T10:02:00Z"}},
                        "lastState": {
                            "terminated": {
                                "exitCode": 137,
                                "reason": "Error",
                                "startedAt": "2026-10-14T10:00:00Z",
                                "finished_at": "2026-10-14T10:01:00Z",
                            }
                        },
                    }
                ],
            }
        },
    )

    container = status["containerStatuses"][0]
    assert container["restartCount"] == 1
    assert container["state"]["running"]["startedAt"] == "2026-10-14T10:02:00Z"
    assert container["lastState"]["terminated"]["exitCode"] == 137
    assert source_timestamp(status) == "2026-10-14T10:02:00Z"


def test_typed_main_and_init_container_ids_are_retained():
    timestamp = datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)
    terminated = client.V1ContainerStateTerminated(
        container_id="containerd://execution",
        exit_code=42,
        finished_at=timestamp,
        reason="Error",
        started_at=timestamp,
    )
    pod_status = client.V1PodStatus(
        phase="Running",
        container_statuses=[
            client.V1ContainerStatus(
                container_id="containerd://main",
                image="trainer:latest",
                image_id="sha256:main",
                last_state=client.V1ContainerState(terminated=terminated),
                name="trainer",
                ready=True,
                restart_count=1,
                state=client.V1ContainerState(running=client.V1ContainerStateRunning(started_at=timestamp)),
            )
        ],
        init_container_statuses=[
            client.V1ContainerStatus(
                container_id="containerd://init",
                image="init:latest",
                image_id="sha256:init",
                last_state=client.V1ContainerState(),
                name="init",
                ready=True,
                restart_count=0,
                state=client.V1ContainerState(terminated=terminated),
            )
        ],
    )

    resource = normalize_resource({"status": pod_status.to_dict()})
    status = selected_status("Pod", resource)

    assert status["containerStatuses"][0]["containerID"] == "containerd://main"
    assert (
        status["containerStatuses"][0]["lastState"]["terminated"]["containerID"]
        == "containerd://execution"
    )
    assert status["initContainerStatuses"][0]["containerID"] == "containerd://init"
    assert (
        status["initContainerStatuses"][0]["state"]["terminated"]["containerID"]
        == "containerd://execution"
    )


def test_pod_status_omits_absent_optional_container_status_lists():
    status = selected_status(
        "Pod",
        {
            "status": {
                "phase": "Pending",
                "container_statuses": None,
                "init_container_statuses": None,
                "ephemeral_container_statuses": None,
            }
        },
    )

    assert status == {"phase": "Pending"}


def test_conditions_are_canonicalized_and_omit_sdk_nulls():
    status = selected_status(
        "Pod",
        {
            "status": {
                "conditions": [
                    {
                        "type": "Ready",
                        "status": "False",
                        "reason": "ContainersNotReady",
                        "message": None,
                        "last_probe_time": None,
                        "last_transition_time": "2026-10-14T10:00:00Z",
                    }
                ]
            }
        },
    )

    assert status["conditions"] == [
        {
            "type": "Ready",
            "status": "False",
            "reason": "ContainersNotReady",
            "lastTransitionTime": "2026-10-14T10:00:00Z",
        }
    ]


def test_controller_and_scheduler_statuses_retain_attempt_boundary_signals():
    job = selected_status(
        "Job",
        {
            "status": {
                "active": 0,
                "failed": 1,
                "succeeded": 1,
                "completionTime": "2026-10-14T10:10:00Z",
                "conditions": [{"type": "Complete", "status": "True"}],
                "ignored": "not retained",
            }
        },
    )
    jobset = selected_status(
        "JobSet",
        {
            "status": {
                "restarts": 1,
                "replicatedJobsStatus": [{"name": "workers", "failed": 1, "succeeded": 4}],
            }
        },
    )
    workload = selected_status(
        "Workload",
        {
            "status": {
                "admission": {"clusterQueue": "gpu"},
                "conditions": [{"type": "Evicted", "status": "False"}],
                "requeueState": {"count": 1},
            }
        },
    )

    assert job["failed"] == 1
    assert job["succeeded"] == 1
    assert "ignored" not in job
    assert jobset["restarts"] == 1
    assert workload["requeueState"] == {"count": 1}


def test_podgroup_retains_allocation_and_scheduler_start_signal():
    status = selected_status(
        "PodGroup",
        {
            "metadata": {
                "annotations": {
                    "kai.scheduler/last-start-timestamp": "2026-10-14T10:00:00Z",
                }
            },
            "status": {
                "resourcesStatus": {
                    "requested": {"cpu": "1"},
                    "allocated": {"cpu": "1"},
                }
            },
        },
    )

    assert status == {
        "resourcesStatus": {
            "requested": {"cpu": "1"},
            "allocated": {"cpu": "1"},
        },
        "lastStartTimestamp": "2026-10-14T10:00:00Z",
    }


def test_attempt_and_scheduler_identity_metadata_is_allowlisted():
    metadata = selected_metadata(
        {
            "metadata": {
                "deletionTimestamp": "2026-10-14T10:05:00Z",
                "labels": {
                    "jobset.sigs.k8s.io/replicatedjob-name": "workers",
                    "jobset.sigs.k8s.io/job-index": "2",
                    "jobset.sigs.k8s.io/restart-attempt": "1",
                    "kueue.x-k8s.io/job-uid": "job-uid",
                    "unrelated.example/label": "discarded",
                },
                "annotations": {
                    "batch.kubernetes.io/job-completion-index": "3",
                    "kai.scheduler/top-owner-metadata": "uid: job-uid",
                    "kueue.x-k8s.io/workload": "workload-a",
                    "pod-group-name": "podgroup-a",
                    "unrelated.example/annotation": "discarded",
                },
            }
        }
    )

    assert metadata == {
        "deletionTimestamp": "2026-10-14T10:05:00Z",
        "labels": {
            "jobset.sigs.k8s.io/job-index": "2",
            "jobset.sigs.k8s.io/replicatedjob-name": "workers",
            "jobset.sigs.k8s.io/restart-attempt": "1",
            "kueue.x-k8s.io/job-uid": "job-uid",
        },
        "annotations": {
            "batch.kubernetes.io/job-completion-index": "3",
            "kai.scheduler/top-owner-metadata": "uid: job-uid",
            "kueue.x-k8s.io/workload": "workload-a",
            "pod-group-name": "podgroup-a",
        },
    }
