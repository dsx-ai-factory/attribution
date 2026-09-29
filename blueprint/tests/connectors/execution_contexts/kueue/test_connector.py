# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from attribution_blueprint.connectors.execution_contexts.kueue import (
    KueueExecutionContextAdapter,
)

FIXTURES = Path(__file__).parent.parent.parent.parent / "fixtures"
WORKLOADS = json.loads((FIXTURES / "kueue_workloads.json").read_text())
PODS = json.loads((FIXTURES / "kueue_pods.json").read_text())
JOBS = json.loads((FIXTURES / "kueue_jobs.json").read_text())


def _kubectl_router(
    workloads_payload=WORKLOADS,
    pods_payload=PODS,
    jobs_payload=JOBS,
    events_payload=None,
):
    """Return a `subprocess.run` side_effect that dispatches by kubectl args.

    The adapter issues these shapes of kubectl call:
      - `kubectl get workloads [-A | -n NS]`            -> list
      - `kubectl get workload <name> -n <ns>`           -> single workload
      - `kubectl get pods -l <owner-selector> -A|-n NS` -> list
      - `kubectl get job <name> -n <ns>`                -> single Job
      - `kubectl get jobs -l jobset.sigs.k8s.io/jobset-name=NAME -n <ns>` -> JobSet children
      - `kubectl get events --field-selector involvedObject.uid=UID,involvedObject.kind=Job,reason=SuccessfulCreate -n <ns>` -> list
      - `kubectl get events --field-selector involvedObject.name=POD,involvedObject.kind=Pod,reason=Scheduled -n <ns>` -> list

    `events_payload` simulates the API server: `_matches_field_selector`
    below filters it by the `--field-selector` clauses on each call, the
    same way a real apiserver would, so tests exercise the actual selector
    strings the adapter sends rather than relying on the adapter's own
    Python-side filtering to compensate for an untargeted mock.
    """
    if events_payload is None:
        events_payload = {"apiVersion": "v1", "kind": "List", "items": []}

    def _by_uid(uid: str) -> dict:
        items = [
            p
            for p in pods_payload["items"]
            if (p.get("metadata") or {}).get("labels", {}).get("batch.kubernetes.io/controller-uid") == uid
        ]
        return {"apiVersion": "v1", "kind": "List", "items": items}

    def _by_name(items: list[dict], name: str, namespace: str | None) -> dict | None:
        for item in items:
            meta = item.get("metadata") or {}
            if meta.get("name") != name:
                continue
            if namespace and meta.get("namespace") != namespace:
                continue
            return item
        return None

    def _by_label(items: list[dict], selector: str | None, namespace: str | None) -> dict:
        if not selector or "=" not in selector:
            return {"apiVersion": "v1", "kind": "List", "items": []}
        key, value = selector.split("=", 1)
        matches = []
        for item in items:
            metadata = item.get("metadata") or {}
            if namespace and metadata.get("namespace") != namespace:
                continue
            labels = metadata.get("labels") or {}
            annotations = metadata.get("annotations") or {}
            if labels.get(key) == value or annotations.get(key) == value:
                matches.append(item)
        return {"apiVersion": "v1", "kind": "List", "items": matches}

    def _matches_field_selector(event: dict, selector: str | None) -> bool:
        """Emulate apiserver-side `--field-selector` filtering for Events."""
        if not selector:
            return True
        involved = event.get("involvedObject") or {}
        field_values = {
            "reason": event.get("reason"),
            "involvedObject.uid": involved.get("uid"),
            "involvedObject.kind": involved.get("kind"),
            "involvedObject.name": involved.get("name"),
            "involvedObject.namespace": involved.get("namespace"),
        }
        for clause in selector.split(","):
            if "=" not in clause:
                continue
            key, value = clause.split("=", 1)
            if field_values.get(key) != value:
                return False
        return True

    def _side_effect(cmd, **kwargs):
        # cmd is the full subprocess argv. Strip kubectl + flags, isolate the
        # kubectl sub-command vs. positional args.
        args = list(cmd)
        # The first non-flag positional is the kubectl verb.
        positional: list[str] = []
        ns: str | None = None
        label_selector: str | None = None
        field_selector: str | None = None
        all_namespaces = False
        i = 0
        while i < len(args):
            tok = args[i]
            if tok in ("--kubeconfig", "--context"):
                i += 2
                continue
            if tok == "-o":
                i += 2
                continue
            if tok == "-n":
                ns = args[i + 1]
                i += 2
                continue
            if tok == "-A":
                all_namespaces = True
                i += 1
                continue
            if tok == "-l":
                label_selector = args[i + 1]
                i += 2
                continue
            if tok == "--field-selector":
                field_selector = args[i + 1]
                i += 2
                continue
            positional.append(tok)
            i += 1

        # positional now starts with the kubectl binary path. Drop it.
        positional = positional[1:]
        if not positional or positional[0] != "get":
            stdout = "{}"
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")
        kind = positional[1] if len(positional) > 1 else ""
        name = positional[2] if len(positional) > 2 else None

        if kind == "workloads":
            stdout = json.dumps(workloads_payload)
        elif kind == "workload":
            obj = _by_name(workloads_payload["items"], name, ns)
            if obj is None:
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="NotFound")
            stdout = json.dumps(obj)
        elif kind == "pods":
            if label_selector and label_selector.startswith("batch.kubernetes.io/controller-uid="):
                stdout = json.dumps(_by_uid(label_selector.split("=", 1)[1]))
            else:
                stdout = json.dumps(_by_label(pods_payload["items"], label_selector, ns))
        elif kind == "job":
            obj = _by_name(jobs_payload["items"], name, ns)
            if obj is None:
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="NotFound")
            stdout = json.dumps(obj)
        elif kind == "jobs":
            stdout = json.dumps(_by_label(jobs_payload["items"], label_selector, ns))
        elif kind == "events":
            matches = [
                event
                for event in events_payload.get("items", [])
                if (not ns or (event.get("involvedObject") or {}).get("namespace", ns) == ns)
                and _matches_field_selector(event, field_selector)
            ]
            stdout = json.dumps({"apiVersion": "v1", "kind": "List", "items": matches})
        else:
            stdout = json.dumps({"items": []})

        _ = all_namespaces  # silence unused-warning; behavior is identical here
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    return _side_effect


def _wire_kubectl(mocker, **kwargs):
    return mocker.patch("subprocess.run", side_effect=_kubectl_router(**kwargs))


def test_discover_returns_empty_with_no_filter(mocker):
    """Like the Slurm adapter, no-arg discover() returns [] rather than scanning all workloads."""
    run_mock = _wire_kubectl(mocker)
    assert KueueExecutionContextAdapter().discover() == []
    assert run_mock.call_count == 0


def test_discover_by_ids_normalizes_finished_workload(mocker):
    _wire_kubectl(mocker)
    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-train-mnist-abc12"])
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["id"] == "uid-aaaa-1001"
    assert ctx["workload_uid"] == "uid-aaaa-1001"
    assert ctx["attempt_id"] == "uid-aaaa-1001"
    assert ctx["workload_type"] == "kueue"
    # BP-internal identity remains independent of the service wire field.
    assert ctx["scheduler"] == "kueue"
    assert ctx["job_id"] == "team-alpha/job-train-mnist-abc12"
    assert ctx["namespace"] == "team-alpha"
    assert ctx["name"] == "job-train-mnist-abc12"
    assert ctx["state"] == "COMPLETED"
    assert ctx["kueue_state"] == "FINISHED_SUCCEEDED"
    assert ctx["nodes"] == ["node005"]
    assert ctx["pod_nodes"] == {"train-mnist-abc12-0": "node005"}
    assert ctx["start_time"] == datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert ctx["end_time"] == datetime(2026, 5, 1, 12, 30, 0, tzinfo=timezone.utc)
    assert ctx["user_name"] == "alice"
    assert ctx["queue_name"] == "research-queue"
    assert ctx["priority_class"] == 100
    assert ctx["time_limit"] == 60
    assert "failed_node" not in ctx
    # kueue_phase was removed - it had no information beyond `state`.
    assert "kueue_phase" not in ctx


def test_finished_success_takes_precedence_over_retried_node_loss(mocker):
    pods = deepcopy(PODS)
    pod = next(
        item
        for item in pods["items"]
        if item["metadata"]["labels"].get("batch.kubernetes.io/controller-uid") == "owner-1001"
    )
    pod["status"] = {"phase": "Failed", "reason": "NodeLost"}
    _wire_kubectl(mocker, pods_payload=pods)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-train-mnist-abc12"])[0]

    assert ctx["state"] == "COMPLETED"
    assert ctx["kueue_state"] == "FINISHED_SUCCEEDED"
    assert ctx["failed_node"] == "node005"


def test_discover_by_ids_carries_kueue_gpu_count(mocker):
    workloads = deepcopy(WORKLOADS)
    workload = workloads["items"][0]
    pod_set = workload["spec"]["podSets"][0]
    pod_set["count"] = 2
    pod_set["template"]["spec"]["containers"][0]["resources"] = {"requests": {"cpu": "4", "nvidia.com/gpu": "3"}}
    _wire_kubectl(mocker, workloads_payload=workloads)

    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-train-mnist-abc12"])

    assert contexts[0]["gpu_count"] == 6


def test_discover_by_ids_carries_zero_gpu_count_for_cpu_only_kueue_workload(mocker):
    workloads = deepcopy(WORKLOADS)
    workload = workloads["items"][0]
    container = workload["spec"]["podSets"][0]["template"]["spec"]["containers"][0]
    container["resources"] = {"requests": {"cpu": "4"}}
    _wire_kubectl(mocker, workloads_payload=workloads)

    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-train-mnist-abc12"])

    assert contexts[0]["gpu_count"] == 0


def test_active_deadline_seconds_rounds_up_to_service_minutes(mocker):
    jobs = deepcopy(JOBS)
    jobs["items"][0]["spec"]["activeDeadlineSeconds"] = 61
    _wire_kubectl(mocker, jobs_payload=jobs)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-train-mnist-abc12"])[0]

    assert ctx["time_limit"] == 2


def test_failed_workload_surfaces_first_nonzero_exit_node(mocker):
    _wire_kubectl(mocker)
    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-llama-7b-pretrain-xyz77"])
    ctx = contexts[0]
    assert ctx["state"] == "FAILED"
    assert ctx["kueue_state"] == "FINISHED_FAILED"
    assert ctx["nodes"] == ["node001", "node002", "node003"]
    # Only node002's container reported exitCode=137, even though all three
    # pods are in Failed phase. failed_node names the node that actually
    # produced the bad exit, not the whole nodelist.
    assert ctx["failed_node"] == "node002"
    assert ctx["time_limit"] == 120


def test_failed_workload_deadline_exceeded_normalizes_to_timeout(mocker):
    jobs = deepcopy(JOBS)
    jobs["items"][1]["status"] = {
        "conditions": [
            {
                "type": "Failed",
                "status": "True",
                "reason": "DeadlineExceeded",
                "lastTransitionTime": "2026-05-01T13:30:00Z",
            }
        ]
    }
    _wire_kubectl(mocker, jobs_payload=jobs)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-llama-7b-pretrain-xyz77"])[0]

    assert ctx["state"] == "TIMEOUT"
    assert ctx["kueue_state"] == "FINISHED_FAILED"


def test_reused_job_name_with_different_uid_is_not_used_as_owner(mocker):
    """Kubernetes names are reusable after deletion. If the Job the Workload
    pointed to (`owner-1002`) was deleted and a new, unrelated Job with the
    same name `llama-7b-pretrain` but a different uid now exists, that new
    Job must never be treated as this Workload's owner — even though a
    namespace/name lookup finds it."""
    jobs = deepcopy(JOBS)
    job = jobs["items"][1]
    assert job["metadata"]["name"] == "llama-7b-pretrain"
    job["metadata"]["uid"] = "owner-1002-REPLACEMENT"
    job["spec"] = {"activeDeadlineSeconds": 60}
    job["status"] = {
        "conditions": [
            {
                "type": "Failed",
                "status": "True",
                "reason": "DeadlineExceeded",
                "lastTransitionTime": "2026-05-01T13:30:00Z",
            }
        ]
    }
    _wire_kubectl(mocker, jobs_payload=jobs)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-llama-7b-pretrain-xyz77"])[0]

    # The replacement Job's DeadlineExceeded/time_limit must not leak in.
    assert ctx["state"] == "FAILED"
    assert ctx["kueue_state"] == "FINISHED_FAILED"
    assert "time_limit" not in ctx


def test_failed_jobset_child_deadline_exceeded_normalizes_to_timeout(mocker):
    workload = {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "Workload",
        "metadata": {
            "name": "jobset-timeout-abc12",
            "namespace": "team-alpha",
            "uid": "workload-jobset-timeout",
            "creationTimestamp": "2026-05-01T12:00:00Z",
            "ownerReferences": [{"kind": "JobSet", "name": "training", "uid": "jobset-uid"}],
        },
        "spec": {"podSets": []},
        "status": {
            "conditions": [
                {
                    "type": "Admitted",
                    "status": "True",
                    "lastTransitionTime": "2026-05-01T12:01:00Z",
                },
                {
                    "type": "Finished",
                    "status": "True",
                    "reason": "Failed",
                    "lastTransitionTime": "2026-05-01T12:02:00Z",
                },
            ]
        },
    }
    child_job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "training-workers-0",
            "namespace": "team-alpha",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "training"},
            "ownerReferences": [{"kind": "JobSet", "name": "training", "uid": "jobset-uid"}],
        },
        "spec": {"activeDeadlineSeconds": 60},
        "status": {
            "conditions": [
                {
                    "type": "Failed",
                    "status": "True",
                    "reason": "DeadlineExceeded",
                    "lastTransitionTime": "2026-05-01T12:02:00Z",
                }
            ]
        },
    }
    _wire_kubectl(
        mocker,
        workloads_payload={"items": [workload]},
        pods_payload={"items": []},
        jobs_payload={"items": [child_job]},
    )

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/jobset-timeout-abc12"])[0]

    assert ctx["state"] == "TIMEOUT"
    assert ctx["kueue_state"] == "FINISHED_FAILED"
    assert ctx["time_limit"] == 1


def test_reused_jobset_name_with_different_uid_is_not_used_as_owner(mocker):
    """The child Job is selected by `jobset-name=training`, a reusable name.
    If it is actually owned by a *replacement* JobSet with a different uid
    than the one on the Workload's ownerReference, it must not be treated
    as this Workload's child Job."""
    workload = {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "Workload",
        "metadata": {
            "name": "jobset-timeout-abc12",
            "namespace": "team-alpha",
            "uid": "workload-jobset-timeout",
            "creationTimestamp": "2026-05-01T12:00:00Z",
            "ownerReferences": [{"kind": "JobSet", "name": "training", "uid": "jobset-uid-OLD"}],
        },
        "spec": {"podSets": []},
        "status": {
            "conditions": [
                {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-05-01T12:01:00Z"},
                {
                    "type": "Finished",
                    "status": "True",
                    "reason": "Failed",
                    "lastTransitionTime": "2026-05-01T12:02:00Z",
                },
            ]
        },
    }
    # Same jobset-name label, but owned by a different (replacement) JobSet uid.
    replacement_child_job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "training-workers-0",
            "namespace": "team-alpha",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "training"},
            "ownerReferences": [{"kind": "JobSet", "name": "training", "uid": "jobset-uid-NEW"}],
        },
        "spec": {"activeDeadlineSeconds": 60},
        "status": {
            "conditions": [
                {
                    "type": "Failed",
                    "status": "True",
                    "reason": "DeadlineExceeded",
                    "lastTransitionTime": "2026-05-01T12:02:00Z",
                }
            ]
        },
    }
    _wire_kubectl(
        mocker,
        workloads_payload={"items": [workload]},
        pods_payload={"items": []},
        jobs_payload={"items": [replacement_child_job]},
    )

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/jobset-timeout-abc12"])[0]

    # The replacement JobSet's child evidence (DeadlineExceeded, time_limit)
    # must not leak into this Workload's attribution.
    assert ctx["state"] == "FAILED"
    assert ctx["kueue_state"] == "FINISHED_FAILED"
    assert "time_limit" not in ctx


def _jobset_workload_and_children(*, first_reason: str, second_reason: str) -> tuple[dict, list[dict]]:
    """Build a JobSet Workload with two children failing at different times.

    `first_reason` is on the child whose `Failed` condition transitions
    earliest; `second_reason` is on the child that fails afterward.
    """
    workload = {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "Workload",
        "metadata": {
            "name": "jobset-mixed-abc12",
            "namespace": "team-alpha",
            "uid": "workload-jobset-mixed",
            "creationTimestamp": "2026-05-01T12:00:00Z",
            "ownerReferences": [{"kind": "JobSet", "name": "training", "uid": "jobset-uid"}],
        },
        "spec": {"podSets": []},
        "status": {
            "conditions": [
                {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-05-01T12:01:00Z"},
                {
                    "type": "Finished",
                    "status": "True",
                    "reason": "Failed",
                    "lastTransitionTime": "2026-05-01T12:05:00Z",
                },
            ]
        },
    }

    def _child(name: str, reason: str, transition_time: str) -> dict:
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "namespace": "team-alpha",
                "labels": {"jobset.sigs.k8s.io/jobset-name": "training"},
                "ownerReferences": [{"kind": "JobSet", "name": "training", "uid": "jobset-uid"}],
            },
            "spec": {},
            "status": {
                "conditions": [
                    {"type": "Failed", "status": "True", "reason": reason, "lastTransitionTime": transition_time}
                ]
            },
        }

    children = [
        _child("training-workers-0", first_reason, "2026-05-01T12:02:00Z"),
        _child("training-workers-1", second_reason, "2026-05-01T12:04:00Z"),
    ]
    return workload, children


def test_jobset_mixed_child_failures_use_earliest_reason_not_deadline_exceeded(mocker):
    """A later DeadlineExceeded child must not override an earlier, different failure.

    Mirrors JobSet's own controller convention (findFirstFailedJob /
    findJobFailureTime): the earliest failing child is authoritative, with
    no special-casing of any particular reason.
    """
    workload, children = _jobset_workload_and_children(
        first_reason="BackoffLimitExceeded", second_reason="DeadlineExceeded"
    )
    _wire_kubectl(
        mocker,
        workloads_payload={"items": [workload]},
        pods_payload={"items": []},
        jobs_payload={"items": children},
    )

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/jobset-mixed-abc12"])[0]

    assert ctx["state"] == "FAILED"
    assert ctx["kueue_state"] == "FINISHED_FAILED"


def test_jobset_earliest_deadline_exceeded_wins_over_later_other_reason(mocker):
    """When the earliest-failing child is the DeadlineExceeded one, it wins --
    because it failed first, not because of any reason-specific preference."""
    workload, children = _jobset_workload_and_children(
        first_reason="DeadlineExceeded", second_reason="BackoffLimitExceeded"
    )
    _wire_kubectl(
        mocker,
        workloads_payload={"items": [workload]},
        pods_payload={"items": []},
        jobs_payload={"items": children},
    )

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/jobset-mixed-abc12"])[0]

    assert ctx["state"] == "TIMEOUT"
    assert ctx["kueue_state"] == "FINISHED_FAILED"


def test_failed_workload_node_lost_normalizes_to_node_fail(mocker):
    pods = deepcopy(PODS)
    failed_pod = pods["items"][1]
    failed_pod["status"] = {"phase": "Failed", "reason": "NodeLost"}
    _wire_kubectl(mocker, pods_payload=pods)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-llama-7b-pretrain-xyz77"])[0]

    assert ctx["state"] == "NODE_FAIL"
    assert ctx["kueue_state"] == "FINISHED_FAILED"
    assert ctx["failed_node"] == "node001"


def test_disruption_target_pod_normalizes_admitted_workload_to_node_fail(mocker):
    workloads = deepcopy(WORKLOADS)
    workload = next(item for item in workloads["items"] if item["metadata"]["name"] == "job-still-running-qq01")
    workload["status"]["conditions"] = [
        condition for condition in workload["status"]["conditions"] if condition["type"] != "PodsReady"
    ]
    pods = deepcopy(PODS)
    pod = next(
        item
        for item in pods["items"]
        if item["metadata"]["labels"].get("batch.kubernetes.io/controller-uid") == "owner-1004"
    )
    pod["status"] = {
        "phase": "Failed",
        "conditions": [
            {
                "type": "DisruptionTarget",
                "status": "True",
                "reason": "DeletionByPodGC",
                "message": "PodGC: node no longer exists",
                "lastTransitionTime": "2026-05-02T08:05:00Z",
            }
        ],
    }
    _wire_kubectl(mocker, workloads_payload=workloads, pods_payload=pods)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-still-running-qq01"])[0]

    assert ctx["state"] == "NODE_FAIL"
    assert ctx["kueue_state"] == "ADMITTED"
    assert ctx["failed_node"] == "node020"


def test_taint_manager_disruption_target_does_not_imply_node_failure(mocker):
    """DeletionByTaintManager only proves the node was unreachable/NotReady
    long enough for the NoExecute toleration window to expire - it does
    NOT prove the Node API object was ever deleted, unlike DeletionByPodGC.
    Treating it as node-loss evidence risks misclassifying transient
    unreachability or a routine eviction as a hardware failure, so it must
    not normalize the Workload to NODE_FAIL on its own.
    """
    workloads = deepcopy(WORKLOADS)
    workload = next(item for item in workloads["items"] if item["metadata"]["name"] == "job-still-running-qq01")
    workload["status"]["conditions"] = [
        condition for condition in workload["status"]["conditions"] if condition["type"] != "PodsReady"
    ]
    pods = deepcopy(PODS)
    pod = next(
        item
        for item in pods["items"]
        if item["metadata"]["labels"].get("batch.kubernetes.io/controller-uid") == "owner-1004"
    )
    pod["status"] = {
        "phase": "Running",
        "conditions": [
            {
                "type": "DisruptionTarget",
                "status": "True",
                "reason": "DeletionByTaintManager",
                "message": "Taint manager: deleting due to NoExecute taint",
                "lastTransitionTime": "2026-05-02T08:05:00Z",
            }
        ],
    }
    _wire_kubectl(mocker, workloads_payload=workloads, pods_payload=pods)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-still-running-qq01"])[0]

    assert ctx["state"] == "ADMITTED"
    assert ctx["kueue_state"] == "ADMITTED"
    assert "failed_node" not in ctx


def test_evicted_workload_state_and_end_time(mocker):
    _wire_kubectl(mocker)
    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-beta/job-hpl-bench-pp99"])
    ctx = contexts[0]
    # Evicted=True at 14:05 with reason=Preempted. No Finished condition,
    # so end_time falls back to the Evicted transition time.
    assert ctx["state"] == "PREEMPTED"
    assert ctx["kueue_state"] == "EVICTED_PREEMPTED"
    assert ctx["end_time"] == datetime(2026, 5, 1, 14, 5, 0, tzinfo=timezone.utc)
    assert ctx["nodes"] == ["node010"]
    # Hpl-bench has no batch/v1 Job owner in our fixture, so time_limit absent.
    assert "time_limit" not in ctx


@pytest.mark.parametrize(
    "reason",
    [
        "ClusterQueueStopped",
        "LocalQueueStopped",
        "PodsReadyTimeout",
        "AdmissionCheck",
        "Deactivated",
    ],
)
def test_noncanonical_eviction_reasons_normalize_to_evicted_for_service_fallback(mocker, reason):
    workloads = deepcopy(WORKLOADS)
    workload = next(item for item in workloads["items"] if item["metadata"]["name"] == "job-hpl-bench-pp99")
    evicted = next(condition for condition in workload["status"]["conditions"] if condition["type"] == "Evicted")
    evicted["reason"] = reason
    _wire_kubectl(mocker, workloads_payload=workloads)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-beta/job-hpl-bench-pp99"])[0]

    assert ctx["state"] == "EVICTED"
    assert ctx["kueue_state"] == f"EVICTED_{reason.upper()}"


def test_topology_node_failure_eviction_normalizes_to_node_fail(mocker):
    workloads = deepcopy(WORKLOADS)
    workload = next(item for item in workloads["items"] if item["metadata"]["name"] == "job-hpl-bench-pp99")
    evicted = next(condition for condition in workload["status"]["conditions"] if condition["type"] == "Evicted")
    evicted["reason"] = "NodeFailures"
    _wire_kubectl(mocker, workloads_payload=workloads)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-beta/job-hpl-bench-pp99"])[0]

    assert ctx["state"] == "NODE_FAIL"
    assert ctx["kueue_state"] == "EVICTED_NODEFAILURES"


def test_native_requeued_condition_takes_precedence_over_retryable_eviction(mocker):
    workloads = deepcopy(WORKLOADS)
    workload = next(item for item in workloads["items"] if item["metadata"]["name"] == "job-hpl-bench-pp99")
    evicted = next(condition for condition in workload["status"]["conditions"] if condition["type"] == "Evicted")
    evicted["reason"] = "PodsReadyTimeout"
    workload["status"]["conditions"].append(
        {
            "type": "Requeued",
            "status": "True",
            "reason": "BackoffFinished",
            "lastTransitionTime": "2026-05-01T14:06:00Z",
        }
    )
    _wire_kubectl(mocker, workloads_payload=workloads)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-beta/job-hpl-bench-pp99"])[0]

    assert ctx["state"] == "REQUEUED"
    assert ctx["kueue_state"] == "EVICTED_PODSREADYTIMEOUT"


def test_stale_requeued_condition_does_not_override_readmission(mocker):
    """A workload readmitted after eviction must report RUNNING, not REQUEUED,
    even though its Requeued=True condition from the earlier eviction is
    still present (Kueue does not retroactively clear it)."""
    workloads = deepcopy(WORKLOADS)
    workload = next(item for item in workloads["items"] if item["metadata"]["name"] == "job-hpl-bench-pp99")
    workload["status"]["conditions"] = [
        {"type": "Admitted", "status": "True", "reason": "Admitted", "lastTransitionTime": "2026-05-01T14:10:00Z"},
        {"type": "PodsReady", "status": "True", "lastTransitionTime": "2026-05-01T14:11:00Z"},
        {"type": "Evicted", "status": "False", "reason": "Preempted", "lastTransitionTime": "2026-05-01T14:10:00Z"},
        {
            "type": "Requeued",
            "status": "True",
            "reason": "BackoffFinished",
            "lastTransitionTime": "2026-05-01T14:06:00Z",
        },
    ]
    _wire_kubectl(mocker, workloads_payload=workloads)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-beta/job-hpl-bench-pp99"])[0]

    assert ctx["state"] == "RUNNING"
    assert ctx["kueue_state"] == "RUNNING"


def test_stale_requeued_condition_does_not_override_finished_timeout(mocker):
    """A workload that later finishes as TIMEOUT must not get stuck on a
    stale Requeued=True condition from an earlier eviction/readmission
    cycle."""
    workloads = deepcopy(WORKLOADS)
    workload = next(item for item in workloads["items"] if item["metadata"]["name"] == "job-hpl-bench-pp99")
    workload["status"]["conditions"] = [
        {"type": "Admitted", "status": "True", "reason": "Admitted", "lastTransitionTime": "2026-05-01T14:10:00Z"},
        {"type": "Evicted", "status": "False", "reason": "Preempted", "lastTransitionTime": "2026-05-01T14:10:00Z"},
        {
            "type": "Requeued",
            "status": "True",
            "reason": "BackoffFinished",
            "lastTransitionTime": "2026-05-01T14:06:00Z",
        },
        {"type": "Finished", "status": "True", "reason": "Failed", "lastTransitionTime": "2026-05-01T14:20:00Z"},
    ]
    jobs = deepcopy(JOBS)
    hpl_bench_job = next(item for item in jobs["items"] if item["metadata"]["name"] == "hpl-bench")
    hpl_bench_job["status"] = {
        "conditions": [
            {
                "type": "Failed",
                "status": "True",
                "reason": "DeadlineExceeded",
                "lastTransitionTime": "2026-05-01T14:19:00Z",
            }
        ]
    }
    _wire_kubectl(mocker, workloads_payload=workloads, jobs_payload=jobs)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-beta/job-hpl-bench-pp99"])[0]

    assert ctx["state"] == "TIMEOUT"
    assert ctx["kueue_state"] == "FINISHED_FAILED"


def test_rejected_admission_check_preserves_native_eviction_state(mocker):
    workloads = deepcopy(WORKLOADS)
    workload = next(item for item in workloads["items"] if item["metadata"]["name"] == "job-hpl-bench-pp99")
    evicted = next(condition for condition in workload["status"]["conditions"] if condition["type"] == "Evicted")
    evicted["reason"] = "Deactivated"
    workload["status"]["admissionChecks"] = [{"name": "provisioning", "state": "Rejected"}]
    _wire_kubectl(mocker, workloads_payload=workloads)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-beta/job-hpl-bench-pp99"])[0]

    assert ctx["state"] == "EVICTED"
    assert ctx["kueue_state"] == "EVICTED_DEACTIVATED"


def test_preemption_takes_precedence_over_node_loss_evidence(mocker):
    pods = deepcopy(PODS)
    pod = next(
        item
        for item in pods["items"]
        if item["metadata"]["labels"].get("batch.kubernetes.io/controller-uid") == "owner-1003"
    )
    pod["status"] = {"phase": "Failed", "reason": "NodeLost"}
    _wire_kubectl(mocker, pods_payload=pods)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-beta/job-hpl-bench-pp99"])[0]

    assert ctx["state"] == "PREEMPTED"
    assert ctx["kueue_state"] == "EVICTED_PREEMPTED"
    assert ctx["failed_node"] == "node010"


def test_preempted_condition_normalizes_to_preempted(mocker):
    workloads = deepcopy(WORKLOADS)
    workload = next(item for item in workloads["items"] if item["metadata"]["name"] == "job-hpl-bench-pp99")
    workload["status"]["conditions"] = [
        condition for condition in workload["status"]["conditions"] if condition["type"] != "Evicted"
    ]
    _wire_kubectl(mocker, workloads_payload=workloads)

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-beta/job-hpl-bench-pp99"])[0]

    assert ctx["state"] == "PREEMPTED"
    assert ctx["kueue_state"] == "PREEMPTED"


def test_running_workload_substitutes_now_for_end_time(mocker):
    _wire_kubectl(mocker)
    before = datetime.now(tz=timezone.utc)
    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-still-running-qq01"])
    after = datetime.now(tz=timezone.utc)
    ctx = contexts[0]
    assert ctx["state"] == "RUNNING"
    # end_time was substituted from now() because Finished hasn't fired yet.
    assert before <= ctx["end_time"] <= after
    assert ctx["start_time"] == datetime(2026, 5, 2, 8, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("conditions", "expected_state"),
    [
        ([], "PENDING"),
        (
            [{"type": "QuotaReserved", "status": "True", "lastTransitionTime": "2026-05-02T09:00:00Z"}],
            "QUOTA_RESERVED",
        ),
        (
            [{"type": "Admitted", "status": "True", "lastTransitionTime": "2026-05-02T09:01:00Z"}],
            "ADMITTED",
        ),
    ],
    ids=["pending", "quota-reserved", "admitted"],
)
def test_pre_execution_states_pass_through_for_service_fallback(mocker, conditions, expected_state):
    workloads = deepcopy(WORKLOADS)
    workload = next(item for item in workloads["items"] if item["metadata"]["name"] == "job-pending-quota-rr11")
    workload["status"] = {"conditions": conditions}
    _wire_kubectl(mocker, workloads_payload=workloads)

    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-pending-quota-rr11"])

    assert len(contexts) == 1
    assert contexts[0]["state"] == expected_state


def test_discover_by_ids_skips_missing_workload(mocker):
    """kubectl returning non-zero (NotFound) must come back as [] for that id."""
    _wire_kubectl(mocker)
    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/does-not-exist"])
    assert contexts == []


def test_bare_id_rejected_without_namespace_config():
    """Bare names without a configured namespace are ambiguous and rejected."""
    with pytest.raises(ValueError, match="missing a namespace"):
        KueueExecutionContextAdapter().discover_by_ids(["job-train-mnist-abc12"])


def test_bare_id_works_when_namespace_configured(mocker):
    _wire_kubectl(mocker)
    eca = KueueExecutionContextAdapter(namespace="team-alpha")
    ctxs = eca.discover_by_ids(["job-train-mnist-abc12"])
    assert len(ctxs) == 1
    assert ctxs[0]["namespace"] == "team-alpha"


def test_discover_by_time_range_filters_window(mocker):
    _wire_kubectl(mocker)
    eca = KueueExecutionContextAdapter()
    # Window that overlaps only the May-1 workloads, not the May-2 ones.
    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 1, 23, 59, 0, tzinfo=timezone.utc)
    contexts = eca.discover_by_time_range(start, end)
    names = {c["name"] for c in contexts}
    assert names == {
        "job-train-mnist-abc12",
        "job-llama-7b-pretrain-xyz77",
        "job-hpl-bench-pp99",
    }


def test_discover_by_time_range_intersects_nodelist(mocker):
    _wire_kubectl(mocker)
    eca = KueueExecutionContextAdapter()
    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 2, 23, 59, 0, tzinfo=timezone.utc)
    contexts = eca.discover_by_time_range(start, end, nodelist=["node002"])
    names = {c["name"] for c in contexts}
    # Only the llama workload's pods landed on node002.
    assert names == {"job-llama-7b-pretrain-xyz77"}


def test_kubectl_invocation_carries_kubeconfig_and_context(mocker):
    run_mock = _wire_kubectl(mocker)
    eca = KueueExecutionContextAdapter(
        kubectl_path="/usr/local/bin/kubectl",
        kubeconfig="/tmp/kc",
        context="kind-kueue-demo",
    )
    eca.discover_by_ids(["team-alpha/job-train-mnist-abc12"])
    first_call = run_mock.call_args_list[0]
    cmd = first_call.args[0]
    assert cmd[0] == "/usr/local/bin/kubectl"
    assert "--kubeconfig" in cmd and "/tmp/kc" in cmd
    assert "--context" in cmd and "kind-kueue-demo" in cmd
    assert "-o" in cmd and "json" in cmd


def test_namespace_scoped_list_uses_minus_n_not_dash_a(mocker):
    run_mock = _wire_kubectl(mocker)
    eca = KueueExecutionContextAdapter(namespace="team-alpha")
    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 2, 0, 0, 0, tzinfo=timezone.utc)
    eca.discover_by_time_range(start, end)
    listing_call = run_mock.call_args_list[0]
    cmd = listing_call.args[0]
    assert "-n" in cmd and "team-alpha" in cmd
    assert "-A" not in cmd


def test_unscoped_list_uses_dash_a(mocker):
    run_mock = _wire_kubectl(mocker)
    eca = KueueExecutionContextAdapter()
    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 2, 0, 0, 0, tzinfo=timezone.utc)
    eca.discover_by_time_range(start, end)
    listing_call = run_mock.call_args_list[0]
    cmd = listing_call.args[0]
    assert "-A" in cmd


def test_invalid_json_from_kubectl_treated_as_empty(mocker):
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="not-json{", stderr="")
    mocker.patch("subprocess.run", return_value=completed)
    eca = KueueExecutionContextAdapter()
    assert (
        eca.discover_by_time_range(
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
        == []
    )


def test_kubectl_timeout_returns_empty(mocker):
    err = subprocess.TimeoutExpired(cmd=["kubectl"], timeout=60)
    mocker.patch("subprocess.run", side_effect=err)
    eca = KueueExecutionContextAdapter()
    assert eca.discover_by_ids(["team-alpha/anything"]) == []


def test_finished_state_wins_over_earlier_evicted(mocker):
    """A workload that was evicted earlier in its lifetime but finished later
    must report FINISHED_*, not EVICTED_*. Mirrors Slurm's terminal-state-wins
    rule."""
    payload = {
        "items": [
            {
                "apiVersion": "kueue.x-k8s.io/v1beta1",
                "kind": "Workload",
                "metadata": {
                    "name": "recovered",
                    "namespace": "team-alpha",
                    "uid": "uid-rec-1",
                    "creationTimestamp": "2026-05-03T11:59:00Z",
                },
                "spec": {"queueName": "q"},
                "status": {
                    "conditions": [
                        {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-05-03T12:00:00Z"},
                        {
                            "type": "Evicted",
                            "status": "True",
                            "lastTransitionTime": "2026-05-03T12:05:00Z",
                            "reason": "PodsReadyTimeout",
                        },
                        {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-05-03T12:10:00Z"},
                        {
                            "type": "Finished",
                            "status": "True",
                            "lastTransitionTime": "2026-05-03T12:30:00Z",
                            "reason": "Succeeded",
                        },
                    ]
                },
            }
        ]
    }
    pods = {"items": []}
    mocker.patch(
        "subprocess.run",
        side_effect=_kubectl_router(workloads_payload=payload, pods_payload=pods, jobs_payload={"items": []}),
    )
    contexts = KueueExecutionContextAdapter().discover_by_time_range(
        datetime(2026, 5, 3, tzinfo=timezone.utc),
        datetime(2026, 5, 4, tzinfo=timezone.utc),
    )
    assert len(contexts) == 1
    assert contexts[0]["state"] == "COMPLETED"
    assert contexts[0]["kueue_state"] == "FINISHED_SUCCEEDED"


def test_missing_uid_workload_skipped(mocker):
    payload = {
        "items": [
            {
                "apiVersion": "kueue.x-k8s.io/v1beta1",
                "kind": "Workload",
                "metadata": {"name": "no-uid", "namespace": "team-alpha"},
                "spec": {},
                "status": {"conditions": []},
            }
        ]
    }
    mocker.patch(
        "subprocess.run",
        side_effect=_kubectl_router(workloads_payload=payload, pods_payload={"items": []}, jobs_payload={"items": []}),
    )
    contexts = KueueExecutionContextAdapter().discover_by_time_range(
        datetime(2026, 5, 1, tzinfo=timezone.utc),
        datetime(2026, 5, 4, tzinfo=timezone.utc),
    )
    assert contexts == []


def test_pods_filtered_by_batch_controller_uid_not_workload_uid_or_kueue_label(mocker):
    """Regression for two related bugs:

      1) Looking up pods by the Workload's `metadata.uid` returns zero pods
         on a real cluster (the original POC bug, masked by fixtures that
         conflated Workload UID with the label value).
      2) Looking up pods by Kueue's `kueue.x-k8s.io/job-uid` ALSO returns
         zero pods - per the Kueue docs that label is set on the Workload,
         not on its pods. The pods carry the standard Kubernetes
         Job-controller label `batch.kubernetes.io/controller-uid=<job uid>`.

    Only the pod carrying `batch.kubernetes.io/controller-uid` whose value
    is the owner Job's UID must be associated with this workload."""
    workloads = {
        "items": [
            {
                "apiVersion": "kueue.x-k8s.io/v1beta1",
                "kind": "Workload",
                "metadata": {
                    "name": "regression",
                    "namespace": "team-alpha",
                    "uid": "WORKLOAD-UID-1",
                    "creationTimestamp": "2026-05-10T11:59:00Z",
                    "ownerReferences": [{"apiVersion": "batch/v1", "kind": "Job", "name": "regr", "uid": "JOB-UID-1"}],
                },
                "spec": {"queueName": "q"},
                "status": {
                    "conditions": [
                        {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-05-10T12:00:00Z"},
                        {
                            "type": "Finished",
                            "status": "True",
                            "lastTransitionTime": "2026-05-10T12:30:00Z",
                            "reason": "Succeeded",
                        },
                    ]
                },
            }
        ]
    }
    pods = {
        "items": [
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "p-batch-controller-uid",
                    "namespace": "team-alpha",
                    "labels": {
                        "batch.kubernetes.io/controller-uid": "JOB-UID-1",
                        "batch.kubernetes.io/job-name": "regr",
                    },
                },
                "spec": {"nodeName": "node-correct"},
                "status": {"phase": "Succeeded"},
            },
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "p-kueue-label-only",
                    "namespace": "team-alpha",
                    # Pod that ONLY carries the kueue label (with the owner
                    # job uid as value) - must NOT match, because pods
                    # don't actually carry this label in real clusters and
                    # the connector keys off the batch controller-uid label.
                    "labels": {"kueue.x-k8s.io/job-uid": "JOB-UID-1"},
                },
                "spec": {"nodeName": "node-wrong-kueue-label"},
                "status": {"phase": "Succeeded"},
            },
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "p-workload-uid",
                    "namespace": "team-alpha",
                    # batch label present but value is the Workload UID,
                    # not the owner Job UID - must NOT match.
                    "labels": {"batch.kubernetes.io/controller-uid": "WORKLOAD-UID-1"},
                },
                "spec": {"nodeName": "node-wrong-workload-uid"},
                "status": {"phase": "Succeeded"},
            },
        ]
    }
    jobs = {
        "items": [
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": "regr", "namespace": "team-alpha", "uid": "JOB-UID-1"},
                "spec": {},
            }
        ]
    }
    mocker.patch(
        "subprocess.run",
        side_effect=_kubectl_router(workloads_payload=workloads, pods_payload=pods, jobs_payload=jobs),
    )
    ctxs = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/regression"])
    assert len(ctxs) == 1
    assert ctxs[0]["nodes"] == ["node-correct"]


def test_missing_live_pods_are_recovered_from_uid_matched_events(mocker):
    """A completed Job may retain its Workload after its pod is deleted.

    Recover the pod from an aggregated, exact owner-UID SuccessfulCreate Event
    and its node from the corresponding scheduler Event. An Event for an older
    Job with the same name must not leak into the recreated Job's attempt.
    """
    events = {
        "items": [
            {
                "metadata": {"creationTimestamp": "2026-05-01T12:00:05Z"},
                "involvedObject": {
                    "kind": "Job",
                    "namespace": "team-alpha",
                    "name": "train-mnist",
                    "uid": "owner-OLD",
                },
                "reason": "SuccessfulCreate",
                "message": "Created pod: train-mnist-old",
                "lastTimestamp": "2026-05-01T12:00:05Z",
            },
            {
                "metadata": {"creationTimestamp": "2026-05-01T12:00:06Z"},
                "involvedObject": {
                    "kind": "Pod",
                    "namespace": "team-alpha",
                    "name": "train-mnist-old",
                    "uid": "pod-OLD",
                },
                "reason": "Scheduled",
                "message": "Successfully assigned team-alpha/train-mnist-old to node-old",
                "lastTimestamp": "2026-05-01T12:00:06Z",
            },
            {
                "metadata": {"creationTimestamp": "2026-05-01T12:00:10Z"},
                "involvedObject": {
                    "kind": "Job",
                    "namespace": "team-alpha",
                    "name": "train-mnist",
                    "uid": "owner-1001",
                },
                "reason": "SuccessfulCreate",
                "message": "(combined from similar events): Created pod: train-mnist-new",
                "lastTimestamp": "2026-05-01T12:00:10Z",
            },
            {
                "metadata": {"creationTimestamp": "2026-05-01T12:00:11Z"},
                "involvedObject": {
                    "kind": "Pod",
                    "namespace": "team-alpha",
                    "name": "train-mnist-new",
                    "uid": "pod-NEW",
                },
                "reason": "Scheduled",
                "message": "Successfully assigned team-alpha/train-mnist-new to node-new",
                "lastTimestamp": "2026-05-01T12:00:11Z",
            },
        ]
    }
    _wire_kubectl(mocker, pods_payload={"items": []}, events_payload=events)

    # Fixture end_time is fixed in the past; disable the age gate so this
    # test isn't coupled to wall-clock time.
    eca = KueueExecutionContextAdapter(event_max_age_seconds=2**31)
    ctx = eca.discover_by_ids(["team-alpha/job-train-mnist-abc12"])[0]

    assert ctx["pod_names"] == ["train-mnist-new"]
    assert ctx["nodes"] == ["node-new"]


def test_event_recovery_skipped_for_workloads_older_than_max_age(mocker):
    """A workload whose end_time is older than `event_max_age_seconds` skips
    Event recovery entirely (no `kubectl get events` calls), even though
    recoverable Events exist, since they've almost certainly expired."""
    events = {
        "items": [
            {
                "metadata": {"creationTimestamp": "2026-05-01T12:00:10Z"},
                "involvedObject": {
                    "kind": "Job",
                    "namespace": "team-alpha",
                    "name": "train-mnist",
                    "uid": "owner-1001",
                },
                "reason": "SuccessfulCreate",
                "message": "Created pod: train-mnist-new",
                "lastTimestamp": "2026-05-01T12:00:10Z",
            },
        ]
    }
    run_mock = _wire_kubectl(mocker, pods_payload={"items": []}, events_payload=events)

    eca = KueueExecutionContextAdapter(event_max_age_seconds=1)
    ctx = eca.discover_by_ids(["team-alpha/job-train-mnist-abc12"])[0]

    assert "pod_names" not in ctx
    assert ctx["nodes"] == []
    for call in run_mock.call_args_list:
        assert "events" not in call.args[0]


def test_missing_pod_and_node_lists_are_reported_when_events_cannot_recover_them(mocker):
    warning = mocker.patch("attribution_blueprint.connectors.execution_contexts.kueue.connector.logger.warning")
    _wire_kubectl(mocker, pods_payload={"items": []}, events_payload={"items": []})

    ctx = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-train-mnist-abc12"])[0]

    assert "pod_names" not in ctx
    assert ctx["nodes"] == []
    warning.assert_any_call(
        "workload team-alpha/job-train-mnist-abc12: "
        "pod list and node list unavailable from live pods and Kubernetes Events"
    )


def test_kubectl_pod_selector_uses_batch_controller_uid_label(mocker):
    """The label selector kubectl sees on the wire must be
    `batch.kubernetes.io/controller-uid=<owner-job-uid>`. Kueue's own label
    (`kueue.x-k8s.io/job-uid`) is set on Workloads, not pods - using it as
    a pod selector matches zero pods on a real cluster."""
    run_mock = _wire_kubectl(mocker)
    KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-train-mnist-abc12"])
    pod_calls = [call for call in run_mock.call_args_list if "pods" in call.args[0] and "-l" in call.args[0]]
    assert pod_calls, "expected at least one `kubectl get pods -l ...` invocation"
    cmd = pod_calls[0].args[0]
    selector = cmd[cmd.index("-l") + 1]
    assert selector.startswith("batch.kubernetes.io/controller-uid=")
    assert "kueue.x-k8s.io/job-uid" not in selector


def test_collect_pod_evidence_rejects_unlabelled_and_mismatched_pods():
    """Defensive filter: pods without the batch.kubernetes.io/controller-uid
    label (or with a different value) must never count toward a workload's
    evidence, even if they leaked into the input list. Previously the
    filter accepted `None` (unlabelled) as a match, which would associate
    any foreign pod that bypassed the server-side label selector."""
    from attribution_blueprint.connectors.execution_contexts.kueue.connector import _collect_pod_evidence

    ns = "team-a"
    pods = [
        {
            "metadata": {"namespace": ns, "labels": {"batch.kubernetes.io/controller-uid": "owner-A"}},
            "spec": {"nodeName": "n-good"},
        },
        {"metadata": {"namespace": ns, "labels": {}}, "spec": {"nodeName": "n-unlabelled"}},
        {
            "metadata": {"namespace": ns, "labels": {"batch.kubernetes.io/controller-uid": "owner-B"}},
            "spec": {"nodeName": "n-other"},
        },
    ]
    selector = (ns, "batch.kubernetes.io/controller-uid", "owner-A")
    nodes, failed_node, _, _ = _collect_pod_evidence(pods, selector)
    assert nodes == ["n-good"]
    assert failed_node is None


def test_discover_by_time_range_skips_pod_and_job_fetches_for_out_of_window_workloads(mocker):
    """`discover_by_time_range` parses workload timestamps cheaply, applies
    the [start, end] window filter, and only THEN fetches pods + owner Jobs for the
    survivors. Previously every workload in the cluster-wide listing took N
    kubectl round-trips up front, even when 99% were obviously outside the
    window. With long-running clusters this turned a millisecond query into
    minutes."""

    def _wl(name: str, ns: str, uid: str, owner_uid: str, owner_name: str, day: int) -> dict:
        return {
            "apiVersion": "kueue.x-k8s.io/v1beta1",
            "kind": "Workload",
            "metadata": {
                "name": name,
                "namespace": ns,
                "uid": uid,
                "creationTimestamp": f"2026-05-{day:02d}T11:59:00Z",
                "ownerReferences": [{"apiVersion": "batch/v1", "kind": "Job", "name": owner_name, "uid": owner_uid}],
            },
            "spec": {"queueName": "q"},
            "status": {
                "conditions": [
                    {"type": "Admitted", "status": "True", "lastTransitionTime": f"2026-05-{day:02d}T12:00:00Z"},
                    {
                        "type": "Finished",
                        "status": "True",
                        "lastTransitionTime": f"2026-05-{day:02d}T12:30:00Z",
                        "reason": "Succeeded",
                    },
                ]
            },
        }

    workloads = {
        "items": [
            _wl("in-window", "team-alpha", "uid-in", "owner-in", "job-in", 1),
            _wl("out-of-window-1", "team-alpha", "uid-out1", "owner-out1", "job-out1", 10),
            _wl("out-of-window-2", "team-alpha", "uid-out2", "owner-out2", "job-out2", 15),
        ]
    }
    pods = {"items": []}
    jobs = {
        "items": [
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": "job-in", "namespace": "team-alpha", "uid": "owner-in"},
                "spec": {},
            }
        ]
    }
    run_mock = mocker.patch(
        "subprocess.run",
        side_effect=_kubectl_router(workloads_payload=workloads, pods_payload=pods, jobs_payload=jobs),
    )
    eca = KueueExecutionContextAdapter()
    contexts = eca.discover_by_time_range(
        datetime(2026, 4, 30, tzinfo=timezone.utc),
        datetime(2026, 5, 2, tzinfo=timezone.utc),
    )
    assert {c["name"] for c in contexts} == {"in-window"}

    # Inspect every kubectl invocation. Pod / owner-Job fetches must only target
    # the in-window workload's owner Job. Out-of-window owners must NOT
    # appear in any selector or positional arg.
    out_owner_uids = {"owner-out1", "owner-out2"}
    out_job_names = {"job-out1", "job-out2"}
    for call in run_mock.call_args_list:
        argv = call.args[0]
        kind = argv[argv.index("get") + 1] if "get" in argv else ""
        if kind == "pods":
            selector = argv[argv.index("-l") + 1]
            assert not any(u in selector for u in out_owner_uids), (
                f"unexpected pod fetch for out-of-window owner: {argv}"
            )
        elif kind == "job":
            positional = argv[argv.index("get") + 2]  # `get job <name>`
            assert positional not in out_job_names, f"unexpected job fetch: {argv}"


def test_failed_node_is_deterministic_across_pod_orderings():
    """`failed_node` must not depend on kubectl's response ordering.
    Previously `_collect_pod_evidence` picked the first iteration-order
    pod with a non-zero exit code, which made `failed_node` flake when
    the API returned pods in a different order between runs. Lock in
    the deterministic choice: lexicographically smallest node name
    among pods that reported a non-zero exit code."""
    from attribution_blueprint.connectors.execution_contexts.kueue.connector import _collect_pod_evidence

    ns = "team-a"

    def _pod(node: str, exit_code: int | None) -> dict:
        cs: dict = {}
        if exit_code is not None:
            cs = {"containerStatuses": [{"state": {"terminated": {"exitCode": exit_code}}}]}
        return {
            "metadata": {"namespace": ns, "labels": {"batch.kubernetes.io/controller-uid": "owner-A"}},
            "spec": {"nodeName": node},
            "status": cs,
        }

    # Three pods with non-zero exits on different nodes. Input order is
    # scrambled - the deterministic result must be the same regardless.
    pods_a = [_pod("n-c", 137), _pod("n-a", 1), _pod("n-b", 2), _pod("n-d", 0)]
    pods_b = list(reversed(pods_a))
    pods_c = [_pod("n-b", 2), _pod("n-d", 0), _pod("n-a", 1), _pod("n-c", 137)]

    selector = (ns, "batch.kubernetes.io/controller-uid", "owner-A")
    _, failed_a, _, _ = _collect_pod_evidence(pods_a, selector)
    _, failed_b, _, _ = _collect_pod_evidence(pods_b, selector)
    _, failed_c, _, _ = _collect_pod_evidence(pods_c, selector)

    assert failed_a == failed_b == failed_c == "n-a"


def test_collect_pod_evidence_prefers_node_loss_over_alphabetically_smaller_exit_code_node():
    """Node-loss evidence must outrank a generic non-zero exit code: a
    collateral exit on an alphabetically-smaller node must not get blamed
    instead of the node that was actually lost."""
    from attribution_blueprint.connectors.execution_contexts.kueue.connector import _collect_pod_evidence

    ns = "team-a"
    node_lost_pod = {
        "metadata": {"namespace": ns, "labels": {"batch.kubernetes.io/controller-uid": "owner-A"}},
        "spec": {"nodeName": "node-z"},
        "status": {"reason": "NodeLost"},
    }
    collateral_exit_pod = {
        "metadata": {"namespace": ns, "labels": {"batch.kubernetes.io/controller-uid": "owner-A"}},
        "spec": {"nodeName": "node-a"},
        "status": {"containerStatuses": [{"state": {"terminated": {"exitCode": 137}}}]},
    }
    selector = (ns, "batch.kubernetes.io/controller-uid", "owner-A")

    _, failed_node, _, _ = _collect_pod_evidence([node_lost_pod, collateral_exit_pod], selector)

    assert failed_node == "node-z"


def test_collect_pod_evidence_returns_empty_for_workload_with_no_owner_job():
    """A Workload without an owner Job (e.g. pending, never-admitted) has no
    pods to attribute. Returning ([], None) avoids accidentally folding
    every pod in the cluster into such workloads."""
    from attribution_blueprint.connectors.execution_contexts.kueue.connector import _collect_pod_evidence

    pods = [{"metadata": {"labels": {"batch.kubernetes.io/controller-uid": "anything"}}, "spec": {"nodeName": "n"}}]
    assert _collect_pod_evidence(pods, None) == ([], None, [], {})


def test_pod_nodes_maps_scheduled_pods_and_omits_unscheduled_ones():
    from attribution_blueprint.connectors.execution_contexts.kueue.connector import _collect_pod_evidence

    ns = "team-a"
    selector = (ns, "batch.kubernetes.io/controller-uid", "owner-A")
    label_key, label_value = selector[1], selector[2]
    pods = [
        {"metadata": {"name": "w-0", "namespace": ns, "labels": {label_key: label_value}}, "spec": {"nodeName": "n-1"}},
        {"metadata": {"name": "w-1", "namespace": ns, "labels": {label_key: label_value}}, "spec": {"nodeName": "n-2"}},
        {"metadata": {"name": "w-pending", "namespace": ns, "labels": {label_key: label_value}}, "spec": {}},
    ]
    evidence = _collect_pod_evidence(pods, selector)
    assert evidence.pod_nodes == {"w-0": "n-1", "w-1": "n-2"}
    assert evidence.pod_names == ["w-0", "w-1", "w-pending"]


def test_same_named_jobsets_in_two_namespaces_keep_separate_pod_evidence():
    """The pod list is pooled across workloads and the JobSet label is only
    unique per namespace, so pods must be scoped to the workload's namespace
    before matching or `pod_nodes` from one tenant overwrites the other's."""
    from attribution_blueprint.connectors.execution_contexts.kueue.connector import _parse_workload

    def _wl(ns: str) -> dict:
        return {
            "metadata": {
                "name": "train",
                "namespace": ns,
                "uid": f"wl-{ns}",
                "creationTimestamp": "2026-05-01T11:59:00Z",
                "ownerReferences": [{"apiVersion": "jobset.x-k8s.io/v1alpha2", "kind": "JobSet", "name": "train"}],
            },
            "spec": {},
            "status": {
                "conditions": [
                    {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-05-01T12:00:00Z"},
                    {"type": "Finished", "status": "True", "lastTransitionTime": "2026-05-01T12:30:00Z"},
                ]
            },
        }

    def _pod(ns: str, node: str) -> dict:
        return {
            "metadata": {"name": "train-w-0-0", "namespace": ns, "labels": {"jobset.sigs.k8s.io/jobset-name": "train"}},
            "spec": {"nodeName": node},
            "status": {},
        }

    pods = [_pod("team-alpha", "node-a"), _pod("team-beta", "node-b")]
    now = datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc)
    alpha = _parse_workload(_wl("team-alpha"), pods, {}, now=now)
    beta = _parse_workload(_wl("team-beta"), pods, {}, now=now)
    assert alpha["nodes"] == ["node-a"] and alpha["pod_nodes"] == {"train-w-0-0": "node-a"}
    assert beta["nodes"] == ["node-b"] and beta["pod_nodes"] == {"train-w-0-0": "node-b"}


def test_identically_named_jobsets_in_different_namespaces_do_not_cross_contaminate(mocker):
    """`jobset.sigs.k8s.io/jobset-name` carries only a name, which is only
    unique per namespace. Cluster-wide discovery pools pods from every
    namespace before filtering, so a namespace-unaware match would let
    team-b's node-loss evidence leak into team-a's identically-named
    JobSet Workload."""

    def _workload(namespace: str, uid: str) -> dict:
        return {
            "apiVersion": "kueue.x-k8s.io/v1beta1",
            "kind": "Workload",
            "metadata": {
                "name": "trainer-wl",
                "namespace": namespace,
                "uid": uid,
                "creationTimestamp": "2026-05-01T12:00:00Z",
                "ownerReferences": [{"kind": "JobSet", "name": "trainer", "uid": f"jobset-{namespace}"}],
            },
            "spec": {"podSets": []},
            "status": {
                "conditions": [
                    {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-05-01T12:01:00Z"},
                    {"type": "PodsReady", "status": "True", "lastTransitionTime": "2026-05-01T12:02:00Z"},
                ]
            },
        }

    def _pod(namespace: str, name: str, node: str, *, node_lost: bool) -> dict:
        pod = {
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {"jobset.sigs.k8s.io/jobset-name": "trainer"},
            },
            "spec": {"nodeName": node},
            "status": {},
        }
        if node_lost:
            pod["status"] = {"phase": "Failed", "reason": "NodeLost"}
        return pod

    workloads = {"items": [_workload("team-a", "wl-team-a"), _workload("team-b", "wl-team-b")]}
    pods = {
        "items": [
            _pod("team-a", "trainer-a-0", "node-a", node_lost=False),
            _pod("team-b", "trainer-b-0", "node-b", node_lost=True),
        ]
    }
    _wire_kubectl(mocker, workloads_payload=workloads, pods_payload=pods, jobs_payload={"items": []})

    contexts = {
        ctx["namespace"]: ctx
        for ctx in KueueExecutionContextAdapter().discover_by_ids(["team-a/trainer-wl", "team-b/trainer-wl"])
    }

    assert contexts["team-a"]["state"] != "NODE_FAIL"
    assert contexts["team-a"]["nodes"] == ["node-a"]
    assert contexts["team-a"]["pod_names"] == ["trainer-a-0"]
    assert "failed_node" not in contexts["team-a"]

    assert contexts["team-b"]["state"] == "NODE_FAIL"
    assert contexts["team-b"]["nodes"] == ["node-b"]
    assert contexts["team-b"]["pod_names"] == ["trainer-b-0"]
    assert contexts["team-b"]["failed_node"] == "node-b"


def test_discover_by_time_range_rejects_naive_datetimes(mocker):
    """Naive `start`/`end` would crash mid-loop (TypeError comparing naive vs
    tz-aware in the per-workload filter). Reject at the boundary instead so
    the failure is local and actionable."""
    _wire_kubectl(mocker)
    eca = KueueExecutionContextAdapter()
    with pytest.raises(ValueError, match="tz-aware"):
        eca.discover_by_time_range(
            datetime(2026, 5, 1),  # naive
            datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
    with pytest.raises(ValueError, match="tz-aware"):
        eca.discover_by_time_range(
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 2),  # naive
        )


def test_start_time_uses_earliest_admitted_not_latest(mocker):
    """When a workload was admitted, preempted, and re-admitted, start_time
    must reflect the FIRST admission so a time-window query covering only
    the original admission interval still finds the workload. The previous
    code used max(lastTransitionTime), which silently shifted start_time
    forward to the re-admission."""
    payload = {
        "items": [
            {
                "apiVersion": "kueue.x-k8s.io/v1beta1",
                "kind": "Workload",
                "metadata": {
                    "name": "reattempted",
                    "namespace": "team-alpha",
                    "uid": "u-reattempted",
                    "creationTimestamp": "2026-05-03T11:59:00Z",
                },
                "spec": {},
                "status": {
                    "conditions": [
                        {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-05-03T12:00:00Z"},
                        {
                            "type": "Evicted",
                            "status": "True",
                            "lastTransitionTime": "2026-05-03T12:05:00Z",
                            "reason": "PodsReadyTimeout",
                        },
                        {"type": "Admitted", "status": "True", "lastTransitionTime": "2026-05-03T12:10:00Z"},
                        {
                            "type": "Finished",
                            "status": "True",
                            "lastTransitionTime": "2026-05-03T12:30:00Z",
                            "reason": "Succeeded",
                        },
                    ]
                },
            }
        ]
    }
    mocker.patch(
        "subprocess.run",
        side_effect=_kubectl_router(workloads_payload=payload, pods_payload={"items": []}, jobs_payload={"items": []}),
    )
    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/reattempted"])
    assert len(contexts) == 1
    assert contexts[0]["start_time"] == datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_build_with_none_timeout_falls_back_to_default():
    """TOML `null` (Python None) for timeout_seconds must fall back to the
    documented 60s default - NOT propagate to `subprocess.run(timeout=None)`,
    which would silently disable the timeout."""
    from attribution_blueprint.connectors.execution_contexts.kueue import _build

    adapter = _build({"timeout_seconds": None}, ctx={"framework": "eca"})
    assert adapter.timeout_seconds == 60


def test_build_rejects_placement_records_without_namespace():
    """A recorder installation watches exactly one namespace. Without a
    configured `namespace`, discovery could span namespaces the recorder
    doesn't cover, and its response would be indistinguishable from `no
    placements recorded yet` for a namespace it does watch."""
    from attribution_blueprint.connectors.execution_contexts.kueue import _build
    from attribution_blueprint.exceptions import MCAError

    spec = {
        "placement_records_enabled": True,
        "placement_recorder": {"endpoint": "http://recorder:8080"},
    }
    with pytest.raises(MCAError, match="eca.kueue.namespace"):
        _build(spec, ctx={"framework": "eca"})


def test_discover_by_time_range_intersects_csv_nodelist(mocker):
    """The kueue adapter accepts CSV-separated node names in its nodelist
    argument and intersects them client-side. Moving this parsing from
    cli.py into the adapter keeps the orchestrator's nodelist contract
    scheduler-agnostic."""
    _wire_kubectl(mocker)
    eca = KueueExecutionContextAdapter()
    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 2, 23, 59, 0, tzinfo=timezone.utc)
    contexts = eca.discover_by_time_range(start, end, nodelist=["node002, node005"])
    names = {c["name"] for c in contexts}
    assert names == {"job-llama-7b-pretrain-xyz77", "job-train-mnist-abc12"}


class TestMergePlacementRecordsFailedNode:
    """`_merge_placement_records` is the only merge source that can recover
    `failed_node` after Pod GC — node-loss/exit-code evidence lives only on
    the Pod object, so neither the PromQL placement route nor the Kubernetes
    Events fallback can reconstruct it."""

    def _record(self, *, node_name, node_failure_reason=None, pod_uid="pod-uid-1", pod_name="pod-a"):
        from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import (
            OwnerIdentity,
            PlacementRecord,
            WorkloadIdentity,
        )

        return PlacementRecord(
            namespace="team-a",
            workload=WorkloadIdentity(name="wl-a", uid="workload-uid-1"),
            owner=OwnerIdentity(api_version="batch/v1", kind="Job", name="job-a", uid="job-uid-1"),
            pod_name=pod_name,
            pod_uid=pod_uid,
            node_name=node_name,
            observed_at="2026-05-01T00:00:00Z",
            resolver_source="owner-uid",
            node_failure_reason=node_failure_reason,
        )

    def _context(self, **overrides):
        ctx = {
            "namespace": "team-a",
            "workload_uid": "workload-uid-1",
            "nodes": [],
            "state": "FAILED",
        }
        ctx.update(overrides)
        return ctx

    def test_record_with_failure_reason_sets_failed_node(self, mocker):
        eca = KueueExecutionContextAdapter()
        mocker.patch.object(
            eca,
            "_fetch_placement_records",
            return_value=[self._record(node_name="node-a", node_failure_reason="NodeLost")],
        )
        ctx = self._context()

        eca._merge_placement_records(ctx)

        assert ctx["failed_node"] == "node-a"
        assert ctx["state"] == "NODE_FAIL"

    def test_record_without_failure_reason_does_not_set_failed_node(self, mocker):
        eca = KueueExecutionContextAdapter()
        mocker.patch.object(eca, "_fetch_placement_records", return_value=[self._record(node_name="node-a")])
        ctx = self._context()

        eca._merge_placement_records(ctx)

        assert "failed_node" not in ctx
        assert ctx["state"] == "FAILED"

    @pytest.mark.parametrize("state", ["COMPLETED", "PREEMPTED", "NODE_FAIL"])
    def test_higher_precedence_or_existing_node_fail_state_is_preserved(self, mocker, state):
        eca = KueueExecutionContextAdapter()
        mocker.patch.object(
            eca,
            "_fetch_placement_records",
            return_value=[self._record(node_name="node-a", node_failure_reason="NodeLost")],
        )
        ctx = self._context(state=state)

        eca._merge_placement_records(ctx)

        assert ctx["state"] == state
        assert ctx["failed_node"] == "node-a"

    def test_retained_container_exit_does_not_reclassify_state(self, mocker):
        eca = KueueExecutionContextAdapter()
        mocker.patch.object(
            eca,
            "_fetch_placement_records",
            return_value=[self._record(node_name="node-a", node_failure_reason="ExitCode:1")],
        )
        ctx = self._context(state="FAILED")

        eca._merge_placement_records(ctx)

        assert ctx["state"] == "FAILED"
        assert ctx["failed_node"] == "node-a"

    def test_retained_node_loss_overrides_live_exit_code(self, mocker):
        eca = KueueExecutionContextAdapter()
        mocker.patch.object(
            eca,
            "_fetch_placement_records",
            return_value=[self._record(node_name="node-b", node_failure_reason="NodeLost")],
        )
        ctx = self._context(failed_node="node-a")

        eca._merge_placement_records(ctx)

        assert ctx["failed_node"] == "node-b"
        assert ctx["state"] == "NODE_FAIL"

    def test_finalize_prefers_deleted_pod_node_loss_over_surviving_pod_exit(self, mocker):
        workload = deepcopy(WORKLOADS["items"][0])
        workload["metadata"]["name"] = "wl-a"
        workload["metadata"]["uid"] = "workload-uid-1"
        finished = next(
            condition
            for condition in workload["status"]["conditions"]
            if condition["type"] == "Finished"
        )
        finished["reason"] = "Failed"

        surviving_pod = {
            "metadata": {
                "namespace": "team-alpha",
                "name": "pod-a",
                "uid": "pod-uid-a",
                "labels": {"batch.kubernetes.io/controller-uid": "owner-1001"},
            },
            "spec": {"nodeName": "node-a"},
            "status": {
                "containerStatuses": [
                    {"state": {"terminated": {"exitCode": 137}}},
                ]
            },
        }
        reader = mocker.Mock()
        reader.list_for_workload.return_value = [
            self._record(
                node_name="node-b",
                node_failure_reason="NodeLost",
                pod_uid="pod-uid-b",
                pod_name="pod-b",
            )
        ]
        eca = KueueExecutionContextAdapter(
            placement_reader=reader,
            placement_records_enabled=True,
        )
        mocker.patch.object(eca, "_fetch_pods_for", return_value=[surviving_pod])
        mocker.patch.object(eca, "_fetch_owner_jobs", return_value={})

        [ctx] = eca._finalize([workload])

        assert ctx["state"] == "NODE_FAIL"
        assert ctx["failed_node"] == "node-b"

    def test_node_loss_evidence_outranks_alphabetically_smaller_exit_code_node(self, mocker):
        """Node-loss evidence must win regardless of node-name ordering: a
        collateral non-zero exit on an alphabetically-smaller node must not
        get blamed instead of the node that was actually lost."""
        eca = KueueExecutionContextAdapter()
        mocker.patch.object(
            eca,
            "_fetch_placement_records",
            return_value=[
                self._record(node_name="node-z", node_failure_reason="NodeLost", pod_uid="pod-uid-1", pod_name="pod-a"),
                self._record(
                    node_name="node-a", node_failure_reason="ExitCode:1", pod_uid="pod-uid-2", pod_name="pod-b"
                ),
            ],
        )
        ctx = self._context()

        eca._merge_placement_records(ctx)

        assert ctx["failed_node"] == "node-z"
        assert ctx["state"] == "NODE_FAIL"


class TestPlacementReader:
    def test_disabled_records_do_not_query_reader(self, mocker):
        reader = mocker.Mock()
        eca = KueueExecutionContextAdapter(placement_reader=reader)

        assert eca._fetch_placement_records({"namespace": "ns", "workload_uid": "wl-uid"}) == []
        reader.list_for_workload.assert_not_called()

    def test_enabled_records_query_reader(self, mocker):
        reader = mocker.Mock()
        reader.list_for_workload.return_value = []
        eca = KueueExecutionContextAdapter(placement_reader=reader, placement_records_enabled=True)

        assert eca._fetch_placement_records({"namespace": "ns", "workload_uid": "wl-uid"}) == []
        reader.list_for_workload.assert_called_once_with("ns", "wl-uid")

    def test_enabled_records_propagate_reader_outage(self, mocker):
        reader = mocker.Mock()
        reader.list_for_workload.side_effect = OSError("unavailable")
        eca = KueueExecutionContextAdapter(placement_reader=reader, placement_records_enabled=True)

        with pytest.raises(RuntimeError, match="placement recorder lookup failed"):
            eca._fetch_placement_records({"namespace": "ns", "workload_uid": "wl-uid"})

    def test_enabled_records_require_reader(self):
        eca = KueueExecutionContextAdapter(placement_records_enabled=True)

        with pytest.raises(RuntimeError, match="no recorder client"):
            eca._fetch_placement_records({"namespace": "ns", "workload_uid": "wl-uid"})
