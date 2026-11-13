# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from attribution_blueprint import mca_registry
from attribution_blueprint.connectors.execution_contexts.kai import KaiExecutionContextAdapter
from attribution_blueprint.connectors.execution_contexts.kai.connector import _resolve_owner
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.kubernetes.client import KubernetesClientError
from attribution_blueprint.kubernetes.daemon.models import PlacementRecord

FIXTURES = Path(__file__).parent.parent.parent.parent / "fixtures"
PODGROUPS = json.loads((FIXTURES / "kai_podgroups.json").read_text())
PODS = json.loads((FIXTURES / "kai_pods.json").read_text())
JOBS = json.loads((FIXTURES / "kai_jobs.json").read_text())


_DEFAULT_OWNER_GROUPS = {
    "JobSet": "jobset.x-k8s.io",
    "LeaderWorkerSet": "leaderworkerset.x-k8s.io",
    "PyTorchJob": "kubeflow.org",
    "RayJob": "ray.io",
    "Deployment": "apps",
    "DynamoGraphDeployment": "nvidia.com",
}

_DEFAULT_OWNER_VERSIONS = {
    "JobSet": "v1alpha2",
    "DynamoGraphDeployment": "v1alpha1",
}


def _podgroup(
    name="pg-a",
    namespace="team-alpha",
    uid="uid-pg-a",
    created="2026-05-01T12:00:00Z",
    owner=True,
    owner_kind="Job",
    owner_name=None,
    owner_group=None,
    owner_version=None,
    omit_group=False,
    spec=None,
    status=None,
):
    meta = {"name": name, "namespace": namespace, "uid": uid, "creationTimestamp": created}
    if owner:
        if owner_name is None:
            owner_name = f"{name}-job"
        if not omit_group and owner_group is None:
            owner_group = _DEFAULT_OWNER_GROUPS.get(owner_kind)
        if owner_version is None:
            owner_version = _DEFAULT_OWNER_VERSIONS.get(owner_kind, "v1")
        group = f"group: {owner_group}\n" if owner_group else ""
        meta["annotations"] = {
            "kai.scheduler/top-owner-metadata": (
                f"name: {owner_name}\nuid: uid-job-{uid}\nkind: {owner_kind}\n{group}version: {owner_version}\n"
            )
        }
    pg = {
        "apiVersion": "scheduling.run.ai/v2alpha2",
        "kind": "PodGroup",
        "metadata": meta,
        "spec": spec if spec is not None else {"queue": "q"},
    }
    if status is not None:
        pg["status"] = status
    return pg


def _pod(
    name,
    podgroup="pg-a",
    namespace="team-alpha",
    node="node001",
    created="2026-05-01T12:00:10Z",
    podgroup_assigned=None,
    phase="Running",
    finished=None,
    controller=None,
    top_owner=None,
):
    meta = {"name": name, "namespace": namespace, "creationTimestamp": created}
    if controller is not None:
        kind, controller_uid = controller
        meta["ownerReferences"] = [{"kind": kind, "name": "c", "uid": controller_uid, "controller": True}]
    if podgroup is not None:
        annotations = {"pod-group-name": podgroup}
        if top_owner is not None:
            top_kind, top_uid = top_owner
            annotations["kai.scheduler/top-owner-metadata"] = (
                f"name: c\nuid: {top_uid}\nkind: {top_kind}\nversion: v1\n"
            )
        meta["annotations"] = annotations
        meta["managedFields"] = [
            {
                "manager": "pod-grouper",
                "operation": "Update",
                "time": podgroup_assigned or created,
                "fieldsV1": {
                    "f:metadata": {
                        "f:annotations": {
                            "f:pod-group-name": {},
                        },
                    },
                },
            },
        ]
    status = {"phase": phase}
    if finished:
        status["containerStatuses"] = [{"name": "c", "state": {"terminated": {"finishedAt": finished}}}]
    spec = {"nodeName": node} if node else {}
    return {"metadata": meta, "spec": spec, "status": status}


def _job(
    name,
    uid,
    completed=None,
    failed_at=None,
    namespace="team-alpha",
    reason=None,
    suspended=False,
    jobset=None,
    deadline_seconds=None,
):
    status = {}
    if completed:
        status["completionTime"] = completed
    if failed_at:
        failed = {"type": "Failed", "status": "True", "lastTransitionTime": failed_at}
        if reason:
            failed["reason"] = reason
        status["conditions"] = [failed]
    if suspended:
        status["conditions"] = [{"type": "Suspended", "status": "True"}]
    metadata = {"name": name, "namespace": namespace, "uid": uid}
    if jobset is not None:
        jobset_name, jobset_uid = jobset
        metadata["labels"] = {"jobset.sigs.k8s.io/jobset-name": jobset_name}
        metadata["ownerReferences"] = [{"kind": "JobSet", "name": jobset_name, "uid": jobset_uid, "controller": True}]
    spec = {"activeDeadlineSeconds": deadline_seconds} if deadline_seconds is not None else {}
    return {"metadata": metadata, "spec": spec, "status": status}


def _jobset(name, uid, completed_at=None, failed_at=None, namespace="team-alpha"):
    status = {}
    if completed_at or failed_at:
        condition_type, transition = ("Completed", completed_at) if completed_at else ("Failed", failed_at)
        status = {
            "terminalState": condition_type,
            "conditions": [
                {
                    "type": condition_type,
                    "status": "True",
                    "lastTransitionTime": transition,
                }
            ],
        }
    return {"metadata": {"name": name, "namespace": namespace, "uid": uid}, "status": status}


def _pytorchjob(
    name,
    uid,
    completed=None,
    succeeded_at=None,
    failed_at=None,
    *,
    succeeded=False,
    failed=False,
    reason=None,
    message=None,
    suspended=False,
    namespace="team-alpha",
):
    status = {}
    if completed:
        status["completionTime"] = completed
    conditions = []
    if succeeded_at or succeeded:
        cond = {"type": "Succeeded", "status": "True"}
        if succeeded_at:
            cond["lastTransitionTime"] = succeeded_at
        conditions.append(cond)
    if failed_at or failed:
        failed_cond = {"type": "Failed", "status": "True"}
        if failed_at:
            failed_cond["lastTransitionTime"] = failed_at
        if reason:
            failed_cond["reason"] = reason
        if message:
            failed_cond["message"] = message
        conditions.append(failed_cond)
    if suspended:
        conditions.append({"type": "Suspended", "status": "True"})
    if conditions:
        status["conditions"] = conditions
    return {"metadata": {"name": name, "namespace": namespace, "uid": uid}, "status": status}


def _rayjob(
    name,
    uid,
    job_status=None,
    deployment_status=None,
    end_time=None,
    ray_job_info=None,
    ray_cluster_status=None,
    reason=None,
    conditions=None,
    namespace="team-alpha",
):
    status = {}
    if job_status is not None:
        status["jobStatus"] = job_status
    if deployment_status is not None:
        status["jobDeploymentStatus"] = deployment_status
    if end_time is not None:
        status["endTime"] = end_time
    if ray_job_info is not None:
        status["rayJobInfo"] = ray_job_info
    if ray_cluster_status is not None:
        status["rayClusterStatus"] = ray_cluster_status
    if reason is not None:
        status["reason"] = reason
    if conditions is not None:
        status["conditions"] = conditions
    return {"metadata": {"name": name, "namespace": namespace, "uid": uid}, "status": status}


def _lws(name, uid, conditions=None, namespace="team-alpha"):
    status = {"conditions": conditions} if conditions is not None else {}
    return {"metadata": {"name": name, "namespace": namespace, "uid": uid}, "status": status}


def _deployment(name, uid, conditions=None, namespace="team-alpha"):
    status = {"conditions": conditions} if conditions is not None else {}
    return {"metadata": {"name": name, "namespace": namespace, "uid": uid}, "status": status}


def _dynamo(name, uid, state=None, conditions=None, reason=None, end_time=None, namespace="team-alpha"):
    status = {}
    if state is not None:
        status["state"] = state
    if conditions is not None:
        status["conditions"] = conditions
    if reason is not None:
        status["reason"] = reason
    if end_time is not None:
        status["endTime"] = end_time
    return {"metadata": {"name": name, "namespace": namespace, "uid": uid}, "status": status}


class _FakeKubernetesClient:
    def __init__(
        self,
        *,
        podgroups=None,
        pods=None,
        jobs=None,
        jobsets=None,
        leaderworkersets=None,
        pytorchjobs=None,
        rayjobs=None,
        deployments=None,
        dynamographdeployments=None,
    ):
        self.podgroups = list(podgroups or [])
        self.pods = list(pods or [])
        if jobs is None:
            jobs = [
                _job(owner["name"], owner["uid"], namespace=pg["metadata"]["namespace"])
                for pg in self.podgroups
                if (owner := _resolve_owner(pg["metadata"])) is not None and owner["kind"] == "Job"
            ]
        self.jobs = list(jobs)
        if jobsets is None:
            jobsets = [
                _jobset(owner["name"], owner["uid"], namespace=pg["metadata"]["namespace"])
                for pg in self.podgroups
                if (owner := _resolve_owner(pg["metadata"])) is not None and owner["kind"] == "JobSet"
            ]
        self.jobsets = list(jobsets)
        if leaderworkersets is None:
            leaderworkersets = [
                _lws(owner["name"], owner["uid"], namespace=pg["metadata"]["namespace"])
                for pg in self.podgroups
                if (owner := _resolve_owner(pg["metadata"])) is not None and owner["kind"] == "LeaderWorkerSet"
            ]
        self.leaderworkersets = list(leaderworkersets)
        if pytorchjobs is None:
            pytorchjobs = [
                _pytorchjob(owner["name"], owner["uid"], namespace=pg["metadata"]["namespace"])
                for pg in self.podgroups
                if (owner := _resolve_owner(pg["metadata"])) is not None and owner["kind"] == "PyTorchJob"
            ]
        self.pytorchjobs = list(pytorchjobs)
        if rayjobs is None:
            rayjobs = [
                _rayjob(owner["name"], owner["uid"], namespace=pg["metadata"]["namespace"])
                for pg in self.podgroups
                if (owner := _resolve_owner(pg["metadata"])) is not None and owner["kind"] == "RayJob"
            ]
        self.rayjobs = list(rayjobs)
        if deployments is None:
            deployments = [
                _deployment(owner["name"], owner["uid"], namespace=pg["metadata"]["namespace"])
                for pg in self.podgroups
                if (owner := _resolve_owner(pg["metadata"])) is not None and owner["kind"] == "Deployment"
            ]
        self.deployments = list(deployments)
        if dynamographdeployments is None:
            dynamographdeployments = [
                _dynamo(owner["name"], owner["uid"], namespace=pg["metadata"]["namespace"])
                for pg in self.podgroups
                if (owner := _resolve_owner(pg["metadata"])) is not None and owner["kind"] == "DynamoGraphDeployment"
            ]
        self.dynamographdeployments = list(dynamographdeployments)
        self.calls: list[dict] = []
        self.failures: dict[str, Exception] = {}
        self.versioned_failures: dict[tuple[str, str | None], Exception] = {}

    def _items(self, kind):
        return {
            "PodGroup": self.podgroups,
            "Pod": self.pods,
            "Job": self.jobs,
            "JobSet": self.jobsets,
            "LeaderWorkerSet": self.leaderworkersets,
            "PyTorchJob": self.pytorchjobs,
            "RayJob": self.rayjobs,
            "Deployment": self.deployments,
            "DynamoGraphDeployment": self.dynamographdeployments,
        }[kind]

    def _selected(self, kind, namespace, label_selector=None):
        selected = [
            item
            for item in self._items(kind)
            if namespace is None or (item.get("metadata") or {}).get("namespace") == namespace
        ]
        if label_selector:
            key, value = label_selector.split("=", 1)
            selected = [
                item for item in selected if ((item.get("metadata") or {}).get("labels") or {}).get(key) == value
            ]
        return selected

    def get_one(self, kind, name, *, namespace, group=None, api_version=None):
        self.calls.append(
            {
                "op": "get_one",
                "kind": kind,
                "name": name,
                "namespace": namespace,
                "group": group,
                "api_version": api_version,
            }
        )
        if (kind, api_version) in self.versioned_failures:
            raise self.versioned_failures[(kind, api_version)]
        if kind in self.failures:
            raise self.failures[kind]
        obj = next(
            (item for item in self._selected(kind, namespace) if (item.get("metadata") or {}).get("name") == name),
            None,
        )
        if obj is None:
            raise KubernetesClientError(f"{kind} {name!r} not found in {namespace!r}", status=404)
        return obj

    def list(
        self,
        kind,
        *,
        namespace,
        field_selector=None,
        label_selector=None,
        group=None,
        api_version=None,
    ):
        self.calls.append(
            {
                "op": "list",
                "kind": kind,
                "namespace": namespace,
                "field_selector": field_selector,
                "label_selector": label_selector,
                "group": group,
                "api_version": api_version,
            }
        )
        if kind in self.failures:
            raise self.failures[kind]
        return self._selected(kind, namespace, label_selector)


def _wire_kubectl(
    mocker,
    *,
    podgroups=None,
    pods=None,
    jobs=None,
    jobsets=None,
    leaderworkersets=None,
    pytorchjobs=None,
    rayjobs=None,
    deployments=None,
    dynamographdeployments=None,
):
    fake = _FakeKubernetesClient(
        podgroups=podgroups,
        pods=pods,
        jobs=jobs,
        jobsets=jobsets,
        leaderworkersets=leaderworkersets,
        pytorchjobs=pytorchjobs,
        rayjobs=rayjobs,
        deployments=deployments,
        dynamographdeployments=dynamographdeployments,
    )
    mocker.patch(
        "attribution_blueprint.connectors.execution_contexts.kai.connector.KubernetesClient",
        return_value=fake,
    )
    return fake


@pytest.fixture
def warnings(mocker):
    return mocker.patch("attribution_blueprint.connectors.execution_contexts.kai.connector.logger.warning")


def _logged(warning) -> str:
    return "\n".join(str(call.args[0]) for call in warning.call_args_list)


START = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
END = datetime(2026, 5, 2, 0, 0, tzinfo=timezone.utc)


def test_current_crd_shape_emits_incarnation_keyed_context(mocker, warnings):
    _wire_kubectl(
        mocker,
        podgroups=PODGROUPS["items"],
        pods=PODS["items"],
        jobs=JOBS["items"],
    )

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-train-mnist-1a2b3c4d"])

    assert ctx["id"] == "uid-pg-1001"
    assert ctx["attempt_id"] == "uid-pg-1001"
    assert ctx["pod_group_uid"] == "uid-pg-1001"
    assert ctx["job_id"] == "team-alpha/pg-train-mnist-1a2b3c4d"
    assert ctx["workload_type"] == "kai"
    assert ctx["scheduler"] == "kai"
    assert ctx["queue_name"] == "research-queue"
    assert ctx["min_member"] == 2
    assert ctx["start_time"] == datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    # The owning Job's completion closes the window.
    assert ctx["end_time"] == datetime(2026, 5, 1, 12, 32, tzinfo=timezone.utc)
    warnings.assert_not_called()
    assert ctx["nodes"] == ["node005", "node006"]
    assert ctx["pod_names"] == ["train-mnist-0", "train-mnist-1"]
    assert ctx["pod_nodes"] == {"train-mnist-0": "node005", "train-mnist-1": "node006"}


def test_context_does_not_claim_a_failed_node(mocker):
    _wire_kubectl(mocker, podgroups=PODGROUPS["items"], pods=PODS["items"])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-train-mnist-1a2b3c4d"])

    assert "failed_node" not in ctx


@pytest.mark.parametrize(
    "owner, state, owner_state",
    [
        (_job("pg-a-job", "uid-job-uid-pg-a", completed="2026-05-01T13:00:00Z"), "COMPLETED", "Complete"),
        (_job("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z"), "FAILED", "Failed"),
        (
            _job("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z", reason="BackoffLimitExceeded"),
            "FAILED",
            "Failed:BackoffLimitExceeded",
        ),
        (
            _job("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z", reason="DeadlineExceeded"),
            "TIMEOUT",
            "Failed:DeadlineExceeded",
        ),
        (_job("pg-a-job", "uid-job-uid-pg-a"), "RUNNING", "Active"),
        (_job("pg-a-job", "uid-job-uid-pg-a", suspended=True), "RUNNING", "Suspended"),
    ],
    ids=["completed", "failed", "backoff-limit", "deadline", "active", "suspended"],
)
def test_job_owner_state(mocker, owner, state, owner_state):
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[], jobs=[owner])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["state"] == state
    assert ctx["owner_state"] == owner_state


def test_job_deadline_is_forwarded_in_minutes(mocker):
    owner = _job(
        "pg-a-job",
        "uid-job-uid-pg-a",
        failed_at="2026-05-01T12:45:00Z",
        reason="DeadlineExceeded",
        deadline_seconds=30,
    )
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[], jobs=[owner])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["state"] == "TIMEOUT"
    assert ctx["time_limit"] == 1


@pytest.mark.parametrize(
    "owner, state, owner_state",
    [
        (_jobset("pg-a-job", "uid-job-uid-pg-a", completed_at="2026-05-01T13:00:00Z"), "COMPLETED", "Completed"),
        (_jobset("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z"), "FAILED", "Failed"),
        (_jobset("pg-a-job", "uid-job-uid-pg-a"), "RUNNING", "Active"),
    ],
    ids=["completed", "failed", "active"],
)
def test_jobset_owner_state(mocker, owner, state, owner_state):
    _wire_kubectl(mocker, podgroups=[_podgroup(owner_kind="JobSet")], pods=[], jobsets=[owner])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["state"] == state
    assert ctx["owner_state"] == owner_state


def test_failed_jobset_reports_the_failed_reason(mocker):
    jobset = _jobset("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z")
    jobset["status"]["conditions"][0]["reason"] = "ReachedMaxRestarts"
    _wire_kubectl(mocker, podgroups=[_podgroup(owner_kind="JobSet")], pods=[], jobsets=[jobset])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["state"] == "FAILED"
    assert ctx["owner_state"] == "Failed:ReachedMaxRestarts"


def test_jobset_uses_verified_child_deadline_and_failure_reason(mocker):
    jobset = _jobset("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z")
    jobset["status"]["conditions"][0]["reason"] = "ReachedMaxRestarts"
    children = [
        _job(
            "worker-0",
            "uid-worker-0",
            failed_at="2026-05-01T12:40:00Z",
            reason="DeadlineExceeded",
            jobset=("pg-a-job", "uid-job-uid-pg-a"),
            deadline_seconds=90,
        ),
        _job(
            "worker-1",
            "uid-worker-1",
            jobset=("pg-a-job", "uid-job-uid-pg-a"),
            deadline_seconds=300,
        ),
        _job(
            "stale-worker",
            "uid-stale",
            failed_at="2026-05-01T12:30:00Z",
            reason="BackoffLimitExceeded",
            jobset=("pg-a-job", "old-jobset-uid"),
            deadline_seconds=30,
        ),
    ]
    _wire_kubectl(
        mocker,
        podgroups=[_podgroup(owner_kind="JobSet")],
        pods=[],
        jobs=children,
        jobsets=[jobset],
    )

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["state"] == "TIMEOUT"
    assert ctx["owner_state"] == "Failed:ReachedMaxRestarts"
    assert ctx["time_limit"] == 2


def _lost_pod(name, node):
    pod = _pod(name, node=node, controller=("Job", "uid-job-uid-pg-a"))
    pod["status"]["reason"] = "NodeLost"
    return pod


@pytest.mark.parametrize(
    "owner, state",
    [
        (_job("pg-a-job", "uid-job-uid-pg-a"), "NODE_FAIL"),
        (_job("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z"), "NODE_FAIL"),
        (_job("pg-a-job", "uid-job-uid-pg-a", completed="2026-05-01T13:00:00Z"), "COMPLETED"),
    ],
    ids=["active", "failed", "completed"],
)
def test_node_loss_on_a_member_pod_sets_node_fail(mocker, owner, state):
    pods = [_lost_pod("a-0", "node002"), _lost_pod("a-1", "node001")]
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=pods, jobs=[owner])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["state"] == state
    if state == "NODE_FAIL":
        assert ctx["failed_node"] == "node001"
    else:
        assert "failed_node" not in ctx


def test_healthy_pods_leave_no_failed_node(mocker):
    pod = _pod("a-0", controller=("Job", "uid-job-uid-pg-a"))
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[pod], jobs=[_job("pg-a-job", "uid-job-uid-pg-a")])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert "failed_node" not in ctx


def test_podgc_disruption_sets_node_fail(mocker):
    pod = _pod("a-0", controller=("Job", "uid-job-uid-pg-a"))
    pod["status"]["conditions"] = [
        {
            "type": "DisruptionTarget",
            "status": "True",
            "reason": "DeletionByPodGC",
            "lastTransitionTime": "2026-05-01T12:30:00Z",
        }
    ]
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[pod], jobs=[_job("pg-a-job", "uid-job-uid-pg-a")])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["state"] == "NODE_FAIL"
    assert ctx["failed_node"] == "node001"


def test_node_loss_on_a_stale_jobset_pod_is_ignored(mocker):
    pg = _podgroup(owner_kind="JobSet")
    jobs = [
        _job("current", "uid-current", jobset=("pg-a-job", "uid-job-uid-pg-a")),
        _job("stale", "uid-stale", jobset=("pg-a-job", "uid-old-jobset")),
    ]
    stale = _pod("stale-0", node="node009", controller=("Job", "uid-stale"))
    stale["status"]["reason"] = "NodeLost"
    pods = [_pod("current-0", controller=("Job", "uid-current")), stale]
    _wire_kubectl(mocker, podgroups=[pg], pods=pods, jobs=jobs)

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["state"] == "RUNNING"
    assert "failed_node" not in ctx


def test_surviving_pods_do_not_change_the_owner_state(mocker):
    pod = _pod("a-0", controller=("Job", "uid-job-uid-pg-a"))
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[pod], jobs=[_job("pg-a-job", "uid-job-uid-pg-a")])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["state"] == "RUNNING"


def test_current_status_is_preserved_raw(mocker):
    _wire_kubectl(mocker, podgroups=PODGROUPS["items"], pods=PODS["items"])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-train-mnist-1a2b3c4d"])

    assert ctx["scheduling_conditions"][0]["type"] == "Unschedulable"
    assert ctx["resources_status"] == {"allocated": {"nvidia.com/gpu": "2"}}


def test_removed_status_fields_are_ignored(mocker):
    old = _podgroup(status={"phase": "Running", "running": 2, "failed": 0})
    _wire_kubectl(mocker, podgroups=[old], pods=[])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["state"] == "RUNNING"
    assert "scheduling_conditions" not in ctx
    assert "resources_status" not in ctx


def test_owner_comes_from_top_owner_annotation(mocker):
    _wire_kubectl(mocker, podgroups=PODGROUPS["items"], pods=PODS["items"])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-train-mnist-1a2b3c4d"])

    assert (ctx["owner_kind"], ctx["owner_name"], ctx["owner_uid"]) == ("Job", "train-mnist", "uid-job-1001")
    assert ctx["podgroup_shape"] == "owned"


@pytest.mark.parametrize(
    "api_version, kind, expected_group, expected_api_version",
    [
        ("batch/v1", "Job", None, "batch/v1"),
        ("jobset.x-k8s.io/v1alpha2", "JobSet", "jobset.x-k8s.io", None),
        ("leaderworkerset.x-k8s.io/v1", "LeaderWorkerSet", "leaderworkerset.x-k8s.io", "v1"),
        ("kubeflow.org/v1", "PyTorchJob", "kubeflow.org", "v1"),
        ("ray.io/v1", "RayJob", "ray.io", "v1"),
        ("ray.io/v1alpha1", "RayJob", "ray.io", "v1alpha1"),
        ("apps/v1", "Deployment", None, "apps/v1"),
        ("nvidia.com/v1alpha1", "DynamoGraphDeployment", "nvidia.com", "v1alpha1"),
    ],
)
def test_owner_falls_back_to_controller_owner_reference(
    mocker,
    api_version,
    kind,
    expected_group,
    expected_api_version,
):
    pg = _podgroup(owner=False)
    pg["metadata"]["ownerReferences"] = [
        {"apiVersion": api_version, "kind": kind, "name": "owner", "uid": "uid-owner", "controller": True},
    ]
    fake = _wire_kubectl(mocker, podgroups=[pg], pods=[])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert (ctx["owner_kind"], ctx["owner_name"], ctx["owner_uid"]) == (kind, "owner", "uid-owner")
    assert ctx["podgroup_shape"] == "owned"
    assert any(
        call["op"] == "get_one"
        and call["kind"] == kind
        and call["group"] == expected_group
        and call["api_version"] == expected_api_version
        for call in fake.calls
    )


@pytest.mark.parametrize(
    "kind",
    [
        "Job",
        "JobSet",
        "LeaderWorkerSet",
        "PyTorchJob",
        "RayJob",
        "Deployment",
        "DynamoGraphDeployment",
    ],
)
def test_top_owner_without_group_is_accepted_for_supported_kinds(mocker, kind):
    pg = _podgroup(owner_kind=kind, omit_group=True)
    fake = _wire_kubectl(mocker, podgroups=[pg], pods=[])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["owner_kind"] == kind
    assert ctx["owner_name"] == "pg-a-job"
    assert ctx["owner_uid"] == "uid-job-uid-pg-a"
    expected_version = _DEFAULT_OWNER_VERSIONS.get(kind, "v1")
    if kind in {"LeaderWorkerSet", "PyTorchJob", "RayJob", "DynamoGraphDeployment"}:
        assert any(
            call["op"] == "get_one" and call["kind"] == kind and call["api_version"] == expected_version
            for call in fake.calls
        )


def test_top_owner_custom_version_is_used_for_group_lookup(mocker):
    pg = _podgroup(owner_kind="RayJob", owner_version="v1alpha1")
    fake = _wire_kubectl(mocker, podgroups=[pg], pods=[])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["owner_kind"] == "RayJob"
    assert any(
        call["op"] == "get_one"
        and call["kind"] == "RayJob"
        and call["group"] == "ray.io"
        and call["api_version"] == "v1alpha1"
        for call in fake.calls
    )


def test_unserved_owner_version_falls_back_to_group_lookup(mocker):
    pg = _podgroup(owner_kind="RayJob", owner_version="v1alpha1")
    fake = _wire_kubectl(mocker, podgroups=[pg], pods=[])
    fake.versioned_failures[("RayJob", "v1alpha1")] = KubernetesClientError(
        "Kubernetes resource discovery failed for 'RayJob': ResourceNotFoundError: No matches found"
    )

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["owner_kind"] == "RayJob"
    assert [
        (call["group"], call["api_version"])
        for call in fake.calls
        if call["op"] == "get_one" and call["kind"] == "RayJob"
    ] == [
        ("ray.io", "v1alpha1"),
        ("ray.io", None),
    ]


def test_malformed_top_owner_annotation_leaves_the_podgroup_unowned(mocker):
    pg = _podgroup(owner=False)
    pg["metadata"]["annotations"] = {"kai.scheduler/top-owner-metadata": "{not: [valid"}
    _wire_kubectl(mocker, podgroups=[pg], pods=[])

    assert KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"]) == []


def test_hierarchical_podgroup_is_emitted_and_flagged(mocker):
    pg = _podgroup(
        owner_kind="JobSet",
        spec={"queue": "q", "subGroups": [{"name": "decode", "minMember": 1}]},
    )
    _wire_kubectl(mocker, podgroups=[pg], pods=[])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["podgroup_shape"] == "hierarchical"


def test_recreated_podgroup_under_unchanged_job_is_a_known_limitation(mocker):
    """Documents an accepted gap: Kubernetes exposes no reliable signal to
    tell a leftover Pod from an earlier, same-named PodGroup incarnation
    apart from a genuine new member when the owning Job is unchanged (see
    `_member_pods`). Both pods share the live owner Job UID, so both are
    treated as members; this is intentional, not a regression to "fix" with
    an unreliable timestamp guard."""
    new_pg = _podgroup(uid="uid-pg-new", created="2026-05-01T14:00:00Z")
    owner_uid = "uid-job-uid-pg-new"
    pods = [
        _pod(
            "old-0",
            created="2026-05-01T12:00:00Z",
            node="node-old",
            phase="Failed",
            controller=("Job", owner_uid),
        ),
        _pod(
            "new-0",
            created="2026-05-01T14:00:01Z",
            node="node-new",
            controller=("Job", owner_uid),
        ),
    ]
    _wire_kubectl(mocker, podgroups=[new_pg], pods=pods)

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["id"] == "uid-pg-new"
    assert ctx["pod_names"] == ["new-0", "old-0"]
    assert ctx["nodes"] == ["node-new", "node-old"]


def test_pod_created_before_its_podgroup_is_a_member(mocker):
    """KAI can create a PodGroup after submitting its Pods, so a Pod
    predating its PodGroup is the normal case, not stale-incarnation
    evidence; membership must not exclude it."""
    pg = _podgroup(created="2026-05-01T12:00:00Z")
    pod = _pod(
        "early-0",
        created="2026-05-01T11:59:00Z",
        node="node-early",
        controller=("Job", "uid-job-uid-pg-a"),
    )
    _wire_kubectl(mocker, podgroups=[pg], pods=[pod])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == ["early-0"]
    assert ctx["nodes"] == ["node-early"]
    assert ctx["start_time"] == datetime(2026, 5, 1, 11, 59, tzinfo=timezone.utc)


def test_pod_membership_does_not_require_managed_fields(mocker):
    pg = _podgroup()
    pod = _pod("ambiguous-0", controller=("Job", "uid-job-uid-pg-a"))
    del pod["metadata"]["managedFields"]
    _wire_kubectl(mocker, podgroups=[pg], pods=[pod])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == ["ambiguous-0"]
    assert ctx["nodes"] == ["node001"]


def test_time_range_uses_a_member_pod_that_predates_its_podgroup(mocker):
    pg = _podgroup(created="2026-05-01T12:00:00Z")
    pod = _pod(
        "early-0",
        created="2026-05-01T11:59:00Z",
        controller=("Job", "uid-job-uid-pg-a"),
    )
    _wire_kubectl(mocker, podgroups=[pg], pods=[pod])

    contexts = KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(
        datetime(2026, 5, 1, 11, 58, tzinfo=timezone.utc),
        datetime(2026, 5, 1, 11, 59, 30, tzinfo=timezone.utc),
    )

    assert [ctx["id"] for ctx in contexts] == ["uid-pg-a"]


def test_pod_with_a_non_job_controller_is_not_joined(mocker):
    pod = _pod("rs-0", node="node-rs", controller=("ReplicaSet", "uid-rs"))
    pg = _podgroup()
    _wire_kubectl(mocker, podgroups=[pg], pods=[pod])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert "pod_names" not in ctx


def test_pod_without_controller_is_not_joined(mocker):
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[_pod("bare-0")])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert "pod_names" not in ctx


def test_same_name_in_another_namespace_is_not_joined(mocker):
    pods = [
        _pod("mine-0", namespace="team-alpha", node="node-a", controller=("Job", "uid-job-uid-pg-a")),
        _pod("theirs-0", namespace="team-beta", node="node-b", controller=("Job", "uid-job-uid-pg-a")),
    ]
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=pods)

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == ["mine-0"]
    assert ctx["nodes"] == ["node-a"]


def test_missing_podgroup_is_skipped(mocker):
    _wire_kubectl(mocker, podgroups=[], pods=[])

    assert KaiExecutionContextAdapter().discover_by_ids(["team-alpha/gone"]) == []


def test_podgroup_deleted_between_list_and_lookup_is_skipped(mocker):
    """The listing saw the PodGroup but a later lookup finds nothing."""
    pg = _podgroup()
    fake = _wire_kubectl(mocker, podgroups=[pg], pods=[])
    adapter = KaiExecutionContextAdapter(namespace="team-alpha")
    assert len(adapter.discover_by_time_range(START, END)) == 1

    fake.podgroups = []

    assert adapter.discover_by_ids(["team-alpha/pg-a"]) == []


def test_malformed_podgroups_are_skipped(mocker):
    no_uid = _podgroup(uid="")
    no_created = _podgroup(name="pg-b", uid="uid-b", created="")
    ok = _podgroup(name="pg-c", uid="uid-c")
    _wire_kubectl(mocker, podgroups=[no_uid, no_created, ok], pods=[])

    contexts = KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)

    assert [c["id"] for c in contexts] == ["uid-c"]


def test_active_job_window_ends_now_without_a_warning(mocker, warnings):
    pod = _pod("a-0", controller=("Job", "uid-job-uid-pg-a"))
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[pod], jobs=[_job("pg-a-job", "uid-job-uid-pg-a")])

    before = datetime.now(tz=timezone.utc)
    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["end_time"] >= before
    warnings.assert_not_called()


def test_completed_job_end_survives_pod_deletion(mocker, warnings):
    job = _job("pg-a-job", "uid-job-uid-pg-a", completed="2026-05-01T13:00:00Z")
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[], jobs=[job])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["end_time"] == datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc)
    assert ctx["nodes"] == []
    assert "pod_names" not in ctx
    assert "node list unavailable" in _logged(warnings)


def test_failed_job_end_is_the_failed_condition_time(mocker, warnings):
    job = _job("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z")
    pod = _pod("a-0", controller=("Job", "uid-job-uid-pg-a"))
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[pod], jobs=[job])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["end_time"] == datetime(2026, 5, 1, 12, 45, tzinfo=timezone.utc)
    warnings.assert_not_called()


@pytest.mark.parametrize(
    "owner, expected_end",
    [
        (_jobset("pg-a-job", "uid-job-uid-pg-a", completed_at="2026-05-01T13:00:00Z"), "2026-05-01T13:00:00Z"),
        (_jobset("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z"), "2026-05-01T12:45:00Z"),
    ],
    ids=["completed", "failed"],
)
def test_terminal_jobset_condition_supplies_end_time(mocker, owner, expected_end):
    pg = _podgroup(owner_kind="JobSet")
    _wire_kubectl(mocker, podgroups=[pg], pods=[], jobsets=[owner])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["owner_kind"] == "JobSet"
    assert ctx["end_time"] == datetime.fromisoformat(expected_end)


def test_active_jobset_window_ends_now(mocker):
    pg = _podgroup(owner_kind="JobSet")
    _wire_kubectl(mocker, podgroups=[pg], pods=[])

    before = datetime.now(tz=timezone.utc)
    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["end_time"] >= before


def test_jobset_joins_only_pods_from_current_child_jobs(mocker):
    pg = _podgroup(owner_kind="JobSet")
    jobset_name = "pg-a-job"
    jobset_uid = "uid-job-uid-pg-a"
    jobs = [
        _job("current", "uid-current", jobset=(jobset_name, jobset_uid)),
        _job("stale", "uid-stale", jobset=(jobset_name, "uid-old-jobset")),
    ]
    pods = [
        _pod("current-0", controller=("Job", "uid-current")),
        _pod("stale-0", controller=("Job", "uid-stale")),
    ]
    fake = _wire_kubectl(mocker, podgroups=[pg], pods=pods, jobs=jobs)

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == ["current-0"]
    assert any(call["kind"] == "JobSet" and call["group"] == "jobset.x-k8s.io" for call in fake.calls)
    assert any(call["kind"] == "Job" and call["api_version"] == "batch/v1" for call in fake.calls)
    assert any(call["kind"] == "Pod" and call["api_version"] == "v1" for call in fake.calls)


def test_member_without_a_node_contributes_no_node(mocker):
    job = _job("pg-a-job", "uid-job-uid-pg-a", completed="2026-05-01T13:00:00Z")
    pods = [
        _pod("a-0", node="node001", controller=("Job", "uid-job-uid-pg-a")),
        _pod("a-1", node=None, controller=("Job", "uid-job-uid-pg-a")),
    ]
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=pods, jobs=[job])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == ["a-0", "a-1"]
    assert ctx["nodes"] == ["node001"]


def test_surviving_terminal_pods_do_not_prove_completion(mocker):
    pod = _pod(
        "a-0",
        phase="Succeeded",
        finished="2026-05-01T12:30:00Z",
        controller=("Job", "uid-job-uid-pg-a"),
    )
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[pod], jobs=[_job("pg-a-job", "uid-job-uid-pg-a")])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["end_time"] > datetime(2026, 5, 2, tzinfo=timezone.utc)


def _non_batch_job_podgroup():
    pg = _podgroup()
    pg["metadata"]["annotations"]["kai.scheduler/top-owner-metadata"] += "group: example.com\n"
    return pg


@pytest.mark.parametrize(
    "podgroup, jobs, reason",
    [
        (_podgroup(owner=False), [], "not owned by a supported workload"),
        (_podgroup(owner_kind="CronJob"), [], "not owned by a supported workload"),
        (_podgroup(owner_kind="JobSet", owner_group="example.com"), [], "not owned by a supported workload"),
        (_podgroup(owner_kind="LeaderWorkerSet", owner_group="example.com"), [], "not owned by a supported workload"),
        (_podgroup(owner_kind="PyTorchJob", owner_group="example.com"), [], "not owned by a supported workload"),
        (_podgroup(owner_kind="RayJob", owner_group="example.com"), [], "not owned by a supported workload"),
        (_podgroup(owner_kind="Deployment", owner_group="example.com"), [], "not owned by a supported workload"),
        (
            _podgroup(owner_kind="DynamoGraphDeployment", owner_group="example.com"),
            [],
            "not owned by a supported workload",
        ),
        (_non_batch_job_podgroup(), [], "not owned by a supported workload"),
        (_podgroup(), [], "owner Job not found"),
        (
            _podgroup(),
            [_job("pg-a-job", "uid-other", completed="2026-05-01T13:00:00Z")],
            "owner Job uid mismatch",
        ),
    ],
    ids=[
        "no-owner",
        "unsupported-kind",
        "non-jobset-group",
        "non-lws-group",
        "non-pytorchjob-group",
        "non-rayjob-group",
        "non-deployment-group",
        "non-dynamo-group",
        "non-batch-job",
        "job-missing",
        "job-reused-name",
    ],
)
def test_podgroup_without_a_verified_owner_job_is_skipped(mocker, podgroup, jobs, reason):
    info = mocker.patch("attribution_blueprint.connectors.execution_contexts.kai.connector.logger.info")
    _wire_kubectl(mocker, podgroups=[podgroup], pods=[], jobs=jobs)

    assert KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"]) == []
    assert any(f"{reason}; skipping" in str(c.args[0]) for c in info.call_args_list)


def test_unsupported_podgroups_are_not_enriched(mocker):
    fake = _wire_kubectl(mocker, podgroups=[_podgroup(owner_kind="CronJob")], pods=[])

    KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)

    assert all(call["kind"] not in {"Pod", "Job"} for call in fake.calls[1:])


@pytest.mark.parametrize(
    "owner, state, owner_state, expected_end",
    [
        (
            _pytorchjob("pg-a-job", "uid-job-uid-pg-a", completed="2026-05-01T13:00:00Z"),
            "COMPLETED",
            "Succeeded",
            "2026-05-01T13:00:00Z",
        ),
        (
            _pytorchjob("pg-a-job", "uid-job-uid-pg-a", succeeded_at="2026-05-01T13:05:00Z"),
            "COMPLETED",
            "Succeeded",
            "2026-05-01T13:05:00Z",
        ),
        (
            _pytorchjob("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z"),
            "FAILED",
            "Failed",
            "2026-05-01T12:45:00Z",
        ),
        (
            _pytorchjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                failed_at="2026-05-01T12:45:00Z",
                reason="BackoffLimitExceeded",
            ),
            "FAILED",
            "Failed:BackoffLimitExceeded",
            "2026-05-01T12:45:00Z",
        ),
        (
            _pytorchjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                failed_at="2026-05-01T12:45:00Z",
                reason="PyTorchJobFailed",
                message="PyTorchJob pg-a-job has failed because it has reached the specified backoff limit",
            ),
            "FAILED",
            "Failed:PyTorchJobFailed",
            "2026-05-01T12:45:00Z",
        ),
        (
            _pytorchjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                failed_at="2026-05-01T12:45:00Z",
                reason="DeadlineExceeded",
            ),
            "TIMEOUT",
            "Failed:DeadlineExceeded",
            "2026-05-01T12:45:00Z",
        ),
        (
            _pytorchjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                failed_at="2026-05-01T12:45:00Z",
                reason="PyTorchJobFailed",
                message="PyTorchJob pg-a-job has failed because it was active longer than specified deadline",
            ),
            "TIMEOUT",
            "Failed:PyTorchJobFailed",
            "2026-05-01T12:45:00Z",
        ),
        (
            _pytorchjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                failed_at="2026-05-01T12:45:00Z",
                message="Job active deadline exceeded",
            ),
            "TIMEOUT",
            "Failed:DeadlineExceeded",
            "2026-05-01T12:45:00Z",
        ),
        (
            _pytorchjob("pg-a-job", "uid-job-uid-pg-a", succeeded=True),
            "COMPLETED",
            "Succeeded",
            None,
        ),
        (
            _pytorchjob("pg-a-job", "uid-job-uid-pg-a", failed=True, reason="BackoffLimitExceeded"),
            "FAILED",
            "Failed:BackoffLimitExceeded",
            None,
        ),
        (
            _pytorchjob("pg-a-job", "uid-job-uid-pg-a", failed=True, reason="DeadlineExceeded"),
            "TIMEOUT",
            "Failed:DeadlineExceeded",
            None,
        ),
        (_pytorchjob("pg-a-job", "uid-job-uid-pg-a"), "RUNNING", "Active", None),
        (_pytorchjob("pg-a-job", "uid-job-uid-pg-a", suspended=True), "RUNNING", "Suspended", None),
    ],
    ids=[
        "completed-time",
        "succeeded-condition",
        "failed",
        "failed-reason",
        "generic-reason-backoff-message",
        "deadline-exceeded",
        "generic-reason-deadline-message",
        "missing-reason-deadline-message",
        "untimed-succeeded-condition",
        "untimed-failed-condition",
        "untimed-deadline-condition",
        "active",
        "suspended",
    ],
)
def test_pytorchjob_owner_state_and_end_time(mocker, owner, state, owner_state, expected_end):
    pg = _podgroup(owner_kind="PyTorchJob")
    pods = [_pod("worker-0", node="node001", controller=("PyTorchJob", "uid-job-uid-pg-a"))]
    _wire_kubectl(mocker, podgroups=[pg], pods=pods, pytorchjobs=[owner])

    before = datetime.now(tz=timezone.utc)
    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["owner_kind"] == "PyTorchJob"
    assert ctx["state"] == state
    assert ctx["owner_state"] == owner_state
    assert ctx["nodes"] == ["node001"]
    if expected_end is not None:
        assert ctx["end_time"] == datetime.fromisoformat(expected_end)
    else:
        assert ctx["end_time"] >= before


@pytest.mark.parametrize(
    "owner, state, owner_state, expected_end",
    [
        (
            _rayjob("pg-a-job", "uid-job-uid-pg-a", job_status="SUCCEEDED", end_time="2026-05-01T13:00:00Z"),
            "COMPLETED",
            "Succeeded",
            "2026-05-01T13:00:00Z",
        ),
        (
            _rayjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                job_status="SUCCEEDED",
                deployment_status="Running",
                ray_job_info={"endTime": "2026-05-01T13:00:00Z"},
            ),
            "COMPLETED",
            "Succeeded",
            "2026-05-01T13:00:00Z",
        ),
        (
            _rayjob("pg-a-job", "uid-job-uid-pg-a", deployment_status="Complete", end_time="2026-05-01T13:00:00Z"),
            "COMPLETED",
            "Succeeded",
            "2026-05-01T13:00:00Z",
        ),
        (
            _rayjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                job_status="STOPPED",
                deployment_status="Complete",
                end_time="2026-05-01T12:40:00Z",
            ),
            "CANCELLED",
            "Stopped",
            "2026-05-01T12:40:00Z",
        ),
        (
            _rayjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                job_status="STOPPED",
                deployment_status="Complete",
                reason="JobStoppedByUser",
                end_time="2026-05-01T12:40:00Z",
            ),
            "CANCELLED",
            "Stopped:JobStoppedByUser",
            "2026-05-01T12:40:00Z",
        ),
        (
            _rayjob("pg-a-job", "uid-job-uid-pg-a", job_status="FAILED", end_time="2026-05-01T12:45:00Z"),
            "FAILED",
            "Failed",
            "2026-05-01T12:45:00Z",
        ),
        (
            _rayjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                job_status="FAILED",
                deployment_status="Running",
                ray_job_info={"endTime": "2026-05-01T12:45:00Z"},
            ),
            "FAILED",
            "Failed",
            "2026-05-01T12:45:00Z",
        ),
        (
            _rayjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                deployment_status="Failed",
                reason="SubmissionFailed",
                end_time="2026-05-01T12:45:00Z",
            ),
            "FAILED",
            "Failed:SubmissionFailed",
            "2026-05-01T12:45:00Z",
        ),
        (
            _rayjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                job_status="FAILED",
                reason="DeadlineExceeded",
                end_time="2026-05-01T12:45:00Z",
            ),
            "TIMEOUT",
            "Failed:DeadlineExceeded",
            "2026-05-01T12:45:00Z",
        ),
        (
            _rayjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                deployment_status="Failed",
                reason="PreRunningDeadlineExceeded",
                end_time="2026-05-01T12:45:00Z",
            ),
            "TIMEOUT",
            "Failed:PreRunningDeadlineExceeded",
            "2026-05-01T12:45:00Z",
        ),
        (
            _rayjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                job_status="FAILED",
                conditions=[{"type": "Failed", "status": "True", "lastTransitionTime": "2026-05-01T12:50:00Z"}],
            ),
            "FAILED",
            "Failed",
            "2026-05-01T12:50:00Z",
        ),
        (
            _rayjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                job_status="FAILED",
                deployment_status="Running",
                ray_cluster_status={
                    "conditions": [
                        {"type": "HeadPodReady", "status": "True", "lastTransitionTime": "2026-05-01T12:05:00Z"}
                    ]
                },
            ),
            "FAILED",
            "Failed",
            None,
        ),
        (
            _rayjob(
                "pg-a-job",
                "uid-job-uid-pg-a",
                job_status="FAILED",
                deployment_status="Retrying",
                ray_job_info={"endTime": "2026-05-01T12:45:00Z"},
            ),
            "RUNNING",
            "Active",
            None,
        ),
        (_rayjob("pg-a-job", "uid-job-uid-pg-a", job_status="RUNNING"), "RUNNING", "Active", None),
        (_rayjob("pg-a-job", "uid-job-uid-pg-a", deployment_status="Suspended"), "RUNNING", "Suspended", None),
    ],
    ids=[
        "succeeded",
        "kuberay-v1-34-succeeded",
        "complete-deployment",
        "stopped",
        "stopped-with-reason",
        "failed",
        "kuberay-v1-34-failed",
        "failed-reason",
        "deadline-exceeded",
        "pre-running-deadline-exceeded",
        "failed-condition-fallback",
        "untimed-failed-ignores-ray-cluster-readiness",
        "retrying-after-failed-attempt",
        "running",
        "suspended",
    ],
)
def test_rayjob_owner_state_and_end_time(mocker, owner, state, owner_state, expected_end):
    pg_name = "pg-pg-a-job-uid-job-uid-pg-a"
    pg = _podgroup(name=pg_name, owner_name="pg-a-job", owner_kind="RayJob")
    pods = [
        _pod(
            "ray-worker-0",
            podgroup=pg_name,
            node="node002",
            controller=("RayCluster", "uid-raycluster-1"),
        )
    ]
    _wire_kubectl(mocker, podgroups=[pg], pods=pods, rayjobs=[owner])

    before = datetime.now(tz=timezone.utc)
    [ctx] = KaiExecutionContextAdapter().discover_by_ids([f"team-alpha/{pg_name}"])

    assert ctx["owner_kind"] == "RayJob"
    assert ctx["state"] == state
    assert ctx["owner_state"] == owner_state
    assert ctx["nodes"] == ["node002"]
    if expected_end is not None:
        assert ctx["end_time"] == datetime.fromisoformat(expected_end)
    else:
        assert ctx["end_time"] >= before


@pytest.mark.parametrize(
    "owner, state, owner_state, expected_end",
    [
        (
            _deployment(
                "pg-a-job",
                "uid-job-uid-pg-a",
                conditions=[
                    {
                        "type": "Progressing",
                        "status": "True",
                        "reason": "NewReplicaSetAvailable",
                        "lastTransitionTime": "2026-05-01T12:05:00Z",
                    }
                ],
            ),
            "RUNNING",
            "Active",
            None,
        ),
        (
            _deployment(
                "pg-a-job",
                "uid-job-uid-pg-a",
                conditions=[
                    {
                        "type": "Progressing",
                        "status": "False",
                        "reason": "ProgressDeadlineExceeded",
                        "lastTransitionTime": "2026-05-01T12:45:00Z",
                    }
                ],
            ),
            "RUNNING",
            "ProgressDeadlineExceeded",
            None,
        ),
    ],
    ids=["active", "progress-deadline-exceeded"],
)
def test_deployment_owner_state_and_end_time(mocker, owner, state, owner_state, expected_end):
    pg_name = "pg-pg-a-job-uid-job-uid-pg-a"
    pg = _podgroup(name=pg_name, owner_name="pg-a-job", owner_kind="Deployment")
    pods = [
        _pod(
            "deploy-pod-0",
            podgroup=pg_name,
            node="node003",
            controller=("ReplicaSet", "uid-rs-1"),
        )
    ]
    _wire_kubectl(mocker, podgroups=[pg], pods=pods, deployments=[owner])

    before = datetime.now(tz=timezone.utc)
    [ctx] = KaiExecutionContextAdapter().discover_by_ids([f"team-alpha/{pg_name}"])

    assert ctx["owner_kind"] == "Deployment"
    assert ctx["state"] == state
    assert ctx["owner_state"] == owner_state
    assert ctx["nodes"] == ["node003"]
    if expected_end is not None:
        assert ctx["end_time"] == datetime.fromisoformat(expected_end)
    else:
        assert ctx["end_time"] >= before


@pytest.mark.parametrize(
    "owner, state, owner_state, expected_end",
    [
        (
            _lws(
                "pg-a-job",
                "uid-job-uid-pg-a",
                conditions=[{"type": "Available", "status": "True", "lastTransitionTime": "2026-05-01T12:05:00Z"}],
            ),
            "RUNNING",
            "Active",
            None,
        ),
        (
            _lws(
                "pg-a-job",
                "uid-job-uid-pg-a",
                conditions=[
                    {"type": "Available", "status": "False", "lastTransitionTime": "2026-05-01T12:40:00Z"},
                    {"type": "Progressing", "status": "False", "lastTransitionTime": "2026-05-01T12:45:00Z"},
                    {"type": "UpdateInProgress", "status": "False", "lastTransitionTime": "2026-05-01T12:30:00Z"},
                ],
            ),
            "RUNNING",
            "Active",
            None,
        ),
        (
            _lws(
                "pg-a-job",
                "uid-job-uid-pg-a",
                conditions=[
                    {
                        "type": "Failed",
                        "status": "True",
                        "reason": "ReplicaFailure",
                        "lastTransitionTime": "2026-05-01T12:45:00Z",
                    }
                ],
            ),
            "FAILED",
            "Failed:ReplicaFailure",
            "2026-05-01T12:45:00Z",
        ),
    ],
    ids=["active", "unavailable-not-terminal", "failed-condition"],
)
def test_leaderworkerset_owner_state_and_end_time(mocker, owner, state, owner_state, expected_end):
    pg_name = "pg-pg-a-job-uid-job-uid-pg-a-group-0"
    pg = _podgroup(name=pg_name, owner_name="pg-a-job", owner_kind="LeaderWorkerSet")
    pods = [
        _pod(
            "lws-0-0",
            podgroup=pg_name,
            node="node004",
            controller=("StatefulSet", "uid-sts-1"),
        )
    ]
    _wire_kubectl(mocker, podgroups=[pg], pods=pods, leaderworkersets=[owner])

    before = datetime.now(tz=timezone.utc)
    [ctx] = KaiExecutionContextAdapter().discover_by_ids([f"team-alpha/{pg_name}"])

    assert ctx["owner_kind"] == "LeaderWorkerSet"
    assert ctx["state"] == state
    assert ctx["owner_state"] == owner_state
    assert ctx["nodes"] == ["node004"]
    if expected_end is not None:
        assert ctx["end_time"] == datetime.fromisoformat(expected_end)
    else:
        assert ctx["end_time"] >= before


@pytest.mark.parametrize(
    "owner, state, owner_state, expected_end",
    [
        (_dynamo("pg-a-job", "uid-job-uid-pg-a", state="successful"), "RUNNING", "Active", None),
        (
            _dynamo(
                "pg-a-job",
                "uid-job-uid-pg-a",
                state="failed",
                conditions=[
                    {
                        "type": "Ready",
                        "status": "False",
                        "reason": "ComponentCrash",
                        "lastTransitionTime": "2026-05-01T12:45:00Z",
                    }
                ],
            ),
            "FAILED",
            "Failed:ComponentCrash",
            "2026-05-01T12:45:00Z",
        ),
        (
            _dynamo(
                "pg-a-job",
                "uid-job-uid-pg-a",
                state="failed",
                reason="StartupCrash",
            ),
            "FAILED",
            "Failed:StartupCrash",
            None,
        ),
        (
            _dynamo(
                "pg-a-job",
                "uid-job-uid-pg-a",
                conditions=[
                    {
                        "type": "Ready",
                        "status": "False",
                        "reason": "Failed",
                        "lastTransitionTime": "2026-05-01T12:50:00Z",
                    }
                ],
            ),
            "FAILED",
            "Failed",
            "2026-05-01T12:50:00Z",
        ),
    ],
    ids=["active", "failed-state", "untimed-failed-state", "failed-ready-condition"],
)
def test_dynamographdeployment_owner_state_and_end_time(mocker, owner, state, owner_state, expected_end):
    pg_name = "pg-uid-job-uid-pg-a-decode"
    pg = _podgroup(name=pg_name, owner_name="pg-a-job", owner_kind="DynamoGraphDeployment")
    pods = [
        _pod(
            "dynamo-worker-0",
            podgroup=pg_name,
            node="node005",
            controller=("StatefulSet", "uid-sts-1"),
        )
    ]
    _wire_kubectl(mocker, podgroups=[pg], pods=pods, dynamographdeployments=[owner])

    before = datetime.now(tz=timezone.utc)
    [ctx] = KaiExecutionContextAdapter().discover_by_ids([f"team-alpha/{pg_name}"])

    assert ctx["owner_kind"] == "DynamoGraphDeployment"
    assert ctx["state"] == state
    assert ctx["owner_state"] == owner_state
    assert ctx["nodes"] == ["node005"]
    if expected_end is not None:
        assert ctx["end_time"] == datetime.fromisoformat(expected_end)
    else:
        assert ctx["end_time"] >= before


@pytest.mark.parametrize(
    "owner_kind",
    [
        "LeaderWorkerSet",
        "PyTorchJob",
        "RayJob",
        "Deployment",
        "DynamoGraphDeployment",
    ],
)
def test_node_loss_on_non_job_owner_pod_sets_node_fail(mocker, owner_kind):
    pg = _podgroup(owner_kind=owner_kind)
    lost = _pod("member-0", node="node007", top_owner=(owner_kind, "uid-job-uid-pg-a"))
    lost["status"]["reason"] = "NodeLost"
    _wire_kubectl(mocker, podgroups=[pg], pods=[lost])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["owner_kind"] == owner_kind
    assert ctx["state"] == "NODE_FAIL"
    assert ctx["owner_state"] == "Active"
    assert ctx["failed_node"] == "node007"


@pytest.mark.parametrize(
    "owner_kind",
    [
        "LeaderWorkerSet",
        "PyTorchJob",
        "RayJob",
        "Deployment",
        "DynamoGraphDeployment",
    ],
)
def test_non_job_owner_filters_out_cross_incarnation_pods(mocker, owner_kind):
    pg = _podgroup(owner_kind=owner_kind)
    owner_uid = "uid-job-uid-pg-a"
    pods = [
        _pod(
            "matching-top-owner",
            node="node001",
            top_owner=(owner_kind, owner_uid),
        ),
        _pod(
            "matching-direct-controller",
            node="node002",
            controller=(owner_kind, owner_uid),
        ),
        _pod(
            "child-controller-with-matching-top-owner",
            node="node003",
            controller=("StatefulSet", "uid-child-sts"),
            top_owner=(owner_kind, owner_uid),
        ),
        _pod(
            "child-controller-without-top-owner",
            node="node006",
            controller=("StatefulSet", "uid-child-sts"),
        ),
        _pod(
            "stale-top-owner",
            node="node008",
            top_owner=(owner_kind, "uid-stale-owner"),
        ),
        _pod(
            "stale-direct-controller",
            node="node009",
            controller=(owner_kind, "uid-stale-owner"),
        ),
        _pod(
            "conflicting-top-and-direct-controller",
            node="node010",
            controller=(owner_kind, "uid-stale-owner"),
            top_owner=(owner_kind, owner_uid),
        ),
    ]
    _wire_kubectl(mocker, podgroups=[pg], pods=pods)

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == [
        "child-controller-with-matching-top-owner",
        "matching-direct-controller",
        "matching-top-owner",
    ]
    assert ctx["nodes"] == ["node001", "node002", "node003"]


@pytest.mark.parametrize(
    "owner_kind, child_kind, rejected_child_kind",
    [
        ("Deployment", "ReplicaSet", "StatefulSet"),
        ("LeaderWorkerSet", "StatefulSet", "ReplicaSet"),
        ("RayJob", "RayCluster", "ReplicaSet"),
        ("DynamoGraphDeployment", "PodClique", "CronJob"),
    ],
)
def test_child_controller_pods_match_uid_named_podgroup_without_top_owner_annotation(
    mocker,
    owner_kind,
    child_kind,
    rejected_child_kind,
):
    pg_name = "pg-pg-a-job-uid-job-uid-pg-a"
    pg = _podgroup(name=pg_name, owner_name="pg-a-job", owner_kind=owner_kind)
    pods = [
        _pod(
            "expected-child-pod",
            podgroup=pg_name,
            node="node001",
            controller=(child_kind, "uid-child-1"),
        ),
        _pod(
            "unrelated-child-pod",
            podgroup=pg_name,
            node="node002",
            controller=(rejected_child_kind, "uid-child-2"),
        ),
    ]
    _wire_kubectl(mocker, podgroups=[pg], pods=pods)

    [ctx] = KaiExecutionContextAdapter().discover_by_ids([f"team-alpha/{pg_name}"])

    assert ctx["pod_names"] == ["expected-child-pod"]
    assert ctx["nodes"] == ["node001"]


@pytest.mark.parametrize(
    "jobsets, reason",
    [
        ([], "owner JobSet not found"),
        ([_jobset("pg-a-job", "uid-other", completed_at="2026-05-01T13:00:00Z")], "owner JobSet uid mismatch"),
    ],
    ids=["missing", "reused-name"],
)
def test_podgroup_without_a_verified_jobset_is_skipped(mocker, jobsets, reason):
    info = mocker.patch("attribution_blueprint.connectors.execution_contexts.kai.connector.logger.info")
    pg = _podgroup(owner_kind="JobSet")
    _wire_kubectl(mocker, podgroups=[pg], pods=[], jobsets=jobsets)

    assert KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"]) == []
    assert any(f"{reason}; skipping" in str(c.args[0]) for c in info.call_args_list)


def test_podgroup_without_pods_has_no_nodes(mocker):
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["nodes"] == []
    assert "pod_names" not in ctx


def test_time_range_keeps_only_intersecting_podgroups(mocker):
    inside = _podgroup(name="pg-in", uid="uid-in", created="2026-05-01T12:00:00Z")
    later = _podgroup(name="pg-late", uid="uid-late", created="2026-05-03T12:00:00Z")
    finished_before = _podgroup(name="pg-old", uid="uid-old", created="2026-04-01T12:00:00Z")
    jobs = [
        _job("pg-in-job", "uid-job-uid-in", completed="2026-05-01T13:00:00Z"),
        _job("pg-old-job", "uid-job-uid-old", completed="2026-04-01T13:00:00Z"),
        _job("pg-late-job", "uid-job-uid-late", completed="2026-05-03T13:00:00Z"),
    ]
    _wire_kubectl(mocker, podgroups=[inside, later, finished_before], pods=[], jobs=jobs)

    contexts = KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)

    assert [c["id"] for c in contexts] == ["uid-in"]


def test_time_range_skips_placement_reads_for_excluded_podgroups(mocker):
    later = _podgroup(created="2026-05-03T12:00:00Z")
    _wire_kubectl(
        mocker,
        podgroups=[later],
        pods=[],
        jobs=[_job("pg-a-job", "uid-job-uid-pg-a", completed="2026-05-03T13:00:00Z")],
    )
    reader = mocker.Mock()
    reader.list_for_resource.side_effect = RuntimeError("must not be called")
    adapter = KaiExecutionContextAdapter(
        namespace="team-alpha",
        placement_records_enabled=True,
        placement_reader=reader,
    )

    assert adapter.discover_by_time_range(START, END) == []
    reader.list_for_resource.assert_not_called()


def test_time_range_nodelist_includes_retained_nodes(mocker):
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[], jobs=[_job("pg-a-job", "uid-job-uid-pg-a")])
    reader = mocker.Mock()
    reader.list_for_resource.return_value = [
        PlacementRecord(
            namespace="team-alpha",
            pod_name="deleted-0",
            pod_uid="pod-uid-deleted",
            node_name="node009",
            observed_at="2026-05-01T12:05:00Z",
        )
    ]
    adapter = KaiExecutionContextAdapter(
        namespace="team-alpha",
        placement_records_enabled=True,
        placement_reader=reader,
    )

    contexts = adapter.discover_by_time_range(START, END, nodelist=["node009"])

    assert [context["id"] for context in contexts] == ["uid-pg-a"]


def test_time_range_nodelist_filters_on_member_pod_nodes(mocker):
    a = _podgroup(name="pg-a", uid="uid-a")
    b = _podgroup(name="pg-b", uid="uid-b")
    pods = [
        _pod("a-0", podgroup="pg-a", node="node001", controller=("Job", "uid-job-uid-a")),
        _pod("b-0", podgroup="pg-b", node="node002", controller=("Job", "uid-job-uid-b")),
    ]
    _wire_kubectl(mocker, podgroups=[a, b], pods=pods)
    adapter = KaiExecutionContextAdapter(namespace="team-alpha")

    assert [c["id"] for c in adapter.discover_by_time_range(START, END, nodelist=["node002, node009"])] == ["uid-b"]
    assert len(adapter.discover_by_time_range(START, END, nodelist=[])) == 2


def test_time_range_rejects_naive_datetimes():
    with pytest.raises(ValueError):
        KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(datetime(2026, 5, 1), END)


def test_pods_are_listed_once_per_namespace(mocker):
    pgs = [
        _podgroup(name="pg-a", uid="uid-a", namespace="team-alpha"),
        _podgroup(name="pg-b", uid="uid-b", namespace="team-alpha"),
        _podgroup(name="pg-c", uid="uid-c", namespace="team-beta"),
    ]
    fake = _wire_kubectl(mocker, podgroups=pgs, pods=[])

    KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a", "team-alpha/pg-b", "team-beta/pg-c"])

    pod_lists = [call for call in fake.calls if call["op"] == "list" and call["kind"] == "Pod"]
    assert len(pod_lists) == 2
    assert {call["namespace"] for call in pod_lists} == {"team-alpha", "team-beta"}


def test_podgroups_are_requested_by_kind_and_group(mocker):
    fake = _wire_kubectl(mocker, podgroups=[], pods=[])

    KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)

    assert fake.calls[0]["kind"] == "PodGroup"
    assert fake.calls[0]["group"] == "scheduling.run.ai"


def test_time_range_uses_configured_namespace(mocker):
    fake = _wire_kubectl(mocker, podgroups=[], pods=[])
    KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)
    assert fake.calls[0]["namespace"] == "team-alpha"

    with pytest.raises(ValueError, match="requires eca.kai.namespace"):
        KaiExecutionContextAdapter().discover_by_time_range(START, END)


def test_bare_id_needs_a_namespace():
    with pytest.raises(ValueError, match="missing a namespace"):
        KaiExecutionContextAdapter().discover_by_ids(["pg-a"])


def test_bare_id_uses_configured_namespace(mocker):
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[])

    [ctx] = KaiExecutionContextAdapter(namespace="team-alpha").discover_by_ids(["pg-a"])

    assert ctx["id"] == "uid-pg-a"


def test_kubernetes_client_carries_kubeconfig_context_and_timeout(mocker):
    client_cls = mocker.patch(
        "attribution_blueprint.connectors.execution_contexts.kai.connector.KubernetesClient",
        return_value=_FakeKubernetesClient(podgroups=[_podgroup()], pods=[]),
    )

    KaiExecutionContextAdapter(
        kubeconfig="/tmp/kc",
        context="kind-kai",
        timeout_seconds=42,
    ).discover_by_ids(["team-alpha/pg-a"])

    client_cls.assert_called_once_with(
        kubeconfig="/tmp/kc",
        context="kind-kai",
        timeout_seconds=42,
    )


def test_failed_podgroup_list_fails_the_pass(mocker):
    fake = _wire_kubectl(mocker, podgroups=[], pods=[])
    fake.failures["PodGroup"] = KubernetesClientError("connection timed out")

    with pytest.raises(KubernetesClientError):
        KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)


def _fail_on(mocker, resource, error):
    fake = _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[], jobs=[])
    fake.failures[resource] = KubernetesClientError(error)


def test_denied_pod_list_fails_the_pass(mocker):
    _fail_on(mocker, "Pod", "Forbidden: pods is forbidden")

    with pytest.raises(KubernetesClientError, match="Forbidden"):
        KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)


def test_denied_owner_lookup_fails_the_pass(mocker):
    _fail_on(mocker, "Job", "Forbidden: jobs.batch is forbidden")

    with pytest.raises(KubernetesClientError, match="Forbidden"):
        KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])


def test_denied_jobset_lookup_fails_the_pass(mocker):
    pg = _podgroup(owner_kind="JobSet")
    fake = _wire_kubectl(mocker, podgroups=[pg], pods=[])
    fake.failures["JobSet"] = KubernetesClientError("Forbidden: jobsets are forbidden")

    with pytest.raises(KubernetesClientError, match="Forbidden"):
        KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])


def test_out_of_namespace_id_rejected_before_kubernetes_read_when_placement_records_enabled(mocker):
    fake = _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[])
    adapter = KaiExecutionContextAdapter(
        namespace="team-alpha",
        placement_records_enabled=True,
        placement_reader=mocker.Mock(),
    )

    with pytest.raises(ValueError, match="outside configured namespace"):
        adapter.discover_by_ids(["team-beta/pg-a"])

    assert fake.calls == []


def test_daemon_placements_restore_deleted_pod_and_node_failure(mocker):
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[], jobs=[_job("pg-a-job", "uid-job-uid-pg-a")])
    reader = mocker.Mock()
    reader.list_for_resource.return_value = [
        PlacementRecord(
            namespace="team-alpha",
            pod_name="deleted-0",
            pod_uid="pod-uid-deleted",
            node_name="node009",
            observed_at="2026-05-01T12:05:00Z",
            node_failure_reason="NodeLost",
        )
    ]
    adapter = KaiExecutionContextAdapter(
        placement_records_enabled=True,
        placement_reader=reader,
    )

    [ctx] = adapter.discover_by_ids(["team-alpha/pg-a"])

    reader.list_for_resource.assert_called_once_with("team-alpha", "uid-pg-a")
    assert ctx["state"] == "NODE_FAIL"
    assert ctx["failed_node"] == "node009"
    assert ctx["nodes"] == ["node009"]
    assert ctx["pod_names"] == ["deleted-0"]
    assert ctx["pod_nodes"] == {"deleted-0": "node009"}
    assert ctx["pod_placements"] == [
        {
            "pod_uid": "pod-uid-deleted",
            "pod_name": "deleted-0",
            "node_name": "node009",
            "source": "kubernetes_daemon",
        }
    ]


def test_retained_placement_does_not_overwrite_live_pod_name_mapping(mocker):
    pod = _pod(
        "trainer-0",
        node="node-live",
        controller=("Job", "uid-job-uid-pg-a"),
    )
    _wire_kubectl(mocker, podgroups=[_podgroup()], pods=[pod])
    reader = mocker.Mock()
    reader.list_for_resource.return_value = [
        PlacementRecord(
            namespace="team-alpha",
            pod_name="trainer-0",
            pod_uid="pod-uid-deleted",
            node_name="node-old",
            observed_at="2026-05-01T12:05:00Z",
        )
    ]
    adapter = KaiExecutionContextAdapter(
        placement_records_enabled=True,
        placement_reader=reader,
    )

    [ctx] = adapter.discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_nodes"] == {"trainer-0": "node-live"}
    assert {placement["pod_uid"] for placement in ctx["pod_placements"]} == {
        "pod-uid-deleted",
    }


def test_denied_podgroup_lookup_fails_instead_of_skipping(mocker):
    fake = _wire_kubectl(mocker, podgroups=[], pods=[])
    fake.failures["PodGroup"] = KubernetesClientError("Forbidden")

    with pytest.raises(KubernetesClientError):
        KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])


def test_discover_without_selector_is_empty():
    assert KaiExecutionContextAdapter().discover() == []


def test_registered_as_eca_component():
    component = mca_registry.get_component("eca", "kai")

    assert component.name == "kai"
    assert {"ids", "kubeconfig", "context", "namespace", "timeout_seconds"} <= set(component.schema)
    assert "kubectl_path" not in component.schema


def test_builder_defaults_and_validation():
    component = mca_registry.get_component("eca", "kai")

    adapter = component.build({"timeout_seconds": None, "namespace": "team-alpha"}, ctx={"framework": "eca"})
    assert isinstance(adapter, KaiExecutionContextAdapter)
    assert adapter.timeout_seconds == 60
    assert adapter.namespace == "team-alpha"

    with pytest.raises(MCAError, match="positive"):
        component.build({"timeout_seconds": 0}, ctx={"framework": "eca"})

    with pytest.raises(MCAError, match="requires eca.kai.kubernetes_daemon"):
        component.build(
            {"namespace": "team-alpha", "placement_records_enabled": True},
            ctx={"framework": "eca"},
        )

    with pytest.raises(MCAError, match="requires eca.kai.namespace"):
        component.build(
            {
                "placement_records_enabled": True,
                "kubernetes_daemon": {"endpoint": "http://daemon:8080"},
            },
            ctx={"framework": "eca"},
        )


def test_builder_coerces_cli_and_environment_boolean_strings():
    component = mca_registry.get_component("eca", "kai")

    disabled = component.build(
        {"placement_records_enabled": "false"},
        ctx={"framework": "eca"},
    )
    enabled = component.build(
        {
            "namespace": "team-alpha",
            "placement_records_enabled": "true",
            "kubernetes_daemon": {"endpoint": "http://daemon:8080"},
        },
        ctx={"framework": "eca"},
    )

    assert disabled.placement_records_enabled is False
    assert enabled.placement_records_enabled is True
    with pytest.raises(MCAError, match="must be a bool"):
        component.build(
            {"placement_records_enabled": "sometimes"},
            ctx={"framework": "eca"},
        )
