# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from attribution_blueprint import mca_registry
from attribution_blueprint.connectors.execution_contexts.kai import KaiExecutionContextAdapter
from attribution_blueprint.connectors.execution_contexts.kai.connector import _resolve_owner
from attribution_blueprint.connectors.execution_contexts.kubernetes_client import KubernetesClientError
from attribution_blueprint.exceptions import MCAError

FIXTURES = Path(__file__).parent.parent.parent.parent / "fixtures"
PODGROUPS = json.loads((FIXTURES / "kai_podgroups.json").read_text())
PODS = json.loads((FIXTURES / "kai_pods.json").read_text())
JOBS = json.loads((FIXTURES / "kai_jobs.json").read_text())


def _podgroup(
    name="pg-a",
    namespace="team-alpha",
    uid="uid-pg-a",
    created="2026-05-01T12:00:00Z",
    owner=True,
    owner_kind="Job",
    owner_group=None,
    spec=None,
    status=None,
):
    meta = {"name": name, "namespace": namespace, "uid": uid, "creationTimestamp": created}
    if owner:
        if owner_group is None and owner_kind == "JobSet":
            owner_group = "jobset.x-k8s.io"
        group = f"group: {owner_group}\n" if owner_group else ""
        meta["annotations"] = {
            "kai.scheduler/top-owner-metadata": (
                f"name: {name}-job\nuid: uid-job-{uid}\nkind: {owner_kind}\n{group}version: v1\n"
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
):
    meta = {"name": name, "namespace": namespace, "creationTimestamp": created}
    if controller is not None:
        kind, controller_uid = controller
        meta["ownerReferences"] = [{"kind": kind, "name": "c", "uid": controller_uid, "controller": True}]
    if podgroup is not None:
        meta["annotations"] = {"pod-group-name": podgroup}
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


def _job(name, uid, completed=None, failed_at=None, namespace="team-alpha", jobset=None):
    status = {}
    if completed:
        status["completionTime"] = completed
    if failed_at:
        status["conditions"] = [{"type": "Failed", "status": "True", "lastTransitionTime": failed_at}]
    metadata = {"name": name, "namespace": namespace, "uid": uid}
    if jobset is not None:
        jobset_name, jobset_uid = jobset
        metadata["labels"] = {"jobset.sigs.k8s.io/jobset-name": jobset_name}
        metadata["ownerReferences"] = [
            {"kind": "JobSet", "name": jobset_name, "uid": jobset_uid, "controller": True}
        ]
    return {"metadata": metadata, "status": status}


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


class _FakeKubernetesClient:
    def __init__(self, podgroups=None, pods=None, jobs=None, jobsets=None):
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
        self.calls: list[dict] = []
        self.failures: dict[str, Exception] = {}

    def _items(self, kind):
        return {
            "PodGroup": self.podgroups,
            "Pod": self.pods,
            "Job": self.jobs,
            "JobSet": self.jobsets,
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
                item
                for item in selected
                if ((item.get("metadata") or {}).get("labels") or {}).get(key) == value
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
        if kind in self.failures:
            raise self.failures[kind]
        obj = next(
            (
                item
                for item in self._selected(kind, namespace)
                if (item.get("metadata") or {}).get("name") == name
            ),
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


def _wire_kubectl(mocker, podgroups=None, pods=None, jobs=None, jobsets=None):
    fake = _FakeKubernetesClient(podgroups, pods, jobs, jobsets)
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
    _wire_kubectl(mocker, PODGROUPS["items"], PODS["items"], JOBS["items"])

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


def test_context_carries_no_state(mocker):
    _wire_kubectl(mocker, PODGROUPS["items"], PODS["items"])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-train-mnist-1a2b3c4d"])

    assert "state" not in ctx
    assert "failed_node" not in ctx


def test_current_status_is_preserved_raw(mocker):
    _wire_kubectl(mocker, PODGROUPS["items"], PODS["items"])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-train-mnist-1a2b3c4d"])

    assert ctx["scheduling_conditions"][0]["type"] == "Unschedulable"
    assert ctx["resources_status"] == {"allocated": {"nvidia.com/gpu": "2"}}


def test_removed_status_fields_are_ignored(mocker):
    old = _podgroup(status={"phase": "Running", "running": 2, "failed": 0})
    _wire_kubectl(mocker, [old], [])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert "state" not in ctx
    assert "scheduling_conditions" not in ctx
    assert "resources_status" not in ctx


def test_owner_comes_from_top_owner_annotation(mocker):
    _wire_kubectl(mocker, PODGROUPS["items"], PODS["items"])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-train-mnist-1a2b3c4d"])

    assert (ctx["owner_kind"], ctx["owner_name"], ctx["owner_uid"]) == ("Job", "train-mnist", "uid-job-1001")
    assert ctx["podgroup_shape"] == "owned"


@pytest.mark.parametrize(
    "api_version, kind",
    [("batch/v1", "Job"), ("jobset.x-k8s.io/v1alpha2", "JobSet")],
)
def test_owner_falls_back_to_controller_owner_reference(mocker, api_version, kind):
    pg = _podgroup(owner=False)
    pg["metadata"]["ownerReferences"] = [
        {"apiVersion": api_version, "kind": kind, "name": "owner", "uid": "uid-owner", "controller": True},
    ]
    _wire_kubectl(mocker, [pg], [])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert (ctx["owner_kind"], ctx["owner_name"], ctx["owner_uid"]) == (kind, "owner", "uid-owner")
    assert ctx["podgroup_shape"] == "owned"


def test_malformed_top_owner_annotation_leaves_the_podgroup_unowned(mocker):
    pg = _podgroup(owner=False)
    pg["metadata"]["annotations"] = {"kai.scheduler/top-owner-metadata": "{not: [valid"}
    _wire_kubectl(mocker, [pg], [])

    assert KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"]) == []


def test_hierarchical_podgroup_is_emitted_and_flagged(mocker):
    pg = _podgroup(
        owner_kind="JobSet",
        spec={"queue": "q", "subGroups": [{"name": "decode", "minMember": 1}]},
    )
    _wire_kubectl(mocker, [pg], [])

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
    _wire_kubectl(mocker, [new_pg], pods)

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
    _wire_kubectl(mocker, [pg], [pod])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == ["early-0"]
    assert ctx["nodes"] == ["node-early"]
    assert ctx["start_time"] == datetime(2026, 5, 1, 11, 59, tzinfo=timezone.utc)


def test_pod_membership_does_not_require_managed_fields(mocker):
    pg = _podgroup()
    pod = _pod("ambiguous-0", controller=("Job", "uid-job-uid-pg-a"))
    del pod["metadata"]["managedFields"]
    _wire_kubectl(mocker, [pg], [pod])

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
    _wire_kubectl(mocker, [pg], [pod])

    contexts = KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(
        datetime(2026, 5, 1, 11, 58, tzinfo=timezone.utc),
        datetime(2026, 5, 1, 11, 59, 30, tzinfo=timezone.utc),
    )

    assert [ctx["id"] for ctx in contexts] == ["uid-pg-a"]


def test_pod_with_a_non_job_controller_is_not_joined(mocker):
    pod = _pod("rs-0", node="node-rs", controller=("ReplicaSet", "uid-rs"))
    pg = _podgroup()
    _wire_kubectl(mocker, [pg], [pod])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert "pod_names" not in ctx


def test_pod_without_controller_is_not_joined(mocker):
    _wire_kubectl(mocker, [_podgroup()], [_pod("bare-0")])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert "pod_names" not in ctx


def test_same_name_in_another_namespace_is_not_joined(mocker):
    pods = [
        _pod("mine-0", namespace="team-alpha", node="node-a", controller=("Job", "uid-job-uid-pg-a")),
        _pod("theirs-0", namespace="team-beta", node="node-b", controller=("Job", "uid-job-uid-pg-a")),
    ]
    _wire_kubectl(mocker, [_podgroup()], pods)

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == ["mine-0"]
    assert ctx["nodes"] == ["node-a"]


def test_missing_podgroup_is_skipped(mocker):
    _wire_kubectl(mocker, [], [])

    assert KaiExecutionContextAdapter().discover_by_ids(["team-alpha/gone"]) == []


def test_podgroup_deleted_between_list_and_lookup_is_skipped(mocker):
    """The listing saw the PodGroup but a later lookup finds nothing."""
    pg = _podgroup()
    fake = _wire_kubectl(mocker, [pg], [])
    adapter = KaiExecutionContextAdapter(namespace="team-alpha")
    assert len(adapter.discover_by_time_range(START, END)) == 1

    fake.podgroups = []

    assert adapter.discover_by_ids(["team-alpha/pg-a"]) == []


def test_malformed_podgroups_are_skipped(mocker):
    no_uid = _podgroup(uid="")
    no_created = _podgroup(name="pg-b", uid="uid-b", created="")
    ok = _podgroup(name="pg-c", uid="uid-c")
    _wire_kubectl(mocker, [no_uid, no_created, ok], [])

    contexts = KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)

    assert [c["id"] for c in contexts] == ["uid-c"]


def test_active_job_window_ends_now_without_a_warning(mocker, warnings):
    pod = _pod("a-0", controller=("Job", "uid-job-uid-pg-a"))
    _wire_kubectl(mocker, [_podgroup()], [pod], [_job("pg-a-job", "uid-job-uid-pg-a")])

    before = datetime.now(tz=timezone.utc)
    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["end_time"] >= before
    warnings.assert_not_called()


def test_completed_job_end_survives_pod_deletion(mocker, warnings):
    job = _job("pg-a-job", "uid-job-uid-pg-a", completed="2026-05-01T13:00:00Z")
    _wire_kubectl(mocker, [_podgroup()], [], [job])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["end_time"] == datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc)
    assert ctx["nodes"] == []
    assert "pod_names" not in ctx
    assert "node list unavailable" in _logged(warnings)


def test_failed_job_end_is_the_failed_condition_time(mocker, warnings):
    job = _job("pg-a-job", "uid-job-uid-pg-a", failed_at="2026-05-01T12:45:00Z")
    pod = _pod("a-0", controller=("Job", "uid-job-uid-pg-a"))
    _wire_kubectl(mocker, [_podgroup()], [pod], [job])

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
    _wire_kubectl(mocker, [pg], [], jobsets=[owner])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["owner_kind"] == "JobSet"
    assert ctx["end_time"] == datetime.fromisoformat(expected_end)


def test_active_jobset_window_ends_now(mocker):
    pg = _podgroup(owner_kind="JobSet")
    _wire_kubectl(mocker, [pg], [])

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
    fake = _wire_kubectl(mocker, [pg], pods, jobs=jobs)

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
    _wire_kubectl(mocker, [_podgroup()], pods, [job])

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
    _wire_kubectl(mocker, [_podgroup()], [pod], [_job("pg-a-job", "uid-job-uid-pg-a")])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["end_time"] > datetime(2026, 5, 2, tzinfo=timezone.utc)


def _non_batch_job_podgroup():
    pg = _podgroup()
    pg["metadata"]["annotations"]["kai.scheduler/top-owner-metadata"] += "group: example.com\n"
    return pg


@pytest.mark.parametrize(
    "podgroup, jobs, reason",
    [
        (_podgroup(owner=False), [], "not owned by a supported Job or JobSet"),
        (_podgroup(owner_kind="Deployment"), [], "not owned by a supported Job or JobSet"),
        (_podgroup(owner_kind="JobSet", owner_group="example.com"), [], "not owned by a supported Job or JobSet"),
        (_non_batch_job_podgroup(), [], "not owned by a supported Job or JobSet"),
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
        "non-batch-job",
        "job-missing",
        "job-reused-name",
    ],
)
def test_podgroup_without_a_verified_owner_job_is_skipped(mocker, podgroup, jobs, reason):
    info = mocker.patch("attribution_blueprint.connectors.execution_contexts.kai.connector.logger.info")
    _wire_kubectl(mocker, [podgroup], [], jobs)

    assert KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"]) == []
    assert any(f"{reason}; skipping" in str(c.args[0]) for c in info.call_args_list)


def test_unsupported_podgroups_are_not_enriched(mocker):
    fake = _wire_kubectl(mocker, [_podgroup(owner_kind="Deployment")], [])

    KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)

    assert all(call["kind"] not in {"Pod", "Job"} for call in fake.calls[1:])


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
    _wire_kubectl(mocker, [pg], [], jobsets=jobsets)

    assert KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"]) == []
    assert any(f"{reason}; skipping" in str(c.args[0]) for c in info.call_args_list)


def test_podgroup_without_pods_has_no_nodes(mocker):
    _wire_kubectl(mocker, [_podgroup()], [])

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
    _wire_kubectl(mocker, [inside, later, finished_before], [], jobs)

    contexts = KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)

    assert [c["id"] for c in contexts] == ["uid-in"]


def test_time_range_nodelist_filters_on_member_pod_nodes(mocker):
    a = _podgroup(name="pg-a", uid="uid-a")
    b = _podgroup(name="pg-b", uid="uid-b")
    pods = [
        _pod("a-0", podgroup="pg-a", node="node001", controller=("Job", "uid-job-uid-a")),
        _pod("b-0", podgroup="pg-b", node="node002", controller=("Job", "uid-job-uid-b")),
    ]
    _wire_kubectl(mocker, [a, b], pods)
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
    fake = _wire_kubectl(mocker, pgs, [])

    KaiExecutionContextAdapter().discover_by_ids(
        ["team-alpha/pg-a", "team-alpha/pg-b", "team-beta/pg-c"]
    )

    pod_lists = [call for call in fake.calls if call["op"] == "list" and call["kind"] == "Pod"]
    assert len(pod_lists) == 2
    assert {call["namespace"] for call in pod_lists} == {"team-alpha", "team-beta"}


def test_podgroups_are_requested_by_kind_and_group(mocker):
    fake = _wire_kubectl(mocker, [], [])

    KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)

    assert fake.calls[0]["kind"] == "PodGroup"
    assert fake.calls[0]["group"] == "scheduling.run.ai"


def test_time_range_uses_configured_namespace(mocker):
    fake = _wire_kubectl(mocker, [], [])
    KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)
    assert fake.calls[0]["namespace"] == "team-alpha"

    with pytest.raises(ValueError, match="requires eca.kai.namespace"):
        KaiExecutionContextAdapter().discover_by_time_range(START, END)


def test_bare_id_needs_a_namespace():
    with pytest.raises(ValueError, match="missing a namespace"):
        KaiExecutionContextAdapter().discover_by_ids(["pg-a"])


def test_bare_id_uses_configured_namespace(mocker):
    _wire_kubectl(mocker, [_podgroup()], [])

    [ctx] = KaiExecutionContextAdapter(namespace="team-alpha").discover_by_ids(["pg-a"])

    assert ctx["id"] == "uid-pg-a"


def test_kubernetes_client_carries_kubeconfig_context_and_timeout(mocker):
    client_cls = mocker.patch(
        "attribution_blueprint.connectors.execution_contexts.kai.connector.KubernetesClient",
        return_value=_FakeKubernetesClient([_podgroup()], []),
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
    fake = _wire_kubectl(mocker, [], [])
    fake.failures["PodGroup"] = KubernetesClientError("connection timed out")

    with pytest.raises(KubernetesClientError):
        KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)


def _fail_on(mocker, resource, error):
    fake = _wire_kubectl(mocker, [_podgroup()], [], [])
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
    fake = _wire_kubectl(mocker, [pg], [])
    fake.failures["JobSet"] = KubernetesClientError("Forbidden: jobsets are forbidden")

    with pytest.raises(KubernetesClientError, match="Forbidden"):
        KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])


def test_denied_podgroup_lookup_fails_instead_of_skipping(mocker):
    fake = _wire_kubectl(mocker, [], [])
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
