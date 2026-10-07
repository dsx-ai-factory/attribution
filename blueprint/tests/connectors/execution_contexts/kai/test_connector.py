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
from datetime import datetime, timezone
from pathlib import Path

import pytest

from attribution_blueprint import mca_registry
from attribution_blueprint.connectors.execution_contexts.kai import KaiExecutionContextAdapter
from attribution_blueprint.connectors.execution_contexts.kai.connector import _resolve_owner
from attribution_blueprint.connectors.execution_contexts.kubectl_client import KubectlError
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
    spec=None,
    status=None,
):
    meta = {"name": name, "namespace": namespace, "uid": uid, "creationTimestamp": created}
    if owner:
        meta["annotations"] = {
            "kai.scheduler/top-owner-metadata": (
                f"name: {name}-job\nuid: uid-job-{uid}\nkind: {owner_kind}\nversion: v1\n"
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
    status = {"phase": phase}
    if finished:
        status["containerStatuses"] = [{"name": "c", "state": {"terminated": {"finishedAt": finished}}}]
    spec = {"nodeName": node} if node else {}
    return {"metadata": meta, "spec": spec, "status": status}


def _job(name, uid, completed=None, failed_at=None, namespace="team-alpha"):
    status = {}
    if completed:
        status["completionTime"] = completed
    if failed_at:
        status["conditions"] = [{"type": "Failed", "status": "True", "lastTransitionTime": failed_at}]
    return {"metadata": {"name": name, "namespace": namespace, "uid": uid}, "status": status}


def _wire_kubectl(mocker, podgroups=None, pods=None, jobs=None):
    """Patch `subprocess.run` with a router over PodGroups, Pods and Jobs.

    Honors `-n NS` on lists, and answers `get <resource> <name>` with the
    object or a NotFound error, like kubectl. When `jobs` is omitted, every
    Job-owned PodGroup gets a matching active Job; pass `[]` for none.
    """
    podgroups = list(podgroups or [])
    pods = list(pods or [])
    if jobs is None:
        jobs = [
            _job(owner["name"], owner["uid"], namespace=pg["metadata"]["namespace"])
            for pg in podgroups
            if (owner := _resolve_owner(pg["metadata"])) is not None and owner["kind"] == "Job"
        ]
    jobs = list(jobs)

    def _side_effect(cmd, **kwargs):
        args = list(cmd)
        positional: list[str] = []
        ns = None
        i = 0
        while i < len(args):
            tok = args[i]
            if tok in ("--kubeconfig", "--context", "-o"):
                i += 2
                continue
            if tok == "-n":
                ns = args[i + 1]
                i += 2
                continue
            if tok == "-A":
                i += 1
                continue
            positional.append(tok)
            i += 1
        _binary, _verb, resource, *rest = positional

        def _in_ns(items):
            return [o for o in items if ns is None or (o.get("metadata") or {}).get("namespace") == ns]

        if resource.startswith("podgroups"):
            items = podgroups
        elif resource == "pods":
            items = pods
        elif resource == "job.batch":
            items = jobs
        else:
            raise AssertionError(f"unexpected resource {resource!r}")
        if rest:
            match = [o for o in _in_ns(items) if o["metadata"].get("name") == rest[0]]
            if not match:
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="NotFound")
            payload = match[0]
        else:
            payload = {"items": _in_ns(items)}
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(payload), stderr="")

    return mocker.patch("subprocess.run", side_effect=_side_effect)


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


def test_owner_falls_back_to_controller_owner_reference(mocker):
    pg = _podgroup(owner=False)
    pg["metadata"]["ownerReferences"] = [
        {"apiVersion": "batch/v1", "kind": "Job", "name": "j", "uid": "uid-j", "controller": True},
    ]
    _wire_kubectl(mocker, [pg], [])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert (ctx["owner_kind"], ctx["owner_name"], ctx["owner_uid"]) == ("Job", "j", "uid-j")
    assert ctx["podgroup_shape"] == "owned"


def test_malformed_top_owner_annotation_leaves_the_podgroup_unowned(mocker):
    pg = _podgroup(owner=False)
    pg["metadata"]["annotations"] = {"kai.scheduler/top-owner-metadata": "{not: [valid"}
    _wire_kubectl(mocker, [pg], [])

    assert KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"]) == []


def test_hierarchical_podgroup_is_emitted_and_flagged(mocker):
    pg = _podgroup(spec={"queue": "q", "subGroups": [{"name": "decode", "minMember": 1}]})
    _wire_kubectl(mocker, [pg], [])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["podgroup_shape"] == "hierarchical"


def test_same_name_new_uid_does_not_inherit_previous_incarnation_pods(mocker):
    new_pg = _podgroup(uid="uid-pg-new", created="2026-05-01T14:00:00Z")
    pods = [
        _pod("old-0", node="node-old", phase="Failed", controller=("Job", "uid-job-uid-pg-old")),
        _pod("new-0", node="node-new", controller=("Job", "uid-job-uid-pg-new")),
    ]
    _wire_kubectl(mocker, [new_pg], pods)

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["id"] == "uid-pg-new"
    assert ctx["pod_names"] == ["new-0"]
    assert ctx["nodes"] == ["node-new"]


def test_pod_created_before_its_podgroup_is_a_member(mocker):
    pg = _podgroup(created="2026-05-01T12:00:00Z")
    pod = _pod("early-0", created="2026-05-01T11:59:00Z", node="node-early", controller=("Job", "uid-job-uid-pg-a"))
    _wire_kubectl(mocker, [pg], [pod])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == ["early-0"]
    assert ctx["nodes"] == ["node-early"]
    assert ctx["start_time"] == datetime(2026, 5, 1, 11, 59, tzinfo=timezone.utc)


def test_time_range_uses_a_member_pod_that_predates_its_podgroup(mocker):
    pg = _podgroup(created="2026-05-01T12:00:00Z")
    pod = _pod("early-0", created="2026-05-01T11:59:00Z", controller=("Job", "uid-job-uid-pg-a"))
    _wire_kubectl(mocker, [pg], [pod])

    contexts = KaiExecutionContextAdapter().discover_by_time_range(
        datetime(2026, 5, 1, 11, 58, tzinfo=timezone.utc),
        datetime(2026, 5, 1, 11, 59, 30, tzinfo=timezone.utc),
    )

    assert [ctx["id"] for ctx in contexts] == ["uid-pg-a"]


def test_different_owner_level_keeps_the_pod(mocker):
    pod = _pod("rs-0", node="node-rs", controller=("ReplicaSet", "uid-rs"))
    pg = _podgroup()
    _wire_kubectl(mocker, [pg], [pod])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == ["rs-0"]


def test_pod_without_controller_is_kept(mocker):
    _wire_kubectl(mocker, [_podgroup()], [_pod("bare-0")])

    [ctx] = KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])

    assert ctx["pod_names"] == ["bare-0"]


def test_same_name_in_another_namespace_is_not_joined(mocker):
    pods = [
        _pod("mine-0", namespace="team-alpha", node="node-a"),
        _pod("theirs-0", namespace="team-beta", node="node-b"),
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
    run_mock = _wire_kubectl(mocker, [pg], [])
    adapter = KaiExecutionContextAdapter()
    assert len(adapter.discover_by_time_range(START, END)) == 1

    run_mock.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["kubectl"], stderr="NotFound")

    assert adapter.discover_by_ids(["team-alpha/pg-a"]) == []


def test_malformed_podgroups_are_skipped(mocker):
    no_uid = _podgroup(uid="")
    no_created = _podgroup(name="pg-b", uid="uid-b", created="")
    ok = _podgroup(name="pg-c", uid="uid-c")
    _wire_kubectl(mocker, [no_uid, no_created, ok], [])

    contexts = KaiExecutionContextAdapter().discover_by_time_range(START, END)

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
    pod = _pod("a-0", phase="Succeeded", finished="2026-05-01T12:30:00Z")
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
        (_podgroup(owner=False), [], "not owned by a batch Job"),
        (_podgroup(owner_kind="JobSet"), [], "not owned by a batch Job"),
        (_non_batch_job_podgroup(), [], "not owned by a batch Job"),
        (_podgroup(), [], "owner Job not found"),
        (
            _podgroup(),
            [_job("pg-a-job", "uid-other", completed="2026-05-01T13:00:00Z")],
            "owner Job uid mismatch",
        ),
    ],
    ids=["no-owner", "jobset-owner", "non-batch-job", "job-missing", "job-reused-name"],
)
def test_podgroup_without_a_verified_owner_job_is_skipped(mocker, podgroup, jobs, reason):
    info = mocker.patch("attribution_blueprint.connectors.execution_contexts.kai.connector.logger.info")
    _wire_kubectl(mocker, [podgroup], [], jobs)

    assert KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"]) == []
    assert any(f"{reason}; skipping" in str(c.args[0]) for c in info.call_args_list)


def test_unsupported_podgroups_are_not_enriched(mocker):
    run = _wire_kubectl(mocker, [_podgroup(owner_kind="JobSet")], [])

    KaiExecutionContextAdapter().discover_by_time_range(START, END)

    assert all("pods" not in c.args[0] and "job.batch" not in c.args[0] for c in run.call_args_list[1:])


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

    contexts = KaiExecutionContextAdapter().discover_by_time_range(START, END)

    assert [c["id"] for c in contexts] == ["uid-in"]


def test_time_range_nodelist_filters_on_member_pod_nodes(mocker):
    a = _podgroup(name="pg-a", uid="uid-a")
    b = _podgroup(name="pg-b", uid="uid-b")
    pods = [_pod("a-0", podgroup="pg-a", node="node001"), _pod("b-0", podgroup="pg-b", node="node002")]
    _wire_kubectl(mocker, [a, b], pods)
    adapter = KaiExecutionContextAdapter()

    assert [c["id"] for c in adapter.discover_by_time_range(START, END, nodelist=["node002, node009"])] == ["uid-b"]
    assert len(adapter.discover_by_time_range(START, END, nodelist=[])) == 2


def test_time_range_rejects_naive_datetimes():
    with pytest.raises(ValueError):
        KaiExecutionContextAdapter().discover_by_time_range(datetime(2026, 5, 1), END)


def test_pods_are_listed_once_per_namespace(mocker):
    pgs = [
        _podgroup(name="pg-a", uid="uid-a", namespace="team-alpha"),
        _podgroup(name="pg-b", uid="uid-b", namespace="team-alpha"),
        _podgroup(name="pg-c", uid="uid-c", namespace="team-beta"),
    ]
    run_mock = _wire_kubectl(mocker, pgs, [])

    KaiExecutionContextAdapter().discover_by_time_range(START, END)

    pod_lists = [c.args[0] for c in run_mock.call_args_list if "pods" in c.args[0]]
    assert len(pod_lists) == 2
    assert all("-A" not in cmd for cmd in pod_lists)


def test_podgroups_are_requested_by_full_resource_name(mocker):
    run_mock = _wire_kubectl(mocker, [], [])

    KaiExecutionContextAdapter().discover_by_time_range(START, END)

    assert "podgroups.scheduling.run.ai" in run_mock.call_args_list[0].args[0]


def test_namespace_scope_uses_minus_n_and_unscoped_uses_dash_a(mocker):
    run_mock = _wire_kubectl(mocker, [], [])
    KaiExecutionContextAdapter(namespace="team-alpha").discover_by_time_range(START, END)
    scoped = run_mock.call_args_list[0].args[0]
    assert "-n" in scoped and "team-alpha" in scoped and "-A" not in scoped

    run_mock.reset_mock()
    KaiExecutionContextAdapter().discover_by_time_range(START, END)
    assert "-A" in run_mock.call_args_list[0].args[0]


def test_bare_id_needs_a_namespace():
    with pytest.raises(ValueError, match="missing a namespace"):
        KaiExecutionContextAdapter().discover_by_ids(["pg-a"])


def test_bare_id_uses_configured_namespace(mocker):
    _wire_kubectl(mocker, [_podgroup()], [])

    [ctx] = KaiExecutionContextAdapter(namespace="team-alpha").discover_by_ids(["pg-a"])

    assert ctx["id"] == "uid-pg-a"


def test_kubectl_invocation_carries_kubeconfig_and_context(mocker):
    run_mock = _wire_kubectl(mocker, [_podgroup()], [])

    KaiExecutionContextAdapter(
        kubectl_path="/usr/local/bin/kubectl", kubeconfig="/tmp/kc", context="kind-kai"
    ).discover_by_ids(["team-alpha/pg-a"])

    cmd = run_mock.call_args_list[0].args[0]
    assert cmd[0] == "/usr/local/bin/kubectl"
    assert "--kubeconfig" in cmd and "/tmp/kc" in cmd
    assert "--context" in cmd and "kind-kai" in cmd


def test_failed_podgroup_list_fails_the_pass(mocker):
    mocker.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["kubectl"], timeout=60))

    with pytest.raises(KubectlError):
        KaiExecutionContextAdapter().discover_by_time_range(START, END)


def _fail_on(mocker, resource, stderr):
    """Route kubectl normally, except that queries for `resource` fail."""
    run = _wire_kubectl(mocker, [_podgroup()], [], [])
    router = run.side_effect

    def _side_effect(cmd, **kwargs):
        if resource in cmd:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr=stderr)
        return router(cmd, **kwargs)

    run.side_effect = _side_effect


def test_denied_pod_list_fails_the_pass(mocker):
    _fail_on(mocker, "pods", "Error from server (Forbidden): pods is forbidden")

    with pytest.raises(KubectlError, match="Forbidden"):
        KaiExecutionContextAdapter().discover_by_time_range(START, END)


def test_denied_owner_lookup_fails_the_pass(mocker):
    _fail_on(mocker, "job.batch", "Error from server (Forbidden): jobs.batch is forbidden")

    with pytest.raises(KubectlError, match="Forbidden"):
        KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])


def test_denied_podgroup_lookup_fails_instead_of_skipping(mocker):
    mocker.patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(returncode=1, cmd=["kubectl"], stderr="Error (Forbidden)"),
    )

    with pytest.raises(KubectlError):
        KaiExecutionContextAdapter().discover_by_ids(["team-alpha/pg-a"])


def test_discover_without_selector_is_empty():
    assert KaiExecutionContextAdapter().discover() == []


def test_registered_as_eca_component():
    component = mca_registry.get_component("eca", "kai")

    assert component.name == "kai"
    assert {"ids", "kubectl_path", "kubeconfig", "context", "namespace", "timeout_seconds"} <= set(
        component.schema
    )


def test_builder_defaults_and_validation():
    component = mca_registry.get_component("eca", "kai")

    adapter = component.build({"timeout_seconds": None, "namespace": "team-alpha"}, ctx={"framework": "eca"})
    assert isinstance(adapter, KaiExecutionContextAdapter)
    assert adapter.timeout_seconds == 60
    assert adapter.namespace == "team-alpha"

    with pytest.raises(MCAError, match="positive"):
        component.build({"timeout_seconds": 0}, ctx={"framework": "eca"})
