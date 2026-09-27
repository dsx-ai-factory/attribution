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


def _kubectl_router(workloads_payload=WORKLOADS, pods_payload=PODS, jobs_payload=JOBS):
    """Return a `subprocess.run` side_effect that dispatches by kubectl args.

    The adapter issues four shapes of kubectl call:
      - `kubectl get workloads [-A | -n NS]`            -> list
      - `kubectl get workload <name> -n <ns>`           -> single workload
      - `kubectl get pods -l batch.kubernetes.io/controller-uid=UID -A|-n NS` -> list
      - `kubectl get job <name> -n <ns>`                -> single Job
    """

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

    def _side_effect(cmd, **kwargs):
        # cmd is the full subprocess argv. Strip kubectl + flags, isolate the
        # kubectl sub-command vs. positional args.
        args = list(cmd)
        # The first non-flag positional is the kubectl verb.
        positional: list[str] = []
        ns: str | None = None
        label_selector: str | None = None
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
            uid = None
            if label_selector and label_selector.startswith("batch.kubernetes.io/controller-uid="):
                uid = label_selector.split("=", 1)[1]
            if uid is None:
                stdout = json.dumps({"items": []})
            else:
                stdout = json.dumps(_by_uid(uid))
        elif kind == "job":
            obj = _by_name(jobs_payload["items"], name, ns)
            if obj is None:
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="NotFound")
            stdout = json.dumps(obj)
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
    # Wire-format value goes straight to the service's workload.type enum,
    # which currently doesn't know "kueue". UNDEFINED ("") is the documented
    # fallback; switch back to "kueue" once the service enum learns it.
    assert ctx["workload_type"] == ""
    # BP-internal identity is independent of the wire field. SQLite reads
    # `scheduler` for the `scheduler_type` column, so kueue rows stay
    # distinguishable from other UNDEFINED sources.
    assert ctx["scheduler"] == "kueue"
    assert ctx["job_id"] == "team-alpha/job-train-mnist-abc12"
    assert ctx["namespace"] == "team-alpha"
    assert ctx["name"] == "job-train-mnist-abc12"
    assert ctx["state"] == "FINISHED_SUCCEEDED"
    assert ctx["nodes"] == ["node005"]
    assert ctx["pod_nodes"] == {"train-mnist-abc12-0": "node005"}
    assert ctx["start_time"] == datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert ctx["end_time"] == datetime(2026, 5, 1, 12, 30, 0, tzinfo=timezone.utc)
    assert ctx["user_name"] == "alice"
    assert ctx["queue_name"] == "research-queue"
    assert ctx["priority_class"] == 100
    assert ctx["time_limit"] == 3600
    assert "failed_node" not in ctx
    # kueue_phase was removed - it had no information beyond `state`.
    assert "kueue_phase" not in ctx


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


def test_failed_workload_surfaces_first_nonzero_exit_node(mocker):
    _wire_kubectl(mocker)
    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-llama-7b-pretrain-xyz77"])
    ctx = contexts[0]
    assert ctx["state"] == "FINISHED_FAILED"
    assert ctx["nodes"] == ["node001", "node002", "node003"]
    # Only node002's container reported exitCode=137, even though all three
    # pods are in Failed phase. failed_node names the node that actually
    # produced the bad exit, not the whole nodelist.
    assert ctx["failed_node"] == "node002"
    assert ctx["time_limit"] == 7200


def test_evicted_workload_state_and_end_time(mocker):
    _wire_kubectl(mocker)
    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-beta/job-hpl-bench-pp99"])
    ctx = contexts[0]
    # Evicted=True at 14:05 with reason=Preempted. No Finished condition,
    # so end_time falls back to the Evicted transition time.
    assert ctx["state"] == "EVICTED_PREEMPTED"
    assert ctx["end_time"] == datetime(2026, 5, 1, 14, 5, 0, tzinfo=timezone.utc)
    assert ctx["nodes"] == ["node010"]
    # Hpl-bench has no batch/v1 Job owner in our fixture, so time_limit absent.
    assert "time_limit" not in ctx


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


def test_pending_workload_falls_back_to_creation_timestamp(mocker):
    """A workload with no Admitted condition uses metadata.creationTimestamp.

    Without any timestamps the connector would have to skip the row - this
    branch ensures pending workloads still produce a usable context window
    so callers can attribute "stuck pending" symptoms too.
    """
    _wire_kubectl(mocker)
    contexts = KueueExecutionContextAdapter().discover_by_ids(["team-alpha/job-pending-quota-rr11"])
    ctx = contexts[0]
    assert ctx["state"] == "PENDING"
    assert ctx["start_time"] == datetime(2026, 5, 2, 8, 59, 30, tzinfo=timezone.utc)
    # No pods owned by this workload yet.
    assert ctx["nodes"] == []


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
    assert contexts[0]["state"] == "FINISHED_SUCCEEDED"


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

    pods = [
        {"metadata": {"labels": {"batch.kubernetes.io/controller-uid": "owner-A"}}, "spec": {"nodeName": "n-good"}},
        {"metadata": {"labels": {}}, "spec": {"nodeName": "n-unlabelled"}},
        {"metadata": {"labels": {"batch.kubernetes.io/controller-uid": "owner-B"}}, "spec": {"nodeName": "n-other"}},
    ]
    nodes, failed_node, _, _ = _collect_pod_evidence(pods, ("batch.kubernetes.io/controller-uid", "owner-A"))
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

    def _pod(node: str, exit_code: int | None) -> dict:
        cs: dict = {}
        if exit_code is not None:
            cs = {"containerStatuses": [{"state": {"terminated": {"exitCode": exit_code}}}]}
        return {
            "metadata": {"labels": {"batch.kubernetes.io/controller-uid": "owner-A"}},
            "spec": {"nodeName": node},
            "status": cs,
        }

    # Three pods with non-zero exits on different nodes. Input order is
    # scrambled - the deterministic result must be the same regardless.
    pods_a = [_pod("n-c", 137), _pod("n-a", 1), _pod("n-b", 2), _pod("n-d", 0)]
    pods_b = list(reversed(pods_a))
    pods_c = [_pod("n-b", 2), _pod("n-d", 0), _pod("n-a", 1), _pod("n-c", 137)]

    selector = ("batch.kubernetes.io/controller-uid", "owner-A")
    _, failed_a, _, _ = _collect_pod_evidence(pods_a, selector)
    _, failed_b, _, _ = _collect_pod_evidence(pods_b, selector)
    _, failed_c, _, _ = _collect_pod_evidence(pods_c, selector)

    assert failed_a == failed_b == failed_c == "n-a"


def test_collect_pod_evidence_returns_empty_for_workload_with_no_owner_job():
    """A Workload without an owner Job (e.g. pending, never-admitted) has no
    pods to attribute. Returning ([], None) avoids accidentally folding
    every pod in the cluster into such workloads."""
    from attribution_blueprint.connectors.execution_contexts.kueue.connector import _collect_pod_evidence

    pods = [{"metadata": {"labels": {"batch.kubernetes.io/controller-uid": "anything"}}, "spec": {"nodeName": "n"}}]
    assert _collect_pod_evidence(pods, None) == ([], None, [], {})


def test_pod_nodes_maps_scheduled_pods_and_omits_unscheduled_ones():
    from attribution_blueprint.connectors.execution_contexts.kueue.connector import _collect_pod_evidence

    selector = ("batch.kubernetes.io/controller-uid", "owner-A")
    pods = [
        {"metadata": {"name": "w-0", "labels": {selector[0]: selector[1]}}, "spec": {"nodeName": "n-1"}},
        {"metadata": {"name": "w-1", "labels": {selector[0]: selector[1]}}, "spec": {"nodeName": "n-2"}},
        {"metadata": {"name": "w-pending", "labels": {selector[0]: selector[1]}}, "spec": {}},
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
