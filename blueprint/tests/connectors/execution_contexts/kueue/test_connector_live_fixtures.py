# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests pinned to raw kubectl JSON captured from a live kind+Kueue
cluster (kueue-demo namespace) on 2026-05-21.

The synthetic fixtures in `tests/fixtures/kueue_*.json` are convenient but
silently elide shape differences real clusters produce. These tests load the
captured payloads under `tests/fixtures/kueue_live/` and assert the connector
behaves the way it did on the live cluster, including JobSet child Job and
pod evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from attribution_blueprint.connectors.execution_contexts.kubernetes_client import (
    KubernetesClientError,
)
from attribution_blueprint.connectors.execution_contexts.kueue import (
    KueueExecutionContextAdapter,
)

LIVE = Path(__file__).parent.parent.parent.parent / "fixtures" / "kueue_live"


def _load(name: str) -> dict:
    return json.loads((LIVE / name).read_text())


class _FakeLiveKubernetesClient:
    """A `KubernetesClient` stand-in answering `get_one`/`list` against
    in-memory lists of captured live objects.

    Mirrors the call shapes the adapter issues (list workloads, get a single
    workload, list pods by owner label, get a direct Job, and list JobSet
    child Jobs). The synthetic fixture fake lives in `test_connector.py`;
    this one is intentionally separate so live payloads don't bleed into the
    synthetic tests' shared module-level state.
    """

    def __init__(self, *args, workloads: list[dict], pods: list[dict], jobs: list[dict], **kwargs):
        del args, kwargs  # accepts and ignores KubernetesClient's constructor kwargs
        self.workloads = workloads
        self.pods = pods
        self.jobs = jobs
        self.calls: list[dict] = []

    @staticmethod
    def _by_name(items: list[dict], name: str, namespace: str | None) -> dict | None:
        for item in items:
            meta = item.get("metadata") or {}
            if meta.get("name") != name:
                continue
            if namespace and meta.get("namespace") != namespace:
                continue
            return item
        return None

    def _pods_by_label(self, key: str, value: str) -> list[dict]:
        return [p for p in self.pods if ((p.get("metadata") or {}).get("labels") or {}).get(key) == value]

    def get_one(
        self,
        resource_kind,
        name: str,
        *,
        namespace: str,
        group=None,
        api_version=None,
    ) -> dict:
        kind = resource_kind.lower() + "s"
        self.calls.append({"op": "get_one", "kind": kind, "name": name, "namespace": namespace, "label_selector": None})
        if kind == "workloads":
            items = self.workloads
        elif kind == "jobs":
            items = self.jobs
        else:
            raise AssertionError(f"_FakeLiveKubernetesClient.get_one: unexpected kind {kind!r}")
        obj = self._by_name(items, name, namespace)
        if obj is None:
            raise KubernetesClientError(f"{kind} {name!r} not found in {namespace!r}", status=404)
        return obj

    def list(
        self,
        resource_kind,
        *,
        namespace: str | None,
        field_selector: str | None = None,
        label_selector: str | None = None,
        group=None,
        api_version=None,
    ) -> list[dict]:
        kind = resource_kind.lower() + "s"
        self.calls.append(
            {"op": "list", "kind": kind, "name": None, "namespace": namespace, "label_selector": label_selector}
        )
        if kind == "workloads":
            return self.workloads
        if kind == "pods":
            if label_selector and "=" in label_selector:
                lkey, lvalue = label_selector.split("=", 1)
                return self._pods_by_label(lkey, lvalue)
            return []
        if kind == "jobs":
            items = []
            if label_selector and "=" in label_selector:
                lkey, lvalue = label_selector.split("=", 1)
                for job in self.jobs:
                    metadata = job.get("metadata") or {}
                    labels = metadata.get("labels") or {}
                    annotations = metadata.get("annotations") or {}
                    if (not namespace or metadata.get("namespace") == namespace) and (
                        labels.get(lkey) == lvalue or annotations.get(lkey) == lvalue
                    ):
                        items.append(job)
            return items
        raise AssertionError(f"_FakeLiveKubernetesClient.list: unexpected kind {kind!r}")


def _wire(mocker, *, workloads, pods, jobs):
    """Patch the adapter's `KubernetesClient` construction to return a fake
    reader backed by the captured live payloads, and return that fake so
    tests can assert on `.calls`."""
    fake = _FakeLiveKubernetesClient(workloads=workloads, pods=pods, jobs=jobs)
    mocker.patch(
        "attribution_blueprint.connectors.execution_contexts.kueue.connector.KubernetesClient",
        return_value=fake,
    )
    return fake


# --- single Job that failed with exit 7 -----------------------------------


def test_live_single_job_failure_attributes_failed_node(mocker):
    """A plain batch/v1 Job that failed with non-zero exit reports the right
    state, the node it ran on, and the same node as `failed_node`.

    Captured live from the kind+Kueue cluster: bp-fail-fast Job, busybox
    `exit 7`, single pod that landed on kueue-demo-worker2.
    """
    workload = _load("workload_single_job_failed.json")
    pods = _load("pods_single_job_failed.json")["items"]
    job = _load("job_single.json")
    _wire(mocker, workloads=[workload], pods=pods, jobs=[job])

    adapter = KueueExecutionContextAdapter(namespace="kueue-demo")
    contexts = adapter.discover_by_ids(["kueue-demo/job-bp-fail-fast-17027"])

    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["state"] == "FAILED"
    assert ctx["kueue_state"] == "FINISHED_FAILED"
    assert ctx["nodes"] == ["kueue-demo-worker2"]
    assert ctx["failed_node"] == "kueue-demo-worker2"
    assert ctx["queue_name"] == "dev"
    assert ctx["namespace"] == "kueue-demo"
    assert ctx["start_time"] == datetime(2026, 5, 21, 23, 16, 10, tzinfo=timezone.utc)
    assert ctx["end_time"] == datetime(2026, 5, 21, 23, 16, 19, tzinfo=timezone.utc)
    # Pod names are recorded so downstream log connectors (kubectl_logs)
    # don't have to re-derive the per-owner pod selector.
    assert ctx["pod_names"] == ["bp-fail-fast-gtldf"]


# --- JobSet failure: pod attribution via jobset-name label ----------------


def test_live_jobset_failure_attributes_failed_node(mocker):
    """A Workload owned by a JobSet (the canonical distributed-training
    shape) gets full pod-node attribution: `nodes` is populated and the
    pod that exited non-zero is reported as `failed_node`.

    Captured live from: bp-train-fail JobSet, completions=2 trainer where
    rank 1 `exit 42`. The Workload's `ownerReferences[0].kind` is `JobSet`
    (not `Job`), so the connector cannot use the `batch.kubernetes.io/
    controller-uid` label - the JobSet's child Jobs each have their own
    distinct UIDs. Instead, the connector selects pods by
    `jobset.sigs.k8s.io/jobset-name=<jobset-name>`, the label JobSet
    stamps on every pod it transitively owns.
    """
    workload = _load("workload_jobset_failed.json")
    pods = _load("pods_jobset_failed.json")["items"]
    _wire(mocker, workloads=[workload], pods=pods, jobs=[])

    adapter = KueueExecutionContextAdapter(namespace="kueue-demo")
    contexts = adapter.discover_by_ids(["kueue-demo/jobset-bp-train-fail-5677c"])

    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["state"] == "FAILED"
    assert ctx["kueue_state"] == "FINISHED_FAILED"
    # Only the failed pod (rank 1, exit 42 on kueue-demo-worker) survived
    # the JobSet's failurePolicy - the succeeding sibling was torn down by
    # the time fixtures were captured. nodes therefore lists just the one
    # surviving pod's node, and failed_node names it.
    assert ctx["nodes"] == ["kueue-demo-worker"]
    assert ctx["failed_node"] == "kueue-demo-worker"
    assert ctx["queue_name"] == "dev"
    # The selected pod's name flows through pod_names for downstream log
    # connectors.
    assert ctx["pod_names"] == ["bp-train-fail-trainer-0-1-sz6ft"]


def test_live_jobset_pod_label_selector_uses_jobset_name(mocker):
    """When the Workload owner is a JobSet, the `kubectl get pods -l ...`
    selector must use `jobset.sigs.k8s.io/jobset-name=<jobset.name>` - not
    the batch Job controller-uid (the Workload owner is the JobSet, which
    has no `controller-uid` to label-match against).
    """
    workload = _load("workload_jobset_failed.json")
    pods = _load("pods_jobset_failed.json")["items"]
    fake = _wire(mocker, workloads=[workload], pods=pods, jobs=[])

    adapter = KueueExecutionContextAdapter(namespace="kueue-demo")
    adapter.discover_by_ids(["kueue-demo/jobset-bp-train-fail-5677c"])

    pod_selectors = [call["label_selector"] for call in fake.calls if call["kind"] == "pods" and call["label_selector"]]
    assert pod_selectors, "connector should have issued at least one pod list query with a label selector"
    assert all(s == "jobset.sigs.k8s.io/jobset-name=bp-train-fail" for s in pod_selectors)


# --- pod-deleted-mid-flight: nodes lost ----------------------------------


def test_live_job_with_deleted_pod_has_no_node_evidence(mocker):
    """When the underlying pod has been garbage-collected (e.g. kubectl
    delete, kueue preemption, GC of completed pods) before the connector
    queries, `nodes` ends up empty even though the Workload's terminal state
    is still reported correctly.

    Captured live from: bp-kill-mid Job. A busybox `trap-on-SIGTERM`
    container was running on worker2; the pod was deleted with --grace
    seconds=2 mid-flight. The trap exited the container cleanly (busybox
    sh trap behavior), so the Job reports `Finished=True reason=Succeeded`,
    but by the time the connector runs the pod object is gone - there is no
    nodeName left to harvest.

    Operationally: anything that gc's pods (Kueue preemption, retention,
    admin delete) deletes node evidence with them. Downstream attributions
    looking for `nodes` will see [].
    """
    workload = _load("workload_pod_deleted.json")
    # No pods captured - the live state had already deleted them.
    _wire(mocker, workloads=[workload], pods=[], jobs=[])

    adapter = KueueExecutionContextAdapter(namespace="kueue-demo")
    contexts = adapter.discover_by_ids(["kueue-demo/job-bp-kill-mid-5e6b5"])

    assert len(contexts) == 1
    ctx = contexts[0]
    # Service state is normalized while the native Kueue state is preserved.
    assert ctx["state"] == "COMPLETED"
    assert ctx["kueue_state"] == "FINISHED_SUCCEEDED"
    assert ctx["nodes"] == []
    assert "failed_node" not in ctx


# --- discover_by_time_range over a mixed live listing --------------------


def test_live_discover_by_time_range_returns_three_workloads(mocker):
    """End-to-end exercise on the three captured workloads: the connector's
    cheap-first-pass window filter must keep all three (they all overlap
    [start, end]), and the full parse must produce the same per-workload
    findings as the targeted tests above.
    """
    workloads = [
        _load("workload_single_job_failed.json"),
        _load("workload_jobset_failed.json"),
        _load("workload_pod_deleted.json"),
    ]
    pods = _load("pods_single_job_failed.json")["items"] + _load("pods_jobset_failed.json")["items"]
    jobs = [_load("job_single.json")]
    _wire(mocker, workloads=workloads, pods=pods, jobs=jobs)

    adapter = KueueExecutionContextAdapter(namespace="kueue-demo")
    start = datetime(2026, 5, 21, 23, 16, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 21, 23, 19, 0, tzinfo=timezone.utc)
    contexts = adapter.discover_by_time_range(start, end)

    by_name = {c["name"]: c for c in contexts}
    assert set(by_name) == {
        "job-bp-fail-fast-17027",
        "jobset-bp-train-fail-5677c",
        "job-bp-kill-mid-5e6b5",
    }
    # Plain Job with surviving pod: nodes populated, failed_node set.
    assert by_name["job-bp-fail-fast-17027"]["nodes"] == ["kueue-demo-worker2"]
    assert by_name["job-bp-fail-fast-17027"]["failed_node"] == "kueue-demo-worker2"
    # JobSet: nodes populated via jobset-name label, failed_node names the
    # rank-1 pod's node.
    assert by_name["jobset-bp-train-fail-5677c"]["nodes"] == ["kueue-demo-worker"]
    assert by_name["jobset-bp-train-fail-5677c"]["failed_node"] == "kueue-demo-worker"
    # Pod-deleted Job: empty nodes (pod already gone before query).
    assert by_name["job-bp-kill-mid-5e6b5"]["nodes"] == []


def test_live_discover_by_time_range_nodelist_intersects_per_node(mocker):
    """The nodelist filter intersects against the connector's `nodes` field.
    With JobSet attribution wired up, the JobSet workload lands on
    `kueue-demo-worker` and the single Job lands on `kueue-demo-worker2`;
    filtering by either node should keep its own workload but drop the
    other one. The pod-deleted Job has no nodes at all and is dropped
    regardless.
    """
    workloads = [
        _load("workload_single_job_failed.json"),
        _load("workload_jobset_failed.json"),
        _load("workload_pod_deleted.json"),
    ]
    pods = _load("pods_single_job_failed.json")["items"] + _load("pods_jobset_failed.json")["items"]
    jobs = [_load("job_single.json")]
    _wire(mocker, workloads=workloads, pods=pods, jobs=jobs)

    adapter = KueueExecutionContextAdapter(namespace="kueue-demo")
    start = datetime(2026, 5, 21, 23, 16, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 21, 23, 19, 0, tzinfo=timezone.utc)
    only_worker2 = adapter.discover_by_time_range(start, end, nodelist=["kueue-demo-worker2"])
    assert {c["name"] for c in only_worker2} == {"job-bp-fail-fast-17027"}
    only_worker = adapter.discover_by_time_range(start, end, nodelist=["kueue-demo-worker"])
    assert {c["name"] for c in only_worker} == {"jobset-bp-train-fail-5677c"}


# --- regression: kubectl is called with the correct controller-uid label --


def test_live_pod_fetch_uses_batch_controller_uid_label(mocker):
    """The owning Job's UID - not the Workload's UID and not the Kueue
    `kueue.x-k8s.io/job-uid` label - is what the connector must put on the
    `kubectl get pods -l ...` selector. This was a real bug twice; pinning
    it against captured payloads keeps the regression visible.
    """
    workload = _load("workload_single_job_failed.json")
    pods = _load("pods_single_job_failed.json")["items"]
    job = _load("job_single.json")
    fake = _wire(mocker, workloads=[workload], pods=pods, jobs=[job])

    adapter = KueueExecutionContextAdapter(namespace="kueue-demo")
    adapter.discover_by_ids(["kueue-demo/job-bp-fail-fast-17027"])

    # Job UID captured live; the pod label-selector in any pod list query
    # must reference exactly this value.
    expected_uid = job["metadata"]["uid"]
    pod_selectors = [call["label_selector"] for call in fake.calls if call["kind"] == "pods" and call["label_selector"]]
    assert pod_selectors, "connector should have issued at least one pod list query with a label selector"
    assert all(s == f"batch.kubernetes.io/controller-uid={expected_uid}" for s in pod_selectors)
