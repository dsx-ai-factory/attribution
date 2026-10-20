# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from types import SimpleNamespace

from attribution_blueprint.kubernetes.daemon.models import ResourceIdentity
from attribution_blueprint.kubernetes.daemon.resolvers import ResolutionPending
from attribution_blueprint.kubernetes.daemon.runtime import KubernetesDaemon
from attribution_blueprint.kubernetes.daemon.sqlite_store import SQLitePlacementStore
from attribution_blueprint.kubernetes.daemon.store import CreateResult


def _pod() -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "trainer-0",
            "namespace": "team-a",
            "uid": "pod-uid-1",
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": "trainer",
                    "uid": "job-uid-1",
                    "controller": True,
                }
            ],
        },
        "spec": {"nodeName": "node-a"},
        "status": {},
    }


class _OwnerResolver:
    def resolve(self, pod):
        return ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")


class _PendingResourceResolver:
    def resolve(self, pod, owner):
        raise ResolutionPending("Workload has not appeared")


class _FailOnEnrichmentStore:
    def __init__(self):
        self.calls = 0

    def create(self, record):
        self.calls += 1
        if self.calls == 1:
            return CreateResult.CREATED
        raise OSError("synthetic storage failure")


def test_records_placement_before_any_resolution(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")

    daemon._record(_pod())

    record = store.get_by_pod_uid("pod-uid-1")
    assert record is not None
    assert record.node_name == "node-a"
    assert record.owner is None
    assert record.associated_resources == ()


def test_complete_listing_reconciles_absent_placements(mocker):
    store = mocker.Mock()
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    pod = _pod()
    listing = SimpleNamespace(
        items=[pod],
        metadata=SimpleNamespace(resource_version="42"),
    )

    assert daemon._enqueue_listing(listing) == "42"
    store.reconcile_active_pods.assert_called_once_with(
        "team-a",
        {"pod-uid-1"},
    )
    assert daemon.items.get_nowait() == (pod, False)


def test_retains_owner_when_optional_scheduler_resolution_exhausts(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_OwnerResolver(),
        resource_resolvers=(_PendingResourceResolver(),),
        max_resolution_attempts=1,
    )

    daemon._record(_pod())

    record = store.get_by_pod_uid("pod-uid-1")
    assert record is not None
    assert record.owner == ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")
    assert record.associated_resources == ()


def test_deleted_event_starts_placement_retention(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")

    daemon._record(_pod(), deleted=True)

    with sqlite3.connect(path) as connection:
        marker = connection.execute(
            "SELECT pod_uid FROM placement_deletion"
        ).fetchone()
    assert marker == ("pod-uid-1",)


def test_resource_version_expiry_resets_readiness_before_relisting(mocker):
    """A relist triggered by an expired watch resourceVersion (HTTP 410) must
    clear `initial_list_complete` until the relist completes, so `/readyz`
    does not report ready while the daemon cannot currently reconcile."""
    from kubernetes.client.exceptions import ApiException

    store = mocker.Mock()
    core_api = mocker.Mock()
    daemon = KubernetesDaemon(core_api=core_api, store=store, namespace="team-a", workers=1)

    readiness_during_lists = []

    def list_namespaced_pod(namespace):
        readiness_during_lists.append(daemon.health.initial_list_complete)
        return SimpleNamespace(items=[], metadata=SimpleNamespace(resource_version="1"))

    core_api.list_namespaced_pod.side_effect = list_namespaced_pod

    watcher = mocker.Mock()
    calls = {"count": 0}

    def stream(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ApiException(status=410, reason="resource version expired")
        daemon.stop()
        return iter(())

    watcher.stream.side_effect = stream
    mocker.patch("kubernetes.watch.Watch", return_value=watcher)

    daemon.run()

    assert readiness_during_lists == [False, False]
    assert daemon.health.initial_list_complete is True


def test_owner_enrichment_write_failure_does_not_escape_or_kill_recording():
    store = _FailOnEnrichmentStore()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_OwnerResolver(),
    )

    daemon._record(_pod())

    assert store.calls == 2
    assert daemon.health.persistence_failed is True


def test_worker_survives_unexpected_recording_failure(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    daemon.items.put((_pod(), False))
    daemon.items.put((_pod(), False))
    daemon.stop_event.set()
    calls = 0

    def fail_once(_pod, *, deleted=False):
        del deleted
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic processing failure")

    daemon._record = fail_once
    daemon._worker()

    assert calls == 2
    assert daemon.items.unfinished_tasks == 0
    assert daemon.health.recording_failed is False
