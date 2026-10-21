# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import threading
from types import SimpleNamespace

from attribution_blueprint.kubernetes.daemon.models import PlacementRecord, ResourceIdentity
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
    def __init__(self):
        self.calls = 0

    def resolve(self, pod):
        self.calls += 1
        return ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")


class _ReassignedThenPendingResolver(_OwnerResolver):
    def resolve(self, pod):
        self.calls += 1
        if self.calls == 1:
            return ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")
        if self.calls == 2:
            return ResourceIdentity(
                "jobset.x-k8s.io/v1alpha2",
                "JobSet",
                "team-a",
                "training",
                "jobset-uid-1",
            )
        raise ResolutionPending("owner disappeared")


class _ReassignedOwnerResolver(_OwnerResolver):
    def resolve(self, pod):
        self.calls += 1
        if self.calls == 1:
            return ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")
        return ResourceIdentity(
            "jobset.x-k8s.io/v1alpha2",
            "JobSet",
            "team-a",
            "training",
            "jobset-uid-1",
        )


class _BlockingReassignmentResolver:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def resolve(self, pod):
        controller_uid = (pod["metadata"]["ownerReferences"][0])["uid"]
        if controller_uid == "job-uid-1":
            self.started.set()
            assert self.release.wait(timeout=2)
            return ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")
        return ResourceIdentity("batch/v1", "Job", "team-a", "trainer-2", "job-uid-2")


class _OwnerTrackingStore:
    def __init__(self, delegate):
        self.delegate = delegate
        self.newer_owner_written = threading.Event()
        self.newer_resource_written = threading.Event()

    def create(self, record):
        result = self.delegate.create(record)
        if record.owner and record.owner.uid == "job-uid-2":
            self.newer_owner_written.set()
        if any(resource.uid == "admission-uid-b" for resource in record.associated_resources):
            self.newer_resource_written.set()
        return result

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _BlockingResourceResolver:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def resolve(self, pod, owner):
        event = (pod.get("metadata") or {}).get("annotations", {}).get("event")
        if event == "old":
            self.started.set()
            assert self.release.wait(timeout=2)
            suffix = "a"
        else:
            suffix = "b"
        return (
            ResourceIdentity(
                "scheduling.example.io/v1",
                "Admission",
                "team-a",
                f"admission-{suffix}",
                f"admission-uid-{suffix}",
            ),
        )


class _PendingResourceResolver:
    def resolve(self, pod, owner):
        raise ResolutionPending("Workload has not appeared")


class _ReassignedResourceResolver:
    def __init__(self):
        self.calls = 0

    def resolve(self, pod, owner):
        self.calls += 1
        if self.calls == 2:
            raise ResolutionPending("resource has not appeared")
        suffix = "a" if self.calls == 1 else "b"
        return (
            ResourceIdentity(
                "scheduling.example.io/v1",
                "Admission",
                "team-a",
                f"admission-{suffix}",
                f"admission-uid-{suffix}",
            ),
        )


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
    queued_pod, deleted, _generation = daemon.items.get_nowait()
    assert queued_pod == pod
    assert deleted is False


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


def test_reassigned_owner_refreshes_controller_reference_for_later_fallback(tmp_path, mocker):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    resolver = _ReassignedThenPendingResolver()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=resolver,
    )
    wait = mocker.patch.object(daemon.stop_event, "wait")
    daemon._record(_pod())
    reassigned = _pod()
    reassigned["metadata"]["ownerReferences"][0]["uid"] = "job-uid-2"

    daemon._record(reassigned)
    daemon._record(reassigned)

    stored = store.get_by_pod_uid("pod-uid-1")
    assert resolver.calls == 3
    wait.assert_not_called()
    assert stored.owner.uid == "jobset-uid-1"
    assert stored.owner_references[0].uid == "job-uid-2"


def test_owner_change_clears_old_association_until_new_resource_resolves(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_ReassignedOwnerResolver(),
        resource_resolvers=(_ReassignedResourceResolver(),),
        max_resolution_attempts=1,
    )
    daemon._record(_pod())
    reassigned = _pod()
    reassigned["metadata"]["ownerReferences"][0]["uid"] = "job-uid-2"

    daemon._record(reassigned)

    stored = store.get_by_pod_uid("pod-uid-1")
    assert stored is not None
    assert stored.owner.uid == "jobset-uid-1"
    assert stored.associated_resources == ()
    assert store.list_for_resource("team-a", "admission-uid-a") == []

    daemon._record(reassigned)

    assert store.list_for_resource("team-a", "admission-uid-b")


def test_stale_owner_write_cannot_overwrite_newer_event(tmp_path):
    backing_store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store = _OwnerTrackingStore(backing_store)
    resolver = _BlockingReassignmentResolver()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=resolver,
        workers=2,
    )
    older = _pod()
    newer = _pod()
    newer["metadata"]["ownerReferences"][0].update(
        {"name": "trainer-2", "uid": "job-uid-2"}
    )
    daemon._enqueue(older)
    daemon._enqueue(newer)
    workers = [threading.Thread(target=daemon._worker) for _ in range(2)]
    for worker in workers:
        worker.start()

    assert resolver.started.wait(timeout=2)
    assert store.newer_owner_written.wait(timeout=2)
    daemon.stop_event.set()
    resolver.release.set()
    daemon.items.join()
    for worker in workers:
        worker.join(timeout=2)

    assert backing_store.get_by_pod_uid("pod-uid-1").owner.uid == "job-uid-2"


def test_stale_resource_write_cannot_overwrite_newer_event(tmp_path):
    backing_store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store = _OwnerTrackingStore(backing_store)
    resolver = _BlockingResourceResolver()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_OwnerResolver(),
        resource_resolvers=(resolver,),
        workers=2,
    )
    older = _pod()
    older["metadata"]["annotations"] = {"event": "old"}
    newer = _pod()
    newer["metadata"]["annotations"] = {"event": "new"}
    daemon._enqueue(older)
    first = threading.Thread(target=daemon._worker)
    first.start()
    assert resolver.started.wait(timeout=2)

    daemon._enqueue(newer)
    second = threading.Thread(target=daemon._worker)
    second.start()
    assert store.newer_resource_written.wait(timeout=2)
    daemon.stop_event.set()
    resolver.release.set()
    daemon.items.join()
    first.join(timeout=2)
    second.join(timeout=2)

    assert backing_store.list_for_resource("team-a", "admission-uid-a") == []
    assert backing_store.list_for_resource("team-a", "admission-uid-b")


def test_missing_generation_tracking_rejects_enrichment_write(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    record = PlacementRecord.for_pod(_pod()).enriched(
        owner=ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")
    )

    assert daemon._persist_enrichment(record, "test enrichment", generation=1) is None
    assert store.get_by_pod_uid("pod-uid-1") is None


def test_stale_event_still_persists_failure_and_deletion_evidence(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    pod = _pod()
    pod["status"]["reason"] = "NodeLost"
    daemon._latest_generation["pod-uid-1"] = 2

    daemon._record(pod, deleted=True, generation=1)

    assert store.get_by_pod_uid("pod-uid-1").node_failure_reason == "NodeLost"
    with sqlite3.connect(path) as connection:
        marker = connection.execute(
            "SELECT pod_uid FROM placement_deletion"
        ).fetchone()
    assert marker == ("pod-uid-1",)


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
    daemon._enqueue(_pod())
    daemon._enqueue(_pod())
    daemon.stop_event.set()
    calls = 0

    def fail_once(_pod, *, deleted=False, generation=None):
        del deleted, generation
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic processing failure")

    daemon._record = fail_once
    daemon._worker()

    assert calls == 2
    assert daemon.items.unfinished_tasks == 0
    assert daemon.health.recording_failed is False
