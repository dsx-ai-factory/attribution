# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import threading
from types import SimpleNamespace

import pytest

from attribution_blueprint.kubernetes.daemon.models import PlacementRecord, ResourceIdentity
from attribution_blueprint.kubernetes.daemon.resolvers import ResolutionPending
from attribution_blueprint.kubernetes.daemon.runtime import (
    KubernetesDaemon,
    RequiredPersistenceError,
    _stored_owner_matches_pod,
)
from attribution_blueprint.kubernetes.daemon.sqlite_store import SQLitePlacementStore
from attribution_blueprint.kubernetes.daemon.store import CreateResult
from attribution_blueprint.kubernetes.daemon.watch_sources import selected_status


def _pod() -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "trainer-0",
            "namespace": "team-a",
            "uid": "pod-uid-1",
            "resourceVersion": "1",
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


class _EchoOwnerResolver:
    """Resolve the owner using the Pod's own controller reference.

    Unlike `_OwnerResolver`, the returned identity reflects whichever Pod
    snapshot was actually passed to `resolve`, so tests can prove which
    snapshot the daemon chose to enrich.
    """

    def __init__(self):
        self.calls = 0

    def resolve(self, pod):
        self.calls += 1
        reference = pod["metadata"]["ownerReferences"][0]
        return ResourceIdentity("batch/v1", "Job", "team-a", reference["name"], reference["uid"])


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


class _PendingThenResolvedOwnerResolver:
    def __init__(self):
        self.calls = 0

    def resolve(self, pod):
        self.calls += 1
        if self.calls == 1:
            raise ResolutionPending("owner not visible yet")
        return ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")


class _AlwaysPendingOwnerResolver:
    def __init__(self):
        self.calls = 0

    def resolve(self, pod):
        self.calls += 1
        raise ResolutionPending("owner never appears")


class _TransientlyFailingOwnerResolver:
    """Raises an error not modeled by any named resolver exception, the way
    an unhandled HTTP 503 from the Kubernetes API would surface."""

    def __init__(self):
        self.calls = 0

    def resolve(self, pod):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("Kubernetes API temporarily unavailable (HTTP 503)")
        return ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")


class _TransientlyFailingResourceResolver:
    def __init__(self):
        self.calls = 0

    def resolve(self, pod, owner):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("Kubernetes API temporarily unavailable (HTTP 503)")
        return (ResourceIdentity("scheduling.example.io/v1", "Admission", "team-a", "admission-a", "admission-uid-a"),)


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


class _FailReadOnceStore:
    def __init__(self, delegate):
        self.delegate = delegate
        self.failed = False

    def get_by_pod_uid(self, pod_uid):
        if not self.failed:
            self.failed = True
            raise OSError("synthetic read failure")
        return self.delegate.get_by_pod_uid(pod_uid)

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
    def __init__(self):
        self.calls = 0

    def resolve(self, pod, owner):
        self.calls += 1
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
        self.base = None

    def create(self, record):
        self.calls += 1
        if self.calls == 1:
            self.base = record
            return CreateResult.CREATED
        raise OSError("synthetic storage failure")

    def get_by_pod_uid(self, pod_uid):
        return self.base


class _FailEnrichmentOnceStore:
    def __init__(self, delegate):
        self.delegate = delegate
        self.failed = False

    def create(self, record):
        if record.owner is not None and not self.failed:
            self.failed = True
            raise OSError("synthetic enrichment write failure")
        return self.delegate.create(record)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _FailCreateOnceStore:
    def __init__(self, delegate):
        self.delegate = delegate
        self.failed = False

    def capture_observation(self, observation, **kwargs):
        if not self.failed:
            self.failed = True
            raise OSError("synthetic placement write failure")
        return self.delegate.capture_observation(observation, **kwargs)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _FailDeletionOnceStore:
    def __init__(self, delegate):
        self.delegate = delegate
        self.failed = False

    def mark_deleted(self, pod_uid):
        if not self.failed:
            self.failed = True
            raise OSError("synthetic deletion write failure")
        return self.delegate.mark_deleted(pod_uid)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _FailReconcileOnceStore:
    def __init__(self, delegate):
        self.delegate = delegate
        self.failed = False

    def reconcile_active_pods(self, namespace, active_pod_uids):
        if not self.failed:
            self.failed = True
            raise OSError("synthetic reconciliation failure")
        return self.delegate.reconcile_active_pods(namespace, active_pod_uids)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _PodWatchSource:
    name = "pods"
    api_version = "v1"
    kind = "Pod"

    def selected_status(self, resource):
        return selected_status(self.kind, resource)


class _FinitePodWatchSource(_PodWatchSource):
    def __init__(self, snapshots, events):
        self.snapshots = snapshots
        self.events = events
        self.daemon = None

    def list_resources(self):
        return self.snapshots, "10"

    def stream(self, resource_version, timeout_seconds):
        assert resource_version == "10"
        yield from self.events
        self.daemon.stop()


class _ReplayPodWatchSource(_PodWatchSource):
    def __init__(self, snapshot, event):
        self.snapshot = snapshot
        self.event = event
        self.stream_resource_versions = []
        self.daemon = None

    def list_resources(self):
        return [self.snapshot], "10"

    def stream(self, resource_version, timeout_seconds):
        self.stream_resource_versions.append(resource_version)
        yield self.event
        if len(self.stream_resource_versions) > 1:
            self.daemon.stop()


class _RetrySnapshotSource(_PodWatchSource):
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.list_calls = 0
        self.daemon = None

    def list_resources(self):
        self.list_calls += 1
        return [self.snapshot], "10"

    def stream(self, resource_version, timeout_seconds):
        self.daemon.stop()
        return ()


class _FailObservationOnceStore:
    def __init__(self, delegate, resource_version):
        self.delegate = delegate
        self.resource_version = resource_version
        self.failed = False

    def capture_observation(self, observation, **kwargs):
        if observation.resource_version == self.resource_version and not self.failed:
            self.failed = True
            raise OSError("synthetic journal failure")
        return self.delegate.capture_observation(observation, **kwargs)

    def create(self, record):
        return self.delegate.create(record)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _NoWaitEvent(threading.Event):
    def wait(self, timeout=None):
        return self.is_set()


def test_records_placement_before_any_resolution(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")

    daemon._capture_placement(_pod())

    record = store.get_by_pod_uid("pod-uid-1")
    assert record is not None
    assert record.node_name == "node-a"
    assert record.owner is None
    assert record.associated_resources == ()


def test_default_retry_window_is_four_seconds(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")

    assert daemon.retry_poll_seconds * (daemon.max_resolution_attempts - 1) == 4


def test_transient_placement_read_failure_still_enriches_owner(tmp_path):
    backing_store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store = _FailReadOnceStore(backing_store)
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_OwnerResolver(),
    )
    daemon._capture_placement(_pod())

    daemon._record(_pod())

    assert backing_store.get_by_pod_uid("pod-uid-1").owner.uid == "job-uid-1"


def test_complete_listing_reconciles_absent_placements(mocker):
    store = mocker.Mock()
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    pod = _pod()

    daemon._reconcile_active_pods([pod])

    store.reconcile_active_pods.assert_called_once_with(
        "team-a",
        {"pod-uid-1"},
    )


def test_relist_marks_queued_but_now_absent_pod_deleted(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a", cluster_id="cluster-a")
    pod = _pod()
    pod["metadata"]["resourceVersion"] = "42"

    daemon._capture(_PodWatchSource(), pod, "SNAPSHOT")
    daemon._reconcile_active_pods([])

    assert store.get_by_pod_uid("pod-uid-1") is not None
    with sqlite3.connect(path) as connection:
        marker = connection.execute(
            "SELECT pod_uid FROM placement_deletion"
        ).fetchone()
    assert marker == ("pod-uid-1",)


def test_failed_reconciliation_replays_listing_and_keeps_deletion_marker(tmp_path):
    path = tmp_path / "placements.db"
    backing_store = SQLitePlacementStore(str(path))
    store = _FailReconcileOnceStore(backing_store)
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    daemon._enqueue(_pod())

    with pytest.raises(RequiredPersistenceError):
        daemon._reconcile_active_pods([])

    assert daemon.health.persistence_failed is True
    daemon._reconcile_active_pods([])
    with sqlite3.connect(path) as connection:
        marker = connection.execute(
            "SELECT pod_uid FROM placement_deletion"
        ).fetchone()
    assert marker == ("pod-uid-1",)
    assert daemon.health.persistence_failed is False


def test_saturation_preserves_admitted_retry_and_skips_new_uid(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        queue_size=2,
    )
    daemon.health.update(initial_list_complete=True)
    # Both capacity slots are already occupied by admitted work.
    daemon._in_flight.add("occupying-uid")
    other_retry = _pod()
    other_retry["metadata"]["uid"] = "pod-uid-other"
    daemon._retry["pod-uid-other"] = (other_retry, -1)
    replacement = _pod()
    replacement["metadata"]["uid"] = "pod-uid-2"
    replacement["metadata"]["name"] = "trainer-1"

    daemon._enqueue(replacement)

    # Placement is never subject to enrichment capacity: it is persisted
    # synchronously regardless of overflow.
    record = store.get_by_pod_uid("pod-uid-2")
    assert record is not None
    assert record.node_name == "node-a"
    # Existing recovery work is preserved; the new distinct UID is skipped.
    assert "pod-uid-other" in daemon._retry
    assert "pod-uid-2" not in daemon._retry
    assert "pod-uid-2" not in daemon._pending
    assert daemon.health.ready() is True


def test_admitted_deferred_enrichment_does_not_fail_readiness(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_AlwaysPendingOwnerResolver(),
        queue_size=1,
    )
    daemon.health.update(initial_list_complete=True)
    daemon._enqueue(_pod())

    # Full enrichment capacity does not make retained reads unavailable.
    assert daemon.health.ready() is True

    daemon.stop_event.set()
    daemon._worker()

    assert "pod-uid-1" in daemon._retry
    assert daemon.health.ready() is True


def test_saturation_skips_new_uid_when_active_work_fills_capacity(tmp_path):
    """New enrichment is skipped when admitted work fills capacity.

    Base placement is still persisted.
    """
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        queue_size=1,
    )
    daemon._in_flight.add("occupying-uid")
    replacement = _pod()
    replacement["metadata"]["uid"] = "pod-uid-2"
    replacement["metadata"]["name"] = "trainer-1"

    daemon._enqueue(replacement)

    assert store.get_by_pod_uid("pod-uid-2") is not None
    assert "pod-uid-2" not in daemon._retry
    assert "pod-uid-2" not in daemon._pending
    assert "pod-uid-2" not in daemon._latest_generation


def test_latest_pod_snapshot_always_wins_without_controller_tracking(tmp_path):
    """A later event for the same Pod UID replaces the earlier snapshot
    outright, even at capacity 1 and even across a controller change. There
    is no special "controller changed" or "controller reverted" case: the
    daemon only ever tracks one latest snapshot per Pod UID."""
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    resolver = _EchoOwnerResolver()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=resolver,
        queue_size=1,
    )
    first = _pod()
    first["metadata"]["resourceVersion"] = "10"
    reassigned = _pod()
    reassigned["metadata"]["ownerReferences"][0].update({"name": "trainer-2", "uid": "job-uid-2"})
    reassigned["metadata"]["resourceVersion"] = "11"
    daemon._capture(_PodWatchSource(), first, "SNAPSHOT")

    daemon._capture(_PodWatchSource(), reassigned, "SNAPSHOT")

    # Updating an already-tracked Pod UID never consumes extra capacity.
    assert daemon._pending["pod-uid-1"][0] == reassigned
    daemon.stop_event.set()
    daemon._worker()

    assert resolver.calls == 1
    assert store.get_by_pod_uid("pod-uid-1").owner.uid == "job-uid-2"


def test_recording_attempt_does_not_retry_scheduler_resolution_inline(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    resolver = _PendingResourceResolver()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_OwnerResolver(),
        resource_resolvers=(resolver,),
        max_resolution_attempts=5,
    )

    daemon._capture_placement(_pod())
    daemon._record(_pod())

    assert resolver.calls == 1
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
    daemon._capture_placement(_pod())
    daemon._record(_pod())
    reassigned = _pod()
    reassigned["metadata"]["ownerReferences"][0]["uid"] = "job-uid-2"

    daemon._capture_placement(reassigned)
    daemon._record(reassigned)
    daemon._capture_placement(reassigned)
    daemon._record(reassigned)

    stored = store.get_by_pod_uid("pod-uid-1")
    assert resolver.calls == 3
    # Retries never block inside a single enrichment attempt; deferred
    # scheduling is entirely the retry loop's responsibility.
    wait.assert_not_called()
    assert stored.owner.uid == "jobset-uid-1"
    assert stored.owner_references[0].uid == "job-uid-2"


def test_jobset_cache_rejects_ambiguous_stored_job_references():
    original = _pod()
    job_a = ResourceIdentity("batch/v1", "Job", "team-a", "trainer-a", "job-uid-a")
    job_b = ResourceIdentity("batch/v1", "Job", "team-a", "trainer-b", "job-uid-b")
    jobset = ResourceIdentity(
        "jobset.x-k8s.io/v1alpha2",
        "JobSet",
        "team-a",
        "training",
        "jobset-uid-1",
    )
    record = PlacementRecord.for_pod(original).enriched(
        owner_references=(job_a, job_b),
        owner=jobset,
    )
    current = _pod()
    current["metadata"]["ownerReferences"][0]["uid"] = "job-uid-b"

    assert _stored_owner_matches_pod(record, current) is False


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
    initial = _pod()
    initial["metadata"]["resourceVersion"] = "1"
    daemon._capture(_PodWatchSource(), initial, "SNAPSHOT")
    daemon._record(initial)
    reassigned = _pod()
    reassigned["metadata"]["ownerReferences"][0]["uid"] = "job-uid-2"
    reassigned["metadata"]["resourceVersion"] = "2"

    daemon._capture(_PodWatchSource(), reassigned, "SNAPSHOT")
    daemon._record(reassigned)

    stored = store.get_by_pod_uid("pod-uid-1")
    assert stored is not None
    assert stored.owner.uid == "jobset-uid-1"
    assert stored.associated_resources == ()
    assert store.list_for_resource("team-a", "admission-uid-a") == []

    resolved = {**reassigned, "metadata": {**reassigned["metadata"], "resourceVersion": "3"}}
    daemon._capture(_PodWatchSource(), resolved, "SNAPSHOT")
    daemon._record(resolved)

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
    older["metadata"]["resourceVersion"] = "10"
    newer["metadata"]["resourceVersion"] = "11"
    newer["metadata"]["ownerReferences"][0].update(
        {"name": "trainer-2", "uid": "job-uid-2"}
    )
    daemon._capture(_PodWatchSource(), older, "SNAPSHOT")
    workers = [threading.Thread(target=daemon._worker) for _ in range(2)]
    for worker in workers:
        worker.start()

    assert resolver.started.wait(timeout=2)
    # The newer event for the same Pod UID cannot start on the other worker
    # while the older one is still in flight; it only runs after release.
    daemon._capture(_PodWatchSource(), newer, "SNAPSHOT")
    resolver.release.set()
    assert store.newer_owner_written.wait(timeout=2)
    daemon.stop_event.set()
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
    older["metadata"]["resourceVersion"] = "10"
    newer = _pod()
    newer["metadata"]["annotations"] = {"event": "new"}
    newer["metadata"]["resourceVersion"] = "11"
    daemon._capture(_PodWatchSource(), older, "SNAPSHOT")
    first = threading.Thread(target=daemon._worker)
    first.start()
    assert resolver.started.wait(timeout=2)

    # The newer event for the same Pod UID cannot start on the second worker
    # while the older one is still in flight; it only runs after release.
    daemon._capture(_PodWatchSource(), newer, "SNAPSHOT")
    second = threading.Thread(target=daemon._worker)
    second.start()
    resolver.release.set()
    assert store.newer_resource_written.wait(timeout=2)
    daemon.stop_event.set()
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

    assert daemon._persist_enrichment(record, "test enrichment", generation=1) == (None, False)
    assert store.get_by_pod_uid("pod-uid-1") is None


def test_failed_capture_preserves_earlier_queued_generation(tmp_path, mocker):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_OwnerResolver(),
    )
    first = _pod()
    first["metadata"]["resourceVersion"] = "10"
    second = _pod()
    second["metadata"]["resourceVersion"] = "11"
    daemon._capture(_PodWatchSource(), first, "SNAPSHOT")
    queued_generation = daemon._latest_generation["pod-uid-1"]
    capture = mocker.patch.object(
        store,
        "capture_observation",
        side_effect=OSError("synthetic storage failure"),
    )

    assert daemon._capture(_PodWatchSource(), second, "SNAPSHOT") is False

    assert daemon._latest_generation["pod-uid-1"] == queued_generation
    mocker.stop(capture)
    daemon.stop_event.set()
    daemon._worker()
    assert store.get_by_pod_uid("pod-uid-1").owner.uid == "job-uid-1"


def test_stale_event_still_persists_failure_and_deletion_evidence(tmp_path):
    path = tmp_path / "placements.db"
    store = SQLitePlacementStore(str(path))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    pod = _pod()
    pod["status"]["reason"] = "NodeLost"
    daemon._latest_generation["pod-uid-1"] = 2

    daemon._capture_placement(pod)
    daemon._mark_deleted("pod-uid-1")
    daemon._record(pod, generation=1)

    assert store.get_by_pod_uid("pod-uid-1").node_failure_reason == "NodeLost"
    with sqlite3.connect(path) as connection:
        marker = connection.execute(
            "SELECT pod_uid FROM placement_deletion"
        ).fetchone()
    assert marker == ("pod-uid-1",)


def test_failed_deletion_write_is_recovered_when_event_is_replayed(tmp_path):
    path = tmp_path / "placements.db"
    backing_store = SQLitePlacementStore(str(path))
    store = _FailDeletionOnceStore(backing_store)
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")

    with pytest.raises(RequiredPersistenceError):
        daemon._enqueue(_pod(), deleted=True)

    assert daemon.health.persistence_failed is True
    daemon._enqueue(_pod(), deleted=True)
    with sqlite3.connect(path) as connection:
        marker = connection.execute(
            "SELECT pod_uid FROM placement_deletion"
        ).fetchone()
    assert marker == ("pod-uid-1",)
    assert daemon.health.persistence_failed is False


def test_watch_replays_event_without_advancing_failed_persistence_checkpoint(tmp_path, mocker):
    backing_store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store = _FailCreateOnceStore(backing_store)
    listing = SimpleNamespace(
        items=[],
        metadata=SimpleNamespace(resource_version="10"),
    )
    core_api = mocker.Mock()
    core_api.list_namespaced_pod.return_value = listing
    daemon = KubernetesDaemon(
        core_api=core_api,
        store=store,
        namespace="team-a",
        workers=1,
    )
    pod = _pod()
    pod["metadata"]["resourceVersion"] = "11"
    event = {"type": "MODIFIED", "object": pod}
    observed_resource_versions = []

    watcher = mocker.Mock()

    def stream(*args, **kwargs):
        observed_resource_versions.append(kwargs["resource_version"])
        yield event
        if len(observed_resource_versions) == 2:
            daemon.stop()

    watcher.stream.side_effect = stream
    mocker.patch("kubernetes.watch.Watch", return_value=watcher)

    daemon.run()

    assert observed_resource_versions == ["10", "10"]
    assert backing_store.get_by_pod_uid("pod-uid-1") is not None
    assert daemon.health.persistence_failed is False


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

    daemon._capture_placement(_pod())
    assert daemon._record(_pod()) is True

    assert store.calls == 2
    assert daemon.health.persistence_failed is True


def test_transient_enrichment_write_failure_retries_without_another_pod_event(tmp_path):
    backing_store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store = _FailEnrichmentOnceStore(backing_store)
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_OwnerResolver(),
    )
    daemon._enqueue(_pod())
    daemon.stop_event.set()

    daemon._worker()

    assert "pod-uid-1" in daemon._retry
    assert backing_store.get_by_pod_uid("pod-uid-1").owner is None

    daemon.stop_event.clear()
    daemon._promote_retries()
    daemon.stop_event.set()
    daemon._worker()

    assert backing_store.get_by_pod_uid("pod-uid-1").owner.uid == "job-uid-1"
    assert "pod-uid-1" not in daemon._retry


def test_retry_promotion_does_not_enqueue_work_after_stop(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    daemon._retry["pod-uid-1"] = (_pod(), 1)
    promotion_started = threading.Event()

    def promote():
        promotion_started.set()
        daemon._promote_retries()

    with daemon._work_available:
        promoter = threading.Thread(target=promote)
        promoter.start()
        assert promotion_started.wait(timeout=2)
        daemon.stop_event.set()

    promoter.join(timeout=2)

    assert not promoter.is_alive()
    assert "pod-uid-1" in daemon._retry
    assert "pod-uid-1" not in daemon._pending


def test_worker_survives_unexpected_recording_failure(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    daemon._enqueue(_pod())
    other = _pod()
    other["metadata"]["uid"] = "pod-uid-2"
    other["metadata"]["name"] = "trainer-1"
    daemon._enqueue(other)
    daemon.stop_event.set()
    calls = 0

    def fail_once(_pod, *, generation=None):
        del generation
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic processing failure")

    daemon._record = fail_once
    daemon._worker()

    assert calls == 2
    assert not daemon._pending and not daemon._in_flight


def test_transient_owner_failure_is_retried_outside_the_worker(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    resolver = _PendingThenResolvedOwnerResolver()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=resolver,
    )
    daemon._enqueue(_pod())
    daemon.stop_event.set()

    daemon._worker()

    # The failed attempt is deferred for retry, not dropped and not retried
    # inline by the worker itself.
    assert resolver.calls == 1
    assert "pod-uid-1" in daemon._retry
    assert "pod-uid-1" not in daemon._pending
    assert store.get_by_pod_uid("pod-uid-1").owner is None

    # Simulate the dedicated retry loop's next tick, outside any worker.
    daemon.stop_event.clear()
    daemon._promote_retries()
    daemon.stop_event.set()
    daemon._worker()

    assert resolver.calls == 2
    assert store.get_by_pod_uid("pod-uid-1").owner.uid == "job-uid-1"


def test_retry_gives_up_after_max_resolution_attempts(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    resolver = _AlwaysPendingOwnerResolver()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=resolver,
        max_resolution_attempts=2,
    )
    daemon._enqueue(_pod())
    daemon.stop_event.set()

    daemon._worker()
    assert "pod-uid-1" in daemon._retry

    daemon.stop_event.clear()
    daemon._promote_retries()
    daemon.stop_event.set()
    daemon._worker()

    assert resolver.calls == 2
    assert "pod-uid-1" not in daemon._retry
    assert "pod-uid-1" not in daemon._pending
    assert store.get_by_pod_uid("pod-uid-1").owner is None


def test_newer_pod_event_waits_instead_of_racing_the_in_flight_attempt(tmp_path):
    """A second event for a Pod UID already in flight must not be picked up
    by another worker concurrently: it waits in `_pending` until the active
    attempt completes."""
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    daemon._enqueue(_pod())
    taken = daemon._take_work(timeout=0.1)
    assert taken is not None
    pod_uid, _older_pod, older_generation = taken

    reassigned = _pod()
    reassigned["metadata"]["ownerReferences"][0]["uid"] = "job-uid-2"
    daemon._enqueue(reassigned)

    # The newer snapshot is queued for this Pod UID, but no worker may start
    # it while the older attempt is still in flight.
    queued_pod, queued_generation = daemon._pending[pod_uid]
    assert queued_pod == reassigned
    assert queued_generation != older_generation
    assert daemon._take_work(timeout=0.1) is None

    daemon._complete_work(pod_uid)

    # Only now does the newer snapshot become available.
    taken_again = daemon._take_work(timeout=0.1)
    assert taken_again is not None
    _pod_uid, newer_pod, newer_generation = taken_again
    assert newer_pod == reassigned
    assert newer_generation != older_generation


def test_in_flight_completion_preserves_generation_for_queued_newer_work(tmp_path):
    """Finishing the older (in-flight) attempt for a Pod UID must not clear
    generation tracking while a newer snapshot for that same UID is still
    queued and waiting to run; otherwise the newer attempt's later write
    would be wrongly rejected as stale."""
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    daemon._capture_placement(_pod())
    daemon._schedule("pod-uid-1", _pod())
    pod_uid, _older_pod, older_generation = daemon._take_work(timeout=0.1)

    reassigned = _pod()
    daemon._schedule("pod-uid-1", reassigned)
    newer_generation = daemon._latest_generation["pod-uid-1"]
    assert newer_generation != older_generation

    # The older attempt finishes (e.g. its own write was rejected as stale).
    daemon._complete_work(pod_uid)

    assert daemon._latest_generation.get("pod-uid-1") == newer_generation
    record = PlacementRecord.for_pod(reassigned).enriched(
        owner=ResourceIdentity("batch/v1", "Job", "team-a", "trainer", "job-uid-1")
    )
    result, retry = daemon._persist_enrichment(record, "test enrichment", newer_generation)
    assert result is not None
    assert retry is False


def test_unexpected_owner_resolution_error_is_retried_not_dropped(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    resolver = _TransientlyFailingOwnerResolver()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=resolver,
    )
    daemon._enqueue(_pod())
    daemon.stop_event.set()

    daemon._worker()

    assert resolver.calls == 1
    assert "pod-uid-1" in daemon._retry
    assert store.get_by_pod_uid("pod-uid-1").owner is None

    daemon.stop_event.clear()
    daemon._promote_retries()
    daemon.stop_event.set()
    daemon._worker()

    assert resolver.calls == 2
    assert store.get_by_pod_uid("pod-uid-1").owner.uid == "job-uid-1"


def test_unexpected_resource_resolution_error_is_retried_not_dropped(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    resolver = _TransientlyFailingResourceResolver()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        owner_resolver=_OwnerResolver(),
        resource_resolvers=(resolver,),
    )
    daemon._enqueue(_pod())
    daemon.stop_event.set()

    daemon._worker()

    assert resolver.calls == 1
    assert "pod-uid-1" in daemon._retry
    assert store.get_by_pod_uid("pod-uid-1").associated_resources == ()

    daemon.stop_event.clear()
    daemon._promote_retries()
    daemon.stop_event.set()
    daemon._worker()

    assert resolver.calls == 2
    assert store.get_by_pod_uid("pod-uid-1").associated_resources[0].uid == "admission-uid-a"


def test_capture_journals_unbound_pod_without_placement(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a", cluster_id="cluster-a")
    pod = _pod()
    pod["metadata"]["resourceVersion"] = "12"
    pod["spec"] = {}
    pod["status"] = {
        "phase": "Pending",
        "containerStatuses": [{"name": "trainer", "restartCount": 0}],
    }

    daemon._capture(_PodWatchSource(), pod, "SNAPSHOT")

    observations = store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1")
    assert len(observations) == 1
    assert observations[0].event_type == "SNAPSHOT"
    assert observations[0].status == {
        "containerStatuses": [{"name": "trainer", "restartCount": 0}],
        "phase": "Pending",
    }
    assert not daemon._pending
    assert store.get_by_pod_uid("pod-uid-1") is None


def test_capture_journals_deletion_before_enqueuing_bound_pod(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a", cluster_id="cluster-a")
    pod = _pod()
    pod["metadata"]["resourceVersion"] = "15"
    pod["metadata"]["deletionTimestamp"] = "2026-10-14T10:00:00Z"

    daemon._capture(_PodWatchSource(), pod, "DELETED")

    observation = store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1")[0]
    assert observation.event_type == "DELETED"
    assert observation.source_timestamp == "2026-10-14T10:00:00Z"
    # Placement is captured synchronously, before enrichment is queued, so it
    # is retained even though this bound Pod is already being deleted.
    record = store.get_by_pod_uid("pod-uid-1")
    assert record is not None
    assert record.node_name == "node-a"
    with sqlite3.connect(tmp_path / "placements.db") as connection:
        marker = connection.execute(
            "SELECT pod_uid FROM placement_deletion"
        ).fetchone()
    assert marker == ("pod-uid-1",)
    assert daemon._pending["pod-uid-1"][0]["metadata"]["uid"] == "pod-uid-1"


def test_saturated_placement_queue_does_not_block_journal_capture(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        cluster_id="cluster-a",
        queue_size=1,
    )
    existing = _pod()
    existing["metadata"]["uid"] = "already-queued"
    daemon._schedule("already-queued", existing)
    pod = _pod()
    pod["metadata"]["resourceVersion"] = "18"

    daemon._capture(_PodWatchSource(), pod, "MODIFIED")

    observations = store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1")
    assert len(observations) == 1
    assert "pod-uid-1" not in daemon._pending
    assert store.get_by_pod_uid("pod-uid-1") is not None


def test_watch_records_snapshots_updates_and_deletions(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    pod = _pod()
    pod["spec"] = {}
    snapshots = [{**pod, "metadata": {**pod["metadata"], "resourceVersion": "10"}}]
    events = [
        {
            "type": "MODIFIED",
            "object": {
                **pod,
                "metadata": {**pod["metadata"], "resourceVersion": "11"},
                "status": {"phase": "Running"},
            },
        },
        {
            "type": "DELETED",
            "object": {
                **pod,
                "metadata": {**pod["metadata"], "resourceVersion": "12"},
                "status": {"phase": "Failed"},
            },
        },
    ]
    source = _FinitePodWatchSource(snapshots, events)
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        cluster_id="cluster-a",
        watch_sources=(source,),
    )
    source.daemon = daemon

    daemon._watch_source(source)

    observations = store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1")
    assert [observation.event_type for observation in observations] == [
        "SNAPSHOT",
        "MODIFIED",
        "DELETED",
    ]
    assert daemon.health.initial_list_complete is True


def test_watch_replays_failed_write_from_last_persisted_resource_version(tmp_path):
    journal = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store = _FailObservationOnceStore(journal, "11")
    pod = _pod()
    pod["spec"] = {}
    snapshot = {**pod, "metadata": {**pod["metadata"], "resourceVersion": "10"}}
    event = {
        "type": "MODIFIED",
        "object": {
            **pod,
            "metadata": {**pod["metadata"], "resourceVersion": "11"},
            "status": {"phase": "Running"},
        },
    }
    source = _ReplayPodWatchSource(snapshot, event)
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        cluster_id="cluster-a",
        watch_sources=(source,),
    )
    daemon.stop_event = _NoWaitEvent()
    source.daemon = daemon

    daemon._watch_source(source)

    observations = journal.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1")
    assert [observation.resource_version for observation in observations] == ["10", "11"]
    assert source.stream_resource_versions == ["10", "10"]
    assert daemon.health.persistence_failed is False


def test_initial_list_is_retried_before_watch_position_advances(tmp_path):
    journal = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store = _FailObservationOnceStore(journal, "10")
    pod = _pod()
    pod["spec"] = {}
    snapshot = {**pod, "metadata": {**pod["metadata"], "resourceVersion": "10"}}
    source = _RetrySnapshotSource(snapshot)
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        cluster_id="cluster-a",
        watch_sources=(source,),
    )
    daemon.stop_event = _NoWaitEvent()
    source.daemon = daemon

    daemon._watch_source(source)

    observations = journal.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1")
    assert [observation.resource_version for observation in observations] == ["10"]
    assert source.list_calls == 2
    assert daemon.health.initial_list_complete is True
    assert daemon.health.persistence_failed is False
