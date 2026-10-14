# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

from attribution_blueprint.kubernetes.daemon.models import ResourceIdentity
from attribution_blueprint.kubernetes.daemon.resolvers import ResolutionPending
from attribution_blueprint.kubernetes.daemon.runtime import KubernetesDaemon
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

    def append_observation(self, observation):
        if observation.resource_version == self.resource_version and not self.failed:
            self.failed = True
            raise OSError("synthetic journal failure")
        return self.delegate.append_observation(observation)

    def create(self, record):
        return self.delegate.create(record)


class _NoWaitEvent(threading.Event):
    def wait(self, timeout=None):
        return self.is_set()


def test_records_placement_before_any_resolution(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")

    daemon._record(_pod())

    record = store.get_by_pod_uid("pod-uid-1")
    assert record is not None
    assert record.node_name == "node-a"
    assert record.owner is None
    assert record.associated_resources == ()


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
    daemon.items.put(_pod())
    daemon.items.put(_pod())
    daemon.stop_event.set()
    calls = 0

    def fail_once(_pod):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic processing failure")

    daemon._record = fail_once
    daemon._worker()

    assert calls == 2
    assert daemon.items.unfinished_tasks == 0
    assert daemon.health.recording_failed is False


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
    assert daemon.items.empty()
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
    assert daemon.items.get_nowait()["metadata"]["uid"] == "pod-uid-1"


def test_saturated_placement_queue_does_not_block_journal_capture(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        cluster_id="cluster-a",
        queue_size=1,
    )
    daemon.items.put({"already": "queued"})
    pod = _pod()
    pod["metadata"]["resourceVersion"] = "18"

    daemon._capture(_PodWatchSource(), pod, "MODIFIED")

    observations = store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1")
    assert len(observations) == 1
    assert daemon.health.queue_saturated is True


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
