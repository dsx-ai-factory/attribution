# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

from attribution_blueprint.kubernetes.daemon.models import PlacementRecord, ResourceIdentity
from attribution_blueprint.kubernetes.daemon.resolvers import ResolutionPending
from attribution_blueprint.kubernetes.daemon.runtime import KubernetesDaemon
from attribution_blueprint.kubernetes.daemon.sqlite_store import SQLitePlacementStore
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


def _initialize_pod_coverage(store) -> None:
    store.complete_watch_list(
        cluster_id="cluster-a",
        namespace="team-a",
        source_name="pods",
        api_version="v1",
        kind="Pod",
        resource_version="10",
        completed_at="2026-10-14T00:00:00Z",
        gap_id=None,
    )


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
        raise OSError("synthetic storage failure")

    def get_by_pod_uid(self, pod_uid):
        return PlacementRecord.for_pod(_pod())


class _PodWatchSource:
    name = "pods"
    api_version = "v1"
    kind = "Pod"

    def selected_status(self, resource):
        return selected_status(self.kind, resource)


class _FinitePodWatchSource(_PodWatchSource):
    def __init__(self, snapshots, events, *, list_resource_version="10"):
        self.snapshots = snapshots
        self.events = events
        self.list_resource_version = list_resource_version
        self.list_calls = 0
        self.stream_resource_versions = []
        self.daemon = None

    def list_resources(self):
        self.list_calls += 1
        return self.snapshots, self.list_resource_version

    def stream(self, resource_version, timeout_seconds):
        self.stream_resource_versions.append(resource_version)
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


class _ExpiredWatchSource(_PodWatchSource):
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.stream_calls = 0
        self.readiness_during_lists = []
        self.daemon = None

    def list_resources(self):
        self.readiness_during_lists.append(self.daemon.health.ready())
        return [self.snapshot], "10"

    def stream(self, resource_version, timeout_seconds):
        from kubernetes.client.exceptions import ApiException

        self.stream_calls += 1
        if self.stream_calls == 1:
            raise ApiException(status=410, reason="resource version expired")
        self.daemon.stop()
        return ()


class _AbsentJobSetWatchSource(_PodWatchSource):
    name = "jobsets"
    api_version = "jobset.x-k8s.io/v1alpha2"
    kind = "JobSet"

    def list_resources(self):
        from kubernetes.client.exceptions import ApiException

        raise ApiException(status=404, reason="the server could not find the requested resource")


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


class _FailPlacementOnceStore:
    """The first atomic capture containing a placement fails."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.capture_calls = 0

    def capture_observation(self, observation, **kwargs):
        self.capture_calls += 1
        if self.capture_calls == 1 and kwargs.get("placement") is not None:
            raise OSError("synthetic placement storage failure")
        return self.delegate.capture_observation(observation, **kwargs)

    def get_by_pod_uid(self, pod_uid):
        return self.delegate.get_by_pod_uid(pod_uid)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class _NoWaitEvent(threading.Event):
    def wait(self, timeout=None):
        return self.is_set()


def test_records_placement_before_any_resolution(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a")
    pod = _pod()
    pod["metadata"]["resourceVersion"] = "1"

    daemon._capture(_PodWatchSource(), pod, "SNAPSHOT")

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

    pod = _pod()
    pod["metadata"]["resourceVersion"] = "1"
    daemon._capture(_PodWatchSource(), pod, "SNAPSHOT")
    daemon._record(pod)

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

    assert store.calls == 1
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
    _initialize_pod_coverage(store)
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
    assert daemon.items.get_nowait()["metadata"]["uid"] == "pod-uid-1"


def test_saturated_enrichment_queue_does_not_lose_placement(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    _initialize_pod_coverage(store)
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

    assert daemon._capture(_PodWatchSource(), pod, "MODIFIED") is True

    observations = store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1")
    assert len(observations) == 1
    assert daemon.health.queue_saturated is True
    # The enrichment queue being full only defers owner/scheduler
    # association; base placement must still be captured synchronously.
    record = store.get_by_pod_uid("pod-uid-1")
    assert record is not None
    assert record.node_name == "node-a"


def test_placement_persistence_failure_is_reported_and_blocks_advancement(tmp_path):
    journal = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store = _FailPlacementOnceStore(journal)
    daemon = KubernetesDaemon(core_api=None, store=store, namespace="team-a", cluster_id="cluster-a")
    pod = _pod()
    pod["metadata"]["resourceVersion"] = "20"

    # Observation, placement, and checkpoint form one capture transaction.
    # A placement failure therefore reports the whole resource version as
    # uncaptured so the watch loop retries it.
    assert daemon._capture(_PodWatchSource(), pod, "ADDED") is False
    assert daemon.health.persistence_failed is True
    assert store.get_by_pod_uid("pod-uid-1") is None
    assert store.list_observations_for_uid("cluster-a", "team-a", "pod-uid-1") == []


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
    coverage = store.get_watch_coverage("cluster-a", "team-a", "pods")
    assert coverage is not None
    assert coverage.resource_version == "12"
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


def test_daemon_restart_resumes_from_durable_checkpoint_without_relist(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
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
    first_source = _FinitePodWatchSource([snapshot], [event])
    first_daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        cluster_id="cluster-a",
        watch_sources=(first_source,),
    )
    first_source.daemon = first_daemon
    first_daemon._watch_source(first_source)
    assert store.list_coverage_gaps("cluster-a", "team-a") == []

    second_source = _FinitePodWatchSource([snapshot], [])
    second_daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        cluster_id="cluster-a",
        watch_sources=(second_source,),
    )
    second_source.daemon = second_daemon
    second_daemon._watch_source(second_source)

    assert second_source.list_calls == 0
    assert second_source.stream_resource_versions == ["11"]
    assert store.list_coverage_gaps("cluster-a", "team-a") == []


def test_expired_watch_records_coverage_gap_around_relist(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    pod = _pod()
    pod["spec"] = {}
    snapshot = {**pod, "metadata": {**pod["metadata"], "resourceVersion": "10"}}
    source = _ExpiredWatchSource(snapshot)
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        cluster_id="cluster-a",
        watch_sources=(source,),
    )
    source.daemon = daemon

    daemon._watch_source(source)

    assert source.readiness_during_lists == [False, False]
    [gap] = store.list_coverage_gaps("cluster-a", "team-a")
    assert gap.source_name == "pods"
    assert gap.reason == "RESOURCE_VERSION_EXPIRED"
    assert gap.start_resource_version == "10"
    assert gap.end_resource_version == "10"
    assert gap.recovered_at is not None


def test_changed_watch_source_relists_instead_of_reusing_checkpoint(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    store.complete_watch_list(
        cluster_id="cluster-a",
        namespace="team-a",
        source_name="pods",
        api_version="v1",
        kind="Pod",
        resource_version="10",
        completed_at="2026-10-14T00:00:00Z",
        gap_id=None,
    )
    pod = _pod()
    pod["apiVersion"] = "example.io/v2"
    pod["spec"] = {}
    snapshot = {**pod, "metadata": {**pod["metadata"], "resourceVersion": "20"}}
    source = _FinitePodWatchSource(
        [snapshot],
        [],
        list_resource_version="20",
    )
    source.api_version = "example.io/v2"
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        cluster_id="cluster-a",
        watch_sources=(source,),
    )
    source.daemon = daemon

    daemon._watch_source(source)

    assert source.list_calls == 1
    assert source.stream_resource_versions == ["20"]
    [gap] = store.list_coverage_gaps("cluster-a", "team-a")
    assert gap.reason == "WATCH_SOURCE_CHANGED"
    assert gap.start_resource_version == "10"
    assert gap.end_resource_version == "20"


def test_absent_jobset_api_disables_watch_without_blocking_readiness(tmp_path):
    store = SQLitePlacementStore(str(tmp_path / "placements.db"))
    source = _AbsentJobSetWatchSource()
    daemon = KubernetesDaemon(
        core_api=None,
        store=store,
        namespace="team-a",
        cluster_id="cluster-a",
        watch_sources=(source,),
    )

    daemon._watch_source(source)

    assert daemon.health.ready()
    assert daemon.health.disabled_watch_status() == {"jobsets": "api-not-found"}
