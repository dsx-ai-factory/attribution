# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest
from kubernetes.client.exceptions import ApiException

from attribution_blueprint.connectors.execution_contexts.kueue.helpers import OwnerIdentity, WorkloadIdentity
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import PlacementRecord
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.runtime import (
    KubernetesIdentityResolver,
    PodPlacementRecorder,
    ResolutionPending,
    StaleOwnerReference,
    _normalize_resource,
)
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.sqlite_store import SQLitePlacementStore
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.store import CreateResult


def _pod(*, name="pod-a", uid="pod-uid-1", node="node-a", status=None):
    return {
        "metadata": {"namespace": "ns", "name": name, "uid": uid},
        "spec": {"nodeName": node},
        "status": status or {},
    }


def _make_recorder(mocker, resolve_return):
    resolver = mocker.Mock()
    resolver.resolve.return_value = resolve_return
    store = mocker.Mock()
    store.get_by_pod_uid.return_value = None
    return PodPlacementRecorder(core_api=mocker.Mock(), resolver=resolver, store=store, namespace="ns"), store


def _resolved():
    return (
        WorkloadIdentity(name="wl-a", uid="wl-uid-1"),
        OwnerIdentity(api_version="batch/v1", kind="Job", name="job-a", uid="job-uid-1"),
        "owner-uid",
    )


def _owned_pod(*, annotation=True):
    pod = _pod()
    pod["metadata"]["ownerReferences"] = [
        {"apiVersion": "batch/v1", "kind": "Job", "name": "job-a", "uid": "job-uid-1", "controller": True}
    ]
    if annotation:
        pod["metadata"]["annotations"] = {"kueue.x-k8s.io/workload": "wl-a"}
    return pod


def _workload(owner_kind="Job", owner_name="job-a", owner_uid="job-uid-1"):
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "Workload",
        "metadata": {
            "namespace": "ns",
            "name": "wl-a",
            "uid": "wl-uid-1",
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1" if owner_kind == "Job" else "jobset.x-k8s.io/v1alpha2",
                    "kind": owner_kind,
                    "name": owner_name,
                    "uid": owner_uid,
                    "controller": True,
                }
            ],
        },
    }


class TestNormalizeResourceTimestamps:
    """G1: the Kubernetes SDK's `.to_dict()` leaves timestamps as `datetime`
    objects; `_normalize_resource` must make them JSON-safe, not just
    rename the key."""

    def test_condition_last_transition_time_becomes_rfc3339_string(self):
        raw = {
            "metadata": {"namespace": "ns", "name": "pod-a", "uid": "pod-uid-1"},
            "spec": {"node_name": "node-a"},
            "status": {
                "conditions": [
                    {
                        "type": "PodScheduled",
                        "status": "True",
                        "last_transition_time": datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
                    }
                ]
            },
        }

        normalized = _normalize_resource(raw)

        scheduled = normalized["status"]["conditions"][0]["lastTransitionTime"]
        assert scheduled == "2026-05-01T12:00:00Z"
        assert isinstance(scheduled, str)

    def test_real_v1pod_style_object_round_trips_through_sqlite(self, tmp_path):
        """End-to-end regression: a real `V1Pod`-shaped `.to_dict()` result
        (datetime timestamps, not strings) must normalize, record, survive
        a store reopen, and read back with an RFC3339 string, not a
        datetime or a `json.dumps` crash."""

        class _FakePod:
            """Stands in for `kubernetes.client.V1Pod.to_dict()`'s output shape."""

            def to_dict(self):
                return {
                    "metadata": {
                        "namespace": "ns",
                        "name": "pod-a",
                        "uid": "pod-uid-1",
                        "owner_references": [
                            {"api_version": "batch/v1", "kind": "Job", "name": "job-a", "uid": "job-uid-1"}
                        ],
                    },
                    "spec": {"node_name": "node-a"},
                    "status": {
                        "conditions": [
                            {
                                "type": "PodScheduled",
                                "status": "True",
                                "last_transition_time": datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
                            }
                        ]
                    },
                }

        normalized = _normalize_resource(_FakePod().to_dict())

        record = PlacementRecord.for_pod(
            normalized,
            WorkloadIdentity(name="wl-a", uid="wl-uid-1"),
            OwnerIdentity(api_version="batch/v1", kind="Job", name="job-a", uid="job-uid-1"),
            "owner-uid",
        )
        assert record.scheduled_at == "2026-05-01T12:00:00Z"

        path = tmp_path / "placements.db"
        SQLitePlacementStore(str(path)).create(record)  # must not raise TypeError from json.dumps

        reopened = SQLitePlacementStore(str(path)).list_for_workload("ns", "wl-uid-1")
        assert reopened[0].scheduled_at == "2026-05-01T12:00:00Z"
        assert isinstance(reopened[0].scheduled_at, str)


@pytest.mark.parametrize(("field", "kwargs"), [("queue_size", {"queue_size": 0}), ("workers", {"workers": 0})])
def test_recorder_rejects_non_positive_capacity(mocker, field, kwargs):
    with pytest.raises(ValueError, match=field):
        PodPlacementRecorder(
            core_api=mocker.Mock(),
            resolver=mocker.Mock(),
            store=mocker.Mock(),
            namespace="ns",
            **kwargs,
        )


class TestKubernetesIdentityResolver:
    def test_resolves_annotated_job(self, mocker):
        batch_api = mocker.Mock()
        batch_api.read_namespaced_job.return_value.to_dict.return_value = {
            "metadata": {"namespace": "ns", "name": "job-a", "uid": "job-uid-1"}
        }
        custom_api = mocker.Mock()
        custom_api.get_namespaced_custom_object.return_value = _workload()

        result = KubernetesIdentityResolver(batch_api, custom_api).resolve(_owned_pod())

        assert result == (*_resolved()[:2], "kueue-workload-annotation")
        custom_api.get_namespaced_custom_object.assert_called_once_with(
            "kueue.x-k8s.io", "v1beta1", "ns", "workloads", "wl-a"
        )

    def test_resolves_jobset_by_owner_uid(self, mocker):
        batch_api = mocker.Mock()
        batch_api.read_namespaced_job.return_value.to_dict.return_value = {
            "metadata": {
                "namespace": "ns",
                "name": "job-a",
                "uid": "job-uid-1",
                "owner_references": [
                    {
                        "api_version": "jobset.x-k8s.io/v1alpha2",
                        "kind": "JobSet",
                        "name": "set-a",
                        "uid": "set-uid-1",
                        "controller": True,
                    }
                ],
            }
        }
        jobset = {
            "apiVersion": "jobset.x-k8s.io/v1alpha2",
            "kind": "JobSet",
            "metadata": {"namespace": "ns", "name": "set-a", "uid": "set-uid-1"},
        }
        custom_api = mocker.Mock()
        custom_api.get_namespaced_custom_object.return_value = jobset
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [_workload("JobSet", "set-a", "set-uid-1")]
        }

        workload, owner, source = KubernetesIdentityResolver(batch_api, custom_api).resolve(
            _owned_pod(annotation=False)
        )

        assert workload == WorkloadIdentity(name="wl-a", uid="wl-uid-1")
        assert owner == OwnerIdentity("jobset.x-k8s.io/v1alpha2", "JobSet", "set-a", "set-uid-1")
        assert source == "owner-uid"
        custom_api.list_namespaced_custom_object.assert_called_once_with(
            "kueue.x-k8s.io",
            "v1beta1",
            "ns",
            "workloads",
            label_selector="kueue.x-k8s.io/job-uid=set-uid-1",
        )

    def test_treats_eventual_consistency_404_as_pending(self, mocker):
        batch_api = mocker.Mock()
        batch_api.read_namespaced_job.side_effect = ApiException(status=404)

        with pytest.raises(ResolutionPending, match="not visible yet"):
            KubernetesIdentityResolver(batch_api, mocker.Mock()).resolve(_owned_pod())

    def test_reused_job_name_with_different_uid_raises_stale_owner_reference(self, mocker):
        """G4: the Pod's controller reference points at `job-uid-1`, but the
        Job namespace/name lookup (the only lookup kubectl/the SDK offer)
        returns a *different* object — e.g. the original Job was deleted
        and a same-named replacement now exists. The mismatch must not be
        trusted as this Pod's owner."""
        batch_api = mocker.Mock()
        batch_api.read_namespaced_job.return_value.to_dict.return_value = {
            "metadata": {"namespace": "ns", "name": "job-a", "uid": "job-uid-1-REPLACEMENT"}
        }

        with pytest.raises(StaleOwnerReference, match="uid mismatch"):
            KubernetesIdentityResolver(batch_api, mocker.Mock()).resolve(_owned_pod())

    def test_reused_jobset_name_with_different_uid_raises_stale_owner_reference(self, mocker):
        """G4: the Job's JobSet controller reference points at
        `set-uid-1`, but the JobSet namespace/name lookup returns a
        different object — e.g. the original JobSet was deleted and a
        same-named replacement now exists."""
        batch_api = mocker.Mock()
        batch_api.read_namespaced_job.return_value.to_dict.return_value = {
            "metadata": {
                "namespace": "ns",
                "name": "job-a",
                "uid": "job-uid-1",
                "owner_references": [
                    {
                        "api_version": "jobset.x-k8s.io/v1alpha2",
                        "kind": "JobSet",
                        "name": "set-a",
                        "uid": "set-uid-1",
                        "controller": True,
                    }
                ],
            }
        }
        custom_api = mocker.Mock()
        custom_api.get_namespaced_custom_object.return_value = {
            "apiVersion": "jobset.x-k8s.io/v1alpha2",
            "kind": "JobSet",
            "metadata": {"namespace": "ns", "name": "set-a", "uid": "set-uid-1-REPLACEMENT"},
        }

        with pytest.raises(StaleOwnerReference, match="uid mismatch"):
            KubernetesIdentityResolver(batch_api, custom_api).resolve(_owned_pod(annotation=False))


class TestRecordSkipsStaleOwnerReference:
    """G4: `_record` must skip (not retry, not crash) a Pod whose owner
    resolution hits a `StaleOwnerReference`, since the mismatch is
    permanent for that Pod."""

    def test_stale_job_owner_reference_is_skipped_without_recording(self, mocker):
        recorder, store = _make_recorder(mocker, None)
        recorder.resolver.resolve.side_effect = StaleOwnerReference("job-a uid mismatch")
        warning = mocker.patch(
            "attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.runtime.logger.warning"
        )

        recorder._record(_pod())

        store.create.assert_not_called()
        recorder.resolver.resolve.assert_called_once()  # no retry
        assert "stale owner reference" in warning.call_args.args[0]


def test_run_relists_after_expired_watch_and_processes_events(mocker):
    recorder, _ = _make_recorder(mocker, _resolved())
    listed_pod = _pod(name="listed")
    watched_pod = _pod(name="watched")
    listing = mocker.Mock()
    listing.metadata.resource_version = "rv-1"
    listing.items = [listed_pod]
    recorder.core_api.list_namespaced_pod.return_value = listing
    enqueue = mocker.patch.object(recorder, "_enqueue")
    mocker.patch(
        "attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.runtime.threading.Thread"
    )
    watcher = mocker.Mock()

    def final_watch():
        yield {"object": watched_pod}
        recorder.stop()

    watcher.stream.side_effect = [ApiException(status=410), final_watch()]
    mocker.patch("kubernetes.watch.Watch", return_value=watcher)

    recorder.run()

    assert recorder.core_api.list_namespaced_pod.call_count == 2
    assert enqueue.call_args_list == [mocker.call(listed_pod), mocker.call(listed_pod), mocker.call(watched_pod)]
    assert recorder.health.ready()


def test_watch_requests_bookmarks(mocker):
    """Bookmarks advance the watch's resourceVersion in a quiet namespace,
    so a restart doesn't immediately hit a stale-resourceVersion 410."""
    recorder, _ = _make_recorder(mocker, _resolved())
    listing = mocker.Mock()
    listing.metadata.resource_version = "rv-1"
    listing.items = []
    recorder.core_api.list_namespaced_pod.return_value = listing
    mocker.patch.object(recorder, "_enqueue")
    mocker.patch(
        "attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.runtime.threading.Thread"
    )
    watcher = mocker.Mock()

    def stop_immediately(*_args, **_kwargs):
        recorder.stop()
        return iter(())

    watcher.stream.side_effect = stop_immediately
    mocker.patch("kubernetes.watch.Watch", return_value=watcher)

    recorder.run()

    assert watcher.stream.call_args.kwargs["allow_watch_bookmarks"] is True


def test_readiness_survives_expired_watch_relist(mocker):
    """SQLite-backed placement records remain readable while the watch is
    relisting after a 410, so readiness must not flip false mid-relist —
    only startup and queue saturation should gate it."""
    recorder, _ = _make_recorder(mocker, _resolved())
    listing = mocker.Mock()
    listing.metadata.resource_version = "rv-1"
    listing.items = []
    recorder.core_api.list_namespaced_pod.return_value = listing
    mocker.patch.object(recorder, "_enqueue")
    mocker.patch(
        "attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.runtime.threading.Thread"
    )
    watcher = mocker.Mock()
    readiness_during_relist: list[bool] = []
    calls = 0

    def stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ApiException(status=410)
        readiness_during_relist.append(recorder.health.ready())
        recorder.stop()
        return iter(())

    watcher.stream.side_effect = stream
    mocker.patch("kubernetes.watch.Watch", return_value=watcher)

    recorder.run()

    assert readiness_during_relist == [True]


class TestRecordCapturesFailureEvidence:
    def test_existing_record_reuses_identity_without_resolving_owner(self, mocker):
        recorder, store = _make_recorder(mocker, _resolved())
        existing = PlacementRecord(
            namespace="ns",
            workload=_resolved()[0],
            owner=_resolved()[1],
            pod_name="pod-a",
            pod_uid="pod-uid-1",
            node_name="node-a",
            observed_at="2026-09-14T00:00:00Z",
            resolver_source="owner-uid",
        )
        store.get_by_pod_uid.return_value = existing
        store.create.return_value = CreateResult.UPDATED

        recorder._record(_pod(status={"reason": "NodeLost"}))

        recorder.resolver.resolve.assert_not_called()
        record = store.create.call_args.args[0]
        assert record.workload == existing.workload
        assert record.owner == existing.owner
        assert record.node_failure_reason == "NodeLost"

    def test_lookup_failure_on_existing_record_path_is_caught(self, mocker):
        """A store error (e.g. `database is locked`) on the fast path must
        not escape `_record` — an uncaught exception here would kill the
        worker thread permanently, since `_worker` has no catch-all around
        `_record`."""
        recorder, store = _make_recorder(mocker, _resolved())
        store.get_by_pod_uid.side_effect = RuntimeError("database is locked")

        recorder._record(_pod())  # must not raise

        recorder.resolver.resolve.assert_not_called()
        store.create.assert_not_called()

    def test_create_failure_on_existing_record_path_is_caught(self, mocker):
        """Same as above, but the failure happens on the write instead of
        the lookup."""
        recorder, store = _make_recorder(mocker, _resolved())
        store.get_by_pod_uid.return_value = PlacementRecord(
            namespace="ns",
            workload=_resolved()[0],
            owner=_resolved()[1],
            pod_name="pod-a",
            pod_uid="pod-uid-1",
            node_name="node-a",
            observed_at="2026-09-14T00:00:00Z",
            resolver_source="owner-uid",
        )
        store.create.side_effect = RuntimeError("database is locked")

        recorder._record(_pod())  # must not raise

        recorder.resolver.resolve.assert_not_called()

    def test_healthy_pod_records_no_failure_reason(self, mocker):
        recorder, store = _make_recorder(mocker, _resolved())
        store.create.return_value = CreateResult.CREATED

        recorder._record(_pod())

        record = store.create.call_args.args[0]
        assert record.node_failure_reason is None

    def test_node_lost_pod_records_failure_reason(self, mocker):
        recorder, store = _make_recorder(mocker, _resolved())
        store.create.return_value = CreateResult.CREATED

        recorder._record(_pod(status={"reason": "NodeLost"}))

        record = store.create.call_args.args[0]
        assert record.node_failure_reason == "NodeLost"

    def test_disruption_target_pod_records_failure_reason(self, mocker):
        recorder, store = _make_recorder(mocker, _resolved())
        store.create.return_value = CreateResult.CREATED

        pod = _pod(
            status={
                "conditions": [
                    {
                        "type": "DisruptionTarget",
                        "status": "True",
                        "reason": "DeletionByPodGC",
                        "lastTransitionTime": "t",
                    },
                ]
            }
        )
        recorder._record(pod)

        record = store.create.call_args.args[0]
        assert record.node_failure_reason == "DeletionByPodGC"

    def test_nonzero_exit_code_records_failure_reason(self, mocker):
        recorder, store = _make_recorder(mocker, _resolved())
        store.create.return_value = CreateResult.CREATED

        pod = _pod(status={"containerStatuses": [{"state": {"terminated": {"exitCode": 1}}}]})
        recorder._record(pod)

        record = store.create.call_args.args[0]
        assert record.node_failure_reason == "ExitCode:1"

    def test_node_loss_takes_precedence_over_exit_code(self, mocker):
        """When both signals are present, the node-loss reason is more
        specific and is preferred (matches `_collect_pod_evidence`'s
        `failed_nodes` set, which folds both into one undifferentiated
        signal — but the recorder retains the more informative one)."""
        recorder, store = _make_recorder(mocker, _resolved())
        store.create.return_value = CreateResult.CREATED

        pod = _pod(
            status={
                "reason": "NodeLost",
                "containerStatuses": [{"state": {"terminated": {"exitCode": 1}}}],
            }
        )
        recorder._record(pod)

        record = store.create.call_args.args[0]
        assert record.node_failure_reason == "NodeLost"

    def test_update_result_is_logged_with_reason(self, mocker):
        recorder, store = _make_recorder(mocker, _resolved())
        store.create.return_value = CreateResult.UPDATED
        info = mocker.patch(
            "attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.runtime.logger.info"
        )

        recorder._record(_pod(status={"reason": "NodeLost"}))

        message = info.call_args.args[0]
        assert "updated" in message
        assert "NodeLost" in message
