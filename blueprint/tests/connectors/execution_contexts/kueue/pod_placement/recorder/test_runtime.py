# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest
from kubernetes.client.exceptions import ApiException

from attribution_blueprint.connectors.execution_contexts.kueue.helpers import OwnerIdentity, WorkloadIdentity
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.runtime import (
    KubernetesIdentityResolver,
    PodPlacementRecorder,
    ResolutionPending,
)
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


class TestRecordCapturesFailureEvidence:
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
