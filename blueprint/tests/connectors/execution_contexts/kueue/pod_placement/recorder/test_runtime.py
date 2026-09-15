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

from attribution_blueprint.connectors.execution_contexts.kueue.helpers import OwnerIdentity, WorkloadIdentity
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder.runtime import (
    PodPlacementRecorder,
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
