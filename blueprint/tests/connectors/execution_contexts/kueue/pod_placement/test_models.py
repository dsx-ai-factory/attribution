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

from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import (
    OwnerIdentity,
    PlacementRecord,
    WorkloadIdentity,
)


def _pod(*, namespace="ns", name="pod-a", uid="pod-uid-1", node="node-a", status=None):
    return {
        "metadata": {"namespace": namespace, "name": name, "uid": uid},
        "spec": {"nodeName": node},
        "status": status or {},
    }


def _identities():
    return (
        WorkloadIdentity(name="wl-a", uid="wl-uid-1"),
        OwnerIdentity(api_version="batch/v1", kind="Job", name="job-a", uid="job-uid-1"),
    )


class TestForPod:
    def test_requires_namespace_name_uid_and_node(self):
        workload, owner = _identities()
        pod = {"metadata": {}, "spec": {}}
        with pytest.raises(ValueError, match="missing namespace, name, UID, or nodeName"):
            PlacementRecord.for_pod(pod, workload, owner, "owner-uid")

    def test_node_failure_reason_defaults_to_none(self):
        workload, owner = _identities()
        record = PlacementRecord.for_pod(_pod(), workload, owner, "owner-uid")
        assert record.node_failure_reason is None

    def test_node_failure_reason_is_stored_when_provided(self):
        workload, owner = _identities()
        record = PlacementRecord.for_pod(_pod(), workload, owner, "owner-uid", node_failure_reason="NodeLost")
        assert record.node_failure_reason == "NodeLost"

    def test_scheduled_at_read_from_pod_scheduled_condition(self):
        workload, owner = _identities()
        pod = _pod(
            status={
                "conditions": [
                    {"type": "PodScheduled", "status": "True", "lastTransitionTime": "2026-05-01T00:00:00Z"},
                ]
            }
        )
        record = PlacementRecord.for_pod(pod, workload, owner, "owner-uid")
        assert record.scheduled_at == "2026-05-01T00:00:00Z"


class TestResourceRoundTrip:
    def test_round_trip_without_node_failure_reason(self):
        workload, owner = _identities()
        record = PlacementRecord.for_pod(_pod(), workload, owner, "owner-uid")
        resource = record.to_resource()
        assert "nodeFailureReason" not in resource["spec"]
        assert PlacementRecord.from_resource(resource).node_failure_reason is None

    def test_round_trip_with_node_failure_reason(self):
        workload, owner = _identities()
        record = PlacementRecord.for_pod(_pod(), workload, owner, "owner-uid", node_failure_reason="DeletionByPodGC")
        resource = record.to_resource()
        assert resource["spec"]["nodeFailureReason"] == "DeletionByPodGC"
        assert PlacementRecord.from_resource(resource).node_failure_reason == "DeletionByPodGC"

    def test_managed_by_label_identifies_the_recorder(self):
        workload, owner = _identities()
        record = PlacementRecord.for_pod(_pod(), workload, owner, "owner-uid")
        labels = record.to_resource()["metadata"]["labels"]
        assert labels["attribution.nvidia.com/managed-by"] == "pod-placement-recorder"
