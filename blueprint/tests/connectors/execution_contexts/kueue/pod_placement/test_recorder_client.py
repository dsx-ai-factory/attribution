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
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.recorder_client import (
    RecorderPlacementClient,
)


def _record(namespace="ns", workload_uid="wl-uid-1"):
    return PlacementRecord(
        namespace=namespace,
        workload=WorkloadIdentity("wl-a", workload_uid),
        owner=OwnerIdentity("batch/v1", "Job", "job-a", "job-uid-1"),
        pod_name="pod-a",
        pod_uid="pod-uid-1",
        node_name="node-a",
        observed_at="2026-09-14T00:00:00Z",
        resolver_source="owner-uid",
    )


def test_client_passes_uid_selectors_and_decodes_records(mocker):
    response = mocker.Mock()
    response.json.return_value = {"items": [_record().to_resource()]}
    client = RecorderPlacementClient("http://recorder:8080/")
    client.session.get = mocker.Mock(return_value=response)

    assert client.list_for_workload("ns", "wl-uid-1") == [_record()]
    client.session.get.assert_called_once_with(
        "http://recorder:8080/v1/placements",
        params={"namespace": "ns", "workload_uid": "wl-uid-1"},
        timeout=10.0,
    )
    response.raise_for_status.assert_called_once()


def test_client_rejects_out_of_scope_response(mocker):
    response = mocker.Mock()
    response.json.return_value = {"items": [_record(namespace="other").to_resource()]}
    client = RecorderPlacementClient("http://recorder:8080")
    client.session.get = mocker.Mock(return_value=response)

    with pytest.raises(ValueError, match="outside the requested workload"):
        client.list_for_workload("ns", "wl-uid-1")
