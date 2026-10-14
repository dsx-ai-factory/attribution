# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import requests

from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.daemon_client import (
    KueueDaemonClient,
)
from attribution_blueprint.kubernetes.daemon.models import PlacementRecord


def test_uses_scheduler_neutral_daemon_query(mocker):
    client = KueueDaemonClient("http://daemon")
    expected = [
        PlacementRecord(
            namespace="ns",
            pod_name="pod-a",
            pod_uid="pod-uid-1",
            node_name="node-a",
            observed_at="2026-10-13T20:00:00Z",
        )
    ]
    mocker.patch.object(client.daemon, "list_for_resource", return_value=expected)
    legacy = mocker.patch.object(client.daemon.session, "get")

    assert client.list_for_resource("ns", "workload-uid-1") == expected
    legacy.assert_not_called()


def test_falls_back_to_legacy_workload_query_on_bad_request(mocker):
    client = KueueDaemonClient("http://recorder")
    response = requests.Response()
    response.status_code = 400
    error = requests.HTTPError(response=response)
    mocker.patch.object(client.daemon, "list_for_resource", side_effect=error)
    legacy_response = mocker.Mock()
    legacy_response.json.return_value = {
        "items": [
            {
                "metadata": {"namespace": "ns"},
                "spec": {
                    "workload": {"name": "wl-a", "uid": "workload-uid-1"},
                    "owner": {
                        "apiVersion": "batch/v1",
                        "kind": "Job",
                        "name": "job-a",
                        "uid": "job-uid-1",
                    },
                    "pod": {"name": "pod-a", "uid": "pod-uid-1"},
                    "node": {"name": "node-a"},
                    "observedAt": "2026-10-13T20:00:00Z",
                },
            }
        ]
    }
    legacy_response.raise_for_status.return_value = None
    mocker.patch.object(client.daemon.session, "get", return_value=legacy_response)

    [record] = client.list_for_resource("ns", "workload-uid-1")

    assert record.pod_uid == "pod-uid-1"
    assert record.owner is not None
    assert record.owner.uid == "job-uid-1"
    assert record.associated_resources[0].uid == "workload-uid-1"
