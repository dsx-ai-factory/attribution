# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_blueprint.kubernetes.daemon.client import KubernetesDaemonClient
from attribution_blueprint.kubernetes.daemon.models import PlacementRecord, ResourceIdentity


class _Auth:
    def request_kwargs(self):
        return {
            "headers": {"Authorization": "Bearer secret"},
            "cert": ("/cert.pem", "/key.pem"),
            "verify": "/ca.pem",
        }


def _record() -> PlacementRecord:
    return PlacementRecord(
        namespace="team-a",
        pod_name="trainer-0",
        pod_uid="pod-uid-1",
        node_name="node-a",
        observed_at="2026-10-13T20:00:00Z",
        associated_resources=(
            ResourceIdentity(
                "scheduling.example.io/v1",
                "Admission",
                "team-a",
                "trainer-admission",
                "admission-uid-1",
            ),
        ),
    )


def test_client_configures_transport_auth():
    client = KubernetesDaemonClient("https://daemon.example/", auth=_Auth(), timeout_seconds=12)

    assert client.endpoint == "https://daemon.example"
    assert client.timeout_seconds == 12
    assert client.session.headers["Authorization"] == "Bearer secret"
    assert client.session.cert == ("/cert.pem", "/key.pem")
    assert client.session.verify == "/ca.pem"


@pytest.mark.parametrize(
    ("endpoint", "timeout", "message"),
    [
        ("", 10, "endpoint"),
        ("http://daemon", 0, "timeout_seconds"),
    ],
)
def test_client_rejects_invalid_configuration(endpoint, timeout, message):
    with pytest.raises(ValueError, match=message):
        KubernetesDaemonClient(endpoint, timeout_seconds=timeout)


def test_list_for_resource_validates_and_returns_records(mocker):
    client = KubernetesDaemonClient("http://daemon")
    response = mocker.Mock()
    response.json.return_value = {"items": [_record().to_resource()]}
    request = mocker.patch.object(client.session, "get", return_value=response)

    assert client.list_for_resource("team-a", "admission-uid-1") == [_record()]
    request.assert_called_once_with(
        "http://daemon/v1/placements",
        params={"namespace": "team-a", "resource_uid": "admission-uid-1"},
        timeout=10,
    )
    response.raise_for_status.assert_called_once_with()


def test_list_for_resource_rejects_invalid_payload(mocker):
    client = KubernetesDaemonClient("http://daemon")
    response = mocker.Mock()
    response.json.return_value = {"records": []}
    mocker.patch.object(client.session, "get", return_value=response)

    with pytest.raises(ValueError, match="items list"):
        client.list_for_resource("team-a", "admission-uid-1")


def test_list_for_resource_rejects_unrelated_record(mocker):
    client = KubernetesDaemonClient("http://daemon")
    response = mocker.Mock()
    response.json.return_value = {"items": [_record().to_resource()]}
    mocker.patch.object(client.session, "get", return_value=response)

    with pytest.raises(ValueError, match="outside the requested resource"):
        client.list_for_resource("team-a", "other-uid")
