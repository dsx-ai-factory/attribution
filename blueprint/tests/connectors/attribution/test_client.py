# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from copy import deepcopy
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path

import pytest
from nvidia_attribution_client import ApiException, AttributorInfoV2, ObservationV2

from attribution_blueprint.connectors.attribution import AttributionConnector

FIXTURES = json.loads((Path(__file__).parents[2] / "fixtures" / "service_responses.json").read_text())

_TRANSPORT = "nvidia_attribution_client.rest.RESTClientObject.request"


class _FakeResponse:
    """Stand-in for the client's RESTResponse: `read()` first, then `.data`."""

    def __init__(self, status: int, body: dict | None):
        self.status = status
        self.reason = ""
        self.data = b"" if body is None else json.dumps(body).encode()
        self.headers = {"content-type": "application/json"}

    def read(self):
        return self.data


def _transport(mocker, *responses: _FakeResponse):
    if len(responses) == 1:
        return mocker.patch(_TRANSPORT, return_value=responses[0])
    return mocker.patch(_TRANSPORT, side_effect=list(responses))


def _info(workload_id: str = "ctx-1") -> AttributorInfoV2:
    return AttributorInfoV2.model_validate(
        {
            "workload": {
                "id": workload_id,
                "type": "slurm",
                "status": "FAILED",
                "framework": "",
                "name": "job",
                "username": "alice",
                "job_start_time": "2026-05-01T12:00:00+00:00",
                "job_end_time": "2026-05-01T12:30:00+00:00",
                "timezone": "UTC",
                "nodes": {"node-001": [0, 1, 2, 3]},
            },
            "metadata": {"cluster": "", "agent": "attribution-blueprint", "tenant": "", "ruleset": ""},
        }
    )


def _observation(**context_overrides) -> ObservationV2:
    context = {
        "source": "syslog",
        "time_interval": {"start": "2026-05-01T12:00:00+00:00", "end": "2026-05-01T12:30:00+00:00"},
    }
    context.update(context_overrides)
    return ObservationV2.model_validate({"context": context, "body": "[]"})


def test_create_attributor_posts_and_returns_id(mocker):
    request = _transport(mocker, _FakeResponse(HTTPStatus.CREATED, FIXTURES["create_response"]))

    conn = AttributionConnector("http://svc:8000")
    result = conn.create_attributor(_info())

    assert result == FIXTURES["create_response"]["attributor_id"]
    args, kwargs = request.call_args
    assert args == ("POST", "http://svc:8000/v2/attributor")
    assert kwargs["body"]["workload"]["id"] == "ctx-1"
    assert kwargs["headers"]["Content-Type"] == "application/json"


def test_ingest_observation_posts_and_returns_id(mocker):
    request = _transport(mocker, _FakeResponse(HTTPStatus.OK, FIXTURES["ingest_response"]))

    conn = AttributionConnector("http://svc:8000")
    result = conn.ingest_observation("attr-7", _observation())

    assert result == FIXTURES["ingest_response"]["observation_id"]
    assert isinstance(result, int)
    args, kwargs = request.call_args
    assert args == ("POST", "http://svc:8000/v2/attributor/attr-7/observation")
    assert kwargs["body"]["context"]["source"] == "syslog"


def test_get_attribution_returns_full_dict(mocker):
    request = _transport(mocker, _FakeResponse(HTTPStatus.OK, FIXTURES["attribution_response"]))

    conn = AttributionConnector("http://svc:8000")
    result = conn.get_attribution("attr-7")

    expected = deepcopy(FIXTURES["attribution_response"])
    expected["verdict"]["decision_graph"] = None
    expected["verdict"]["root_symptom"]["catalog_id"] = None
    assert result == expected
    args, _ = request.call_args
    assert args == ("GET", "http://svc:8000/v2/attributor/attr-7/attribution")


def test_delete_attributor_returns_none(mocker):
    request = _transport(mocker, _FakeResponse(HTTPStatus.NO_CONTENT, None))

    conn = AttributionConnector("http://svc:8000")

    assert conn.delete_attributor("attr-7") is None
    args, _ = request.call_args
    assert args == ("DELETE", "http://svc:8000/v2/attributor/attr-7")


def test_create_or_replace_attributor_handles_409_by_deleting_and_retrying(mocker):
    _transport(
        mocker,
        _FakeResponse(HTTPStatus.CONFLICT, FIXTURES["conflict_response"]),
        _FakeResponse(HTTPStatus.NO_CONTENT, None),
        _FakeResponse(HTTPStatus.CREATED, FIXTURES["create_response"]),
    )
    conn = AttributionConnector("http://svc:8000")
    delete = mocker.spy(conn, "delete_attributor")

    assert conn.create_or_replace_attributor(_info()) == FIXTURES["create_response"]["attributor_id"]
    delete.assert_called_once_with("stale-attr-id")


def test_create_or_replace_attributor_propagates_non_409_http_error(mocker):
    _transport(mocker, _FakeResponse(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "boom"}))
    conn = AttributionConnector("http://svc:8000")
    delete = mocker.spy(conn, "delete_attributor")

    with pytest.raises(ApiException) as excinfo:
        conn.create_or_replace_attributor(_info())

    assert excinfo.value.status == HTTPStatus.INTERNAL_SERVER_ERROR
    delete.assert_not_called()


def test_create_or_replace_attributor_propagates_409_without_att_id(mocker):
    _transport(mocker, _FakeResponse(HTTPStatus.CONFLICT, {"unexpected": "shape"}))
    conn = AttributionConnector("http://svc:8000")
    delete = mocker.spy(conn, "delete_attributor")

    with pytest.raises(ApiException) as excinfo:
        conn.create_or_replace_attributor(_info())

    assert excinfo.value.status == HTTPStatus.CONFLICT
    delete.assert_not_called()


def test_create_attributor_propagates_http_error(mocker):
    _transport(mocker, _FakeResponse(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": []}))

    conn = AttributionConnector("http://svc:8000")
    with pytest.raises(ApiException):
        conn.create_attributor(_info())


def test_delete_attributor_tolerates_404(mocker):
    """The server's 24h TTL can expire an attributor before the orchestrator's
    cleanup runs; treating that 404 as success keeps cleanup idempotent."""
    _transport(
        mocker, _FakeResponse(HTTPStatus.NOT_FOUND, {"model_type": "UnknownAttributorIdError", "att_id": "attr-7"})
    )

    conn = AttributionConnector("http://svc:8000")

    # Must not raise.
    assert conn.delete_attributor("attr-7") is None


def test_delete_attributor_propagates_other_errors(mocker):
    """Only 404 is silently tolerated; 500 must still raise."""
    _transport(mocker, _FakeResponse(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "boom"}))

    conn = AttributionConnector("http://svc:8000")
    with pytest.raises(ApiException) as excinfo:
        conn.delete_attributor("attr-7")

    assert excinfo.value.status == HTTPStatus.INTERNAL_SERVER_ERROR
