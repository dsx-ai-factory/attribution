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
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import pytest
from nvidia_attribution_client import ApiException, AttributorInfoV2, Observation
from nvidia_attribution_client.rest import RESTResponse
from urllib3 import HTTPResponse
from urllib3.exceptions import ReadTimeoutError

from attribution_blueprint.connectors.attribution import AttributionConnector

FIXTURES = json.loads((Path(__file__).parents[2] / "fixtures" / "service_responses.json").read_text())

_TRANSPORT = "nvidia_attribution_client.rest.RESTClientObject.request"


class _FakeResponse(RESTResponse):
    """Use the SDK response wrapper with an in-memory HTTP body."""

    def __init__(self, status: int, body: dict | None):
        super().__init__(
            HTTPResponse(
                body=BytesIO(b"" if body is None else json.dumps(body).encode()),
                status=status,
                headers={"content-type": "application/json"},
                preload_content=False,
            )
        )


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


def _observation(**context_overrides) -> Observation:
    context = {
        "source": "syslog",
        "format": "loki_dict_list",
        "resources": "AllJobResources",
        "time_interval": {"start": "2026-05-01T12:00:00+00:00", "end": "2026-05-01T12:30:00+00:00"},
    }
    context.update(context_overrides)
    return Observation.model_validate({"context": context, "body": "[]"})


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


@pytest.mark.parametrize(
    ("verdict_fields", "site_status"),
    [
        ({}, "bad"),
        ({"decision_graph": None}, "bad"),
        ({"decision_graph": {"nodes": [], "edges": []}}, "bad"),
        ({"decision_graph": "invalid"}, "bad"),
        ({}, "invalid"),
    ],
    ids=["absent-graph", "null-graph", "empty-graph", "invalid-graph", "invalid-site"],
)
def test_get_attribution_preserves_response_json(mocker, verdict_fields, site_status):
    expected = deepcopy(FIXTURES["attribution_response"])
    expected["verdict"].update(verdict_fields)
    expected["extra_evidence"] = {"count": 0, "available": False}
    expected["symptoms"]["GpuFell"][0]["source"]["pos"] = 0
    expected["symptoms"]["GpuFell"][0]["status"] = site_status
    response = _FakeResponse(HTTPStatus.OK, expected)
    close = mocker.spy(response.response, "close")
    release = mocker.spy(response.response, "release_conn")
    request = _transport(mocker, response)

    conn = AttributionConnector("http://svc:8000", timeout_seconds=12.0)
    result = conn.get_attribution("attr-7")

    assert result == expected
    args, kwargs = request.call_args
    assert args == ("GET", "http://svc:8000/v2/attributor/attr-7/attribution")
    assert kwargs["_request_timeout"] == (3.05, 12.0)
    close.assert_called_once_with()
    release.assert_called()


@pytest.mark.parametrize(
    ("status", "body", "error_model"),
    [
        (404, {"model_type": "UnknownAttributorIdError", "att_id": "attr-7"}, "UnknownAttributorIdError"),
        (507, {"model_type": "EventStorageOverflowError", "att_id": "attr-7"}, "EventStorageOverflowError"),
        (422, {"errors": [], "body": None, "message": "Invalid request"}, "ValidationErrorResponse"),
        (500, {"detail": "boom"}, None),
    ],
)
def test_get_attribution_preserves_http_errors_and_releases_response(mocker, status, body, error_model):
    response = _FakeResponse(status, body)
    close = mocker.spy(response.response, "close")
    release = mocker.spy(response.response, "release_conn")
    _transport(mocker, response)

    with pytest.raises(ApiException) as excinfo:
        AttributionConnector("http://svc:8000").get_attribution("attr-7")

    assert excinfo.value.status == status
    assert json.loads(excinfo.value.body) == body
    if error_model is None:
        assert excinfo.value.data is None
    else:
        assert type(excinfo.value.data).__name__ == error_model
    close.assert_called_once_with()
    release.assert_called()


@pytest.mark.parametrize("body", [b"{", b"[]"])
def test_get_attribution_rejects_invalid_json_and_releases_response(mocker, body):
    response = _FakeResponse(HTTPStatus.OK, None)
    response.response = HTTPResponse(
        body=BytesIO(body), status=HTTPStatus.OK, headers={"content-type": "application/json"}, preload_content=False
    )
    close = mocker.spy(response.response, "close")
    release = mocker.spy(response.response, "release_conn")
    _transport(mocker, response)

    with pytest.raises(ValueError):
        AttributionConnector("http://svc:8000").get_attribution("attr-7")

    close.assert_called_once_with()
    release.assert_called()


def test_get_attribution_propagates_read_failure_and_releases_response(mocker):
    response = _FakeResponse(HTTPStatus.OK, FIXTURES["attribution_response"])
    error = ReadTimeoutError(None, "/v2/attributor/attr-7/attribution", "Read timed out")
    mocker.patch.object(response.response, "read", side_effect=error)
    close = mocker.spy(response.response, "close")
    release = mocker.spy(response.response, "release_conn")
    _transport(mocker, response)

    with pytest.raises(ReadTimeoutError) as excinfo:
        AttributionConnector("http://svc:8000").get_attribution("attr-7")

    assert excinfo.value is error
    close.assert_called_once_with()
    release.assert_called_once_with()


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
