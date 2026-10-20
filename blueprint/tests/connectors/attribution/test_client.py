# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from copy import deepcopy
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path

import pytest
from nvidia_attribution_client import ApiException, AttributorInfoV2, DuplicateAttributorError, ObservationV2

from attribution_blueprint.connectors.attribution import (
    AttributionConnector,
    AttributorConflictError,
    ObservationConflictError,
)
from attribution_blueprint.orchestrator import AttributionOrchestrator

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


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_conflict_error_pickle_round_trip(protocol):
    error = AttributorConflictError("active-attr-id")
    error.add_note("The existing session belongs to another run.")

    restored = pickle.loads(pickle.dumps(error, protocol=protocol))

    assert type(restored) is AttributorConflictError
    assert restored.attributor_id == error.attributor_id
    assert str(restored) == str(error)
    assert restored.args == error.args
    assert restored.__notes__ == error.__notes__


def test_create_attributor_posts_and_returns_id(mocker):
    request = _transport(mocker, _FakeResponse(HTTPStatus.CREATED, FIXTURES["create_response"]))

    conn = AttributionConnector("http://svc:8000")
    result = conn.create_attributor(_info())

    assert result == FIXTURES["create_response"]["attributor_id"]
    args, kwargs = request.call_args
    assert args == ("POST", "http://svc:8000/v2/attributor")
    assert kwargs["body"]["workload"]["id"] == "ctx-1"
    assert kwargs["headers"]["Content-Type"] == "application/json"


def test_duplicate_blueprint_run_does_not_claim_or_delete_existing_session(mocker):
    """A 409 identifies another caller's session; it does not transfer ownership."""
    warning = mocker.patch("loguru.logger.warning")
    request = _transport(
        mocker,
        _FakeResponse(HTTPStatus.CONFLICT, FIXTURES["conflict_response"]),
        _FakeResponse(HTTPStatus.NO_CONTENT, None),
        _FakeResponse(HTTPStatus.CREATED, FIXTURES["create_response"]),
        _FakeResponse(HTTPStatus.OK, FIXTURES["ingest_response"]),
        _FakeResponse(HTTPStatus.OK, FIXTURES["attribution_response"]),
        _FakeResponse(HTTPStatus.NO_CONTENT, None),
    )
    eca = mocker.Mock()
    eca.discover.return_value = [
        {
            "id": "ctx-1",
            "nodes": ["node-001"],
            "start_time": datetime(2026, 5, 1, 12, tzinfo=timezone.utc),
            "end_time": datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc),
        }
    ]
    collector = mocker.Mock(source="syslog")
    collector.collect.return_value = [json.loads(_observation().to_json())]
    output = mocker.Mock()
    pipeline = AttributionOrchestrator(
        execution_context_adapters=[eca],
        log_connectors=[collector],
        metrics_connectors=[],
        attribution=AttributionConnector("http://svc:8000"),
        outputs=[output],
    )

    assert pipeline.analyze() == []

    assert pipeline.last_run_stats == {"attempted": 1, "succeeded": 0, "conflicted": 1}
    assert pipeline.last_run_summaries == [
        {
            "context_id": "ctx-1",
            "job_id": "ctx-1",
            "attempt_id": "",
            "status": "conflicted",
            "attributor_id": "active-attr-id",
        }
    ]
    output.write.assert_not_called()
    output.close.assert_called_once_with()
    assert request.call_count == 1
    assert request.call_args.args == ("POST", "http://svc:8000/v2/attributor")
    warning.assert_called_once_with(
        "attribution conflict for ctx ctx-1: Attributor active-attr-id already exists; "
        "preserving the existing session. Retry after its owner finishes and deletes it."
    )


def test_ingest_observation_posts_and_returns_response(mocker):
    request = _transport(mocker, _FakeResponse(HTTPStatus.OK, FIXTURES["ingest_response"]))

    conn = AttributionConnector("http://svc:8000")
    result = conn.ingest_observation("attr-7", _observation())

    assert result.observation_id == FIXTURES["ingest_response"]["observation_id"]
    args, kwargs = request.call_args
    assert args == ("POST", "http://svc:8000/v2/attributor/attr-7/observation")
    assert kwargs["body"]["context"]["source"] == "syslog"


def test_ingest_observation_http_error_raises_bounded_message(mocker):
    _transport(mocker, _FakeResponse(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": "x" * 10000}))

    conn = AttributionConnector("http://svc:8000")
    with pytest.raises(RuntimeError) as excinfo:
        conn.ingest_observation("attr-7", _observation())

    assert str(excinfo.value).startswith("HTTP 422")
    assert len(str(excinfo.value)) < 400


def test_ingest_observation_duplicate_conflict_raises_observation_conflict(mocker):
    _transport(mocker, _FakeResponse(HTTPStatus.CONFLICT, {"model_type": "DuplicateObservationError", "obs_id": 42}))

    conn = AttributionConnector("http://svc:8000")
    with pytest.raises(ObservationConflictError) as excinfo:
        conn.ingest_observation("attr-7", _observation())

    assert excinfo.value.observation_id == 42


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


def test_create_attributor_preserves_existing_session_on_conflict(mocker):
    request = _transport(mocker, _FakeResponse(HTTPStatus.CONFLICT, FIXTURES["conflict_response"]))
    warning = mocker.patch("loguru.logger.warning")
    conn = AttributionConnector("http://svc:8000")

    with pytest.raises(AttributorConflictError) as excinfo:
        conn.create_attributor(_info())

    assert excinfo.value.attributor_id == "active-attr-id"
    cause = excinfo.value.__cause__
    assert isinstance(cause, ApiException)
    assert cause.status == HTTPStatus.CONFLICT
    assert isinstance(cause.data, DuplicateAttributorError)
    assert cause.data.att_id == "active-attr-id"
    assert json.loads(cause.body) == FIXTURES["conflict_response"]
    assert cause.headers == {"content-type": "application/json"}
    assert request.call_count == 1
    assert request.call_args.args == ("POST", "http://svc:8000/v2/attributor")
    warning.assert_not_called()


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "boom"}),
        (HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": []}),
        (HTTPStatus.CONFLICT, {"unexpected": "shape"}),
        (HTTPStatus.CONFLICT, {"model_type": "DuplicateAttributorError"}),
        (HTTPStatus.CONFLICT, None),
    ],
    ids=["server-error", "validation-error", "unrecognized-conflict", "missing-id", "empty-conflict"],
)
def test_create_attributor_propagates_other_errors_without_recovery(mocker, status, body):
    request = _transport(mocker, _FakeResponse(status, body))
    warning = mocker.patch("loguru.logger.warning")
    conn = AttributionConnector("http://svc:8000")

    with pytest.raises(ApiException) as excinfo:
        conn.create_attributor(_info())

    assert excinfo.value.status == status
    assert request.call_count == 1
    assert request.call_args.args == ("POST", "http://svc:8000/v2/attributor")
    warning.assert_not_called()


def test_create_attributor_preserves_original_conflict_as_cause(mocker):
    error = ApiException(
        status=HTTPStatus.CONFLICT,
        reason="Conflict",
        data=DuplicateAttributorError(model_type="DuplicateAttributorError", att_id="active-attr-id"),
    )
    conn = AttributionConnector("http://svc:8000")
    create = mocker.patch.object(conn._api, "new_attribution_v2_attributor_post", side_effect=error)

    with pytest.raises(AttributorConflictError) as excinfo:
        conn.create_attributor(_info())

    assert excinfo.value.__cause__ is error
    create.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [
        ApiException(status=409, data={"unexpected": "shape"}),
        ApiException(
            status=500,
            data=DuplicateAttributorError(model_type="DuplicateAttributorError", att_id="active-attr-id"),
        ),
        RuntimeError("transport failure"),
    ],
    ids=["unrecognized-conflict", "non-conflict-status", "non-api-error"],
)
def test_create_attributor_preserves_other_exception_identity(mocker, error):
    conn = AttributionConnector("http://svc:8000")
    create = mocker.patch.object(conn._api, "new_attribution_v2_attributor_post", side_effect=error)

    with pytest.raises(type(error)) as excinfo:
        conn.create_attributor(_info())

    assert excinfo.value is error
    create.assert_called_once()


def test_create_attributor_can_be_retried_after_owner_cleanup(mocker):
    request = _transport(
        mocker,
        _FakeResponse(HTTPStatus.CONFLICT, FIXTURES["conflict_response"]),
        _FakeResponse(HTTPStatus.CREATED, FIXTURES["create_response"]),
    )
    conn = AttributionConnector("http://svc:8000")
    info = _info()
    original_info = info.to_json()

    with pytest.raises(AttributorConflictError):
        conn.create_attributor(info)

    # The owner has since cleaned up; an explicit later attempt may succeed.
    assert conn.create_attributor(info) == FIXTURES["create_response"]["attributor_id"]
    assert info.to_json() == original_info
    assert [call.args for call in request.call_args_list] == [("POST", "http://svc:8000/v2/attributor")] * 2
    assert request.call_args_list[0].kwargs["body"] == request.call_args_list[1].kwargs["body"]


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
