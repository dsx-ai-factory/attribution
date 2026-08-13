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
from pathlib import Path

import pytest
import requests

from attribution_blueprint.connectors.attribution import AttributionConnector

FIXTURES = json.loads((Path(__file__).parents[2] / "fixtures" / "service_responses.json").read_text())


def _mock_response(mocker, body: dict | None, status_ok: bool = True):
    fake = mocker.Mock()
    fake.json.return_value = body
    if status_ok:
        fake.raise_for_status = mocker.Mock()
    else:
        fake.raise_for_status.side_effect = requests.HTTPError("boom")
    return fake


def _http_error(mocker, status_code: int, body: dict | None = None) -> requests.HTTPError:
    response = mocker.Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = body or {}
    return requests.HTTPError(response=response)


def test_create_attributor_posts_and_returns_id(mocker):
    fake = _mock_response(mocker, FIXTURES["create_response"])
    post = mocker.patch("requests.Session.post", return_value=fake)

    conn = AttributionConnector("http://svc:8000")
    info = {"workload": {"id": "ctx-1"}, "metadata": {"agent": "blueprint"}}
    result = conn.create_attributor(info)

    assert result == FIXTURES["create_response"]["attributor_id"]
    args, kwargs = post.call_args
    assert args == ("http://svc:8000/v2/attributor",)
    assert json.loads(kwargs["data"]) == info
    assert kwargs["headers"] == {"Content-Type": "application/json"}
    assert kwargs["timeout"] == 60.0
    assert "Authorization" not in conn._session.headers


def test_ingest_observation_posts_and_returns_id(mocker):
    fake = _mock_response(mocker, FIXTURES["ingest_response"])
    post = mocker.patch("requests.Session.post", return_value=fake)

    conn = AttributionConnector("http://svc:8000")
    obs = {"context": {"source": "SYSLOG"}, "body": "line1\nline2"}
    result = conn.ingest_observation("attr-7", obs)

    assert result == FIXTURES["ingest_response"]["observation_id"]
    assert isinstance(result, int)
    args, kwargs = post.call_args
    assert args == ("http://svc:8000/v2/attributor/attr-7/observation",)
    assert json.loads(kwargs["data"]) == obs
    assert kwargs["timeout"] == 60.0


def test_get_attribution_returns_full_dict(mocker):
    fake = _mock_response(mocker, FIXTURES["attribution_response"])
    get = mocker.patch("requests.Session.get", return_value=fake)

    conn = AttributionConnector("http://svc:8000")
    result = conn.get_attribution("attr-7")

    assert result == FIXTURES["attribution_response"]
    get.assert_called_once_with("http://svc:8000/v2/attributor/attr-7/attribution", timeout=60.0)


def test_delete_attributor_returns_none(mocker):
    fake = _mock_response(mocker, None)
    delete = mocker.patch("requests.Session.delete", return_value=fake)

    conn = AttributionConnector("http://svc:8000")
    result = conn.delete_attributor("attr-7")

    assert result is None
    delete.assert_called_once_with("http://svc:8000/v2/attributor/attr-7", timeout=60.0)


def test_create_or_replace_attributor_handles_409_by_deleting_and_retrying(mocker):
    conn = AttributionConnector("http://svc:8000")
    mocker.patch.object(
        conn,
        "create_attributor",
        side_effect=[_http_error(mocker, 409, {"att_id": "stale-attr-id"}), "fresh-attr-id"],
    )
    delete = mocker.patch.object(conn, "delete_attributor")

    assert conn.create_or_replace_attributor({"workload": {"id": "1"}}) == "fresh-attr-id"
    delete.assert_called_once_with("stale-attr-id")
    assert conn.create_attributor.call_count == 2


def test_create_or_replace_attributor_propagates_non_409_http_error(mocker):
    conn = AttributionConnector("http://svc:8000")
    mocker.patch.object(conn, "create_attributor", side_effect=_http_error(mocker, 500, {"error": "boom"}))
    delete = mocker.patch.object(conn, "delete_attributor")

    with pytest.raises(requests.HTTPError):
        conn.create_or_replace_attributor({"workload": {"id": "1"}})
    delete.assert_not_called()


def test_create_or_replace_attributor_propagates_409_without_att_id(mocker):
    conn = AttributionConnector("http://svc:8000")
    mocker.patch.object(conn, "create_attributor", side_effect=_http_error(mocker, 409, {}))
    delete = mocker.patch.object(conn, "delete_attributor")

    with pytest.raises(requests.HTTPError):
        conn.create_or_replace_attributor({"workload": {"id": "1"}})
    delete.assert_not_called()


def test_raise_for_status_propagates(mocker):
    fake = _mock_response(mocker, None, status_ok=False)
    mocker.patch("requests.Session.post", return_value=fake)

    conn = AttributionConnector("http://svc:8000")
    with pytest.raises(requests.HTTPError):
        conn.create_attributor({})


def test_trailing_slash_normalized(mocker):
    fake = _mock_response(mocker, FIXTURES["create_response"])
    post = mocker.patch("requests.Session.post", return_value=fake)

    conn = AttributionConnector("http://svc:8000/")
    conn.create_attributor({})

    assert post.call_args.args == ("http://svc:8000/v2/attributor",)


def test_delete_attributor_tolerates_404(mocker):
    """The server's 24h TTL can expire an attributor before the orchestrator's
    cleanup runs; treating that 404 as success keeps cleanup idempotent."""
    fake = mocker.Mock()
    fake.status_code = 404
    fake.raise_for_status = mocker.Mock()
    mocker.patch("requests.Session.delete", return_value=fake)

    conn = AttributionConnector("http://svc:8000")
    # Must not raise.
    assert conn.delete_attributor("attr-7") is None
    # We short-circuited before raise_for_status, so it must NOT have been called.
    fake.raise_for_status.assert_not_called()


def test_delete_attributor_propagates_other_errors(mocker):
    """Only 404 is silently tolerated; 500 must still raise."""
    fake = mocker.Mock()
    fake.status_code = 500
    fake.raise_for_status.side_effect = requests.HTTPError("boom")
    mocker.patch("requests.Session.delete", return_value=fake)

    conn = AttributionConnector("http://svc:8000")
    with pytest.raises(requests.HTTPError):
        conn.delete_attributor("attr-7")


def test_datetime_in_observation_serializes(mocker):
    from datetime import datetime, timezone

    fake = _mock_response(mocker, FIXTURES["ingest_response"])
    post = mocker.patch("requests.Session.post", return_value=fake)

    conn = AttributionConnector("http://svc:8000")
    obs = {
        "context": {
            "time_interval": {
                "start": datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 5, 1, 12, 30, 0, tzinfo=timezone.utc),
            },
            "source": "SYSLOG",
        },
        "body": "[]",
    }
    conn.ingest_observation("attr-7", obs)
    sent = json.loads(post.call_args.kwargs["data"])
    assert sent["context"]["time_interval"]["start"] == "2026-05-01T12:00:00+00:00"
