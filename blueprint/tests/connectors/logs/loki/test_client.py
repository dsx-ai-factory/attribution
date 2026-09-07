# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime, timedelta, timezone

import pytest
import requests

from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth
from attribution_blueprint.connectors.logs.loki.client import (
    LokiQueryBudgetExceeded,
    LokiQueryClient,
    datetime_to_nanoseconds,
    node_from_labels,
)

EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)


class _Response:
    def __init__(self, streams=None, status_code: int = 200):
        self._streams = streams or []
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(f"{self.status_code}")
            error.response = self
            raise error

    def json(self):
        return {"data": {"result": self._streams}}


class _Session:
    """Serve responses selected by a callback on the posted query parameters."""

    def __init__(self, respond):
        self.respond = respond
        self.calls: list[dict] = []

    def post(self, url, *, data, timeout):
        self.calls.append({"url": url, "data": dict(data), "timeout": timeout})
        return self.respond(data)


def _stream(values, **labels):
    return {"stream": labels or {"host_name": "node-a"}, "values": values}


def _client(session, **kwargs):
    defaults = dict(endpoint="https://loki.example/", auth=TokenAuth("tok"), session=session)
    defaults.update(kwargs)
    return LokiQueryClient(**defaults)


def test_query_range_bounds_requests_to_query_window_and_walks_forward():
    session = _Session(lambda data: _Response([_stream([[str(int(data["start"]) + 1), "line"]])]))
    client = _client(session, query_window=timedelta(minutes=30), max_pages=4)

    entries = client.query_range("query", EPOCH, EPOCH + timedelta(minutes=70))

    starts = [call["data"]["start"] for call in session.calls]
    ends = [call["data"]["end"] for call in session.calls]
    minute = 60 * 1_000_000_000
    assert starts == [0, 30 * minute + 1, 60 * minute + 2]
    assert ends == [30 * minute, 60 * minute + 1, 70 * minute]
    assert all(call["data"]["direction"] == "forward" for call in session.calls)
    assert all(call["url"] == "https://loki.example/loki/api/v1/query_range" for call in session.calls)
    assert [entry["timestamp"] for entry in entries] == [1, 30 * minute + 2, 60 * minute + 3]


def test_query_range_bisects_saturated_pages_without_losing_equal_timestamps():
    # Timestamps in nanoseconds; the request spans four microseconds.
    lines = {1000: ["a", "b"], 2000: ["c"], 4000: ["d", "e"]}

    def respond(data):
        start, end = int(data["start"]), int(data["end"])
        values = [[str(ts), body] for ts in sorted(lines) if start <= ts <= end for body in lines[ts]]
        return _Response([_stream(values[: int(data["limit"])])])

    session = _Session(respond)
    client = _client(session, limit=3, max_pages=5)

    entries = client.query_range("query", EPOCH, EPOCH + timedelta(microseconds=4))

    assert [(entry["timestamp"], entry["body"]) for entry in entries] == [
        (1000, "a"),
        (1000, "b"),
        (2000, "c"),
        (4000, "d"),
        (4000, "e"),
    ]
    assert len(session.calls) == 5


def test_query_range_fails_closed_when_a_saturated_timestamp_cannot_split():
    session = _Session(lambda data: _Response([_stream([["7", "a"], ["7", "b"]])]))
    client = _client(session, limit=2, max_pages=1)

    with pytest.raises(RuntimeError, match="raise limit"):
        client.query_range("query", EPOCH + timedelta(microseconds=7), EPOCH + timedelta(microseconds=7))


def test_query_range_raises_budget_with_partial_entries():
    def respond(data):
        start = int(data["start"])
        return _Response([_stream([[str(start), "x"], [str(start + 1), "y"]])])

    session = _Session(respond)
    client = _client(session, limit=2, max_pages=2)

    with pytest.raises(LokiQueryBudgetExceeded) as excinfo:
        client.query_range("query", EPOCH, EPOCH + timedelta(seconds=1))

    assert [entry["body"] for entry in excinfo.value.entries] == ["x", "y"]
    assert len(session.calls) == 2


def test_http_errors_propagate_from_the_session():
    session = _Session(lambda data: _Response(status_code=400))
    client = _client(session)

    with pytest.raises(requests.exceptions.HTTPError):
        client.query_range("query", EPOCH, EPOCH + timedelta(seconds=1))

    assert len(session.calls) == 1


def test_flatten_resolves_node_from_preferred_label_then_explicit_fallbacks():
    streams = [
        _stream([["1", "preferred"]], host_name="n1", hostname="ignored"),
        _stream([["2", "fallback"]], hostname="n2"),
        _stream([["3", "none", {"node_name": "n3"}]], service="x"),
    ]
    client = _client(_Session(lambda data: _Response(streams)), node_label_fallbacks=("hostname",))

    entries = client.query_range("query", EPOCH, EPOCH + timedelta(seconds=1))

    assert [entry["node"] for entry in entries] == ["n1", "n2", None]
    assert entries[0]["labels"] == {"host_name": "n1", "hostname": "ignored"}
    # Structured metadata travels with the line; it is not a node fallback.
    assert entries[2]["structured_metadata"] == {"node_name": "n3"}
    assert "structured_metadata" not in entries[0]


def test_node_from_labels_prefers_configured_label_and_assumes_no_fallbacks():
    assert node_from_labels({"hostname": "b", "host_name": "a"}, preferred="hostname") == "b"
    assert node_from_labels({"instance": "c"}, preferred="host_name") is None
    assert node_from_labels({"instance": "c"}, preferred="host_name", fallbacks=("instance",)) == "c"
    assert node_from_labels({"host_name": ""}, preferred="host_name") is None


def test_datetime_to_nanoseconds_is_exact():
    at = datetime(2026, 5, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
    assert datetime_to_nanoseconds(at) == int(at.timestamp()) * 1_000_000_000 + 123_456_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"max_pages": 0},
        {"query_window": timedelta(0)},
        {"endpoint": ""},
    ],
)
def test_client_rejects_invalid_settings(kwargs):
    defaults = dict(endpoint="https://loki.example", auth=None)
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        LokiQueryClient(**defaults)


def test_query_range_rejects_naive_or_inverted_bounds():
    client = _client(_Session(lambda data: _Response()))
    with pytest.raises(ValueError):
        client.query_range("q", EPOCH + timedelta(seconds=1), EPOCH)
    with pytest.raises(ValueError):
        client.query_range("q", datetime(2026, 1, 1), datetime(2026, 1, 2))


def test_default_session_carries_auth_and_unbounded_window_is_one_request(mocker):
    client = LokiQueryClient(
        endpoint="https://loki.example",
        auth=MTLSAuth("/c.crt", "/c.key", ca_cert_path="/ca.crt", org_id="tenant"),
        query_window=None,
    )
    post = mocker.patch.object(client._session, "post", return_value=_Response())

    client.query_range("query", EPOCH, EPOCH + timedelta(days=3))

    assert client._session.cert == ("/c.crt", "/c.key")
    assert client._session.verify == "/ca.crt"
    assert client._session.headers["X-Scope-OrgID"] == "tenant"
    assert [(call.kwargs["data"]["start"], call.kwargs["data"]["end"]) for call in post.call_args_list] == [
        (0, 3 * 86_400 * 1_000_000_000)
    ]
