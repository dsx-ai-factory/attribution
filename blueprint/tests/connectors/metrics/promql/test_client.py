# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone

import pytest
import requests

from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth
from attribution_blueprint.connectors.metrics.promql.client import (
    PrometheusClient,
    PromQLQueryClient,
    nodes_regex,
    promql_string,
    re2_escape,
)

START = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
END = START + timedelta(minutes=30)


class _Response:
    def __init__(self, payload=None, status_code: int = 200):
        self._payload = payload if payload is not None else {"status": "success", "data": {"result": []}}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(str(self.status_code))
            error.response = self
            raise error

    def json(self):
        return self._payload


class _Session:
    def __init__(self, respond):
        self.respond = respond
        self.calls: list[dict] = []

    def post(self, url, *, data, timeout):
        self.calls.append({"url": url, "data": dict(data), "timeout": timeout})
        return self.respond(data)


def _client(session, **kwargs):
    defaults = dict(endpoint="https://prom.example/", auth=TokenAuth("tok"), session=session)
    defaults.update(kwargs)
    return PromQLQueryClient(**defaults)


def _series(values, **labels):
    return {"metric": labels, "values": values}


def test_query_range_posts_seconds_and_flattens_samples_with_node_identity():
    payload = {
        "status": "success",
        "data": {
            "result": [
                _series([[1700000000, "1"], ["1700000060.5", "2"]], net_host_name="n1", s_cluster="c1", device="ib0"),
                _series([[1700000000, "9"]], job="no-node"),
            ]
        },
    }
    session = _Session(lambda data: _Response(payload))
    client = _client(session)

    entries = client.query_range("up", 'up{net_host_name=~"n1"}', START, END, step_seconds=15, cluster="fallback")

    call = session.calls[0]
    assert call["url"] == "https://prom.example/api/v1/query_range"
    assert call["data"] == {
        "query": 'up{net_host_name=~"n1"}',
        "start": int(START.timestamp()),
        "end": int(END.timestamp()),
        "step": 15,
    }
    assert entries == [
        {
            "timestamp": 1700000000,
            "value": "1",
            "metric": "up",
            "labels": {"net_host_name": "n1", "s_cluster": "c1", "device": "ib0"},
            "node": "n1",
            "cluster": "c1",
            "component": "ib0",
        },
        {
            "timestamp": 1700000060,
            "value": "2",
            "metric": "up",
            "labels": {"net_host_name": "n1", "s_cluster": "c1", "device": "ib0"},
            "node": "n1",
            "cluster": "c1",
            "component": "ib0",
        },
    ]


def test_query_range_falls_back_to_caller_cluster_when_label_dropped():
    payload = {"status": "success", "data": {"result": [_series([[1, "1"]], net_host_name="n1")]}}
    client = _client(_Session(lambda data: _Response(payload)))

    entries = client.query_range("m", "m", START, END, cluster="c-fallback")

    assert entries[0]["cluster"] == "c-fallback"
    assert "component" not in entries[0]


def test_query_instant_posts_time_and_reads_single_value():
    payload = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [{"metric": {"net_host_name": "n1"}, "value": [1700000000.5, "42"]}],
        },
    }
    session = _Session(lambda data: _Response(payload))
    client = _client(session)

    entries = client.query_instant("serial", "DCGM_FI_DEV_SERIAL", END)

    assert session.calls[0]["url"] == "https://prom.example/api/v1/query"
    assert session.calls[0]["data"] == {"query": "DCGM_FI_DEV_SERIAL", "time": int(END.timestamp())}
    assert [(entry["timestamp"], entry["value"], entry["node"]) for entry in entries] == [(1700000000, "42", "n1")]


def test_status_error_payload_returns_empty(caplog):
    payload = {"status": "error", "error": "query timeout"}
    client = _client(_Session(lambda data: _Response(payload)))

    assert client.query_range("m", "m", START, END) == []


def test_http_errors_propagate_from_the_session():
    session = _Session(lambda data: _Response(status_code=400))
    client = _client(session)

    with pytest.raises(requests.exceptions.HTTPError):
        client.query_range("m", "m", START, END)

    assert len(session.calls) == 1


def test_custom_node_label_selects_node_identity():
    payload = {"status": "success", "data": {"result": [_series([[1, "1"]], instance="host-a:9100")]}}
    client = _client(_Session(lambda data: _Response(payload)), node_label="instance")

    assert client.query_range("m", "m", START, END)[0]["node"] == "host-a:9100"


def test_nodes_regex_escapes_only_re2_metacharacters():
    assert nodes_regex(["example-node-014", "host.example.com", ""]) == "example-node-014|host\\.example\\.com"


@pytest.mark.parametrize("kwargs", [{"endpoint": ""}, {"node_label": ""}])
def test_client_rejects_invalid_settings(kwargs):
    defaults = dict(endpoint="https://prom.example", auth=None)
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        PromQLQueryClient(**defaults)


def test_query_range_rejects_bad_bounds():
    client = _client(_Session(lambda data: _Response()))
    with pytest.raises(ValueError):
        client.query_range("m", "m", END, START)
    with pytest.raises(ValueError):
        client.query_range("m", "m", START, END, step_seconds=0)
    with pytest.raises(ValueError):
        client.query_range("m", "m", datetime(2026, 1, 1), datetime(2026, 1, 2))


def test_require_node_false_keeps_series_without_node_label():
    payload = {"status": "success", "data": {"result": [_series([[1, "1"]], CapacityBlockId="cb-1")]}}
    client = _client(_Session(lambda data: _Response(payload)), require_node=False)

    entries = client.query_range("health", "health", START, END)

    assert entries == [
        {"timestamp": 1, "value": "1", "metric": "health", "labels": {"CapacityBlockId": "cb-1"}, "node": ""}
    ]


def test_default_session_carries_auth(mocker):
    client = PromQLQueryClient(endpoint="https://prom.example", auth=TokenAuth("tok", org_id="t"))
    mocker.patch.object(client._session, "post", return_value=_Response())

    client.query_instant("m", "m", END)

    assert client._session.headers["Authorization"] == "Bearer tok"
    assert client._session.headers["X-Scope-OrgID"] == "t"


def _mock_response(mocker, result, status="success", error=None):
    fake = mocker.Mock()
    payload = {"status": status, "data": {"result": result}}
    if error is not None:
        payload["error"] = error
    fake.json.return_value = payload
    fake.raise_for_status = mocker.Mock()
    return mocker.patch("requests.Session.post", return_value=fake)


def test_query_posts_to_query_endpoint(mocker):
    post = _mock_response(mocker, [{"metric": {"a": "b"}}])
    client = PrometheusClient(endpoint="https://prom.example/")
    result = client.query(name="q", query="up", timestamp=1000)

    assert result == [{"metric": {"a": "b"}}]
    args, kwargs = post.call_args
    assert args[0] == "https://prom.example/api/v1/query"
    assert kwargs["data"] == {"query": "up", "time": 1000}


def test_query_range_posts_to_query_range_endpoint(mocker):
    post = _mock_response(mocker, [])
    client = PrometheusClient(endpoint="https://prom.example")
    client.query_range(name="q", query="up", start=0, end=60, step=15)

    args, kwargs = post.call_args
    assert args[0] == "https://prom.example/api/v1/query_range"
    assert kwargs["data"] == {"query": "up", "start": 0, "end": 60, "step": 15}


def test_status_error_returns_empty_and_warns(mocker):
    _mock_response(mocker, [], status="error", error="query timeout")
    client = PrometheusClient(endpoint="https://prom.example")
    assert client.query(name="q", query="up", timestamp=0) == []


def test_non_list_result_returns_empty(mocker):
    fake = mocker.Mock()
    fake.json.return_value = {"status": "success", "data": {"result": "not-a-list"}}
    fake.raise_for_status = mocker.Mock()
    mocker.patch("requests.Session.post", return_value=fake)
    client = PrometheusClient(endpoint="https://prom.example")
    assert client.query(name="q", query="up", timestamp=0) == []


def test_token_auth_sets_authorization_header():
    client = PrometheusClient(endpoint="https://prom.example", auth=TokenAuth("tok", org_id="tenant"))
    assert client.session.headers["Authorization"] == "Bearer tok"
    assert client.session.headers["X-Scope-OrgID"] == "tenant"


def test_mtls_auth_sets_session_cert():
    client = PrometheusClient(endpoint="https://prom.example", auth=MTLSAuth(cert_path="/c", key_path="/k"))
    assert client.session.cert == ("/c", "/k")


def test_auth_is_optional():
    client = PrometheusClient(endpoint="https://prom.example")
    assert "Authorization" not in client.session.headers


@pytest.mark.parametrize("prefix", ["http://", "https://"])
def test_transient_failures_use_bounded_retries(prefix):
    client = PrometheusClient(endpoint="https://prom.example")
    retry = client.session.adapters[prefix].max_retries

    assert retry.total == 5
    assert retry.backoff_factor == 0.5
    assert retry.status_forcelist == (429, 502, 503, 504)
    assert retry.allowed_methods == frozenset({"GET", "POST"})


def test_re2_escape_does_not_backslash_hyphens():
    assert re2_escape("host-01") == "host-01"
    assert re2_escape("a.b") == "a\\.b"


def test_promql_string_escapes_quotes_and_backslashes():
    assert promql_string('va"lue') == '"va\\"lue"'
    assert promql_string("a\\b") == '"a\\\\b"'


@pytest.mark.parametrize("input_value", ["", "plain"])
def test_promql_string_roundtrips_simple_values(input_value):
    assert promql_string(input_value) == f'"{input_value}"'
