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

from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth
from attribution_blueprint.connectors.metrics.promql.client import (
    PrometheusClient,
    promql_string,
    re2_escape,
)


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
