# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check connector measurements without contacting a backend."""

import io
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from attribution_blueprint import telemetry


@pytest.fixture(scope="module")
def reader():
    """Set the process provider once.

    Returns:
        The reader used by this module.
    """
    reader = InMemoryMetricReader()
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
    return reader


def points(reader, connector, name):
    """Select measurements for one test connector.

    Args:
        reader: Metric reader for this process.
        connector: Connector name to select.
        name: Metric suffix to select.

    Returns:
        Matching data points.
    """
    data = reader.get_metrics_data()
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == f"attribution.connector.{name}"
        for point in metric.data.data_points
        if point.attributes.get("connector") == connector
    ]


def test_track_success(reader):
    """Record counts, bytes, duration, and fixed labels.

    Args:
        reader: Metric reader for this process.
    """
    with telemetry.track("track_success", "query") as call:
        call.response_bytes = 17

    labels = {"connector": "track_success", "operation": "query"}
    for name, expected in (("requests", 1), ("successes", 1), ("response.size", 17)):
        (point,) = points(reader, "track_success", name)
        assert point.value == expected
        assert point.attributes == labels
    assert points(reader, "track_success", "errors") == []
    (duration,) = points(reader, "track_success", "request.duration")
    assert duration.count == 1
    assert duration.sum >= 0
    assert duration.attributes == labels
    assert duration.explicit_bounds == (
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1,
        2.5,
        5,
        7.5,
        10,
        30,
        60,
        120,
    )


class StatusError(Exception):
    """Represent an API error with an HTTP status."""

    status = 503


@pytest.mark.parametrize("error, error_type", [(StatusError(), "503"), (ValueError(), "ValueError")])
def test_track_exception(reader, error, error_type):
    """Preserve exceptions and label both errors and duration.

    Args:
        reader: Metric reader for this process.
        error: Exception raised by the call.
        error_type: Expected error label.
    """
    connector = f"exception_{error_type}"
    with pytest.raises(type(error)) as raised:
        with telemetry.track(connector, "read"):
            raise error
    assert raised.value is error
    assert points(reader, connector, "requests")[0].value == 1
    assert points(reader, connector, "successes") == []
    assert points(reader, connector, "response.size") == []
    (failures,) = points(reader, connector, "errors")
    assert failures.value == 1
    labels = {"connector": connector, "operation": "read", "error.type": error_type}
    assert failures.attributes == labels
    (duration,) = points(reader, connector, "request.duration")
    assert duration.count == 1
    assert duration.attributes == labels


@pytest.mark.parametrize(
    "body, response_bytes, expected_size", [("héllo", None, 6), (None, None, None), (b"ignored", 3, 3)]
)
def test_track_exception_response_size(reader, body, response_bytes, expected_size):
    """Measure error bodies without replacing a recorded size.

    Args:
        reader: Metric reader for this process.
        body: Error body supplied by the client.
        response_bytes: Size already recorded by the connector.
        expected_size: Expected byte count, or None when no size is recorded.
    """
    connector = f"error_body_{expected_size}"
    error = StatusError()
    error.body = body
    with pytest.raises(StatusError) as raised:
        with telemetry.track(connector, "read") as call:
            call.response_bytes = response_bytes
            raise error
    assert raised.value is error
    sizes = points(reader, connector, "response.size")
    if expected_size is None:
        assert sizes == []
    else:
        (size,) = sizes
        assert size.value == expected_size


class FakeAdapter(requests.adapters.BaseAdapter):
    """Return a response or raise without opening a connection."""

    def __init__(self, status=200, body=b"hello", error=None):
        """Store the transport result.

        Args:
            status: HTTP response status.
            body: Decoded response body.
            error: Optional transport exception.
        """
        self.error = error
        self.response = requests.Response()
        self.response.status_code = status
        self.response.raw = io.BytesIO(body)

    def send(self, request, **kwargs):
        """Return the configured result.

        Args:
            request: Prepared HTTP request.
            **kwargs: Transport options.

        Returns:
            The configured response.
        """
        if self.error is not None:
            raise self.error
        self.response.request = request
        self.response.url = request.url
        return self.response

    def close(self):
        """No transport resources need cleanup."""


@pytest.mark.parametrize("backend", ["loki", "promql"])
@pytest.mark.parametrize("entry_point", ["client", "collect", "collect_batch"])
def test_query_clients_meter_requests(reader, backend, entry_point):
    """Measure direct and connector queries without losing auth or retries.

    Args:
        reader: Metric reader for this process.
        backend: Query backend to exercise.
        entry_point: Client or connector method to call.
    """
    from attribution_blueprint.connectors.auth import MTLSAuth
    from attribution_blueprint.connectors.logs.loki.client import LokiQueryClient
    from attribution_blueprint.connectors.logs.loki.connector import LokiConnector
    from attribution_blueprint.connectors.metrics.promql.client import PromQLQueryClient
    from attribution_blueprint.connectors.metrics.promql.connector import PromQLMetricsConnector
    from attribution_blueprint.types import Source

    auth = MTLSAuth("/c.crt", "/c.key", ca_cert_path="/ca.crt", org_id="tenant")
    kwargs = {"endpoint": "http://h/prefix", "auth": auth}
    if backend == "loki":
        client_class = LokiQueryClient
        connector = LokiConnector(
            **kwargs, source=Source.SYSLOG, query_kwargs={}, templates={Source.SYSLOG: '{{app="test"}}'}
        )
        operations = ["/loki/api/v1/query_range"]
    else:
        client_class = PromQLQueryClient
        connector = PromQLMetricsConnector(
            **kwargs, cluster="test", templates={Source.NODE_METRICS: {"metric": "{name}"}}
        )
        operations = ["/api/v1/query_range"]
        if entry_point == "client":
            operations.append("/api/v1/query")
    client = client_class(**kwargs) if entry_point == "client" else connector._client
    session = client._session
    assert isinstance(session, telemetry.MeteredSession)
    assert session.cert == ("/c.crt", "/c.key")
    assert session.verify == "/ca.crt"
    assert session.headers["X-Scope-OrgID"] == "tenant"
    retry = session.get_adapter("http://h").max_retries
    assert retry.total == 2
    assert retry.read == 2
    assert retry.backoff_jitter == 1.0
    assert retry.allowed_methods == frozenset({"POST"})
    assert retry.raise_on_status is False

    body = b'{"status":"success","data":{"result":[]}}'
    session.mount("http://", FakeAdapter(body=body))
    before = {
        name: {point.attributes["operation"]: point.value for point in points(reader, backend, name)}
        for name in ("requests", "successes", "response.size")
    }
    at = datetime(2026, 5, 1, tzinfo=timezone.utc)
    context = {"id": "test", "nodes": ["n1"], "start_time": at, "end_time": at}
    if entry_point == "client":
        if backend == "loki":
            assert client.query_range('{app="test"}', at, at) == []
        else:
            assert client.query_range("metric", "metric", at, at) == []
            assert client.query_instant("metric", "metric", at) == []
    elif entry_point == "collect":
        assert connector.collect(context) == []
    else:
        assert connector.collect_batch([context]) == {"test": []}
    for name, expected in (("requests", 1), ("successes", 1), ("response.size", len(body))):
        after = {point.attributes["operation"]: point.value for point in points(reader, backend, name)}
        for operation in operations:
            assert after[operation] - before[name].get(operation, 0) == expected
    with requests.Session() as supplied:
        assert client_class(**kwargs, session=supplied)._session is supplied


@pytest.mark.parametrize("status", [200, 404])
def test_session_response(reader, status):
    """Count HTTP errors without raising for status.

    Args:
        reader: Metric reader for this process.
        status: Response status to measure.
    """
    connector = f"session_{status}"
    adapter = FakeAdapter(status=status)
    with telemetry.MeteredSession(connector, ("/api/v1/query",)) as session:
        session.mount("http://", adapter)
        response = session.get("http://h/api/v1/query")
    assert response is adapter.response
    assert response.content == b"hello"
    labels = {"connector": connector, "operation": "/api/v1/query"}
    assert points(reader, connector, "requests")[0].value == 1
    (size,) = points(reader, connector, "response.size")
    assert size.value == 5
    assert size.attributes == labels
    metric = "errors" if status >= 400 else "successes"
    (result,) = points(reader, connector, metric)
    assert result.value == 1
    if status >= 400:
        labels["error.type"] = str(status)
        assert points(reader, connector, "successes") == []
    else:
        assert points(reader, connector, "errors") == []
    assert result.attributes == labels
    assert points(reader, connector, "request.duration")[0].attributes == labels


def test_session_connection_error(reader):
    """Record transport failures and preserve the exception.

    Args:
        reader: Metric reader for this process.
    """
    error = requests.ConnectionError("connection failed")
    with telemetry.MeteredSession("connection_error") as session:
        session.mount("http://", FakeAdapter(error=error))
        with pytest.raises(requests.ConnectionError) as raised:
            session.get("http://h/")
    assert raised.value is error
    (failure,) = points(reader, "connection_error", "errors")
    assert failure.value == 1
    assert failure.attributes["error.type"] == "ConnectionError"


@pytest.mark.parametrize(
    "method,path,operation",
    [
        ("get", "/prefix/api/v1/query", "/api/v1/query"),
        ("get", "/v1/jobs/12345", "GET"),
        ("post", "/api/v1/query_range", "POST"),
    ],
)
def test_session_path_guard(reader, method, path, operation):
    """Allow base prefixes without recording arbitrary paths.

    Args:
        reader: Metric reader for this process.
        method: HTTP method.
        path: Requested URL path.
        operation: Expected operation label.
    """
    connector = f"path_{operation}"
    with telemetry.MeteredSession(connector, ("/api/v1/query",)) as session:
        session.mount("http://", FakeAdapter())
        session.request(method, f"http://h{path}")
    assert points(reader, connector, "requests")[0].attributes["operation"] == operation


@pytest.mark.parametrize(
    "case, session_stream, args, kwargs, expected_stream",
    [
        ("keyword", False, (), {"stream": True}, True),
        ("session", True, (), {}, True),
        ("positional", False, (None, None, None, None, None, None, None, True, None, None, True), {}, True),
        ("override", True, (), {"stream": False}, False),
    ],
)
def test_session_stream_does_not_read_body(reader, case, session_stream, args, kwargs, expected_stream):
    """Read bodies and record their size only when streaming is disabled.

    Args:
        reader: Metric reader for this process.
        case: Unique name for this test case.
        session_stream: Session default for streaming.
        args: Positional request arguments after the URL.
        kwargs: Keyword request arguments.
        expected_stream: Whether the response body must remain unread.
    """
    connector = f"stream_{case}"
    adapter = FakeAdapter()
    with telemetry.MeteredSession(connector) as session:
        session.stream = session_stream
        session.mount("http://", adapter)
        response = session.request("GET", "http://h/", *args, **kwargs)
        if expected_stream is True:
            assert response.raw.tell() == 0
            assert points(reader, connector, "response.size") == []
        else:
            assert response.raw.tell() == 5
            assert response.content == b"hello"
            (size,) = points(reader, connector, "response.size")
            assert size.value == 5
        assert points(reader, connector, "successes")[0].value == 1


def test_prometheus_endpoint():
    """Verify the actual scrape names in a separate provider process."""
    pytest.importorskip("opentelemetry.exporter.prometheus")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import socket
from urllib.request import urlopen
from attribution_blueprint.telemetry import enable_prometheus_endpoint_from_env, track

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
os.environ["ATTRIBUTION_METRICS_ADDR"] = f"127.0.0.1:{port}"
enable_prometheus_endpoint_from_env(os.environ)
with track("scrape_test", "read") as call:
    call.response_bytes = 12
try:
    with track("scrape_test", "read"):
        raise ValueError("failed")
except ValueError:
    pass
with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
    print(response.read().decode())
""",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    for name in (
        "attribution_connector_requests_total",
        "attribution_connector_successes_total",
        "attribution_connector_errors_total",
        "attribution_connector_response_size_bytes_total",
        "attribution_connector_request_duration_seconds_bucket",
        "attribution_connector_request_duration_seconds_sum",
        "attribution_connector_request_duration_seconds_count",
    ):
        assert name in result.stdout
    assert 'connector="scrape_test"' in result.stdout
    assert 'operation="read"' in result.stdout
    assert 'error_type="ValueError"' in result.stdout


def test_prometheus_endpoint_bind_error(mocker):
    """Report the address without setting a provider when the server cannot bind.

    Args:
        mocker: Pytest mock helper.

    Returns:
        None.
    """
    pytest.importorskip("opentelemetry.exporter.prometheus")
    error = OSError(98, "Address already in use")
    mocker.patch("prometheus_client.start_http_server", side_effect=error)
    set_provider = mocker.patch.object(metrics, "set_meter_provider")
    with pytest.raises(RuntimeError, match=r"127\.0\.0\.1:9464.*Address already in use") as raised:
        telemetry.enable_prometheus_endpoint("127.0.0.1", 9464)
    assert raised.value.__cause__ is error
    set_provider.assert_not_called()


def test_cli_prometheus_endpoint_bind_error(monkeypatch, mocker):
    """Turn a metrics bind failure into a CLI argument error.

    Args:
        monkeypatch: Pytest patch helper.
        mocker: Pytest mock helper.

    Returns:
        None.
    """
    pytest.importorskip("opentelemetry.exporter.prometheus")
    from attribution_blueprint.cli import main

    mocker.patch("prometheus_client.start_http_server", side_effect=OSError(98, "Address already in use"))
    mocker.patch.object(metrics, "set_meter_provider")
    monkeypatch.setenv("ATTRIBUTION_METRICS_ADDR", "127.0.0.1:9464")
    with pytest.raises(SystemExit) as raised:
        main(["--config", "/nonexistent.toml"])
    assert raised.value.code == 2


def test_missing_metrics_extra(monkeypatch):
    """Explain how to install a missing exporter.

    Args:
        monkeypatch: Pytest patch helper.
    """
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.prometheus", None)
    with pytest.raises(RuntimeError, match=r"nvidia-attribution-blueprint\[metrics\]"):
        telemetry.enable_prometheus_endpoint("127.0.0.1", 9464)


@pytest.mark.parametrize("address", ["9464", "host:", "host:abc", "host:70000", ":9464", "host:0"])
def test_invalid_address(address, mocker):
    """Reject bad addresses before starting an endpoint.

    Args:
        address: Invalid environment value.
        mocker: Pytest mock helper.
    """
    enable = mocker.patch.object(telemetry, "enable_prometheus_endpoint")
    with pytest.raises(ValueError, match="ATTRIBUTION_METRICS_ADDR.*host:port"):
        telemetry.enable_prometheus_endpoint_from_env({"ATTRIBUTION_METRICS_ADDR": address})
    enable.assert_not_called()


@pytest.mark.parametrize("environ", [{}, {"ATTRIBUTION_METRICS_ADDR": ""}])
def test_endpoint_default_off(environ, mocker):
    """Keep an unset or empty address free of side effects.

    Args:
        environ: Environment variables to parse.
        mocker: Pytest mock helper.
    """
    enable = mocker.patch.object(telemetry, "enable_prometheus_endpoint")
    telemetry.enable_prometheus_endpoint_from_env(environ)
    enable.assert_not_called()


def test_ipv6_address(mocker):
    """Remove IPv6 brackets before starting the endpoint.

    Args:
        mocker: Pytest mock helper.
    """
    enable = mocker.patch.object(telemetry, "enable_prometheus_endpoint")
    telemetry.enable_prometheus_endpoint_from_env({"ATTRIBUTION_METRICS_ADDR": "[::1]:9464"})
    enable.assert_called_once_with("::1", 9464)


def test_noop_default():
    """Run measurements without an SDK provider or warnings."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from attribution_blueprint.telemetry import track
with track("noop", "read") as call:
    call.response_bytes = 12
""",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stderr == ""


def test_attribution_create_response_size(reader, mocker):
    """Measure service response bytes through the generated client.

    Args:
        reader: Metric reader for this process.
        mocker: Pytest mock helper.
    """
    from nvidia_attribution_client import AttributorInfoV2

    from attribution_blueprint.connectors.attribution import AttributionConnector

    body = b'{"attributor_id":"metrics-test","model_type":"NewResponseV2"}'
    mocker.patch(
        "nvidia_attribution_client.rest.RESTClientObject.request",
        return_value=SimpleNamespace(
            status=201, reason="Created", data=body, headers={"content-type": "application/json"}, read=lambda: body
        ),
    )
    info = AttributorInfoV2.model_validate(
        {
            "workload": {
                "id": "metrics-test",
                "type": "slurm",
                "status": "FAILED",
                "name": "test",
                "username": "test",
                "framework": "",
                "nodes": {},
                "timezone": "UTC",
                "job_start_time": "2026-05-01T12:00:00+00:00",
            },
            "metadata": {"cluster": "", "agent": "test", "tenant": "", "ruleset": ""},
        }
    )
    assert AttributionConnector("http://h").create_attributor(info) == "metrics-test"
    (size,) = points(reader, "attribution_service", "response.size")
    assert size.value == len(body)
    assert size.attributes == {"connector": "attribution_service", "operation": "create_attributor"}


@pytest.mark.parametrize("status", [422, 503])
def test_attribution_error_response_size(reader, mocker, status):
    """Count error response bytes through the generated client.

    Args:
        reader: Metric reader for this process.
        mocker: Pytest mock helper.
        status: HTTP error status to measure.
    """
    from nvidia_attribution_client import ApiException, AttributorInfoV2

    from attribution_blueprint.connectors.attribution import AttributionConnector

    body = b'{"detail":[]}'
    mocker.patch(
        "nvidia_attribution_client.rest.RESTClientObject.request",
        return_value=SimpleNamespace(
            status=status, reason="Error", data=body, headers={"content-type": "application/json"}, read=lambda: body
        ),
    )
    info = AttributorInfoV2.model_validate(
        {
            "workload": {
                "id": "metrics-test",
                "type": "slurm",
                "status": "FAILED",
                "name": "test",
                "username": "test",
                "framework": "",
                "nodes": {},
                "timezone": "UTC",
                "job_start_time": "2026-05-01T12:00:00+00:00",
            },
            "metadata": {"cluster": "", "agent": "test", "tenant": "", "ruleset": ""},
        }
    )
    before = sum(
        point.value
        for point in points(reader, "attribution_service", "response.size")
        if point.attributes["operation"] == "create_attributor"
    )
    with pytest.raises(ApiException) as raised:
        AttributionConnector("http://h").create_attributor(info)
    assert raised.value.status == status
    (failure,) = [
        point
        for point in points(reader, "attribution_service", "errors")
        if point.attributes["operation"] == "create_attributor" and point.attributes["error.type"] == str(status)
    ]
    assert failure.value == 1
    after = sum(
        point.value
        for point in points(reader, "attribution_service", "response.size")
        if point.attributes["operation"] == "create_attributor"
    )
    assert after - before == len(body)


def test_kubernetes_error_response_size(reader, mocker):
    """Preserve API errors and count the response bytes they carry.

    Args:
        reader: Metric reader for this process.
        mocker: Pytest mock helper.
    """
    from kubernetes.client.exceptions import ApiException

    from attribution_blueprint.kubernetes.client import _set_default_request_timeout

    error = ApiException(
        http_resp=SimpleNamespace(status=403, reason="Forbidden", data=b"denied", getheaders=lambda: {})
    )
    client = SimpleNamespace(call_api=mocker.Mock(side_effect=error))
    _set_default_request_timeout(client, 15)
    method = "ERROR_RESPONSE"
    before = sum(
        point.value
        for point in points(reader, "kubernetes", "response.size")
        if point.attributes["operation"] == method
    )
    with pytest.raises(ApiException) as raised:
        client.call_api("/api/v1/pods", method, _preload_content=False)
    assert raised.value is error
    (failure,) = [point for point in points(reader, "kubernetes", "errors") if point.attributes["operation"] == method]
    assert failure.value == 1
    assert failure.attributes["error.type"] == "403"
    (size,) = [
        point for point in points(reader, "kubernetes", "response.size") if point.attributes["operation"] == method
    ]
    assert size.value - before == 6


def test_kubernetes_response_size(reader, mocker):
    """Measure bytes for a DynamicClient request that does not stream.

    Args:
        reader: Metric reader for this process.
        mocker: Pytest mock helper.
    """
    from attribution_blueprint.kubernetes.client import _set_default_request_timeout

    response = mocker.Mock(data=b"body")
    client = SimpleNamespace(call_api=mocker.Mock(return_value=response))
    _set_default_request_timeout(client, 15)
    assert (
        client.call_api("/api/v1/pods", "GET", {}, [("labelSelector", "a=b")], {}, _preload_content=False) is response
    )
    (size,) = [
        point for point in points(reader, "kubernetes", "response.size") if point.attributes["operation"] == "GET"
    ]
    assert size.value == 4
    assert size.attributes == {"connector": "kubernetes", "operation": "GET"}


@pytest.mark.parametrize("keyword", [False, True])
def test_kubernetes_response_size_without_query_params(reader, mocker, keyword):
    """Count response bytes when query parameters are None.

    Args:
        reader: Metric reader for this process.
        mocker: Pytest mock helper.
        keyword: Whether to pass query parameters by keyword.
    """
    from attribution_blueprint.kubernetes.client import _set_default_request_timeout

    response = SimpleNamespace(data=b"body")
    client = SimpleNamespace(call_api=mocker.Mock(return_value=response))
    _set_default_request_timeout(client, 15)
    method = f"NO_QUERY_PARAMS_{keyword}"
    if keyword is True:
        result = client.call_api("/api/v1/pods", method, query_params=None, _preload_content=False)
    else:
        result = client.call_api("/api/v1/pods", method, {}, None, {}, _preload_content=False)
    assert result is response
    (size,) = [
        point for point in points(reader, "kubernetes", "response.size") if point.attributes["operation"] == method
    ]
    assert size.value == 4


@pytest.mark.parametrize("body, expected_size", [(None, None), ("body", None), (b"", 0)])
def test_kubernetes_response_body_types(reader, mocker, body, expected_size):
    """Record only byte bodies, including an empty byte body.

    Args:
        reader: Metric reader for this process.
        mocker: Pytest mock helper.
        body: Response data supplied by the client.
        expected_size: Expected byte count, or None when no size is recorded.
    """
    from attribution_blueprint.kubernetes.client import _set_default_request_timeout

    response = SimpleNamespace(data=body)
    client = SimpleNamespace(call_api=mocker.Mock(return_value=response))
    _set_default_request_timeout(client, 15)
    method = f"BODY_TYPE_{type(body).__name__}"
    assert client.call_api("/api/v1/pods", method, _preload_content=False) is response
    sizes = [
        point for point in points(reader, "kubernetes", "response.size") if point.attributes["operation"] == method
    ]
    if expected_size is None:
        assert sizes == []
    else:
        (size,) = sizes
        assert size.value == expected_size


@pytest.mark.parametrize("query_params", [[("watch", True)], [("follow", True)]])
@pytest.mark.parametrize("keyword", [False, True])
def test_kubernetes_stream_without_reading_response_body(reader, mocker, query_params, keyword):
    """Leave watch and log-follow response bodies unread.

    Args:
        reader: Metric reader for this process.
        mocker: Pytest mock helper.
        query_params: Query parameters that identify a stream.
        keyword: Whether to pass query parameters by keyword.
    """
    from attribution_blueprint.kubernetes.client import _set_default_request_timeout

    response = mocker.Mock()
    type(response).data = mocker.PropertyMock(side_effect=AssertionError("Response body must not be read"))
    client = SimpleNamespace(call_api=mocker.Mock(return_value=response))
    _set_default_request_timeout(client, 15)
    method = f"STREAM_{query_params[0][0]}_{keyword}"
    if keyword is True:
        result = client.call_api("/api/v1/pods", method, query_params=query_params, _preload_content=False)
    else:
        result = client.call_api("/api/v1/pods", method, {}, query_params, {}, _preload_content=False)
    assert result is response
    assert [
        point for point in points(reader, "kubernetes", "response.size") if point.attributes["operation"] == method
    ] == []


def test_kubernetes_timeout_without_reading_preloaded_response(reader, mocker):
    """Retain timeouts and method forms without reading preloaded responses.

    Args:
        reader: Metric reader for this process.
        mocker: Pytest mock helper.

    Returns:
        None.
    """
    from attribution_blueprint.kubernetes.client import _set_default_request_timeout

    response = mocker.Mock()
    type(response).data = mocker.PropertyMock(side_effect=AssertionError("Response body must not be read"))
    transport = mocker.Mock(return_value=response)
    client = SimpleNamespace(call_api=transport)
    _set_default_request_timeout(client, 15)
    assert client.call_api("/apis", "GET", _request_timeout=None) is response
    transport.assert_called_once_with("/apis", "GET", _request_timeout=15)
    assert client.call_api(resource_path="/apis", method="POST", _preload_content=True, _request_timeout=3) is response
    transport.assert_called_with(resource_path="/apis", method="POST", _preload_content=True, _request_timeout=3)
    assert [
        point for point in points(reader, "kubernetes", "response.size") if point.attributes["operation"] == "POST"
    ] == []
    assert any(point.attributes["operation"] == "POST" for point in points(reader, "kubernetes", "requests"))
