# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check connector measurements without contacting a backend."""

import io
import subprocess
import sys
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


def test_session_stream_does_not_read_body(reader):
    """Leave streamed bodies unread and omit their size.

    Args:
        reader: Metric reader for this process.
    """
    adapter = FakeAdapter()
    with telemetry.MeteredSession("stream") as session:
        session.mount("http://", adapter)
        response = session.get("http://h/", stream=True)
        assert response.raw.tell() == 0
        assert points(reader, "stream", "response.size") == []
        assert points(reader, "stream", "successes")[0].value == 1


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


def test_kubernetes_timeout_without_reading_response_body(reader, mocker):
    """Retain timeouts and request metrics without reading response bodies.

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
    assert points(reader, "kubernetes", "response.size") == []
    assert client.call_api(resource_path="/apis", method="POST", _preload_content=False, _request_timeout=3) is response
    transport.assert_called_with(resource_path="/apis", method="POST", _preload_content=False, _request_timeout=3)
    assert points(reader, "kubernetes", "response.size") == []
    assert any(point.attributes["operation"] == "POST" for point in points(reader, "kubernetes", "requests"))
