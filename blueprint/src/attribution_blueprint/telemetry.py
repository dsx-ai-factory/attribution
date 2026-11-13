# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure connector requests through the optional OpenTelemetry provider."""

import inspect
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

import requests
from loguru import logger
from opentelemetry import metrics

_request_signature = inspect.signature(requests.Session.request)

_meter = metrics.get_meter("attribution_blueprint")
_requests = _meter.create_counter(
    "attribution.connector.requests", unit="{request}", description="Connector requests started."
)
_successes = _meter.create_counter(
    "attribution.connector.successes", unit="{request}", description="Connector requests that succeeded."
)
_errors = _meter.create_counter(
    "attribution.connector.errors", unit="{request}", description="Connector requests that failed."
)
_response_size = _meter.create_counter(
    "attribution.connector.response.size", unit="By", description="Decoded response bytes received."
)
_duration = _meter.create_histogram(
    "attribution.connector.request.duration",
    unit="s",
    description="Time spent on each connector request.",
    explicit_bucket_boundaries_advisory=[
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
    ],
)


def enable_prometheus_endpoint(host: str, port: int) -> None:
    """Start a Prometheus scrape endpoint.

    Args:
        host: Address on which to listen.
        port: TCP port on which to listen.
    """
    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider
        from prometheus_client import start_http_server
    except ImportError as error:
        raise RuntimeError(
            "The metrics extra is not installed; install nvidia-attribution-blueprint[metrics]"
        ) from error
    reader = PrometheusMetricReader()
    try:
        start_http_server(port, addr=host)
    except OSError as error:
        raise RuntimeError(f"Cannot start the metrics endpoint on {host}:{port}: {error.strerror or error}") from error
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
    logger.info("Connector metrics endpoint listening on {}:{}", host, port)


def enable_prometheus_endpoint_from_env(environ: Mapping[str, str]) -> None:
    """Start the endpoint only when ATTRIBUTION_METRICS_ADDR is set.

    Args:
        environ: Environment variables for this process.
    """
    address = environ.get("ATTRIBUTION_METRICS_ADDR")
    if address is None or address == "":
        return
    host, _, port_text = address.rpartition(":")
    host = host.strip("[]")
    try:
        port = int(port_text)
        if host == "" or not 1 <= port <= 65535:
            raise ValueError
    except ValueError as error:
        raise ValueError(
            "ATTRIBUTION_METRICS_ADDR must use host:port with a non-empty host and port in 1..65535"
        ) from error
    enable_prometheus_endpoint(host, port)


@dataclass
class _Call:
    """Response measurements supplied by the connector."""

    response_bytes: int | None = None
    error_type: str | None = None


@contextmanager
def track(connector: str, operation: str) -> Iterator[_Call]:
    """Record one request and preserve any exception from its body.

    Args:
        connector: Fixed connector name.
        operation: Fixed path, method, or command name.

    Returns:
        A context manager that yields mutable response measurements.
    """
    labels = {"connector": connector, "operation": operation}
    call = _Call()
    _requests.add(1, labels)
    start = time.perf_counter()
    try:
        yield call
    except BaseException as error:
        status = getattr(error, "status", None)
        call.error_type = str(status) if isinstance(status, int) and status > 0 else type(error).__name__
        # Generated API clients raise ApiException with the error body they already read.
        body = getattr(error, "body", None)
        if isinstance(body, str):
            body = body.encode("utf-8")
        if isinstance(body, bytes) and call.response_bytes is None:
            call.response_bytes = len(body)
        raise
    finally:
        elapsed = time.perf_counter() - start
        duration_labels = labels
        if call.error_type is not None:
            duration_labels = {**labels, "error.type": call.error_type}
            _errors.add(1, duration_labels)
        else:
            _successes.add(1, labels)
        _duration.record(elapsed, duration_labels)
        if call.response_bytes is not None:
            _response_size.add(call.response_bytes, labels)


class MeteredSession(requests.Session):
    """Measure HTTP calls without changing response or retry behavior."""

    def __init__(self, connector: str, operations: tuple[str, ...] = ()):
        """Use fixed paths to keep IDs out of metric labels.

        Args:
            connector: Fixed connector name.
            operations: Allowed API path suffixes.
        """
        super().__init__()
        self._connector = connector
        self._operations = operations

    def request(self, method, url, *args, **kwargs):
        """Measure one HTTP call, including its transport retries.

        Args:
            method: HTTP method.
            url: Request URL.
            *args: Positional session arguments.
            **kwargs: Keyword session arguments.

        Returns:
            The unchanged HTTP response.
        """
        path = urlsplit(url).path
        operation = next((allowed for allowed in self._operations if path.endswith(allowed)), method.upper())
        with track(self._connector, operation) as call:
            response = super().request(method, url, *args, **kwargs)
            if response.status_code >= 400:
                call.error_type = str(response.status_code)
            stream = _request_signature.bind(self, method, url, *args, **kwargs).arguments.get("stream")
            if stream is None:
                stream = self.stream
            if not stream:
                call.response_bytes = len(response.content)
            return response
