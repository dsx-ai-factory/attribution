# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PromQL query client shared by the connector and other callers.

`PromQLQueryClient` serves metric collection. It works against Prometheus, Mimir, and Cortex. Range and instant queries are
POSTed form-encoded so long `net_host_name=~"n1|n2|..."` selectors stay under
URL limits, and every returned sample carries its node identity so callers can
slice a batch result per context. Transient HTTP failures are retried by the
session (see `attribution_blueprint.connectors.auth.build_session`).

`PrometheusClient` is the small raw-result client the Kueue Pod placement
resolver uses.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, NotRequired, TypedDict

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from attribution_blueprint.connectors.auth import Auth, build_session
from attribution_blueprint.types import assert_aware

DEFAULT_STEP_SECONDS = 60
DEFAULT_NODE_LABEL = "net_host_name"
DEFAULT_CLUSTER_LABEL = "s_cluster"
DEFAULT_COMPONENT_LABEL = "device"

_RE2_METAS = frozenset(r".\+*?()[]{}|^$")


def re2_escape(value: str) -> str:
    """Escape only RE2 metacharacters so hyphenated hostnames stay valid."""
    return "".join("\\" + c if c in _RE2_METAS else c for c in value)


def promql_string(value: str) -> str:
    """Quote a string for use as a PromQL label-matcher value."""
    escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return f'"{escaped}"'


def nodes_regex(nodes: list[str]) -> str:
    """Render node names as the right-hand side of a `=~` selector."""
    return "|".join(re2_escape(node) for node in nodes if node)


class PrometheusMetricEntry(TypedDict):
    """One flattened sample in the shape the service's PROMETHEUS_METRICS parser reads."""

    timestamp: int  # seconds since the Unix epoch
    value: str
    metric: str
    labels: dict[str, str]
    node: str
    component: NotRequired[str]
    cluster: NotRequired[str]


class PromQLQueryClient:
    """Fetch PromQL range and instant results with node identity."""

    def __init__(
        self,
        *,
        endpoint: str,
        auth: Auth | None = None,
        timeout_seconds: float = 60.0,
        node_label: str = DEFAULT_NODE_LABEL,
        cluster_label: str = DEFAULT_CLUSTER_LABEL,
        component_label: str = DEFAULT_COMPONENT_LABEL,
        require_node: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        """Create a client.

        `require_node=True` drops series without the node label, which the
        service needs on every PROMETHEUS_METRICS entry. Set it to False for
        callers that consume series without node identity themselves.
        `session` is used as-is; otherwise `build_session(auth)` supplies one.
        """
        if not endpoint:
            raise ValueError("endpoint must be a nonempty PromQL base URL")
        if not node_label:
            raise ValueError("node_label must be a nonempty label name")
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout_seconds
        self._node_label = node_label
        self._cluster_label = cluster_label
        self._component_label = component_label
        self._require_node = require_node
        self._session = build_session(auth) if session is None else session

    def query_range(
        self,
        name: str,
        query: str,
        start: datetime,
        end: datetime,
        *,
        step_seconds: int = DEFAULT_STEP_SECONDS,
        cluster: str | None = None,
    ) -> list[PrometheusMetricEntry]:
        """Return flattened samples for `query` between `start` and `end`."""
        start = assert_aware(start, "start")
        end = assert_aware(end, "end")
        if end < start:
            raise ValueError("end must not precede start")
        if step_seconds <= 0:
            raise ValueError(f"step_seconds must be positive, got {step_seconds}")
        payload = self._post(
            "/api/v1/query_range",
            {"query": query, "start": int(start.timestamp()), "end": int(end.timestamp()), "step": step_seconds},
            name,
        )
        return self._flatten(name, payload, cluster)

    def query_instant(
        self,
        name: str,
        query: str,
        at: datetime,
        *,
        cluster: str | None = None,
    ) -> list[PrometheusMetricEntry]:
        """Return flattened samples for an instant `query` evaluated at `at`."""
        at = assert_aware(at, "at")
        payload = self._post("/api/v1/query", {"query": query, "time": int(at.timestamp())}, name)
        return self._flatten(name, payload, cluster)

    def _post(self, path: str, data: dict[str, object], name: str) -> Mapping[str, object]:
        response = self._session.post(f"{self._endpoint}{path}", data=data, timeout=self._timeout)
        response.raise_for_status()
        payload = response.json() or {}
        if not isinstance(payload, Mapping):
            payload = {}
        # Prometheus/Mimir returns HTTP 200 with `{"status":"error", "error":...}`
        # for execution failures (query timeout, sample-limit, parse error).
        # `raise_for_status` doesn't catch these; verifying status=="success"
        # surfaces them as a warning instead of silently dropping data.
        if payload.get("status") != "success":
            logger.warning(
                f"promql query {name!r} returned status={payload.get('status')!r}: "
                f"{payload.get('error') or payload.get('errorType') or '<no detail>'}"
            )
            return {}
        return payload

    def _flatten(self, name: str, payload: Mapping[str, object], cluster: str | None) -> list[PrometheusMetricEntry]:
        data = payload.get("data", {})
        result = data.get("result", []) if isinstance(data, Mapping) else []
        out: list[PrometheusMetricEntry] = []
        for series in result if isinstance(result, list) else []:
            if not isinstance(series, Mapping):
                continue
            labels = series.get("metric", {}) or {}
            if not isinstance(labels, Mapping):
                continue
            labels = {str(key): str(value) for key, value in labels.items()}
            node = labels.get(self._node_label, "")
            if not node and self._require_node:
                # The service requires node identity on every PROMETHEUS_METRICS
                # entry. Aggregations that drop the node label are
                # unattributable; skip rather than emit unscoped points.
                continue
            # Aggregating templates (e.g. `max by (net_host_name) (...)`) drop
            # the cluster label, so fall back to the caller's cluster.
            series_cluster = labels.get(self._cluster_label) or cluster
            component = labels.get(self._component_label)
            instant = series.get("value")
            values = [instant] if isinstance(instant, list) else series.get("values", []) or []
            for sample in values if isinstance(values, list) else []:
                if not isinstance(sample, list) or len(sample) < 2:
                    continue
                try:
                    timestamp = int(float(sample[0]))
                except (TypeError, ValueError):
                    continue
                entry: PrometheusMetricEntry = {
                    "timestamp": timestamp,
                    "value": str(sample[1]),
                    "metric": name,
                    "labels": labels,
                    "node": node,
                }
                if series_cluster:
                    entry["cluster"] = series_cluster
                if component:
                    entry["component"] = component
                out.append(entry)
        return out


class PrometheusClient:
    """Small synchronous client for Prometheus-compatible query APIs."""

    def __init__(
        self,
        *,
        endpoint: str,
        auth: Auth | None = None,
        timeout_seconds: float = 60.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=0.5,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        if auth is not None:
            kwargs = auth.request_kwargs()
            if "headers" in kwargs:
                self.session.headers.update(kwargs["headers"])
            if "cert" in kwargs:
                self.session.cert = kwargs["cert"]
            if "verify" in kwargs:
                self.session.verify = kwargs["verify"]

    def query(self, *, name: str, query: str, timestamp: int) -> list[dict[str, Any]]:
        return self._post_result(
            name=name,
            path="/api/v1/query",
            data={"query": query, "time": timestamp},
        )

    def query_range(
        self,
        *,
        name: str,
        query: str,
        start: int,
        end: int,
        step: int,
    ) -> list[dict[str, Any]]:
        return self._post_result(
            name=name,
            path="/api/v1/query_range",
            data={"query": query, "start": start, "end": end, "step": step},
        )

    def _post_result(self, *, name: str, path: str, data: dict[str, Any]) -> list[dict[str, Any]]:
        response = self.session.post(
            f"{self.endpoint}{path}",
            data=data,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json() or {}
        if payload.get("status") != "success":
            logger.warning(
                f"promql query {name!r} returned status={payload.get('status')!r}: "
                f"{payload.get('error') or payload.get('errorType') or '<no detail>'}"
            )
            return []
        result = payload.get("data", {}).get("result", []) or []
        if not isinstance(result, list):
            logger.warning(f"promql query {name!r} returned a non-list result")
            return []
        return result


__all__ = [
    "DEFAULT_CLUSTER_LABEL",
    "DEFAULT_COMPONENT_LABEL",
    "DEFAULT_NODE_LABEL",
    "DEFAULT_STEP_SECONDS",
    "PrometheusClient",
    "PrometheusMetricEntry",
    "PromQLQueryClient",
    "nodes_regex",
    "promql_string",
    "re2_escape",
]
