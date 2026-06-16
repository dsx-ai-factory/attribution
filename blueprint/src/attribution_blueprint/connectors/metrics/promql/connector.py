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
from typing import NotRequired, TypedDict

import requests
from loguru import logger

from attribution_blueprint.connectors.auth import Auth
from attribution_blueprint.connectors.metrics.promql.templates import TEMPLATES
from attribution_blueprint.types import (
    ExecutionContext,
    Format,
    Observation,
    Source,
    assert_aware,
)

DEFAULT_STEP_SECONDS = 60
DEFAULT_NODE_LABEL = "net_host_name"
DEFAULT_CLUSTER_LABEL = "s_cluster"
DEFAULT_COMPONENT_LABEL = "device"

# Prometheus and Mimir use RE2 for label-match regexes. Python's re.escape()
# backslashes too much for RE2, including hyphens (`\-`), which RE2 rejects.
_RE2_METAS = frozenset(r".\+*?()[]{}|^$")


def _re2_escape(s: str) -> str:
    return "".join("\\" + c if c in _RE2_METAS else c for c in s)


class PrometheusMetricEntry(TypedDict):
    timestamp: int
    value: str
    metric: str
    labels: dict[str, str]
    node: str
    component: NotRequired[str]
    cluster: NotRequired[str]


class PromQLMetricsConnector:
    """Collector that fetches one Source from a PromQL backend (Mimir, Prometheus, Cortex).

    Each instance handles exactly one Source. For multiple Sources from the same
    backend, instantiate per Source — failure isolation lives at the instance
    boundary. Templates are bundled with the connector; the customer supplies a
    cluster name and (optionally) a templates dict to extend or override.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        cluster: str,
        auth: Auth | None = None,
        source: Source = Source.NODE_METRICS,
        templates: dict[Source, dict[str, str]] = TEMPLATES,
        timeout_seconds: float = 60.0,
        step_seconds: int = DEFAULT_STEP_SECONDS,
    ):
        if source not in templates:
            raise ValueError(f"no templates registered for source {source}")
        self._source = source
        self._templates = templates[source]
        self._endpoint = endpoint.rstrip("/")
        self._cluster = cluster
        self._timeout = timeout_seconds
        self._step = step_seconds
        self._auth = auth

        self._session = requests.Session()
        if auth is not None:
            kwargs = auth.request_kwargs()
            if "headers" in kwargs:
                self._session.headers.update(kwargs["headers"])
            if "cert" in kwargs:
                self._session.cert = kwargs["cert"]
            if "verify" in kwargs:
                self._session.verify = kwargs["verify"]

    @property
    def source(self) -> Source:
        return self._source

    def collect(self, context: ExecutionContext) -> list[Observation]:
        # Drop empty strings: `re.escape("")` is `""`, which would render the
        # selector as `net_host_name=~""` and match every series in the cluster.
        nodes = [n for n in (context.get("nodes") or []) if n]
        if not nodes:
            return []
        # Reject naive datetimes; the service rejects them and parsing them
        # into Unix epoch silently anchors to the blueprint host's local TZ.
        start = assert_aware(context["start_time"], "start_time")
        end = assert_aware(context["end_time"], "end_time")
        start_s = int(start.timestamp())
        end_s = int(end.timestamp())
        entries: list[PrometheusMetricEntry] = []
        # Each node lands inside a PromQL `=~ "..."` regex on net_host_name,
        # so regex metacharacters in hostnames would silently broaden the
        # match. Escape only RE2 metacharacters so common hyphenated hostnames
        # don't produce invalid regexes.
        nodes_regex = "|".join(_re2_escape(n) for n in nodes)
        for name, tmpl in self._templates.items():
            try:
                query = tmpl.format(name=name, cluster=self._cluster, nodes=nodes_regex)
            except KeyError as e:
                logger.warning(f"promql template {name!r} missing kwarg {e}; skipping")
                continue
            entries.extend(self._query_range(name, query, start_s, end_s))

        if not entries:
            return []
        return [
            {
                "context": {
                    "time_interval": {"start": context["start_time"], "end": context["end_time"]},
                    "source": self._source,
                    "format": Format.PROMETHEUS_METRICS,
                    "resources": "AllJobResources",
                },
                "body": json.dumps(entries),
            }
        ]

    def _query_range(self, name: str, query: str, start_s: int, end_s: int) -> list[PrometheusMetricEntry]:
        # POST form-encoded rather than GET: ExecutionContexts with many nodes produce
        # `net_host_name=~"n1|n2|..."` selectors that easily blow past
        # Mimir/Cortex default URL limits (~4-8 KB). The Prometheus HTTP API
        # accepts POST on /api/v1/query_range for exactly this reason and the
        # response shape is identical.
        response = self._session.post(
            f"{self._endpoint}/api/v1/query_range",
            data={"query": query, "start": start_s, "end": end_s, "step": self._step},
            timeout=self._timeout,
        )
        response.raise_for_status()
        # Prometheus/Mimir returns HTTP 200 with `{"status":"error", "error":...}`
        # for execution failures (query timeout, sample-limit, parse error).
        # `raise_for_status` doesn't catch these; verifying status=="success"
        # surfaces them as a warning instead of silently dropping data.
        payload = response.json() or {}
        if payload.get("status") != "success":
            logger.warning(
                f"promql query {name!r} returned status={payload.get('status')!r}: "
                f"{payload.get('error') or payload.get('errorType') or '<no detail>'}"
            )
            return []
        result = payload.get("data", {}).get("result", []) or []
        out: list[PrometheusMetricEntry] = []
        for series in result:
            labels = series.get("metric", {}) or {}
            node = labels.get(DEFAULT_NODE_LABEL)
            if not node:
                # The service requires node identity on every PROMETHEUS_METRICS
                # entry. Aggregations that drop net_host_name are unattributable;
                # skip rather than emit unscoped points.
                continue
            # Templates that aggregate (e.g. `max by (net_host_name) (...)`)
            # drop s_cluster, so fall back to the connector's configured
            # cluster — the value is known either way.
            cluster = labels.get(DEFAULT_CLUSTER_LABEL) or self._cluster
            component = labels.get(DEFAULT_COMPONENT_LABEL)
            for ts, val in series.get("values", []) or []:
                try:
                    timestamp = int(float(ts))
                except (TypeError, ValueError):
                    continue
                entry: PrometheusMetricEntry = {
                    "timestamp": timestamp,
                    "value": str(val),
                    "metric": name,
                    "labels": labels,
                    "node": node,
                }
                if cluster:
                    entry["cluster"] = cluster
                if component:
                    entry["component"] = component
                out.append(entry)
        return out
