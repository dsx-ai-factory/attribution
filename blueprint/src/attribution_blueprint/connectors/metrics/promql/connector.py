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

from loguru import logger

from attribution_blueprint.connectors.auth import Auth
from attribution_blueprint.connectors.metrics.promql.client import (
    PrometheusClient,
    re2_escape,
)
from attribution_blueprint.connectors.metrics.promql.templates import TEMPLATES
from attribution_blueprint.types import (
    ExecutionContext,
    Observation,
    Source,
    TimeseriesEntry,
    assert_aware,
)

DEFAULT_STEP_SECONDS = 60
DEFAULT_NODE_LABEL = "net_host_name"
DEFAULT_CLUSTER_LABEL = "s_cluster"
DEFAULT_COMPONENT_LABEL = "device"
_NS = 10**9  # PromQL sample times are unix seconds; `ts` is nanoseconds


class PrometheusMetricEntry(TypedDict):
    timestamp: int
    value: str
    metric: str
    labels: dict[str, str]
    node: str
    component: NotRequired[str]
    cluster: NotRequired[str]


def _render_metric_body(metric: PrometheusMetricEntry) -> str:
    """Render one sample as the line the service's node-metric rules match.

    The rules read only the `metric_name: <name>, metric_value: <value>` pair; the
    remaining fields keep the line identical to what other producers emit.
    """
    raw_value = metric.get("value", 0)
    metric_value: int | float
    if isinstance(raw_value, str):
        try:
            metric_value = int(raw_value)
        except ValueError:
            metric_value = float(raw_value)
    else:
        metric_value = raw_value

    labels = metric.get("labels", {})
    # Extract component information - preserve empty strings, only None if truly missing
    component = metric.get("component")
    if component is None:
        component = labels.get("device")
    # Extract cluster information - preserve empty strings, only None if truly missing
    cluster = metric.get("cluster")
    if cluster is None:
        cluster = labels.get("s_cluster")

    return (
        f"time: {metric['timestamp']}, metric_name: {metric['metric']}, metric_value: {metric_value}, "
        f"threshold: {metric.get('threshold')}, cluster: {cluster}, node: {metric['node']}, component: {component}"
    )


def _timeseries_entry(sample: PrometheusMetricEntry) -> TimeseriesEntry:
    return {
        # `ts` is what the service filters on; the body's `time:` (seconds) is never
        # parsed, it only keeps the line identical to other producers' output.
        "ts": sample["timestamp"] * _NS,
        "resources": [{"kind": "node", "hostname": sample["node"]}],
        "body": _render_metric_body(sample),
    }


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
        self._client = PrometheusClient(
            endpoint=endpoint,
            auth=auth,
            timeout_seconds=timeout_seconds,
        )
        # Kept as an alias for callers that inspect the configured session.
        self._session = self._client.session

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
        samples: list[PrometheusMetricEntry] = []
        # Each node lands inside a PromQL `=~ "..."` regex on net_host_name,
        # so regex metacharacters in hostnames would silently broaden the
        # match. Escape only RE2 metacharacters so common hyphenated hostnames
        # don't produce invalid regexes.
        nodes_regex = "|".join(re2_escape(n) for n in nodes)
        for name, tmpl in self._templates.items():
            try:
                query = tmpl.format(name=name, cluster=self._cluster, nodes=nodes_regex)
            except KeyError as e:
                logger.warning(f"promql template {name!r} missing kwarg {e}; skipping")
                continue
            samples.extend(self._query_range(name, query, start_s, end_s))

        if not samples:
            return []
        return [
            {
                "context": {
                    "time_interval": {"start": context["start_time"], "end": context["end_time"]},
                    "source": self._source,
                },
                "body": json.dumps([_timeseries_entry(sample) for sample in samples]),
            }
        ]

    def _query_range(self, name: str, query: str, start_s: int, end_s: int) -> list[PrometheusMetricEntry]:
        result = self._client.query_range(
            name=name,
            query=query,
            start=start_s,
            end=end_s,
            step=self._step,
        )
        out: list[PrometheusMetricEntry] = []
        for series in result:
            labels = series.get("metric", {}) or {}
            node = labels.get(DEFAULT_NODE_LABEL)
            if not node:
                # A node_metrics entry must declare its node; the service never
                # parses one out of a metric body. Aggregations that drop
                # net_host_name are unattributable; skip rather than emit
                # unscoped points.
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
