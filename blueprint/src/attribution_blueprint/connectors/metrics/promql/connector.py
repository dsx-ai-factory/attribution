# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import requests
from loguru import logger

from attribution_blueprint.connectors.auth import Auth
from attribution_blueprint.connectors.metrics.promql.client import (
    DEFAULT_CLUSTER_LABEL,
    DEFAULT_COMPONENT_LABEL,
    DEFAULT_NODE_LABEL,
    DEFAULT_STEP_SECONDS,
    PrometheusMetricEntry,
    PromQLQueryClient,
    nodes_regex,
)
from attribution_blueprint.connectors.metrics.promql.templates import TEMPLATES
from attribution_blueprint.types import (
    ExecutionContext,
    Observation,
    Source,
    TimeseriesEntry,
    assert_aware,
)

DEFAULT_NODE_CHUNK_SIZE = 100
DEFAULT_LOOKAHEAD_SECONDS = 300
# Concurrent queries per batch; enough to overlap network latency without
# hammering the backend.
_MAX_IN_FLIGHT = 4
_NS = 10**9  # PromQL sample times are unix seconds; `ts` is nanoseconds

__all__ = ["PrometheusMetricEntry", "PromQLMetricsConnector", "DEFAULT_NODE_LABEL", "DEFAULT_STEP_SECONDS"]


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

    `collect(context)` queries per context. `collect_batch(contexts)` queries the
    union of a batch's nodes once per template (chunked, concurrent, best
    effort) and slices the samples per context by node and time.
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
        node_chunk_size: int = DEFAULT_NODE_CHUNK_SIZE,
        lookahead_seconds: int = DEFAULT_LOOKAHEAD_SECONDS,
        node_label: str = DEFAULT_NODE_LABEL,
        cluster_label: str = DEFAULT_CLUSTER_LABEL,
        component_label: str = DEFAULT_COMPONENT_LABEL,
        session: requests.Session | None = None,
    ):
        if source not in templates:
            raise ValueError(f"no templates registered for source {source}")
        if step_seconds <= 0:
            raise ValueError(f"step_seconds must be positive, got {step_seconds}")
        if node_chunk_size <= 0:
            raise ValueError(f"node_chunk_size must be positive, got {node_chunk_size}")
        if lookahead_seconds < 0:
            raise ValueError("lookahead_seconds must not be negative")
        self._source = source
        self._templates = templates[source]
        self._endpoint = endpoint.rstrip("/")
        self._cluster = cluster
        self._timeout = timeout_seconds
        self._step = step_seconds
        self._auth = auth
        self._node_chunk_size = node_chunk_size
        self._lookahead = timedelta(seconds=lookahead_seconds)
        self._client = PromQLQueryClient(
            endpoint=endpoint,
            auth=auth,
            timeout_seconds=timeout_seconds,
            node_label=node_label,
            cluster_label=cluster_label,
            component_label=component_label,
            session=session,
        )

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
        entries: list[PrometheusMetricEntry] = []
        for name, query in self._render(nodes):
            entries.extend(
                self._client.query_range(name, query, start, end, step_seconds=self._step, cluster=self._cluster)
            )
        return self._observation(context, entries)

    def collect_batch(self, contexts: Sequence[ExecutionContext]) -> dict[str, list[Observation]]:
        """Query the union of the batch's nodes once per template and slice per context."""
        contexts = list(contexts)
        if not contexts:
            return {}
        windows = {
            str(ctx["id"]): (
                assert_aware(ctx["start_time"], "start_time"),
                assert_aware(ctx["end_time"], "end_time") + self._lookahead,
            )
            for ctx in contexts
        }
        nodes = sorted({node for ctx in contexts for node in (ctx.get("nodes") or []) if node})
        if not nodes:
            return {str(ctx["id"]): [] for ctx in contexts}
        start = min(window[0] for window in windows.values())
        end = max(window[1] for window in windows.values())
        chunks = [
            nodes[offset : offset + self._node_chunk_size] for offset in range(0, len(nodes), self._node_chunk_size)
        ]
        queries = [(name, query) for chunk in chunks for name, query in self._render(chunk)]
        entries = self._query_batch(queries, start, end)
        result: dict[str, list[Observation]] = {}
        for ctx in contexts:
            ctx_id = str(ctx["id"])
            window_start, window_end = windows[ctx_id]
            ctx_nodes = {node.lower() for node in (ctx.get("nodes") or []) if node}
            start_s = int(window_start.timestamp())
            end_s = int(window_end.timestamp())
            selected = [
                entry
                for entry in entries
                if entry["node"].lower() in ctx_nodes and start_s <= entry["timestamp"] <= end_s
            ]
            result[ctx_id] = self._observation(ctx, selected)
        return result

    def _render(self, nodes: list[str]) -> list[tuple[str, str]]:
        rendered: list[tuple[str, str]] = []
        regex = nodes_regex(nodes)
        for name, tmpl in self._templates.items():
            try:
                rendered.append((name, tmpl.format(name=name, cluster=self._cluster, nodes=regex)))
            except KeyError as e:
                logger.warning(f"promql template {name!r} missing kwarg {e}; skipping")
        return rendered

    def _query_batch(
        self, queries: list[tuple[str, str]], start: datetime, end: datetime
    ) -> list[PrometheusMetricEntry]:
        failures: list[Exception] = []

        def run(item: tuple[str, str]) -> list[PrometheusMetricEntry]:
            name, query = item
            try:
                return self._client.query_range(name, query, start, end, step_seconds=self._step, cluster=self._cluster)
            except Exception as exc:
                logger.warning(f"promql batch query {name!r} failed: {exc}")
                failures.append(exc)
                return []

        if len(queries) <= 1:
            pages = [run(item) for item in queries]
        else:
            with ThreadPoolExecutor(max_workers=min(_MAX_IN_FLIGHT, len(queries))) as executor:
                pages = list(executor.map(run, queries))
        if failures and len(failures) == len(queries):
            raise failures[0]
        return [entry for page in pages for entry in page]

    def _observation(self, context: ExecutionContext, entries: list[PrometheusMetricEntry]) -> list[Observation]:
        if not entries:
            return []
        return [
            {
                "context": {
                    "time_interval": {"start": context["start_time"], "end": context["end_time"]},
                    "source": self._source,
                },
                "body": json.dumps([_timeseries_entry(entry) for entry in entries]),
            }
        ]
