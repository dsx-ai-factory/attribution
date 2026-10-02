# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.auth import AUTH_SCHEMA, build_auth
from attribution_blueprint.connectors.metrics.promql.client import (
    PrometheusMetricEntry,
    PromQLQueryClient,
    nodes_regex,
)
from attribution_blueprint.connectors.metrics.promql.connector import (
    PromQLMetricsConnector,
)
from attribution_blueprint.connectors.metrics.promql.templates import TEMPLATES
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM
from attribution_blueprint.types import Source


def _build(spec: dict, ctx: dict) -> PromQLMetricsConnector:
    from attribution_blueprint.mca import _coerce_float, _coerce_int

    source: Source = ctx["source"]
    prefix = f"metrics.{source.name.lower()}.promql"
    endpoint = spec.get("endpoint")
    if not endpoint:
        raise MCAError(f"{prefix} requires endpoint")
    cluster = spec.get("cluster")
    if not cluster:
        raise MCAError(f"{prefix} requires cluster")
    auth_spec = spec.get("auth")
    auth = build_auth(auth_spec, "promql.auth") if auth_spec is not None else None
    timeout = _coerce_float(spec.get("timeout_seconds", 60.0), "timeout_seconds")
    step = _coerce_int(spec.get("step_seconds", 60), "step_seconds")

    queries = spec.get("queries") or {}
    if queries:
        templates = {**TEMPLATES, source: {**TEMPLATES.get(source, {}), **queries}}
    else:
        templates = TEMPLATES

    kwargs: dict[str, Any] = dict(
        endpoint=endpoint,
        auth=auth,
        cluster=cluster,
        source=source,
        templates=templates,
        timeout_seconds=timeout,
        step_seconds=step,
    )
    if "node_chunk_size" in spec:
        value = _coerce_int(spec["node_chunk_size"], f"{prefix}.node_chunk_size")
        if value <= 0:
            raise MCAError(f"{prefix}.node_chunk_size must be positive, got {value}")
        kwargs["node_chunk_size"] = value
    if "lookahead_seconds" in spec:
        value = _coerce_int(spec["lookahead_seconds"], f"{prefix}.lookahead_seconds")
        if value < 0:
            raise MCAError(f"{prefix}.lookahead_seconds must not be negative, got {value}")
        kwargs["lookahead_seconds"] = value
    for key in ("node_label", "cluster_label", "component_label"):
        if key in spec:
            value = str(spec[key])
            if value == "":
                raise MCAError(f"{prefix}.{key} must be a nonempty label name")
            kwargs[key] = value
    try:
        return PromQLMetricsConnector(**kwargs)
    except ValueError as exc:
        raise MCAError(f"{prefix}: {exc}") from exc


_registry.register_component(
    framework="metrics",
    name="promql",
    schema={
        "endpoint": _PARAM,
        "cluster": _PARAM,
        "timeout_seconds": _PARAM,
        "step_seconds": _PARAM,
        "node_chunk_size": _PARAM,
        "lookahead_seconds": _PARAM,
        "node_label": _PARAM,
        "cluster_label": _PARAM,
        "component_label": _PARAM,
        "auth": AUTH_SCHEMA,
        "queries": {"_open": True},
    },
    build=_build,
)

__all__ = ["PromQLMetricsConnector", "PromQLQueryClient", "PrometheusMetricEntry", "TEMPLATES", "nodes_regex"]
