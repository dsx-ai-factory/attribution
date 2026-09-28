# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.auth import AUTH_SCHEMA, build_auth
from attribution_blueprint.connectors.logs.loki.client import (
    LokiEntry,
    LokiQueryBudgetExceeded,
    LokiQueryClient,
    node_from_labels,
)
from attribution_blueprint.connectors.logs.loki.connector import LokiConnector
from attribution_blueprint.connectors.logs.loki.range_cache import FileRangeCacheStore, LokiRangeCache
from attribution_blueprint.connectors.logs.loki.templates import BARE_TEMPLATES, TEMPLATES
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM, _coerce_bool, _coerce_float, _coerce_int
from attribution_blueprint.types import Source


def _build(spec: dict, ctx: dict) -> LokiConnector:
    source: Source = ctx["source"]
    prefix = f"log.{source.name.lower()}.loki"
    endpoint = spec.get("endpoint")
    if not endpoint:
        raise MCAError(f"{prefix} requires endpoint")
    auth_spec = spec.get("auth")
    if not auth_spec:
        raise MCAError(f"{prefix} requires auth")
    auth = build_auth(auth_spec, "loki.auth")
    timeout = _coerce_float(spec.get("timeout_seconds", 60.0), "timeout_seconds")

    queries = spec.get("queries")
    if queries is None and "query" in spec:
        queries = [spec["query"]]
    elif isinstance(queries, str):
        queries = [queries]

    kwargs: dict[str, Any] = dict(
        endpoint=endpoint,
        auth=auth,
        source=source,
        query_kwargs=dict(spec.get("query_kwargs", {})),
        timeout_seconds=timeout,
    )
    if queries is not None:
        kwargs["templates"] = {source: list(queries)}
    # `node_label` controls which Loki stream label the connector reads to
    # populate the per-entry `node` field and to splice the per-ExecutionContext
    # node filter into queries. Defaults to "host_name" (the bundled set's
    # convention); deployments whose streams use another label override here.
    if "node_label" in spec:
        kwargs["node_label"] = spec["node_label"]
    for key in ("limit", "node_chunk_size"):
        if key in spec:
            value = _coerce_int(spec[key], f"{prefix}.{key}")
            if value <= 0:
                raise MCAError(f"{prefix}.{key} must be positive, got {value}")
            kwargs[key] = value
    if "batch_window_hours" in spec:
        hours = _coerce_float(spec["batch_window_hours"], f"{prefix}.batch_window_hours")
        if hours < 0:
            raise MCAError(f"{prefix}.batch_window_hours must not be negative, got {hours}")
        kwargs["batch_window_hours"] = hours
    if "batch_node_filter" in spec:
        kwargs["batch_node_filter"] = _coerce_bool(spec["batch_node_filter"], f"{prefix}.batch_node_filter")
    cache_dir = spec.get("cache_dir")
    if cache_dir:
        kwargs["range_cache"] = LokiRangeCache(FileRangeCacheStore(str(cache_dir)))
    try:
        return LokiConnector(**kwargs)
    except ValueError as exc:
        raise MCAError(f"{prefix}: {exc}") from exc


_registry.register_component(
    framework="log",
    name="loki",
    schema={
        "endpoint": _PARAM,
        "timeout_seconds": _PARAM,
        "query": _PARAM,
        "queries": _PARAM,
        "node_label": _PARAM,
        "limit": _PARAM,
        "batch_window_hours": _PARAM,
        "batch_node_filter": _PARAM,
        "node_chunk_size": _PARAM,
        "cache_dir": _PARAM,
        "auth": AUTH_SCHEMA,
        "query_kwargs": {"_open": True},
    },
    build=_build,
)

__all__ = [
    "BARE_TEMPLATES",
    "FileRangeCacheStore",
    "LokiConnector",
    "LokiEntry",
    "LokiQueryBudgetExceeded",
    "LokiQueryClient",
    "LokiRangeCache",
    "TEMPLATES",
    "node_from_labels",
]
