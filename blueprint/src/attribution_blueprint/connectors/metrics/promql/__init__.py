# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.auth import Auth, build_auth
from attribution_blueprint.connectors.metrics.promql.connector import (
    DEFAULT_CLUSTER_LABEL,
    DEFAULT_NODE_LABEL,
    PromQLMetricsConnector,
)
from attribution_blueprint.connectors.metrics.promql.templates import TEMPLATES
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM
from attribution_blueprint.types import Source


def _build_auth(spec: dict) -> Auth:
    return build_auth(spec, "promql")


def _build(spec: dict, ctx: dict) -> PromQLMetricsConnector:
    from attribution_blueprint.mca import _coerce_float, _coerce_int

    source: Source = ctx["source"]
    endpoint = spec.get("endpoint")
    if not endpoint:
        raise MCAError(f"metrics.{source.name.lower()}.promql requires endpoint")
    cluster = spec.get("cluster")
    if not cluster:
        raise MCAError(f"metrics.{source.name.lower()}.promql requires cluster")
    auth_spec = spec.get("auth")
    auth = _build_auth(auth_spec) if auth_spec is not None else None
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
        node_label=spec.get("node_label", DEFAULT_NODE_LABEL),
        cluster_label=spec.get("cluster_label", DEFAULT_CLUSTER_LABEL),
        source=source,
        templates=templates,
        timeout_seconds=timeout,
        step_seconds=step,
    )
    return PromQLMetricsConnector(**kwargs)


_registry.register_component(
    framework="metrics",
    name="promql",
    schema={
        "endpoint": _PARAM,
        "cluster": _PARAM,
        "node_label": _PARAM,
        "cluster_label": _PARAM,
        "timeout_seconds": _PARAM,
        "step_seconds": _PARAM,
        "auth": {
            "type": _PARAM,
            "cert_path": _PARAM,
            "key_path": _PARAM,
            "ca_cert_path": _PARAM,
            "token_env": _PARAM,
            "org_id": _PARAM,
        },
        "queries": {"_open": True},
    },
    build=_build,
)


__all__ = ["PromQLMetricsConnector", "TEMPLATES"]
