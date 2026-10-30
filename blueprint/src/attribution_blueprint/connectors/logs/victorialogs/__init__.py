# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.auth import Auth, build_auth
from attribution_blueprint.connectors.logs.victorialogs.connector import (
    DEFAULT_LIMIT,
    DEFAULT_NODE_FIELD,
    VictoriaLogsConnector,
)
from attribution_blueprint.connectors.logs.victorialogs.templates import BARE_TEMPLATES, TEMPLATES
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM
from attribution_blueprint.types import Source


def _build(spec: dict, ctx: dict) -> VictoriaLogsConnector:
    from attribution_blueprint.mca import _coerce_float, _coerce_int

    source: Source = ctx["source"]
    endpoint = spec.get("endpoint")
    if not endpoint:
        raise MCAError(f"log.{source.name.lower()}.victorialogs requires endpoint")
    auth_spec = spec.get("auth")
    auth: Auth | None = build_auth(auth_spec, "victorialogs") if auth_spec is not None else None
    timeout = _coerce_float(spec.get("timeout_seconds", 60.0), "timeout_seconds")
    limit = _coerce_int(spec.get("limit", DEFAULT_LIMIT), "limit")
    if limit <= 0:
        raise MCAError("limit must be greater than 0")
    queries = spec.get("queries")
    if queries is None and "query" in spec:
        queries = [spec["query"]]
    elif isinstance(queries, str):
        queries = [queries]
    node_field = spec.get("node_field", DEFAULT_NODE_FIELD)
    if node_field == "":
        node_field = None
    kwargs: dict[str, Any] = dict(
        endpoint=endpoint,
        auth=auth,
        source=source,
        query_kwargs=dict(spec.get("query_kwargs", {})),
        timeout_seconds=timeout,
        limit=limit,
        node_field=node_field,
    )
    if queries is not None:
        kwargs["templates"] = {source: list(queries)}
    return VictoriaLogsConnector(**kwargs)


_registry.register_component(
    framework="log",
    name="victorialogs",
    schema={
        "endpoint": _PARAM,
        "timeout_seconds": _PARAM,
        "limit": _PARAM,
        "query": _PARAM,
        "queries": _PARAM,
        "node_field": _PARAM,
        "auth": {
            "type": _PARAM,
            "cert_path": _PARAM,
            "key_path": _PARAM,
            "ca_cert_path": _PARAM,
            "token_env": _PARAM,
        },
        "query_kwargs": {"_open": True},
    },
    build=_build,
)


__all__ = ["VictoriaLogsConnector", "TEMPLATES", "BARE_TEMPLATES"]
