# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.auth import Auth, build_auth
from attribution_blueprint.connectors.logs.loki.connector import LokiConnector
from attribution_blueprint.connectors.logs.loki.templates import BARE_TEMPLATES, TEMPLATES
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM
from attribution_blueprint.types import Source


def _build_auth(spec: dict) -> Auth:
    return build_auth(spec, "loki")


def _build(spec: dict, ctx: dict) -> LokiConnector:
    from attribution_blueprint.mca import _coerce_float

    source: Source = ctx["source"]
    endpoint = spec.get("endpoint")
    if not endpoint:
        raise MCAError(f"log.{source.name.lower()}.loki requires endpoint")
    auth_spec = spec.get("auth")
    if not auth_spec:
        raise MCAError(f"log.{source.name.lower()}.loki requires auth")
    auth = _build_auth(auth_spec)
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
    # populate the per-entry `node` field and to splice the per-ExecutionContext node
    # filter into queries. Defaults to "host_name" (the bundled set's
    # convention); customers whose streams use "hostname" / "instance" /
    # "kubernetes_pod_node_name" / etc. override here.
    if "node_label" in spec:
        kwargs["node_label"] = spec["node_label"]
    return LokiConnector(**kwargs)


_registry.register_component(
    framework="log",
    name="loki",
    schema={
        "endpoint": _PARAM,
        "timeout_seconds": _PARAM,
        "query": _PARAM,
        "queries": _PARAM,
        "node_label": _PARAM,
        "auth": {
            "type": _PARAM,
            "cert_path": _PARAM,
            "key_path": _PARAM,
            "ca_cert_path": _PARAM,
            "token_env": _PARAM,
            "org_id": _PARAM,
        },
        "query_kwargs": {"_open": True},
    },
    build=_build,
)


__all__ = ["LokiConnector", "TEMPLATES", "BARE_TEMPLATES"]
