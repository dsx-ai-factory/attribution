# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from typing import Any

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.auth import Auth, MTLSAuth, TokenAuth
from attribution_blueprint.connectors.metrics.promql.connector import (
    PromQLMetricsConnector,
)
from attribution_blueprint.connectors.metrics.promql.templates import TEMPLATES
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM
from attribution_blueprint.types import Source


def _build_auth(spec: dict) -> Auth:
    atype = spec.get("type")
    if atype == "token":
        env_name = spec.get("token_env")
        if not env_name:
            raise MCAError("promql.auth type=token requires token_env")
        if env_name not in os.environ:
            # Surface as MCAError so the CLI's parser.error path produces a
            # clean usage message and exit code 2, rather than a Python
            # traceback from an uncaught KeyError. Mirrors loki/_build_auth.
            raise MCAError(f"auth.token_env references unset environment variable: {env_name}")
        return TokenAuth(token=os.environ[env_name], org_id=spec.get("org_id"))
    if atype == "mtls":
        if not spec.get("cert_path") or not spec.get("key_path"):
            raise MCAError("promql.auth type=mtls requires cert_path and key_path")
        return MTLSAuth(
            cert_path=spec["cert_path"],
            key_path=spec["key_path"],
            ca_cert_path=spec.get("ca_cert_path"),
            org_id=spec.get("org_id"),
        )
    raise MCAError(f"unknown auth type: {atype!r}")


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
