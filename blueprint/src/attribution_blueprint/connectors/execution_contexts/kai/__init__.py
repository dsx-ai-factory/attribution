# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth
from attribution_blueprint.connectors.execution_contexts.kai.connector import (
    KaiExecutionContext,
    KaiExecutionContextAdapter,
)
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM


def _build_kubernetes_daemon(spec: dict):
    from attribution_blueprint.kubernetes.daemon.client import KubernetesDaemonClient
    from attribution_blueprint.mca import _coerce_float

    endpoint = spec.get("endpoint")
    if not endpoint:
        raise MCAError("eca.kai.placement_recorder requires endpoint")
    auth = None
    auth_spec = spec.get("auth")
    if auth_spec is not None:
        auth_type = auth_spec.get("type")
        if auth_type == "token":
            env_name = auth_spec.get("token_env")
            if not env_name:
                raise MCAError("eca.kai.placement_recorder.auth type=token requires token_env")
            if env_name not in os.environ:
                raise MCAError(f"auth.token_env references unset environment variable: {env_name}")
            auth = TokenAuth(token=os.environ[env_name], org_id=auth_spec.get("org_id"))
        elif auth_type == "mtls":
            if not auth_spec.get("cert_path") or not auth_spec.get("key_path"):
                raise MCAError("eca.kai.placement_recorder.auth type=mtls requires cert_path and key_path")
            auth = MTLSAuth(
                cert_path=auth_spec["cert_path"],
                key_path=auth_spec["key_path"],
                ca_cert_path=auth_spec.get("ca_cert_path"),
                org_id=auth_spec.get("org_id"),
            )
        else:
            raise MCAError(f"unknown KubernetesDaemon auth type: {auth_type!r}")
    timeout = _coerce_float(spec.get("timeout_seconds", 10.0), "eca.kai.placement_recorder.timeout_seconds")
    try:
        return KubernetesDaemonClient(endpoint, auth=auth, timeout_seconds=timeout)
    except ValueError as error:
        raise MCAError(f"invalid eca.kai.placement_recorder configuration: {error}") from error


def _build(spec: dict, ctx: dict) -> KaiExecutionContextAdapter:
    from attribution_blueprint.mca import _coerce_int

    # A literal `null` arrives as None and must fall back to the default;
    # otherwise None would reach the Kubernetes client and silently disable
    # the request timeout.
    timeout = spec.get("timeout_seconds")
    if timeout is None:
        timeout = 60
    else:
        timeout = _coerce_int(timeout, "eca.kai.timeout_seconds")
        if timeout <= 0:
            raise MCAError(f"eca.kai.timeout_seconds must be positive, got {timeout}")
    placement_records_enabled = spec.get("placement_records_enabled", False)
    if not isinstance(placement_records_enabled, bool):
        raise MCAError("eca.kai.placement_records_enabled must be a bool")
    recorder_spec = spec.get("placement_recorder")
    if placement_records_enabled and recorder_spec is None:
        raise MCAError("eca.kai.placement_records_enabled requires eca.kai.placement_recorder")
    if placement_records_enabled and not spec.get("namespace"):
        raise MCAError("eca.kai.placement_records_enabled requires eca.kai.namespace")
    return KaiExecutionContextAdapter(
        kubeconfig=spec.get("kubeconfig"),
        context=spec.get("context"),
        namespace=spec.get("namespace"),
        timeout_seconds=timeout,
        placement_records_enabled=placement_records_enabled,
        placement_reader=_build_kubernetes_daemon(recorder_spec) if placement_records_enabled else None,
    )


_registry.register_component(
    framework="eca",
    name="kai",
    schema={
        "ids": _PARAM,
        "start_time": _PARAM,
        "end_time": _PARAM,
        "nodelist": _PARAM,
        # Optional path to a kubeconfig file. When unset, the client falls back
        # to its default discovery (KUBECONFIG, ~/.kube/config, or the
        # in-cluster service account).
        "kubeconfig": _PARAM,
        # Optional context name within the resolved kubeconfig.
        "context": _PARAM,
        # Optional namespace scope. The empty string (or unset) queries every
        # namespace.
        "namespace": _PARAM,
        "timeout_seconds": _PARAM,
        "placement_records_enabled": _PARAM,
        "placement_recorder": {
            "endpoint": _PARAM,
            "timeout_seconds": _PARAM,
            "auth": {
                "type": _PARAM,
                "cert_path": _PARAM,
                "key_path": _PARAM,
                "ca_cert_path": _PARAM,
                "token_env": _PARAM,
                "org_id": _PARAM,
            },
        },
    },
    build=_build,
)


__all__ = ["KaiExecutionContext", "KaiExecutionContextAdapter"]
