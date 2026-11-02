# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from attribution_blueprint.connectors.auth import MTLSAuth, TokenAuth
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.kubernetes.daemon.client import KubernetesDaemonClient


def build_kubernetes_daemon_client(
    spec: dict,
    *,
    config_path: str,
) -> KubernetesDaemonClient:
    """Build the shared daemon client from one adapter's MCA configuration."""
    endpoint = spec.get("endpoint")
    if not endpoint:
        raise MCAError(f"{config_path} requires endpoint")

    auth = None
    auth_spec = spec.get("auth")
    if auth_spec is not None:
        auth_type = auth_spec.get("type")
        auth_path = f"{config_path}.auth"
        if auth_type == "token":
            env_name = auth_spec.get("token_env")
            if not env_name:
                raise MCAError(f"{auth_path} type=token requires token_env")
            if env_name not in os.environ:
                raise MCAError(f"auth.token_env references unset environment variable: {env_name}")
            auth = TokenAuth(token=os.environ[env_name], org_id=auth_spec.get("org_id"))
        elif auth_type == "mtls":
            if not auth_spec.get("cert_path") or not auth_spec.get("key_path"):
                raise MCAError(f"{auth_path} type=mtls requires cert_path and key_path")
            auth = MTLSAuth(
                cert_path=auth_spec["cert_path"],
                key_path=auth_spec["key_path"],
                ca_cert_path=auth_spec.get("ca_cert_path"),
                org_id=auth_spec.get("org_id"),
            )
        else:
            raise MCAError(f"unknown KubernetesDaemon auth type: {auth_type!r}")

    timeout_value = spec.get("timeout_seconds", 10.0)
    try:
        timeout = float(timeout_value)
    except (TypeError, ValueError) as error:
        raise MCAError(f"{config_path}.timeout_seconds={timeout_value!r} must be a float") from error
    try:
        return KubernetesDaemonClient(endpoint, auth=auth, timeout_seconds=timeout)
    except ValueError as error:
        raise MCAError(f"invalid {config_path} configuration: {error}") from error
