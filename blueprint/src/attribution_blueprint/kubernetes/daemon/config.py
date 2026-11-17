# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from attribution_blueprint.connectors.auth import build_auth
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

    auth_spec = spec.get("auth")
    auth = build_auth(auth_spec, config_path) if auth_spec is not None else None

    timeout_value = spec.get("timeout_seconds", 10.0)
    try:
        timeout = float(timeout_value)
    except (TypeError, ValueError) as error:
        raise MCAError(f"{config_path}.timeout_seconds={timeout_value!r} must be a float") from error
    try:
        return KubernetesDaemonClient(endpoint, auth=auth, timeout_seconds=timeout)
    except ValueError as error:
        raise MCAError(f"invalid {config_path} configuration: {error}") from error
