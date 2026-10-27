# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.execution_contexts.kai.connector import (
    KaiExecutionContext,
    KaiExecutionContextAdapter,
)
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM


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
    return KaiExecutionContextAdapter(
        kubeconfig=spec.get("kubeconfig"),
        context=spec.get("context"),
        namespace=spec.get("namespace"),
        timeout_seconds=timeout,
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
        # Namespace scope. Required for time-range discovery and for bare
        # `<name>` IDs. Without it, ID discovery accepts only
        # `<namespace>/<name>`. The adapter never scans the whole cluster.
        "namespace": _PARAM,
        "timeout_seconds": _PARAM,
    },
    build=_build,
)


__all__ = ["KaiExecutionContext", "KaiExecutionContextAdapter"]
