# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.execution_contexts.kueue.connector import (
    KueueExecutionContext,
    KueueExecutionContextAdapter,
)
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM


def _build(spec: dict, ctx: dict) -> KueueExecutionContextAdapter:
    from attribution_blueprint.mca import _coerce_int

    # `spec.get(..., 60)` returns 60 only when the key is absent; a TOML
    # literal `null` arrives as Python None and must fall back to the same
    # default. Otherwise None would propagate to `subprocess.run(timeout=None)`,
    # silently disabling the timeout entirely.
    timeout = spec.get("timeout_seconds")
    if timeout is None:
        timeout = 60
    else:
        timeout = _coerce_int(timeout, "eca.kueue.timeout_seconds")
        if timeout <= 0:
            raise MCAError(f"eca.kueue.timeout_seconds must be positive, got {timeout}")
    return KueueExecutionContextAdapter(
        kubectl_path=spec.get("kubectl_path", "kubectl"),
        kubeconfig=spec.get("kubeconfig"),
        context=spec.get("context"),
        namespace=spec.get("namespace"),
        timeout_seconds=timeout,
    )


_registry.register_component(
    framework="eca",
    name="kueue",
    schema={
        "ids": _PARAM,
        "start_time": _PARAM,
        "end_time": _PARAM,
        "nodelist": _PARAM,
        "kubectl_path": _PARAM,
        # Optional path to a kubeconfig file. When unset, kubectl falls back
        # to its default discovery (KUBECONFIG env var, ~/.kube/config, or an
        # in-cluster service-account token when running inside a pod).
        "kubeconfig": _PARAM,
        # Optional context name within the resolved kubeconfig. When unset,
        # kubectl uses the kubeconfig's current-context.
        "context": _PARAM,
        # Optional namespace scope. The empty string (or unset) queries every
        # namespace via `-A`. Set this to limit blast radius on multi-tenant
        # clusters or to allow bare `<name>` --ids values.
        "namespace": _PARAM,
        "timeout_seconds": _PARAM,
    },
    build=_build,
)


__all__ = ["KueueExecutionContext", "KueueExecutionContextAdapter"]
