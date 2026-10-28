# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility import for scheduler-neutral Kubernetes client code."""

from attribution_blueprint.connectors.execution_contexts.kubernetes.client import (
    KubernetesClient,
    KubernetesClientError,
)

__all__ = ["KubernetesClient", "KubernetesClientError"]
