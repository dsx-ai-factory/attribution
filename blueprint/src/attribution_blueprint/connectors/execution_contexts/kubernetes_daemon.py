# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility import for KubernetesDaemon connector configuration."""

from attribution_blueprint.kubernetes.daemon.config import (
    build_kubernetes_daemon_client,
)

__all__ = ["build_kubernetes_daemon_client"]
