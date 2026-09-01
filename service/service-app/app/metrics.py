# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prometheus metrics for the NVIDIA Attribution Service."""

from prometheus_client import Gauge

VERSION_INFO = Gauge(
    "version_info", "Version information for the attribution service", ["version"], "nvidia_attribution_service"
)
