# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for scheduler-neutral Kubernetes job helpers."""

from attribution_blueprint.connectors.execution_contexts.kubernetes.helpers import (
    earliest_job_failure_reason,
    job_time_limit_minutes,
)

__all__ = ["earliest_job_failure_reason", "job_time_limit_minutes"]
