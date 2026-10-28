# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility import for Kueue resource resolution."""

from attribution_blueprint.connectors.execution_contexts.kueue.resource_resolver import (
    KueueResourceResolver,
)

__all__ = ["KueueResourceResolver"]
