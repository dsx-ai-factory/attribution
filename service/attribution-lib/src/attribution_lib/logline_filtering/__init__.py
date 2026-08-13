# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Log line filtering utilities for consistent logging of ignored log lines.

This package provides centralized logging utilities for when log lines are
dropped during processing for various reasons (node filtering, time boundaries,
content patterns, etc.).

Key functions:
- log_ignored_node_line: Log lines from nodes not in job's node list (with built-in error handling)
- log_ignored_time_line: Log lines outside job time boundaries (with built-in error handling)
- log_ignored_content_line: Log lines matching exclusion patterns (with built-in error handling)

Future extensions:
- Format error logging
- Size limit logging
- Rate limiting logging
- Business rule logging
"""

from .logger import (
    log_ignored_content_line,
    log_ignored_node_line,
    log_ignored_time_line,
)

__all__ = [
    "log_ignored_node_line",
    "log_ignored_time_line",
    "log_ignored_content_line",
]
