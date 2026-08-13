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
Node metrics rules ported from attribution_lib/node_metrics.py check().

Threshold-based attribution on Prometheus metrics.  Each rule matches the
metric name inside the ``MetricEntry.body()`` text and captures the numeric
value for threshold comparison in the processor.
"""

import common.node_metrics_spec  # noqa: F401 — triggers metric spec registration

from attribution_lib.rules_engine import declare_event_types

rule_author = "NVIDIA Attribution team"

declare_event_types("Attribution.node.bad", "Attribution.node.interesting")
