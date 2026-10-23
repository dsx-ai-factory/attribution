# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Node metrics rules ported from attribution_lib/node_metrics.py check().

Threshold-based attribution on Prometheus metrics.  Each rule matches the
metric name inside the ``render_metric_body`` text and captures the numeric
value for threshold comparison in the processor.
"""

import common.node_metrics_spec

from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rules import MetricSpec, declare_event_types, register_metric_spec

rule_author = "NVIDIA Attribution team"

register_metric_spec(
    ObservationSource.NODE_METRICS,
    MetricSpec(thresholds=common.node_metrics_spec.metric_thresholds),
)
declare_event_types("Attribution.node.bad", "Attribution.node.interesting")
