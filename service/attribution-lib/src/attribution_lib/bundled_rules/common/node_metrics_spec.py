# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Node metrics specification: threshold-based rules for Prometheus metric attribution.

Ported from attribution_lib/rules/node_metrics_to_symptom.py.
"""

import operator

from attribution_lib.symptoms.symptom_enums import RegisteredSymptoms

# ---------------------------------------------------------------------------
# Symptom type definitions
# ---------------------------------------------------------------------------
LINK_DOWN_EAST_WEST = "link down(east-west)"
LINK_DOWN_NVLINK_PHYSICAL_LAYER = RegisteredSymptoms.LINK_DOWN_NVLINK_PHYSICAL_LAYER.value
PROMETHEUS_LOST_CONNECTION = "prometheus lost connection"
FAILURE = "failure"
PERFORMANCE = "performance"

# ---------------------------------------------------------------------------
# Threshold-based rules for metric attribution
# ---------------------------------------------------------------------------
# {
# metric_name: {
# threshold: int/float,
# operator: callable,  # operator.gt, operator.lt, operator.ge, operator.le, operator.eq, operator.ne  (default: operator.gt)
# symptom_type: str,   # e.g. LINK_DOWN_EAST_WEST
# attribution_type: str,
# }
metric_thresholds = {
    "node_infiniband_link_downed_total": {
        "threshold": 0,  # int: Any increase > 0 indicates link flap
        "operator": operator.gt,
        "symptom_type": LINK_DOWN_EAST_WEST,
        "attribution_type": FAILURE,
    },
    "node_network_carrier_down_changes_total": {
        "threshold": 0,  # int: Any increase > 0 indicates link flap
        "operator": operator.gt,
        "symptom_type": LINK_DOWN_EAST_WEST,
        "attribution_type": FAILURE,
    },
    "DCGM_FI_DEV_NVLINK_PPCNT_PHYSICAL_LINK_DOWN_COUNTER": {
        "threshold": 0,  # int: any non-zero delta indicates an NVLink physical-layer down event
        "operator": operator.gt,
        "symptom_type": LINK_DOWN_NVLINK_PHYSICAL_LAYER,
        "attribution_type": FAILURE,
    },
    "ib_roce_adp_retrans": {
        # we don't know what is right threshold for this metric for now,
        # starts with 50 and we should adjust it later if needed
        "threshold": 50,
        "operator": operator.gt,
        "symptom_type": "performance_roce_adp_retrans_high",
        "attribution_type": PERFORMANCE,
    },
    "up": {
        "threshold": 0,  # if the value is 0, it means the network connection is lost
        "operator": operator.eq,
        "symptom_type": PROMETHEUS_LOST_CONNECTION,
        "attribution_type": FAILURE,
    },
}

# ---------------------------------------------------------------------------
# Backward-compatible derived data
# ---------------------------------------------------------------------------
supported_node_metrics = [metric for metric in metric_thresholds.keys()]
node_metrics_thresholds = {metric: rule["threshold"] for metric, rule in metric_thresholds.items()}
