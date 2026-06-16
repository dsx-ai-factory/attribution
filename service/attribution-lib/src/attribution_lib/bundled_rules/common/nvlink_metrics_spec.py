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
NVLink metrics specification: threshold-based rules for NVLink Prometheus metric attribution.

Ported from attribution_lib/rules/nvlink_metrics_to_symptom.py.  Importing this
module registers the metric spec with the rules-engine framework.
"""

import operator

from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rules_engine import MetricSpec, register_metric_spec

# ---------------------------------------------------------------------------
# Symptom type definitions
# ---------------------------------------------------------------------------
FAILURE = "failure"
PERFORMANCE = "performance"
LINK_DOWN_NVLINK = "link down(nvlink)"

# ---------------------------------------------------------------------------
# Threshold-based rules for metric attribution
# ---------------------------------------------------------------------------
metric_thresholds = {
    "stackdriver_gce_instance_compute_googleapis_com_instance_gpu_nvswitch_link_downed_counter": {
        "threshold": 0,  # int: Any increase > 0 indicates link flap
        "operator": operator.gt,
        "symptom_type": LINK_DOWN_NVLINK,
        "attribution_type": FAILURE,
    },
    "phy_effective_errors": {
        # we don't know what is right threshold for this metric for now,
        # starts with 10000 and we should adjust it later if needed
        "threshold": 10000,
        "operator": operator.gt,
        "symptom_type": "performance_phy_effective_errors_high",
        "attribution_type": PERFORMANCE,
    },
    "effective_ber_magnitude": {
        # We want to know when effective_ber >=1e-7
        # so setting the threshold for the magnitude to -8.
        "threshold": -8,
        "operator": operator.gt,
        "symptom_type": "effective_ber_high",
        "attribution_type": FAILURE,
    },
    "plr_tx_bw_loss_1m_5pct": {
        "threshold": 0.05,
        "operator": operator.gt,
        "symptom_type": "plr_tx_bw_loss_high_1m_5pct",
        "attribution_type": FAILURE,
    },
    "plr_tx_bw_loss_5m_1pct": {
        "threshold": 0.01,
        "operator": operator.gt,
        "symptom_type": "plr_tx_bw_loss_high_5m_1pct",
        "attribution_type": FAILURE,
    },
    "plr_tx_bw_loss_10m_0p5pct": {
        "threshold": 0.005,
        "operator": operator.gt,
        "symptom_type": "plr_tx_bw_loss_high_10m_0p5pct",
        "attribution_type": FAILURE,
    },
    "plr_rx_bw_loss_1m_5pct": {
        "threshold": 0.05,
        "operator": operator.gt,
        "symptom_type": "plr_rx_bw_loss_high_1m_5pct",
        "attribution_type": PERFORMANCE,
    },
    "plr_rx_bw_loss_5m_1pct": {
        "threshold": 0.01,
        "operator": operator.gt,
        "symptom_type": "plr_rx_bw_loss_high_5m_1pct",
        "attribution_type": PERFORMANCE,
    },
    "plr_rx_bw_loss_10m_0p5pct": {
        "threshold": 0.005,
        "operator": operator.gt,
        "symptom_type": "plr_rx_bw_loss_high_10m_0p5pct",
        "attribution_type": PERFORMANCE,
    },
    "hist15": {
        "threshold": 0,
        "operator": operator.gt,
        "symptom_type": "hist15_high",
        "attribution_type": FAILURE,
    },
    "port_xmit_wait_lost_bw_gbps_1m_5pct": {
        "threshold": 20,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_1m_5pct",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_1m_1pct": {
        "threshold": 4,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_1m_1pct",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_1m_0p5pct": {
        "threshold": 2,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_1m_0p5pct",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_1m_10pct": {
        "threshold": 40,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_1m_10pct",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_1m_50pct": {
        "threshold": 200,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_1m_50pct",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_1m_gt0": {
        "threshold": 0,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_1m_gt0",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_5m_5pct": {
        "threshold": 20,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_5m_5pct",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_5m_1pct": {
        "threshold": 4,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_5m_1pct",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_5m_0p5pct": {
        "threshold": 2,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_5m_0p5pct",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_5m_10pct": {
        "threshold": 40,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_5m_10pct",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_5m_50pct": {
        "threshold": 200,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_5m_50pct",
        "attribution_type": PERFORMANCE,
    },
    "port_xmit_wait_lost_bw_gbps_5m_gt0": {
        "threshold": 0,
        "operator": operator.gt,
        "symptom_type": "performance_port_xmit_wait_bw_loss_host_5m_gt0",
        "attribution_type": PERFORMANCE,
    },
}

# ---------------------------------------------------------------------------
# Backward-compatible derived data
# ---------------------------------------------------------------------------
supported_nvlink_metrics = [(metric_name, rule) for metric_name, rule in metric_thresholds.items()]

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
register_metric_spec(
    ObservationSource.NVLINK_METRICS,
    MetricSpec(thresholds=metric_thresholds),
)
