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
NVLink metrics rules ported from attribution_lib/nvlink_metrics.py check().

Threshold-based attribution on NVLink Prometheus metrics.
"""

from typing import Dict, List

import common.nvlink_metrics_spec  # noqa: F401 — triggers metric spec registration

from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rule_helpers import _bad, _interesting
from attribution_lib.rules_engine import EventTriple, RuleContext, declare_event_types, extractor_def
from attribution_lib.sources import Source

rule_author = "FACT Team"
rule_sources = {ObsSource.NVLINK_METRICS}

declare_event_types("Attribution.node.bad", "Attribution.node.interesting")


def _threshold(metric_name: str) -> float:
    return float(common.nvlink_metrics_spec.metric_thresholds[metric_name]["threshold"])


@extractor_def(
    name="NvlinkMetrics_nvswitch_link_downed",
    match_re=(
        r"metric_name: stackdriver_gce_instance_compute_googleapis_com"
        r"_instance_gpu_nvswitch_link_downed_counter, metric_value: (?P<value>[^,]+)"
    ),
    derived_rules=["nvlink_stackdriver_gce_instance_compute_googleapis_com_instance_gpu_nvswitch_link_downed_counter"],
)
def nvlink_nvswitch_link_downed(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        metric = "stackdriver_gce_instance_compute_googleapis_com_instance_gpu_nvswitch_link_downed_counter"
        return [_bad(node, "link down(nvlink)", source.derived(rule="nvlink_" + metric))]
    return []


@extractor_def(
    name="NvlinkMetrics_phy_effective_errors",
    match_re=r"metric_name: phy_effective_errors, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_phy_effective_errors"],
)
def nvlink_phy_effective_errors(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 10000:
        return [
            _interesting(
                node, "performance_phy_effective_errors_high", source.derived(rule="nvlink_phy_effective_errors")
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_effective_ber_magnitude",
    match_re=r"metric_name: effective_ber_magnitude, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_effective_ber_magnitude"],
)
def nvlink_effective_ber_magnitude(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > -8:
        return [_bad(node, "effective_ber_high", source.derived(rule="nvlink_effective_ber_magnitude"))]
    return []


@extractor_def(
    name="NvlinkMetrics_plr_tx_bw_loss_1m_5pct",
    match_re=r"metric_name: plr_tx_bw_loss_1m_5pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_plr_tx_bw_loss_1m_5pct"],
)
def nvlink_plr_tx_bw_loss_1m_5pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > _threshold("plr_tx_bw_loss_1m_5pct"):
        return [_bad(node, "plr_tx_bw_loss_high_1m_5pct", source.derived(rule="nvlink_plr_tx_bw_loss_1m_5pct"))]
    return []


@extractor_def(
    name="NvlinkMetrics_plr_tx_bw_loss_5m_1pct",
    match_re=r"metric_name: plr_tx_bw_loss_5m_1pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_plr_tx_bw_loss_5m_1pct"],
)
def nvlink_plr_tx_bw_loss_5m_1pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > _threshold("plr_tx_bw_loss_5m_1pct"):
        return [_bad(node, "plr_tx_bw_loss_high_5m_1pct", source.derived(rule="nvlink_plr_tx_bw_loss_5m_1pct"))]
    return []


@extractor_def(
    name="NvlinkMetrics_plr_tx_bw_loss_10m_0p5pct",
    match_re=r"metric_name: plr_tx_bw_loss_10m_0p5pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_plr_tx_bw_loss_10m_0p5pct"],
)
def nvlink_plr_tx_bw_loss_10m_0p5pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > _threshold("plr_tx_bw_loss_10m_0p5pct"):
        return [_bad(node, "plr_tx_bw_loss_high_10m_0p5pct", source.derived(rule="nvlink_plr_tx_bw_loss_10m_0p5pct"))]
    return []


@extractor_def(
    name="NvlinkMetrics_plr_rx_bw_loss_1m_5pct",
    match_re=r"metric_name: plr_rx_bw_loss_1m_5pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_plr_rx_bw_loss_1m_5pct"],
)
def nvlink_plr_rx_bw_loss_1m_5pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > _threshold("plr_rx_bw_loss_1m_5pct"):
        return [
            _interesting(
                node,
                "plr_rx_bw_loss_high_1m_5pct",
                source.derived(rule="nvlink_plr_rx_bw_loss_1m_5pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_plr_rx_bw_loss_5m_1pct",
    match_re=r"metric_name: plr_rx_bw_loss_5m_1pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_plr_rx_bw_loss_5m_1pct"],
)
def nvlink_plr_rx_bw_loss_5m_1pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > _threshold("plr_rx_bw_loss_5m_1pct"):
        return [
            _interesting(
                node,
                "plr_rx_bw_loss_high_5m_1pct",
                source.derived(rule="nvlink_plr_rx_bw_loss_5m_1pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_plr_rx_bw_loss_10m_0p5pct",
    match_re=r"metric_name: plr_rx_bw_loss_10m_0p5pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_plr_rx_bw_loss_10m_0p5pct"],
)
def nvlink_plr_rx_bw_loss_10m_0p5pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > _threshold("plr_rx_bw_loss_10m_0p5pct"):
        return [
            _interesting(
                node,
                "plr_rx_bw_loss_high_10m_0p5pct",
                source.derived(rule="nvlink_plr_rx_bw_loss_10m_0p5pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_1m_5pct",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_1m_5pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_1m_5pct"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_1m_5pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_1m_5pct",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_1m_5pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_1m_1pct",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_1m_1pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_1m_1pct"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_1m_1pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_1m_1pct",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_1m_1pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_1m_0p5pct",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_1m_0p5pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_1m_0p5pct"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_1m_0p5pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_1m_0p5pct",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_1m_0p5pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_1m_10pct",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_1m_10pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_1m_10pct"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_1m_10pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_1m_10pct",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_1m_10pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_1m_50pct",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_1m_50pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_1m_50pct"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_1m_50pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_1m_50pct",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_1m_50pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_1m_gt0",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_1m_gt0, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_1m_gt0"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_1m_gt0(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_1m_gt0",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_1m_gt0"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_5m_5pct",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_5m_5pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_5m_5pct"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_5m_5pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_5m_5pct",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_5m_5pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_5m_1pct",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_5m_1pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_5m_1pct"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_5m_1pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_5m_1pct",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_5m_1pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_5m_0p5pct",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_5m_0p5pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_5m_0p5pct"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_5m_0p5pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_5m_0p5pct",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_5m_0p5pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_5m_10pct",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_5m_10pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_5m_10pct"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_5m_10pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_5m_10pct",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_5m_10pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_5m_50pct",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_5m_50pct, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_5m_50pct"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_5m_50pct(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_5m_50pct",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_5m_50pct"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_port_xmit_wait_lost_bw_gbps_5m_gt0",
    match_re=r"metric_name: port_xmit_wait_lost_bw_gbps_5m_gt0, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_port_xmit_wait_lost_bw_gbps_5m_gt0"],
)
def nvlink_port_xmit_wait_lost_bw_gbps_5m_gt0(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                "performance_port_xmit_wait_bw_loss_host_5m_gt0",
                source.derived(rule="nvlink_port_xmit_wait_lost_bw_gbps_5m_gt0"),
            )
        ]
    return []


@extractor_def(
    name="NvlinkMetrics_hist15",
    match_re=r"metric_name: hist15, metric_value: (?P<value>[^,]+)",
    derived_rules=["nvlink_hist15"],
)
def nvlink_hist15(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        return [_bad(node, "hist15_high", source.derived(rule="nvlink_hist15"))]
    return []
