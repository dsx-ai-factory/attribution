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

from typing import Dict, List

import common.node_metrics_spec  # noqa: F401 — triggers metric spec registration

from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rule_helpers import _bad, _interesting
from attribution_lib.rules_engine import EventTriple, RuleContext, declare_event_types, extractor_def
from attribution_lib.sources import Source
from attribution_lib.symptoms.symptom_enums import RegisteredSymptoms
from attribution_lib.utils import is_event_at_end_of_job

rule_author = "FACT Team"
rule_sources = {ObsSource.NODE_METRICS}

declare_event_types("Attribution.node.bad", "Attribution.node.interesting")

LINK_DOWN_EAST_WEST = "link down(east-west)"
LINK_DOWN_NVLINK_PHYSICAL_LAYER = RegisteredSymptoms.LINK_DOWN_NVLINK_PHYSICAL_LAYER.value


@extractor_def(
    name="NodeMetrics_infiniband_link_downed",
    match_re=r"metric_name: node_infiniband_link_downed_total, metric_value: (?P<value>[^,]+)",
    derived_rules=["node-metric-infiniband-link-downed"],
)
def nm_infiniband_link_downed(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        if is_event_at_end_of_job(source.time, ctx.job_info.job_end_time):
            return [_bad(node, LINK_DOWN_EAST_WEST, source.derived(rule="node-metric-infiniband-link-downed"))]
    return []


@extractor_def(
    name="NodeMetrics_carrier_down_changes",
    match_re=r"metric_name: node_network_carrier_down_changes_total, metric_value: (?P<value>[^,]+)",
    derived_rules=["node-metric-carrier-down"],
)
def nm_carrier_down_changes(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    if float(groups["value"]) > 0:
        if is_event_at_end_of_job(source.time, ctx.job_info.job_end_time):
            return [_bad(node, LINK_DOWN_EAST_WEST, source.derived(rule="node-metric-carrier-down"))]
    return []


@extractor_def(
    name="NodeMetrics_roce_adp_retrans",
    match_re=r"metric_name: ib_roce_adp_retrans, metric_value: (?P<value>[^,]+)",
    derived_rules=["node-metric-roce-adp-retrans"],
)
def nm_roce_adp_retrans(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    if float(groups["value"]) > 50:
        return [
            _interesting(node, "performance_roce_adp_retrans_high", source.derived(rule="node-metric-roce-adp-retrans"))
        ]
    return []


@extractor_def(
    name="NodeMetrics_prometheus_up",
    match_re=r"metric_name: up, metric_value: (?P<value>[^,]+)",
    derived_rules=["node-metric-prometheus-down"],
)
def nm_prometheus_up(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    if float(groups["value"]) == 0:
        return [_bad(node, "prometheus lost connection", source.derived(rule="node-metric-prometheus-down"))]
    return []


@extractor_def(
    name="NodeMetrics_dcgm_nvlink_physical_link_down",
    match_re=r"metric_name: DCGM_FI_DEV_NVLINK_PPCNT_PHYSICAL_LINK_DOWN_COUNTER, metric_value: (?P<value>[^,]+)",
    derived_rules=["node-metric-dcgm-nvlink-physical-link-down"],
)
def nm_dcgm_nvlink_physical_link_down(
    groups: Dict[str, str], node: str, source: Source, ctx: RuleContext
) -> List[EventTriple]:
    """Emit _interesting when the DCGM NVLink physical-layer down counter is non-zero.

    Counter delta > 0 means the NVLink physical layer dropped at least once
    during the scrape window. Co-occurs with XID-149.x; emit as _interesting
    (signal-only) to surface the event without forcing a hard verdict on
    transient ticks. Unlike the sibling east/west link-down rules, this
    intentionally omits the is_event_at_end_of_job guard: those rules emit
    _bad and need the guard to avoid false verdicts from mid-job flaps,
    whereas an _interesting event is safe to emit any time during the job.
    """
    if float(groups["value"]) > 0:
        return [
            _interesting(
                node,
                LINK_DOWN_NVLINK_PHYSICAL_LAYER,
                source.derived(rule="node-metric-dcgm-nvlink-physical-link-down"),
            )
        ]
    return []
