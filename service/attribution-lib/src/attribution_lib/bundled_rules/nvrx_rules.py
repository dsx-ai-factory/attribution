# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""External NVRX rules and collector integration."""

from attribution_lib.bundled_rules.common.nvrx import (
    NVRXStragglerParams,
    StragglerParseState,
    filter_nvrx_stragglers,
    resolve_reported_node,
)
from attribution_lib.rule_helpers import _bad, _interesting
from attribution_lib.rules import (
    EventTriple,
    RuleContext,
    collector_def,
    declare_event_types,
    rule_processor,
)
from attribution_lib.sources import Source

rule_author = "NVIDIA Attribution team"

declare_event_types("Attribution.node.bad", "Attribution.node.interesting")

# NVRX extractors backed by these processors are declared in
# nvrx_leaf_rules.toml and nvrx_processor_rules.toml.


@rule_processor("nvrx:NVRXGpuMemReclaimTimeout")
def nvrx_gpu_mem_reclaim_timeout(
    groups: dict[str, str], node: str, source: Source, ctx: RuleContext
) -> list[EventTriple]:
    """Surface the NVRx GPU-memory reclamation timeout as interesting evidence."""
    attributed_node = resolve_reported_node(groups, node, ctx)
    return [_interesting(attributed_node, "NVRx GPU memory reclamation timeout", source)]


@rule_processor("nvrx:NVRXFailureDetected")
def nvrx_failure_detected(groups: dict[str, str], node: str, source: Source, ctx: RuleContext) -> list[EventTriple]:
    """Surface an NVRx cycle failure as a non-node-attributable symptom.

    Args:
        groups: Regex captures from the failure event.
        node: Source node supplied by the observation reader.
        source: Source attribution for the matching log line.
        ctx: Rule context with optional job metadata.

    Returns:
        A temporary node attribution that is aggregated into job symptoms and
        removed from final bad-node reporting.
    """
    return [_bad(node, "NVRx detected failure", source)]


@rule_processor("nvrx:NVRXUnhealthyNode")
def nvrx_unhealthy_node(groups: dict[str, str], node: str, source: Source, ctx: RuleContext) -> list[EventTriple]:
    """Surface the node directly named by an organic NVRx health-check failure.

    Args:
        groups: Regex captures containing the unhealthy node.
        node: Source node supplied by the observation reader.
        source: Source attribution for the matching log line.
        ctx: Rule context with optional job metadata.

    Returns:
        A bad-node attribution for the unhealthy short hostname.
    """
    attributed_node = resolve_reported_node(groups, node, ctx)
    return [_bad(attributed_node, "NVRx unhealthy node", source)]


@collector_def(
    name="Collect.NVRXStraggler",
    formatter=lambda raw: {"nvrx_straggler": raw},
    params_model=NVRXStragglerParams,
)
def _collect_nvrx_straggler(event_stream, _job_info, params):
    groups = StragglerParseState.collect(event_stream)
    return filter_nvrx_stragglers(groups, params)
