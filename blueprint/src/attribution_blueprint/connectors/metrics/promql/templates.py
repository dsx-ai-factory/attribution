# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.types import Source

# Bundled PromQL templates per Source. Each entry is a complete query with the
# threshold and comparison operator embedded by the template author; the
# connector substitutes only {name}, {cluster}, and {nodes} at render time.
# {nodes} resolves to a "|".join of the ExecutionContext's hostnames, suitable for use as
# the right-hand side of a Prometheus =~ regex selector.
#
# PromQL uses {} for label selectors, which collides with str.format() field
# syntax — literal { and } must be doubled in templates. Substitution
# placeholders stay as single braces.
#
# Drift caveat: the thresholds here exist in two places conceptually.
# Service-side attribution rules also carry per-metric thresholds. If a
# Service-side threshold changes and these templates are not updated, the
# connector will filter at the old threshold and the Service will see fewer
# (or more) data points than expected. Keep template thresholds aligned with
# the Service-side rule set.
# Range windows are 10m rather than 4m for resilience to scrape gaps: at a
# 60s evaluation step a [4m] window covers only ~4 scrape points and a single
# missed scrape causes changes() to see no delta. [10m] gives 2-3 scrapes of
# headroom against gaps without dramatically broadening time resolution.
TEMPLATES: dict[Source, dict[str, str]] = {
    Source.NODE_METRICS: {
        "node_infiniband_link_downed_total": (
            'changes({name}{{s_cluster="{cluster}", net_host_name=~"{nodes}"}}[10m]) > 0'
        ),
        # attribution_lib node_metrics_rules treats this as the second
        # LINK_DOWN_EAST_WEST signal alongside node_infiniband_link_downed_total;
        # without this template the carrier-down symptom path never fires.
        "node_network_carrier_down_changes_total": (
            'changes({name}{{s_cluster="{cluster}", net_host_name=~"{nodes}"}}[10m]) > 0'
        ),
        # increase() is correct for counters; delta() returns wrong values when
        # scrape intervals are uneven (Prometheus docs warn against delta on
        # counters).
        "ib_roce_adp_retrans": ('increase({name}{{s_cluster="{cluster}", net_host_name=~"{nodes}"}}[10m]) > 50'),
        "up": ('max by (net_host_name) ({name}{{s_cluster="{cluster}", net_host_name=~"{nodes}"}}) == 0'),
    },
}
