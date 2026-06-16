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
CSP events rules.

Matches symptom strings produced from cloud service provider event data
and emits the corresponding attribution events. Currently handles host
termination events; extensible to other CSP-provided signals.
"""

from typing import Dict, List

from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rule_helpers import _bad, _interesting
from attribution_lib.rules.csp_events_to_symptom import INTERESTING_CATEGORIES, get_symptom as csp_get_symptom
from attribution_lib.rules_engine import EventTriple, RuleContext, declare_event_types, extractor_def
from attribution_lib.sources import Source

rule_author = "FACT Team"
rule_sources = {ObsSource.CSP_EVENT}

declare_event_types("Attribution.node.bad", "Attribution.node.interesting")


@extractor_def(
    name="CspEvent",
    match_re=r"csp_event: category=(?P<category>\w*), reason=(?P<reason>\w*), status=(?P<status>\w*)",
    derived_rules=["csp-event"],
)
def csp_event(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    symptom = csp_get_symptom(groups["category"], groups["reason"], groups["status"])
    if symptom is None:
        return []
    # HEALTH_EVENT-class signals (e.g. capacity reductions, completed maintenance) are investigation-only:
    # emit interesting (not bad) so they reach s_interesting_nodes without driving the outcome.
    # Termination categories stay bad.
    emit = _interesting if groups["category"] in INTERESTING_CATEGORIES else _bad
    return [emit(node, symptom, source.derived(rule="csp-event"))]
