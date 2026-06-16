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
UFM log rules ported from attribution_lib/ufmlog.py check().
"""

from typing import Dict, List

from common.ufmlog_prefilter import LINK_DOWN_EAST_WEST, NODE_DOWN_UFM

from attribution_lib.observation.inbound import Source as ObsSource
from attribution_lib.rule_helpers import _bad, _interesting
from attribution_lib.rules_engine import EventTriple, RuleContext, declare_event_types, extractor_def
from attribution_lib.sources import Source
from attribution_lib.utils import is_event_at_end_of_job

rule_author = "FACT Team"
rule_sources = {ObsSource.UFMLOG}

declare_event_types("Attribution.node.bad", "Attribution.node.interesting")


@extractor_def(name="UfmlogLinkDown", match_re="Link went down", derived_rules=["ufm"])
def ufmlog_link_down(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    if is_event_at_end_of_job(source.time, ctx.job_info.job_end_time):
        return [_bad(node, LINK_DOWN_EAST_WEST, source.derived(rule="ufm"))]
    return []


@extractor_def(name="UfmlogNodeDown", match_re="node is Down")
def ufmlog_node_down(groups: Dict[str, str], node: str, source: Source, ctx: RuleContext) -> List[EventTriple]:
    if is_event_at_end_of_job(source.time, ctx.job_info.job_end_time):
        return [_interesting(node, NODE_DOWN_UFM, source.derived(rule="ufm"))]
    return []
