# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
UFM log rules ported from attribution_lib/ufmlog.py check().
"""

from attribution_lib.rules import declare_event_types

rule_author = "NVIDIA Attribution team"

declare_event_types("Attribution.node.bad", "Attribution.node.interesting")

# The two end-of-job leaf rules (UfmlogLinkDown / UfmlogNodeDown) are now
# declared in ufmlog_leaf_rules.toml using the at_end_of_job() lifecycle
# predicate. This module retains the source/event-type registration only.
