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

from attribution_lib.rules_engine import declare_event_types

rule_author = "NVIDIA Attribution team"

declare_event_types("Attribution.node.bad", "Attribution.node.interesting")

# The two end-of-job leaf rules (UfmlogLinkDown / UfmlogNodeDown) are now
# declared in ufmlog_leaf_rules.toml using the at_end_of_job() lifecycle
# predicate. This module retains the source/event-type registration only.
