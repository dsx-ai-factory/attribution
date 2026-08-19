# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Exercise record-table projection through ``catalog_lookup`` on a fresh loader.

The fixture covers a record table whose row columns are projected into emitted
events, including both catalog-hit and catalog-miss behavior.
"""

import re
from pathlib import Path

from attribution_lib.declarative import load_toml_rules
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rules_engine import RuleContext, RuleLoader
from attribution_lib.sources import Source

FIX = Path(__file__).parent / "fixtures" / "record_table_project.toml"


def _load():
    """Load the fixture into a fresh loader (coverage is enforced at load)."""
    loader = RuleLoader()
    return load_toml_rules(loader, str(FIX))["SxidRecordProject"]


def _run(rule, line, dummy_ctx):
    m = re.search(rule.match_re, line)
    assert m is not None, "line did not match match_re"
    src = Source(
        source_context=dummy_ctx,
        line_num=1,
        datum=line,
        time=None,
        resources=AllJobResources,
        rule=None,
    )
    return rule.process(m, src, RuleContext(job_info=None, hostname="node01"))


def test_record_hit_projects_nice_name(dummy_ctx):
    events = _run(_load(), "SXID 48 detected on gpu", dummy_ctx)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == "Attribution.node.interesting"
    # symptom PROJECTED from the record's `nice_name` column, not the line capture.
    assert ev.payload["symptom"] == "GPU fell off the bus"


def test_record_miss_emits_fallback_and_credits_missing_rule(dummy_ctx):
    events = _run(_load(), "SXID 12345 detected on gpu", dummy_ctx)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == "Attribution.node.interesting"
    # not_found leaf renders the line capture and credits the fixed rule name.
    assert ev.payload["symptom"] == "SXID-12345"
    assert ev.source.rule == "sxid-id-missing-in-catalog"


def test_record_fixture_loads_with_coverage():
    """Loading on a fresh loader enforces row-complete coverage; if the cases did
    not cover every row plus not_found the load would raise LoadError."""
    loader = RuleLoader()
    added = load_toml_rules(loader, str(FIX))
    assert set(added) == {"SxidRecordProject"}
