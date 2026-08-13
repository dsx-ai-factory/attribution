# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Exercise ``SyslogLustreError`` event sets through the declarative engine.

The parity fixture mirrors the bundled TOML and is loaded into a fresh
``RuleEngine`` so load-time branch coverage is enforced independently.
"""

import re
from pathlib import Path

from attribution_lib.declarative import load_toml_rules
from attribution_lib.events import DemotionTypes
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rules_engine import RuleContext, RuleEngine
from attribution_lib.sources import Source

FIX = Path(__file__).parent / "fixtures" / "syslog_lustre_rule.toml"

# Real parity lines from origin/main:syslog_rules.test.toml.
EVICTED = (
    "LustreError: 167-0: testfs-OST001c-osc-ffff94dc5684f800: This client was "
    "evicted by testfs-OST001c; in progress operations using this service will fail."
)
NONFATAL = "LustreError: 52716:0:(events.c:651:ptlrpc_init_portals()) network initialisation failed"


def _load():
    """Load the fixture into a FRESH engine (coverage is enforced at load)."""
    engine = RuleEngine()
    return load_toml_rules(engine, str(FIX))["SyslogLustreError"]


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


def _triples(events):
    return {(e.type, e.payload["symptom"], e.source.rule) for e in events}


def test_lustre_evicted_emits_bad_pair(dummy_ctx):
    events = _run(_load(), EVICTED, dummy_ctx)

    # try_all emits exactly two events: the guarded bad event and the evicted branch.
    assert len(events) == 2
    assert _triples(events) == {
        ("Attribution.node.bad", "LustreError", "lustre-error"),
        ("Attribution.node.bad", "Lustre Client Evicted", "lustre-client-evicted"),
    }

    # The demotion map is preserved on the guarded bad event.
    always = next(e for e in events if e.payload["symptom"] == "LustreError")
    assert always.payload[DemotionTypes.NOT_NEAR_WATCHDOG_EVENT_DEMOTION] == "lustre-error-non-fatal-watchdog"
    assert always.payload[DemotionTypes.BEFORE_DATALOADER_DEMOTION] == "lustre-error-non-fatal-dataloader-exception"


def test_lustre_nonfatal_emits_bad_plus_interesting(dummy_ctx):
    events = _run(_load(), NONFATAL, dummy_ctx)

    assert len(events) == 2
    assert _triples(events) == {
        ("Attribution.node.bad", "LustreError", "lustre-error"),
        ("Attribution.node.interesting", "(non-fatal) LustreError", "lustre-error-non-fatal"),
    }

    # Non-evicted line reaches the pick_one default, so no client-evicted event
    # and the guarded bad event still carries the demotion map.
    always = next(e for e in events if e.payload["symptom"] == "LustreError")
    assert always.payload[DemotionTypes.NOT_NEAR_WATCHDOG_EVENT_DEMOTION] == "lustre-error-non-fatal-watchdog"
    assert always.payload[DemotionTypes.BEFORE_DATALOADER_DEMOTION] == "lustre-error-non-fatal-dataloader-exception"


def test_lustre_rule_loads_with_coverage():
    """Loading on a fresh engine enforces branch-complete coverage; if the cases
    did not cover the outer and nested pick_one paths, loading would fail."""
    engine = RuleEngine()
    added = load_toml_rules(engine, str(FIX))
    assert set(added) == {"SyslogLustreError"}
    assert engine.extractor_rules["SyslogLustreError"] is added["SyslogLustreError"]
