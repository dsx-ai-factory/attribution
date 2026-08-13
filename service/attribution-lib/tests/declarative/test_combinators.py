# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest

from attribution_lib.declarative.coverage import (
    check_coverage,
    covered_obligations,
    element_obligations,
)
from attribution_lib.declarative.emit import LoadError, compile_element, compile_pick_one, compile_try_all
from attribution_lib.events import DemotionTypes
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rules_engine import RuleContext
from attribution_lib.sources import Source


def _src(dummy_ctx, line="x"):
    return Source(source_context=dummy_ctx, line_num=1, datum=line, time=None, resources=AllJobResources, rule=None)


def _ctx():
    return RuleContext(job_info=None, hostname="node01")


PICK = {
    "type": "pick_one",
    "branches": [
        {"type": "bad_node", "symptom": "hi-val", "when": "int(value) > 100"},
        {"type": "interesting_node", "symptom": "mid-val", "when": "int(value) > 10"},
    ],
    "default": {"type": "interesting_node", "symptom": "low-val"},
}


def test_pick_one_selects_first_matching_branch(dummy_ctx):
    proc = compile_element(PICK, rule_name="r")
    ev = proc({"value": "200"}, "n", _src(dummy_ctx), _ctx())
    assert len(ev) == 1 and ev[0].type == "Attribution.node.bad" and ev[0].payload["symptom"] == "hi-val"


def test_pick_one_first_match_wins_not_all(dummy_ctx):
    # value 200 satisfies BOTH branch whens; only the first fires
    proc = compile_element(PICK, rule_name="r")
    ev = proc({"value": "200"}, "n", _src(dummy_ctx), _ctx())
    assert len(ev) == 1 and ev[0].payload["symptom"] == "hi-val"


def test_pick_one_reaches_default(dummy_ctx):
    proc = compile_element(PICK, rule_name="r")
    ev = proc({"value": "5"}, "n", _src(dummy_ctx), _ctx())
    assert len(ev) == 1 and ev[0].payload["symptom"] == "low-val"


def test_pick_one_default_none_emits_nothing(dummy_ctx):
    pick = {
        "type": "pick_one",
        "branches": [{"type": "bad_node", "symptom": "x", "when": "int(value) > 100"}],
        "default": "none",
    }
    proc = compile_element(pick, rule_name="r")
    assert proc({"value": "5"}, "n", _src(dummy_ctx), _ctx()) == []


def test_pick_one_branch_without_when_is_load_error():
    bad = {"type": "pick_one", "branches": [{"type": "bad_node", "symptom": "x"}], "default": "none"}
    with pytest.raises(LoadError):
        compile_element(bad, rule_name="r")


def test_pick_one_missing_default_is_load_error():
    bad = {"type": "pick_one", "branches": [{"type": "bad_node", "symptom": "x", "when": "int(value) > 1"}]}
    with pytest.raises(LoadError):
        compile_element(bad, rule_name="r")


def test_try_all_emits_each_firing_event(dummy_ctx):
    ta = {
        "type": "try_all",
        "events": [
            {"type": "bad_node", "symptom": "always"},
            {"type": "interesting_node", "symptom": "cond", "when": "int(value) > 10"},
        ],
    }
    proc = compile_element(ta, rule_name="r")
    ev = proc({"value": "50"}, "n", _src(dummy_ctx), _ctx())
    assert [e.payload["symptom"] for e in ev] == ["always", "cond"]
    ev2 = proc({"value": "1"}, "n", _src(dummy_ctx), _ctx())
    assert [e.payload["symptom"] for e in ev2] == ["always"]


def test_try_all_outer_when_gates_all(dummy_ctx):
    ta = {"type": "try_all", "when": "int(value) > 0", "events": [{"type": "bad_node", "symptom": "x"}]}
    proc = compile_element(ta, rule_name="r")
    assert proc({"value": "0"}, "n", _src(dummy_ctx), _ctx()) == []


def test_demotion_dict_maps_names_to_enum_keys(dummy_ctx):
    leaf = {
        "type": "bad_node",
        "symptom": "L",
        "demotion": {"NOT_NEAR_WATCHDOG_EVENT_DEMOTION": "w", "BEFORE_DATALOADER_DEMOTION": "d"},
    }
    proc = compile_element(leaf, rule_name="r")
    ev = proc({}, "n", _src(dummy_ctx), _ctx())
    p = ev[0].payload
    assert p[DemotionTypes.NOT_NEAR_WATCHDOG_EVENT_DEMOTION] == "w"
    assert p[DemotionTypes.BEFORE_DATALOADER_DEMOTION] == "d"


def test_demotion_unknown_name_is_load_error():
    leaf = {"type": "bad_node", "symptom": "L", "demotion": {"NOPE": "x"}}
    with pytest.raises(LoadError):
        compile_element(leaf, rule_name="r")


def test_when_can_reference_datum(dummy_ctx):
    # `search` (not `match`): the token appears mid-line, and `datum` is the full
    # line. `re.match` anchors at start-of-string and would never fire here.
    leaf = {"type": "bad_node", "symptom": "x", "when": "search('evicted', datum)"}
    proc = compile_element(leaf, rule_name="r")
    ev_hit = proc({}, "n", _src(dummy_ctx, "host was evicted now"), _ctx())
    ev_miss = proc({}, "n", _src(dummy_ctx, "nothing here"), _ctx())
    assert len(ev_hit) == 1 and ev_miss == []


# --- combinator coverage obligations --------------------------------------------


def _value_groups(case):
    """Toy match_groups: capture the numeric token after 'value '."""
    return {"value": case["line"].split()[1]}


def test_pick_one_obligations_enumerated():
    obs = element_obligations(PICK)
    assert "branch[0]:selected" in obs
    assert "branch[1]:selected" in obs
    assert "default:reached" in obs


def test_pick_one_missing_default_case_fails_coverage():
    # covers branch0 + branch1 but never the default -> load fails, naming default
    cases = [{"line": "value 200", "expect": "bad"}, {"line": "value 50", "expect": "interesting"}]
    with pytest.raises(LoadError, match="default"):
        check_coverage("r", PICK, cases, untested=None, match_groups=_value_groups)


def test_pick_one_full_coverage_passes():
    cases = [
        {"line": "value 200", "expect": "bad"},
        {"line": "value 50", "expect": "interesting"},
        {"line": "value 5", "expect": "interesting"},
    ]
    # must not raise
    check_coverage("r", PICK, cases, untested=None, match_groups=_value_groups)


def test_pick_one_missing_a_branch_fails_coverage():
    # never selects branch1 (nothing in (10, 100]) -> branch[1]:selected uncovered
    cases = [
        {"line": "value 200", "expect": "bad"},
        {"line": "value 5", "expect": "interesting"},
    ]
    with pytest.raises(LoadError, match="branch"):
        check_coverage("r", PICK, cases, untested=None, match_groups=_value_groups)


def test_pick_one_combinator_when_adds_obligations():
    pick = {
        "type": "pick_one",
        "when": "int(value) > 0",
        "branches": [{"type": "bad_node", "symptom": "x", "when": "int(value) > 100"}],
        "default": "none",
    }
    obs = element_obligations(pick)
    assert "when:true" in obs and "when:false" in obs
    assert "branch[0]:selected" in obs and "default:reached" in obs


def test_try_all_outer_when_has_obligations():
    emit = {
        "type": "try_all",
        "when": "enabled",
        "events": [{"type": "bad_node", "symptom": "x"}],
    }
    assert {"when:true", "when:false", "event[0]:emit"} <= element_obligations(emit)


def test_pick_one_combinator_branch_has_nested_obligations():
    emit = {
        "type": "pick_one",
        "branches": [
            {
                "type": "try_all",
                "when": "enabled",
                "events": [{"type": "bad_node", "symptom": "x"}],
            }
        ],
        "default": "none",
    }
    assert "branch[0].event[0]:emit" in element_obligations(emit)


def test_combinator_coverage_requires_match_groups():
    emit = {"type": "try_all", "events": [{"type": "bad_node", "symptom": "x"}]}
    with pytest.raises(LoadError, match="captured groups"):
        covered_obligations(emit, [{"line": "x"}])


def test_nonmatching_line_does_not_cover_combinator():
    emit = {"type": "try_all", "events": [{"type": "bad_node", "symptom": "x"}]}
    assert covered_obligations(emit, [{"line": "x"}], match_groups=lambda case: {}) == set()


def test_combinator_outer_when_covers_both_sides():
    emit = {
        "type": "try_all",
        "when": "enabled",
        "events": [{"type": "bad_node", "symptom": "x"}],
    }
    cases = [{"line": "on", "enabled": True}, {"line": "off", "enabled": False}]
    hits = covered_obligations(emit, cases, match_groups=lambda case: {"enabled": case["enabled"]})
    assert {"when:true", "when:false"} <= hits


def test_empty_pick_one_fails_coverage_cleanly():
    emit = {"type": "pick_one", "branches": [], "default": "none"}
    with pytest.raises(LoadError, match="non-empty"):
        covered_obligations(emit, [{"line": "x"}], match_groups=lambda case: {"x": 1})


def test_selected_combinator_branch_is_traversed():
    emit = {
        "type": "pick_one",
        "branches": [
            {
                "type": "try_all",
                "when": "selected",
                "events": [{"type": "bad_node", "symptom": "x"}],
            }
        ],
        "default": "none",
    }
    hits = covered_obligations(emit, [{"line": "x"}], match_groups=lambda case: {"selected": True})
    assert "branch[0].event[0]:emit" in hits


# Capstone: try_all whose second event is a pick_one (nested path prefixes).
NESTED = {
    "type": "try_all",
    "events": [
        {"type": "bad_node", "symptom": "always"},
        {
            "type": "pick_one",
            "branches": [{"type": "interesting_node", "symptom": "hi", "when": "int(value) > 100"}],
            "default": {"type": "interesting_node", "symptom": "lo"},
        },
    ],
}


def test_nested_obligations_are_path_prefixed():
    obs = element_obligations(NESTED)
    assert "event[0]:emit" in obs
    assert "event[1].branch[0]:selected" in obs
    assert "event[1].default:reached" in obs


def test_nested_full_coverage_passes():
    cases = [
        {"line": "value 200", "expect": "bad"},  # event0 emits, pick_one selects branch0
        {"line": "value 5", "expect": "bad"},  # event0 emits, pick_one reaches default
    ]
    check_coverage("r", NESTED, cases, untested=None, match_groups=_value_groups)  # no raise


def test_nested_missing_default_case_fails():
    cases = [{"line": "value 200", "expect": "bad"}]  # never reaches the nested default
    with pytest.raises(LoadError, match="default"):
        check_coverage("r", NESTED, cases, untested=None, match_groups=_value_groups)


def test_nested_missing_branch_case_fails():
    cases = [{"line": "value 5", "expect": "bad"}]  # never selects the nested branch
    with pytest.raises(LoadError, match="branch"):
        check_coverage("r", NESTED, cases, untested=None, match_groups=_value_groups)


# --- try_all sibling guards vs. whole-rule `expect` cross-check ------------------

# Two independent `when`-bearing events: no single value can satisfy both guards,
# so a whole-rule `expect` label cannot describe any one event. The `expect` must
# NOT be cross-checked per-event (that would be a category error).
TWO_GUARD_TRY_ALL = {
    "type": "try_all",
    "events": [
        {"type": "bad_node", "symptom": "hi", "when": "int(value) > 5"},
        {"type": "interesting_node", "symptom": "lo", "when": "int(value) < 3"},
    ],
}


def test_try_all_sibling_guards_expect_not_cross_checked():
    # value 10: event0 `when` TRUE, event1 `when` FALSE.
    # value 1:  event0 `when` FALSE, event1 `when` TRUE.
    # Both cases carry `expect` labels; per-event cross-checking would (wrongly)
    # raise "mislabeled" because a whole-rule expect can't be true for both guards.
    cases = [
        {"line": "value 10", "expect": "bad"},
        {"line": "value 1", "expect": "interesting"},
    ]
    # Must load AND be complete: both events' true/false sides are exercised.
    check_coverage("r", TWO_GUARD_TRY_ALL, cases, untested=None, match_groups=_value_groups)

    need = element_obligations(TWO_GUARD_TRY_ALL)
    have = covered_obligations(TWO_GUARD_TRY_ALL, cases, match_groups=_value_groups, name="r")
    assert need <= have


def test_top_level_leaf_expect_cross_check_still_catches_mislabel():
    # A genuinely mislabeled top-level leaf: `when` TRUE over the captures but the
    # case claims no-emit. The cross-check MUST still fire here.
    leaf = {"type": "bad_node", "symptom": "x", "when": "int(value) > 5"}
    cases = [{"line": "value 10", "expect": "none"}]
    with pytest.raises(LoadError, match="mislabeled"):
        check_coverage("r", leaf, cases, untested=None, match_groups=_value_groups)


def test_pick_one_branch_expect_cross_check_still_catches_mislabel():
    # A pick_one selects exactly one branch (or its default), so a single `expect`
    # IS meaningful there and the cross-check must still catch a mislabel. Here the
    # default leaf carries its own `when`, so it flows through the `_hit` cross-check.
    pick = {
        "type": "pick_one",
        "branches": [{"type": "bad_node", "symptom": "x", "when": "int(value) > 100"}],
        "default": {"type": "interesting_node", "symptom": "lo", "when": "int(value) > 5"},
    }
    # value 10: no branch matches -> default reached; default `when` TRUE (emits) but
    # the case claims no-emit -> genuinely mislabeled, cross-check must fire.
    cases = [{"line": "value 10", "expect": "none"}]
    with pytest.raises(LoadError, match="mislabeled"):
        check_coverage("r", pick, cases, untested=None, match_groups=_value_groups)


# --- fix 2: malformed combinators fail coverage with a friendly LoadError --------


def test_pick_one_missing_default_raises_load_error_not_keyerror():
    # `default` absent: `_cover` (which runs before compile validates) must raise a
    # friendly LoadError naming the rule, not a raw KeyError.
    malformed = {
        "type": "pick_one",
        "branches": [{"type": "bad_node", "symptom": "x", "when": "int(value) > 100"}],
    }
    cases = [{"line": "value 200", "expect": "bad"}]
    with pytest.raises(LoadError, match="default"):
        check_coverage("r", malformed, cases, untested=None, match_groups=_value_groups)


def test_pick_one_branch_missing_when_raises_load_error_not_keyerror():
    malformed = {
        "type": "pick_one",
        "branches": [{"type": "bad_node", "symptom": "x"}],
        "default": "none",
    }
    cases = [{"line": "value 200", "expect": "bad"}]
    with pytest.raises(LoadError, match="when"):
        check_coverage("r", malformed, cases, untested=None, match_groups=_value_groups)


def test_try_all_missing_events_raises_load_error_not_keyerror():
    malformed = {"type": "try_all"}
    cases = [{"line": "value 200", "expect": "bad"}]
    with pytest.raises(LoadError, match="events"):
        check_coverage("r", malformed, cases, untested=None, match_groups=_value_groups)


# --- fix 3: a selected leaf branch's `expect` label is cross-checked --------------

LEAF_BRANCH_PICK = {
    "type": "pick_one",
    "branches": [{"type": "bad_node", "symptom": "x", "when": "int(value) > 5"}],
    "default": "none",
}


def test_pick_one_leaf_branch_mislabel_raises():
    # value 10 selects the leaf branch (its `when` is TRUE), but the case claims
    # no-emit: a genuine mislabel that the cross-check must now catch.
    cases = [{"line": "value 10", "expect": "none"}]
    with pytest.raises(LoadError, match="mislabeled"):
        check_coverage("r", LEAF_BRANCH_PICK, cases, untested=None, match_groups=_value_groups)


def test_pick_one_leaf_branch_correctly_labeled_passes():
    # branch selected (value 10, expect bad) + default reached (value 1) -> full,
    # correctly-labeled coverage must load without raising.
    cases = [
        {"line": "value 10", "expect": "bad"},
        {"line": "value 1", "expect": "none"},
    ]
    check_coverage("r", LEAF_BRANCH_PICK, cases, untested=None, match_groups=_value_groups)


@pytest.mark.parametrize(
    ("compiler", "emit", "message"),
    [
        (compile_pick_one, {"type": "pick_one", "branches": [], "default": "none"}, "non-empty"),
        (
            compile_pick_one,
            {
                "type": "pick_one",
                "branches": [{"when": "True", "type": "bad_node", "symptom": "x"}],
                "default": "bad",
            },
            "default.*element",
        ),
        (compile_try_all, {"type": "try_all", "events": []}, "non-empty"),
    ],
)
def test_combinator_compilers_reject_malformed_elements(compiler, emit, message):
    with pytest.raises(LoadError, match=message):
        compiler(emit, rule_name="r")


@pytest.mark.parametrize(
    ("compiler", "emit"),
    [
        (
            compile_pick_one,
            {
                "type": "pick_one",
                "when": "enabled",
                "branches": [{"when": "True", "type": "bad_node", "symptom": "x"}],
                "default": "none",
            },
        ),
        (
            compile_try_all,
            {
                "type": "try_all",
                "when": "enabled",
                "events": [{"type": "bad_node", "symptom": "x"}],
            },
        ),
    ],
)
def test_combinator_outer_when_can_gate(compiler, emit, dummy_ctx):
    proc = compiler(emit, rule_name="r")
    assert proc({"enabled": False}, "node", _src(dummy_ctx), _ctx()) == []
