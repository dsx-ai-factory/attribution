# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_lib.declarative.errors import LoadError
from attribution_lib.declarative.expressions import (
    check_guarded,
    check_template_syntax,
    eval_bool,
    optional_captures,
    render_template,
)


def test_plain_literal_passthrough():
    assert render_template("GspRmAlloc failed", {}) == "GspRmAlloc failed"


def test_capture_interpolation():
    assert render_template("XID-{sxid}", {"sxid": "31"}) == "XID-31"


def test_computed_expression():
    assert render_template("{int(round(int(seconds) / 60))}", {"seconds": "150"}) == "2"


def test_literal_braces_escaped():
    assert render_template("a {{literal}} {x}", {"x": "v"}) == "a {literal} v"


def test_eval_bool_numeric_when():
    assert eval_bool("float(value) > 50.0", {"value": "60"}) is True
    assert eval_bool("float(value) > 50.0", {"value": "10"}) is False


def test_optional_captures_detects_optional_group():
    assert optional_captures(r"a(?P<x>\d+)?b(?P<y>\d+)") == {"x"}  # y is required


def test_optional_captures_detects_enclosing_optional_group():
    # loc lives inside an optional non-capturing group; code is required.
    optional = optional_captures(r"code (?P<code>\w+)(?: at (?P<loc>\S+))?")
    assert optional == {"loc"}


def test_guarded_use_accepted():
    # bare `when`-style expression (is_template=False): exercises the real guard
    # logic on the whole string, not the empty placeholder set of a brace-less template.
    check_guarded("uid is not None and int(uid) > 0", {"uid"}, is_template=False)  # no raise
    check_guarded("{file_line if file_line else ''}", {"file_line"})


def test_guarded_and_chain_accepted():
    # an and-chain: each conjunct may assume earlier ones held. Bare expression, so
    # is_template=False forces the checker to actually parse and walk the chain.
    check_guarded("a is not None and b is not None and int(a) + int(b) > 0", {"a", "b"}, is_template=False)  # no raise


def test_unguarded_bare_when_rejected():
    # production `when`-check path: an unguarded optional capture in a bare
    # expression must raise (loader passes is_template=False for `when`).
    with pytest.raises(LoadError):
        check_guarded("int(uid) > 0", {"uid"}, is_template=False)


def test_broken_and_chain_rejected():
    # the guard for `b` is missing, so `int(b)` is an unguarded use of an optional.
    with pytest.raises(LoadError):
        check_guarded("a is not None and int(a) + int(b) > 0", {"a", "b"}, is_template=False)


def test_literal_symptom_template_not_rejected():
    # a literal symptom string (no placeholders) under an optional-capture stanza
    # must NOT raise when checked as a template -- this is the check_guarded fix
    # that unblocked FmLost's literal "Fabric manager lost connection to GPU" symptom.
    check_guarded("Fabric manager lost connection to GPU", {"gpu_guid"})  # no raise


def test_unguarded_use_rejected():
    with pytest.raises(LoadError):
        check_guarded("SXID-{file_line}", {"file_line"})


def test_optional_via_question_and_star():
    assert optional_captures(r"a(?P<x>\d+)?b(?P<y>\d+)") == {"x"}


def test_optional_via_bounded_repeat_zero_min():
    assert optional_captures(r"a(?P<z>\d+){0,3}") == {"z"}
    assert optional_captures(r"a(?P<z>\d+){0,}") == {"z"}


def test_required_via_repeat_nonzero_min():
    assert optional_captures(r"a(?P<z>\d+){1,3}") == set()


def test_optional_via_alternation():
    # y only participates when the second alternative matches
    assert optional_captures(r"(?P<x>a)|(?P<y>b)") == {"x", "y"}


def test_optional_via_enclosing_optional_group():
    assert optional_captures(r"a(?:.*?gpuGUID (?P<g>0x[0-9a-f]+))?") == {"g"}


def test_required_capture_not_flagged():
    assert optional_captures(r"code (?P<code>\w+)") == set()


# --- check_template_syntax ---------------------------------------------------


def test_valid_templates_pass_syntax_check():
    for tmpl in ("plain text", "{expr}", "{{literal}}", "{a} and {b}", "{{}} ok {x}"):
        check_template_syntax(tmpl, where="test")


def test_stray_open_brace_raises():
    with pytest.raises(LoadError, match="stray"):
        check_template_syntax("hello {world", where="r")


def test_stray_close_brace_raises():
    with pytest.raises(LoadError, match="stray"):
        check_template_syntax("hello world}", where="r")


def test_balanced_nested_braces_are_valid():
    # Balanced nested braces (e.g. set literals) are legal placeholder content.
    check_template_syntax("{x if x in {'a', 'b'} else 'c'}", where="r")  # no raise


def test_unmatched_open_brace_in_placeholder_raises():
    # An outer { with no matching } is still a stray brace.
    with pytest.raises(LoadError, match="stray"):
        check_template_syntax("{outer {inner}", where="r")


def test_escaped_braces_are_valid():
    check_template_syntax("{{ not a placeholder }}", where="r")
    check_template_syntax("{expr} and {{literal}}", where="r")


def test_brace_inside_string_literal_in_placeholder():
    # A regex quantifier like \d{3} inside a quoted string must not be mistaken
    # for a nested brace delimiter; the template should parse and render correctly.
    check_template_syntax(r"{search(r'\d{3}', datum).group(0) if search(r'\d{3}', datum) else datum}", where="r")
    result = render_template(
        r"{search(r'\d{3}', datum).group(0) if search(r'\d{3}', datum) else datum}",
        {"datum": "code 123"},
    )
    assert result == "123"


def test_expression_scanner_handles_triple_quoted_braces():
    check_template_syntax('{"""literal { brace }"""}', where="r")


def test_or_chain_treats_tested_optional_names_as_safe():
    check_guarded("a or b or ''", {"a", "b"}, is_template=False)


@pytest.mark.parametrize(
    "expr",
    [
        "None is x",
        "None is not x and int(x) > 0",
        "0 if x is None else int(x)",
        "0 if None is x else int(x)",
    ],
)
def test_none_guards_accept_reversed_and_false_forms(expr):
    check_guarded(expr, {"x"}, is_template=False)
