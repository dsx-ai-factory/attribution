"""Tests for the @expr_helper registration seam (Plan 5a, Task 1)."""

import pytest

from attribution_lib.declarative.errors import LoadError
from attribution_lib.declarative.expressions import (
    _FIXED_FUNCS,
    called_function_names,
    check_helpers_resolvable,
    eval_bool,
    merge_functions,
    render_template,
)
from attribution_lib.rules_engine import (
    _EXPR_HELPER_RESERVED,
    EXPR_HELPER_INJECTED_PARAMS,
    RuleEngine,
    RuleIngestionError,
    _rules_engine,
    expr_helper,
)


def _with_engine():
    """Context: an active engine bound to the _rules_engine ContextVar."""
    engine = RuleEngine()
    token = _rules_engine.set(engine)
    return engine, token


def test_register_and_retrieve():
    engine, token = _with_engine()
    try:

        @expr_helper("is_user_path")
        def _impl(path):
            return path.startswith("/home/")

        assert engine.expr_helpers["is_user_path"] is _impl
        # decorator returns the function unchanged (still directly callable)
        assert _impl("/home/x") is True
    finally:
        _rules_engine.reset(token)


def test_duplicate_name_is_error():
    engine, token = _with_engine()
    try:

        @expr_helper("dup")
        def _a(x):
            return x

        with pytest.raises(RuleIngestionError, match="already registered"):

            @expr_helper("dup")
            def _b(x):
                return x
    finally:
        _rules_engine.reset(token)


def test_shadowing_fixed_function_is_error():
    engine, token = _with_engine()
    try:
        with pytest.raises(RuleIngestionError, match="shadows a fixed function"):

            @expr_helper("int")
            def _impl(x):
                return x
    finally:
        _rules_engine.reset(token)


def test_injection_metadata_recorded():
    engine, token = _with_engine()
    try:

        @expr_helper("needs_ctx")
        def _h(x, *, ctx):
            return x

        @expr_helper("needs_both")
        def _h2(x, *, source, ctx):
            return x

        @expr_helper("plain")
        def _h3(x):
            return x

        assert engine.expr_helper_injections["needs_ctx"] == frozenset({"ctx"})
        assert engine.expr_helper_injections["needs_both"] == frozenset({"source", "ctx"})
        assert engine.expr_helper_injections["plain"] == frozenset()
    finally:
        _rules_engine.reset(token)


def test_no_active_engine_is_error():
    # No _rules_engine bound.
    with pytest.raises(RuleIngestionError, match="No active rules engine"):

        @expr_helper("orphan")
        def _impl(x):
            return x


def test_fresh_engine_starts_empty():
    assert RuleEngine().expr_helpers == {}


def test_merge_functions_empty_returns_fixed():
    assert merge_functions(None) is _FIXED_FUNCS
    assert merge_functions({}) is _FIXED_FUNCS


def test_merge_functions_adds_helpers():
    fns = merge_functions({"is_user_path": lambda p: True})
    assert fns["is_user_path"]("x") is True
    assert fns["int"] is int  # fixed functions still present
    assert "is_user_path" not in _FIXED_FUNCS  # original not mutated


def test_eval_bool_with_helper():
    fns = merge_functions({"is_user_path": lambda p: p.startswith("/home/")})
    assert eval_bool("is_user_path(path)", {"path": "/home/x"}, functions=fns) is True
    assert eval_bool("is_user_path(path)", {"path": "/etc/x"}, functions=fns) is False


def test_render_template_with_helper():
    fns = merge_functions({"pfx": lambda line, msg: f"fn: {msg}"})
    out = render_template("{pfx(datum, 'boom')}", {"datum": "x"}, functions=fns)
    assert out == "fn: boom"


def test_called_function_names():
    assert called_function_names("is_user_path(path)") == {"is_user_path"}
    assert called_function_names("int(round(int(seconds) / 60))") == {"int", "round"}
    assert called_function_names("value") == set()


def test_check_helpers_resolvable_unknown_raises():
    with pytest.raises(LoadError, match="unknown function 'mystery'"):
        check_helpers_resolvable("mystery(path)", allowed=set(), is_template=False)


def test_check_helpers_resolvable_allows_fixed_and_registered():
    # a template with a registered helper and a fixed function -> no raise
    check_helpers_resolvable("{is_user_path(path)} {int(x)}", allowed={"is_user_path"})


def test_keyword_capture_gives_clear_error():
    with pytest.raises(LoadError, match="keyword 'class'"):
        check_helpers_resolvable("{class}", allowed=set())


def test_reserved_matches_fixed_funcs():
    assert _EXPR_HELPER_RESERVED == set(_FIXED_FUNCS)


@pytest.mark.parametrize("pname", EXPR_HELPER_INJECTED_PARAMS)
def test_positional_only_injected_param_is_error(pname):
    # A helper declaring source/ctx as positional-only (before /) can't receive
    # the keyword injection the engine performs; reject it at registration time.
    engine, token = _with_engine()
    try:
        with pytest.raises(RuleIngestionError, match=f"{pname!r}.*positional-only"):
            exec(
                f"from attribution_lib.rules_engine import expr_helper\n"
                f"@expr_helper('bad_{pname}')\n"
                f"def _h({pname}, /, x): pass",
                {},
            )
    finally:
        _rules_engine.reset(token)
