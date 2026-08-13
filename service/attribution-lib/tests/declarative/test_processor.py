"""Tests for the @rule_processor seam (Plan 5e, Task 1)."""

import textwrap

import pytest

from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.emit import compile_processor
from attribution_lib.declarative.errors import LoadError
from attribution_lib.rules_engine import (
    RuleEngine,
    RuleIngestionError,
    _rules_engine,
    rule_processor,
)


def _with_engine():
    engine = RuleEngine()
    token = _rules_engine.set(engine)
    return engine, token


def test_register_and_retrieve():
    engine, token = _with_engine()
    try:

        @rule_processor("applog:pushrecv")
        def _p(groups, node, source, ctx):
            return []

        assert engine.rule_processors["applog:pushrecv"] is _p
    finally:
        _rules_engine.reset(token)


def test_duplicate_ns_name_is_error():
    engine, token = _with_engine()
    try:

        @rule_processor("applog:dup")
        def _a(groups):
            return []

        with pytest.raises(RuleIngestionError, match="applog:dup"):

            @rule_processor("applog:dup")
            def _b(groups):
                return []
    finally:
        _rules_engine.reset(token)


def test_no_active_engine_is_error():
    with pytest.raises(RuleIngestionError, match="No active rules engine"):

        @rule_processor("applog:orphan")
        def _p(groups):
            return []


def test_fresh_engine_starts_empty():
    assert RuleEngine().rule_processors == {}


def test_processor_injects_only_declared_params():
    seen = {}

    def proc(groups, *, ctx):
        seen["groups"] = groups
        seen["ctx"] = ctx
        return ["ok"]

    closure = compile_processor(proc, rule_name="r")
    out = closure({"a": "1"}, "node-1", "SRC", "CTX")  # type: ignore[arg-type]
    assert out == ["ok"]
    assert seen["groups"] == {"a": "1"}
    assert seen["ctx"] == "CTX"  # ctx injected; node/source NOT passed (not declared)


def test_processor_full_signature():
    seen = {}

    def proc(groups, node, source, ctx):
        seen.update(groups=groups, node=node, source=source, ctx=ctx)
        return []

    closure = compile_processor(proc, rule_name="r")
    closure({"x": "y"}, "n", "s", "c")  # type: ignore[arg-type]
    assert seen == {"groups": {"x": "y"}, "node": "n", "source": "s", "ctx": "c"}


def test_processor_groups_only():
    def proc(groups):
        return [groups["k"]]

    closure = compile_processor(proc, rule_name="r")
    assert closure({"k": "v"}, "n", "s", "c") == ["v"]  # type: ignore[arg-type]


def _load(tmp_path, toml_text, register=None):
    engine = RuleEngine()
    token = _rules_engine.set(engine)
    try:
        if register:
            register()
        f = tmp_path / "rules.toml"
        f.write_text(textwrap.dedent(toml_text))
        return engine, load_toml_rules(engine, str(f))
    finally:
        _rules_engine.reset(token)


def test_processor_stanza_registers_and_declares(tmp_path):
    def register():
        @rule_processor("applog:pr")
        def _pr(groups, node, source, ctx):
            return []

    toml = r"""
    sources = ["APPLOG"]
    [[extractor]]
    name = "pr"
    match_re = 'pushrecv'
    processor = "applog:pr"
    derived_rules = ["pushrecv"]
    derived_event_types = ["Attribution.node.bad", "Attribution.node.bad_peers"]
    """
    engine, added = _load(tmp_path, toml, register)
    assert "pr" in added
    assert "pushrecv" in engine.declared_rule_names
    assert "Attribution.node.bad_peers" in engine.declared_event_types


def test_processor_stanza_merges_file_and_rule_tags(tmp_path):
    def register():
        @rule_processor("applog:pr")
        def _pr(groups, node, source, ctx):
            return []

    toml = r"""
    sources = ["APPLOG"]
    tags = ["oom"]
    [[extractor]]
    name = "pr"
    match_re = 'pushrecv'
    processor = "applog:pr"
    tags = ["external"]
    """
    _engine, added = _load(tmp_path, toml, register)
    assert added["pr"].tags == ("external", "oom")


def test_processor_and_emit_both_present_is_error(tmp_path):
    def register():
        @rule_processor("applog:pr")
        def _pr(groups):
            return []

    toml = r"""
    sources = ["APPLOG"]
    [[extractor]]
    name = "pr"
    match_re = 'x'
    processor = "applog:pr"
    emit = { type = "bad_node", symptom = "s" }
    """
    with pytest.raises(LoadError, match="exactly one"):
        _load(tmp_path, toml, register)


def test_unknown_processor_is_error(tmp_path):
    toml = r"""
    sources = ["APPLOG"]
    [[extractor]]
    name = "pr"
    match_re = 'x'
    processor = "applog:missing"
    """
    with pytest.raises(LoadError, match="applog:missing"):
        _load(tmp_path, toml)


@pytest.mark.parametrize(
    ("func", "message"),
    [
        (lambda: None, "first positional"),
        (lambda groups, required: [], "unsatisfied required"),
    ],
)
def test_compile_processor_rejects_unsatisfied_signatures(func, message):
    with pytest.raises(LoadError, match=message):
        compile_processor(func, rule_name="r")


def test_compile_processor_allows_variadic_and_defaulted_parameters():
    def processor(groups, optional=None, *args, **kwargs):
        return []

    compile_processor(processor, rule_name="r")


def test_compile_processor_rejects_positional_only_injection():
    def processor(groups, node, /):
        return []

    with pytest.raises(LoadError, match="positional-only"):
        compile_processor(processor, rule_name="r")
