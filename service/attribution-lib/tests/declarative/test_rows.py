"""Tests for the @rows record-table provider seam (Plan 5d, Task 1)."""

import textwrap

import pytest

from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.errors import LoadError
from attribution_lib.rules_engine import (
    RuleEngine,
    RuleIngestionError,
    _rules_engine,
    rows,
)


def _with_engine():
    engine = RuleEngine()
    token = _rules_engine.set(engine)
    return engine, token


def test_provider_registers_str_coerced_keys():
    engine, token = _with_engine()
    try:

        @rows("xid")
        def _x():
            return {63: {"nice_name": "GSP RPC Timeout"}, 74: {"nice_name": "NVLink Error"}}

        assert engine.record_tables["xid"] == {
            "63": {"nice_name": "GSP RPC Timeout"},
            "74": {"nice_name": "NVLink Error"},
        }
        assert _x()[63]["nice_name"] == "GSP RPC Timeout"  # returned unchanged
    finally:
        _rules_engine.reset(token)


def test_duplicate_table_name_is_error():
    engine, token = _with_engine()
    try:

        @rows("xid")
        def _a():
            return {1: {"nice_name": "a"}}

        with pytest.raises(RuleIngestionError, match="xid"):

            @rows("xid")
            def _b():
                return {2: {"nice_name": "b"}}
    finally:
        _rules_engine.reset(token)


def test_provider_must_return_dict():
    engine, token = _with_engine()
    try:
        with pytest.raises(RuleIngestionError, match="must return a dict"):

            @rows("bad")
            def _bad():
                return [(1, {"nice_name": "x"})]
    finally:
        _rules_engine.reset(token)


def test_no_active_engine_is_error():
    with pytest.raises(RuleIngestionError, match="No active rules engine"):

        @rows("orphan")
        def _x():
            return {1: {"nice_name": "x"}}


def test_fresh_engine_starts_empty():
    assert RuleEngine().record_tables == {}


def _load(tmp_path, toml_text, register=None):
    engine = RuleEngine()
    token = _rules_engine.set(engine)
    try:
        if register:
            register()  # register @rows providers on this engine (mimics a sibling .py)
        f = tmp_path / "rules.toml"
        f.write_text(textwrap.dedent(toml_text))
        return engine, load_toml_rules(engine, str(f))
    finally:
        _rules_engine.reset(token)


def test_catalog_lookup_over_provider_record_table(tmp_path):
    def register():
        @rows("xid")
        def _x():
            return {63: {"nice_name": "GSP RPC Timeout"}}

    toml = r"""
    sources = ["SYSLOG"]
    [[extractor]]
    name = "SXid"
    match_re = 'SXid.*: (?P<sxid>\d+)'
    [extractor.emit]
    type = "catalog_lookup"
    catalog = "xid"
    key = "int(sxid)"
    derived_rule = "known-sxid-event"
    project = { type = "interesting_node", symptom = "{nice_name}" }
    not_found = { type = "interesting_node", symptom = "SXID-{sxid}", derived_rule = "sxid-id-missing-in-catalog" }
    [[extractor.cases]]
    description = "hit"
    expect = "interesting"
    line = "SXid (PCI:x): 63"
    symptom = "GSP RPC Timeout"
    rule = "known-sxid-event"
    [[extractor.cases]]
    description = "miss"
    expect = "interesting"
    line = "SXid (PCI:x): 999"
    symptom = "SXID-999"
    rule = "sxid-id-missing-in-catalog"
    """
    engine, added = _load(tmp_path, toml, register)
    assert "SXid" in added
    # both credits declared (hit via lookup derived_rule, miss via not_found)
    assert "known-sxid-event" in engine.declared_rule_names
    assert "sxid-id-missing-in-catalog" in engine.declared_rule_names


def test_provider_table_name_collides_with_inline_rows(tmp_path):
    def register():
        @rows("xid")
        def _x():
            return {1: {"nice_name": "a"}}

    toml = r"""
    sources = ["SYSLOG"]
    [rows.xid]
    2 = { nice_name = "b" }
    """
    with pytest.raises(LoadError, match="xid"):
        _load(tmp_path, toml, register)


def test_record_project_own_derived_rule_is_declared(tmp_path):
    # A record project with its OWN derived_rule and no lookup derived_rule must be
    # declared (the Plan 3 gap). No lookup derived_rule here.
    def register():
        @rows("t")
        def _t():
            return {5: {"sym": "five"}}

    toml = r"""
    sources = ["SYSLOG"]
    [[extractor]]
    name = "R"
    match_re = 'k=(?P<k>\d+)'
    [extractor.emit]
    type = "catalog_lookup"
    catalog = "t"
    key = "int(k)"
    project = { type = "bad_node", symptom = "{sym}", derived_rule = "project-own-credit" }
    not_found = "none"
    [[extractor.cases]]
    expect = "bad"
    line = "k=5"
    symptom = "five"
    rule = "project-own-credit"
    [[extractor.cases]]
    expect = "no_match"
    line = "k=6"
    """
    engine, added = _load(tmp_path, toml, register)
    assert "project-own-credit" in engine.declared_rule_names
