# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.catalogs import Tables, parse_tables, resolve_catalog_table
from attribution_lib.declarative.coverage import (
    check_catalog_coverage,
    covered_catalog_obligations,
    expand_catalog_cases,
)
from attribution_lib.declarative.emit import LoadError, compile_catalog_lookup
from attribution_lib.declarative.loader import _check_catalog_guards, _collect_catalog_declarations
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.rule_types.definitions import RuleContext
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.sources import Source


def _write(tmp_path, text):
    p = tmp_path / "r.toml"
    p.write_text(text)
    return str(p)


def test_parse_element_catalog():
    doc = {"catalog": {"device_checks": {"hw_x": {"type": "bad_node", "symptom": "X inactive"}}}}
    t = parse_tables(doc)
    assert t.element_catalogs["device_checks"]["hw_x"] == {"type": "bad_node", "symptom": "X inactive"}
    assert "device_checks" not in t.record_tables


def test_parse_record_table_str_keys():
    doc = {"rows": {"xid": {1: {"nice_name": "n1"}, "2": {"nice_name": "n2"}}}}
    t = parse_tables(doc)
    assert set(t.record_tables["xid"]) == {"1", "2"}  # int key coerced to str


def test_name_in_both_kinds_is_load_error():
    doc = {"catalog": {"foo": {"k": {"type": "bad_node", "symptom": "s"}}}, "rows": {"foo": {"k": {"c": "v"}}}}
    with pytest.raises(LoadError):
        parse_tables(doc)


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ({"catalog": []}, "catalog.*table"),
        ({"rows": []}, "rows.*table"),
        ({"catalog": {"x": []}}, "catalog.x.*table"),
        ({"catalog": {"x": {1: {}, "1": {}}}}, "both coerce"),
    ],
)
def test_parse_tables_rejects_malformed_sections(doc, message):
    with pytest.raises(LoadError, match=message):
        parse_tables(doc)


def test_resolve_catalog_requires_name():
    with pytest.raises(LoadError, match=r"missing.*catalog"):
        resolve_catalog_table({}, Tables(), rule_name="r")


def _src(dummy_ctx, line="x"):
    return Source(source_context=dummy_ctx, line_num=1, datum=line, time=None, resources=AllJobResources, rule=None)


def _ctx():
    return RuleContext(job_info=None, hostname="n")


ELEM = Tables(
    element_catalogs={
        "device_checks": {
            "hw_a": {"type": "bad_node", "symptom": "A bad"},
            "hw_b": {"type": "interesting_node", "symptom": "B interesting"},
        }
    }
)


def test_element_catalog_hit_emits_row(dummy_ctx):
    emit = {"type": "catalog_lookup", "catalog": "device_checks", "key": "check", "not_found": "none"}
    proc = compile_catalog_lookup(emit, rule_name="r", tables=ELEM)
    ev = proc({"check": "hw_a"}, "n", _src(dummy_ctx), _ctx())
    assert ev[0].type == "Attribution.node.bad" and ev[0].payload["symptom"] == "A bad"


def test_not_found_none_emits_nothing(dummy_ctx):
    emit = {"type": "catalog_lookup", "catalog": "device_checks", "key": "check", "not_found": "none"}
    proc = compile_catalog_lookup(emit, rule_name="r", tables=ELEM)
    assert proc({"check": "nope"}, "n", _src(dummy_ctx), _ctx()) == []


def test_derived_rule_template_credits_per_key(dummy_ctx):
    emit = {
        "type": "catalog_lookup",
        "catalog": "device_checks",
        "key": "check",
        "derived_rule": "DeviceCheck/{key}",
        "not_found": "none",
    }
    proc = compile_catalog_lookup(emit, rule_name="r", tables=ELEM)
    ev = proc({"check": "hw_a"}, "n", _src(dummy_ctx), _ctx())
    assert ev[0].source.rule == "DeviceCheck/hw_a"


def test_key_expr_str_coerced(dummy_ctx):
    tables = Tables(record_tables={"xid": {"1": {"nice_name": "XID-1"}}})
    emit = {
        "type": "catalog_lookup",
        "catalog": "xid",
        "key": "int(code)",
        "project": {"type": "interesting_node", "symptom": "{nice_name}"},
        "not_found": "none",
    }
    proc = compile_catalog_lookup(emit, rule_name="r", tables=tables)
    ev = proc({"code": "01"}, "n", _src(dummy_ctx), _ctx())  # int("01")=1 -> "1"
    assert ev[0].payload["symptom"] == "XID-1"


def test_record_table_requires_project():
    tables = Tables(record_tables={"xid": {"1": {"nice_name": "n"}}})
    emit = {"type": "catalog_lookup", "catalog": "xid", "key": "code", "not_found": "none"}
    with pytest.raises(LoadError):
        compile_catalog_lookup(emit, rule_name="r", tables=tables)


def test_element_catalog_forbids_project():
    emit = {
        "type": "catalog_lookup",
        "catalog": "device_checks",
        "key": "check",
        "project": {"type": "bad_node", "symptom": "x"},
        "not_found": "none",
    }
    with pytest.raises(LoadError):
        compile_catalog_lookup(emit, rule_name="r", tables=ELEM)


# --- credit precedence (row own > lookup template > none) ---

_PRECEDENCE = Tables(
    element_catalogs={
        "device_checks": {
            "with_own": {"type": "bad_node", "symptom": "s", "derived_rule": "OWN/rule"},
            "no_own": {"type": "bad_node", "symptom": "s"},
        }
    }
)


def test_row_own_derived_rule_wins_over_template(dummy_ctx):
    emit = {
        "type": "catalog_lookup",
        "catalog": "device_checks",
        "key": "check",
        "derived_rule": "TEMPLATE/{key}",
        "not_found": "none",
    }
    proc = compile_catalog_lookup(emit, rule_name="r", tables=_PRECEDENCE)
    ev = proc({"check": "with_own"}, "n", _src(dummy_ctx), _ctx())
    assert ev[0].source.rule == "OWN/rule"


def test_lookup_template_credits_when_no_own(dummy_ctx):
    emit = {
        "type": "catalog_lookup",
        "catalog": "device_checks",
        "key": "check",
        "derived_rule": "TEMPLATE/{key}",
        "not_found": "none",
    }
    proc = compile_catalog_lookup(emit, rule_name="r", tables=_PRECEDENCE)
    ev = proc({"check": "no_own"}, "n", _src(dummy_ctx), _ctx())
    assert ev[0].source.rule == "TEMPLATE/no_own"


def test_neither_own_nor_template_leaves_rule_unset(dummy_ctx):
    emit = {"type": "catalog_lookup", "catalog": "device_checks", "key": "check", "not_found": "none"}
    proc = compile_catalog_lookup(emit, rule_name="r", tables=_PRECEDENCE)
    ev = proc({"check": "no_own"}, "n", _src(dummy_ctx), _ctx())
    assert not ev[0].source.rule  # stanza-name fallback fills it later


# --- when gate ---


def test_when_false_gates_lookup(dummy_ctx):
    emit = {
        "type": "catalog_lookup",
        "catalog": "device_checks",
        "key": "check",
        "when": "gate == 'on'",
        "not_found": "none",
    }
    proc = compile_catalog_lookup(emit, rule_name="r", tables=ELEM)
    assert proc({"check": "hw_a", "gate": "off"}, "n", _src(dummy_ctx), _ctx()) == []
    ev = proc({"check": "hw_a", "gate": "on"}, "n", _src(dummy_ctx), _ctx())
    assert ev[0].payload["symptom"] == "A bad"


# --- not_found: leaf emits on a miss; "none" emits nothing ---


def test_not_found_leaf_emits_on_miss(dummy_ctx):
    emit = {
        "type": "catalog_lookup",
        "catalog": "device_checks",
        "key": "check",
        "not_found": {"type": "interesting_node", "symptom": "unknown {check}"},
    }
    proc = compile_catalog_lookup(emit, rule_name="r", tables=ELEM)
    ev = proc({"check": "nope"}, "n", _src(dummy_ctx), _ctx())
    assert ev[0].type == "Attribution.node.interesting" and ev[0].payload["symptom"] == "unknown nope"


def test_not_found_combinator_raises_load_error():
    emit = {
        "type": "catalog_lookup",
        "catalog": "device_checks",
        "key": "check",
        "not_found": {
            "type": "pick_one",
            "branches": [{"when": "1", "type": "bad_node", "symptom": "s"}],
            "default": "none",
        },
    }
    with pytest.raises(LoadError):
        compile_catalog_lookup(emit, rule_name="r", tables=ELEM)


# --- LoadErrors ---


def test_unknown_catalog_name_raises():
    emit = {"type": "catalog_lookup", "catalog": "missing", "key": "check", "not_found": "none"}
    with pytest.raises(LoadError):
        compile_catalog_lookup(emit, rule_name="r", tables=ELEM)


def test_malformed_element_row_raises_at_load():
    tables = Tables(element_catalogs={"device_checks": {"bad": {"type": "pick_one"}}})
    emit = {"type": "catalog_lookup", "catalog": "device_checks", "key": "check", "not_found": "none"}
    with pytest.raises(LoadError):  # compile-time, not match-time
        compile_catalog_lookup(emit, rule_name="r", tables=tables)


def test_malformed_record_project_raises_at_load():
    tables = Tables(record_tables={"xid": {"1": {"nice_name": "n"}}})
    emit = {
        "type": "catalog_lookup",
        "catalog": "xid",
        "key": "code",
        "project": {"type": "pick_one"},
        "not_found": "none",
    }
    with pytest.raises(LoadError):
        compile_catalog_lookup(emit, rule_name="r", tables=tables)


# --- record project renders from the RECORD's columns, not line captures ---


def test_record_project_renders_from_record_columns(dummy_ctx):
    tables = Tables(record_tables={"xid": {"48": {"nice_name": "GPU fell off bus", "sev": "high"}}})
    emit = {
        "type": "catalog_lookup",
        "catalog": "xid",
        "key": "code",
        "derived_rule": "XID/{key}",
        "project": {"type": "bad_node", "symptom": "{nice_name} ({sev})"},
        "not_found": "none",
    }
    proc = compile_catalog_lookup(emit, rule_name="r", tables=tables)
    # 'nice_name'/'sev' come from the RECORD, not the line captures.
    ev = proc({"code": "48", "nice_name": "IGNORED"}, "n", _src(dummy_ctx), _ctx())
    assert ev[0].payload["symptom"] == "GPU fell off bus (high)"
    assert ev[0].source.rule == "XID/48"


def test_record_catalog_outer_when_can_gate(dummy_ctx):
    emit = {
        "type": "catalog_lookup",
        "catalog": "x",
        "key": "key",
        "when": "enabled",
        "project": {"type": "bad_node", "symptom": "{value}"},
        "not_found": "none",
    }
    proc = compile_catalog_lookup(
        emit,
        rule_name="r",
        tables=Tables(record_tables={"x": {"a": {"value": "bad"}}}),
    )
    assert proc({"key": "a", "enabled": False}, "node", _src(dummy_ctx), _ctx()) == []


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({}, "required 'key'"),
        ({"key": "key"}, "requires 'not_found'"),
        ({"key": "key", "not_found": "bad"}, "must be a leaf"),
    ],
)
def test_catalog_lookup_rejects_missing_fields(update, message):
    emit = {"type": "catalog_lookup", "catalog": "x", **update}
    with pytest.raises(LoadError, match=message):
        compile_catalog_lookup(emit, rule_name="r", tables=Tables(element_catalogs={"x": {}}))


def test_catalog_expansion_requires_line_template():
    with pytest.raises(LoadError, match="line_template"):
        expand_catalog_cases([{"expansion": "catalog"}], {"a": {}}, "r")


def test_catalog_coverage_skips_nonmatches_and_false_guard():
    emit = {"type": "catalog_lookup", "key": "key", "when": "enabled"}
    cases = [{"kind": "miss"}, {"kind": "guard"}, {"kind": "hit"}]

    def groups(case):
        if case["kind"] == "miss":
            return {}
        return {"key": "a", "enabled": case["kind"] == "hit"}

    assert covered_catalog_obligations(emit, {"a": {}}, cases, match_groups=groups) == {"row:a"}


def test_catalog_cases_and_untested_are_mutually_exclusive():
    with pytest.raises(LoadError, match="mutually exclusive"):
        check_catalog_coverage(
            "r",
            {"type": "catalog_lookup", "key": "key"},
            {"a": {}},
            [{"line": "a"}],
            "waived",
            None,
            match_groups=lambda case: {"key": "a"},
        )


def test_catalog_declarations_collect_generic_event_types():
    rules: set[str] = set()
    event_types: set[str] = set()
    emit = {
        "project": {"type": "generic_event", "event_type": "project.event"},
        "not_found": "none",
    }
    table = {"a": {"type": "generic_event", "event_type": "row.event"}}
    _collect_catalog_declarations(emit, table, True, rules, event_types)
    _collect_catalog_declarations(emit, table, False, rules, event_types)
    assert event_types == {"row.event", "project.event"}


def test_catalog_guard_can_dominate_project_templates():
    emit = {
        "when": "value is not None",
        "key": "value",
        "project": {"type": "bad_node", "symptom": "literal"},
    }
    _check_catalog_guards(emit, {"value"}, where="r")


# --- Part 1: row-complete coverage obligations (loader integration) ---

_THREE_ROW = """
sources = ["SYSLOG"]
[catalog.hc]
alpha = {{ type = "bad_node", symptom = "alpha bad" }}
beta = {{ type = "bad_node", symptom = "beta bad" }}
gamma = {{ type = "interesting_node", symptom = "gamma int" }}

[[extractor]]
name = "HC"
match_re = "HC (?P<check>\\\\w+)"
emit = {{ type = "catalog_lookup", catalog = "hc", key = "check", derived_rule = "HC/{{key}}", not_found = "none" }}
{cases}
"""


def test_partial_row_coverage_names_uncovered_row(tmp_path):
    cases = """
[[extractor.cases]]
line = "HC alpha"
[[extractor.cases]]
line = "HC beta"
"""
    with pytest.raises(LoadError, match="row:gamma"):
        load_toml_rules(RuleLoader(), _write(tmp_path, _THREE_ROW.format(cases=cases)))


def test_all_rows_plus_not_found_covers(tmp_path):
    cases = """
[[extractor.cases]]
line = "HC alpha"
[[extractor.cases]]
line = "HC beta"
[[extractor.cases]]
line = "HC gamma"
[[extractor.cases]]
line = "HC unknown_check"
"""
    added = load_toml_rules(RuleLoader(), _write(tmp_path, _THREE_ROW.format(cases=cases)))
    assert "HC" in added


def test_missing_not_found_case_fails(tmp_path):
    cases = """
[[extractor.cases]]
line = "HC alpha"
[[extractor.cases]]
line = "HC beta"
[[extractor.cases]]
line = "HC gamma"
"""
    with pytest.raises(LoadError, match="not_found"):
        load_toml_rules(RuleLoader(), _write(tmp_path, _THREE_ROW.format(cases=cases)))


def test_untested_rows_waives_a_row(tmp_path):
    toml = """
sources = ["SYSLOG"]
[catalog.hc]
alpha = { type = "bad_node", symptom = "alpha bad" }
beta = { type = "bad_node", symptom = "beta bad" }
gamma = { type = "interesting_node", symptom = "gamma int" }

[[extractor]]
name = "HC"
match_re = "HC (?P<check>\\\\w+)"
emit = { type = "catalog_lookup", catalog = "hc", key = "check", not_found = "none" }
untested_rows = ["gamma"]
[[extractor.cases]]
line = "HC alpha"
[[extractor.cases]]
line = "HC beta"
[[extractor.cases]]
line = "HC unknown_check"
"""
    added = load_toml_rules(RuleLoader(), _write(tmp_path, toml))
    assert "HC" in added


def test_no_cases_no_untested_fails(tmp_path):
    toml = """
sources = ["SYSLOG"]
[catalog.hc]
alpha = { type = "bad_node", symptom = "alpha bad" }

[[extractor]]
name = "HC"
match_re = "HC (?P<check>\\\\w+)"
emit = { type = "catalog_lookup", catalog = "hc", key = "check", not_found = "none" }
"""
    with pytest.raises(LoadError):
        load_toml_rules(RuleLoader(), _write(tmp_path, toml))


# --- Part 2: the `catalog` expansion case form ---


def test_catalog_expansion_covers_all_rows(tmp_path):
    toml = """
sources = ["SYSLOG"]
[catalog.hc]
alpha = { type = "bad_node", symptom = "alpha bad" }
beta = { type = "bad_node", symptom = "beta bad" }
gamma = { type = "interesting_node", symptom = "gamma int" }

[[extractor]]
name = "HC"
match_re = "HC (?P<check>\\\\w+)"
emit = { type = "catalog_lookup", catalog = "hc", key = "check", not_found = "none" }
[[extractor.cases]]
expansion = "catalog"
line_template = "HC {key}"
[[extractor.cases]]
line = "HC unknown_check"
"""
    added = load_toml_rules(RuleLoader(), _write(tmp_path, toml))
    assert "HC" in added


def test_catalog_expansion_missing_not_found_still_fails(tmp_path):
    toml = """
sources = ["SYSLOG"]
[catalog.hc]
alpha = { type = "bad_node", symptom = "alpha bad" }

[[extractor]]
name = "HC"
match_re = "HC (?P<check>\\\\w+)"
emit = { type = "catalog_lookup", catalog = "hc", key = "check", not_found = "none" }
[[extractor.cases]]
expansion = "catalog"
line_template = "HC {key}"
"""
    with pytest.raises(LoadError, match="not_found"):
        load_toml_rules(RuleLoader(), _write(tmp_path, toml))


# --- Part 3: guard-check wired for catalog_lookup in the loader ---


def test_catalog_lookup_unguarded_optional_capture_is_load_error(tmp_path):
    toml = """
sources = ["SYSLOG"]
[catalog.hc]
alpha = { type = "bad_node", symptom = "alpha bad" }

[[extractor]]
name = "HC"
match_re = "HC (?P<check>\\\\w+)(?: at (?P<loc>\\\\S+))?"
emit = { type = "catalog_lookup", catalog = "hc", key = "check", not_found = { type = "interesting_node", symptom = "unknown at {loc}" } }
[[extractor.cases]]
expansion = "catalog"
line_template = "HC {key}"
[[extractor.cases]]
line = "HC nope at x"
"""
    with pytest.raises(LoadError, match="loc"):
        load_toml_rules(RuleLoader(), _write(tmp_path, toml))


# --- Part 4: declare catalog-hit credit names ---


def test_catalog_lookup_declares_rendered_per_key_credits(tmp_path):
    toml = """
sources = ["SYSLOG"]
[catalog.hc]
alpha = { type = "bad_node", symptom = "alpha bad" }
beta = { type = "bad_node", symptom = "beta bad" }

[[extractor]]
name = "HC"
match_re = "HC (?P<check>\\\\w+)"
emit = { type = "catalog_lookup", catalog = "hc", key = "check", derived_rule = "X/{key}", not_found = "none" }
[[extractor.cases]]
expansion = "catalog"
line_template = "HC {key}"
[[extractor.cases]]
line = "HC nope"
"""
    loader = RuleLoader()
    load_toml_rules(loader, _write(tmp_path, toml))
    assert "X/alpha" in loader.declared_rule_names
    assert "X/beta" in loader.declared_rule_names
    assert "X/{key}" not in loader.declared_rule_names


def test_catalog_lookup_declares_row_own_and_not_found(tmp_path):
    toml = """
sources = ["SYSLOG"]
[catalog.hc]
alpha = { type = "bad_node", symptom = "alpha bad", derived_rule = "OWN/alpha" }
beta = { type = "bad_node", symptom = "beta bad" }

[[extractor]]
name = "HC"
match_re = "HC (?P<check>\\\\w+)"
emit = { type = "catalog_lookup", catalog = "hc", key = "check", derived_rule = "X/{key}", not_found = { type = "interesting_node", symptom = "unknown", derived_rule = "MISS/none" } }
[[extractor.cases]]
expansion = "catalog"
line_template = "HC {key}"
[[extractor.cases]]
line = "HC nope"
"""
    loader = RuleLoader()
    load_toml_rules(loader, _write(tmp_path, toml))
    assert "OWN/alpha" in loader.declared_rule_names  # row own wins
    assert "X/beta" in loader.declared_rule_names  # lookup template for no-own row
    assert "MISS/none" in loader.declared_rule_names  # not_found leaf credit
