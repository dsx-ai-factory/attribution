import re

import pytest

from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.catalogs import Tables
from attribution_lib.declarative.coverage import (
    check_family_coverage,
    covered_family_obligations,
    expand_family_cases,
    family_obligations,
)
from attribution_lib.declarative.emit import LoadError
from attribution_lib.declarative.errors import FinalizeError
from attribution_lib.declarative.families import GeneratedRule, _subst_emit, stamp_family, substitute_columns
from attribution_lib.rules_engine import RuleLoader


def test_substitute_columns():
    assert substitute_columns("{key} bad", {"key": "OOM"}, where="r") == "OOM bad"


def test_substitute_missing_column_is_load_error():
    with pytest.raises(LoadError):
        substitute_columns("{nope}", {"key": "x"}, where="r")


FAM = {
    "name": "gk",
    "pattern": "{key}",
    "emit": {"type": "bad_node", "symptom": "{symptom}", "derived_rule": "{rule}"},
    "rows": [
        {"key": "Segmentation fault", "symptom": "Segmentation fault", "rule": "alias else"},
        {"key": "local catastrophic error", "symptom": "local catastrophic error", "rule": "local catastrophic error"},
    ],
}


def test_stamp_family_names_and_match_re():
    gen = stamp_family(FAM, Tables())
    assert {g.name for g in gen} == {"gk:Segmentation fault", "gk:local catastrophic error"}
    g0 = next(g for g in gen if g.name == "gk:Segmentation fault")
    assert g0.match_re == re.escape("Segmentation fault")  # literal
    assert g0.emit == {"type": "bad_node", "symptom": "Segmentation fault", "derived_rule": "alias else"}
    assert "alias else" in g0.credit_names


def test_stamp_family_row_missing_key_is_load_error():
    with pytest.raises(LoadError):
        stamp_family(
            {"name": "x", "pattern": "{key}", "emit": {"type": "bad_node", "symptom": "s"}, "rows": [{"symptom": "s"}]},
            Tables(),
        )


def test_stamp_family_duplicate_generated_name_is_load_error():
    with pytest.raises(LoadError):
        stamp_family(
            {
                "name": "x",
                "pattern": "{key}",
                "emit": {"type": "bad_node", "symptom": "{key}"},
                "rows": [{"key": "a"}, {"key": "a"}],
            },
            Tables(),
        )


@pytest.mark.parametrize(
    ("family", "message"),
    [
        ({"name": "f", "pattern": "x", "emit": {}, "rows": "bad"}, "rows.*list"),
        ({"name": "f", "pattern": "x", "emit": {}, "rows": ["bad"]}, r"rows\[0\].*table"),
        ({"name": "f", "pattern": "x", "emit": {}}, "rows.*rows_from"),
        ({"rows": [], "pattern": "x", "emit": {}}, "requires a 'name'"),
        ({"name": "f", "rows": [], "patterns": [1], "emit": {}}, r"patterns\[0\].*string"),
        ({"name": "f", "rows": [], "patterns": "x", "emit": {}}, "patterns.*list"),
        ({"name": "f", "rows": [], "pattern": 1, "emit": {}}, "pattern.*string"),
        ({"name": "f", "rows": [], "emit": {}}, "pattern.*patterns"),
        ({"name": "f", "rows": [], "pattern": "x"}, "requires 'emit'"),
        ({"name": "f", "rows": [], "pattern": "x", "emit": "bad"}, "emit.*table"),
    ],
)
def test_stamp_family_rejects_malformed_stanzas(family, message):
    with pytest.raises(LoadError, match=message):
        stamp_family(family, Tables())


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ({"k": "bad"}, "entry 'k'.*table"),
        ({"k": {"key": "reserved"}}, "reserved 'key'"),
    ],
)
def test_rows_from_rejects_malformed_records(records, message):
    family = {"name": "f", "pattern": "{key}", "emit": {}, "rows_from": "source"}
    with pytest.raises(LoadError, match=message):
        stamp_family(family, Tables(record_tables={"source": records}))


def test_rows_from_rejects_unknown_table():
    family = {"name": "f", "pattern": "{key}", "emit": {}, "rows_from": "missing"}
    with pytest.raises(LoadError, match="unknown record table"):
        stamp_family(family, Tables())


def test_substitute_emit_recurses_lists_and_preserves_scalars():
    value = {"items": ["{key}", 7]}
    assert _subst_emit(value, {"key": "x"}, where="f") == {"items": ["x", 7]}


def test_family_cases_and_untested_are_mutually_exclusive():
    generated = [GeneratedRule("f:a", "a", {}, row_key="a")]
    with pytest.raises(LoadError, match="mutually exclusive"):
        check_family_coverage("f", generated, [{"line": "a"}], "waived", None)


def _write(tmp_path, text):
    p = tmp_path / "r.toml"
    p.write_text(text)
    return str(p)


def test_load_family_registers_one_rule_per_row(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[family]]
name = "gk"
pattern = "{key}"
emit = { type = "bad_node", symptom = "{symptom}" }
untested = "pilot"

[[family.rows]]
key = "Segmentation fault"
symptom = "Segmentation fault"

[[family.rows]]
key = "OOM"
symptom = "out of memory"
"""
    loader = RuleLoader()
    added = load_toml_rules(loader, _write(tmp_path, toml))
    assert "gk:Segmentation fault" in added
    assert "gk:OOM" in added
    rule = added["gk:Segmentation fault"]
    assert rule.match_re == re.escape("Segmentation fault")
    assert loader.extractor_rules["gk:OOM"] is added["gk:OOM"]


# --- Task 2: patterns (indexed) + rows_from a record table ----------------------


def test_patterns_generate_indexed_names():
    fam = {
        "name": "cuda",
        "patterns": ["Cuda failure '{desc}'", "CUDA error: {desc}"],
        "emit": {"type": "bad_node", "symptom": "{nice}"},
        "rows": [{"key": "e1", "desc": "oom", "nice": "CUDA OOM"}],
    }
    gen = stamp_family(fam, Tables())
    names = {g.name for g in gen}
    assert names == {"cuda:e1:0", "cuda:e1:1"}
    assert {g.match_re for g in gen} == {re.escape("Cuda failure 'oom'"), re.escape("CUDA error: oom")}


def test_rows_from_record_table():
    tables = Tables(record_tables={"xt": {"63": {"nice": "XID 63"}, "94": {"nice": "XID 94"}}})
    fam = {
        "name": "xid",
        "pattern": " XID {key}=",
        "emit": {"type": "bad_node", "symptom": "{nice}"},
        "rows_from": "xt",
    }
    gen = stamp_family(fam, tables)
    assert {g.name for g in gen} == {"xid:63", "xid:94"}
    g = next(x for x in gen if x.name == "xid:63")
    assert g.emit["symptom"] == "XID 63"


# --- Task 3: family row-complete coverage + rows expansion + declarations --------


def test_family_rows_expansion_covers_all_generated_rules(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[family]]
name = "gk"
pattern = "{key}"
emit = { type = "bad_node", symptom = "{symptom}", derived_rule = "{rule}" }

[[family.rows]]
key = "Segmentation fault"
symptom = "Segmentation fault"
rule = "A"

[[family.rows]]
key = "OOM"
symptom = "out of memory"
rule = "B"

[[family.cases]]
expansion = "rows"
expect = "bad"
"""
    loader = RuleLoader()
    added = load_toml_rules(loader, _write(tmp_path, toml))
    assert "gk:Segmentation fault" in added and "gk:OOM" in added


def test_family_missing_cases_and_untested_fails(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[family]]
name = "gk"
pattern = "{key}"
emit = { type = "bad_node", symptom = "{symptom}" }

[[family.rows]]
key = "OOM"
symptom = "out of memory"
"""
    loader = RuleLoader()
    with pytest.raises(LoadError):
        load_toml_rules(loader, _write(tmp_path, toml))


def test_family_untested_rows_waives_a_row(tmp_path):
    # Only "OOM" is exercised by an explicit case; "rare" is waived by untested_rows.
    toml = """
sources = ["SYSLOG"]
[[family]]
name = "gk"
pattern = "{key}"
emit = { type = "bad_node", symptom = "{symptom}" }
untested_rows = ["rare"]

[[family.rows]]
key = "OOM"
symptom = "out of memory"

[[family.rows]]
key = "rare"
symptom = "rare thing"

[[family.cases]]
line = "OOM"
expect = "bad"
"""
    loader = RuleLoader()
    added = load_toml_rules(loader, _write(tmp_path, toml))
    assert "gk:OOM" in added and "gk:rare" in added


def test_family_untested_rows_still_requires_others(tmp_path):
    # Waiving "rare" does not waive "OOM"; with no case covering it, the load fails.
    toml = """
sources = ["SYSLOG"]
[[family]]
name = "gk"
pattern = "{key}"
emit = { type = "bad_node", symptom = "{symptom}" }
untested_rows = ["rare"]

[[family.rows]]
key = "OOM"
symptom = "out of memory"

[[family.rows]]
key = "rare"
symptom = "rare thing"

[[family.cases]]
line = "nomatch"
expect = "none"
"""
    loader = RuleLoader()
    with pytest.raises(LoadError):
        load_toml_rules(loader, _write(tmp_path, toml))


def test_family_declares_generated_credit_names(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[family]]
name = "gk"
pattern = "{key}"
emit = { type = "bad_node", symptom = "{symptom}", derived_rule = "{rule}" }

[[family.rows]]
key = "seg"
symptom = "Segmentation fault"
rule = "A"

[[family.rows]]
key = "oom"
symptom = "out of memory"
rule = "B"

[[family.cases]]
expansion = "rows"
expect = "bad"
"""
    loader = RuleLoader()
    load_toml_rules(loader, _write(tmp_path, toml))
    assert "A" in loader.declared_rule_names
    assert "B" in loader.declared_rule_names


def test_family_tags_merge_with_file_tags_on_every_generated_row(tmp_path):
    toml = """
sources = ["SYSLOG"]
tags = ["oom"]

[[extractor]]
name = "sibling"
match_re = "sibling"
emit = { type = "bad_node", symptom = "sibling" }
untested = "validation"

[[family]]
name = "gk"
pattern = "{key}"
emit = { type = "bad_node", symptom = "{symptom}" }
untested = "validation"
tags = ["external"]

[[family.rows]]
key = "seg"
symptom = "Segmentation fault"

[[family.rows]]
key = "oom"
symptom = "out of memory"

[[family.rows]]
key = "rare"
symptom = "rare thing"
"""
    added = load_toml_rules(RuleLoader(), _write(tmp_path, toml))
    assert len(added) == 4
    # Every generated row carries both layers; the sibling gets the file layer alone.
    assert {rule.tags for rule in added.values()} == {("external", "oom"), ("oom",)}


def test_family_tags_conflicting_with_file_tags_are_rejected_naming_the_file(tmp_path):
    toml = """
sources = ["SYSLOG"]
tags = ["internal"]

[[family]]
name = "gk"
pattern = "{key}"
emit = { type = "bad_node", symptom = "{symptom}" }
untested = "validation"
tags = ["external"]

[[family.rows]]
key = "oom"
symptom = "out of memory"
"""
    path = _write(tmp_path, toml)
    with pytest.raises(FinalizeError, match=re.escape(path)):
        load_toml_rules(RuleLoader(), path)


def test_expand_family_cases_preserves_per_case_data():
    # expansion="rows" must carry forward all per-case annotations (expect,
    # lifecycle booleans, etc.) and only override `line` with the generated line.
    generated = [
        GeneratedRule("f:a", "line_a", {}, row_key="a"),
        GeneratedRule("f:b", "line_b", {}, row_key="b"),
    ]
    cases = [{"expansion": "rows", "expect": "bad", "at_end_of_job": True}]
    result = expand_family_cases(cases, generated)
    assert len(result) == 2
    for expanded, g in zip(result, generated):
        assert expanded["line"] == g.line
        assert expanded["expect"] == "bad"
        assert expanded["at_end_of_job"] is True
        assert expanded["expansion"] == "rows"


def test_family_rows_expansion_skips_redundant_regex_checks(monkeypatch):
    """Skip regex searches when row expansion guarantees family coverage.

    Args:
        monkeypatch: Pytest fixture used to reject unexpected regex searches.
    """
    generated = stamp_family(FAM, Tables())
    cases = expand_family_cases([{"expansion": "rows", "expect": "bad"}], generated)

    def fail_search(*_args, **_kwargs):
        """Fail when row-expanded coverage performs a regex search."""
        raise AssertionError("rows expansion must not re-check every generated regex")

    monkeypatch.setattr("attribution_lib.declarative.coverage.re.search", fail_search)

    assert covered_family_obligations(generated, cases) == family_obligations(generated)
