import re

import pytest
from simpleeval import NameNotDefined

from attribution_lib.declarative import load_toml_rules
from attribution_lib.declarative.coverage import (
    check_coverage,
    covered_obligations,
    leaf_obligations,
)
from attribution_lib.declarative.emit import LoadError
from attribution_lib.rules_engine import RuleEngine

# --- leaf_obligations -----------------------------------------------------------


def test_leaf_with_when_needs_true_and_false():
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(value) > 50.0"}
    assert leaf_obligations(emit) == {"when:true", "when:false"}


def test_leaf_without_when_needs_emit():
    emit = {"type": "interesting_node", "symptom": "x"}
    assert leaf_obligations(emit) == {"emit"}


# --- check_coverage (no match_groups: coverage inferred from `expect`) ----------


def test_missing_false_case_fails_coverage():
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(value) > 50.0"}
    cases = [{"line": "value 60", "expect": "interesting"}]  # only true side
    with pytest.raises(LoadError):
        check_coverage("R", emit, cases, untested=None)


def test_fully_covered_leaf_passes():
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(value) > 50.0"}
    cases = [
        {"line": "value 60", "expect": "interesting"},  # true side
        {"line": "value 10", "expect": "none"},  # false side
    ]
    check_coverage("R", emit, cases, untested=None)  # must not raise


def test_no_when_leaf_needs_one_case():
    emit = {"type": "bad_node", "symptom": "boom"}
    check_coverage("R", emit, [{"line": "boom", "expect": "bad"}], untested=None)


def test_untested_and_cases_mutually_exclusive():
    emit = {"type": "bad_node", "symptom": "boom"}
    cases = [{"line": "boom", "expect": "bad"}]
    with pytest.raises(LoadError):
        check_coverage("R", emit, cases, untested="flaky pattern")


def test_no_cases_and_no_untested_fails():
    emit = {"type": "bad_node", "symptom": "boom"}
    with pytest.raises(LoadError):
        check_coverage("R", emit, [], untested=None)


def test_untested_waives_coverage():
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(value) > 50.0"}
    check_coverage("R", emit, [], untested="hard to reproduce")  # must not raise


# --- covered_obligations with a real match_groups (loader-style) ----------------


def test_covered_obligations_evaluates_when_against_captures():
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(value) > 50.0"}
    compiled = re.compile(r"value (?P<value>\d+)")

    def match_groups(case):
        m = compiled.search(case["line"])
        return m.groupdict() if m else {}

    cases = [
        {"line": "value 60"},  # when true
        {"line": "value 10"},  # when false
    ]
    assert covered_obligations(emit, cases, match_groups=match_groups) == {
        "when:true",
        "when:false",
    }


def test_covered_obligations_nonmatching_line_is_false_side():
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(value) > 50.0"}
    compiled = re.compile(r"value (?P<value>\d+)")

    def match_groups(case):
        m = compiled.search(case["line"])
        return m.groupdict() if m else {}

    # A line that does not match => no captures => treated as the false side.
    assert covered_obligations(emit, [{"line": "nothing here"}], match_groups=match_groups) == {"when:false"}


# --- machine-verified side + expect cross-check ---------------------------------


def _matcher(pattern):
    compiled = re.compile(pattern)

    def match_groups(case):
        m = compiled.search(case["line"])
        return m.groupdict() if m else {}

    return match_groups


def test_mislabeled_none_case_whose_when_is_true_is_rejected():
    # Both lines make `when` TRUE, but the second is mislabeled expect="none".
    # The load must fail: the false side is author-asserted, not machine-verified.
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(value) > 50.0"}
    cases = [
        {"line": "value 60", "expect": "interesting"},  # true, correctly labeled
        {"line": "value 70", "expect": "none"},  # true, but mislabeled no-emit
    ]
    with pytest.raises(LoadError, match="mislabeled"):
        check_coverage("R", emit, cases, untested=None, match_groups=_matcher(r"value (?P<value>\d+)"))


def test_mislabeled_emit_case_whose_when_is_false_is_rejected():
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(value) > 50.0"}
    cases = [
        {"line": "value 60", "expect": "interesting"},  # true
        {"line": "value 10", "expect": "interesting"},  # false, but labeled emit
    ]
    with pytest.raises(LoadError, match="mislabeled"):
        check_coverage("R", emit, cases, untested=None, match_groups=_matcher(r"value (?P<value>\d+)"))


def test_correctly_labeled_true_and_false_cases_pass():
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(value) > 50.0"}
    cases = [
        {"line": "value 60", "expect": "interesting"},  # matches, when true
        {"line": "value 10", "expect": "none"},  # matches, when false
    ]
    check_coverage("R", emit, cases, untested=None, match_groups=_matcher(r"value (?P<value>\d+)"))


def test_nonmatching_line_labeled_none_covers_false_side():
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(value) > 50.0"}
    cases = [
        {"line": "value 60", "expect": "interesting"},  # true
        {"line": "no number here", "expect": "no_match"},  # line doesn't match => false side
    ]
    check_coverage("R", emit, cases, untested=None, match_groups=_matcher(r"value (?P<value>\d+)"))


def test_malformed_when_unknown_name_surfaces_loudly():
    # `bogus` is not a capture; on a MATCHED line all captures are present, so the
    # missing name is a genuine rule error and must not collapse to when:false.
    emit = {"type": "interesting_node", "symptom": "x", "when": "float(bogus) > 50.0"}
    cases = [{"line": "value 60", "expect": "interesting"}]
    with pytest.raises(NameNotDefined):
        check_coverage("R", emit, cases, untested=None, match_groups=_matcher(r"value (?P<value>\d+)"))


# --- loader integration: coverage enforced at load ------------------------------


def _write(tmp_path, text):
    p = tmp_path / "r.toml"
    p.write_text(text)
    return str(p)


def test_loader_enforces_coverage_when_branch_missing(tmp_path):
    # A leaf with `when` but only a true-side case must FAIL the load.
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "HighValue"
match_re = "value (?P<value>\\\\d+)"
emit = { type = "interesting_node", symptom = "value {value}", when = "float(value) > 50.0" }
[[extractor.cases]]
line = "value 60"
expect = "interesting"
"""
    with pytest.raises(LoadError):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))


def test_loader_accepts_branch_complete_leaf(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "HighValue"
match_re = "value (?P<value>\\\\d+)"
emit = { type = "interesting_node", symptom = "value {value}", when = "float(value) > 50.0" }
[[extractor.cases]]
line = "value 60"
expect = "interesting"
[[extractor.cases]]
line = "value 10"
expect = "none"
"""
    added = load_toml_rules(RuleEngine(), _write(tmp_path, toml))
    assert "HighValue" in added


def test_loader_untested_waiver_loads(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "boom"
emit = { type = "bad_node", symptom = "boom" }
untested = "covered by integration tests"
"""
    added = load_toml_rules(RuleEngine(), _write(tmp_path, toml))
    assert "R" in added


def test_loader_no_cases_no_untested_fails(tmp_path):
    toml = """
sources = ["SYSLOG"]
[[extractor]]
name = "R"
match_re = "boom"
emit = { type = "bad_node", symptom = "boom" }
"""
    with pytest.raises(LoadError):
        load_toml_rules(RuleEngine(), _write(tmp_path, toml))
