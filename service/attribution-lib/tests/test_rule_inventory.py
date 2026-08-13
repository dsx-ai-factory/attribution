# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    """Load the standalone inventory script as an importable test module."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "rule_inventory.py"
    spec = importlib.util.spec_from_file_location("rule_inventory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = _load_module()


def _rule(name, match_re="match", *, kind="extractor", sources=("syslog",), schema=None, details=None):
    """Build a compact versioned rule record for comparison tests."""
    rule = {
        "kind": kind,
        "name": name,
        "author": "test",
        "location": {"file": "rules.py", "line": 1},
    }
    data = {}
    if kind in {"extractor", "extractor_accumulator"}:
        data.update(match_re=match_re, sources=list(sources))
    elif kind in {"filter", "refiner"}:
        data["priority"] = 1000
    elif kind == "collector":
        data["params_model"] = None
    if details is not None:
        data.update(details)
    rule["definition"] = {
        "schema": inventory._python_schema(kind) if schema is None else schema,
        "data": data,
    }
    return rule


def _snapshot(*rules, discovered=("rules.py",)):
    """Build a minimal valid inventory envelope around test rule records."""
    return {
        "schema_name": inventory.SCHEMA_NAME,
        "schema_version": 1,
        "artifacts": {"rules_dirs": ["/rules"], "discovered": list(discovered)},
        "rules": list(rules),
        "declared_rule_names": ["credited"],
        "declared_event_types": ["Attribution.node.bad"],
    }


def test_identical_inventory_passes():
    """Identical inventories pair every rule without differences."""
    snapshot = _snapshot(_rule("R"), _rule("F", kind="filter"))
    result = inventory.compare_inventories(snapshot, snapshot)
    assert result["safe"] is True
    assert result["counts"] == {
        "unchanged": 2,
        "renamed": 0,
        "approved_change": 0,
        "unapproved_change": 0,
    }


@pytest.mark.parametrize(
    "field",
    ["artifacts", "rules", "declared_rule_names", "declared_event_types"],
)
def test_missing_required_inventory_field_fails(field):
    """A current-schema inventory cannot omit a required collection."""
    snapshot = _snapshot(_rule("R"))
    del snapshot[field]
    with pytest.raises(ValueError, match="missing required fields"):
        inventory.compare_inventories(snapshot, _snapshot(_rule("R")))


@pytest.mark.parametrize("field", ["rules_dirs", "discovered"])
def test_missing_required_artifact_field_fails(field):
    """Artifact metadata cannot silently omit a required collection."""
    snapshot = _snapshot(_rule("R"))
    del snapshot["artifacts"][field]
    with pytest.raises(ValueError, match="artifacts are missing required fields"):
        inventory.compare_inventories(snapshot, _snapshot(_rule("R")))


def test_matcher_equivalent_rename_is_automatic():
    """An extractor rename with the same runtime signature pairs automatically."""
    result = inventory.compare_inventories(_snapshot(_rule("Old")), _snapshot(_rule("New")))
    assert result["safe"] is True
    assert result["counts"]["renamed"] == 1
    assert "`extractor:Old` → `New`" in inventory.render_report(result)


def test_removed_rule_fails_even_when_total_count_is_unchanged():
    """A replacement cannot hide a missing reference rule through equal totals."""
    base = _snapshot(_rule("Kept"), _rule("Dropped", "old"))
    current = _snapshot(_rule("Kept"), _rule("Replacement", "new"))
    result = inventory.compare_inventories(base, current)
    assert result["safe"] is False
    assert {item["name"] for item in result["unmatched"]} == {"Dropped"}
    assert {item["name"] for item in result["additions"]} == {"Replacement"}
    report = inventory.render_report(result)
    assert "## Missing from candidate" in report
    assert "`extractor:Dropped`" in report
    assert "## Candidate additions" in report
    assert "`extractor:Replacement`" in report


def test_non_extractor_rename_requires_explicit_mapping():
    """Non-extractor renames remain explicit because their signatures are weak."""
    base = _snapshot(_rule("OldFilter", kind="filter"))
    current = _snapshot(_rule("NewFilter", kind="filter"))
    result = inventory.compare_inventories(base, current)
    assert result["safe"] is False
    assert result["counts"]["renamed"] == 0
    assert {item["name"] for item in result["unmatched"]} == {"OldFilter"}


def test_changed_matcher_requires_explicit_mapping():
    """A matcher change fails until an explicit reviewed mapping approves it."""
    base = _snapshot(_rule("R", "old"))
    current = _snapshot(_rule("R", "new"))
    unapproved = inventory.compare_inventories(base, current)
    assert unapproved["safe"] is False
    assert unapproved["counts"]["unapproved_change"] == 1
    assert unapproved["unmatched"] == []
    assert unapproved["additions"] == []

    result = inventory.compare_inventories(
        base,
        current,
        [{"kind": "extractor", "reference": "R", "candidate": "R", "reason": "intentional expansion"}],
    )
    assert result["safe"] is True
    assert result["counts"]["approved_change"] == 1


def test_python_to_declarative_uses_common_runtime_projection():
    """Python and declarative extractors negotiate their shared runtime schema."""
    base = _snapshot(_rule("R"))
    current = _snapshot(
        _rule(
            "R",
            schema="extractor/declarative/v1",
            details={"form": "emit", "emit": {"type": "bad_node"}},
        )
    )
    result = inventory.compare_inventories(base, current)
    assert result["safe"] is True
    assert result["definition_changes"] == []
    assert result["schema_transitions"] == [
        {
            "kind": "extractor",
            "reference": "R",
            "candidate": "R",
            "reference_schema": "extractor/python/v1",
            "candidate_schema": "extractor/declarative/v1",
            "projection": "extractor/runtime/v1",
        }
    ]


def test_same_declarative_schema_compares_native_fields():
    """Rules sharing a declarative schema compare their full native payloads."""
    base = _snapshot(
        _rule("R", schema="extractor/declarative/v1", details={"form": "emit", "emit": {"type": "bad_node"}})
    )
    current = _snapshot(
        _rule(
            "R",
            schema="extractor/declarative/v1",
            details={"form": "emit", "emit": {"type": "interesting_node"}},
        )
    )
    result = inventory.compare_inventories(base, current)
    assert result["safe"] is False
    assert [item["field"] for item in result["definition_changes"]] == ["emit"]
    assert result["definition_changes"][0]["projection"] == "extractor/declarative/v1"

    approved = inventory.compare_inventories(
        base,
        current,
        [{"kind": "extractor", "reference": "R", "candidate": "R", "reason": "intentional emit change"}],
    )
    assert approved["safe"] is True
    assert approved["definition_changes"][0]["approved"] is True


def test_added_tag_is_reported_without_failing_the_comparison():
    """A tag edit is reported on its own and still passes: classification is not a gate."""
    base = _snapshot(_rule("R") | {"tags": []})
    current = _snapshot(_rule("R") | {"tags": ["external"]})
    result = inventory.compare_inventories(base, current)
    assert result["safe"] is True
    assert result["errors"] == []
    assert result["definition_changes"] == []
    assert result["tag_changes"] == [
        {
            "kind": "extractor",
            "reference": "R",
            "candidate": "R",
            "reference_tags": [],
            "candidate_tags": ["external"],
        }
    ]
    assert "`extractor:R`: `[]` → `['external']`" in inventory.render_report(result)


def test_absent_tags_field_matches_an_empty_tag_list():
    """An inventory captured before tags existed compares clean against an untagged one."""
    result = inventory.compare_inventories(_snapshot(_rule("R")), _snapshot(_rule("R") | {"tags": []}))
    assert result["safe"] is True
    assert result["tag_changes"] == []
    assert result["counts"]["unchanged"] == 1


def test_declarative_processor_gets_its_own_rule_schema():
    """A declarative processor descriptor produces a declarative rule definition."""
    definition = inventory._definition(
        "extractor",
        {"match_re": "xid", "sources": ["syslog"]},
        '{"processor":"syslog:xid","derived_rules":["known-xid"],"derived_event_types":["bad"]}',
    )
    assert definition["schema"] == "extractor/declarative/v1"
    assert definition["data"]["form"] == "processor"
    assert definition["data"]["processor"] == "syslog:xid"
    assert definition["data"]["derived_rules"] == ["known-xid"]


def test_unknown_inventory_envelope_version_is_rejected():
    """An unsupported inventory envelope version fails validation."""
    snapshot = _snapshot(_rule("R"))
    snapshot["schema_version"] = 99
    with pytest.raises(ValueError, match="unsupported inventory schema version"):
        inventory.compare_inventories(snapshot, _snapshot(_rule("R")))


def test_unrelated_rule_schemas_without_common_projection_fail():
    """Rule definitions with no common projection are reported as incomparable."""
    base = _snapshot(_rule("R", schema="extractor/vendor-a/v1"))
    current = _snapshot(_rule("R", schema="extractor/vendor-b/v1"))
    result = inventory.compare_inventories(base, current)
    assert result["safe"] is False
    assert len(result["incomparable"]) == 1
    assert "no common projection" in " ".join(result["errors"])


def test_rule_schema_can_project_through_an_intermediate_version(monkeypatch):
    """Projection negotiation can traverse intermediate rule-schema versions."""

    def v2_to_v1(_kind, data):
        """Discard the candidate-only field while projecting custom v2 to v1."""
        return {key: value for key, value in data.items() if key != "new_field"}

    monkeypatch.setitem(
        inventory.RULE_PROJECTORS,
        ("extractor/custom/v2", "extractor/custom/v1"),
        v2_to_v1,
    )
    monkeypatch.setitem(
        inventory.RULE_PROJECTORS,
        ("extractor/custom/v1", "extractor/runtime/v1"),
        inventory._runtime_data,
    )
    reference = _snapshot(_rule("R", schema="extractor/custom/v1", details={"setting": "stable"}))
    candidate = _snapshot(
        _rule("R", schema="extractor/custom/v2", details={"setting": "stable", "new_field": "candidate-only"})
    )

    result = inventory.compare_inventories(reference, candidate)
    assert result["safe"] is True
    assert result["schema_transitions"][0]["projection"] == "extractor/custom/v1"
    assert result["definition_changes"] == []


@pytest.mark.parametrize("field", ["declared_rule_names", "declared_event_types"])
def test_dropped_declaration_fails(field):
    """Dropping a declared rule name or event type fails the comparison."""
    base = _snapshot(_rule("R"))
    current = _snapshot(_rule("R"))
    current[field] = []
    result = inventory.compare_inventories(base, current)
    assert result["safe"] is False
    assert any(f"dropped {field}" in error for error in result["errors"]) is True


def test_change_file_requires_a_reason(tmp_path):
    """Explicit mappings require a nonempty review reason."""
    path = tmp_path / "changes.toml"
    path.write_text('[[change]]\nkind="extractor"\nreference="A"\ncandidate="B"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        inventory.load_changes(path)


def test_dump_inventory_accepts_multiple_rule_directories(tmp_path):
    """Dump loads repeated rule directories and disambiguates duplicate filenames."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    template = """\
[[extractor]]
name = {name!r}
match_re = {pattern!r}
untested = true
[extractor.emit]
type = "bad_node"
symptom = "TestSymptom"
"""
    (first / "rules.toml").write_text(template.format(name="First", pattern="first"), encoding="utf-8")
    (second / "rules.toml").write_text(template.format(name="Second", pattern="second"), encoding="utf-8")

    try:
        result = inventory.dump_inventory([first, second])
    finally:
        inventory.setup_rules(force=True)

    assert result["artifacts"]["rules_dirs"] == [first.as_posix(), second.as_posix()]
    assert result["artifacts"]["discovered"] == ["0:rules.toml", "1:rules.toml"]
    assert {(rule["name"], rule["location"]["file"]) for rule in result["rules"]} == {
        ("First", "0:rules.toml"),
        ("Second", "1:rules.toml"),
    }


def test_dumped_records_carry_declared_tags_and_stay_empty_when_untagged(tmp_path):
    """Effective tags reach the record top level, defaulting to an empty list."""
    (tmp_path / "rules.toml").write_text(
        """\
[[extractor]]
name = "Tagged"
match_re = "tagged"
tags = ["external"]
untested = true
[extractor.emit]
type = "bad_node"
symptom = "TestSymptom"

[[extractor]]
name = "Untagged"
match_re = "untagged"
untested = true
[extractor.emit]
type = "bad_node"
symptom = "TestSymptom"
""",
        encoding="utf-8",
    )

    try:
        result = inventory.dump_inventory([tmp_path])
    finally:
        inventory.setup_rules(force=True)

    assert {rule["name"]: rule["tags"] for rule in result["rules"]} == {"Tagged": ["external"], "Untagged": []}


def test_dump_inventory_rejects_missing_rule_directory(tmp_path):
    """Dump rejects a rule-directory override that does not exist."""
    with pytest.raises(ValueError, match="does not exist"):
        inventory.dump_inventory([tmp_path / "missing"])
