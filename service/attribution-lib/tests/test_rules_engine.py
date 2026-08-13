# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob
import os
from typing import Optional

import pytest

import attribution_lib
from attribution_lib.events import EventManager
from attribution_lib.exceptions import InvalidRuleParams, RuleAcceptsNoParamsError
from attribution_lib.rules_engine import (
    CascadeScope,
    CollectorRule,
    DuplicateRuleError,
    FilterPriority,
    FilterRule,
    OutcomePriority,
    RuleEngine,
    RuleIngestionError,
    RuleLocation,
    RuleParameters,
    RuleParamsBase,
    SymptomCategory,
)


def test_symptom_declarations_register_during_load(tmp_path):
    rule_file = tmp_path / "symptom_rules.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules_engine import CascadeScope, OutcomePriority, SymptomCategory, cascade_symptom, outcome_symptom",
                "",
                "outcome_symptom(",
                "    'bus error',",
                "    outcome='Fail MultiCategory',",
                "    category=SymptomCategory.CSP_EVENT,",
                "    priority=OutcomePriority.LOW_USER,",
                "    doc='bus-error',",
                "    cascade_discards=('PyTorch NCCL watchdog',),",
                ")",
                "cascade_symptom(",
                "    'bus error',",
                "    cascade_discards=('XID-145',),",
                "    scope=CascadeScope.NVL_DOMAIN,",
                ")",
                "cascade_symptom('Python bug', cascade_discards=('IB Error-12',))",
            ]
        )
    )

    engine = RuleEngine()
    engine.load(str(rule_file))

    assert len(engine.symptom_declarations) == 3
    outcome_decl = engine.symptom_declarations[0]
    assert outcome_decl.prefix == "bus error"
    assert outcome_decl.outcome == "Fail MultiCategory"
    assert outcome_decl.priority == OutcomePriority.LOW_USER
    assert outcome_decl.category is SymptomCategory.CSP_EVENT
    assert outcome_decl.doc == "bus-error"
    assert outcome_decl.cascade_discards == ("PyTorch NCCL watchdog",)
    assert outcome_decl.scope is CascadeScope.NODE
    assert outcome_decl.location is not None

    domain_decl = engine.symptom_declarations[1]
    assert domain_decl.prefix == "bus error"
    assert domain_decl.cascade_discards == ("XID-145",)
    assert domain_decl.scope is CascadeScope.NVL_DOMAIN

    cascade_decl = engine.symptom_declarations[2]
    assert cascade_decl.prefix == "Python bug"
    assert cascade_decl.outcome is None
    assert cascade_decl.priority is None
    assert cascade_decl.doc is None
    assert cascade_decl.cascade_discards == ("IB Error-12",)


def test_outcome_symptom_requires_doc_anchor(tmp_path):
    rule_file = tmp_path / "bad_symptom_rules.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules_engine import OutcomePriority, outcome_symptom",
                "",
                "outcome_symptom(",
                "    'bus error',",
                "    outcome='Fail MultiCategory',",
                "    priority=OutcomePriority.LOW_USER,",
                "    doc='',",
                ")",
            ]
        )
    )

    engine = RuleEngine()
    with pytest.raises(RuleIngestionError, match="requires a non-empty doc anchor"):
        engine.load(str(rule_file))


def test_known_symptom_registers_vocabulary_only_declaration(tmp_path):
    rule_file = tmp_path / "known_symptom_rules.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules_engine import known_symptom",
                "",
                "known_symptom('BrokenPipeError')",
            ]
        )
    )

    engine = RuleEngine()
    engine.load(str(rule_file))

    assert len(engine.symptom_declarations) == 1
    declaration = engine.symptom_declarations[0]
    assert declaration.prefix == "BrokenPipeError"
    assert declaration.outcome is None
    assert declaration.priority is None
    assert declaration.doc is None
    assert declaration.cascade_discards == ()
    assert declaration.location is not None


def test_duplicate_identical_symptom_declaration_is_ignored(tmp_path):
    rule_file = tmp_path / "duplicate_symptom_rules.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules_engine import OutcomePriority, outcome_symptom",
                "",
                "outcome_symptom(",
                "    'bus error',",
                "    outcome='Fail MultiCategory',",
                "    priority=OutcomePriority.LOW_USER,",
                "    doc='bus-error',",
                ")",
                "outcome_symptom(",
                "    'bus error',",
                "    outcome='Fail MultiCategory',",
                "    priority=OutcomePriority.LOW_USER,",
                "    doc='bus-error',",
                ")",
            ]
        )
    )

    engine = RuleEngine()
    engine.load(str(rule_file))

    assert len(engine.symptom_declarations) == 1


def test_conflicting_symptom_declaration_raises(tmp_path):
    rule_file = tmp_path / "conflicting_symptom_rules.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules_engine import OutcomePriority, outcome_symptom",
                "",
                "outcome_symptom(",
                "    'bus error',",
                "    outcome='Fail MultiCategory',",
                "    priority=OutcomePriority.LOW_USER,",
                "    doc='bus-error',",
                ")",
                "outcome_symptom(",
                "    'bus error',",
                "    outcome='Fail User',",
                "    priority=OutcomePriority.LOW_USER,",
                "    doc='bus-error',",
                ")",
            ]
        )
    )

    engine = RuleEngine()
    with pytest.raises(RuleIngestionError, match="Conflicting symptom declaration"):
        engine.load(str(rule_file))
    assert engine.symptom_declarations == []


def test_failed_load_restores_symptom_predicates(tmp_path):
    """Verify a rejected rule module cannot leak predicate registrations.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """

    rule_file = tmp_path / "failing_predicate_rules.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules_engine import symptom_predicate",
                "",
                "@symptom_predicate",
                "def leaked_predicate(symptoms, job_info):",
                "    return True",
                "",
                "raise RuntimeError('load failed')",
            ]
        )
    )

    engine = RuleEngine()
    with pytest.raises(RuntimeError, match="load failed"):
        engine.load(str(rule_file))

    assert engine.symptom_predicates == {}


# Filter priority ordering tests
class TestFilterPriorityOrdering:
    """Tests for the filter rule priority/ordering system."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.execution_log: list[str] = []
        self.mgr = EventManager(RuleEngine())
        yield
        self.mgr.close()

    def _make_rule(self, name: str, priority: int = FilterPriority.STANDARD) -> FilterRule:
        return FilterRule(
            name=name,
            author="test",
            location=RuleLocation("test.py", 0),
            processor=lambda ctx, n=name: self.execution_log.append(n),
            priority=priority,
        )

    def _run_filters(self, engine: RuleEngine, **kwargs) -> list[str]:
        stream = self.mgr.create_stream("test")
        engine.process_event_filters(stream, None, **kwargs)
        return list(self.execution_log)

    def test_default_priority_is_standard(self):
        rule = self._make_rule("R")
        assert rule.priority == FilterPriority.STANDARD

    def test_priority_ordering(self):
        """Rules registered in arbitrary order execute sorted by priority.

        phase=X runs only filters at that priority; phase=None runs all phases.
        """
        engine = RuleEngine()
        engine.register_rule(self._make_rule("Late", FilterPriority.LATE))
        engine.register_rule(self._make_rule("Mid", FilterPriority.MID))
        engine.register_rule(self._make_rule("Std", FilterPriority.STANDARD))
        engine.register_rule(self._make_rule("Early", FilterPriority.EARLY))

        assert self._run_filters(engine) == ["Early", "Mid", "Std", "Late"]

        self.execution_log.clear()
        assert self._run_filters(engine, phase=FilterPriority.MID) == ["Mid"]

    def test_stable_tiebreaking(self):
        """Rules with the same priority execute in registration order."""
        engine = RuleEngine()
        engine.register_rule(self._make_rule("A"))
        engine.register_rule(self._make_rule("B"))
        engine.register_rule(self._make_rule("C"))

        assert self._run_filters(engine) == ["A", "B", "C"]

    def test_mixed_priorities_and_tiebreaking(self):
        """Within the same priority, registration order wins."""
        engine = RuleEngine()
        engine.register_rule(self._make_rule("S1", FilterPriority.STANDARD))
        engine.register_rule(self._make_rule("M1", FilterPriority.MID))
        engine.register_rule(self._make_rule("S2", FilterPriority.STANDARD))
        engine.register_rule(self._make_rule("M2", FilterPriority.MID))

        assert self._run_filters(engine) == ["M1", "M2", "S1", "S2"]

        self.execution_log.clear()
        assert self._run_filters(engine, phase=FilterPriority.MID) == ["M1", "M2"]

    def test_duplicate_filter_rule_raises(self):
        engine = RuleEngine()
        engine.register_rule(self._make_rule("Dup"))

        with pytest.raises(DuplicateRuleError, match="Duplicate rule 'Dup'"):
            engine.register_rule(self._make_rule("Dup", FilterPriority.MID))

    def test_backward_compatible_default_priority(self):
        """Existing rules without explicit priority preserve registration order."""
        engine = RuleEngine()
        for name in ["DemoteTflops", "DemoteWatchdog", "DemoteDataloader", "ProcessCompletes"]:
            engine.register_rule(self._make_rule(name))

        assert self._run_filters(engine) == ["DemoteTflops", "DemoteWatchdog", "DemoteDataloader", "ProcessCompletes"]


class _SampleParams(RuleParamsBase):
    value: int = 0


def _collector(name: str, params_model: Optional[type[RuleParamsBase]]) -> CollectorRule:
    return CollectorRule(
        name=name,
        author="test",
        location=RuleLocation("test_rules_engine.py", 0),
        processor=lambda _es, _ji, _p: {},
        formatter=lambda raw: raw,
        params_model=params_model,
    )


class TestParseCollectorParams:
    """Behavior of ``parse_collector_params`` across model/no-model and valid/invalid input."""

    def test_no_model_rejects_supplied_params(self):
        """Passing params for a rule with no params_model raises."""
        with pytest.raises(RuleAcceptsNoParamsError) as exc:
            RuleEngine.parse_collector_params(_collector("NoModel", None), RuleParameters(params={"value": 1}))
        assert exc.value.rule_id == "NoModel"

    def test_no_model_accepts_empty_envelope(self):
        """Empty envelope is fine. Callers often pass {} for all rules indiscriminately."""
        assert RuleEngine.parse_collector_params(_collector("NoModel", None), RuleParameters()) is None

    def test_model_parses_valid_params(self):
        out = RuleEngine.parse_collector_params(
            _collector("WithModel", _SampleParams), RuleParameters(params={"value": 7})
        )
        assert isinstance(out, _SampleParams) and out.value == 7

    def test_model_rejects_invalid_params(self):
        with pytest.raises(InvalidRuleParams) as exc:
            RuleEngine.parse_collector_params(
                _collector("WithModel", _SampleParams), RuleParameters(params={"value": "nope"})
            )
        assert exc.value.rule_id == "WithModel"


def test_no_bundled_rule_resolves_to_engine_module():
    """Sweep: no registered bundled rule should have its location pointing at
    rules_engine.py (the registration machinery)."""

    d = os.path.join(os.path.dirname(attribution_lib.__file__), "bundled_rules")
    engine = RuleEngine()
    for f in sorted(glob.glob("*.py", root_dir=d)):
        engine.load(os.path.join(d, f))
    offenders = [
        (name, r.location.file)
        for name, r in engine.rules.items()
        if r.location and os.path.basename(r.location.file) == "rules_engine.py"
    ]
    assert not offenders, f"{len(offenders)} rules resolve to rules_engine.py, e.g. {offenders[:3]}"
