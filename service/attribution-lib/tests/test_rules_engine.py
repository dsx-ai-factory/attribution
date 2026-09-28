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
from attribution_lib.exceptions import (
    DuplicateRuleError,
    InvalidRuleParams,
    RuleAcceptsNoParamsError,
    RuleIngestionError,
)
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rule_types.definitions import (
    CollectorRule,
    ExtractorAccumulatorRule,
    ExtractorRule,
    FilterContext,
    FilterPriority,
    FilterRule,
    RuleLocation,
    RuleParameters,
    RuleParamsBase,
)
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.symptoms.declarations import CascadeScope, OutcomePriority, SymptomCategory


def test_symptom_declarations_register_during_load(tmp_path):
    rule_file = tmp_path / "symptom_rules.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules import SymptomFamily, cascade_symptom, outcome_symptom",
                "from attribution_lib.symptoms.declarations import CascadeScope, OutcomePriority, SymptomCategory",
                "",
                "outcome_symptom(",
                "    'bus error',",
                "    family=SymptomFamily(name='network', properties={'kind': 'bus'}),",
                "    outcome='Fail MultiCategory',",
                "    category=SymptomCategory.CSP_EVENT,",
                "    priority=OutcomePriority.LOW_USER,",
                "    doc='bus-error',",
                "    cascade_discards=('PyTorch NCCL watchdog',),",
                ")",
                "cascade_symptom(",
                "    'bus error',",
                "    family=SymptomFamily(name='network', properties={'kind': 'bus'}),",
                "    cascade_discards=('XID-145',),",
                "    scope=CascadeScope.NVL_DOMAIN,",
                ")",
                "cascade_symptom('Python bug', cascade_discards=('IB Error-12',))",
            ]
        )
    )

    loader = RuleLoader()
    loader.load(str(rule_file))

    assert len(loader.symptom_declarations) == 3
    outcome_decl = loader.symptom_declarations[0]
    assert outcome_decl.prefix == "bus error"
    assert outcome_decl.family is not None
    assert outcome_decl.family.name == "network"
    assert outcome_decl.family.properties == {"kind": "bus"}
    assert outcome_decl.outcome == "Fail MultiCategory"
    assert outcome_decl.priority == OutcomePriority.LOW_USER
    assert outcome_decl.category is SymptomCategory.CSP_EVENT
    assert outcome_decl.doc == "bus-error"
    assert outcome_decl.cascade_discards == ("PyTorch NCCL watchdog",)
    assert outcome_decl.scope is CascadeScope.NODE
    assert outcome_decl.location is not None

    domain_decl = loader.symptom_declarations[1]
    assert domain_decl.prefix == "bus error"
    assert domain_decl.family == outcome_decl.family
    assert domain_decl.cascade_discards == ("XID-145",)
    assert domain_decl.scope is CascadeScope.NVL_DOMAIN

    cascade_decl = loader.symptom_declarations[2]
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
                "from attribution_lib.rules import outcome_symptom",
                "from attribution_lib.symptoms.declarations import OutcomePriority",
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

    loader = RuleLoader()
    with pytest.raises(RuleIngestionError, match="requires a non-empty doc anchor"):
        loader.load(str(rule_file))


def test_known_symptom_registers_vocabulary_only_declaration(tmp_path):
    rule_file = tmp_path / "known_symptom_rules.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules import SymptomFamily, known_symptom",
                "",
                "known_symptom(",
                "    'BrokenPipeError',",
                "    family=SymptomFamily(name='python-exception', properties={'type': 'BrokenPipeError'}),",
                ")",
            ]
        )
    )

    loader = RuleLoader()
    loader.load(str(rule_file))

    assert len(loader.symptom_declarations) == 1
    declaration = loader.symptom_declarations[0]
    assert declaration.prefix == "BrokenPipeError"
    assert declaration.family is not None
    assert declaration.family.name == "python-exception"
    assert declaration.family.properties == {"type": "BrokenPipeError"}
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
                "from attribution_lib.rules import outcome_symptom",
                "from attribution_lib.symptoms.declarations import OutcomePriority",
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

    loader = RuleLoader()
    loader.load(str(rule_file))

    assert len(loader.symptom_declarations) == 1


def test_conflicting_symptom_declaration_raises(tmp_path):
    rule_file = tmp_path / "conflicting_symptom_rules.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules import outcome_symptom",
                "from attribution_lib.symptoms.declarations import OutcomePriority",
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

    loader = RuleLoader()
    with pytest.raises(RuleIngestionError, match="Conflicting symptom declaration"):
        loader.load(str(rule_file))
    assert loader.symptom_declarations == []


def test_failed_load_restores_symptom_predicates(tmp_path):
    """Verify a rejected rule module cannot leak predicate registrations.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """

    rule_file = tmp_path / "failing_predicate_rules.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules import symptom_predicate",
                "",
                "@symptom_predicate",
                "def leaked_predicate(symptoms, job_info):",
                "    return True",
                "",
                "raise RuntimeError('load failed')",
            ]
        )
    )

    loader = RuleLoader()
    with pytest.raises(RuntimeError, match="load failed"):
        loader.load(str(rule_file))

    assert loader.symptom_predicates == {}


def test_explicit_empty_accumulator_source_filters_override_rule_sources(tmp_path):
    rule_file = tmp_path / "empty_accumulator_sources.py"
    rule_file.write_text(
        "\n".join(
            [
                "from attribution_lib.rules import ExtractorAccumulator, extractor_accumulator_def",
                "from attribution_lib.observation.inbound import Source as ObservationSource",
                "",
                "rule_sources = {ObservationSource.APPLOG}",
                "",
                "@extractor_accumulator_def('ExplicitEmpty', 'trigger', source_filters=set())",
                "class ExplicitEmpty(ExtractorAccumulator):",
                "    def process_line(self, source): pass",
                "    @property",
                "    def is_consuming(self): return False",
                "    def finalize(self): return []",
                "",
                "@extractor_accumulator_def('Omitted', 'trigger')",
                "class Omitted(ExplicitEmpty): pass",
            ]
        )
    )

    loader = RuleLoader()
    loader.load(str(rule_file))

    explicit = loader.rules["ExplicitEmpty"]
    omitted = loader.rules["Omitted"]
    assert isinstance(explicit, ExtractorAccumulatorRule)
    assert isinstance(omitted, ExtractorAccumulatorRule)
    assert explicit.source_filters == frozenset()
    assert omitted.source_filters == frozenset({ObservationSource.APPLOG})


# Filter priority ordering tests
class TestFilterPriorityOrdering:
    """Tests for the filter rule priority/ordering system."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.execution_log: list[str] = []
        self.mgr = EventManager(RuleLoader().build())
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

    def _run_filters(self, loader: RuleLoader, **kwargs) -> list[str]:
        stream = self.mgr.create_stream("test")
        loader.build().execute(FilterContext(event_stream=stream, job_info=None, **kwargs))
        return list(self.execution_log)

    def test_default_priority_is_standard(self):
        rule = self._make_rule("R")
        assert rule.priority == FilterPriority.STANDARD

    def test_priority_ordering(self):
        """Rules registered in arbitrary order execute sorted by priority.

        phase=X runs only filters at that priority; phase=None runs all phases.
        """
        loader = RuleLoader()
        loader.register_rule(self._make_rule("Late", FilterPriority.LATE))
        loader.register_rule(self._make_rule("Mid", FilterPriority.MID))
        loader.register_rule(self._make_rule("Std", FilterPriority.STANDARD))
        loader.register_rule(self._make_rule("Early", FilterPriority.EARLY))

        assert self._run_filters(loader) == ["Early", "Mid", "Std", "Late"]

        self.execution_log.clear()
        assert self._run_filters(loader, phase=FilterPriority.MID) == ["Mid"]

    def test_stable_tiebreaking(self):
        """Rules with the same priority execute in registration order."""
        loader = RuleLoader()
        loader.register_rule(self._make_rule("A"))
        loader.register_rule(self._make_rule("B"))
        loader.register_rule(self._make_rule("C"))

        assert self._run_filters(loader) == ["A", "B", "C"]

    def test_mixed_priorities_and_tiebreaking(self):
        """Within the same priority, registration order wins."""
        loader = RuleLoader()
        loader.register_rule(self._make_rule("S1", FilterPriority.STANDARD))
        loader.register_rule(self._make_rule("M1", FilterPriority.MID))
        loader.register_rule(self._make_rule("S2", FilterPriority.STANDARD))
        loader.register_rule(self._make_rule("M2", FilterPriority.MID))

        assert self._run_filters(loader) == ["M1", "M2", "S1", "S2"]

        self.execution_log.clear()
        assert self._run_filters(loader, phase=FilterPriority.MID) == ["M1", "M2"]

    def test_duplicate_filter_rule_raises(self):
        loader = RuleLoader()
        loader.register_rule(self._make_rule("Dup"))

        with pytest.raises(DuplicateRuleError, match="Duplicate rule 'Dup'"):
            loader.register_rule(self._make_rule("Dup", FilterPriority.MID))

    def test_duplicate_name_across_rule_types_raises(self):
        """A name registered as one rule type cannot be reused as a different type."""
        loader = RuleLoader()
        loader.register_rule(
            ExtractorRule(
                name="Shared",
                author="test",
                location=RuleLocation("first.py", 7),
                match_re=r"x",
                processor=lambda groups, hostname, source, context: [],
            )
        )
        with pytest.raises(DuplicateRuleError, match="first at first.py:7"):
            loader.register_rule(self._make_rule("Shared", FilterPriority.MID))

    def test_backward_compatible_default_priority(self):
        """Existing rules without explicit priority preserve registration order."""
        loader = RuleLoader()
        for name in ["DemoteTflops", "DemoteWatchdog", "DemoteDataloader", "ProcessCompletes"]:
            loader.register_rule(self._make_rule(name))

        assert self._run_filters(loader) == ["DemoteTflops", "DemoteWatchdog", "DemoteDataloader", "ProcessCompletes"]


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


def test_engine_get_rule_is_independent_of_rule_type():
    loader = RuleLoader()
    rule = _collector("Collector", None)
    loader.register_rule(rule)

    engine = loader.build()

    assert engine.get_rule(rule.name) is rule
    assert engine.get_rule("Missing") is None


class TestParseCollectorParams:
    """Behavior of collector parameter parsing across model/no-model and valid/invalid input."""

    def test_no_model_accepts_none(self):
        assert _collector("NoModel", None).parse_params(None) is None

    def test_no_model_rejects_supplied_params(self):
        """Passing params for a rule with no params_model raises."""
        rule = _collector("NoModel", None)
        with pytest.raises(RuleAcceptsNoParamsError) as exc:
            rule.parse_params(RuleParameters(params={"value": 1}))
        assert exc.value.rule_id == "NoModel"

    def test_no_model_accepts_empty_envelope(self):
        """Empty envelope is fine. Callers often pass {} for all rules indiscriminately."""
        assert _collector("NoModel", None).parse_params(RuleParameters()) is None

    def test_model_parses_valid_params(self):
        out = _collector("WithModel", _SampleParams).parse_params(RuleParameters(params={"value": 7}))
        assert isinstance(out, _SampleParams) and out.value == 7

    def test_model_uses_defaults_for_none(self):
        out = _collector("WithModel", _SampleParams).parse_params(None)
        assert out == _SampleParams(value=0)

    def test_model_rejects_invalid_params(self):
        rule = _collector("WithModel", _SampleParams)
        with pytest.raises(InvalidRuleParams) as exc:
            rule.parse_params(RuleParameters(params={"value": "nope"}))
        assert exc.value.rule_id == "WithModel"


def test_no_bundled_rule_resolves_to_engine_module():
    """Sweep: no registered bundled rule should have its location pointing at
    rules_engine.py (the registration machinery)."""

    d = os.path.join(os.path.dirname(attribution_lib.__file__), "bundled_rules")
    loader = RuleLoader()
    for f in sorted(glob.glob("*.py", root_dir=d)):
        loader.load(os.path.join(d, f))
    offenders = [
        (name, r.location.file)
        for name, r in loader.rules.items()
        if r.location and os.path.basename(r.location.file) == "rules_engine.py"
    ]
    assert not offenders, f"{len(offenders)} rules resolve to rules_engine.py, e.g. {offenders[:3]}"
