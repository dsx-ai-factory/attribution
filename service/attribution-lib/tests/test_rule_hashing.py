# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from attribution_lib.core import get_bundled_rules_path
from attribution_lib.events import EventManager
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rule_types.definitions import (
    ExtractorAccumulatorRule,
    ExtractorRule,
    FilterRule,
    RuleEmission,
    RuleLocation,
)
from attribution_lib.rules_engine import RuleLoader


def _make_extractor(
    name: str,
    regex: str,
    sources: frozenset = frozenset(),
    credited_rules: frozenset[str] = frozenset(),
) -> ExtractorRule:
    return ExtractorRule(
        name=name,
        author="test",
        location=RuleLocation("test.py", 1),
        match_re=regex,
        processor=lambda ctx, m, emit: None,
        source_filters=sources,
        credited_rule_ids=credited_rules,
    )


def _make_accumulator(name: str, regex: str, sources: frozenset = frozenset()) -> ExtractorAccumulatorRule:
    return ExtractorAccumulatorRule(
        name=name,
        author="test",
        location=RuleLocation("test.py", 1),
        match_re=regex,
        factory=lambda: None,
        source_filters=sources,
    )


def _make_filter(name: str, priority: int = 1000) -> FilterRule:
    return FilterRule(
        name=name,
        author="test",
        location=RuleLocation("test.py", 1),
        processor=lambda es, ji, **kw: None,
        priority=priority,
    )


RULES_DIR = get_bundled_rules_path()


class TestExtractorHash:
    def test_stable_across_calls(self):
        loader = RuleLoader()
        loader.register_rule(_make_extractor("r1", r"error\s+\d+"))
        assert loader.build().extractor_hash() == loader.build().extractor_hash()

    def test_changes_when_rule_added(self):
        loader = RuleLoader()
        loader.register_rule(_make_extractor("r1", r"error\s+\d+"))
        h1 = loader.build().extractor_hash()

        loader.register_rule(_make_extractor("r2", r"warning"))
        h2 = loader.build().extractor_hash()
        assert h1 != h2

    def test_changes_when_regex_differs(self):
        loader1 = RuleLoader()
        loader1.register_rule(_make_extractor("r1", r"pattern_a"))

        loader2 = RuleLoader()
        loader2.register_rule(_make_extractor("r1", r"pattern_b"))

        assert loader1.build().extractor_hash() != loader2.build().extractor_hash()

    def test_independent_of_filter_changes(self):
        loader = RuleLoader()
        loader.register_rule(_make_extractor("r1", r"error"))
        h1 = loader.build().extractor_hash()

        loader.register_rule(_make_filter("f1"))
        h2 = loader.build().extractor_hash()
        assert h1 == h2

    def test_source_filter_affects_hash(self):
        loader1 = RuleLoader()
        loader1.register_rule(_make_extractor("r1", r"error", frozenset()))

        loader2 = RuleLoader()
        loader2.register_rule(_make_extractor("r1", r"error", frozenset({ObservationSource.APPLOG})))

        assert loader1.build().extractor_hash() != loader2.build().extractor_hash()

    def test_accumulator_source_filter_affects_hash(self):
        loader1 = RuleLoader()
        loader1.register_rule(_make_accumulator("r1", r"error", frozenset()))

        loader2 = RuleLoader()
        loader2.register_rule(_make_accumulator("r1", r"error", frozenset({ObservationSource.APPLOG})))

        assert loader1.build().extractor_hash() != loader2.build().extractor_hash()

    def test_credited_rules_affect_hash(self):
        loader1 = RuleLoader()
        loader1.declared_rule_names.add("derived-credit-a")
        loader1.register_rule(_make_extractor("r1", r"error", credited_rules=frozenset({"derived-credit-a"})))
        loader2 = RuleLoader()
        loader2.declared_rule_names.add("derived-credit-b")
        loader2.register_rule(_make_extractor("r1", r"error", credited_rules=frozenset({"derived-credit-b"})))

        assert loader1.build().extractor_hash() != loader2.build().extractor_hash()


class TestRuleEmissions:
    def test_storage_key_round_trips_without_rules_schema_columns(self):
        emission = RuleEmission(
            credited_rule_id='credit: "quoted" / ünicode',
            emitter_rule_id="extractor:[arm]",
        )

        assert RuleEmission.from_storage_key(emission.storage_key) == emission

        manager = EventManager(RuleLoader().build())
        try:
            columns = [row[1] for row in manager._conn.execute("PRAGMA table_info('rules')").fetchall()]
        finally:
            manager.close()

        assert columns == ["id", "name"]

    def test_extractor_declares_identity_and_derived_credits(self):
        loader = RuleLoader()
        loader.declared_rule_names.add("shared-credit")
        loader.register_rule(_make_extractor("extractor-a", r"error", credited_rules=frozenset({"shared-credit"})))

        engine = loader.build()

        assert {(emission.credited_rule_id, emission.emitter_rule_id) for emission in engine.rule_emissions} == {
            ("extractor-a", "extractor-a"),
            ("shared-credit", "extractor-a"),
        }

    def test_shared_credit_retains_every_extractor_relationship(self):
        loader = RuleLoader()
        loader.declared_rule_names.add("shared-credit")
        loader.register_rule(_make_extractor("extractor-a", r"first", credited_rules=frozenset({"shared-credit"})))
        loader.register_rule(_make_extractor("extractor-b", r"second", credited_rules=frozenset({"shared-credit"})))

        engine = loader.build()

        assert engine.get_rule_emission("shared-credit", "extractor-a") is not None
        assert engine.get_rule_emission("shared-credit", "extractor-b") is not None

    def test_event_manager_persists_rule_credit_metadata(self):
        loader = RuleLoader()
        loader.declared_rule_names.add("shared-credit")
        loader.register_rule(_make_extractor("extractor-a", r"error", credited_rules=frozenset({"shared-credit"})))
        manager = EventManager(loader.build())

        try:
            storage_names = [row[0] for row in manager._conn.execute("SELECT name FROM rules").fetchall()]
        finally:
            manager.close()

        emissions = {
            emission
            for storage_name in storage_names
            if (emission := RuleEmission.from_storage_key(storage_name)) is not None
        }
        assert RuleEmission(credited_rule_id="shared-credit", emitter_rule_id="extractor-a") in emissions


class TestRuleProvenance:
    def test_exposes_stable_id_regex_and_deterministic_hash(self):
        loader = RuleLoader()
        loader.register_rule(_make_extractor("stable_id", r"error\s+\d+", frozenset({ObservationSource.APPLOG})))
        engine = loader.build()
        rule = engine.get_rule("stable_id")

        assert rule is not None
        first = rule.provenance()
        second = rule.provenance()

        assert first == second
        assert first.rule_id == "stable_id"
        assert first.rule_type == "ExtractorRule"
        assert first.matcher is not None
        assert first.matcher.regex == r"error\s+\d+"
        assert first.matcher.sources == ("applog",)
        assert first.matcher_hash is not None
        assert len(first.matcher_hash) == 64

    def test_hash_changes_with_effective_matching_criteria(self):
        loader1 = RuleLoader()
        loader1.register_rule(_make_extractor("stable_id", r"pattern_a"))
        loader2 = RuleLoader()
        loader2.register_rule(_make_extractor("stable_id", r"pattern_b"))

        first_rule = loader1.build().get_rule("stable_id")
        second_rule = loader2.build().get_rule("stable_id")

        assert first_rule is not None
        assert second_rule is not None
        assert first_rule.provenance().matcher_hash != second_rule.provenance().matcher_hash

    def test_hash_changes_with_source_filters(self):
        loader1 = RuleLoader()
        loader1.register_rule(_make_extractor("stable_id", r"pattern", frozenset({ObservationSource.APPLOG})))
        loader2 = RuleLoader()
        loader2.register_rule(_make_extractor("stable_id", r"pattern", frozenset({ObservationSource.SYSLOG})))

        first_rule = loader1.build().get_rule("stable_id")
        second_rule = loader2.build().get_rule("stable_id")

        assert first_rule is not None
        assert second_rule is not None
        assert first_rule.provenance().matcher_hash != second_rule.provenance().matcher_hash

    def test_matcher_hash_is_explicitly_not_a_full_rule_hash(self):
        loader1 = RuleLoader()
        loader1.register_rule(_make_extractor("first_id", r"pattern"))
        loader2 = RuleLoader()
        loader2.register_rule(_make_extractor("second_id", r"pattern"))

        first_rule = loader1.build().get_rule("first_id")
        second_rule = loader2.build().get_rule("second_id")

        assert first_rule is not None
        assert second_rule is not None
        assert first_rule.provenance().matcher_hash == second_rule.provenance().matcher_hash
        assert first_rule.provenance().rule_id != second_rule.provenance().rule_id

    def test_non_extractor_has_basic_provenance(self):
        loader = RuleLoader()
        loader.register_rule(_make_filter("filter_rule"))

        rule = loader.build().get_rule("filter_rule")
        assert rule is not None
        provenance = rule.provenance()
        assert provenance.rule_id == "filter_rule"
        assert provenance.rule_type == "FilterRule"
        assert provenance.matcher_hash is None
        assert provenance.matcher is None


class TestAttributionHash:
    def test_stable_across_calls(self):
        loader = RuleLoader()
        loader.register_rule(_make_filter("f1", priority=500))
        assert loader.build().attribution_hash() == loader.build().attribution_hash()

    def test_changes_when_filter_added(self):
        loader = RuleLoader()
        loader.register_rule(_make_filter("f1"))
        h1 = loader.build().attribution_hash()

        loader.register_rule(_make_filter("f2"))
        h2 = loader.build().attribution_hash()
        assert h1 != h2

    def test_changes_when_priority_differs(self):
        loader1 = RuleLoader()
        loader1.register_rule(_make_filter("f1", priority=100))

        loader2 = RuleLoader()
        loader2.register_rule(_make_filter("f1", priority=200))

        assert loader1.build().attribution_hash() != loader2.build().attribution_hash()

    def test_independent_of_extractor_changes(self):
        loader = RuleLoader()
        loader.register_rule(_make_filter("f1"))
        h1 = loader.build().attribution_hash()

        loader.register_rule(_make_extractor("r1", r"error"))
        h2 = loader.build().attribution_hash()
        assert h1 == h2


class TestHashWithLoadedRules:
    """Verify hashing works with real rule files loaded via RuleLoader.load()."""

    def test_loaded_rules_produce_stable_hash(self):
        loader1 = RuleLoader()
        loader1.load(RULES_DIR / "applog_rules.py")

        loader2 = RuleLoader()
        loader2.load(RULES_DIR / "applog_rules.py")

        assert loader1.build().extractor_hash() == loader2.build().extractor_hash()
        assert loader1.build().attribution_hash() == loader2.build().attribution_hash()
