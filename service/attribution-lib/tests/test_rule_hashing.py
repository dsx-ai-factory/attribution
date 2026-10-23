# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_lib.core import get_bundled_rules_path
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rule_types.definitions import (
    ExtractorAccumulatorRule,
    ExtractorRule,
    FilterRule,
    RuleLocation,
)
from attribution_lib.rules_engine import RuleLoader


def _make_extractor(name: str, regex: str, sources: frozenset = frozenset()) -> ExtractorRule:
    return ExtractorRule(
        name=name,
        author="test",
        location=RuleLocation("test.py", 1),
        match_re=regex,
        processor=lambda ctx, m, emit: None,
        source_filters=sources,
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
