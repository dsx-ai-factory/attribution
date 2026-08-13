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
from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rules_engine import (
    ExtractorRule,
    FilterRule,
    RuleEngine,
    RuleLocation,
)


def _make_extractor(name: str, regex: str, sources: frozenset = frozenset()) -> ExtractorRule:
    return ExtractorRule(
        name=name,
        author="test",
        location=RuleLocation("test.py", 1),
        match_re=regex,
        processor=lambda ctx, m, emit: None,
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
        engine = RuleEngine()
        engine.register_rule(_make_extractor("r1", r"error\s+\d+"))
        assert engine.extractor_hash() == engine.extractor_hash()

    def test_changes_when_rule_added(self):
        engine = RuleEngine()
        engine.register_rule(_make_extractor("r1", r"error\s+\d+"))
        h1 = engine.extractor_hash()

        engine.register_rule(_make_extractor("r2", r"warning"))
        h2 = engine.extractor_hash()
        assert h1 != h2

    def test_changes_when_regex_differs(self):
        e1 = RuleEngine()
        e1.register_rule(_make_extractor("r1", r"pattern_a"))

        e2 = RuleEngine()
        e2.register_rule(_make_extractor("r1", r"pattern_b"))

        assert e1.extractor_hash() != e2.extractor_hash()

    def test_independent_of_filter_changes(self):
        engine = RuleEngine()
        engine.register_rule(_make_extractor("r1", r"error"))
        h1 = engine.extractor_hash()

        engine.register_rule(_make_filter("f1"))
        h2 = engine.extractor_hash()
        assert h1 == h2

    def test_source_filter_affects_hash(self):
        e1 = RuleEngine()
        e1.register_rule(_make_extractor("r1", r"error", frozenset()))

        e2 = RuleEngine()
        e2.register_rule(_make_extractor("r1", r"error", frozenset({ObservationSource.APPLOG})))

        assert e1.extractor_hash() != e2.extractor_hash()


class TestAttributionHash:
    def test_stable_across_calls(self):
        engine = RuleEngine()
        engine.register_rule(_make_filter("f1", priority=500))
        assert engine.attribution_hash() == engine.attribution_hash()

    def test_changes_when_filter_added(self):
        engine = RuleEngine()
        engine.register_rule(_make_filter("f1"))
        h1 = engine.attribution_hash()

        engine.register_rule(_make_filter("f2"))
        h2 = engine.attribution_hash()
        assert h1 != h2

    def test_changes_when_priority_differs(self):
        e1 = RuleEngine()
        e1.register_rule(_make_filter("f1", priority=100))

        e2 = RuleEngine()
        e2.register_rule(_make_filter("f1", priority=200))

        assert e1.attribution_hash() != e2.attribution_hash()

    def test_independent_of_extractor_changes(self):
        engine = RuleEngine()
        engine.register_rule(_make_filter("f1"))
        h1 = engine.attribution_hash()

        engine.register_rule(_make_extractor("r1", r"error"))
        h2 = engine.attribution_hash()
        assert h1 == h2


class TestHashWithLoadedRules:
    """Verify hashing works with real rule files loaded via RuleEngine.load()."""

    def test_loaded_rules_produce_stable_hash(self):
        e1 = RuleEngine()
        e1.load(RULES_DIR / "applog_rules.py")

        e2 = RuleEngine()
        e2.load(RULES_DIR / "applog_rules.py")

        assert e1.extractor_hash() == e2.extractor_hash()
        assert e1.attribution_hash() == e2.attribution_hash()
