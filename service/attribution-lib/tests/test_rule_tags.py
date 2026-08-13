# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import dataclasses

import pytest

from attribution_lib.rules_engine import (
    CollectorRule,
    ExtractorAccumulatorRule,
    ExtractorRule,
    FilterRule,
    RefinerRule,
    Rule,
    RuleIngestionError,
    RuleLocation,
    normalize_tags,
)

WHERE = "RuleName (rules.py:123)"


class TestNormalizeTags:
    def test_omitted_tags_yield_empty_tuple(self):
        assert normalize_tags(where=WHERE) == ()

    def test_none_yields_empty_tuple(self):
        assert normalize_tags(None, where=WHERE) == ()

    def test_empty_list_yields_empty_tuple(self):
        assert normalize_tags([], where=WHERE) == ()

    def test_duplicates_and_declaration_order_yield_sorted_unique_tuple(self):
        assert normalize_tags(["oom", "external", "oom"], where=WHERE) == ("external", "oom")

    def test_tuple_input_yields_sorted_tuple(self):
        assert normalize_tags(("oom", "external"), where=WHERE) == ("external", "oom")

    def test_surrounding_whitespace_is_stripped_before_dedupe(self):
        assert normalize_tags(["  external ", "external", "\toom\n"], where=WHERE) == ("external", "oom")

    def test_capitalized_internal_alongside_external_is_kept_as_an_ordinary_tag(self):
        assert normalize_tags(["Internal", "external"], where=WHERE) == ("Internal", "external")

    @pytest.mark.parametrize(
        "tags",
        [
            pytest.param(None, id="none"),
            pytest.param([], id="empty"),
            pytest.param(["external"], id="single"),
            pytest.param(["  oom ", "external", "oom"], id="dirty"),
            pytest.param(("syslog", "external"), id="tuple"),
        ],
    )
    def test_normalizing_an_already_normalized_value_changes_nothing(self, tags):
        normalized = normalize_tags(tags, where=WHERE)
        assert normalize_tags(normalized, where=WHERE) == normalized


class TestNormalizeTagsRejections:
    @pytest.mark.parametrize(
        "tags",
        [
            pytest.param("external", id="bare-string"),
            pytest.param(5, id="int"),
            pytest.param({"external"}, id="set"),
            pytest.param(["external", 5], id="non-string-entry"),
            pytest.param(["external", ""], id="empty-entry"),
            pytest.param(["external", "   "], id="whitespace-only-entry"),
            pytest.param(["internal", "external"], id="conflict"),
            pytest.param([" internal ", "external"], id="conflict-after-strip"),
        ],
    )
    def test_invalid_tags_are_rejected(self, tags):
        with pytest.raises(RuleIngestionError):
            normalize_tags(tags, where=WHERE)


def _untagged_rules() -> list[Rule]:
    """One rule of every kind, none of them declaring tags."""
    location = RuleLocation("test.py", 1)
    return [
        ExtractorRule(
            name="extractor",
            author="test",
            location=location,
            match_re="trigger",
            processor=lambda groups, hostname, source, context: [],
        ),
        FilterRule(name="filter", author="test", location=location, processor=lambda context: None),
        ExtractorAccumulatorRule(
            name="accumulator", author="test", location=location, match_re="trigger", factory=lambda: None
        ),
        RefinerRule(name="refiner", author="test", location=location, processor=lambda context: None),
        CollectorRule(
            name="collector",
            author="test",
            location=location,
            processor=lambda es, ji, params: None,
            formatter=lambda raw: {},
        ),
    ]


@pytest.mark.parametrize("rule", _untagged_rules(), ids=lambda rule: type(rule).__name__)
class TestRuleTagsField:
    def test_rule_constructed_without_tags_exposes_empty_tuple(self, rule):
        assert rule.tags == ()

    def test_assigning_tags_to_a_constructed_rule_is_rejected(self, rule):
        with pytest.raises(dataclasses.FrozenInstanceError):
            rule.tags = ("external",)
