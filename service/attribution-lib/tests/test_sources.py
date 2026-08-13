# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from datetime import datetime

import pytest

from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.sources import (
    Source,
    SourceSet,
    SymptomSet,
    json_helper,
    set_if_none,
)


class TestSource:
    def test_creation_and_json(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        dict_data = src.to_dict()

        assert dict_data["source_context"]["id"] == 12345
        assert dict_data["line_num"] == 1
        assert dict_data["datum"] == "test_datum"
        assert dict_data["time"] == now.isoformat()
        assert dict_data["rule"] == "test_rule"

    def test_json_with_none_values(self, full_context, timezone):
        """Test JSON serialization with None values"""
        now = datetime.now(tz=timezone)
        src = Source(full_context, None, None, now, AllJobResources, None)

        json_data = src.to_dict()
        assert json_data["line_num"] is None
        assert json_data["datum"] is None
        assert json_data["rule"] is None


class TestSourceSet:
    def test_creation_and_basic_operations(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        source_set = SourceSet("test_value")

        source_set.add(src)
        assert len(source_set) == 1
        assert source_set.has_bad()
        assert source_set.one_source() == src

    def test_discard_all(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        source_set = SourceSet("test_value")

        source_set.add(src)
        assert len(source_set) == 1
        source_set.discard_all()
        assert len(source_set) == 0
        assert not source_set.has_bad()

    def test_update(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src1 = Source(full_context, 1, "test_datum1", now, AllJobResources, "test_rule1")
        src2 = Source(full_context, 2, "test_datum2", now, AllJobResources, "test_rule2")

        set1 = SourceSet("test_value")
        set2 = SourceSet("test_value")

        set1.add(src1)
        set2.add(src2)

        set1.update(set2)
        assert len(set1) == 2
        assert set1.has_bad()

    def test_update_mismatched_values(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        set1 = SourceSet("value1")
        set2 = SourceSet("value2")

        set1.add(src)
        with pytest.raises(Exception):  # MismatchedSourceSetValue
            set1.update(set2)

    def test_discard_by_rule_no_match(self, dummy_ctx):
        source_set = SourceSet("value")
        source1 = Source(dummy_ctx, 1, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source2 = Source(dummy_ctx, 2, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule2")
        source_set.add(source1)
        source_set.add(source2)

        source_set.discard_by_rule(frozenset(["rule3"]))

        assert len(source_set) == 2
        assert source1 in source_set
        assert source2 in source_set

    def test_discard_by_rule_match(self, dummy_ctx):
        source_set = SourceSet("value")
        source1 = Source(dummy_ctx, 1, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source2 = Source(dummy_ctx, 2, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule2")
        source_set.add(source1)
        source_set.add(source2)

        source_set.discard_by_rule(frozenset(["rule1"]))

        assert len(source_set) == 1
        assert source1 not in source_set
        assert source2 in source_set

    def test_discard_by_rule_multiple_match(self, dummy_ctx):
        source_set = SourceSet("value")
        source1 = Source(dummy_ctx, 1, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source2 = Source(dummy_ctx, 2, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule2")
        source3 = Source(dummy_ctx, 3, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source_set.add(source1)
        source_set.add(source2)
        source_set.add(source3)

        source_set.discard_by_rule(frozenset(["rule1"]))

        assert len(source_set) == 1
        assert source1 not in source_set
        assert source2 in source_set
        assert source3 not in source_set

    def test_discard_by_rule_empty_rules(self, dummy_ctx):
        source_set = SourceSet("value")
        source1 = Source(dummy_ctx, 1, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source2 = Source(dummy_ctx, 2, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule2")
        source_set.add(source1)
        source_set.add(source2)

        source_set.discard_by_rule(frozenset())

        assert len(source_set) == 2
        assert source1 in source_set
        assert source2 in source_set

    def test_discard_selected_keeps_unmatched_bad_sources(self, dummy_ctx):
        now = datetime.now(tz=dummy_ctx.timezone)
        selected = Source(dummy_ctx, 1, "selected", now, AllJobResources, "rule1")
        remaining = Source(dummy_ctx, 2, "remaining", now, AllJobResources, "rule2")
        absent = Source(dummy_ctx, 3, "absent", now, AllJobResources, "rule3")
        interesting = Source(dummy_ctx, 4, "interesting", now, AllJobResources, "rule4")
        source_set = SourceSet("value").add(selected).add(remaining).add_interesting(interesting)

        matched = source_set.discard_selected(source for source in (selected, absent, interesting))

        assert matched == {selected}
        assert set(source_set) == {remaining}
        assert set(source_set.all_sources()) == {selected, remaining, interesting}
        assert source_set.has_interesting()
        assert source_set.discard_selected({selected, absent, interesting}) == set()

    def test_discard_selected_can_purge_sources(self, dummy_ctx):
        now = datetime.now(tz=dummy_ctx.timezone)
        selected = Source(dummy_ctx, 1, "selected", now, AllJobResources, "rule1")
        remaining = Source(dummy_ctx, 2, "remaining", now, AllJobResources, "rule2")
        source_set = SourceSet("value").add(selected).add(remaining)

        matched = source_set.discard_selected({selected}, purge=True)

        assert matched == {selected}
        assert set(source_set) == {remaining}
        assert set(source_set.all_sources()) == {remaining}
        assert not source_set.has_interesting()


class TestSymptomSet:
    def test_creation_and_basic_operations(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        symptom_set = SymptomSet()

        symptom_set.add(SourceSet.new_one("test_value", src))
        assert len(symptom_set) == 1
        assert "test_value" in symptom_set
        assert symptom_set.has_bad()

    def test_items_and_symptoms(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src1 = Source(full_context, 1, "test_datum1", now, AllJobResources, "test_rule1")
        src2 = Source(full_context, 2, "test_datum2", now, AllJobResources, "test_rule2")
        symptom_set = SymptomSet()

        symptom_set.add(SourceSet.new_one("value1", src1))
        symptom_set.add(SourceSet.new_one("value2", src2))

        items = list(symptom_set.items())
        assert len(items) == 2
        assert all(isinstance(v, SourceSet) for (k, v) in items)

        symptoms = list(symptom_set.symptoms())
        assert len(symptoms) == 2
        assert "value1" in symptoms
        assert "value2" in symptoms

    def test_discard(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        symptom_set = SymptomSet()

        symptom_set.add(SourceSet.new_one("test_value", src))
        assert "test_value" in symptom_set
        symptom_set.discard("test_value")
        assert "test_value" not in symptom_set
        assert not symptom_set.has_bad()
        assert symptom_set["test_value"].has_interesting()

    def test_discard_with_purge(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")
        symptom_set = SymptomSet()

        symptom_set.add(SourceSet.new_one("test_value", src))
        assert "test_value" in symptom_set
        symptom_set.discard("test_value", purge=True)
        assert symptom_set.is_void()

    def test_discard_all(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src1 = Source(full_context, 1, "test_datum1", now, AllJobResources, "test_rule1")
        src2 = Source(full_context, 2, "test_datum2", now, AllJobResources, "test_rule2")
        symptom_set = SymptomSet()

        symptom_set.add(SourceSet.new_one("value1", src1))
        symptom_set.add(SourceSet.new_one("value2", src2))

        assert len(symptom_set) == 2
        symptom_set.discard_all()
        assert len(symptom_set) == 0
        assert not symptom_set.has_bad()
        assert symptom_set["value1"].has_interesting()

    def test_discard_all_with_purge(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src1 = Source(full_context, 1, "test_datum1", now, AllJobResources, "test_rule1")
        src2 = Source(full_context, 2, "test_datum2", now, AllJobResources, "test_rule2")
        symptom_set = SymptomSet()

        symptom_set.add(SourceSet.new_one("value1", src1))
        symptom_set.add(SourceSet.new_one("value2", src2))

        assert len(symptom_set) == 2
        symptom_set.discard_all(purge=True)
        assert symptom_set.is_void()

    def test_update(self, full_context, timezone):
        now = datetime.now(tz=timezone)
        src1 = Source(full_context, 1, "test_datum1", now, AllJobResources, "test_rule1")
        src2 = Source(full_context, 2, "test_datum2", now, AllJobResources, "test_rule2")

        set1 = SymptomSet()
        set2 = SymptomSet()

        set1.add(SourceSet.new_one("value1", src1))
        set2.add(SourceSet.new_one("value2", src2))

        set1.update(set2)
        assert len(set1) == 2
        assert set1.has_bad()
        assert "value1" in set1
        assert "value2" in set1

    def test_discard_by_rule_no_match(self, dummy_ctx):
        symptom_set = SymptomSet()
        source_set1 = SourceSet("value1")
        source_set2 = SourceSet("value2")
        source1 = Source(dummy_ctx, 1, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source2 = Source(dummy_ctx, 2, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule2")
        source_set1.add(source1)
        source_set2.add(source2)
        symptom_set.add(source_set1)
        symptom_set.add(source_set2)

        symptom_set.discard_by_rule(frozenset(["rule3"]))

        assert len(symptom_set) == 2
        assert "value1" in symptom_set
        assert "value2" in symptom_set

    def test_discard_by_rule_match(self, dummy_ctx):
        symptom_set = SymptomSet()
        source_set1 = SourceSet("value1")
        source_set2 = SourceSet("value2")
        source1 = Source(dummy_ctx, 1, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source2 = Source(dummy_ctx, 2, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule2")
        source_set1.add(source1)
        source_set2.add(source2)
        symptom_set.add(source_set1)
        symptom_set.add(source_set2)

        symptom_set.discard_by_rule(frozenset(["rule1"]))

        assert len(symptom_set) == 1
        assert "value1" not in symptom_set
        assert "value2" in symptom_set

    def test_discard_by_rule_match_with_purge(self, dummy_ctx):
        symptom_set = SymptomSet()
        source_set1 = SourceSet("value1")
        source_set2 = SourceSet("value2")
        source1 = Source(dummy_ctx, 1, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source2 = Source(dummy_ctx, 2, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule2")
        source_set1.add(source1)
        source_set2.add(source2)
        symptom_set.add(source_set1)
        symptom_set.add(source_set2)

        symptom_set.discard_by_rule(frozenset(["rule1"]), purge=True)

        assert len(symptom_set) == 1
        assert "value1" not in symptom_set
        assert "value2" in symptom_set
        # Neither entry will have interesting symptoms as value2 isn't discarded
        # and value1 is purged
        assert "value1" not in symptom_set._data
        assert not symptom_set["value2"].has_interesting()

    def test_discard_by_rule_multiple_match(self, dummy_ctx):
        symptom_set = SymptomSet()
        source_set1 = SourceSet("value1")
        source_set2 = SourceSet("value2")
        source_set3 = SourceSet("value3")
        source1 = Source(dummy_ctx, 1, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source2 = Source(dummy_ctx, 2, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule2")
        source3 = Source(dummy_ctx, 3, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source_set1.add(source1)
        source_set2.add(source2)
        source_set3.add(source3)
        symptom_set.add(source_set1)
        symptom_set.add(source_set2)
        symptom_set.add(source_set3)

        symptom_set.discard_by_rule(frozenset(["rule1"]))

        assert len(symptom_set) == 1
        assert "value1" not in symptom_set
        assert "value2" in symptom_set
        assert "value3" not in symptom_set

    def test_discard_by_rule_empty_rules(self, dummy_ctx):
        symptom_set = SymptomSet()
        source_set1 = SourceSet("value1")
        source_set2 = SourceSet("value2")
        source1 = Source(dummy_ctx, 1, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule1")
        source2 = Source(dummy_ctx, 2, "datum", datetime.now(tz=dummy_ctx.timezone), AllJobResources, "rule2")
        source_set1.add(source1)
        source_set2.add(source2)
        symptom_set.add(source_set1)
        symptom_set.add(source_set2)

        symptom_set.discard_by_rule(frozenset())

        assert len(symptom_set) == 2
        assert "value1" in symptom_set
        assert "value2" in symptom_set


class TestJsonHelper:
    def test_json_helper_with_to_dict(self, full_context, timezone):
        """Test json_helper with object that has to_dict method"""
        now = datetime.now(tz=timezone)
        src = Source(full_context, 1, "test_datum", now, AllJobResources, "test_rule")

        result = json_helper(src)
        assert isinstance(result, dict)
        assert result["source_context"]["id"] == 12345
        assert result["line_num"] == 1
        assert result["datum"] == "test_datum"
        assert result["time"] == now.isoformat()
        assert result["rule"] == "test_rule"

    def test_json_helper_with_dataclass(self):
        """Test json_helper with a dataclass object"""

        @dataclass
        class TestData:
            field1: str
            field2: int

        test_obj = TestData("test", 123)
        result = json_helper(test_obj)
        assert result == {"field1": "test", "field2": 123}


class TestSetIfNone:
    def test_set_if_none_with_none(self):
        """Test set_if_none when first value is None"""
        result = set_if_none(None, "default")
        assert result == "default"

    def test_set_if_none_with_value(self):
        """Test set_if_none when first value is not None"""
        result = set_if_none("existing", "default")
        assert result == "existing"

    def test_set_if_none_with_none_default(self):
        """Test set_if_none when both values are None"""
        result = set_if_none(None, None)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
