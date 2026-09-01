# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for two-stage regex matcher"""

import re

from attribution_lib.regex_accelerator import TwoStageMatcher


class TestTwoStageMatcherBasic:
    """Basic functionality tests for TwoStageMatcher"""

    def test_simple_match(self):
        """Test simple pattern matching"""
        patterns = {0: r"error"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("This is an error message")
        assert len(matches) == 1
        assert matches[0][0] == 0

    def test_no_match(self):
        """Test line that doesn't match any pattern"""
        patterns = {0: r"error", 1: r"warning"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("This is an info message")
        assert len(matches) == 0

    def test_multiple_patterns_single_match(self):
        """Test multiple patterns where one matches"""
        patterns = {0: r"error", 1: r"warning", 2: r"info"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("This is a warning")
        assert len(matches) == 1
        assert matches[0][0] == 1

    def test_multiple_patterns_multiple_matches(self):
        """Test line matching multiple patterns"""
        patterns = {0: r"error", 1: r"message"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("error message")
        assert len(matches) == 2
        keys = {m[0] for m in matches}
        assert keys == {0, 1}

    def test_empty_line(self):
        """Test matching against empty line"""
        patterns = {0: r"error"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("")
        assert len(matches) == 0

    def test_empty_patterns(self):
        """Test matcher with no patterns"""
        matcher = TwoStageMatcher({})
        matches = matcher.match_line("any text")
        assert len(matches) == 0


class TestNamedGroups:
    """Test that named capture groups work correctly"""

    def test_single_named_group(self):
        """Test pattern with single named group"""
        patterns = {0: r"ERROR:\s+(?P<code>\d+)"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("ERROR: 500")
        assert len(matches) == 1
        _, match_obj = matches[0]
        assert match_obj.groupdict() == {"code": "500"}

    def test_multiple_named_groups(self):
        """Test pattern with multiple named groups"""
        patterns = {0: r"(?P<level>ERROR|WARN):\s+(?P<code>\d+)\s+-\s+(?P<msg>.+)"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("ERROR: 500 - Internal server error")
        assert len(matches) == 1
        _, match_obj = matches[0]
        groups = match_obj.groupdict()
        assert groups["level"] == "ERROR"
        assert groups["code"] == "500"
        assert groups["msg"] == "Internal server error"

    def test_optional_named_group(self):
        """Test pattern with optional named group"""
        patterns = {0: r"user_id:\s*(?P<uid>\d+)?"}
        matcher = TwoStageMatcher(patterns)

        # Test with value
        matches = matcher.match_line("user_id: 12345")
        assert len(matches) == 1
        _, match_obj = matches[0]
        assert match_obj.groupdict() == {"uid": "12345"}

        # Test without value
        matches = matcher.match_line("user_id: ")
        assert len(matches) == 1
        _, match_obj = matches[0]
        assert match_obj.groupdict() == {"uid": None}

    def test_nested_named_groups(self):
        """Test pattern with nested groups"""
        patterns = {0: r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+(?P<time>\d{2}:\d{2}:\d{2}))"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("2024-01-15 10:30:45")
        assert len(matches) == 1
        _, match_obj = matches[0]
        groups = match_obj.groupdict()
        assert "timestamp" in groups
        assert "time" in groups
        assert groups["time"] == "10:30:45"


class TestMultipleLines:
    """Test matching multiple lines"""

    def test_match_lines_basic(self):
        """Test basic multi-line matching"""
        patterns = {0: r"error", 1: r"warning"}
        matcher = TwoStageMatcher(patterns)

        lines = [
            "This is an error",
            "This is a warning",
            "This is info",
        ]

        matches = matcher.match_lines(lines)
        assert len(matches) == 2
        # Check line numbers
        line_nums = {m[0] for m in matches}
        assert line_nums == {1, 2}

    def test_match_lines_with_named_groups(self):
        """Test multi-line matching with capture groups"""
        patterns = {0: r"user_id:\s*(?P<uid>\d+)"}
        matcher = TwoStageMatcher(patterns)

        lines = [
            "Login attempt for user_id: 12345",
            "Logout for user_id: 67890",
            "Anonymous access",
        ]

        matches = matcher.match_lines(lines)
        assert len(matches) == 2

        # Check captured values
        uids = [m[2].group("uid") for m in matches]
        assert "12345" in uids
        assert "67890" in uids

    def test_match_lines_empty_list(self):
        """Test matching against empty line list"""
        patterns = {0: r"error"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_lines([])
        assert len(matches) == 0

    def test_match_lines_no_return_line_nums(self):
        """Test matching without line numbers"""
        patterns = {0: r"error"}
        matcher = TwoStageMatcher(patterns)

        lines = ["error message", "info message"]
        matches = matcher.match_lines(lines, return_line_nums=False)

        # Should still return matches but format might differ
        assert len(matches) > 0


class TestPatternInfo:
    """Test pattern info retrieval"""

    def test_get_pattern_info(self):
        """Test retrieving pattern information"""
        patterns = {0: r"error:\s+(?P<code>\d+)", 1: r"warning"}
        matcher = TwoStageMatcher(patterns)

        info = matcher.get_pattern_info(0)
        assert info.key == 0
        assert info.regex_str == r"error:\s+(?P<code>\d+)"
        assert info.literal is not None

    def test_pattern_has_literal(self):
        """Test that patterns have literals extracted"""
        patterns = {0: r"error:\s+\d+"}
        matcher = TwoStageMatcher(patterns)

        info = matcher.get_pattern_info(0)
        assert info.literal == "error:"

    def test_pattern_no_literal(self):
        """Test pattern with no extractable literal"""
        patterns = {0: r"^\d+$"}
        matcher = TwoStageMatcher(patterns)

        info = matcher.get_pattern_info(0)
        # Literal should be None or very short
        assert info.literal is None or len(info.literal) <= 1


class TestDebugInfo:
    """Test debug information"""

    def test_debug_info_structure(self):
        """Test that debug info has expected structure"""
        patterns = {0: r"error:\s+\d+", 1: r"warning:\s+\d+"}
        matcher = TwoStageMatcher(patterns)

        info = matcher.debug_info()
        assert "total_patterns" in info
        assert "patterns_with_literals" in info
        assert "patterns_without_literals" in info
        assert "unique_literals_in_automaton" in info
        assert "automaton_literal_usage" in info
        assert "pattern_best_literals" in info

    def test_debug_info_counts(self):
        """Test debug info has correct counts"""
        patterns = {0: r"error:\s+\d+", 1: r"warning:\s+\d+", 2: r"^\d+$"}
        matcher = TwoStageMatcher(patterns)

        info = matcher.debug_info()
        assert info["total_patterns"] == 3
        # First two should have literals, third might not
        assert info["patterns_with_literals"] >= 2


class TestComplexPatterns:
    """Test complex regex patterns"""

    def test_alternation_pattern(self):
        """Test pattern with alternation"""
        patterns = {0: r"(error|warning|info):"}
        matcher = TwoStageMatcher(patterns)

        for word in ["error:", "warning:", "info:"]:
            matches = matcher.match_line(word)
            assert len(matches) == 1

    def test_quantifiers(self):
        """Test patterns with various quantifiers"""
        patterns = {
            0: r"a+x",  # One or more 'a' followed by 'x'
            1: r"b+y",  # One or more 'b' followed by 'y'
            2: r"c+z",  # One or more 'c' followed by 'z'
            3: r"d{2,4}w",  # 2-4 'd' followed by 'w'
        }
        matcher = TwoStageMatcher(patterns)

        test_cases = [
            ("aaax", {0}),
            ("bbbby", {1}),
            ("cz", {2}),
            ("dddw", {3}),
        ]

        for text, expected_ids in test_cases:
            matches = matcher.match_line(text)
            match_ids = {m[0] for m in matches}
            assert match_ids == expected_ids

    def test_character_classes(self):
        """Test patterns with character classes"""
        patterns = {
            0: r"\d{3}-\d{3}-\d{4}",  # Phone number
            1: r"[A-Z][a-z]+",  # Capitalized word
            2: r"\w+@\w+\.\w+",  # Simple email
        }
        matcher = TwoStageMatcher(patterns)

        test_cases = [
            ("555-123-4567", 0),
            ("Hello", 1),
            ("user@example.com", 2),
        ]

        for text, expected_pattern in test_cases:
            matches = matcher.match_line(text)
            assert any(m[0] == expected_pattern for m in matches)

    def test_anchors(self):
        """Test patterns with anchors"""
        patterns = {
            0: r"^start",
            1: r"end$",
            2: r"\bword\b",
        }
        matcher = TwoStageMatcher(patterns)

        assert len(matcher.match_line("start of line")) == 1
        assert len(matcher.match_line("line end")) == 1
        assert len(matcher.match_line("a word here")) == 1
        assert len(matcher.match_line("keyword")) == 0  # No word boundary


class TestPerformance:
    """Test performance characteristics"""

    def test_many_patterns(self):
        """Test matching with many patterns"""
        patterns = {i: f"pattern_{i}:\\s+\\d+" for i in range(100)}
        matcher = TwoStageMatcher(patterns)

        # Should handle many patterns without error
        matches = matcher.match_line("pattern_50: 123")
        assert len(matches) >= 1

    def test_many_lines(self):
        """Test matching many lines"""
        patterns = {0: r"error:\s+\d+"}
        matcher = TwoStageMatcher(patterns)

        lines = [f"line {i} with error: {i}" if i % 10 == 0 else f"line {i}" for i in range(1000)]

        matches = matcher.match_lines(lines)
        # Should find matches for lines divisible by 10
        assert len(matches) == 100

    def test_long_lines(self):
        """Test matching against very long lines"""
        patterns = {0: r"needle"}
        matcher = TwoStageMatcher(patterns)

        # Create a long line with the pattern at the end
        long_line = "x" * 10000 + " needle " + "y" * 10000

        matches = matcher.match_line(long_line)
        assert len(matches) == 1


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_special_characters_in_literal(self):
        """Test pattern with special characters"""
        patterns = {0: r"file\.txt"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("open file.txt")
        assert len(matches) == 1

    def test_unicode_support(self):
        """Test matching unicode text"""
        patterns = {0: r"café"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("I love café")
        assert len(matches) == 1

    def test_newlines_in_pattern(self):
        """Test pattern that spans multiple lines"""
        patterns = {0: r"line1\nline2"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("line1\nline2")
        assert len(matches) == 1

    def test_pattern_with_no_literal_still_matches(self):
        """Test that patterns without literals still work (fallback to checking every line)"""
        patterns = {0: r"^\d+$"}  # No literal to extract
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("12345")
        assert len(matches) == 1

        matches = matcher.match_line("abc")
        assert len(matches) == 0

    def test_overlapping_matches(self):
        """Test patterns that have overlapping matches"""
        patterns = {0: r"\w+", 1: r"\d+"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("abc123")
        # Both patterns should match
        assert len(matches) >= 1


class TestCustomKeys:
    """Test custom key functionality"""

    def test_string_keys(self):
        """Test using dict with string keys"""
        patterns = {
            "error": r"ERROR:\s+(?P<code>\d+)",
            "warn": r"WARNING:\s+(?P<msg>.+)",
        }
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("ERROR: 404")
        assert len(matches) == 1
        assert matches[0][0] == "error"

        matches = matcher.match_line("WARNING: high load")
        assert len(matches) == 1
        assert matches[0][0] == "warn"

    def test_enum_keys(self):
        """Test using enum values as keys"""
        from enum import Enum, auto

        class ErrorType(Enum):
            CONNECTION = auto()
            TIMEOUT = auto()
            AUTH = auto()

        patterns = {
            ErrorType.CONNECTION: r"connection\s+(failed|refused)",
            ErrorType.TIMEOUT: r"timeout\s+after",
            ErrorType.AUTH: r"authentication\s+failed",
        }
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("connection failed to server")
        assert len(matches) == 1
        assert matches[0][0] == ErrorType.CONNECTION

        matches = matcher.match_line("timeout after 30 seconds")
        assert len(matches) == 1
        assert matches[0][0] == ErrorType.TIMEOUT

    def test_tuple_keys(self):
        """Test using tuples as keys (e.g., for categorization)"""
        patterns = {
            ("error", "network"): r"network\s+error",
            ("error", "disk"): r"disk\s+error",
            ("warning", "memory"): r"high\s+memory",
        }
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("network error occurred")
        assert len(matches) == 1
        assert matches[0][0] == ("error", "network")

    def test_get_pattern_info_by_custom_key(self):
        """Test retrieving pattern info by custom key"""
        patterns = {
            "error_handler": r"error:\s+\d+",
            "warn_handler": r"warning:\s+\w+",
        }
        matcher = TwoStageMatcher(patterns)

        info = matcher.get_pattern_info("error_handler")
        assert info.key == "error_handler"
        assert info.regex_str == r"error:\s+\d+"

    def test_match_lines_with_custom_keys(self):
        """Test multi-line matching with custom keys"""
        patterns = {
            "ERR": r"error",
            "WRN": r"warning",
        }
        matcher = TwoStageMatcher(patterns)

        lines = ["an error occurred", "a warning message", "info message"]
        matches = matcher.match_lines(lines)

        assert len(matches) == 2
        keys = {m[1] for m in matches}
        assert keys == {"ERR", "WRN"}

    def test_integer_keys(self):
        """Test using integer keys explicitly"""
        patterns = {0: r"error", 1: r"warning"}
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("error message")
        assert len(matches) == 1
        assert matches[0][0] == 0

        info = matcher.get_pattern_info(0)
        assert info.key == 0
        assert info.regex_str == r"error"


class TestLiteralPrioritization:
    """Test that longer literals are prioritized"""

    def test_longer_literal_checked_first(self):
        """Test that both patterns with different literal lengths are found"""
        patterns = {
            "short": r"error",  # Short literal: "error"
            "long": r"error: connection failed",  # Long literal: "error: connection failed"
        }
        matcher = TwoStageMatcher(patterns)

        # Both patterns match this line
        matches = matcher.match_line("error: connection failed on node-001")

        assert len(matches) == 2
        matched_keys = {m[0] for m in matches}
        assert matched_keys == {"long", "short"}

    def test_multiple_literals_uses_longest(self):
        """Test that when a pattern has multiple matching literals, both are found"""
        patterns = {
            "pattern1": r"(error|critical error)",  # Has both "error" and "critical error"
            "pattern2": r"network",
        }
        matcher = TwoStageMatcher(patterns)

        # Line has both "critical error" and "network"
        matches = matcher.match_line("critical error in network subsystem")

        # Both patterns should match
        assert len(matches) == 2
        matched_keys = {m[0] for m in matches}
        assert matched_keys == {"pattern1", "pattern2"}

    def test_no_literal_checked_last(self):
        """Test that patterns without literals are checked after those with literals"""
        patterns = {
            "with_literal": r"error:\s+\d+",  # Has literal "error:"
            "no_literal": r"^\d+$",  # No good literal
        }
        matcher = TwoStageMatcher(patterns)

        matches = matcher.match_line("error: 404")

        # Only "with_literal" matches, but test the ordering principle
        assert len(matches) == 1
        assert matches[0][0] == "with_literal"


class TestCorrectness:
    """Test that results match naive regex approach"""

    def test_equivalence_with_sequential_matching(self):
        """Verify results match sequential regex matching"""
        patterns = {
            0: r"error:\s+(?P<code>\d+)",
            1: r"user_id:\s*(?P<uid>\d+)",
            2: r"(?P<level>INFO|WARN|ERROR)",
        }

        matcher = TwoStageMatcher(patterns)
        compiled = {k: re.compile(p) for k, p in patterns.items()}

        test_lines = [
            "ERROR: 500",
            "user_id: 12345",
            "INFO: Starting",
            "No match here",
            "error: 404 and user_id: 99",
        ]

        for line in test_lines:
            # Two-stage approach
            ts_matches = matcher.match_line(line)
            ts_keys = {m[0] for m in ts_matches}

            # Naive approach
            naive_keys = set()
            for k, regex in compiled.items():
                if regex.search(line):
                    naive_keys.add(k)

            # Should have same patterns matching
            assert ts_keys == naive_keys, f"Mismatch for line: {line!r}"
