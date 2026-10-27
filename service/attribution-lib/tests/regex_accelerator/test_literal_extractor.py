# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for regex literal extraction"""

from attribution_lib.regex_accelerator import (
    extract_best_literal,
    extract_best_literals_from_corpus,
    extract_literals_from_regex,
)


class TestLiteralExtraction:
    """Test extraction of literal substrings from regexes"""

    def test_simple_literal(self):
        """Test extraction from simple literal pattern"""
        literals = extract_literals_from_regex(r"error")
        assert "error" in literals

    def test_literal_with_special_chars(self):
        """Test extraction when regex has special characters"""
        literals = extract_literals_from_regex(r"ERROR:\s+\d+")
        assert "ERROR:" in literals

    def test_named_group_with_literal_prefix(self):
        """Test extraction from pattern with named capture group"""
        literals = extract_literals_from_regex(r"user_id:\s*(?P<id>\d+)")
        assert "user_id:" in literals

    def test_alternation(self):
        """Test extraction from alternation patterns"""
        literals = extract_literals_from_regex(r"(failed|failure)")
        # Should extract literals from branches
        assert any("fail" in lit for lit in literals)

    def test_literal_with_whitespace_pattern(self):
        """Test extraction when whitespace patterns are present"""
        literals = extract_literals_from_regex(r"connection\s+timeout")
        assert "connection" in literals
        assert "timeout" in literals

    def test_no_literals_pattern(self):
        """Test pattern with no extractable literals"""
        literals = extract_literals_from_regex(r"^\d+$")
        # Should either be empty or contain only very short literals
        assert all(len(lit) <= 1 for lit in literals)

    def test_escaped_special_char(self):
        """Test extraction of escaped special characters"""
        literals = extract_literals_from_regex(r"file\.txt")
        assert "file.txt" in literals

    def test_optional_constructs(self):
        """Test that optional parts are handled correctly"""
        literals = extract_literals_from_regex(r"(pre_)?process(ing|ed)?")
        # Should extract 'process' which is required
        assert "process" in literals

    def test_character_class_not_extracted(self):
        """Test that character classes are not extracted as literals"""
        literals = extract_literals_from_regex(r"[abc]\d+error")
        # Should extract 'error' but not the character class
        assert "error" in literals
        assert not any("[abc]" in lit for lit in literals)

    def test_quantified_literal(self):
        """Test literals before quantifiers"""
        literals = extract_literals_from_regex(r"ERROR:+")
        # Should extract 'ERROR:' or 'ERROR' (the colon might be included)
        assert any("ERROR" in lit for lit in literals)

    def test_anchors_ignored(self):
        """Test that anchors don't prevent literal extraction"""
        literals = extract_literals_from_regex(r"^ERROR.*database.*$")
        assert "ERROR" in literals
        assert "database" in literals

    def test_empty_pattern(self):
        """Test handling of empty pattern"""
        literals = extract_literals_from_regex(r"")
        assert literals == []

    def test_complex_nested_groups(self):
        """Test extraction from complex nested groups"""
        literals = extract_literals_from_regex(r"(?:(?:error|warn):)\s*(\d+)")
        # Should extract literals from nested groups
        assert any(lit in ["error", "warn"] for lit in literals)


class TestBestLiteralSelection:
    """Test selection of best literal from a pattern"""

    def test_longest_literal_selected(self):
        """Test that longest literal is preferred"""
        best = extract_best_literal(r"a\s+longer_literal\s+b")
        assert best == "longer_literal"

    def test_single_literal(self):
        """Test pattern with only one literal"""
        best = extract_best_literal(r"error:\s+\d+")
        assert best == "error:"

    def test_no_literals_returns_none(self):
        """Test that None is returned when no literals exist"""
        best = extract_best_literal(r"^\d+$")
        # Should return None or a very short/empty literal
        assert best is None or len(best) <= 1

    def test_multiple_equal_length_literals(self):
        """Test that one is chosen when multiple literals have same length"""
        best = extract_best_literal(r"abc\s+def")
        # Should return one of them
        assert best in ["abc", "def"]


class TestCorpusLiteralExtraction:
    """Test extraction across multiple patterns (corpus)"""

    def test_uniqueness_strategy(self):
        """Test that uniqueness strategy penalizes common literals"""
        patterns = [
            r"error:\s+connection",
            r"error:\s+database",
            r"warning:\s+timeout",
        ]
        best = extract_best_literals_from_corpus(patterns, strategy="uniqueness")

        # 'error:' appears in 2 patterns, so unique literals should be preferred
        # Pattern 0 might choose 'connection' over 'error:'
        # Pattern 2 should choose 'warning:' or 'timeout'
        assert 0 in best
        assert 1 in best
        assert 2 in best

    def test_length_strategy(self):
        """Test that length strategy picks longest literal per pattern"""
        patterns = [
            r"a\s+longer",
            r"b\s+shorter",
        ]
        best = extract_best_literals_from_corpus(patterns, strategy="length")

        assert best[0] == "longer"
        assert best[1] == "shorter"

    def test_empty_corpus(self):
        """Test handling of empty pattern corpus"""
        best = extract_best_literals_from_corpus([])
        assert best == {}

    def test_corpus_with_no_literals(self):
        """Test corpus where patterns have no literals"""
        patterns = [r"^\d+$", r"^\w+$"]
        best = extract_best_literals_from_corpus(patterns)
        # Should return something for each pattern, possibly None
        assert 0 in best
        assert 1 in best

    def test_mixed_patterns_with_and_without_literals(self):
        """Test corpus with mix of patterns with/without literals"""
        patterns = [
            r"error:\s+\d+",  # Has literal
            r"^\d+$",  # No literal
            r"user_id:\s*\d+",  # Has literal
        ]
        best = extract_best_literals_from_corpus(patterns)

        assert best[0] == "error:"
        assert best[2] == "user_id:"
        # Pattern 1 should return None or very short literal
        assert best[1] is None or len(best[1]) <= 1

    def test_shared_literals(self):
        """Test patterns that share the same literal"""
        patterns = [
            r"error:\s+timeout",
            r"error:\s+connection",
        ]
        best = extract_best_literals_from_corpus(patterns, strategy="uniqueness")

        # Both patterns share 'error:', so uniqueness should favor
        # the unique parts: 'timeout' and 'connection'
        assert best[0] in ["error:", "timeout"]
        assert best[1] in ["error:", "connection"]


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_unicode_pattern(self):
        """Test pattern with unicode characters"""
        literals = extract_literals_from_regex(r"errör:\s+\d+")
        assert "errör:" in literals

    def test_very_long_literal(self):
        """Test extraction of very long literal"""
        long_literal = "a" * 1000
        literals = extract_literals_from_regex(long_literal)
        assert long_literal in literals

    def test_special_regex_sequences(self):
        """Test patterns with special regex sequences"""
        patterns = [
            r"\bword\b",  # Word boundaries
            r"(?i)case",  # Case insensitive flag
            r"(?:group)",  # Non-capturing group
        ]
        for pattern in patterns:
            literals = extract_literals_from_regex(pattern)
            # Should extract some literals without erroring
            assert isinstance(literals, list)

    def test_invalid_regex_handled(self):
        """Test that invalid regex doesn't crash"""
        # Use fallback extraction for invalid patterns
        literals = extract_literals_from_regex(r"[invalid")
        # Should not crash, might return empty or use fallback
        assert isinstance(literals, list)

    def test_lookahead_lookbehind(self):
        """Test patterns with lookahead/lookbehind assertions"""
        literals = extract_literals_from_regex(r"(?<=prefix)literal(?=suffix)")
        # Should extract 'literal'
        assert "literal" in literals

    def test_backreferences(self):
        """Test pattern with backreferences"""
        literals = extract_literals_from_regex(r"(word)\s+\1")
        # Should extract 'word'
        assert "word" in literals
