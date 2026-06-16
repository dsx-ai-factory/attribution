#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Extract literal substrings from Python regexes for use as pre-filters.

This module provides functions to:
1. Parse regexes and find all literal substrings that MUST appear
2. Score literals by length and uniqueness across a corpus
3. Select the best literal for pre-filtering with Aho-Corasick
"""

import re
import sre_constants
import sre_parse
from collections import Counter
from typing import Any, Optional

LiteralScore = tuple[Optional[str], float]


def extract_literals_from_regex(pattern: str) -> list[str]:
    """
    Extract all literal substrings that MUST appear in a match.

    This parses the regex and finds sequences of literal characters,
    avoiding optional constructs, alternations, and character classes.

    Args:
        pattern: The regex pattern string

    Returns:
        list of literal strings found in the pattern
    """
    try:
        parsed = sre_parse.parse(pattern)
    except re.error:
        # If regex parsing fails, try to extract simple literals with a fallback
        return _fallback_literal_extraction(pattern)

    literals: list[str] = []
    _extract_from_parsed(parsed, literals)
    return literals


def _extract_from_parsed(parsed: Any, literals: list[str], current: str = "") -> list[str]:
    """
    Recursively extract literals from parsed regex structure.

    Args:
        parsed: Parsed regex from sre_parse
        literals: list to accumulate found literals
        current: Current literal string being built
    """
    for op, av in parsed:
        if op == sre_constants.LITERAL:
            # Add literal character
            char = chr(av)
            current += char

        elif op == sre_constants.NOT_LITERAL:
            # Not a literal - flush current and skip
            if current:
                literals.append(current)
                current = ""

        elif op in (sre_constants.SUBPATTERN, sre_constants.GROUPREF_EXISTS):
            # Recurse into subpattern
            if current:
                literals.append(current)
                current = ""
            if isinstance(av, tuple) and len(av) > 1:
                _extract_from_parsed(av[-1], literals, "")

        elif op == sre_constants.BRANCH:
            # Alternation (a|b) - we can't guarantee which branch
            # Flush current and recursively extract from each branch
            if current:
                literals.append(current)
                current = ""
            # av is (None, [branch1, branch2, ...])
            for branch in av[1]:
                _extract_from_parsed(branch, literals, "")

        elif op in (sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT):
            # Repetition: {m,n}, *, +, ?
            min_count, _max_count, subpattern = av
            if current:
                literals.append(current)
                current = ""

            # Only recurse if the subpattern is required (min_count > 0)
            if min_count > 0:
                _extract_from_parsed(subpattern, literals, "")

        elif op == sre_constants.IN:
            # Character class [...] - not a literal
            if current:
                literals.append(current)
                current = ""

        elif op in (
            sre_constants.ANY,
            sre_constants.AT,
            sre_constants.ASSERT,
            sre_constants.ASSERT_NOT,
            sre_constants.AT_BEGINNING,
            sre_constants.AT_END,
        ):
            # Non-literal constructs - flush current
            if current:
                literals.append(current)
                current = ""

        else:
            # Unknown or other constructs - be conservative
            if current:
                literals.append(current)
                current = ""

    # Flush any remaining literal
    if current:
        literals.append(current)

    return literals


def _fallback_literal_extraction(pattern: str) -> list[str]:
    """
    Fallback: extract simple literal sequences if parsing fails.

    This is a simple approach that finds sequences of non-special characters.
    """
    # Remove common regex escapes and find remaining text
    # This is imperfect but works for simple patterns
    special_chars = r"\.^$*+?{}[]()|\\"

    literals = []
    current = ""

    i = 0
    while i < len(pattern):
        char = pattern[i]

        if char == "\\" and i + 1 < len(pattern):
            # Handle escape sequences
            next_char = pattern[i + 1]
            if next_char in "dDwWsS":
                # Character class - not literal
                if current:
                    literals.append(current)
                    current = ""
                i += 2
                continue
            elif next_char in special_chars:
                # Escaped special char - treat as literal
                current += next_char
                i += 2
                continue
            else:
                # Other escape - skip
                if current:
                    literals.append(current)
                    current = ""
                i += 2
                continue
        elif char in special_chars:
            # Special character - not literal
            if current:
                literals.append(current)
                current = ""
            i += 1
        else:
            # Regular character - add to current literal
            current += char
            i += 1

    if current:
        literals.append(current)

    return literals


def score_literals_by_length(literals: list[str]) -> list[tuple[str, float]]:
    """
    Score literals by length (longer is better).

    Returns:
        list of (literal, score) tuples sorted by score descending
    """
    scored: list[tuple[str, float]] = [(lit, float(len(lit))) for lit in literals]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def score_literals_by_uniqueness(pattern_literals: dict[int, list[str]]) -> dict[int, LiteralScore]:
    """
    Score literals across a corpus by uniqueness.

    Uses a TF-IDF (term frequency–inverse document frequency)-like approach:
    - Longer literals get higher base scores
    - Literals that appear in fewer patterns get boosted
    - Common literals (appearing in many patterns) get penalized

    Args:
        pattern_literals: dict mapping pattern_id -> list of literal strings

    Returns:
        dict mapping pattern_id -> (best_literal, score)
    """
    # Count how many patterns each literal appears in
    literal_pattern_count: Counter[str] = Counter()
    for _pattern_id, literals in pattern_literals.items():
        unique_literals = set(literals)
        for lit in unique_literals:
            literal_pattern_count[lit] += 1

    total_patterns = len(pattern_literals)
    best_for_pattern: dict[int, LiteralScore] = {}

    for pattern_id, literals in pattern_literals.items():
        if not literals:
            best_for_pattern[pattern_id] = (None, 0.0)
            continue

        # Score each literal
        scored: list[tuple[str, float]] = []
        for lit in literals:
            # Base score: length (favor longer strings)
            length_score = len(lit)

            # Uniqueness score: inverse document frequency
            pattern_count = literal_pattern_count[lit]
            uniqueness_score = total_patterns / pattern_count

            # Combined score (you can adjust the weights)
            combined_score = length_score * uniqueness_score
            scored.append((lit, combined_score))

        # Pick the best
        scored.sort(key=lambda x: x[1], reverse=True)
        best_for_pattern[pattern_id] = scored[0]

    return best_for_pattern


def extract_best_literal(pattern: str) -> Optional[str]:
    """
    Extract the single best literal substring from a regex pattern.

    Args:
        pattern: Regex pattern string

    Returns:
        The best literal string, or None if no literals found
    """
    literals = extract_literals_from_regex(pattern)
    if not literals:
        return None

    # Score by length
    scored = score_literals_by_length(literals)
    return scored[0][0] if scored else None


def extract_best_literals_from_corpus(patterns: list[str], strategy: str = "uniqueness") -> dict[int, Optional[str]]:
    """
    Extract the best literal from each pattern in a corpus.

    Args:
        patterns: list of regex pattern strings
        strategy: "length" or "uniqueness"

    Returns:
        dict mapping pattern index -> best literal string
    """

    # Extract all literals
    pattern_literals = {}
    for i, pattern in enumerate(patterns):
        literals = extract_literals_from_regex(pattern)
        pattern_literals[i] = literals

    if strategy == "uniqueness":
        # Score by uniqueness across corpus
        best = score_literals_by_uniqueness(pattern_literals)
        return {i: lit for i, (lit, _score) in best.items()}
    else:
        # Score by length only
        result: dict[int, Optional[str]] = {}
        for i, literals in pattern_literals.items():
            if literals:
                scored = score_literals_by_length(literals)
                result[i] = scored[0][0]
            else:
                result[i] = None
        return result


def demonstrate_extraction() -> None:
    """Demonstrate the literal extraction on various regex patterns"""

    test_patterns = [
        (r"error", "Simple literal"),
        (r"ERROR:\s+\d+", "Literal with special chars"),
        (r"user_id:\s*(?P<id>\d+)", "Named group with literal prefix"),
        (r"(failed|failure)", "Alternation"),
        (r"connection\s+timeout", "Literal with whitespace"),
        (r"\d{3}-\d{3}-\d{4}", "Phone number (no literals)"),
        (r"(?:log|error):\s*(.+)", "Non-capturing group with alternation"),
        (r"^ERROR.*database.*", "Start anchor with literals"),
        (r"file\.txt", "Escaped dot"),
        (r"(pre_)?process(ing|ed)?", "Optional prefix and suffix"),
    ]

    print("=" * 80)
    print("LITERAL EXTRACTION EXAMPLES")
    print("=" * 80)

    for pattern, description in test_patterns:
        print(f"\nPattern: {pattern!r}")
        print(f"Description: {description}")
        literals = extract_literals_from_regex(pattern)
        print(f"Literals found: {literals}")
        best = extract_best_literal(pattern)
        print(f"Best literal: {best!r}")

    # Demonstrate corpus-level uniqueness scoring
    print("\n" + "=" * 80)
    print("CORPUS UNIQUENESS SCORING")
    print("=" * 80)

    corpus_patterns = [
        r"long_common_error:\s+\d+",
        r"long_common_error:\s+connection",
        r"long_common_error:\s+database",
        r"warning:\s+\d+",
        r"info:\s+started",
    ]

    print("\nPatterns in corpus:")
    for i, p in enumerate(corpus_patterns):
        print(f"  {i}: {p!r}")

    best_by_length = extract_best_literals_from_corpus(corpus_patterns, strategy="length")
    print("\nBest by length:")
    for i, lit in best_by_length.items():
        print(f"  Pattern {i}: {lit!r}")

    best_by_uniqueness = extract_best_literals_from_corpus(corpus_patterns, strategy="uniqueness")
    print("\nBest by uniqueness (penalizes common 'error'):")
    for i, lit in best_by_uniqueness.items():
        print(f"  Pattern {i}: {lit!r}")


if __name__ == "__main__":
    demonstrate_extraction()
