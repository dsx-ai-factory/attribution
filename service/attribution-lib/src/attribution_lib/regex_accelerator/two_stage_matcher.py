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
Two-stage regex matcher using Aho-Corasick pre-filtering.

Stage 1: Use Aho-Corasick to quickly find lines containing literal substrings
Stage 2: Apply full Python regexes (with capturing groups) only to candidates

"""

import re
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional, overload

import ahocorasick

from .regex_literal_extractor import (
    extract_best_literals_from_corpus,
    extract_literals_from_regex,
)

AutomatonValue = tuple[str, list[Hashable]]


@dataclass
class RegexPattern:
    """Represents a regex pattern with its literal pre-filter"""

    key: Hashable  # User-provided hashable identifier for this pattern
    regex_str: str
    compiled: re.Pattern[str]
    literal: Optional[str]


class TwoStageMatcher:
    """
    Efficient multi-regex matcher using Aho-Corasick pre-filtering.

    This dramatically speeds up matching when:
    1. You have many regex patterns
    2. Most lines don't match any pattern
    3. Your regexes contain literal substrings
    """

    def __init__(
        self,
        patterns: Mapping[Any, str],
        uniqueness_strategy: str = "uniqueness",
        min_literal_length: int = 4,
    ):
        """
        Initialize the two-stage matcher.

        Args:
            patterns: Dict mapping keys to regex pattern strings.
                      Keys can be any hashable type (str, int, enum, tuple, etc.)
                      Example: {"error": r"error:\\s+\\d+", "warning": r"warning:\\s+\\w+"}
            uniqueness_strategy: "uniqueness" or "length" for literal selection
            min_literal_length: minimum literal length to use (default 4)
        """
        self.patterns: dict[Hashable, RegexPattern] = {}
        self.automaton: ahocorasick.Automaton[AutomatonValue] = ahocorasick.Automaton()
        self.has_automaton = False

        # Extract just the pattern strings for corpus analysis
        pattern_strings = list(patterns.values())

        # Use corpus-level analysis to find best literals considering uniqueness
        # The indices in best_literals_map correspond to positions in pattern_strings list
        best_literals_map = extract_best_literals_from_corpus(pattern_strings, strategy=uniqueness_strategy)

        # Build patterns and Aho-Corasick automaton
        literal_to_patterns: dict[str, list[Hashable]] = {}

        def add_literal(literal: str, key: Hashable) -> None:
            """Helper to add a literal->pattern mapping"""
            if literal not in literal_to_patterns:
                literal_to_patterns[literal] = []
            if key not in literal_to_patterns[literal]:
                literal_to_patterns[literal].append(key)

        for idx, (key, pattern_str) in enumerate(patterns.items()):
            # Get the best literal from corpus analysis (indexed by position in list)
            best_literal = best_literals_map.get(idx)

            # Filter out literals that are too short
            if best_literal and len(best_literal) < min_literal_length:
                best_literal = None

            compiled = re.compile(pattern_str)

            regex_pattern = RegexPattern(
                key=key,
                regex_str=pattern_str,
                compiled=compiled,
                literal=best_literal,
            )
            self.patterns[key] = regex_pattern

            # Add literal(s) to automaton
            # Strategy: Use the single best literal UNLESS we need multiple
            # alternation branches to cover all cases

            if best_literal and len(best_literal) >= min_literal_length:
                # Check if we need multiple literals for alternation branches
                all_literals = extract_literals_from_regex(pattern_str)
                good_literals = [lit for lit in all_literals if len(lit) >= min_literal_length]

                # Identify substantial literals (double the minimum)
                substantial_threshold = min_literal_length * 2
                substantial_literals = [lit for lit in good_literals if len(lit) >= substantial_threshold]

                # Strategy 1: If we have 2+ substantial literals, add all of them
                # (likely different branches in a complex pattern)
                if len(substantial_literals) >= 2:
                    for literal in substantial_literals:
                        add_literal(literal, key)

                # Strategy 2: For smaller literals, check if they're similar length
                # (simple alternation like INFO|DEBUG)
                elif len(good_literals) >= 2:
                    lengths = [len(lit) for lit in good_literals]
                    max_len = max(lengths)
                    min_len = min(lengths)
                    ratio = max_len / max(min_len, 1)

                    # Similar lengths suggest simple alternation
                    if ratio < 2.5:
                        # Add all good literals
                        for literal in good_literals:
                            add_literal(literal, key)
                    else:
                        # One literal is dominant - use just the best
                        add_literal(best_literal, key)
                else:
                    # Single good literal - use it
                    add_literal(best_literal, key)

            elif best_literal is None or len(best_literal) < min_literal_length:
                # No good single literal - fall back to adding any good literals we find
                all_literals = extract_literals_from_regex(pattern_str)
                good_literals = [lit for lit in all_literals if len(lit) >= min_literal_length]

                # Add all good literals (alternation fallback)
                if good_literals:
                    for literal in good_literals:
                        add_literal(literal, key)

        # Add literals to Aho-Corasick automaton
        for literal, pattern_keys in literal_to_patterns.items():
            # Store the pattern keys that use this literal
            self.automaton.add_word(literal, (literal, pattern_keys))

        # Finalize the automaton (only if we added any words)
        if literal_to_patterns:
            self.automaton.make_automaton()
            self.has_automaton = True

        # Patterns with no literal need to be checked on every line
        self.always_check = [p.key for p in self.patterns.values() if p.literal is None]

    def candidates(self, line: str) -> set[Hashable]:
        """
        Stage 1 only: find candidate pattern keys using Aho-Corasick pre-filter.

        Returns the set of pattern keys whose literal substrings appear in the line,
        plus any patterns that have no extractable literal (always_check).
        The caller is responsible for confirming candidates with full regex if needed.

        Args:
            line: The line to check

        Returns:
            set of candidate pattern keys
        """
        result: set[Hashable] = set()
        if self.has_automaton:
            for _, (_literal, pattern_keys) in self.automaton.iter(line):
                result.update(pattern_keys)
        result.update(self.always_check)
        return result

    def confirm_match(self, key: Hashable, line: str) -> Optional[re.Match[str]]:
        """
        Stage 2: confirm a candidate with full regex match.

        Args:
            key: The pattern key (must be a valid key in this matcher)
            line: The line to match against

        Returns:
            Match object if the full regex matches, None otherwise
        """
        return self.patterns[key].compiled.search(line)

    def match_line(self, line: str) -> list[tuple[Hashable, re.Match[str]]]:
        """
        Match a line against all patterns (both stages).

        Args:
            line: The line to match

        Returns:
            list of (key, Match object) tuples for all matches
        """
        return [(key, match) for key in self.candidates(line) if (match := self.confirm_match(key, line))]

    @overload
    def match_lines(
        self, lines: list[str], return_line_nums: Literal[True] = True
    ) -> list[tuple[int, Hashable, re.Match[str]]]: ...

    @overload
    def match_lines(
        self, lines: list[str], return_line_nums: Literal[False]
    ) -> list[tuple[Hashable, re.Match[str]]]: ...

    def match_lines(
        self, lines: list[str], return_line_nums: bool = True
    ) -> list[tuple[int, Hashable, re.Match[str]]] | list[tuple[Hashable, re.Match[str]]]:
        """
        Match multiple lines against all patterns.

        Args:
            lines: list of lines to match
            return_line_nums: If True, include line numbers in results

        Returns:
            If return_line_nums=True: list of (line_num, key, Match) tuples
            If return_line_nums=False: list of (key, Match) tuples
        """
        if return_line_nums:
            return [
                (line_num, key, match) for line_num, line in enumerate(lines, 1) for key, match in self.match_line(line)
            ]
        else:
            return [(key, match) for line in lines for key, match in self.match_line(line)]

    def get_pattern_info(self, key: Hashable) -> RegexPattern:
        """Get information about a pattern by its key"""
        return self.patterns[key]

    def debug_info(self) -> dict[str, Any]:
        """Get debugging information about the matcher"""
        patterns_with_literals = sum(1 for p in self.patterns.values() if p.literal)
        patterns_without_literals = len(self.patterns) - patterns_with_literals

        # Get what's actually in the automaton (not just pattern.literal)
        automaton_literals: dict[str, list[Hashable]] = {}
        if self.has_automaton:
            for _, (literal, pattern_keys) in self.automaton.items():
                if literal not in automaton_literals:
                    automaton_literals[literal] = []
                automaton_literals[literal].extend(pattern_keys)

        # Also track the "best" literal chosen for each pattern
        pattern_best_literals: dict[Hashable, str] = {}
        for pattern in self.patterns.values():
            if pattern.literal:
                pattern_best_literals[pattern.key] = pattern.literal

        return {
            "total_patterns": len(self.patterns),
            "patterns_with_literals": patterns_with_literals,
            "patterns_without_literals": patterns_without_literals,
            "unique_literals_in_automaton": len(automaton_literals),
            "automaton_literal_usage": automaton_literals,
            "pattern_best_literals": pattern_best_literals,
        }


def demo_with_named_groups() -> None:
    """Demonstrate the two-stage matcher with named groups"""

    print("=" * 80)
    print("TWO-STAGE MATCHER DEMO WITH NAMED GROUPS")
    print("=" * 80)

    # Example patterns with named groups
    patterns = {
        0: r"ERROR:\s+(?P<code>\d+)\s+-\s+(?P<message>.+)",
        1: r"user_id:\s*(?P<user_id>\d+)",
        2: r"(?P<level>ERROR|WARNING|INFO):\s+(?P<text>.+)",
        3: r"connection\s+(?P<status>established|failed|timeout)",
        4: r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
    }

    # Test lines
    test_lines = [
        "INFO: Application started successfully",
        "ERROR: 500 - Internal server error",
        "User logged in with user_id: 12345",
        "DEBUG: Cache hit for key user_123",
        "connection established with 192.168.1.1",
        "2024-01-15 10:30:45 - Processing request",
        "WARNING: High memory usage detected",
        "connection timeout after 30 seconds",
    ]

    print("\nPatterns:")
    for i, p in enumerate(patterns):
        print(f"  {i}: {p!r}")

    print("\nBuilding matcher...")
    matcher = TwoStageMatcher(patterns, uniqueness_strategy="uniqueness")

    print("\nMatcher info:")
    debug_info = matcher.debug_info()
    print(f"  Total patterns: {debug_info['total_patterns']}")
    print(f"  Patterns with literals: {debug_info['patterns_with_literals']}")
    print(f"  Patterns without literals (always checked): {debug_info['patterns_without_literals']}")
    print(f"  Unique literals in Aho-Corasick automaton: {debug_info['unique_literals_in_automaton']}")
    print("\nBest literal per pattern:")
    for pattern in matcher.patterns.values():
        print(f"  Pattern {pattern.key}: {pattern.literal!r}")
    print("\nAll literals in automaton:")
    for literal, pattern_keys in debug_info["automaton_literal_usage"].items():
        print(f"  {literal!r} → patterns {pattern_keys}")

    print("\n" + "=" * 80)
    print("MATCHING RESULTS")
    print("=" * 80)

    matches = matcher.match_lines(test_lines)

    for line_num, key, match in matches:
        line = test_lines[line_num - 1]
        pattern = matcher.get_pattern_info(key)

        print(f"\nLine {line_num}: {line!r}")
        print(f"  Pattern {key}: {pattern.regex_str!r}")
        print(f"  Full match: {match.group(0)!r}")

        # Show named groups if present
        if match.groupdict():
            print("  Named groups:")
            for name, value in match.groupdict().items():
                print(f"    {name}: {value!r}")


def benchmark_comparison() -> None:
    """Benchmark: Naive vs Two-Stage approach"""
    import time

    # Generate test data
    patterns = {
        0: r"error:\s+\d+",
        1: r"ERROR:\s+(?P<code>\d+)",
        2: r"warning:\s+(?P<message>.+)",
        3: r"user_id:\s*(?P<uid>\d+)",
        4: r"connection\s+(failed|timeout)",
        5: r"(?P<level>INFO|DEBUG):\s+.+",
        6: r"exception:\s+(?P<type>\w+)",
        7: r"timeout\s+after\s+(?P<seconds>\d+)",
        8: r"memory\s+usage:\s+(?P<percent>\d+)%",
        9: r"request\s+from\s+(?P<ip>\d{1,3}(\.\d{1,3}){3})",
    }

    # Generate test lines (mostly non-matching)
    import random

    templates = [
        "INFO: Request processed successfully",
        "DEBUG: Cache hit for key user_123",
        "TRACE: Function called with args=[]",
        "Performance metric: latency=50ms",
        "Query executed in 10ms",
    ]
    matching_templates = [
        "ERROR: 500",
        "warning: high CPU usage",
        "user_id: 12345",
        "connection failed",
        "exception: ValueError",
    ]

    # 90% non-matching, 10% matching
    test_lines = []
    for _ in range(1000000):
        if random.random() < 0.9:
            test_lines.append(random.choice(templates))
        else:
            test_lines.append(random.choice(matching_templates))

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"\nTest data: {len(test_lines):,} lines, {len(patterns)} patterns")

    # Method 1: Naive approach
    print("\n1. Naive approach (try every regex on every line)...")
    compiled = {k: re.compile(p) for k, p in patterns.items()}

    start = time.time()
    naive_matches = []
    for line_num, line in enumerate(test_lines, 1):
        for key, regex in compiled.items():
            match = regex.search(line)
            if match:
                naive_matches.append((line_num, key, match))
    naive_time = time.time() - start

    print(f"   Time: {naive_time:.3f}s")
    print(f"   Matches: {len(naive_matches):,}")
    print(f"   Throughput: {len(test_lines) / naive_time:,.0f} lines/sec")

    # Method 2: Two-stage approach
    print("\n2. Two-stage approach (Aho-Corasick + regex)...")

    start = time.time()
    matcher = TwoStageMatcher(patterns, uniqueness_strategy="uniqueness")
    compile_time = time.time() - start

    # Show matcher stats
    debug_info = matcher.debug_info()
    print(f"   Patterns with literals: {debug_info['patterns_with_literals']}/{debug_info['total_patterns']}")
    print(f"   Unique literals in automaton: {debug_info['unique_literals_in_automaton']}")

    start = time.time()
    two_stage_matches = matcher.match_lines(test_lines)
    scan_time = time.time() - start
    total_time = compile_time + scan_time

    print(f"   Compile time: {compile_time:.3f}s")
    print(f"   Scan time: {scan_time:.3f}s")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Matches: {len(two_stage_matches):,}")
    print(f"   Throughput: {len(test_lines) / scan_time:,.0f} lines/sec")

    # Comparison
    print("\n" + "=" * 80)
    print(f"SPEEDUP: {naive_time / scan_time:.1f}x faster (scanning only)")
    print(f"Time saved: {naive_time - scan_time:.3f}s")
    print(f"\nNote: Two-stage has {compile_time:.3f}s one-time compile overhead")
    print(f"Break-even point: ~{compile_time / (naive_time / len(test_lines)):.0f} lines")

    # Verify results match
    naive_set = {(ln, key) for ln, key, _ in naive_matches}
    two_stage_set = {(ln, key) for ln, key, _ in two_stage_matches}
    if naive_set == two_stage_set:
        print("\n✓ Results match between both methods")
    else:
        print("\n⚠ Results differ!")
        print(f"   Only in naive: {len(naive_set - two_stage_set)}")
        print(f"   Only in two-stage: {len(two_stage_set - naive_set)}")


if __name__ == "__main__":
    demo_with_named_groups()
    benchmark_comparison()
