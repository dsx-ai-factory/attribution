# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
High-performance multi-pattern regex matching with Aho-Corasick pre-filtering.

This package preserves full Python regex behavior, including named capture
groups, while reducing the amount of full regex evaluation needed when matching
many patterns against many lines.

How it works:
- Stage 1: extract required literal substrings and index them with Aho-Corasick
- Stage 2: run the full Python regex only on candidate lines

Quick start:
    >>> from attribution_lib.regex_accelerator import TwoStageMatcher
    >>> patterns = {
    ...     0: r"ERROR:\\s+(?P<code>\\d+)",
    ...     1: r"user_id:\\s*(?P<uid>\\d+)",
    ... }
    >>> matcher = TwoStageMatcher(patterns)
    >>> matches = matcher.match_line("ERROR: 500")
    >>> matches[0][1].groupdict()
    {'code': '500'}

Tests and benchmarks:
- `uv run pytest tests/regex_accelerator/ -v`
- `uv run pytest tests/regex_accelerator/test_benchmarks.py --benchmark-only`
"""

from .regex_literal_extractor import (
    extract_best_literal,
    extract_best_literals_from_corpus,
    extract_literals_from_regex,
)
from .two_stage_matcher import RegexPattern, TwoStageMatcher

__all__ = [
    "TwoStageMatcher",
    "RegexPattern",
    "extract_literals_from_regex",
    "extract_best_literal",
    "extract_best_literals_from_corpus",
]
