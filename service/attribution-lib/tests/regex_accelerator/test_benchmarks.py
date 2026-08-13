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
Performance benchmarks for regex_accelerator using pytest-benchmark.

These benchmarks compare the two-stage matcher against naive sequential
regex matching with realistic workloads based on NVIDIA Attribution service patterns.

Run with: uv pytest tests/test_benchmarks.py --benchmark-only
"""

import random
import re
from typing import List

import pytest
from attribution_lib.regex_accelerator import TwoStageMatcher

# Realistic patterns from NVIDIA Attribution
ATTRIBUTION_PATTERNS = {
    0: r"\[(?P<isotime>[^\]]+)\]\s+iteration\s+\d+/\s*\d+.*TFLOP/s/GPU\):\s*\d+\.\d+",
    1: r"socketStartConnect: Connect to (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> failed : (?P<reason>.*)",
    2: r"socketPollConnect: Connect to (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> returned \d+\((?P<reason>[^)]+)\) errno",
    3: r"socketProgressOpt: Call to recv from (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> failed : (?P<reason>.*)",
    4: r"IPv4 network addresses of \((?P<hostname>[^,]+), \d+\) cannot be retrieved",
    5: r"client socket has timed out after \d+ms while trying to connect to \((?P<hostname>[^,]+), \d+\)",
    6: r"Connection closed by remote peer (?P<hostname>\S+)[^<]*<",
    7: r"(?:AttributeError|TypeError): (?:'[\w\d]+' object has no attribute '[\w\d]+'|object of type '[\w\d]+' has no [\w\d()]+)",
    8: r"KeyValidationError: ListConfig indices must be .+, not .+",
    9: r"OFI Request .* completed with error\. RC: (?P<error_code>\d+)",
    10: r"Got completion from peer (?P<peer_ip>\d+\.\d+\.\d+\.\d+).* with \w+[= ](?P<error_code>\d+).*vendor err (?P<vendor_err>\d+)",
    11: r"Call to ibv_modify_qp failed with (?P<error_code>\d+) .* on dev (?P<local_hca>[^,\s]+)",
    12: r"Caught signal \d+ \(Bus error: nonexistent physical address\)",
    13: r"NCCL WARN Cuda failure",
    14: r"Segmentation fault",
    15: r"terminate called",
    16: r"Disk quota exceeded",
    17: r"torch\.OutOfMemoryError",
    18: r"CUDA error: out of memory",
    19: r"No such file or directory",
}


# Sample log lines (mix of matching and non-matching)
NON_MATCHING_LINES = [
    "INFO: Starting training job",
    "DEBUG: Processing batch 42",
    "INFO: Checkpoint saved at iteration 1000",
    "WARNING: High memory usage detected: 85%",
    "INFO: Model loaded successfully",
    "DEBUG: Learning rate: 0.001",
    "INFO: Epoch 5/100 completed",
    "DEBUG: Gradient norm: 1.234",
    "INFO: Validation accuracy: 92.5%",
    "DEBUG: Cache hit ratio: 78%",
    "INFO: Training resumed from checkpoint",
    "DEBUG: Data loader initialized",
    "INFO: Metrics logged to wandb",
    "DEBUG: Memory allocated: 12GB",
    "INFO: Forward pass completed",
]

MATCHING_LINES = [
    "[2024-01-15T10:30:45] iteration 100/1000 | throughput (TFLOP/s/GPU): 245.6",
    "socketStartConnect: Connect to 192.168.1.100<12345> failed : Connection refused",
    "socketPollConnect: Connect to 10.0.0.5<8080> returned 110(Connection timed out) errno",
    "IPv4 network addresses of (gpu-node-005, 12345) cannot be retrieved",
    "client socket has timed out after 5000ms while trying to connect to (node-042, 8080)",
    "Connection closed by remote peer node-123.cluster.com<5555>",
    "AttributeError: 'NoneType' object has no attribute 'shape'",
    "KeyValidationError: ListConfig indices must be integers, not str",
    "OFI Request 0x7f1234567890 completed with error. RC: 5",
    "Got completion from peer 192.168.10.5:12345 with error=5 status failed vendor err 129",
    "Call to ibv_modify_qp failed with 22 (Invalid argument) on dev mlx5_0",
    "Caught signal 7 (Bus error: nonexistent physical address)",
    "NCCL WARN Cuda failure 'out of memory'",
    "Segmentation fault (core dumped)",
    "terminate called after throwing an instance",
    "Disk quota exceeded on /lustre/fs1",
    "torch.OutOfMemoryError: CUDA out of memory",
    "CUDA error: out of memory",
    "No such file or directory: /path/to/file",
]


def generate_test_lines(num_lines: int, match_rate: float = 0.05) -> List[str]:
    """
    Generate test lines with specified match rate.

    Args:
        num_lines: Total number of lines to generate
        match_rate: Fraction of lines that will match (0.0 to 1.0)

    Returns:
        List of test lines
    """
    lines = []
    num_matching = int(num_lines * match_rate)
    num_non_matching = num_lines - num_matching

    # Add matching lines
    for _ in range(num_matching):
        lines.append(random.choice(MATCHING_LINES))

    # Add non-matching lines
    for _ in range(num_non_matching):
        lines.append(random.choice(NON_MATCHING_LINES))

    # Shuffle to distribute matches throughout
    random.shuffle(lines)

    return lines


@pytest.fixture
def compiled_patterns():
    """Pre-compiled regex patterns for naive approach"""
    return {k: re.compile(p) for k, p in ATTRIBUTION_PATTERNS.items()}


@pytest.fixture
def two_stage_matcher():
    """Pre-built two-stage matcher"""
    return TwoStageMatcher(ATTRIBUTION_PATTERNS)


# =============================================================================
# Small Scale Benchmarks (baseline)
# =============================================================================


class TestSmallScaleBenchmarks:
    """Benchmarks with small datasets (100 lines)"""

    @pytest.fixture
    def small_dataset_5pct(self):
        """100 lines, 5% match rate"""
        random.seed(42)
        return generate_test_lines(100, match_rate=0.05)

    def test_naive_small_5pct(self, benchmark, compiled_patterns, small_dataset_5pct):
        """Naive: 100 lines, 5% match rate"""

        def naive_match():
            matches = []
            for line_num, line in enumerate(small_dataset_5pct, 1):
                for pattern_id, regex in compiled_patterns.items():
                    if regex.search(line):
                        matches.append((line_num, pattern_id))
            return matches

        result = benchmark(naive_match)
        assert len(result) > 0

    def test_two_stage_small_5pct(self, benchmark, two_stage_matcher, small_dataset_5pct):
        """Two-stage: 100 lines, 5% match rate"""
        result = benchmark(two_stage_matcher.match_lines, small_dataset_5pct)
        assert len(result) > 0


# =============================================================================
# Medium Scale Benchmarks (realistic workload)
# =============================================================================


class TestMediumScaleBenchmarks:
    """Benchmarks with medium datasets (1,000 lines)"""

    @pytest.fixture
    def medium_dataset_5pct(self):
        """1,000 lines, 5% match rate (typical log file)"""
        random.seed(42)
        return generate_test_lines(1000, match_rate=0.05)

    @pytest.fixture
    def medium_dataset_20pct(self):
        """1,000 lines, 20% match rate (high error rate)"""
        random.seed(42)
        return generate_test_lines(1000, match_rate=0.20)

    def test_naive_medium_5pct(self, benchmark, compiled_patterns, medium_dataset_5pct):
        """Naive: 1,000 lines, 5% match rate"""

        def naive_match():
            matches = []
            for line_num, line in enumerate(medium_dataset_5pct, 1):
                for pattern_id, regex in compiled_patterns.items():
                    if regex.search(line):
                        matches.append((line_num, pattern_id))
            return matches

        result = benchmark(naive_match)
        assert len(result) > 0

    def test_two_stage_medium_5pct(self, benchmark, two_stage_matcher, medium_dataset_5pct):
        """Two-stage: 1,000 lines, 5% match rate"""
        result = benchmark(two_stage_matcher.match_lines, medium_dataset_5pct)
        assert len(result) > 0

    def test_naive_medium_20pct(self, benchmark, compiled_patterns, medium_dataset_20pct):
        """Naive: 1,000 lines, 20% match rate"""

        def naive_match():
            matches = []
            for line_num, line in enumerate(medium_dataset_20pct, 1):
                for pattern_id, regex in compiled_patterns.items():
                    if regex.search(line):
                        matches.append((line_num, pattern_id))
            return matches

        result = benchmark(naive_match)
        assert len(result) > 0

    def test_two_stage_medium_20pct(self, benchmark, two_stage_matcher, medium_dataset_20pct):
        """Two-stage: 1,000 lines, 20% match rate"""
        result = benchmark(two_stage_matcher.match_lines, medium_dataset_20pct)
        assert len(result) > 0


# =============================================================================
# Large Scale Benchmarks (stress test)
# =============================================================================


class TestLargeScaleBenchmarks:
    """Benchmarks with large datasets (10,000 lines) - stress test"""

    @pytest.fixture
    def large_dataset_1pct(self):
        """10,000 lines, 1% match rate (mostly clean logs)"""
        random.seed(42)
        return generate_test_lines(10000, match_rate=0.01)

    @pytest.fixture
    def large_dataset_5pct(self):
        """10,000 lines, 5% match rate (typical)"""
        random.seed(42)
        return generate_test_lines(10000, match_rate=0.05)

    @pytest.fixture
    def large_dataset_10pct(self):
        """10,000 lines, 10% match rate (high error scenario)"""
        random.seed(42)
        return generate_test_lines(10000, match_rate=0.10)

    def test_naive_large_1pct(self, benchmark, compiled_patterns, large_dataset_1pct):
        """Naive: 10,000 lines, 1% match rate"""

        def naive_match():
            matches = []
            for line_num, line in enumerate(large_dataset_1pct, 1):
                for pattern_id, regex in compiled_patterns.items():
                    if regex.search(line):
                        matches.append((line_num, pattern_id))
            return matches

        result = benchmark(naive_match)
        assert len(result) > 0

    def test_two_stage_large_1pct(self, benchmark, two_stage_matcher, large_dataset_1pct):
        """Two-stage: 10,000 lines, 1% match rate"""
        result = benchmark(two_stage_matcher.match_lines, large_dataset_1pct)
        assert len(result) > 0

    def test_naive_large_5pct(self, benchmark, compiled_patterns, large_dataset_5pct):
        """Naive: 10,000 lines, 5% match rate"""

        def naive_match():
            matches = []
            for line_num, line in enumerate(large_dataset_5pct, 1):
                for pattern_id, regex in compiled_patterns.items():
                    if regex.search(line):
                        matches.append((line_num, pattern_id))
            return matches

        result = benchmark(naive_match)
        assert len(result) > 0

    def test_two_stage_large_5pct(self, benchmark, two_stage_matcher, large_dataset_5pct):
        """Two-stage: 10,000 lines, 5% match rate"""
        result = benchmark(two_stage_matcher.match_lines, large_dataset_5pct)
        assert len(result) > 0

    def test_naive_large_10pct(self, benchmark, compiled_patterns, large_dataset_10pct):
        """Naive: 10,000 lines, 10% match rate"""

        def naive_match():
            matches = []
            for line_num, line in enumerate(large_dataset_10pct, 1):
                for pattern_id, regex in compiled_patterns.items():
                    if regex.search(line):
                        matches.append((line_num, pattern_id))
            return matches

        result = benchmark(naive_match)
        assert len(result) > 0

    def test_two_stage_large_10pct(self, benchmark, two_stage_matcher, large_dataset_10pct):
        """Two-stage: 10,000 lines, 10% match rate"""
        result = benchmark(two_stage_matcher.match_lines, large_dataset_10pct)
        assert len(result) > 0


# =============================================================================
# Very Large Scale Benchmarks (extreme stress test)
# =============================================================================


@pytest.mark.slow
class TestVeryLargeScaleBenchmarks:
    """Benchmarks with very large datasets (100,000 lines) - extreme stress test"""

    @pytest.fixture
    def very_large_dataset_1pct(self):
        """100,000 lines, 1% match rate"""
        random.seed(42)
        return generate_test_lines(100000, match_rate=0.01)

    @pytest.fixture
    def very_large_dataset_5pct(self):
        """100,000 lines, 5% match rate"""
        random.seed(42)
        return generate_test_lines(100000, match_rate=0.05)

    def test_naive_very_large_1pct(self, benchmark, compiled_patterns, very_large_dataset_1pct):
        """Naive: 100,000 lines, 1% match rate"""

        def naive_match():
            matches = []
            for line_num, line in enumerate(very_large_dataset_1pct, 1):
                for pattern_id, regex in compiled_patterns.items():
                    if regex.search(line):
                        matches.append((line_num, pattern_id))
            return matches

        result = benchmark.pedantic(naive_match, rounds=3, iterations=1)
        assert len(result) > 0

    def test_two_stage_very_large_1pct(self, benchmark, two_stage_matcher, very_large_dataset_1pct):
        """Two-stage: 100,000 lines, 1% match rate"""
        result = benchmark.pedantic(
            two_stage_matcher.match_lines, args=(very_large_dataset_1pct,), rounds=3, iterations=1
        )
        assert len(result) > 0

    def test_naive_very_large_5pct(self, benchmark, compiled_patterns, very_large_dataset_5pct):
        """Naive: 100,000 lines, 5% match rate"""

        def naive_match():
            matches = []
            for line_num, line in enumerate(very_large_dataset_5pct, 1):
                for pattern_id, regex in compiled_patterns.items():
                    if regex.search(line):
                        matches.append((line_num, pattern_id))
            return matches

        result = benchmark.pedantic(naive_match, rounds=3, iterations=1)
        assert len(result) > 0

    def test_two_stage_very_large_5pct(self, benchmark, two_stage_matcher, very_large_dataset_5pct):
        """Two-stage: 100,000 lines, 5% match rate"""
        result = benchmark.pedantic(
            two_stage_matcher.match_lines, args=(very_large_dataset_5pct,), rounds=3, iterations=1
        )
        assert len(result) > 0


# =============================================================================
# Pattern Scaling Benchmarks
# =============================================================================


class TestPatternScalingBenchmarks:
    """Test how performance scales with number of patterns"""

    @pytest.fixture
    def medium_lines(self):
        """1,000 lines for pattern scaling tests"""
        random.seed(42)
        return generate_test_lines(1000, match_rate=0.05)

    def test_naive_10_patterns(self, benchmark, medium_lines):
        """Naive: 1,000 lines, 10 patterns"""
        patterns = {k: v for k, v in list(ATTRIBUTION_PATTERNS.items())[:10]}
        compiled = {k: re.compile(p) for k, p in patterns.items()}

        def naive_match():
            matches = []
            for line_num, line in enumerate(medium_lines, 1):
                for pattern_id, regex in compiled.items():
                    if regex.search(line):
                        matches.append((line_num, pattern_id))
            return matches

        benchmark(naive_match)

    def test_two_stage_10_patterns(self, benchmark, medium_lines):
        """Two-stage: 1,000 lines, 10 patterns"""
        patterns = {k: v for k, v in list(ATTRIBUTION_PATTERNS.items())[:10]}
        matcher = TwoStageMatcher(patterns)
        benchmark(matcher.match_lines, medium_lines)

    def test_naive_20_patterns(self, benchmark, medium_lines):
        """Naive: 1,000 lines, 20 patterns (all)"""
        compiled = {k: re.compile(p) for k, p in ATTRIBUTION_PATTERNS.items()}

        def naive_match():
            matches = []
            for line_num, line in enumerate(medium_lines, 1):
                for pattern_id, regex in compiled.items():
                    if regex.search(line):
                        matches.append((line_num, pattern_id))
            return matches

        benchmark(naive_match)

    def test_two_stage_20_patterns(self, benchmark, medium_lines):
        """Two-stage: 1,000 lines, 20 patterns (all)"""
        matcher = TwoStageMatcher(ATTRIBUTION_PATTERNS)
        benchmark(matcher.match_lines, medium_lines)
