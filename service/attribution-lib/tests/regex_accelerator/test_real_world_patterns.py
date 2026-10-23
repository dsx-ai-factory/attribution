# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests using real-world regex patterns from NVIDIA Attribution.

These patterns are extracted from applog_rules.py which analyzes
GPU training job failures in HPC clusters.
"""

from attribution_lib.regex_accelerator import TwoStageMatcher

# Real-world patterns from NVIDIA Attribution applog_rules.py
ATTRIBUTION_PATTERNS = {
    "mfu_iteration": r"\[(?P<isotime>[^\]]+)\]\s+iteration\s+\d+/\s*\d+.*TFLOP/s/GPU\):\s*\d+\.\d+",
    "socket_start_connect": r"socketStartConnect: Connect to (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> failed : (?P<reason>.*)",
    "socket_poll_connect": r"socketPollConnect: Connect to (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> returned \d+\((?P<reason>[^)]+)\) errno",
    "socket_progress_opt": r"socketProgressOpt: Call to recv from (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> failed : (?P<reason>.*)",
    "ipv4_retrieval_error": r"IPv4 network addresses of \((?P<hostname>[^,]+), \d+\) cannot be retrieved",
    "socket_timeout": r"client socket has timed out after \d+ms while trying to connect to \((?P<hostname>[^,]+), \d+\)",
    "connection_closed": r"Connection closed by remote peer (?P<hostname>\S+)[^<]*<",
    "python_exception": r"(?:AttributeError|TypeError): (?:'[\w\d]+' object has no attribute '[\w\d]+'|object of type '[\w\d]+' has no [\w\d()]+)",
    "key_validation_error": r"KeyValidationError: ListConfig indices must be .+, not .+",
    "ofi_error": r"OFI Request .* completed with error\. RC: (?P<error_code>\d+)",
    "ib_completion_error": r"Got completion from peer (?P<peer_ip>\d+\.\d+\.\d+\.\d+).* with \w+[= ](?P<error_code>\d+).*vendor err (?P<vendor_err>\d+)",
    "ibv_modify_qp_error": r"Call to ibv_modify_qp failed with (?P<error_code>\d+) .* on dev (?P<local_hca>[^,\s]+)",
    "bus_error": r"Caught signal \d+ \(Bus error: nonexistent physical address\)",
}


class TestRealWorldAttributionPatterns:
    """Test using actual NVIDIA Attribution patterns"""

    def test_create_matcher_with_rule_patterns(self):
        """Test that matcher can be created with complex real-world patterns"""
        matcher = TwoStageMatcher(ATTRIBUTION_PATTERNS)

        assert len(matcher.patterns) == len(ATTRIBUTION_PATTERNS)

        # Check that literals were extracted
        patterns_with_literals = sum(1 for p in matcher.patterns.values() if p.literal)
        # Most patterns should have extractable literals
        assert patterns_with_literals > len(ATTRIBUTION_PATTERNS) // 2

    def test_mfu_pattern_matching(self):
        """Test MFU (Model FLOPs Utilization) iteration pattern"""
        patterns = {0: r"\[(?P<isotime>[^\]]+)\]\s+iteration\s+\d+/\s*\d+.*TFLOP/s/GPU\):\s*\d+\.\d+"}
        matcher = TwoStageMatcher(patterns)

        test_line = "[2024-01-15T10:30:45] iteration 100/1000 | throughput (TFLOP/s/GPU): 245.6"

        matches = matcher.match_line(test_line)
        assert len(matches) == 1

        _, match = matches[0]
        assert match.group("isotime") == "2024-01-15T10:30:45"

    def test_socket_connect_failure(self):
        """Test socket connection failure pattern"""
        patterns = {0: r"socketStartConnect: Connect to (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> failed : (?P<reason>.*)"}
        matcher = TwoStageMatcher(patterns)

        test_line = "socketStartConnect: Connect to 192.168.1.100<12345> failed : Connection refused"

        matches = matcher.match_line(test_line)
        assert len(matches) == 1

        _, match = matches[0]
        assert match.group("ip") == "192.168.1.100"
        assert match.group("reason") == "Connection refused"

    def test_multiple_socket_patterns(self):
        """Test multiple socket-related patterns"""
        patterns = {
            0: r"socketStartConnect: Connect to (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> failed : (?P<reason>.*)",
            1: r"socketPollConnect: Connect to (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> returned \d+\((?P<reason>[^)]+)\) errno",
            2: r"socketProgressOpt: Call to recv from (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> failed : (?P<reason>.*)",
        }
        matcher = TwoStageMatcher(patterns)

        test_lines = [
            ("socketStartConnect: Connect to 10.0.0.1<5000> failed : Timeout", 0),
            ("socketPollConnect: Connect to 10.0.0.2<5001> returned 110(Connection timed out) errno", 1),
            ("socketProgressOpt: Call to recv from 10.0.0.3<5002> failed : Network unreachable", 2),
        ]

        for line, expected_pattern in test_lines:
            matches = matcher.match_line(line)
            assert len(matches) == 1
            assert matches[0][0] == expected_pattern

    def test_hostname_patterns(self):
        """Test hostname extraction patterns"""
        patterns = {
            0: r"IPv4 network addresses of \((?P<hostname>[^,]+), \d+\) cannot be retrieved",
            1: r"client socket has timed out after \d+ms while trying to connect to \((?P<hostname>[^,]+), \d+\)",
        }
        matcher = TwoStageMatcher(patterns)

        test_cases = [
            ("IPv4 network addresses of (node-001.cluster.com, 12345) cannot be retrieved", 0, "node-001.cluster.com"),
            (
                "client socket has timed out after 5000ms while trying to connect to (gpu-node-05, 8080)",
                1,
                "gpu-node-05",
            ),
        ]

        for line, expected_pattern, expected_hostname in test_cases:
            matches = matcher.match_line(line)
            assert len(matches) == 1
            assert matches[0][0] == expected_pattern
            assert matches[0][1].group("hostname") == expected_hostname

    def test_python_exception_patterns(self):
        """Test Python exception patterns"""
        patterns = {
            0: r"(?:AttributeError|TypeError): (?:'[\w\d]+' object has no attribute '[\w\d]+'|object of type '[\w\d]+' has no [\w\d()]+)",
        }
        matcher = TwoStageMatcher(patterns)

        test_lines = [
            "AttributeError: 'NoneType' object has no attribute 'shape'",
            "TypeError: object of type 'int' has no len()",
        ]

        for line in test_lines:
            matches = matcher.match_line(line)
            assert len(matches) == 1, f"Failed to match: {line}"

    def test_infiniband_error_pattern(self):
        """Test InfiniBand completion error pattern"""
        patterns = {
            0: r"Got completion from peer (?P<peer_ip>\d+\.\d+\.\d+\.\d+).* with \w+[= ](?P<error_code>\d+).*vendor err (?P<vendor_err>\d+)",
        }
        matcher = TwoStageMatcher(patterns)

        test_line = "Got completion from peer 192.168.10.5:12345 with error=5 status failed vendor err 129"

        matches = matcher.match_line(test_line)
        assert len(matches) == 1

        _, match = matches[0]
        assert match.group("peer_ip") == "192.168.10.5"
        assert match.group("error_code") == "5"
        assert match.group("vendor_err") == "129"

    def test_ibv_modify_qp_pattern(self):
        """Test IB verb modify QP failure pattern"""
        patterns = {
            0: r"Call to ibv_modify_qp failed with (?P<error_code>\d+) .* on dev (?P<local_hca>[^,\s]+)",
        }
        matcher = TwoStageMatcher(patterns)

        test_line = "Call to ibv_modify_qp failed with 22 (Invalid argument) on dev mlx5_0"

        matches = matcher.match_line(test_line)
        assert len(matches) == 1

        _, match = matches[0]
        assert match.group("error_code") == "22"
        assert match.group("local_hca") == "mlx5_0"

    def test_efa_error_pattern(self):
        """Test EFA/OFI error pattern"""
        patterns = {0: r"OFI Request .* completed with error\. RC: (?P<error_code>\d+)"}
        matcher = TwoStageMatcher(patterns)

        test_line = "OFI Request 0x7f1234567890 completed with error. RC: 5"

        matches = matcher.match_line(test_line)
        assert len(matches) == 1

        _, match = matches[0]
        assert match.group("error_code") == "5"

    def test_bus_error_pattern(self):
        """Test bus error signal pattern"""
        patterns = {0: r"Caught signal \d+ \(Bus error: nonexistent physical address\)"}
        matcher = TwoStageMatcher(patterns)

        test_line = "Caught signal 7 (Bus error: nonexistent physical address)"

        matches = matcher.match_line(test_line)
        assert len(matches) == 1


class TestAttributionPerformance:
    """Test performance with Attribution-like workloads"""

    def test_many_rule_patterns_performance(self):
        """Test matching with many real-world patterns"""
        # Use all Attribution patterns
        matcher = TwoStageMatcher(ATTRIBUTION_PATTERNS)

        # Simulate log lines (mix of matching and non-matching)
        test_lines = [
            "INFO: Starting training job",
            "[2024-01-15T10:30:45] iteration 100/1000 | throughput (TFLOP/s/GPU): 245.6",
            "socketStartConnect: Connect to 192.168.1.100<12345> failed : Connection refused",
            "DEBUG: Processing batch 42",
            "WARNING: High memory usage detected",
            "Got completion from peer 10.0.0.5:8080 with error=5 status failed vendor err 129",
            "INFO: Checkpoint saved",
            "AttributeError: 'NoneType' object has no attribute 'shape'",
            "Call to ibv_modify_qp failed with 22 (Invalid argument) on dev mlx5_0",
            "INFO: Training completed",
        ] * 10  # 100 lines total

        matches = matcher.match_lines(test_lines)

        # Should find matches for the error lines
        assert len(matches) > 0

        # Verify no false positives on INFO/DEBUG lines
        info_line_nums = {
            i + 1 for i, line in enumerate(test_lines) if line.startswith("INFO:") or line.startswith("DEBUG:")
        }
        match_line_nums = {m[0] for m in matches}
        assert not (info_line_nums & match_line_nums), "False positives on INFO/DEBUG lines"

    def test_literal_extraction_quality(self):
        """Test that good literals are extracted from Attribution patterns"""
        matcher = TwoStageMatcher(ATTRIBUTION_PATTERNS)

        # Check debug info
        info = matcher.debug_info()

        # Most patterns should have literals
        assert info["patterns_with_literals"] > info["total_patterns"] * 0.6

        # Check some specific expected literals
        literals = [p.literal for p in matcher.patterns.values() if p.literal]

        # Should extract good literals like "socketStartConnect", "AttributeError", etc.
        assert any("socket" in lit.lower() for lit in literals)
        assert any("error" in lit.lower() for lit in literals)

    def test_alternation_in_python_errors(self):
        """Test that alternation in error patterns works correctly"""
        patterns = {
            0: r"(?:AttributeError|TypeError): (?:'[\w\d]+' object has no attribute '[\w\d]+'|object of type '[\w\d]+' has no [\w\d()]+)",
        }
        matcher = TwoStageMatcher(patterns)

        # Both AttributeError and TypeError should match
        test_cases = [
            "AttributeError: 'dict' object has no attribute 'get_value'",
            "TypeError: object of type 'str' has no len()",
        ]

        for line in test_cases:
            matches = matcher.match_line(line)
            assert len(matches) == 1, f"Pattern should match: {line}"


class TestAttributionMatchCorrectness:
    """Verify correctness against naive regex matching"""

    def test_equivalence_with_sequential_matching(self):
        """Verify Attribution patterns give same results as naive approach"""
        import re

        # Use a subset of Attribution patterns
        patterns = {
            0: r"socketStartConnect: Connect to (?P<ip>\d+\.\d+\.\d+\.\d+)<\d+> failed : (?P<reason>.*)",
            1: r"AttributeError: '[\w\d]+' object has no attribute '[\w\d]+'",
            2: r"OFI Request .* completed with error\. RC: (?P<error_code>\d+)",
        }

        matcher = TwoStageMatcher(patterns)
        compiled = {k: re.compile(p) for k, p in patterns.items()}

        test_lines = [
            "INFO: Normal log line",
            "socketStartConnect: Connect to 10.0.0.1<5000> failed : Timeout",
            "AttributeError: 'NoneType' object has no attribute 'value'",
            "DEBUG: Another normal line",
            "OFI Request 0x123456 completed with error. RC: -1",
            "WARNING: Memory usage high",
        ]

        for line in test_lines:
            # Two-stage approach
            ts_matches = matcher.match_line(line)
            ts_pattern_ids = {m[0] for m in ts_matches}

            # Naive approach
            naive_pattern_ids = set()
            for k, regex in compiled.items():
                if regex.search(line):
                    naive_pattern_ids.add(k)

            # Should have same patterns matching
            assert ts_pattern_ids == naive_pattern_ids, (
                f"Mismatch for line: {line!r}\n  Two-stage: {ts_pattern_ids}\n  Naive: {naive_pattern_ids}"
            )
