# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from unittest.mock import patch

import pytest

import attribution_lib
from attribution_lib.attr_types import JobNodeAttributions, _normalize_node
from attribution_lib.sources import SourceSet
from attribution_lib.utils import (
    applog_extract_node_rank,
    convert_hostname_to_short_format,
    get_hostname,
    matches_ignored_hostname_prefix,
    syslog_extract_node,
)


class TestGetHostname:
    """Test cases for the get_hostname function."""

    def test_hostname_is_shortened_without_dns(self):
        """A hostname is already an identifier and does not require DNS."""
        assert get_hostname("yard95.example.com") == "yard95"

    def test_known_ip_mapping(self):
        """Test IP to hostname mapping from known dictionary."""
        with patch.dict("attribution_lib.ip_to_nodenames", {"192.168.1.1": "node1"}):
            assert get_hostname("192.168.1.1") == "node1"

    def test_unmapped_ip_is_returned_without_dns(self):
        """An unmapped IP remains unresolved without querying DNS."""
        with patch.dict("attribution_lib.ip_to_nodenames", {}, clear=True):
            assert get_hostname("192.168.1.1") == "192.168.1.1"

    def test_invalid_hostname_with_non_printable_char_returns_input(self):
        """Test that a hostname containing non-printable bytes (e.g. from corrupted log data) is returned as-is."""
        invalid_hostname = "node\x90invalid"
        assert get_hostname(invalid_hostname) == invalid_hostname

    def test_mapping_changes_are_visible_immediately(self):
        """Resolution must not retain stale results when the mapping changes."""
        with patch.dict("attribution_lib.ip_to_nodenames", {}, clear=True):
            assert get_hostname("192.168.1.2") == "192.168.1.2"
            with patch.dict("attribution_lib.ip_to_nodenames", {"192.168.1.2": "node2"}):
                assert get_hostname("192.168.1.2") == "node2"

    def test_ipv6_known_mapping(self):
        """Test IPv6 to hostname mapping from known dictionary."""
        with patch.dict("attribution_lib.ip_to_nodenames", {"2001:db8::1": "node1"}):
            assert get_hostname("2001:db8::1") == "node1"

    def test_ipv6_unmapped_returns_ip(self):
        """An unmapped IPv6 address remains unresolved."""
        with patch.dict("attribution_lib.ip_to_nodenames", {}, clear=True):
            assert get_hostname("2001:db8::2") == "2001:db8::2"


class TestConvertHostnameToShortFormat:
    """Test cases for the convert_hostname_to_short_format function."""

    @pytest.mark.parametrize(
        "input_hostname,expected",
        [
            ("192.168.1.1", "192.168.1.1"),
            ("10.0.0.1", "10.0.0.1"),
            ("172.16.0.1", "172.16.0.1"),
        ],
    )
    def test_ipv4_addresses_unchanged(self, input_hostname, expected):
        """Test that IPv4 addresses are returned unchanged."""
        assert convert_hostname_to_short_format(input_hostname) == expected

    @pytest.mark.parametrize(
        "input_hostname,expected",
        [
            ("2001:db8::1", "2001:db8::1"),
            ("2001:0db8:85a3::8a2e:0370:7334", "2001:0db8:85a3::8a2e:0370:7334"),
            ("::1", "::1"),
            ("fe80::1", "fe80::1"),
        ],
    )
    def test_ipv6_addresses_unchanged(self, input_hostname, expected):
        """Test that IPv6 addresses are returned unchanged."""
        assert convert_hostname_to_short_format(input_hostname) == expected

    @pytest.mark.parametrize(
        "input_hostname,expected",
        [
            ("test-node-001.us-east4-b.c.test-infra-20241015.internal", "test-node-001"),
            ("node1.cluster.internal", "node1"),
        ],
    )
    def test_internal_hostnames_shortened(self, input_hostname, expected):
        """Test that .internal hostnames are shortened."""
        assert convert_hostname_to_short_format(input_hostname) == expected

    @pytest.mark.parametrize(
        "input_hostname,expected",
        [
            ("simple-hostname", "simple-hostname"),
            ("node.example.com", "node"),
        ],
    )
    def test_regular_hostnames(self, input_hostname, expected):
        """Test regular hostname handling."""
        assert convert_hostname_to_short_format(input_hostname) == expected


class TestMatchesIgnoredHostnamePrefix:
    """Ignored prefixes match short hostnames and full IPs, not IPv4 first octets."""

    def test_hostname_prefix_matches_short_and_fqdn(self):
        prefixes = ["scanner-"]
        assert matches_ignored_hostname_prefix("scanner-001", prefixes)
        assert matches_ignored_hostname_prefix("scanner-001.example.com", prefixes)
        assert not matches_ignored_hostname_prefix("compute-node-003", prefixes)

    def test_unmapped_ipv4_matches_full_address_not_first_octet(self):
        """Splitting 192.0.2.1 on dots yields '192' and never matches the address."""
        prefixes = ["192.0.2.1"]
        assert matches_ignored_hostname_prefix("192.0.2.1", prefixes)
        assert not matches_ignored_hostname_prefix("192.0.2.1", ["scanner-"])
        assert "192.0.2.1".split(".")[0] == "192"
        assert not "192".startswith("192.0.2.1")

    def test_unmapped_ipv6_matches_full_address(self):
        prefixes = ["2001:db8::1"]
        assert matches_ignored_hostname_prefix("2001:db8::1", prefixes)
        assert not matches_ignored_hostname_prefix("2001:db8::2", prefixes)

    def test_defaults_to_library_prefix_list(self):
        with patch.object(attribution_lib, "ignored_hostname_prefixes", ["scanner-"]):
            assert matches_ignored_hostname_prefix("scanner-001")
            assert not matches_ignored_hostname_prefix("compute-node-003")
        with patch.object(attribution_lib, "ignored_hostname_prefixes", []):
            assert not matches_ignored_hostname_prefix("scanner-001")


class TestApplogExtractNodeRank:
    """Test cases for the applog_extract_node_rank function."""

    def test_direct_node_name(self):
        """Test extraction of direct node name."""
        assert applog_extract_node_rank("node1: some error") == ("node1", None)

    @pytest.mark.parametrize(
        "log_line,expected_rank",
        [
            ("[rank3]: some error", 3),
            ("[rank: 3] some error", 3),
            ("[rank:3] some error", 3),
            ("[rank:3]: some error", 3),
        ],
    )
    def test_rank_based_extraction(self, log_line, expected_rank):
        """Test various rank-based extraction patterns."""
        assert applog_extract_node_rank(log_line) == (None, expected_rank)

    @pytest.mark.parametrize(
        "log_line",
        [
            "terminate called after throwing...",
            "wandb: error occurred",
            "Writing temporary files back into original files",
            "invalid rank: error",
        ],
    )
    def test_special_cases_return_rank_zero(self, log_line):
        """Test that special cases return rank 0."""
        assert applog_extract_node_rank(log_line) == (None, 0)

    def test_empty_string(self):
        """Test empty string handling."""
        assert applog_extract_node_rank("") == ("", None)


class TestNormalizeNode:
    """Test cases for _normalize_node in attr_types."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("TEST-NODE-ALPHA<33751>", "test-node-alpha"),
            ('TEST-NODE-ALPHA<33751>\n"', "test-node-alpha"),
            ("TEST-NODE-BRAVO.example.test<16048>", "test-node-bravo.example.test"),
            ("host-001<999>extra-garbage", "host-001"),
            ("TEST-NODE-ALPHA", "test-node-alpha"),
            ("simple-host", "simple-host"),
        ],
    )
    def test_strips_nccl_pid_suffix(self, raw, expected):
        assert _normalize_node(raw) == expected

    def test_empty_after_strip_returns_original(self):
        """If stripping would produce an empty string, return the original."""
        assert _normalize_node("<12345>") == "<12345>"


class TestJobNodeAttributionsNormalization:
    """Verify that add() and update() normalize NCCL <pid> suffixes."""

    def test_add_normalizes_pid_suffix(self):
        """Dirty node name in add() should be stored under the clean key."""
        jna = JobNodeAttributions()
        attr = SourceSet.new_one("Connection closed", None)
        jna.add(["TEST-NODE-ALPHA<33751>"], attr)

        assert "test-node-alpha" in jna.attributions
        assert "TEST-NODE-ALPHA<33751>" not in jna.attributions
        assert "TEST-NODE-ALPHA" not in jna.attributions

    def test_add_merges_clean_and_dirty_into_one_key(self):
        """A clean entry and a dirty entry for the same host should merge."""
        jna = JobNodeAttributions()
        attr1 = SourceSet.new_one("Route Unhealthy", None)
        attr2 = SourceSet.new_one("Connection closed", None)

        jna.add(["TEST-NODE-ALPHA"], attr1)
        jna.add(["TEST-NODE-ALPHA<33751>"], attr2)

        assert jna.attributions.keys() == {"test-node-alpha"}
        symptoms = set(jna.attributions["test-node-alpha"].symptoms())
        assert "Route Unhealthy" in symptoms
        assert "Connection closed" in symptoms

    def test_add_merges_different_casing(self):
        """Different casings of the same hostname should merge into one key."""
        jna = JobNodeAttributions()
        attr1 = SourceSet.new_one("NCCL invalid NIC selection", None)
        attr2 = SourceSet.new_one("Connection closed", None)

        jna.add(["test-node-charlie"], attr1)
        jna.add(["TEST-NODE-CHARLIE"], attr2)

        assert len(jna.attributions) == 1
        symptoms = set(jna.attributions["test-node-charlie"].symptoms())
        assert "NCCL invalid NIC selection" in symptoms
        assert "Connection closed" in symptoms

    def test_update_normalizes_pid_suffix(self):
        """Dirty node name arriving via update() should be normalized."""
        jna_src = JobNodeAttributions()
        attr = SourceSet.new_one("Connection closed", None)
        jna_src.add(["TEST-NODE-ALPHA<33751>"], attr)

        jna_dst = JobNodeAttributions()
        jna_dst.update(jna_src)

        assert "test-node-alpha" in jna_dst.attributions
        assert "TEST-NODE-ALPHA<33751>" not in jna_dst.attributions


class TestSyslogExtractNode:
    """Test cases for the syslog_extract_node function."""

    @pytest.mark.parametrize(
        "log_line,expected_hostname",
        [
            ("Mar 15 10:30:45 hostname program[123]: message", "hostname"),
            ("Mar 15 10:30:45 hostname program: message", "hostname"),
        ],
    )
    def test_valid_syslog_formats(self, log_line, expected_hostname):
        """Test valid syslog format extraction."""
        assert syslog_extract_node(log_line) == expected_hostname

    @pytest.mark.parametrize(
        "log_line",
        [
            "This is not a syslog message",
            "",
        ],
    )
    def test_invalid_formats_return_none(self, log_line):
        """Test that invalid formats return None."""
        assert syslog_extract_node(log_line) is None
