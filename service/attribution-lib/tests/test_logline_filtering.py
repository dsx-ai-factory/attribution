# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the logline filtering logger utility.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from attribution_lib.exceptions import InvalidRankError, UnidentifiableNodeError, UnknownNodeError, UnknownResourceError
from attribution_lib.logline_filtering import log_ignored_content_line, log_ignored_node_line, log_ignored_time_line
from attribution_lib.observation.inbound import Source


@pytest.fixture
def logline_jobinfo(simple_job_factory):
    """JobInfo fixture for logline filtering tests."""
    return simple_job_factory(job_id=12345, nodes=["job-node-001", "job-node-002"])


@pytest.fixture
def minimal_jobinfo(simple_job_factory):
    """JobInfo fixture with minimal nodes for testing edge cases."""
    return simple_job_factory(job_id=12345, nodes=["single-node"])


@pytest.fixture
def mock_entry():
    """Mock entry fixture for node filtering tests."""
    entry = Mock()
    entry.body.return_value = "Mar 15 10:30:45 invalid-node-999 kernel: Test log message"
    return entry


@pytest.fixture
def mock_entry_content():
    """Mock entry fixture for content filtering tests."""
    entry = Mock()
    entry.body.return_value = "This log contains prolog error information"
    return entry


@pytest.fixture
def mock_entry_with_time(timezone):
    """Mock entry fixture with time for time filtering tests."""
    entry = Mock()
    entry.body.return_value = "Early log entry before job start"
    entry.time.return_value = datetime.fromtimestamp(1699990000, tz=timezone)
    return entry


class TestLogIgnoredNodeLine:
    """Test the log_ignored_node_line function."""

    def test_log_with_node_in_exception(self, logline_jobinfo, mock_entry):
        """Test logging when exception contains node information."""
        exception = UnknownNodeError("invalid-node-999")
        exception.node = "invalid-node-999"  # Simulate node attribute

        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.SYSLOG)

        # Verify debug was called
        mock_logger.debug.assert_called_once()

        # Check log message content
        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.SYSLOG ignored" in call_args
        assert "node 'invalid-node-999' not in job nodelist" in call_args
        assert "detail=node 'invalid-node-999' not found in job's 2 allocated nodes" in call_args
        assert "Mar 15 10:30:45 invalid-node-999" in call_args

    def test_log_with_fallback_node_extraction(self, logline_jobinfo, mock_entry):
        """Test logging when exception has no node, falls back to parsing."""
        # Create a custom exception without node attribute to test fallback
        exception = UnknownResourceError("generic error")
        # This exception type doesn't set self.node

        with (
            patch("attribution_lib.logline_filtering.logger.logger") as mock_logger,
            patch("attribution_lib.logline_filtering.logger.syslog_extract_node", return_value="parsed-node-123"),
        ):
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.HEALTH_CHECK)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.HEALTH_CHECK ignored" in call_args
        assert "node 'parsed-node-123' not in job nodes" in call_args
        assert "detail=generic resource error for node 'parsed-node-123'" in call_args

    def test_log_with_failed_node_extraction(self, logline_jobinfo, mock_entry):
        """Test logging when both exception and parsing fail to extract node."""
        # Use base exception without node attribute and make parsing return None
        exception = UnknownResourceError("generic error")

        with (
            patch("attribution_lib.logline_filtering.logger.logger") as mock_logger,
            patch("attribution_lib.logline_filtering.logger.syslog_extract_node", return_value=None),
        ):
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.APPLOG)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.APPLOG ignored" in call_args
        assert "node 'unknown' not in job nodes" in call_args
        assert "detail=generic resource error for node 'unknown'" in call_args

    def test_different_log_levels(self, logline_jobinfo, mock_entry):
        """Test that different log levels work correctly."""
        exception = UnknownNodeError("test-node")
        exception.node = "test-node"

        # Test INFO level
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, "Syslog", "INFO")
            mock_logger.info.assert_called_once()

        # Test WARNING level
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, "Syslog", "WARNING")
            mock_logger.warning.assert_called_once()

    def test_no_job_nodes(self, minimal_jobinfo, mock_entry):
        """Test logging when job has minimal nodes."""
        exception = UnknownNodeError("test-node")

        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_node_line(exception, mock_entry, minimal_jobinfo, Source.SYSLOG)

        call_args = mock_logger.debug.call_args[0][0]
        assert "node 'test-node' not found in job's 1 allocated nodes" in call_args

    def test_long_line_truncation(self, logline_jobinfo):
        """Test that long lines are properly truncated."""
        exception = UnknownNodeError("test-node")
        exception.node = "test-node"

        # Create a very long log line
        long_line = "A" * 100  # 100 characters
        mock_entry = Mock()
        mock_entry.body.return_value = long_line

        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.SYSLOG)

        call_args = mock_logger.debug.call_args[0][0]
        # Should be truncated to 80 chars + "..."
        assert "A" * 80 + "..." in call_args
        assert "A" * 81 not in call_args  # Should not contain 81+ chars

    def test_safe_error_handling_for_public_api(self, logline_jobinfo, mock_entry):
        """Test that the public API handles known vs unknown exceptions appropriately."""
        exception = UnknownNodeError("test-node")
        exception.node = "test-node"

        # Test known exceptions - should be logged but not raised
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            mock_logger.debug.side_effect = AttributeError("Known error")
            mock_logger.error.return_value = None

            try:
                log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.SYSLOG)
                # Should succeed without raising
            except Exception:
                pytest.fail("Known exceptions should be logged but not raised")

            mock_logger.error.assert_called_with(
                "Known error in node logline filtering for Source.SYSLOG: AttributeError: Known error"
            )

        # Test unknown exceptions - should be logged and raised
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            mock_logger.debug.side_effect = RuntimeError("Unknown error")
            mock_logger.error.return_value = None

            with pytest.raises(RuntimeError, match="Unknown error"):
                log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.SYSLOG)

            mock_logger.error.assert_called_with(
                "Unknown error in node logline filtering for Source.SYSLOG: RuntimeError: Unknown error"
            )

    def test_safe_fallback_parsing_error_handling(self, logline_jobinfo, mock_entry):
        """Test that the public API handles fallback parsing exceptions appropriately."""
        # Use base exception without node attribute to trigger fallback
        exception = UnknownResourceError("generic error")

        # Test known exception in fallback parsing - should be logged but not raised
        with (
            patch(
                "attribution_lib.logline_filtering.logger.syslog_extract_node",
                side_effect=TypeError("Known parse error"),
            ),
            patch("attribution_lib.logline_filtering.logger.logger") as mock_logger,
        ):
            mock_logger.error.return_value = None

            try:
                log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.SYSLOG)
                # Should succeed without raising
            except Exception:
                pytest.fail("Known exceptions in fallback parsing should be logged but not raised")

        # Test unknown exception in fallback parsing - should be logged and raised
        with (
            patch(
                "attribution_lib.logline_filtering.logger.syslog_extract_node",
                side_effect=RuntimeError("Unknown parse error"),
            ),
            patch("attribution_lib.logline_filtering.logger.logger") as mock_logger,
        ):
            mock_logger.error.return_value = None

            with pytest.raises(RuntimeError, match="Unknown parse error"):
                log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.SYSLOG)

    def test_unidentifiable_node_error_message(self, logline_jobinfo, mock_entry):
        """Test specific message for UnidentifiableNodeError."""
        exception = UnidentifiableNodeError("log line with no node info")

        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.APPLOG)

        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.APPLOG ignored" in call_args
        assert "no node or rank identifier found in log line" in call_args
        assert "detail=unable to extract node information from content" in call_args

    def test_invalid_rank_error_message(self, logline_jobinfo, mock_entry):
        """Test specific message for InvalidRankError."""
        exception = InvalidRankError(200)  # Rank too high for 2-node job

        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.APPLOG)

        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.APPLOG ignored" in call_args
        assert "rank 200 not found in job info" in call_args
        assert "rank 200 cannot be processed (job has 2 nodes, max rank: 15)" in call_args  # 2 * 8 - 1 = 15

    def test_unknown_node_error_message(self, logline_jobinfo, mock_entry):
        """Test specific message for UnknownNodeError uses node from exception."""
        exception = UnknownNodeError("specific-invalid-node")

        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.SYSLOG)

        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.SYSLOG ignored" in call_args
        assert "node 'specific-invalid-node' not in job nodelist" in call_args
        assert "detail=node 'specific-invalid-node' not found in job's 2 allocated nodes" in call_args

    def test_base_unknown_resource_error_fallback(self, logline_jobinfo, mock_entry):
        """Test that base UnknownResourceError uses fallback case with node extraction."""
        # Use base class, not subclass
        exception = UnknownResourceError("generic resource error")
        exception.node = "fallback-node"  # Manually set node attribute

        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.HEALTH_CHECK)

        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.HEALTH_CHECK ignored" in call_args
        assert "node 'fallback-node' not in job nodes" in call_args
        assert "detail=generic resource error for node 'fallback-node'" in call_args

    def test_unknown_exception_type_fallback(self, logline_jobinfo, mock_entry):
        """Test that unexpected exception types are handled by fallback case."""

        # Create a custom exception that inherits from UnknownResourceError
        class CustomResourceError(UnknownResourceError):
            pass

        exception = CustomResourceError("unexpected error type")

        with (
            patch("attribution_lib.logline_filtering.logger.logger") as mock_logger,
            patch("attribution_lib.logline_filtering.logger.syslog_extract_node", return_value="parsed-from-line"),
        ):
            log_ignored_node_line(exception, mock_entry, logline_jobinfo, Source.APPLOG)

        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.APPLOG ignored" in call_args
        assert "node 'parsed-from-line' not in job nodes" in call_args
        assert "detail=generic resource error for node 'parsed-from-line'" in call_args


class TestLogIgnoredTimeLine:
    """Test the log_ignored_time_line function."""

    def test_time_before_job_start(self, logline_jobinfo, mock_entry_with_time):
        """Test logging for entries before job start."""
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_time_line(mock_entry_with_time, logline_jobinfo, "before_job_start", Source.SYSLOG)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.SYSLOG ignored" in call_args
        assert "time_before_job_start" in call_args
        assert "entry_time=" in call_args
        assert "job_start=" in call_args

    def test_time_after_job_end(self, logline_jobinfo, timezone):
        """Test logging for entries after job end."""
        late_entry = Mock()
        late_entry.body.return_value = "Late log entry after job end"
        late_entry.time.return_value = datetime.fromtimestamp(1700020000, tz=timezone)

        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_time_line(late_entry, logline_jobinfo, "after_job_end", Source.SYSLOG)

        call_args = mock_logger.debug.call_args[0][0]
        assert "time_after_job_end" in call_args


class TestLogIgnoredContentLine:
    """Test the log_ignored_content_line function."""

    def test_content_filtering_log(self, mock_entry_content):
        """Test logging for content-based filtering."""
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_content_line(mock_entry_content, "prolog", "skip_prolog_errors", Source.HEALTH_CHECK)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.HEALTH_CHECK ignored" in call_args
        assert "content_filter" in call_args
        assert "reason=skip_prolog_errors" in call_args
        assert "pattern='prolog'" in call_args
        assert "This log contains prolog" in call_args

    def test_content_different_log_level(self, mock_entry_content):
        """Test content filtering with different log level."""
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            log_ignored_content_line(mock_entry_content, "diagnostic", "skip_diagnostic", "Epilog", "INFO")

        mock_logger.info.assert_called_once()


class TestIntegration:
    """Integration tests for realistic usage patterns."""

    def test_typical_syslog_usage_pattern(self, simple_job_factory):
        """Test the typical usage pattern in syslog processing."""
        # This simulates how the function would be used in actual code
        job_info = simple_job_factory(job_id=12345, nodes=["valid-node"])

        # Mock entry from invalid node
        mock_entry = Mock()
        mock_entry.node.side_effect = UnknownNodeError("invalid-node")
        mock_entry.body.return_value = "Test syslog message from invalid node"

        # This is how it would be used in syslog.py
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            try:
                mock_entry.node(job_info)
            except UnknownResourceError as e:
                log_ignored_node_line(e, mock_entry, job_info, Source.SYSLOG)
                # continue  # In real code, this would continue the loop

        # Verify logging occurred
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "Source.SYSLOG ignored" in call_args
        assert "node 'invalid-node' not in job nodelist" in call_args

    def test_multiple_source_names(self):
        """Test that different source names work correctly."""
        exception = UnknownNodeError("test-node")
        exception.node = "test-node"
        mock_entry = Mock()
        mock_entry.body.return_value = "Test message"
        mock_job_info = Mock()
        mock_job_info.nodes = ["other-node"]

        sources = ["Syslog", "Epilog", "Applog", "CustomProcessor"]

        for source in sources:
            with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
                log_ignored_node_line(exception, mock_entry, mock_job_info, source)
                call_args = mock_logger.debug.call_args[0][0]
                assert f"{source} ignored" in call_args
