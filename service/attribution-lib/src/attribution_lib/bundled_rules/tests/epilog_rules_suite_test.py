# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from unittest.mock import patch

import attribution_lib
from attribution_lib.core import process_check
from attribution_lib.observation.inbound import (
    Format as ObservationFormat,
    Source as ObservationSource,
    TimeInterval,
)


class TestCheckResults:
    def test_node_filtering_logging(self, simple_job_factory, obs_data, event_manager):
        """Test that epilog lines from nodes not in job are logged and ignored."""
        # Create job info with limited node list
        job_info = simple_job_factory(nodes=["job-node-001", "job-node-002"])

        # Create test epilog data with mixed nodes; the node is parsed from the body
        test_epilog_data = [
            {
                "ts": 1700005000000000000,
                "body": "Nov 15 10:30:00 job-node-001 epilog_12345: Job completed successfully",
            },
            {"ts": 1700005100000000000, "body": "Nov 15 10:31:00 other-node-999 epilog_12345: Some epilog error"},
        ]

        obs = obs_data(job_info, test_epilog_data, ObservationSource.HEALTH_CHECK)

        event_stream = event_manager.create_stream("test_node_filtering_logging")

        # Mock the logger to capture debug calls
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            # Run the epilog check
            process_check(
                event_stream=event_stream,
                job_info=job_info,
                observation_source=obs,
                rules_engine=attribution_lib.get_rules_engine(),
            )

            # Verify that debug logging was called for the ignored node
            mock_logger.debug.assert_called()

            # Check the log calls contain the expected node filtering message
            debug_calls = mock_logger.debug.call_args_list
            debug_logged = any(
                len(call.args) > 0
                and "Source.HEALTH_CHECK ignored:" in str(call.args[0])
                and "other-node-999" in str(call.args[0])
                for call in debug_calls
            )
            assert debug_logged, "Expected debug log for node filtering"

    def test_time_filtering_logging(self, simple_job_factory, event_manager, obs_data):
        """Test that epilog time filtering is logged correctly."""
        # Create job info
        job_info = simple_job_factory(
            nodes=["test-node"],
        )

        # Entries outside the declared interval, which defaults to the job's
        test_epilog_data = [
            {"ts": 1699995000000000000, "body": "Nov 14 22:00:00 test-node epilog_12345: Before job start"},
            {"ts": 1700015000000000000, "body": "Nov 15 14:00:00 test-node epilog_12345: After job end"},
        ]
        observation_source = obs_data(job_info, test_epilog_data, ObservationSource.HEALTH_CHECK)

        event_stream = event_manager.create_stream("time_filtering_logging")

        # Mock the logger to capture debug calls
        with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
            process_check(
                event_stream=event_stream,
                job_info=job_info,
                observation_source=observation_source,
                rules_engine=attribution_lib.get_rules_engine(),
            )

            debug_calls = mock_logger.debug.call_args_list

            # Check for before job start filtering
            before_logged = any(len(call.args) > 0 and "time_before_start" in str(call.args[0]) for call in debug_calls)

            # Check for after job end filtering
            after_logged = any(len(call.args) > 0 and "time_after_end" in str(call.args[0]) for call in debug_calls)

            assert before_logged, "Expected before job start filtering log"
            assert after_logged, "Expected after job end filtering log"

    def test_content_filtering_skips_excluded_entries(self, simple_job_factory, obs_data, event_manager):
        """Test that pre-filter excludes prolog and unrelated epilog entries."""
        job_info = simple_job_factory(
            nodes=["test-node"],
        )

        # Both entries should be excluded by the HEALTH_CHECK pre-filter:
        # - prolog: keyword exclude
        # - epilog_99999: callable (wrong job id)
        test_epilog_data = [
            {
                "ts": 1700005000000000000,
                "resources": [{"kind": "node", "hostname": "test-node"}],
                "body": "Nov 15 10:30:00 test-node This is a prolog issue",
            },
            {
                "ts": 1700005100000000000,
                "resources": [{"kind": "node", "hostname": "test-node"}],
                "body": "Nov 15 10:31:00 test-node This is an epilog_99999 unrelated issue",
            },
        ]
        observation_source = obs_data(job_info, test_epilog_data, ObservationSource.HEALTH_CHECK)
        event_stream = event_manager.create_stream("content_filtering")

        process_check(
            event_stream=event_stream,
            observation_source=observation_source,
            job_info=job_info,
            rules_engine=attribution_lib.get_rules_engine(),
        )

        # All entries should have been excluded - no extractor events produced
        rows = event_stream.query_raw(
            "SELECT COUNT(*) FROM events WHERE context_id=?",
            [event_stream.ctx_id],
        )
        assert rows[0][0] == 0, "Expected no events from excluded epilog entries"
