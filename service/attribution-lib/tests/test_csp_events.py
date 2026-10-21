# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CSP event attribution."""

import pytest

from attribution_lib.attributor import Attributor
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Observation as InboundObservation,
    Source as ObservationSource,
    TimeInterval,
)

# ---------------------------------------------------------------------------
# End-to-end test through the Attributor
# ---------------------------------------------------------------------------


class TestCspEventIngestion:
    """CSP events remain ingestible without the internal attribution rule."""

    @pytest.fixture
    def csp_job_info(self, simple_job_factory):
        return simple_job_factory(
            job_id="csp_e2e_test",
            cluster="test_cluster",
            job_type="batch",
            state="FAILED",
            framework="test",
            ranks_per_node=1,
            start_timestamp=1718452700,
            end_timestamp=1718452900,
            nodes=["test-node-001"],
            source_name="test_slurm",
        )

    def _make_csp_entries(self, hostname="test-node-001"):
        return [
            {
                "ts": 1718452800 * 10**9,
                "resources": [{"kind": "node", "hostname": hostname}],
                "body": "csp_event: category=HOST_FAILURE, reason=BAD_GPU, status=",
            },
        ]

    def _make_csp_context(self, csp_job_info, timezone):
        time_interval = TimeInterval(csp_job_info.job_start_time, csp_job_info.job_end_time)
        return InboundContext(
            time_interval=time_interval,
            source=ObservationSource.CSP_EVENT,
            format=ObservationFormat.ENTRY_TIMESERIES,
            timezone=timezone,
        )

    def test_attributor_ingestion(self, request, event_manager, csp_job_info, timezone):
        """CSP termination events should be ingestible through the Attributor."""
        event_stream = event_manager.create_stream(request.node.name)
        attributor = Attributor.new(event_stream, csp_job_info)

        entries = self._make_csp_entries()
        context = self._make_csp_context(csp_job_info, timezone)
        inbound_obs = InboundObservation(context=context, lines=entries)
        ingest_id = attributor.ingest(inbound_obs)
        assert ingest_id in attributor.ingested_observations

        result = attributor.attribute()
        assert result is not None
        # Verify the observation was consumed and produced a result
        ingested = attributor.ingested_observations[ingest_id]
        assert ingested.consumed is not None
