# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CSP event attribution."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from attribution_lib.attributor import Attributor
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    CspEventEntry,
    Format as ObservationFormat,
    Observation as InboundObservation,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.observation.processing import (
    CspEventEntryObj,
)
from attribution_lib.resources.node import Node as NodeResource
from attribution_lib.resources.resource_set import AllJobResources

_SOURCE_EVENT = {
    "source": "databricks_maintenance_fault",
    "hostname": "test-node-001",
    "fault_id": "GPU_RETIRED_PAGE",
    "event_region": "us-west",
    "event_id": "event-1",
    "event_created_at": "2024-06-15T12:00:00+00:00",
}

# ---------------------------------------------------------------------------
# CspEventEntryObj unit tests
# ---------------------------------------------------------------------------


class TestCspEventEntryObj:
    """Tests for CspEventEntryObj."""

    @pytest.fixture
    def timezone(self):
        return ZoneInfo("UTC")

    @pytest.fixture
    def context(self, timezone):
        now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone)
        return InboundContext(
            time_interval=TimeInterval(now, now),
            resources=AllJobResources,
            source=ObservationSource.CSP_EVENT,
            format=ObservationFormat.CSP_EVENT_LIST,
            timezone=timezone,
        )

    @pytest.fixture
    def sample_entry(self) -> CspEventEntry:
        return CspEventEntry(
            hostname="test-node-001",
            termination_timestamp_s=1718452800.0,  # 2024-06-15T12:00:00 UTC
            category="HOST_FAILURE",
            reason="BAD_GPU",
            status="",
            version=1,
        )

    def test_body_returns_raw_event(self, context, sample_entry):
        obj = CspEventEntryObj(context, pos=1, entry=sample_entry)
        assert obj.body() == "csp_event: category=HOST_FAILURE, reason=BAD_GPU, status="

    def test_attribution_source_preserves_raw_event_without_source_event(self, context, sample_entry):
        source = CspEventEntryObj(context, pos=1, entry=sample_entry).attribution_source_base(None)

        assert source.datum == "csp_event: category=HOST_FAILURE, reason=BAD_GPU, status="

    def test_attribution_source_preserves_optional_source_event(self, context):
        entry = CspEventEntry(
            hostname="test-node-001",
            termination_timestamp_s=1718452800.0,
            category="HOST_FAILURE",
            reason="BAD_HARDWARE",
            status="CLASSIFICATION_READY",
            version=1,
            source_event={**_SOURCE_EVENT, "fault_component": "GPU"},
        )

        source = CspEventEntryObj(context, pos=1, entry=entry).attribution_source_base(None)

        assert source.datum == (
            "csp_event: category=HOST_FAILURE, reason=BAD_HARDWARE, status=CLASSIFICATION_READY, "
            'source_event={"event_created_at":"2024-06-15T12:00:00+00:00","event_id":"event-1",'
            '"event_region":"us-west","fault_component":"GPU","fault_id":"GPU_RETIRED_PAGE",'
            '"hostname":"test-node-001","source":"databricks_maintenance_fault"}'
        )

    def test_body_accepts_source_event_without_optional_component(self, context, sample_entry):
        sample_entry["source_event"] = {**_SOURCE_EVENT}

        assert CspEventEntryObj(context, pos=1, entry=sample_entry).body() == (
            "csp_event: category=HOST_FAILURE, reason=BAD_GPU, status=, "
            'source_event={"event_created_at":"2024-06-15T12:00:00+00:00","event_id":"event-1",'
            '"event_region":"us-west","fault_id":"GPU_RETIRED_PAGE","hostname":"test-node-001",'
            '"source":"databricks_maintenance_fault"}'
        )

    @pytest.mark.parametrize(
        "source_event",
        [
            pytest.param(None, id="explicit-none"),
            pytest.param("", id="blank-string"),
            pytest.param(0, id="zero"),
            pytest.param({}, id="empty-object"),
            pytest.param(
                ["source", "event_id", "event_created_at", "event_region", "hostname", "fault_id"],
                id="non-dict-with-required-fields",
            ),
            {
                "source": "databricks_maintenance_fault",
                "event_id": "event-1",
                "event_created_at": "2024-06-15T12:00:00+00:00",
                "event_region": "us-west",
                "hostname": "test-node-001",
                "fault_id": "GPU_RETIRED_PAGE",
                "raw_payload": "not part of the public contract",
            },
            {
                "source": "databricks_maintenance_fault",
                "event_id": "event-1",
                "event_created_at": "2024-06-15T12:00:00+00:00",
                "event_region": "us-west",
                "hostname": "test-node-001",
            },
            {
                "source": "databricks_maintenance_fault",
                "event_id": 1,
                "event_created_at": "2024-06-15T12:00:00+00:00",
                "event_region": "us-west",
                "hostname": "test-node-001",
                "fault_id": "GPU_RETIRED_PAGE",
            },
            {
                "source": "databricks_maintenance_fault",
                "event_id": " ",
                "event_created_at": "2024-06-15T12:00:00+00:00",
                "event_region": "us-west",
                "hostname": "test-node-001",
                "fault_id": "GPU_RETIRED_PAGE",
            },
            pytest.param(
                {**_SOURCE_EVENT, "fault_component": " "},
                id="blank-fault-component",
            ),
            pytest.param(
                {**_SOURCE_EVENT, "fault_component": ""},
                id="empty-fault-component",
            ),
            pytest.param(
                {**_SOURCE_EVENT, "fault_component": None},
                id="none-fault-component",
            ),
            pytest.param(
                {**_SOURCE_EVENT, "fault_component": 1},
                id="non-string-fault-component",
            ),
        ],
    )
    def test_body_rejects_source_event_outside_public_contract(self, context, source_event):
        entry = CspEventEntry(
            hostname="test-node-001",
            termination_timestamp_s=1718452800.0,
            category="HOST_FAILURE",
            reason="BAD_HARDWARE",
            status="CLASSIFICATION_READY",
            version=1,
            source_event=source_event,
        )

        with pytest.raises(ValueError, match="Invalid CSP source_event"):
            CspEventEntryObj(context, pos=1, entry=entry).body()

    def test_time_returns_datetime(self, context, sample_entry, timezone):
        obj = CspEventEntryObj(context, pos=1, entry=sample_entry)
        expected = datetime.fromtimestamp(1718452800.0, timezone)
        assert obj.time() == expected

    def test_resources_returns_node_resource(self, context, sample_entry):
        obj = CspEventEntryObj(context, pos=1, entry=sample_entry)
        rs = obj.resources()
        node_resources = [r for r in rs if isinstance(r, NodeResource)]
        assert len(node_resources) == 1
        assert node_resources[0].hostname == "test-node-001"

    def test_node_returns_hostname(self, context, sample_entry):
        obj = CspEventEntryObj(context, pos=1, entry=sample_entry)
        assert obj.node() == "test-node-001"

    def test_node_with_job_info(self, context, sample_entry):
        """node() returns the hostname regardless of job_info value."""
        obj = CspEventEntryObj(context, pos=1, entry=sample_entry)
        assert obj.node(job_info=None) == "test-node-001"

    def test_body_returns_raw_for_skip_category(self, context, timezone):
        entry = CspEventEntry(
            hostname="test-node-001",
            termination_timestamp_s=1718452800.0,
            category="PLANNED_MAINTENANCE",
            reason="",
            status="",
            version=1,
        )
        obj = CspEventEntryObj(context, pos=1, entry=entry)
        # body() returns raw event data; skip logic is now in the rules layer
        assert obj.body() == "csp_event: category=PLANNED_MAINTENANCE, reason=, status="


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
            CspEventEntry(
                hostname=hostname,
                termination_timestamp_s=1718452800.0,
                category="HOST_FAILURE",
                reason="BAD_GPU",
                status="",
                version=1,
            ),
        ]

    def _make_csp_context(self, csp_job_info, timezone):
        time_interval = TimeInterval(csp_job_info.job_start_time, csp_job_info.job_end_time)
        return InboundContext(
            time_interval=time_interval,
            resources=AllJobResources,
            source=ObservationSource.CSP_EVENT,
            format=ObservationFormat.CSP_EVENT_LIST,
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
