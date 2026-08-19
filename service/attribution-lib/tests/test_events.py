# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import AwareDatetime, BaseModel

from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import EventManager
from attribution_lib.observation import Context as ObservationContext
from attribution_lib.observation.inbound import (
    Format as ObsFormat,
    Id,
    Source as ObsSource,
    TimeInterval,
)
from attribution_lib.resources import AllJobResources, Node, ResourceSet
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.sources import Source


@pytest.fixture(autouse=True)
def _preload_test_rules(event_manager):
    """Preload rule names specific to this test file."""
    event_manager.preload_rule_names({"thermal_monitor_rule", "attribution_analyzer", "mfu_monitor"})


@pytest.fixture
def sample_source(dummy_ctx):
    return Source(
        source_context=dummy_ctx,
        line_num=42,
        datum="High GPU temperature detected",
        time=datetime.now(),
        resources=AllJobResources,
        rule="thermal_monitor_rule",
    )


class TestEnumPopulation:
    """Test that observation source and format enums are automatically populated."""

    def test_obs_sources_populated_on_init(self):
        """Test that obs_sources table is populated with all Source enum values."""
        mgr = EventManager(RuleLoader().build())

        # Get all sources from the database
        actual_sources = [row[0] for row in mgr._conn.execute("SELECT name FROM obs_sources ORDER BY name").fetchall()]

        # Get expected sources from enum (excluding empty values)
        expected_sources = sorted([s.value for s in ObsSource if s.value])

        assert actual_sources == expected_sources, f"Expected {expected_sources}, got {actual_sources}"
        assert len(actual_sources) > 0, "obs_sources table should not be empty"

        mgr.close()

    def test_obs_formats_populated_on_init(self):
        """Test that obs_formats table is populated with all Format enum values."""
        mgr = EventManager(RuleLoader().build())

        # Get all formats from the database
        actual_formats = [row[0] for row in mgr._conn.execute("SELECT name FROM obs_formats ORDER BY name").fetchall()]

        # Get expected formats from enum (excluding empty values)
        expected_formats = sorted([f.value for f in ObsFormat if f.value])

        assert actual_formats == expected_formats, f"Expected {expected_formats}, got {actual_formats}"
        assert len(actual_formats) > 0, "obs_formats table should not be empty"

        mgr.close()

    def test_enum_values_have_valid_ids(self):
        """Test that populated enum values can be looked up and have valid IDs."""
        mgr = EventManager(RuleLoader().build())

        # Test a few known source values
        for source in ["syslog", "applog", "health_check"]:
            source_id = mgr._conn.execute("SELECT id FROM obs_sources WHERE name=?", [source]).fetchone()
            assert source_id is not None, f"Source '{source}' should have an ID"
            assert source_id[0] > 0, f"Source '{source}' ID should be positive"

        # Test a few known format values
        for format_val in ["raw_loki_streams", "loki_dict_list", "prometheus_metrics"]:
            format_id = mgr._conn.execute("SELECT id FROM obs_formats WHERE name=?", [format_val]).fetchone()
            assert format_id is not None, f"Format '{format_val}' should have an ID"
            assert format_id[0] > 0, f"Format '{format_val}' ID should be positive"

        mgr.close()

    def test_lookup_helpers_work_with_populated_enums(self):
        """Test that the lookup helpers can find the pre-populated enum values."""
        mgr = EventManager(RuleLoader().build())

        # Test obs_sources lookup helper
        syslog_id = mgr._obs_sources.get_id("syslog")
        assert syslog_id is not None, "Should find pre-populated 'syslog'"

        syslog_str = mgr._obs_sources.get_str(syslog_id)
        assert syslog_str == "syslog", "Should retrieve 'syslog' string from ID"

        # Test obs_formats lookup helper
        loki_id = mgr._obs_formats.get_id("loki_dict_list")
        assert loki_id is not None, "Should find pre-populated 'loki_dict_list'"

        loki_str = mgr._obs_formats.get_str(loki_id)
        assert loki_str == "loki_dict_list", "Should retrieve 'loki_dict_list' string from ID"

        mgr.close()


class TestEventRule:
    def test_event_creation_with_source(self, event_stream, sample_source):
        """Test creating events with proper Source objects"""
        stream = event_stream("myjob")
        stream.register_observation_context(sample_source.source_context)
        stream.add_event("Attribution.node.bad", {"node": "gpu-node-01", "symptom": "High temperature"}, sample_source)

        # Verify via query_events_with_source (bulk pattern)

        typ_id = stream.mgr.types.get_id("Attribution.node.bad")
        results = list(query_events_with_source(stream, typ_id))
        assert len(results) == 1
        event_id, event_type, payload, retrieved_source = results[0]
        assert payload["node"] == "gpu-node-01"
        assert payload["symptom"] == "High temperature"

        # Verify source was reconstructed correctly
        assert retrieved_source is not None
        assert retrieved_source.rule == "thermal_monitor_rule"
        assert retrieved_source.line_num == 42
        assert retrieved_source.datum == "High GPU temperature detected"

    def test_flush_serializes_pydantic_payload_with_datetime(self, event_stream, sample_source):
        """Regression: Pydantic model with a raw datetime in payload must not crash _flush."""

        class _PydanticTimeInterval(BaseModel):
            start: AwareDatetime
            end: AwareDatetime

        stream = event_stream("test-event-job")
        stream.register_observation_context(sample_source.source_context)

        dt = datetime(2026, 5, 3, 23, 15, 1, tzinfo=ZoneInfo("America/Los_Angeles"))
        stream.add_event(
            "Application.mfu",
            {"time_interval": _PydanticTimeInterval(start=dt, end=dt)},
            sample_source,
        )

        stream.flush()

        typ_id = stream.mgr.types.get_id("Application.mfu")
        results = list(query_events_with_source(stream, typ_id))
        assert len(results) == 1
        _, _, payload, _ = results[0]
        assert isinstance(payload["time_interval"]["start"], str)
        assert datetime.fromisoformat(payload["time_interval"]["start"]) == dt
        assert datetime.fromisoformat(payload["time_interval"]["end"]) == dt

    def test_mfu_rule(self, event_stream):
        """Test event creation and SQL queries"""
        stream = event_stream("myjob")
        nodes = [Node(hostname="gpu-node-01")]
        resources = ResourceSet(nodes)

        # Create a valid observation context
        context = ObservationContext(
            time_interval=TimeInterval(datetime.now() - timedelta(minutes=15), datetime.now()),
            resources=resources,
            source=ObsSource.APPLOG,
            format=ObsFormat.RAW_LOKI_STREAMS,
            timezone=ZoneInfo("UTC"),
            id=Id(54321),
        )
        stream.register_observation_context(context)
        bad_source = Source(
            source_context=context,
            line_num=100,
            datum="Bad attribution detected",
            time=datetime.now() - timedelta(minutes=10),
            resources=resources,
            rule="attribution_analyzer",
        )

        mfu_source = Source(
            source_context=context,
            line_num=200,
            datum="MFU measurement",
            time=datetime.now() - timedelta(minutes=5),
            resources=resources,
            rule="mfu_monitor",
        )

        # Add events
        stream.add_event("Attribution.node.bad", {"node": "gpu-node-01"}, bad_source)
        stream.add_event("Application.mfu", {"tflops": 2.1}, mfu_source)

        # Verify events via SQL
        rows = stream.query_raw(
            "SELECT id, event_type FROM events WHERE context_id=? AND status=1 ORDER BY id",
            [stream.ctx_id],
        )
        assert len(rows) == 2

        # Verify bad event
        bad_typ = stream.mgr.types.get_id("Attribution.node.bad")
        mfu_typ = stream.mgr.types.get_id("Application.mfu")

        bad_rows = stream.query_raw(
            "SELECT id FROM events WHERE context_id=? AND event_type=? AND status=1",
            [stream.ctx_id, bad_typ],
        )
        assert len(bad_rows) == 1

        mfu_rows = stream.query_raw(
            "SELECT id FROM events WHERE context_id=? AND event_type=? AND status=1",
            [stream.ctx_id, mfu_typ],
        )
        assert len(mfu_rows) == 1


class TestTimestampAndTimezoneFlow:
    """Test that timestamps and timezones are preserved through the entire flow."""

    def test_timezone_preserved_through_roundtrip(self):
        """Test that a timezone-aware timestamp is preserved when stored and retrieved."""
        mgr = EventManager(RuleLoader().build())
        mgr.preload_rule_names({"test"})
        mgr.preload_event_types({"Test.event"})
        stream = mgr.create_stream("test-context")

        # Create timestamp in EST (UTC-5)
        est = ZoneInfo("America/New_York")
        original_time = datetime(2025, 1, 17, 15, 30, 45, tzinfo=est)

        # Create source with EST timestamp
        nodes = [Node(hostname="test-node")]
        resources = ResourceSet(nodes)
        context = ObservationContext(
            time_interval=TimeInterval(datetime(2025, 1, 17, 10, 0, 0), datetime(2025, 1, 17, 18, 0, 0)),
            resources=resources,
            source=ObsSource.SYSLOG,
            format=ObsFormat.RAW_LOKI_STREAMS,
            timezone=est,
            id=Id(1),
        )
        source = Source(
            source_context=context,
            line_num=100,
            datum="test",
            time=original_time,
            resources=resources,
            rule="test",
        )
        stream.register_observation_context(context)
        # Add event
        stream.add_event("Test.event", {"test": "data"}, source)

        # Retrieve source using bulk helper
        typ_id = stream.mgr.types.get_id("Test.event")
        results = list(query_events_with_source(stream, typ_id))
        assert len(results) == 1
        event_id, event_type, payload, retrieved_source = results[0]

        assert retrieved_source is not None
        assert retrieved_source.time is not None

        # Check that time and timezone are preserved
        assert retrieved_source.time.year == original_time.year
        assert retrieved_source.time.month == original_time.month
        assert retrieved_source.time.day == original_time.day
        assert retrieved_source.time.hour == original_time.hour
        assert retrieved_source.time.minute == original_time.minute
        assert retrieved_source.time.second == original_time.second

        # Check that the absolute time (UTC equivalent) is the same
        utc = ZoneInfo("UTC")
        retrieved_utc = retrieved_source.time.astimezone(utc)
        original_utc = original_time.astimezone(utc)

        print(f"Original:   {original_time} (tzinfo={original_time.tzinfo})")
        print(f"Retrieved:  {retrieved_source.time} (tzinfo={retrieved_source.time.tzinfo})")
        print(f"Original UTC:   {original_utc}")
        print(f"Retrieved UTC:  {retrieved_utc}")

        assert retrieved_utc == original_utc, f"UTC times don't match: {retrieved_utc} != {original_utc}"

    def test_different_timezones_stored_correctly(self):
        """Test that multiple events with different timezones are stored correctly."""
        mgr = EventManager(RuleLoader().build())
        mgr.preload_rule_names({"test"})
        mgr.preload_event_types({"Test.event"})
        stream = mgr.create_stream("test-context")

        # Same absolute time in different timezones
        est = ZoneInfo("America/New_York")  # UTC-5
        pst = ZoneInfo("America/Los_Angeles")  # UTC-8
        utc = ZoneInfo("UTC")

        # 3pm EST = 8pm UTC = 12pm PST
        time_est = datetime(2025, 1, 17, 15, 0, 0, tzinfo=est)
        time_utc = datetime(2025, 1, 17, 20, 0, 0, tzinfo=utc)
        time_pst = datetime(2025, 1, 17, 12, 0, 0, tzinfo=pst)

        # Verify they're the same absolute time
        assert time_est.astimezone(utc) == time_utc
        assert time_pst.astimezone(utc) == time_utc

        # Store events with each timezone
        nodes = [Node(hostname="test-node")]
        resources = ResourceSet(nodes)

        event_ids = []
        obs_id = 1
        for tz, time_val in [(est, time_est), (utc, time_utc), (pst, time_pst)]:
            context = ObservationContext(
                time_interval=TimeInterval(datetime(2025, 1, 17, 10, 0, 0), datetime(2025, 1, 17, 18, 0, 0)),
                resources=resources,
                source=ObsSource.SYSLOG,
                format=ObsFormat.RAW_LOKI_STREAMS,
                timezone=tz,
                id=obs_id,
            )
            obs_id += 1
            source = Source(
                source_context=context,
                line_num=100,
                datum="test",
                time=time_val,
                resources=resources,
                rule="test",
            )
            stream.register_observation_context(context)
            stream.add_event("Test.event", {"tz": str(tz)}, source)
            event_ids.append((tz, time_val))

        # Retrieve all events with sources using bulk helper
        typ_id = stream.mgr.types.get_id("Test.event")
        results = list(query_events_with_source(stream, typ_id, order_by="e.id ASC"))
        assert len(results) == 3

        # Verify each preserves its timezone
        for i, (expected_tz, expected_time) in enumerate(event_ids):
            event_id, event_type, payload, retrieved_source = results[i]
            assert retrieved_source is not None
            assert retrieved_source.time is not None

            # Check the local time is preserved
            assert retrieved_source.time.hour == expected_time.hour
            # Note: tzinfo objects may differ (ZoneInfo vs pytz) but represent same timezone
            assert str(retrieved_source.time.tzinfo) == str(expected_tz)

            # Check the absolute time (UTC) is the same for all
            assert retrieved_source.time.astimezone(utc) == time_utc

    def test_query_raw_returns_correct_timezone(self):
        """Test that query_raw returns timestamps with correct timezone information."""
        mgr = EventManager(RuleLoader().build())
        mgr.preload_rule_names({"test"})
        mgr.preload_event_types({"Test.event"})
        stream = mgr.create_stream("test-context")

        # Create timestamp in EST
        est = ZoneInfo("America/New_York")
        original_time = datetime(2025, 1, 17, 15, 30, 45, tzinfo=est)

        nodes = [Node(hostname="test-node")]
        resources = ResourceSet(nodes)
        context = ObservationContext(
            time_interval=TimeInterval(datetime(2025, 1, 17, 10, 0, 0), datetime(2025, 1, 17, 18, 0, 0)),
            resources=resources,
            source=ObsSource.SYSLOG,
            format=ObsFormat.RAW_LOKI_STREAMS,
            timezone=est,
            id=Id(1),
        )
        stream.register_observation_context(context)
        source = Source(
            source_context=context,
            line_num=100,
            datum="test",
            time=original_time,
            resources=resources,
            rule="test",
        )

        stream.add_event("Test.event", {"test": "data"}, source)

        # Query raw timestamp from database
        rows = stream.query_raw("SELECT id, timestamp FROM events WHERE context_id=?", [stream.ctx_id])

        assert len(rows) == 1
        event_id, db_timestamp = rows[0]

        # Check what type and value we get back
        print(f"DEBUG: db_timestamp type={type(db_timestamp)}, value={db_timestamp}")

        # Convert to datetime if needed
        if isinstance(db_timestamp, str):
            db_time = datetime.fromisoformat(db_timestamp)
        else:
            db_time = db_timestamp

        print(f"DEBUG: db_time={db_time}, db_time.tzinfo={db_time.tzinfo}")
        print(f"DEBUG: original_time={original_time}, original_time.tzinfo={original_time.tzinfo}")

        # The absolute time (UTC) should be the same
        utc = ZoneInfo("UTC")
        assert db_time.astimezone(utc) == original_time.astimezone(utc)
