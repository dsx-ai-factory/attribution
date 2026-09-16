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
from attribution_lib.events import EventManager, _stable_id
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


class TestContentAddressedLookupIds:
    """event_types/rules/obs_sources/obs_formats ids must depend only on the
    name's own bytes, not on insertion order or what other names are present
    in the same build -- otherwise two independently-built EventManagers
    (e.g. across process restarts, where Python's hash-seed randomization
    makes set iteration order differ) can silently disagree on a name's id.
    This matters for from_snapshot(): a snapshot's ids must remain valid
    against a freshly built EventManager, not just the one that exported it.
    """

    def test_same_name_gets_same_id_regardless_of_insertion_order(self):
        class EngineA:
            all_rule_names = {"rule_x", "rule_y", "rule_z"}
            all_event_types: set[str] = set()

        class EngineB:
            all_rule_names = {"rule_z", "rule_y", "rule_x", "rule_extra"}
            all_event_types: set[str] = set()

        mgr_a = EventManager(EngineA())
        mgr_b = EventManager(EngineB())
        try:
            for name in ("rule_x", "rule_y", "rule_z"):
                assert mgr_a.rules.get_id(name) == mgr_b.rules.get_id(name), (
                    f"{name} got different ids in two independently-built EventManagers"
                )
        finally:
            mgr_a.close()
            mgr_b.close()

    def test_id_is_stable_across_separate_processes(self):
        """Same guarantee as above, but forcing an actual separate Python
        process (a fresh, randomized PYTHONHASHSEED) rather than relying on
        two in-process builds to exercise the same non-determinism."""
        import os
        import subprocess
        import sys

        script = (
            "from attribution_lib.events import EventManager\n"
            "class E:\n"
            "    all_rule_names = {'rule_x', 'rule_y', 'rule_z'}\n"
            "    all_event_types = set()\n"
            "mgr = EventManager(E())\n"
            "print(mgr.rules.get_id('rule_x'), mgr.rules.get_id('rule_y'), mgr.rules.get_id('rule_z'))\n"
            "mgr.close()\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        outputs = set()
        for _ in range(3):
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
            )
            outputs.add(result.stdout.strip())
        assert len(outputs) == 1, f"id assignment differed across process runs: {outputs}"

        expected = " ".join(str(_stable_id(n)) for n in ("rule_x", "rule_y", "rule_z"))
        assert outputs == {expected}, f"subprocess ids {outputs} disagree with in-process ids {expected!r}"


class TestStableIdCollisionDetection:
    """_stable_id() is a 31-bit hash of the name, so two distinct names can in
    principle collide. preload() must detect this rather than silently
    letting the second name's cache entry point at the first name's row.
    """

    def test_colliding_name_raises_instead_of_silently_aliasing(self, monkeypatch):
        import attribution_lib.events as events_module

        class Engine:
            all_rule_names: set[str] = set()
            all_event_types: set[str] = set()

        mgr = EventManager(Engine())
        try:
            mgr.preload_rule_names({"rule_a"})
            colliding_id = mgr.rules.get_id("rule_a")

            monkeypatch.setattr(events_module, "_stable_id", lambda name: colliding_id)

            with pytest.raises(events_module.StableIdCollisionError):
                mgr.preload_rule_names({"rule_b"})

            # The cache must not have been polluted by the failed preload.
            with pytest.raises(events_module.UndeclaredRuleError):
                mgr.rules.get_id("rule_b")
        finally:
            mgr.close()

    def test_collision_within_same_batch_raises(self, monkeypatch):
        """Same guarantee, but both colliding names are new and preloaded in
        one preload_rule_names() call -- INSERT OR IGNORE only persists one
        of the two rows, so the other must be detected via the readback too.
        """
        import attribution_lib.events as events_module

        class Engine:
            all_rule_names: set[str] = set()
            all_event_types: set[str] = set()

        mgr = EventManager(Engine())
        try:
            monkeypatch.setattr(events_module, "_stable_id", lambda name: 999)

            with pytest.raises(events_module.StableIdCollisionError):
                mgr.preload_rule_names({"rule_c", "rule_d"})

            with pytest.raises(events_module.UndeclaredRuleError):
                mgr.rules.get_id("rule_c")
            with pytest.raises(events_module.UndeclaredRuleError):
                mgr.rules.get_id("rule_d")
        finally:
            mgr.close()

    def test_collision_among_rules_engine_names_raises_at_construction(self, monkeypatch):
        """The same collision hazard exists when `EventManager.__init__` bulk-seeds
        the shared lookup db from `rules_engine.all_rule_names`, not just when
        `preload_rule_names()` later adds to the local overlay. Without a check
        here, two colliding names would instead surface as an opaque DuckDB
        primary-key constraint violation.
        """
        import attribution_lib.events as events_module

        class Engine:
            all_rule_names: set[str] = {"rule_e", "rule_f"}
            all_event_types: set[str] = set()

        # Only rule_e/rule_f collide. A blanket stub would also collide the
        # obs_sources/obs_formats enum values bulk-seeded before "rules" (see
        # _build_shared_lookup_db), so the raised error would report table
        # "obs_sources" instead of exercising the rules collision this test
        # is actually about.
        real_stable_id = events_module._stable_id
        monkeypatch.setattr(
            events_module, "_stable_id", lambda name: 999 if name in ("rule_e", "rule_f") else real_stable_id(name)
        )

        with pytest.raises(events_module.StableIdCollisionError) as exc_info:
            EventManager(Engine())
        assert exc_info.value.table == "rules"

    def test_preload_detects_collision_against_existing_shared_tier_name(self, monkeypatch):
        """`preload()`'s collision check previously only compared the
        requested name's own persisted row, via a dict comprehension that
        silently dropped one side whenever two different names (one already
        in the shared tier, one newly preloaded) shared the same id. Which
        row won the dict overwrite depended on SQL row return order, so the
        same collision could raise or silently alias depending on it.
        """
        import attribution_lib.events as events_module

        class Engine:
            all_rule_names: set[str] = {"shared_rule"}
            all_event_types: set[str] = set()

        mgr = EventManager(Engine())
        try:
            shared_id = mgr.rules.get_id("shared_rule")

            real_stable_id = events_module._stable_id
            monkeypatch.setattr(
                events_module,
                "_stable_id",
                lambda name: shared_id if name == "new_local_rule" else real_stable_id(name),
            )

            with pytest.raises(events_module.StableIdCollisionError) as exc_info:
                mgr.preload_rule_names({"new_local_rule"})
            assert exc_info.value.table == "rules"

            # Must not have aliased new_local_rule to shared_rule's id.
            with pytest.raises(events_module.UndeclaredRuleError):
                mgr.rules.get_id("new_local_rule")
            assert mgr.rules.get_id("shared_rule") == shared_id
        finally:
            mgr.close()

    def test_message_when_persisted_name_is_missing(self):
        """persisted_name is None when the id wasn't found under any name at
        all after insertion -- an unexpected state distinct from a genuine
        collision, so it gets its own diagnostic message.
        """
        from attribution_lib.exceptions import StableIdCollisionError

        error = StableIdCollisionError("rules", 999, "rule_e", None)

        assert "not found under any name" in str(error)
        assert "rule_e" in str(error)


class TestConnectionTuning:
    """`EventManager` creates its own private DuckDB connection per instance. DuckDB
    defaults threads to the host's full CPU count and memory_limit to ~80% of system
    RAM. Callers that create many short-lived `EventManagers` (one per unit of work)
    need a way to cap that per-connection overhead."""

    def test_defaults_are_unchanged_when_not_specified(self):
        """No threads/memory_limit args -> today's DuckDB host-sized defaults, unchanged."""
        import os

        mgr = EventManager(RuleLoader().build())
        (threads,) = mgr._conn.execute("SELECT current_setting('threads')").fetchone()
        assert threads == os.cpu_count()
        mgr.close()

    def test_threads_can_be_capped(self):
        mgr = EventManager(RuleLoader().build(), threads=1)
        (threads,) = mgr._conn.execute("SELECT current_setting('threads')").fetchone()
        assert threads == 1
        mgr.close()

    def test_memory_limit_can_be_capped(self):
        mgr = EventManager(RuleLoader().build(), memory_limit="128MB")
        (memory_limit,) = mgr._conn.execute("SELECT current_setting('memory_limit')").fetchone()
        # DuckDB reports this as a normalized "<value> <unit>" string (e.g.
        # "122.0 MiB" for a 128MB request). "GiB" not in memory_limit alone
        # would pass for any non-GiB-scale value, including a wrong one, so
        # parse it and check it's actually ~128MB.
        value, unit = memory_limit.split()
        assert unit == "MiB", f"expected a MiB-scale memory_limit, got {memory_limit!r}"
        assert float(value) == pytest.approx(128_000_000 / 1024**2, rel=0.01)
        mgr.close()

    def test_disk_spill_enabled_by_default(self):
        """Unchanged from today's DuckDB default: deployments with disk to
        spare (e.g. bare metal) get graceful degradation instead of a hard
        failure when memory_limit is exceeded."""
        mgr = EventManager(RuleLoader().build())
        (temp_dir,) = mgr._conn.execute("SELECT current_setting('temp_directory')").fetchone()
        assert temp_dir != ""
        mgr.close()

    def test_disk_spill_can_be_disabled(self):
        """Deployments with no writable disk (e.g. read-only containers) need
        to opt out of DuckDB's default spill-to-disk-on-overflow behavior."""
        mgr = EventManager(RuleLoader().build(), allow_disk_spill=False)
        (temp_dir,) = mgr._conn.execute("SELECT current_setting('temp_directory')").fetchone()
        assert temp_dir == ""
        mgr.close()


class TestOutOfMemoryHandling:
    """With disk spill disabled, exceeding memory_limit must surface as a
    catchable, stable library exception. It should not force every caller
    to know about (and catch) duckdb's own exception hierarchy."""

    def test_overflow_raises_library_exception_not_raw_duckdb_exception(self):
        from attribution_lib.exceptions import EventStorageOutOfMemoryError

        mgr = EventManager(RuleLoader().build(), memory_limit="16MB", allow_disk_spill=False)
        stream = mgr.create_stream("job_1")
        try:
            with pytest.raises(EventStorageOutOfMemoryError):
                stream.execute("CREATE TABLE big AS SELECT repeat('x', 1000000) FROM range(50)")
        finally:
            mgr.close()

    def test_overflow_with_disk_spill_enabled_does_not_raise(self):
        """Same oversized write, but with disk spill allowed (the default).
        It should succeed by spilling to disk instead of erroring."""
        mgr = EventManager(RuleLoader().build(), memory_limit="16MB")
        stream = mgr.create_stream("job_1")
        try:
            stream.execute("CREATE TABLE big AS SELECT repeat('x', 1000000) FROM range(50)")
        finally:
            mgr.close()


class TestCloseRetryability:
    """`close()` is documented as idempotent (`if self._closed: return`), but
    a `stream.flush()` failure -- e.g. `EventStorageOutOfMemoryError`, which
    this exact memory-limit-enforcement branch can trigger -- must not mark
    the manager closed anyway. Otherwise the idempotency guard makes a retry
    silently no-op instead of finishing the interrupted cleanup (remaining
    streams unflushed, `self._conn` never closed)."""

    def test_flush_failure_leaves_manager_retryable(self, monkeypatch):
        mgr = EventManager(RuleLoader().build())
        stream = mgr.create_stream("job_1")

        calls = {"n": 0}
        real_flush = stream.flush

        def flaky_flush():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated flush failure")
            real_flush()

        monkeypatch.setattr(stream, "flush", flaky_flush)

        with pytest.raises(RuntimeError, match="simulated flush failure"):
            mgr.close()
        assert not mgr._closed, "close() must not mark the manager closed when flush() fails"

        # Retry: flush succeeds this time, and cleanup should actually finish.
        mgr.close()
        assert mgr._closed
        with pytest.raises(Exception):
            mgr._conn.execute("SELECT 1")


class TestCheckpointRestoreAtomicity:
    """`checkpoint()`/`restore()` each run a paired DROP/CREATE or
    DELETE/INSERT. DuckDB auto-commits each statement independently unless
    wrapped in an explicit transaction, so if the second statement fails,
    the first one's effect (dropping the prior checkpoint, or deleting all
    events) would otherwise already be permanently committed.
    """

    @staticmethod
    def _insert_event(stream, line_num):
        stream.mgr.preload_event_types({"Test.checkpoint_event"})
        type_id = stream.mgr.types.get_id("Test.checkpoint_event")
        stream.execute(
            "INSERT INTO events (context_id, event_type, status, line_num, payload) VALUES (?, ?, 1, ?, '{}')",
            [stream.ctx_id, type_id, line_num],
        )

    def test_checkpoint_failure_rolls_back_without_destroying_prior_checkpoint(self, monkeypatch):
        import duckdb as duckdb_module

        mgr = EventManager(RuleLoader().build())
        try:
            stream = mgr.create_stream("job_1")
            self._insert_event(stream, 1)
            mgr.checkpoint("v1")
            (count_before,) = mgr._conn.execute('SELECT COUNT(*) FROM "_ckpt_v1"').fetchone()
            assert count_before == 1

            self._insert_event(stream, 2)

            real_execute = duckdb_module.DuckDBPyConnection.execute

            def flaky_execute(self, sql, *args, **kwargs):
                if 'CREATE TABLE "_ckpt_v1"' in sql:
                    raise RuntimeError("synthetic failure")
                return real_execute(self, sql, *args, **kwargs)

            monkeypatch.setattr(duckdb_module.DuckDBPyConnection, "execute", flaky_execute)
            with pytest.raises(RuntimeError):
                mgr.checkpoint("v1")

            # The prior checkpoint must survive: the DROP TABLE that ran
            # before the failing CREATE TABLE must have been rolled back too.
            (count_after,) = mgr._conn.execute('SELECT COUNT(*) FROM "_ckpt_v1"').fetchone()
            assert count_after == 1
        finally:
            mgr.close()

    def test_restore_failure_rolls_back_without_emptying_events_table(self, monkeypatch):
        import duckdb as duckdb_module

        mgr = EventManager(RuleLoader().build())
        try:
            stream = mgr.create_stream("job_1")
            self._insert_event(stream, 1)
            mgr.checkpoint("v1")
            self._insert_event(stream, 2)

            (count_before,) = mgr._conn.execute("SELECT COUNT(*) FROM events").fetchone()
            assert count_before == 2

            real_execute = duckdb_module.DuckDBPyConnection.execute

            def flaky_execute(self, sql, *args, **kwargs):
                if sql.strip().startswith("INSERT INTO events SELECT"):
                    raise RuntimeError("synthetic failure")
                return real_execute(self, sql, *args, **kwargs)

            monkeypatch.setattr(duckdb_module.DuckDBPyConnection, "execute", flaky_execute)
            with pytest.raises(RuntimeError):
                mgr.restore("v1")

            # The DELETE FROM events that ran before the failing INSERT must
            # have been rolled back too, not left with an emptied table.
            (count_after,) = mgr._conn.execute("SELECT COUNT(*) FROM events").fetchone()
            assert count_after == 2
        finally:
            mgr.close()

    def test_commit_failure_also_rolls_back(self, monkeypatch):
        """`conn.commit()` itself can fail. `_atomic()` must roll back in
        that case too, not just when a statement inside the block fails.
        """
        import duckdb as duckdb_module

        mgr = EventManager(RuleLoader().build())
        try:
            stream = mgr.create_stream("job_1")
            self._insert_event(stream, 1)
            mgr.checkpoint("v1")
            self._insert_event(stream, 2)

            (count_before,) = mgr._conn.execute("SELECT COUNT(*) FROM events").fetchone()
            assert count_before == 2

            def flaky_commit(self):
                raise RuntimeError("synthetic commit failure")

            monkeypatch.setattr(duckdb_module.DuckDBPyConnection, "commit", flaky_commit)
            with pytest.raises(RuntimeError):
                mgr.restore("v1")

            (count_after,) = mgr._conn.execute("SELECT COUNT(*) FROM events").fetchone()
            assert count_after == 2
        finally:
            mgr.close()

    def test_commit_time_out_of_memory_is_translated(self, monkeypatch):
        """A `duckdb.OutOfMemoryException` raised by `conn.commit()` itself
        (not just by a statement inside the transaction) must still surface
        as `EventStorageOutOfMemoryError`, matching every other DuckDB
        write path in this class.
        """
        import duckdb as duckdb_module

        from attribution_lib.exceptions import EventStorageOutOfMemoryError

        mgr = EventManager(RuleLoader().build())
        try:
            stream = mgr.create_stream("job_1")
            self._insert_event(stream, 1)

            def flaky_commit(self):
                raise duckdb_module.OutOfMemoryException("synthetic commit OOM")

            monkeypatch.setattr(duckdb_module.DuckDBPyConnection, "commit", flaky_commit)
            with pytest.raises(EventStorageOutOfMemoryError):
                mgr.checkpoint("v1")
        finally:
            mgr.close()


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
