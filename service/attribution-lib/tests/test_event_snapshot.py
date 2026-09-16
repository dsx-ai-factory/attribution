# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime
from pathlib import Path

import pytest

import attribution_lib
from attribution_lib.events import EventManager
from attribution_lib.exceptions import InvalidSnapshotError
from attribution_lib.observation.inbound import (
    Format as ObsFormat,
    Source as ObsSource,
)
from attribution_lib.resources import AllJobResources
from attribution_lib.sources import Source


def _add_events(mgr: EventManager, stream_id: str, dummy_ctx, count: int = 3) -> list[str]:
    """Add test events to a stream, return the event type names."""
    mgr.preload_event_types({f"Test.event_{i}" for i in range(count)})
    mgr.preload_rule_names({f"rule_{i}" for i in range(count)})
    stream = mgr.create_stream(stream_id)
    stream.register_observation_context(dummy_ctx)

    event_types = []
    for i in range(count):
        name = f"Test.event_{i}"
        event_types.append(name)
        stream.add_event(
            name,
            {"index": i},
            Source(
                source_context=dummy_ctx,
                line_num=i,
                datum=f"datum_{i}",
                time=datetime.now(),
                resources=AllJobResources,
                rule=f"rule_{i}",
            ),
        )
    stream.flush()
    return event_types


class TestSnapshotRoundTrip:
    def test_export_import_preserves_events(self, tmp_path, dummy_ctx):
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx, count=5)
        snapshot_dir = tmp_path / "snap"

        mgr.export_snapshot(snapshot_dir)
        mgr.close()

        restored = EventManager.from_snapshot(snapshot_dir, attribution_lib.get_rules_engine())
        rows = restored._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert rows[0] == 5
        restored.close()

    def test_export_import_preserves_event_content(self, tmp_path, dummy_ctx):
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        mgr.preload_event_types({"Test.specific"})
        mgr.preload_rule_names({"specific_rule"})
        stream = mgr.create_stream("job_1")
        stream.register_observation_context(dummy_ctx)
        stream.add_event(
            "Test.specific",
            {"key": "value"},
            Source(
                source_context=dummy_ctx,
                line_num=99,
                datum="specific_datum",
                time=datetime.now(),
                resources=AllJobResources,
                rule="specific_rule",
            ),
        )
        stream.flush()
        mgr.export_snapshot(tmp_path / "snap")
        mgr.close()

        restored = EventManager.from_snapshot(tmp_path / "snap", attribution_lib.get_rules_engine())
        stream2 = restored.create_stream("job_1")

        rows = stream2.query_raw(
            "SELECT source_datum FROM events WHERE context_id=?",
            [stream2.ctx_id],
        )
        assert len(rows) == 1
        assert rows[0][0] == "specific_datum"
        restored.close()

    def test_restoring_a_name_already_in_shared_tier_does_not_duplicate(self, tmp_path, dummy_ctx):
        """A rule name that's both in the exported snapshot and in the
        current rules_engine's shared tier must resolve to exactly one id
        after restore, not a duplicate row split across the shared tier
        and a restored local overlay. `from_snapshot()` builds its shared
        lookup db from the union of rules_engine's names and the snapshot's
        own names (see `_SnapshotRuleNames`). Since that union is a plain
        Python set, a name present in both collapses to one row by
        construction, not via a runtime dedup check.
        """
        rules_engine = attribution_lib.get_rules_engine()
        shared_rule_name = next(iter(rules_engine.all_rule_names))

        mgr = EventManager(rules_engine=rules_engine)
        mgr.preload_event_types({"Test.shared_rule_event"})
        stream = mgr.create_stream("job_1")
        stream.register_observation_context(dummy_ctx)
        stream.add_event(
            "Test.shared_rule_event",
            {"k": "v"},
            Source(
                source_context=dummy_ctx,
                line_num=1,
                datum="d",
                time=datetime.now(),
                resources=AllJobResources,
                rule=shared_rule_name,
            ),
        )
        stream.flush()
        mgr.export_snapshot(tmp_path / "snap")
        original_id = mgr.rules.get_id(shared_rule_name)
        mgr.close()

        restored = EventManager.from_snapshot(tmp_path / "snap", rules_engine)
        try:
            (count,) = restored._conn.execute(
                "SELECT COUNT(*) FROM rules WHERE name = ?", [shared_rule_name]
            ).fetchone()
            assert count == 1, f"expected exactly one row for {shared_rule_name!r}, found {count}"
            assert restored.rules.get_id(shared_rule_name) == original_id
        finally:
            restored.close()

    def test_restoring_a_name_not_in_current_rules_engine_still_resolves(self, tmp_path, dummy_ctx):
        """A rule name that was only ever registered via `preload_rule_names()`
        (never declared to rules_engine, e.g. a since-retired rule) must
        still resolve correctly after restore, even though it's absent from
        the current rules_engine's `all_rule_names`. `from_snapshot()` folds the
        snapshot's own rule names into the shared lookup db it builds, not
        just rules_engine's (see `_SnapshotRuleNames`).
        """
        rules_engine = attribution_lib.get_rules_engine()
        assert "never_declared_rule" not in rules_engine.all_rule_names

        mgr = EventManager(rules_engine=rules_engine)
        mgr.preload_rule_names({"never_declared_rule"})
        mgr.preload_event_types({"Test.retired_rule_event"})
        stream = mgr.create_stream("job_1")
        stream.register_observation_context(dummy_ctx)
        stream.add_event(
            "Test.retired_rule_event",
            {"k": "v"},
            Source(
                source_context=dummy_ctx,
                line_num=1,
                datum="d",
                time=datetime.now(),
                resources=AllJobResources,
                rule="never_declared_rule",
            ),
        )
        stream.flush()
        mgr.export_snapshot(tmp_path / "snap")
        original_id = mgr.rules.get_id("never_declared_rule")
        mgr.close()

        restored = EventManager.from_snapshot(tmp_path / "snap", rules_engine)
        try:
            (count,) = restored._conn.execute(
                "SELECT COUNT(*) FROM rules WHERE name = ?", ["never_declared_rule"]
            ).fetchone()
            assert count == 1, f"expected exactly one row for 'never_declared_rule', found {count}"
            assert restored.rules.get_id("never_declared_rule") == original_id
        finally:
            restored.close()

    def test_lookup_helpers_work_after_import(self, tmp_path, dummy_ctx):
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx, count=2)
        mgr.export_snapshot(tmp_path / "snap")
        mgr.close()

        restored = EventManager.from_snapshot(tmp_path / "snap", attribution_lib.get_rules_engine())
        type_id = restored.types.get_id("Test.event_0")
        assert type_id is not None
        assert restored.types.get_str(type_id) == "Test.event_0"

        rule_id = restored.rules.get_id("rule_0")
        assert rule_id is not None
        restored.close()

    def test_import_merges_current_lookup_catalogs(self, tmp_path):
        """A restored historical snapshot includes newly declared runtime names.

        `obs_sources`/`obs_formats` aren't rules_engine-derived and aren't part
        of the exported snapshot data at all here (see `_build_shared_lookup_db`):
        every `EventManager`, restored or not, is unconditionally seeded from the
        full current `ObsSource`/`ObsFormat` enum, so `new_source`/`new_format`
        resolving after restore is guaranteed by construction, not by any merge
        step. This just pins that guarantee against regression.
        """

        # Not SimpleNamespace: _build_shared_lookup_db() caches per rules_engine
        # in a WeakKeyDictionary, and SimpleNamespace instances can't be weakly
        # referenced. A real RuleEngine is a plain class, so it works fine.
        class _Engine:
            def __init__(self, all_rule_names, all_event_types):
                self.all_rule_names = all_rule_names
                self.all_event_types = all_event_types

        historical_engine = _Engine(
            all_rule_names={"historical_rule"},
            all_event_types={"Historical.event"},
        )
        mgr = EventManager(rules_engine=historical_engine)

        new_source = ObsSource.CSP_EVENT.value
        new_format = ObsFormat.CSP_EVENT_LIST.value
        mgr.export_snapshot(tmp_path / "snap")
        mgr.close()

        current_engine = _Engine(
            all_rule_names={"historical_rule", "current_rule"},
            all_event_types={"Historical.event", "Current.event"},
        )
        restored = EventManager.from_snapshot(tmp_path / "snap", current_engine)

        assert restored.rules.get_id("historical_rule") is not None
        assert restored.rules.get_id("current_rule") is not None
        assert restored.types.get_id("Historical.event") is not None
        assert restored.types.get_id("Current.event") is not None
        assert restored._obs_sources.get_id(new_source) is not None
        assert restored._obs_formats.get_id(new_format) is not None
        restored.close()

    def test_snapshot_resolves_rule_declared_after_it_was_taken(self, tmp_path, dummy_ctx):
        """A rule declared only by the current rules_engine, never seen by a
        snapshot taken before it existed, must still resolve after restore,
        matching the failure mode of `filter_rules.py`'s
        `event_stream.mgr.rules.get_ids(...)` call, which raises
        `UndeclaredRuleError` for any name that doesn't resolve.

        `_SnapshotRuleNames` already unions the live rules_engine's declared
        names with the snapshot's own historical names when building the
        restored manager's shared lookup db (see `from_snapshot()`). No cache
        invalidation or explicit merge step is needed; this is a property of
        how the shared lookup db is built, not an action taken after the fact.
        """

        class _Engine:
            def __init__(self, all_rule_names, all_event_types):
                self.all_rule_names = all_rule_names
                self.all_event_types = all_event_types

        old_engine = _Engine(all_rule_names={"OldRule"}, all_event_types={"Old.event"})
        mgr = EventManager(rules_engine=old_engine)
        mgr.preload_event_types({"Old.event"})
        stream = mgr.create_stream("job_1")
        stream.register_observation_context(dummy_ctx)
        stream.add_event(
            "Old.event",
            {"k": "v"},
            Source(
                source_context=dummy_ctx,
                line_num=1,
                datum="historical datum",
                time=datetime.now(),
                resources=AllJobResources,
                rule="OldRule",
            ),
        )
        stream.flush()
        mgr.export_snapshot(tmp_path / "snap")
        original_event_rows = mgr._conn.execute(
            "SELECT id, context_id, event_type, source_rule FROM events ORDER BY id"
        ).fetchall()
        original_rule_id = mgr.rules.get_id("OldRule")
        mgr.close()

        # The rules engine has since gained a new filter rule the snapshot
        # predates entirely.
        new_engine = _Engine(
            all_rule_names={"OldRule", "NewRule"},
            all_event_types={"Old.event"},
        )
        restored = EventManager.from_snapshot(tmp_path / "snap", new_engine)
        try:
            # The exact call shape that raised UndeclaredRuleError in the
            # original incident: get_ids() over names not necessarily known
            # to the snapshot.
            (new_rule_id,) = restored.rules.get_ids(["NewRule"])
            assert new_rule_id is not None

            # Historical data and ids must be untouched by restoring under a
            # newer rules_engine.
            assert restored.rules.get_id("OldRule") == original_rule_id
            restored_event_rows = restored._conn.execute(
                "SELECT id, context_id, event_type, source_rule FROM events ORDER BY id"
            ).fetchall()
            assert restored_event_rows == original_event_rows
        finally:
            restored.close()

    def test_export_import_survives_single_quote_in_snapshot_path(self, tmp_path, dummy_ctx):
        """Both `export_snapshot()` (`COPY ... TO '...'`) and `from_snapshot()`
        (`read_parquet('...')`) interpolate the snapshot path directly into SQL
        string literals. A single quote in the path (a valid filename character
        on Linux) must not break or corrupt either query, matching the
        quote-doubling already applied to the ATTACH path in
        `EventManager.__init__`.
        """
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx, count=2)
        quoted_snapshot_dir = tmp_path / "it's_a_snapshot"
        mgr.export_snapshot(quoted_snapshot_dir)
        mgr.close()

        restored = EventManager.from_snapshot(quoted_snapshot_dir, attribution_lib.get_rules_engine())
        rows = restored._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert rows[0] == 2
        restored.close()

    def test_close_removes_the_one_off_snapshot_lookup_file(self, tmp_path, dummy_ctx):
        """`from_snapshot()` builds its own one-off shared lookup db (see
        `_SnapshotRuleNames`), keyed in `_shared_lookup_cache` by an object
        nothing else keeps alive after `from_snapshot()` returns. Without
        `close()` unlinking it, that `.duckdb` file leaks on disk for the
        life of the process on every restore.
        """
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx, count=2)
        snapshot_dir = tmp_path / "snap"
        mgr.export_snapshot(snapshot_dir)
        mgr.close()

        restored = EventManager.from_snapshot(snapshot_dir, attribution_lib.get_rules_engine())
        (lookup_path,) = restored._conn.execute(
            "SELECT path FROM duckdb_databases() WHERE database_name='lookup'"
        ).fetchone()
        assert Path(lookup_path).exists()

        restored.close()
        assert not Path(lookup_path).exists(), "the one-off snapshot lookup file must be removed on close()"

        # Idempotent: a second close() must not raise (nothing left to clean up).
        restored.close()

    def test_close_retry_after_unlink_failure_does_not_redetach(self, tmp_path, dummy_ctx, monkeypatch):
        """If unlink() fails after DETACH+conn.close() already succeeded,
        close() must not have marked itself done (see the general retry
        contract), but a retry must not re-issue `DETACH lookup` against
        the now-already-closed connection either.
        """
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx, count=1)
        snapshot_dir = tmp_path / "snap"
        mgr.export_snapshot(snapshot_dir)
        mgr.close()

        restored = EventManager.from_snapshot(snapshot_dir, attribution_lib.get_rules_engine())

        real_unlink = Path.unlink
        calls = {"n": 0}

        def flaky_unlink(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("simulated unlink failure")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", flaky_unlink)

        with pytest.raises(OSError, match="simulated unlink failure"):
            restored.close()
        assert not restored._closed

        # Retry must not try to DETACH again on the already-closed connection.
        restored.close()
        assert restored._closed


class TestSnapshotSequences:
    def test_no_id_collision_after_import(self, tmp_path, dummy_ctx):
        """New events added after import should not collide with imported IDs."""
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx, count=3)
        mgr.export_snapshot(tmp_path / "snap")
        mgr.close()

        restored = EventManager.from_snapshot(tmp_path / "snap", attribution_lib.get_rules_engine())
        restored.preload_event_types({"Test.new_event"})
        restored.preload_rule_names({"new_rule"})
        stream = restored.create_stream("job_1")
        stream.add_event(
            "Test.new_event",
            {"new": True},
            Source(
                source_context=dummy_ctx,
                line_num=100,
                datum="new",
                time=datetime.now(),
                resources=AllJobResources,
                rule="new_rule",
            ),
        )
        stream.flush()

        rows = restored._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert rows[0] == 4

        ids = [r[0] for r in restored._conn.execute("SELECT id FROM events ORDER BY id").fetchall()]
        assert len(ids) == len(set(ids)), "Event IDs must be unique"
        restored.close()


class TestSnapshotErrors:
    def test_missing_snapshot_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Snapshot directory does not exist"):
            EventManager.from_snapshot(tmp_path / "nonexistent", attribution_lib.get_rules_engine())

    def test_partial_snapshot_raises(self, tmp_path, dummy_ctx):
        """If only some Parquet files exist, from_snapshot should fail."""
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx)
        mgr.export_snapshot(tmp_path / "partial")
        mgr.close()

        # Delete one file to simulate partial snapshot
        (tmp_path / "partial" / "events.parquet").unlink()

        with pytest.raises(FileNotFoundError, match="events.parquet"):
            EventManager.from_snapshot(tmp_path / "partial", attribution_lib.get_rules_engine())

    @pytest.mark.parametrize(
        "parquet_file,column",
        [
            ("events", "event_type"),
            ("events", "source_rule"),
            ("observation_contexts", "source"),
            ("observation_contexts", "format"),
        ],
    )
    def test_dangling_lookup_reference_raises(self, tmp_path, dummy_ctx, parquet_file, column):
        """`from_snapshot()` restores `events`/`observation_contexts` via a raw
        `INSERT ... SELECT * FROM read_parquet(...)`, bypassing `get_id()`'s
        usual guarantee that a stored id always resolves to a real lookup
        name. A corrupted or hand-edited snapshot with a dangling id must be
        rejected rather than silently imported.
        """
        import duckdb

        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx)
        snapshot_dir = tmp_path / "snap"
        mgr.export_snapshot(snapshot_dir)
        mgr.close()

        target = snapshot_dir / f"{parquet_file}.parquet"
        with duckdb.connect(":memory:") as scratch:
            scratch.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{target}')")
            scratch.execute(f"UPDATE t SET {column} = 999999999")
            scratch.execute(f"COPY t TO '{target}' (FORMAT PARQUET)")

        with pytest.raises(InvalidSnapshotError, match=column):
            EventManager.from_snapshot(snapshot_dir, attribution_lib.get_rules_engine())

    def test_obs_sources_local_overlay_import_detects_id_collision(self, tmp_path, dummy_ctx):
        """`from_snapshot()` restores `obs_sources`/`obs_formats` local-overlay
        rows via a raw SQL `INSERT` that only filters out rows whose *name*
        already resolves, with no id-based check at all. A snapshot row whose
        id happens to collide with an existing shared-tier name (under a
        different name) must be rejected, not silently create two names
        aliased to the same id.
        """
        import duckdb

        from attribution_lib.exceptions import StableIdCollisionError

        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx)
        snapshot_dir = tmp_path / "snap"
        mgr.export_snapshot(snapshot_dir)
        existing_source_id = mgr._obs_sources.get_id(ObsSource.SYSLOG.value)
        mgr.close()

        target = snapshot_dir / "obs_sources.parquet"
        with duckdb.connect(":memory:") as scratch:
            scratch.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{target}')")
            scratch.execute("INSERT INTO t VALUES (?, ?)", [existing_source_id, "not_a_real_source_name"])
            scratch.execute(f"COPY t TO '{target}' (FORMAT PARQUET)")

        with pytest.raises(StableIdCollisionError):
            EventManager.from_snapshot(snapshot_dir, attribution_lib.get_rules_engine())
