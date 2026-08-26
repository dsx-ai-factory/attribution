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

import duckdb
import pytest

import attribution_lib
from attribution_lib.events import EventManager
from attribution_lib.resources import AllJobResources
from attribution_lib.rule_types.definitions import RuleEmission
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
    def test_rejects_incompatible_schema_version(self, tmp_path):
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        (snapshot_dir / EventManager._SCHEMA_VERSION_FILE).write_text("0")

        with pytest.raises(ValueError, match="stored=0, current=3"):
            EventManager.from_snapshot(snapshot_dir, attribution_lib.get_rules_engine())

    def test_export_import_preserves_events(self, tmp_path, dummy_ctx):
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx, count=5)
        snapshot_dir = tmp_path / "snap's"

        mgr.export_snapshot(snapshot_dir)
        mgr.close()

        restored = EventManager.from_snapshot(snapshot_dir, attribution_lib.get_rules_engine())
        rows = restored._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert rows[0] == 5
        restored.close()

    @pytest.mark.parametrize("with_version_marker", [True, False])
    def test_imports_schema_v1_without_reextraction(self, tmp_path, dummy_ctx, with_version_marker):
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _add_events(mgr, "job_1", dummy_ctx, count=1)
        snapshot_dir = tmp_path / "snap's"
        mgr.export_snapshot(snapshot_dir)
        mgr.close()

        events_path = snapshot_dir / "events.parquet"
        legacy_path = snapshot_dir / "events-v1.parquet"
        legacy_columns = (
            "id, context_id, event_type, timestamp, status, line_num, "
            "source_datum, source_rule, resources, obs_context_id, payload"
        )
        duckdb.read_parquet(str(events_path)).project(legacy_columns).write_parquet(str(legacy_path))
        events_path.unlink()
        legacy_path.rename(events_path)
        version_path = snapshot_dir / EventManager._SCHEMA_VERSION_FILE
        if with_version_marker:
            version_path.write_text("1")
        else:
            version_path.unlink()

        restored = EventManager.from_snapshot(snapshot_dir, attribution_lib.get_rules_engine())
        rows = restored._conn.execute("SELECT source_datum, source_rule FROM events").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "datum_0"
        assert restored.rules.get_str(rows[0][1]) == "rule_0"
        restored.close()

    def test_imports_schema_v2_emitter_provenance(self, tmp_path, dummy_ctx):
        mgr = EventManager(rules_engine=attribution_lib.get_rules_engine())
        mgr.preload_event_types({"Test.specific"})
        mgr.preload_rule_names({"specific_rule", "specific_extractor"})
        stream = mgr.create_stream("job_1")
        stream.register_observation_context(dummy_ctx)
        stream.add_event(
            "Test.specific",
            {"symptom": "Specific Error"},
            Source(
                source_context=dummy_ctx,
                line_num=99,
                datum="specific_datum",
                time=datetime.now(),
                resources=AllJobResources,
                rule="specific_rule",
            ),
        )
        snapshot_dir = tmp_path / "snap"
        mgr.export_snapshot(snapshot_dir)
        mgr.close()

        rules_path = snapshot_dir / "rules.parquet"
        emitter_id = (
            duckdb.read_parquet(str(rules_path)).filter("name = 'specific_extractor'").project("id").fetchone()[0]
        )
        events_path = snapshot_dir / "events.parquet"
        legacy_path = snapshot_dir / "events-v2.parquet"
        v2_columns = (
            "id, context_id, event_type, timestamp, status, line_num, "
            f"source_datum, source_rule, {emitter_id}::INTEGER AS emitter_rule, "
            "resources, obs_context_id, payload"
        )
        duckdb.read_parquet(str(events_path)).project(v2_columns).write_parquet(str(legacy_path))
        events_path.unlink()
        legacy_path.rename(events_path)
        (snapshot_dir / EventManager._SCHEMA_VERSION_FILE).write_text("2")

        restored = EventManager.from_snapshot(
            snapshot_dir,
            attribution_lib.get_rules_engine(),
        )
        rows = restored._conn.execute(
            """SELECT json_extract_string(e.payload, '$.symptom'), r.name
               FROM events e
               JOIN rules r ON e.source_rule = r.id"""
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "Specific Error"
        assert RuleEmission.from_storage_key(rows[0][1]) == RuleEmission(
            credited_rule_id="specific_rule",
            emitter_rule_id="specific_extractor",
        )
        restored.close()

    def test_export_import_preserves_event_content(self, tmp_path, dummy_ctx):
        rules_engine = attribution_lib.get_rules_engine()
        emission = next(
            emission
            for emission in rules_engine.rule_emissions
            if emission.credited_rule_id != emission.emitter_rule_id
        )
        mgr = EventManager(rules_engine=rules_engine)
        mgr.preload_event_types({"Test.specific"})
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
                rule=emission.credited_rule_id,
            ),
            emitter_rule_id=emission.emitter_rule_id,
        )
        stream.flush()
        mgr.export_snapshot(tmp_path / "snap")
        mgr.close()

        restored = EventManager.from_snapshot(tmp_path / "snap", attribution_lib.get_rules_engine())
        stream2 = restored.create_stream("job_1")

        rows = stream2.query_raw(
            """SELECT e.source_datum, r.name
               FROM events e
               JOIN rules r ON e.source_rule = r.id
               WHERE e.context_id=?""",
            [stream2.ctx_id],
        )
        assert len(rows) == 1
        assert rows[0][0] == "specific_datum"
        assert RuleEmission.from_storage_key(rows[0][1]) == emission
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
