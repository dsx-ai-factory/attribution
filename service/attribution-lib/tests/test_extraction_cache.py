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

import pytest

import attribution_lib
from attribution_lib.checkpoint_backends import MemoryCheckpointBackend
from attribution_lib.events import EventManager
from attribution_lib.extraction_cache import (
    DataDiffResult,
    DataFingerprint,
    ExtractionCacheMetadata,
    _hash_entry,
    compute_data_fingerprint,
    diff_data,
    load_extraction_cache,
    load_or_update_extraction_cache,
    save_extraction_cache,
)
from attribution_lib.resources import AllJobResources
from attribution_lib.sources import Source


def _populate(event_manager: EventManager, dummy_ctx, count: int = 3) -> None:
    event_manager.preload_event_types({f"Test.event_{i}" for i in range(count)})
    event_manager.preload_rule_names({f"rule_{i}" for i in range(count)})
    stream = event_manager.create_stream("job_42")
    stream.register_observation_context(dummy_ctx)
    for i in range(count):
        stream.add_event(
            f"Test.event_{i}",
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


class TestHashEntry:
    def test_string_entry(self):
        hash_result = _hash_entry("some log line")
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # sha256 hex digest

    def test_string_determinism(self):
        assert _hash_entry("hello") == _hash_entry("hello")
        assert _hash_entry("hello") != _hash_entry("world")

    def test_dict_determinism(self):
        entry_a = {"b": 2, "a": 1}
        entry_b = {"a": 1, "b": 2}
        assert _hash_entry(entry_a) == _hash_entry(entry_b)

    def test_dict_with_datetime(self):
        entry = {"ts": datetime(2026, 1, 1, 12, 0), "msg": "test"}
        hash_result = _hash_entry(entry)
        assert isinstance(hash_result, str)
        assert _hash_entry(entry) == hash_result


class TestComputeDataFingerprint:
    def test_empty_list(self):
        assert compute_data_fingerprint([]) is None

    def test_single_entry(self):
        fingerprint = compute_data_fingerprint(["only line"])
        assert fingerprint is not None
        assert fingerprint.count == 1
        assert len(fingerprint.sample_hashes) == 1
        assert fingerprint.first_hash == fingerprint.sample_hashes[0]

    def test_multiple_entries(self):
        fingerprint = compute_data_fingerprint(["line_0", "line_1", "line_2"])
        assert fingerprint is not None
        assert fingerprint.count == 3
        assert fingerprint.first_hash == _hash_entry("line_0")
        assert fingerprint.sample_hashes[-1] == _hash_entry("line_2")

    def test_spread_sampling(self):
        entries = [f"line_{i}" for i in range(20)]
        fingerprint = compute_data_fingerprint(entries)
        assert fingerprint is not None
        assert fingerprint.count == 20
        assert len(fingerprint.sample_hashes) == 8
        assert fingerprint.sample_hashes[0] == _hash_entry("line_0")
        assert fingerprint.sample_hashes[-1] == _hash_entry("line_19")

    def test_middle_change_detected(self):
        entries_a = ["start", "middle_a", "end"]
        entries_b = ["start", "middle_b", "end"]
        fp_a = compute_data_fingerprint(entries_a)
        fp_b = compute_data_fingerprint(entries_b)
        assert fp_a != fp_b

    def test_dict_entries(self):
        entries = [{"node": "n1", "val": 10}, {"node": "n2", "val": 20}]
        fingerprint = compute_data_fingerprint(entries)
        assert fingerprint is not None
        assert fingerprint.count == 2
        assert fingerprint.first_hash == _hash_entry(entries[0])

    def test_round_trip(self):
        fingerprint = compute_data_fingerprint(["a", "b", "c"])
        restored = DataFingerprint.from_dict(fingerprint.to_dict())
        assert restored == fingerprint

    def test_backward_compat_from_dict(self):
        old_format = {"count": 10, "first_hash": "aaa", "last_hash": "bbb"}
        restored = DataFingerprint.from_dict(old_format)
        assert restored.count == 10
        assert restored.sample_hashes == ("aaa", "bbb")
        assert restored.first_hash == "aaa"


class TestDiffData:
    def _fingerprint(self, count, first="aaa", last="zzz"):
        return DataFingerprint(count=count, sample_hashes=(first, last))

    def test_unchanged(self):
        cached = {"applog": self._fingerprint(100)}
        current = {"applog": self._fingerprint(100)}
        diffs = diff_data(cached, current)
        assert diffs["applog"].result == DataDiffResult.UNCHANGED

    def test_appended(self):
        cached = {"applog": self._fingerprint(100, first="aaa", last="old_last")}
        current = {"applog": self._fingerprint(150, first="aaa", last="new_last")}
        diffs = diff_data(cached, current)
        assert diffs["applog"].result == DataDiffResult.APPENDED
        assert diffs["applog"].cached_count == 100

    def test_modified_first_hash_changed(self):
        cached = {"applog": self._fingerprint(100, first="aaa")}
        current = {"applog": self._fingerprint(100, first="bbb")}
        diffs = diff_data(cached, current)
        assert diffs["applog"].result == DataDiffResult.MODIFIED

    def test_modified_count_decreased(self):
        cached = {"applog": self._fingerprint(100, first="aaa")}
        current = {"applog": self._fingerprint(50, first="aaa")}
        diffs = diff_data(cached, current)
        assert diffs["applog"].result == DataDiffResult.MODIFIED

    def test_new_source(self):
        cached = {}
        current = {"syslog": self._fingerprint(50)}
        diffs = diff_data(cached, current)
        assert diffs["syslog"].result == DataDiffResult.NEW_SOURCE
        assert diffs["syslog"].cached_count == 0

    def test_removed_source(self):
        cached = {"syslog": self._fingerprint(50)}
        current = {}
        diffs = diff_data(cached, current)
        assert diffs["syslog"].result == DataDiffResult.REMOVED
        assert diffs["syslog"].cached_count == 50

    def test_mixed_sources(self):
        cached = {
            "applog": self._fingerprint(100, first="aaa", last="bbb"),
            "syslog": self._fingerprint(50, first="ccc", last="ddd"),
        }
        current = {
            "applog": self._fingerprint(100, first="aaa", last="bbb"),
            "ufmlog": self._fingerprint(20, first="eee", last="fff"),
        }
        diffs = diff_data(cached, current)
        assert diffs["applog"].result == DataDiffResult.UNCHANGED
        assert diffs["syslog"].result == DataDiffResult.REMOVED
        assert diffs["ufmlog"].result == DataDiffResult.NEW_SOURCE

    def test_skipped_source_treated_as_unchanged(self):
        cached = {"applog": self._fingerprint(100), "nvlink_metrics": self._fingerprint(50)}
        current = {"applog": self._fingerprint(100)}
        diffs = diff_data(cached, current, skipped_sources={"nvlink_metrics"})
        assert diffs["applog"].result == DataDiffResult.UNCHANGED
        assert diffs["nvlink_metrics"].result == DataDiffResult.UNCHANGED
        assert diffs["nvlink_metrics"].cached_count == 50

    def test_skipped_source_without_flag_is_removed(self):
        cached = {"applog": self._fingerprint(100), "nvlink_metrics": self._fingerprint(50)}
        current = {"applog": self._fingerprint(100)}
        diffs = diff_data(cached, current)
        assert diffs["nvlink_metrics"].result == DataDiffResult.REMOVED

    def test_empty_both(self):
        diffs = diff_data({}, {})
        assert diffs == {}


class TestExtractionCacheMetadata:
    def test_round_trip(self):
        metadata = ExtractionCacheMetadata(extractor_hash="abc123", attribution_hash="def456", job_id="42")
        restored = ExtractionCacheMetadata.from_dict(metadata.to_dict())
        assert restored == metadata

    def test_round_trip_with_fingerprints(self):
        fingerprints = {
            "applog": DataFingerprint(count=100, sample_hashes=("aaa", "bbb")),
            "syslog": DataFingerprint(count=50, sample_hashes=("ccc", "ddd")),
        }
        metadata = ExtractionCacheMetadata(
            extractor_hash="abc", attribution_hash="def", job_id="42", data_fingerprints=fingerprints
        )
        restored = ExtractionCacheMetadata.from_dict(metadata.to_dict())
        assert restored == metadata
        assert restored.data_fingerprints["applog"].count == 100

    def test_backward_compat_no_fingerprints(self):
        raw = {"extractor_hash": "abc", "attribution_hash": "def", "job_id": "42"}
        metadata = ExtractionCacheMetadata.from_dict(raw)
        assert metadata.data_fingerprints == {}
        assert metadata.raw_keywords is None

    def test_round_trip_with_raw_keywords(self):
        """Preserve raw keyword strings in checkpoint metadata."""
        metadata = ExtractionCacheMetadata(
            extractor_hash="abc",
            attribution_hash="def",
            job_id="42",
            raw_keywords=('File "', "RuntimeError"),
        )

        restored = ExtractionCacheMetadata.from_dict(metadata.to_dict())

        assert restored == metadata

    def test_round_trip_with_previous_outcome(self):
        metadata = ExtractionCacheMetadata(
            extractor_hash="abc", attribution_hash="def", job_id="42", previous_outcome="Fail InfraHW"
        )
        data = metadata.to_dict()
        assert data["previous_outcome"] == "Fail InfraHW"
        restored = ExtractionCacheMetadata.from_dict(data)
        assert restored == metadata
        assert restored.previous_outcome == "Fail InfraHW"

    def test_previous_outcome_none_omitted_from_dict(self):
        metadata = ExtractionCacheMetadata(extractor_hash="abc", attribution_hash="def", job_id="42")
        data = metadata.to_dict()
        assert "previous_outcome" not in data

    def test_backward_compat_no_previous_outcome(self):
        raw = {"extractor_hash": "abc", "attribution_hash": "def", "job_id": "42"}
        metadata = ExtractionCacheMetadata.from_dict(raw)
        assert metadata.previous_outcome is None

    def test_round_trip_with_attempt_count_and_timestamp(self):
        metadata = ExtractionCacheMetadata(
            extractor_hash="abc",
            attribution_hash="def",
            job_id="42",
            attribution_attempt_count=3,
            last_attribution_timestamp="2026-02-23T10:00:00+00:00",
        )
        data = metadata.to_dict()
        assert data["attribution_attempt_count"] == 3
        assert data["last_attribution_timestamp"] == "2026-02-23T10:00:00+00:00"
        restored = ExtractionCacheMetadata.from_dict(data)
        assert restored == metadata

    def test_attempt_count_zero_omitted_from_dict(self):
        metadata = ExtractionCacheMetadata(extractor_hash="abc", attribution_hash="def", job_id="42")
        data = metadata.to_dict()
        assert "attribution_attempt_count" not in data
        assert "last_attribution_timestamp" not in data

    def test_backward_compat_no_attempt_fields(self):
        raw = {"extractor_hash": "abc", "attribution_hash": "def", "job_id": "42"}
        metadata = ExtractionCacheMetadata.from_dict(raw)
        assert metadata.attribution_attempt_count == 0
        assert metadata.last_attribution_timestamp is None


class TestMemoryCheckpointBackend:
    def test_cache_without_raw_keywords_requires_reextraction(self, dummy_ctx):
        """Invalidate legacy metadata that cannot restore raw keywords.

        Args:
            dummy_ctx: Observation context fixture used to populate events.
        """
        backend = MemoryCheckpointBackend()
        fingerprints = {"applog": DataFingerprint(count=10, sample_hashes=("aaa", "bbb"))}
        event_manager = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _populate(event_manager, dummy_ctx, count=4)
        metadata = ExtractionCacheMetadata(
            extractor_hash=attribution_lib.get_rules_engine().extractor_hash(),
            attribution_hash=attribution_lib.get_rules_engine().attribution_hash(),
            job_id="job_42",
            data_fingerprints=fingerprints,
            raw_keywords=None,
        )
        backend.save("cache_key", event_manager, metadata)

        result = load_or_update_extraction_cache(backend, "cache_key", fingerprints)

        assert result.needs_full_reextraction

    def test_save_then_load(self, dummy_ctx):
        backend = MemoryCheckpointBackend()
        event_manager = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _populate(event_manager, dummy_ctx, count=4)

        save_extraction_cache(backend, "cache_key", event_manager, "job_42")

        result = load_extraction_cache(backend, "cache_key")
        assert result is not None

        restored_event_manager, metadata = result
        assert metadata.job_id == "job_42"

        rows = restored_event_manager._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert rows[0] == 4
        restored_event_manager.close()

    def test_load_returns_none_when_missing(self):
        backend = MemoryCheckpointBackend()
        assert load_extraction_cache(backend, "nonexistent") is None

    def test_cache_hit_all_unchanged(self, dummy_ctx):
        backend = MemoryCheckpointBackend()
        fingerprints = {"applog": DataFingerprint(count=10, sample_hashes=("aaa", "bbb"))}

        event_manager = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _populate(event_manager, dummy_ctx, count=4)
        save_extraction_cache(backend, "cache_key", event_manager, "job_42", data_fingerprints=fingerprints)

        result = load_or_update_extraction_cache(backend, "cache_key", fingerprints)
        assert not result.needs_full_reextraction
        assert result.event_manager is not None
        assert result.data_diffs["applog"].result == DataDiffResult.UNCHANGED
        result.event_manager.close()

    def test_incremental_on_appended_source(self, dummy_ctx):
        backend = MemoryCheckpointBackend()
        cached_fingerprints = {"applog": DataFingerprint(count=10, sample_hashes=("aaa", "bbb"))}

        event_manager = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _populate(event_manager, dummy_ctx, count=4)
        save_extraction_cache(backend, "cache_key", event_manager, "job_42", data_fingerprints=cached_fingerprints)

        current_fingerprints = {"applog": DataFingerprint(count=15, sample_hashes=("aaa", "new_last"))}
        result = load_or_update_extraction_cache(backend, "cache_key", current_fingerprints)
        assert not result.needs_full_reextraction
        assert result.event_manager is not None
        assert result.data_diffs["applog"].result == DataDiffResult.APPENDED
        result.event_manager.close()

    def test_overwrite_replaces_previous(self, dummy_ctx):
        backend = MemoryCheckpointBackend()

        event_manager_1 = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _populate(event_manager_1, dummy_ctx, count=2)
        save_extraction_cache(backend, "cache_key", event_manager_1, "job_42")

        event_manager_2 = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _populate(event_manager_2, dummy_ctx, count=5)
        save_extraction_cache(backend, "cache_key", event_manager_2, "job_42")

        result = load_extraction_cache(backend, "cache_key")
        assert result is not None
        restored_event_manager, _metadata = result
        rows = restored_event_manager._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert rows[0] == 5
        restored_event_manager.close()

    def test_evicts_oldest_when_over_max_entries(self, dummy_ctx):
        backend = MemoryCheckpointBackend(max_entries=2)

        for i in range(3):
            em = EventManager(rules_engine=attribution_lib.get_rules_engine())
            _populate(em, dummy_ctx, count=1)
            save_extraction_cache(backend, f"key_{i}", em, f"job_{i}")

        assert len(backend._store) == 2
        assert "key_0" not in backend._store
        assert "key_1" in backend._store
        assert "key_2" in backend._store

    def test_cached_state_isolated_from_mutations(self, dummy_ctx):
        """Mutating the EventManager after save must not affect cached state."""
        backend = MemoryCheckpointBackend()
        event_manager = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _populate(event_manager, dummy_ctx, count=3)
        save_extraction_cache(backend, "cache_key", event_manager, "job_42")

        _populate(event_manager, dummy_ctx, count=5)
        rows_after_mutation = event_manager._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert rows_after_mutation[0] == 8

        result = load_extraction_cache(backend, "cache_key")
        assert result is not None
        restored_em, _ = result
        rows_cached = restored_em._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert rows_cached[0] == 3
        restored_em.close()
        event_manager.close()

    def test_cleanup_is_noop(self):
        backend = MemoryCheckpointBackend()
        assert backend.cleanup() == 0
        assert backend.cleanup(max_age_hours=0) == 0


class TestSaveRequiresRulesEngine:
    def test_save_raises_without_rules_engine(self, dummy_ctx):
        backend = MemoryCheckpointBackend()
        event_manager = EventManager(rules_engine=attribution_lib.get_rules_engine())
        _populate(event_manager, dummy_ctx)

        saved = attribution_lib._rules_engine
        attribution_lib._rules_engine = None
        try:
            with pytest.raises(Exception):
                save_extraction_cache(backend, "cache_key", event_manager, "job_42")
        finally:
            attribution_lib._rules_engine = saved
            event_manager.close()
