# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import stat
from datetime import datetime, timedelta, timezone

import pytest

from attribution_blueprint.connectors.logs.loki import range_cache as range_cache_module
from attribution_blueprint.connectors.logs.loki.client import LokiQueryBudgetExceeded, datetime_to_nanoseconds
from attribution_blueprint.connectors.logs.loki.range_cache import FileRangeCacheStore, LokiRangeCache

NOW = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)


def _entry(at: datetime, value: str) -> dict:
    return {"timestamp": datetime_to_nanoseconds(at), "value": value}


def _timestamp(item: dict) -> int:
    return int(item["timestamp"])


def test_range_cache_reuses_complete_buckets_and_filters_requested_window(tmp_path):
    cache = LokiRangeCache(FileRangeCacheStore(tmp_path))
    start = NOW - timedelta(hours=1)
    end = start + timedelta(minutes=10)
    calls: list[tuple[datetime, datetime]] = []

    def fetch(_query, missing_start, missing_end):
        calls.append((missing_start, missing_end))
        return [
            _entry(missing_start, "before"),
            _entry(missing_start + timedelta(minutes=3), "inside-a"),
            _entry(missing_end - timedelta(minutes=3), "inside-b"),
        ]

    first = cache.query(
        query='{job="7"}',
        config={"endpoint": "loki-a"},
        start=start + timedelta(minutes=2),
        end=end - timedelta(minutes=2),
        fetch=fetch,
        timestamp=_timestamp,
        now=NOW,
    )
    second = cache.query(
        query='{job="7"}',
        config={"endpoint": "loki-a"},
        start=start + timedelta(minutes=3),
        end=end - timedelta(minutes=3),
        fetch=fetch,
        timestamp=_timestamp,
        now=NOW,
    )

    # One fetch covers both aligned buckets; the second query is served from
    # cache and its exclusive end drops the entry at exactly end - 3 minutes.
    assert calls == [(start, end)]
    assert [item["value"] for item in first] == ["inside-a", "inside-b"]
    assert [item["value"] for item in second] == ["inside-a"]


def test_range_cache_fetches_only_missing_and_hot_buckets(tmp_path):
    cache = LokiRangeCache(FileRangeCacheStore(tmp_path))
    cold_start = NOW - timedelta(hours=1)
    cache._store_bucket(
        query="query",
        config={"tenant": "one"},
        bucket_start=cold_start,
        entries=[_entry(cold_start + timedelta(minutes=1), "cached")],
        now=NOW,
    )
    calls: list[tuple[datetime, datetime]] = []

    def fetch(_query, start, end):
        calls.append((start, end))
        return [_entry(start + timedelta(minutes=1), f"fresh-{len(calls)}")]

    merged = cache.query(
        query="query",
        config={"tenant": "one"},
        start=cold_start,
        end=NOW,
        fetch=fetch,
        timestamp=_timestamp,
        now=NOW,
    )

    # The cached bucket is skipped; everything after it is one missing range.
    assert calls == [(cold_start + timedelta(minutes=5), NOW)]
    assert [item["value"] for item in merged] == ["cached", "fresh-1"]
    # Hot buckets (within ten minutes of now) are never stored.
    store = cache._store
    cache._store_bucket("query", {"tenant": "one"}, NOW - timedelta(minutes=5), [], now=NOW)
    assert store.get(cache._bucket_key("query", {"tenant": "one"}, NOW - timedelta(minutes=5))) is None
    cache._store_bucket("query", {"tenant": "one"}, NOW - timedelta(minutes=20), [], now=NOW)
    assert store.get(cache._bucket_key("query", {"tenant": "one"}, NOW - timedelta(minutes=20))) == []


def test_range_cache_isolates_query_and_configuration(tmp_path):
    cache = LokiRangeCache(FileRangeCacheStore(tmp_path))
    bucket = NOW - timedelta(hours=2)
    cache._store_bucket("query", {"tenant": "one"}, bucket, [_entry(bucket, "one")], now=NOW)
    cache._store_bucket("query", {"tenant": "two"}, bucket, [_entry(bucket, "two")], now=NOW)
    cache._store_bucket("other", {"tenant": "one"}, bucket, [_entry(bucket, "other")], now=NOW)

    def fail(_query, _start, _end):
        raise AssertionError("cached bucket must not be fetched")

    for query, config, expected in (("query", {"tenant": "one"}, "one"), ("query", {"tenant": "two"}, "two")):
        result = cache.query(
            query=query,
            config=config,
            start=bucket,
            end=bucket + timedelta(minutes=5),
            fetch=fail,
            timestamp=_timestamp,
            now=NOW,
        )
        assert [item["value"] for item in result] == [expected]


def test_range_cache_keeps_partial_results_when_budget_runs_out(tmp_path):
    cache = LokiRangeCache(FileRangeCacheStore(tmp_path))
    start = NOW - timedelta(hours=1)
    cache._store_bucket("query", {}, start, [_entry(start + timedelta(minutes=1), "cached")], now=NOW)

    def fetch(_query, missing_start, _missing_end):
        raise LokiQueryBudgetExceeded([_entry(missing_start + timedelta(minutes=1), "partial")])

    with pytest.raises(LokiQueryBudgetExceeded) as excinfo:
        cache.query(query="query", config={}, start=start, end=NOW, fetch=fetch, timestamp=_timestamp, now=NOW)

    assert [item["value"] for item in excinfo.value.entries] == ["cached", "partial"]
    # The failed range was not written as a complete bucket.
    assert cache.query(
        query="query",
        config={},
        start=start + timedelta(minutes=5),
        end=start + timedelta(minutes=10),
        fetch=lambda _q, s, _e: [_entry(s, "refetched")],
        timestamp=_timestamp,
        now=NOW,
    ) == [_entry(start + timedelta(minutes=5), "refetched")]


def test_range_cache_honors_inclusive_end(tmp_path):
    cache = LokiRangeCache(FileRangeCacheStore(tmp_path))
    start = NOW - timedelta(hours=1)
    end = start + timedelta(minutes=5)
    entries = [_entry(start, "start"), _entry(end, "at-end")]

    def fetch(_query, range_start, range_end):
        return [
            item
            for item in entries
            if datetime_to_nanoseconds(range_start) <= item["timestamp"] < datetime_to_nanoseconds(range_end)
        ]

    exclusive = cache.query(query="q", config={}, start=start, end=end, fetch=fetch, timestamp=_timestamp, now=NOW)
    inclusive = cache.query(
        query="q",
        config={},
        start=start,
        end=end + timedelta(microseconds=1),
        inclusive_end_ns=datetime_to_nanoseconds(end),
        fetch=fetch,
        timestamp=_timestamp,
        now=NOW,
    )

    assert [item["value"] for item in exclusive] == ["start"]
    assert [item["value"] for item in inclusive] == ["start", "at-end"]


def test_range_cache_refetches_expired_bucket(tmp_path, monkeypatch):
    current = [1_000_000.0]
    monkeypatch.setattr(range_cache_module.time, "time", lambda: current[0])
    cache = LokiRangeCache(FileRangeCacheStore(tmp_path), ttl_seconds=60)
    bucket = NOW - timedelta(hours=1)
    cache._store_bucket("q", {}, bucket, [_entry(bucket, "old")], now=NOW)
    calls = []

    def fetch(_query, start, _end):
        calls.append(start)
        return [_entry(start, "fresh")]

    current[0] += 61
    result = cache.query(
        query="q",
        config={},
        start=bucket,
        end=bucket + timedelta(minutes=5),
        fetch=fetch,
        timestamp=_timestamp,
        now=NOW,
    )

    assert calls == [bucket]
    assert [item["value"] for item in result] == ["fresh"]


def test_range_cache_rejects_non_json_configuration(tmp_path):
    cache = LokiRangeCache(FileRangeCacheStore(tmp_path))
    with pytest.raises(TypeError):
        cache._store_bucket("q", {"when": NOW}, NOW - timedelta(hours=1), [], now=NOW)


def test_file_store_uses_private_permissions_and_expires_files(tmp_path, monkeypatch):
    current = [1_000_000.0]
    monkeypatch.setattr(range_cache_module.time, "time", lambda: current[0])
    root = tmp_path / "range-cache"
    store = FileRangeCacheStore(root)
    key = "a" * 64 + ":1700000000"
    store.put(key, [{"timestamp": 1}], ttl_seconds=30)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    written = next(root.glob("*/*.json"))
    assert stat.S_IMODE(written.stat().st_mode) == 0o600
    assert store.get(key) == [{"timestamp": 1}]
    (root / "aa" / "corrupt.json").write_text("{not json")

    current[0] += 31
    assert store.get(key) is None
    assert list(root.glob("*/*.json")) == [root / "aa" / "corrupt.json"]


def test_file_store_rejects_unexpected_keys(tmp_path):
    store = FileRangeCacheStore(tmp_path)
    with pytest.raises(ValueError):
        store.get("../escape")


def test_cached_returns_stored_buckets_without_fetching(tmp_path):
    cache = LokiRangeCache(FileRangeCacheStore(tmp_path))
    start = NOW - timedelta(hours=2)
    cache._store_bucket("q", {}, start, [_entry(start + timedelta(minutes=1), "stored")], now=NOW)

    entries = cache.cached(
        query="q", config={}, start=start, end=start + timedelta(minutes=15), timestamp=_timestamp, now=NOW
    )

    assert [item["value"] for item in entries] == ["stored"]
    assert cache.cached(query="q", config={}, start=start, end=start, timestamp=_timestamp, now=NOW) == []
    # A hot bucket is never read from the store, and a different query sees nothing.
    assert (
        cache.cached(query="q", config={}, start=NOW - timedelta(minutes=5), end=NOW, timestamp=_timestamp, now=NOW)
        == []
    )
    assert (
        cache.cached(
            query="other", config={}, start=start, end=start + timedelta(minutes=15), timestamp=_timestamp, now=NOW
        )
        == []
    )
