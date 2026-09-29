# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Bucketed range cache for Loki query results.

Batch collection re-reads the same trailing window many times: every run
covers the previous run's tail. The cache stores completed five-minute
buckets per (query, configuration) and only fetches the buckets it lacks.
Buckets whose end is within `hot_buffer` of now are never cached because
Loki may still be ingesting late lines for them.

The store is any object with `get(key) -> list | None` and
`put(key, value, *, ttl_seconds)`; `FileRangeCacheStore` is the bundled one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypeVar

from attribution_blueprint.connectors.logs.loki.client import LokiQueryBudgetExceeded, datetime_to_nanoseconds

T = TypeVar("T")

RangeFetcher = Callable[[str, datetime, datetime], list[T]]
TimestampExtractor = Callable[[T], int]
NodeExtractor = Callable[[T], str]

_KEY_RE = re.compile(r"^[0-9a-f]{64}:\d+$")


class FileRangeCacheStore:
    """Dependency-free store: one JSON file per bucket below `root`.

    Files are written atomically and expire by the timestamp stored inside
    them, so a store can be shared by sequential runs. The directory and files
    are created with owner-only permissions because cached log lines may carry
    tenant data.

    A bucket value is a flat list of entries or a ``{node: [entries]}`` mapping
    when the range cache partitions by node. A reader from before node
    partitioning treats the mapping as corrupt and drops the file, so a cache
    directory shared across versions only loses hits, never returns wrong data.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if _KEY_RE.match(key) is None:
            raise ValueError(f"unsupported range cache key: {key!r}")
        return self.root / key[:2] / f"{key.replace(':', '_')}.json"

    def get(self, key: str) -> list | dict | None:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("value"), (list, dict)):
            path.unlink(missing_ok=True)
            return None
        expires_at = payload.get("expires_at")
        if isinstance(expires_at, (int, float)) and expires_at <= time.time():
            path.unlink(missing_ok=True)
            return None
        return payload["value"]

    def put(self, key: str, value: list | dict, *, ttl_seconds: float) -> None:
        path = self._path(key)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = json.dumps({"expires_at": time.time() + ttl_seconds, "value": value}, separators=(",", ":"))
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(payload)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        os.replace(temporary, path)


class LokiRangeCache:
    """Five-minute, range-aware cache for Loki query results."""

    BUCKET = timedelta(minutes=5)
    TTL_SECONDS = 5 * 24 * 60 * 60
    HOT_BUFFER = timedelta(minutes=10)

    def __init__(
        self,
        store,
        *,
        bucket: timedelta = BUCKET,
        ttl_seconds: float = TTL_SECONDS,
        hot_buffer: timedelta = HOT_BUFFER,
    ) -> None:
        if bucket <= timedelta(0):
            raise ValueError("bucket must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if hot_buffer < timedelta(0):
            raise ValueError("hot_buffer must not be negative")
        self._store = store
        self._bucket = bucket
        self._ttl_seconds = ttl_seconds
        self._hot_buffer = hot_buffer

    def query(
        self,
        *,
        query: str,
        config: Mapping[str, object],
        start: datetime,
        end: datetime,
        inclusive_end_ns: int | None = None,
        fetch: RangeFetcher[T],
        timestamp: TimestampExtractor[T],
        now: datetime,
        node_of: NodeExtractor[T] | None = None,
    ) -> list[T]:
        """Return one requested window, fetching only the missing ranges.

        `start` is inclusive and `end` exclusive. `inclusive_end_ns` restores an
        inclusive upper bound on the returned entries when the caller's request
        was inclusive. `fetch(query, range_start, range_end)` must return every
        entry inside the half-open missing range. A `LokiQueryBudgetExceeded`
        raised by `fetch` stops fetching; the partial result is re-raised the
        same way so callers can keep what was retrieved.

        `node_of`, when set, partitions each stored bucket by node so a later
        `read(nodes=...)` materializes only those nodes' lines. This method
        still returns every node's entries for the requested window.
        """
        if end <= start:
            return []
        buckets = self._buckets(start, end)
        # Buckets are held by index, never by their start datetime. Two starts
        # an hour apart in real time can carry the same local time either side
        # of a fall-back, and an aware datetime compares and hashes on its
        # local fields alone when the zone matches, so such a pair would
        # collapse into one dict key and one of the two buckets would be lost.
        cached: dict[int, list[T]] = {}
        missing: list[tuple[int, int]] = []
        missing_start: int | None = None
        for index, (bucket_start, _bucket_end) in enumerate(buckets):
            value: object | None = None
            if not self._is_hot(bucket_start, now):
                value = self._store.get(self._bucket_key(query, config, bucket_start))
            if value is not None:
                if missing_start is not None:
                    missing.append((missing_start, index))
                    missing_start = None
                cached[index] = list(self._iter_bucket_entries(value, node_of, None))
            elif missing_start is None:
                missing_start = index
        if missing_start is not None:
            missing.append((missing_start, len(buckets)))

        # Buckets are uniform, contiguous and increasing on the absolute grid,
        # so dividing by the width locates each fetched entry's single bucket
        # instead of rescanning every entry once per bucket. The width is taken
        # from the first bucket so it cannot drift from what `_buckets` built.
        first_edge = datetime_to_nanoseconds(buckets[0][0])
        bucket_ns = datetime_to_nanoseconds(buckets[0][1]) - first_edge
        fetched: list[T] = []
        budget_exceeded = False
        for first, last in missing:
            try:
                values = fetch(query, buckets[first][0], buckets[last - 1][1])
            except LokiQueryBudgetExceeded as exc:
                fetched.extend(exc.entries)
                budget_exceeded = True
                break
            fetched.extend(values)
            grouped: dict[int, list[T]] = {}
            for item in values:
                index = (timestamp(item) - first_edge) // bucket_ns
                if first <= index < last:
                    grouped.setdefault(index, []).append(item)
            # A missing range always spans whole buckets, so every bucket from
            # `first` to `last` lies inside the range that was just fetched.
            for index in range(first, last):
                self._store_bucket(query, config, buckets[index][0], grouped.get(index, []), now=now, node_of=node_of)

        merged: list[T] = []
        for index in sorted(cached):
            merged.extend(cached[index])
        merged.extend(fetched)
        start_ns = datetime_to_nanoseconds(start)
        end_ns = datetime_to_nanoseconds(end)
        bounded: list[T] = []
        for item in merged:
            item_ns = timestamp(item)
            if item_ns < start_ns:
                continue
            if inclusive_end_ns is None and item_ns >= end_ns:
                continue
            if inclusive_end_ns is not None and item_ns > inclusive_end_ns:
                continue
            bounded.append(item)
        result = sorted(bounded, key=timestamp)
        if budget_exceeded:
            raise LokiQueryBudgetExceeded(result)
        return result

    def cached(
        self,
        *,
        query: str,
        config: Mapping[str, object],
        start: datetime,
        end: datetime,
        timestamp: TimestampExtractor[T],
        now: datetime,
        max_entries: int | None = None,
    ) -> list[T]:
        """Return the entries of every bucket already cached inside `[start, end)`.

        Nothing is fetched: buckets that were never stored, expired, or are still
        hot contribute no entries. This is how a polling deployment attributes
        a long workload from the windows earlier polls already fetched.

        `start` is first clamped forward to `now - ttl`: buckets older than the
        cache lifetime have expired, so probing them wastes one store lookup per
        bucket and returns nothing. `max_entries`, when set, caps how many
        entries are materialized: buckets are read newest first and reading stops
        once the cap is reached, so the newest entries are kept and the peak
        memory a single caller can pin is bounded to one bucket beyond the cap.
        """
        return self._read_window(
            query=query,
            config=config,
            start=start,
            end=end,
            nodes=None,
            timestamp=timestamp,
            now=now,
            max_entries=max_entries,
        )

    def read(
        self,
        *,
        query: str,
        config: Mapping[str, object],
        start: datetime,
        end: datetime,
        nodes: Iterable[str],
        timestamp: TimestampExtractor[T],
        now: datetime,
        node_of: NodeExtractor[T],
        max_entries: int | None = None,
    ) -> list[T]:
        """Return already-cached entries inside `[start, end)` for `nodes` only.

        Like `cached`, but node-scoped: buckets stored node-partitioned by
        `query(node_of=...)` yield only the requested nodes' lines, so the memory
        a batch pins scales with its nodes, not the cluster. Legacy flat buckets
        fall back to filtering each entry by `node_of`.
        """
        return self._read_window(
            query=query,
            config=config,
            start=start,
            end=end,
            nodes=set(nodes),
            timestamp=timestamp,
            now=now,
            max_entries=max_entries,
            node_of=node_of,
        )

    def _read_window(
        self,
        *,
        query: str,
        config: Mapping[str, object],
        start: datetime,
        end: datetime,
        nodes: set[str] | None,
        timestamp: TimestampExtractor[T],
        now: datetime,
        max_entries: int | None = None,
        node_of: NodeExtractor[T] | None = None,
    ) -> list[T]:
        """Read cached buckets newest first, node-scoped and capped."""
        if end <= start:
            return []
        ttl_floor = now - timedelta(seconds=self._ttl_seconds)
        if start < ttl_floor:
            start = ttl_floor
        if end <= start:
            return []
        if max_entries is not None and max_entries <= 0:
            return []
        start_ns = datetime_to_nanoseconds(start)
        end_ns = datetime_to_nanoseconds(end)
        merged: list[T] = []
        for bucket_start, _bucket_end in reversed(self._buckets(start, end)):
            if self._is_hot(bucket_start, now):
                continue
            value = self._store.get(self._bucket_key(query, config, bucket_start))
            if value is None:
                continue
            merged.extend(
                item
                for item in self._iter_bucket_entries(value, node_of, nodes)
                if start_ns <= timestamp(item) < end_ns
            )
            if max_entries is not None and len(merged) >= max_entries:
                break
        merged.sort(key=timestamp)
        if max_entries is not None and len(merged) > max_entries:
            merged = merged[-max_entries:]
        return merged

    def iter_partitions(
        self,
        *,
        query: str,
        config: Mapping[str, object],
        start: datetime,
        end: datetime,
        nodes: Iterable[str],
        timestamp: TimestampExtractor[T],
        now: datetime,
        node_of: NodeExtractor[T],
    ) -> Iterator[tuple[datetime, list[T]]]:
        """Yield `(bucket_start, entries)` one bucket at a time, node-scoped, newest first.

        One bucket is decoded whole (a partitioned bucket holds every node) and
        then scoped to the requested nodes, so the resident unit is a single
        bucket and a caller streams a window of any length with memory independent
        of how many buckets it spans; the bucket start lets it cut cleanly on a
        bucket boundary. `start` is clamped forward to `now - ttl`; buckets never
        stored, expired, or still hot yield nothing.
        """
        if end <= start:
            return
        ttl_floor = now - timedelta(seconds=self._ttl_seconds)
        if start < ttl_floor:
            start = ttl_floor
        if end <= start:
            return
        node_set = set(nodes)
        start_ns = datetime_to_nanoseconds(start)
        end_ns = datetime_to_nanoseconds(end)
        for bucket_start, _bucket_end in reversed(self._buckets(start, end)):
            if self._is_hot(bucket_start, now):
                continue
            value = self._store.get(self._bucket_key(query, config, bucket_start))
            if value is None:
                continue
            entries = [
                item
                for item in self._iter_bucket_entries(value, node_of, node_set)
                if start_ns <= timestamp(item) < end_ns
            ]
            if entries:
                yield bucket_start, entries

    @staticmethod
    def _iter_bucket_entries(value: object, node_of: NodeExtractor[T] | None, nodes: set[str] | None):
        """Yield a stored bucket's entries, optionally only those on `nodes`.

        A bucket is stored node-partitioned as ``{node: [entries]}`` when a node
        extractor was in use, or as a flat list otherwise (and by older writers).
        For the partitioned form only the requested nodes are materialized; for
        the flat form each entry is filtered by `node_of`.
        """
        if isinstance(value, dict):
            if nodes is None:
                for node_entries in value.values():
                    yield from node_entries
            else:
                for node in nodes:
                    yield from value.get(node, ())
        elif isinstance(value, list):
            if nodes is None or node_of is None:
                yield from value
            else:
                for entry in value:
                    if node_of(entry) in nodes:
                        yield entry

    def _store_bucket(
        self,
        query: str,
        config: Mapping[str, object],
        bucket_start: datetime,
        entries: list[object],
        *,
        now: datetime,
        node_of: NodeExtractor[T] | None = None,
    ) -> None:
        """Store one complete bucket unless it is still hot.

        With `node_of` the bucket is partitioned by node so a later
        `read(nodes=...)` materializes only those nodes' lines.
        """
        aligned = self._align(bucket_start)
        if self._is_hot(aligned, now):
            return
        if node_of is None:
            payload: object = list(entries)
        else:
            partitioned: dict[str, list[object]] = {}
            for entry in entries:
                partitioned.setdefault(node_of(entry), []).append(entry)
            payload = partitioned
        self._store.put(self._bucket_key(query, config, aligned), payload, ttl_seconds=self._ttl_seconds)

    def _align(self, value: datetime) -> datetime:
        seconds = int(value.timestamp())
        bucket_seconds = int(self._bucket.total_seconds())
        aligned = seconds - seconds % bucket_seconds
        return datetime.fromtimestamp(aligned, tz=value.tzinfo)

    def _buckets(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """Return the buckets covering `[start, end)`, oldest first.

        A bucket is identified by the absolute instant it starts at: `_align`
        floors Unix seconds and `_bucket_key` keys on them. So the grid is
        stepped in absolute seconds rather than by adding `self._bucket` to a
        local datetime, which advances the wall clock and disagrees with that
        identity across both DST transitions. A spring-forward would resolve
        the nonexistent local hour at the pre-transition offset, repeating
        instants the next real hour covers again and running the starts
        backwards in real time; a fall-back would skip the repeated hour, so
        its instants fall in no bucket at all. Absolute seconds have neither
        transition, so the starts stay aligned, contiguous and increasing.

        Args:
            start: Window start; the bucket containing it comes first.
            end: Exclusive window end.

        Returns:
            `(bucket_start, bucket_end)` pairs, each `self._bucket` of real
            time wide, contiguous and strictly increasing.
        """
        bucket_seconds = int(self._bucket.total_seconds())
        end_seconds = end.timestamp()
        result: list[tuple[datetime, datetime]] = []
        edge = int(self._align(start).timestamp())
        while edge < end_seconds:
            result.append(
                (
                    datetime.fromtimestamp(edge, tz=start.tzinfo),
                    datetime.fromtimestamp(edge + bucket_seconds, tz=start.tzinfo),
                )
            )
            edge += bucket_seconds
        return result

    def _is_hot(self, bucket_start: datetime, now: datetime) -> bool:
        """Return whether `bucket_start`'s bucket is too recent to cache.

        The bucket end and the buffer are applied in absolute seconds, like the
        grid itself. Adding `self._bucket` to `bucket_start` would advance the
        wall clock and put the end an hour from where it is either side of a
        fall-back: too late and a settled bucket is refetched forever, too
        early and a bucket still collecting entries is cached incomplete, which
        every later read then serves.

        Args:
            bucket_start: Start of the bucket being considered.
            now: Current time.

        Returns:
            Whether the bucket ends later than `now` less the hot buffer.
        """
        bucket_end = bucket_start.timestamp() + self._bucket.total_seconds()
        return bucket_end >= now.timestamp() - self._hot_buffer.total_seconds()

    @staticmethod
    def _bucket_key(query: str, config: Mapping[str, object], bucket_start: datetime) -> str:
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{query}\x1e{config_json}".encode()).hexdigest()
        return f"{digest}:{int(bucket_start.timestamp())}"


__all__ = ["FileRangeCacheStore", "LokiRangeCache"]
