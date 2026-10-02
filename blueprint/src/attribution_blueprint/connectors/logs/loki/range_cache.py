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
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypeVar

from attribution_blueprint.connectors.logs.loki.client import LokiQueryBudgetExceeded, datetime_to_nanoseconds

T = TypeVar("T")

RangeFetcher = Callable[[str, datetime, datetime], list[T]]
TimestampExtractor = Callable[[T], int]

_KEY_RE = re.compile(r"^[0-9a-f]{64}:\d+$")


class FileRangeCacheStore:
    """Dependency-free store: one JSON file per bucket below `root`.

    Files are written atomically and expire by the timestamp stored inside
    them, so a store can be shared by sequential runs. The directory and files
    are created with owner-only permissions because cached log lines may carry
    tenant data.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if _KEY_RE.match(key) is None:
            raise ValueError(f"unsupported range cache key: {key!r}")
        return self.root / key[:2] / f"{key.replace(':', '_')}.json"

    def get(self, key: str) -> list | None:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("value"), list):
            path.unlink(missing_ok=True)
            return None
        expires_at = payload.get("expires_at")
        if isinstance(expires_at, (int, float)) and expires_at <= time.time():
            path.unlink(missing_ok=True)
            return None
        return payload["value"]

    def put(self, key: str, value: list, *, ttl_seconds: float) -> None:
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
    ) -> list[T]:
        """Return one requested window, fetching only the missing ranges.

        `start` is inclusive and `end` exclusive. `inclusive_end_ns` restores an
        inclusive upper bound on the returned entries when the caller's request
        was inclusive. `fetch(query, range_start, range_end)` must return every
        entry inside the half-open missing range. A `LokiQueryBudgetExceeded`
        raised by `fetch` stops fetching; the partial result is re-raised the
        same way so callers can keep what was retrieved.
        """
        if end <= start:
            return []
        buckets = self._buckets(start, end)
        cached: dict[datetime, list[T]] = {}
        missing: list[tuple[datetime, datetime]] = []
        missing_start: datetime | None = None
        for bucket_start, bucket_end in buckets:
            value: object | None = None
            if not self._is_hot(bucket_start, now):
                value = self._store.get(self._bucket_key(query, config, bucket_start))
            if isinstance(value, list):
                if missing_start is not None:
                    missing.append((missing_start, bucket_start))
                    missing_start = None
                cached[bucket_start] = value
            elif missing_start is None:
                missing_start = bucket_start
            if bucket_end == buckets[-1][1] and missing_start is not None:
                missing.append((missing_start, bucket_end))
                missing_start = None

        fetched: list[T] = []
        budget_exceeded = False
        for range_start, range_end in missing:
            try:
                values = fetch(query, range_start, range_end)
            except LokiQueryBudgetExceeded as exc:
                fetched.extend(exc.entries)
                budget_exceeded = True
                break
            fetched.extend(values)
            for bucket_start, bucket_end in buckets:
                if bucket_start < range_start or bucket_end > range_end:
                    continue
                bucket_values = [
                    item
                    for item in values
                    if datetime_to_nanoseconds(bucket_start) <= timestamp(item) < datetime_to_nanoseconds(bucket_end)
                ]
                self._store_bucket(query, config, bucket_start, list(bucket_values), now=now)

        merged: list[T] = []
        for bucket_start in sorted(cached):
            merged.extend(cached[bucket_start])
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

    def _store_bucket(
        self,
        query: str,
        config: Mapping[str, object],
        bucket_start: datetime,
        entries: list[object],
        *,
        now: datetime,
    ) -> None:
        """Store one complete bucket unless it is still hot."""
        aligned = self._align(bucket_start)
        if self._is_hot(aligned, now):
            return
        self._store.put(self._bucket_key(query, config, aligned), list(entries), ttl_seconds=self._ttl_seconds)

    def _align(self, value: datetime) -> datetime:
        seconds = int(value.timestamp())
        bucket_seconds = int(self._bucket.total_seconds())
        aligned = seconds - seconds % bucket_seconds
        return datetime.fromtimestamp(aligned, tz=value.tzinfo)

    def _buckets(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        result: list[tuple[datetime, datetime]] = []
        current = self._align(start)
        while current < end:
            result.append((current, current + self._bucket))
            current += self._bucket
        return result

    def _is_hot(self, bucket_start: datetime, now: datetime) -> bool:
        return bucket_start + self._bucket >= now - self._hot_buffer

    @staticmethod
    def _bucket_key(query: str, config: Mapping[str, object], bucket_start: datetime) -> str:
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{query}\x1e{config_json}".encode()).hexdigest()
        return f"{digest}:{int(bucket_start.timestamp())}"


__all__ = ["FileRangeCacheStore", "LokiRangeCache"]
