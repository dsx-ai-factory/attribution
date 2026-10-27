# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint storage backends for extraction caching.

Defines the CheckpointBackend protocol and the in-memory implementation.
I/O-based backends can be provided by integrating applications.
"""

from typing import Optional, Protocol

import duckdb
from loguru import logger

from attribution_lib.cache_metadata import (
    _DEFAULT_MAX_AGE_HOURS,
    ExtractionCacheMetadata,
)
from attribution_lib.events import EventManager


class CheckpointBackend(Protocol):
    """Storage backend for extraction cache checkpoints.

    Implementations handle the physical storage of EventManager state
    and metadata.
    """

    def save(self, cache_key: str, event_manager: EventManager, metadata: ExtractionCacheMetadata) -> None:
        """Persist an EventManager and its metadata under the given key."""
        ...

    def load_metadata(self, cache_key: str) -> Optional[ExtractionCacheMetadata]:
        """Load metadata for the given key, or None if not found."""
        ...

    def load_event_manager(self, cache_key: str) -> EventManager:
        """Load the EventManager for the given key.

        Only called after load_metadata() confirmed the cache exists and
        validation passed, so the key is guaranteed to be present.
        """
        ...

    def cleanup(self, max_age_hours: float = _DEFAULT_MAX_AGE_HOURS) -> int:
        """Remove stale entries older than max_age_hours. Returns count removed."""
        ...


class NullCheckpointBackend:
    """No-op backend -- discards saves and reports no cached state.

    Allows attribute() to be the single public entry point without
    requiring callers to configure a storage backend.
    """

    def save(self, cache_key: str, event_manager: EventManager, metadata: ExtractionCacheMetadata) -> None:
        pass

    def load_metadata(self, cache_key: str) -> Optional[ExtractionCacheMetadata]:
        return None

    def load_event_manager(self, cache_key: str) -> EventManager:
        raise RuntimeError("NullCheckpointBackend stores no state")

    def cleanup(self, max_age_hours: float = _DEFAULT_MAX_AGE_HOURS) -> int:
        return 0


class MemoryCheckpointBackend:
    """Stores extraction caches using DuckDB shadow tables within the EventManager.

    On save, EventManager.checkpoint() copies the events table to a shadow
    table in the same DuckDB connection via CREATE TABLE AS SELECT. On load,
    EventManager.restore() copies it back via INSERT INTO ... SELECT FROM.

    This avoids Python serialization overhead entirely — all data stays in
    DuckDB's native columnar format and the copy is a single vectorized
    operation. Lookup tables (event_types, rules, obs_sources, obs_formats)
    and observation_contexts don't need checkpointing since they are stable
    after initialization.

    A max_entries bound prevents unbounded growth in long-running processes.
    When exceeded, the oldest entries are evicted on save. Cleanup is a
    no-op since eviction is handled inline.
    """

    _DEFAULT_MAX_ENTRIES = 256

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._store: dict[str, tuple[EventManager, ExtractionCacheMetadata]] = {}
        self._max_entries = max_entries

    def save(self, cache_key: str, event_manager: EventManager, metadata: ExtractionCacheMetadata) -> None:
        event_manager.checkpoint(cache_key)

        old = self._store.pop(cache_key, None)
        if old is not None and old[0] is not event_manager:
            try:
                old[0].drop_checkpoint(cache_key)
            except duckdb.Error as e:
                logger.debug("Could not drop old checkpoint (key={}): {}", cache_key, e)
        self._store[cache_key] = (event_manager, metadata)

        while len(self._store) > self._max_entries:
            evicted_key = next(iter(self._store))
            evicted_em, _ = self._store.pop(evicted_key)
            try:
                evicted_em.drop_checkpoint(evicted_key)
            except duckdb.Error as e:
                logger.debug("Could not drop evicted checkpoint (key={}): {}", evicted_key, e)
            logger.debug("Evicted in-memory cache entry (key={})", evicted_key)
        logger.info("Extraction cache saved in memory (key={})", cache_key)

    def load_metadata(self, cache_key: str) -> Optional[ExtractionCacheMetadata]:
        entry = self._store.get(cache_key)
        if entry is None:
            return None
        return entry[1]

    def load_event_manager(self, cache_key: str) -> EventManager:
        event_manager, _ = self._store[cache_key]
        event_manager.restore(cache_key)
        return event_manager

    def cleanup(self, max_age_hours: float = _DEFAULT_MAX_AGE_HOURS) -> int:
        return 0
