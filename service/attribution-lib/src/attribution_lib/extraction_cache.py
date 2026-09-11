# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Extraction cache fingerprinting, diffing, and orchestration.

Provides data fingerprinting (spread-sampled hashing), diffing logic
to classify changes (unchanged / appended / modified / new / removed),
and backend-agnostic orchestration functions for loading and saving
extraction caches.

The fingerprint heuristic assumes append-only telemetry sources
(e.g. Loki logs that grow monotonically for a given job). Middle-content
changes on otherwise identical count/boundary data are unlikely but
not guaranteed to be detected; a spread-sampled hash minimizes this
risk while keeping fingerprint cost O(1).

Cache orchestration (when to restore/snapshot) lives in
Attributor.attribute() and is owned by the attribution library.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from loguru import logger

import attribution_lib
from attribution_lib.cache_metadata import (
    DataFingerprint,
    ExtractionCacheMetadata,
)
from attribution_lib.checkpoint_backends import CheckpointBackend
from attribution_lib.events import EventManager
from attribution_lib.exceptions import SnapshotSchemaVersionError
from attribution_lib.observation.inbound import (
    Context as ObservationContext,
    Format as ObservationFormat,
)
from attribution_lib.observation.processing import BodyFallbacks
from attribution_lib.resources.resource_set import serialize_resource_set

# Re-export types so existing callers that import from here keep working
__all__ = [
    "CheckpointBackend",
    "ExtractionCacheMetadata",
    "DataFingerprint",
    "DataDiffResult",
    "DataDiff",
    "CacheLoadResult",
    "compute_data_fingerprint",
    "diff_data",
    "save_extraction_cache",
    "load_extraction_cache",
    "load_or_update_extraction_cache",
]


# Data fingerprinting
_NUM_SAMPLE_POINTS = 8


def _hash_entry(entry: Any) -> str:
    """Produce a stable hash for a single observation entry.

    Handles strings directly; dicts and other types are serialized via
    json.dumps with sort_keys for determinism.
    """
    if isinstance(entry, str):
        raw = entry
    else:
        raw = json.dumps(entry, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _sample_indices(count: int, num_samples: int = _NUM_SAMPLE_POINTS) -> list[int]:
    """Return evenly spaced indices across a list of given count.

    Always includes the first and last index. For lists smaller than
    num_samples, returns all indices.
    """
    if count <= num_samples:
        return list(range(count))
    step = (count - 1) / (num_samples - 1)
    return [round(i * step) for i in range(num_samples)]


def compute_data_fingerprint(
    entries: list[Any],
    extraction_context: Optional[ObservationContext] = None,
    body_fallbacks: Optional[BodyFallbacks] = None,
) -> Optional[DataFingerprint]:
    """Compute a fingerprint for observation entries and extraction context.

    Args:
        entries: Observation entries to sample.
        extraction_context: Context that controls parsing, filtering, and
            attribution provenance. Omit for a data-only fingerprint.
        body_fallbacks: What `entry_timeseries` bodies may be parsed for; its
            `year_fallback` is hashed when it decides how year-less timestamps resolve.

    Returns:
        A context-bound fingerprint, or ``None`` when entries is empty.
    """
    if not entries:
        return None
    indices = _sample_indices(len(entries))
    sample_hashes = tuple(_hash_entry(entries[i]) for i in indices)
    if extraction_context is not None:
        serialized_resources = serialize_resource_set(extraction_context.resources)
        resources: Any
        if isinstance(serialized_resources, str):
            resources = serialized_resources
        else:
            resources = [resource.model_dump(mode="json") for resource in serialized_resources]
        time_start = extraction_context.time_interval.start
        time_end = extraction_context.time_interval.end
        context_fields: dict[str, Any] = {
            "format": extraction_context.format.value,
            "pre_filtered": extraction_context.pre_filtered,
            "resources": resources,
            "source": extraction_context.source.value,
            "time_end": time_end.isoformat() if time_end is not None else None,
            "time_start": time_start.isoformat() if time_start is not None else None,
            "timezone": extraction_context.timezone.key,
        }
        # Only `entry_timeseries` reads the year off `year_fallback`; every other format reads it off
        # `time_end`, hashed above. Keyed only here so their fingerprints stay byte-identical.
        if (
            extraction_context.format == ObservationFormat.ENTRY_TIMESERIES
            and body_fallbacks is not None
            and body_fallbacks.parse_timestamp
        ):
            year_fallback = body_fallbacks.year_fallback
            context_fields["year_fallback"] = year_fallback.isoformat() if year_fallback is not None else None
        context_hash = _hash_entry(context_fields)
        sample_hashes = tuple(
            _hash_entry({"context_hash": context_hash, "entry_hash": sample_hash}) for sample_hash in sample_hashes
        )
    return DataFingerprint(count=len(entries), sample_hashes=sample_hashes)


# Data diffing
class DataDiffResult(Enum):
    """Classification of how current data compares to cached data."""

    UNCHANGED = "unchanged"
    APPENDED = "appended"
    MODIFIED = "modified"
    NEW_SOURCE = "new_source"
    REMOVED = "removed"


@dataclass(frozen=True)
class DataDiff:
    """Result of comparing one observation between cached and current fingerprints."""

    source: str
    result: DataDiffResult
    cached_count: int


def diff_data(
    cached_fingerprints: dict[str, DataFingerprint],
    current_fingerprints: dict[str, DataFingerprint],
    skipped_sources: Optional[set[str]] = None,
) -> dict[str, DataDiff]:
    """Compare cached and current data fingerprints, returning a diff per observation.

    Classification logic:
    - UNCHANGED: count and all sample hashes match, or source was skipped
    - APPENDED: count increased, first_hash matches (new entries added at end)
    - NEW_SOURCE: exists in current but not in cached
    - REMOVED: exists in cached but not in current (and not skipped)
    - MODIFIED: catch-all for any other data change

    Sources in skipped_sources are treated as UNCHANGED even if absent from
    current_fingerprints, allowing callers to intentionally skip collecting
    expensive telemetry without invalidating the cache.
    """
    diffs: dict[str, DataDiff] = {}
    skipped = skipped_sources or set()

    for source in sorted(cached_fingerprints.keys() | current_fingerprints.keys()):
        cached_fingerprint = cached_fingerprints.get(source)
        current_fingerprint = current_fingerprints.get(source)

        if cached_fingerprint is None:
            diff_result = DataDiffResult.NEW_SOURCE
            count = 0
        elif current_fingerprint is None:
            if source in skipped:
                diff_result = DataDiffResult.UNCHANGED
            else:
                diff_result = DataDiffResult.REMOVED
            count = cached_fingerprint.count
        elif cached_fingerprint == current_fingerprint:
            diff_result = DataDiffResult.UNCHANGED
            count = cached_fingerprint.count
        elif (
            current_fingerprint.count > cached_fingerprint.count
            and current_fingerprint.first_hash == cached_fingerprint.first_hash
        ):
            diff_result = DataDiffResult.APPENDED
            count = cached_fingerprint.count
        else:
            diff_result = DataDiffResult.MODIFIED
            count = cached_fingerprint.count

        diffs[source] = DataDiff(source=source, result=diff_result, cached_count=count)

    return diffs


# Cache load result
@dataclass
class CacheLoadResult:
    """Result of load_or_update_extraction_cache.

    Attributes:
        event_manager: Restored EventManager if cache is valid, None if full re-extraction needed.
        metadata: Cached metadata if cache was loaded, None otherwise.
        data_diffs: Per-source diff results. Empty dict if no cache found or extractor hash mismatch.
        needs_full_reextraction: True if cache is missing, rule hash changed, or any source was MODIFIED.
    """

    event_manager: Optional[EventManager]
    metadata: Optional[ExtractionCacheMetadata]
    data_diffs: dict[str, DataDiff]
    needs_full_reextraction: bool


# Orchestration functions (backend-agnostic)
def save_extraction_cache(
    backend: CheckpointBackend,
    cache_key: str,
    event_manager: EventManager,
    job_id: str,
    data_fingerprints: Optional[dict[str, DataFingerprint]] = None,
    raw_keywords: Optional[set[str]] = None,
) -> None:
    """Build metadata from current rule hashes and delegate to the backend.

    Saves extraction state (events + rule hashes + fingerprints + raw keywords).
    Post-attribution metadata (outcome, attempt count, timestamp) should
    be written separately via backend.update_metadata() after attribution.
    """
    rules_engine = attribution_lib.get_rules_engine()

    metadata = ExtractionCacheMetadata(
        extractor_hash=rules_engine.extractor_hash(),
        attribution_hash=rules_engine.attribution_hash(),
        job_id=job_id,
        data_fingerprints=data_fingerprints or {},
        raw_keywords=tuple(sorted(raw_keywords if raw_keywords is not None else ())),
    )
    backend.save(cache_key, event_manager, metadata)


def load_extraction_cache(
    backend: CheckpointBackend,
    cache_key: str,
) -> Optional[tuple[EventManager, ExtractionCacheMetadata]]:
    """Load a cached extraction if it exists and extractor rules haven't changed.

    Returns (EventManager, ExtractionCacheMetadata) if valid, None otherwise.

    This is the simple validation path: it checks only extractor_hash (rule
    changes), not source fingerprints (data changes). Use this when callers
    don't compute fingerprints or don't need incremental extraction support.
    For data-aware validation, use load_or_update_extraction_cache instead.
    """
    metadata = backend.load_metadata(cache_key)
    if metadata is None:
        return None

    current_extractor_hash = attribution_lib.get_rules_engine().extractor_hash()
    if metadata.extractor_hash != current_extractor_hash:
        logger.info(
            "Extraction cache hash mismatch (stored={}, current={}); re-extraction needed",
            metadata.extractor_hash[:12],
            current_extractor_hash[:12],
        )
        return None
    if metadata.raw_keywords is None:
        logger.info("Extraction cache lacks raw keywords; re-extraction needed")
        return None

    try:
        event_manager = backend.load_event_manager(cache_key)
    except SnapshotSchemaVersionError as error:
        logger.info("Extraction cache incompatible (key={}): {}; full re-extraction needed", cache_key, error)
        return None
    logger.info("Loaded extraction cache (key={}, extractor hash: {})", cache_key, metadata.extractor_hash[:12])
    return event_manager, metadata


def load_or_update_extraction_cache(
    backend: CheckpointBackend,
    cache_key: str,
    current_fingerprints: dict[str, DataFingerprint],
    skipped_sources: Optional[set[str]] = None,
) -> CacheLoadResult:
    """Load a cached extraction with source fingerprint validation.

    Validates both extractor hash (rule changes) and source fingerprints
    (data changes). Returns a CacheLoadResult that tells the caller whether
    to do full re-extraction, incremental extraction, or skip extraction.

    Decision logic:
    - No cache / extractor hash mismatch → needs_full_reextraction=True
    - Any source MODIFIED or REMOVED → needs_full_reextraction=True
      (cache invalidated; REMOVED means stale events would participate)
    - All sources UNCHANGED → cache hit (skip extraction entirely)
    - Some sources APPENDED or NEW_SOURCE → cache loaded, caller runs
      incremental extraction on those sources only
    """
    metadata = backend.load_metadata(cache_key)
    if metadata is None:
        return CacheLoadResult(event_manager=None, metadata=None, data_diffs={}, needs_full_reextraction=True)

    current_extractor_hash = attribution_lib.get_rules_engine().extractor_hash()
    if metadata.extractor_hash != current_extractor_hash:
        logger.info(
            "Extraction cache hash mismatch (stored={}, current={}); re-extraction needed",
            metadata.extractor_hash[:12],
            current_extractor_hash[:12],
        )
        return CacheLoadResult(event_manager=None, metadata=metadata, data_diffs={}, needs_full_reextraction=True)
    if metadata.raw_keywords is None:
        logger.info("Extraction cache lacks raw keywords; full re-extraction needed")
        return CacheLoadResult(event_manager=None, metadata=metadata, data_diffs={}, needs_full_reextraction=True)

    data_diffs = diff_data(metadata.data_fingerprints, current_fingerprints, skipped_sources=skipped_sources)

    _INVALIDATING = {DataDiffResult.MODIFIED, DataDiffResult.REMOVED}
    has_invalidated = any(data_diff.result in _INVALIDATING for data_diff in data_diffs.values())
    if has_invalidated:
        invalidated_names = [data_diff.source for data_diff in data_diffs.values() if data_diff.result in _INVALIDATING]
        logger.info("Source data modified/removed for {}, full re-extraction needed", invalidated_names)
        return CacheLoadResult(
            event_manager=None, metadata=metadata, data_diffs=data_diffs, needs_full_reextraction=True
        )

    try:
        event_manager = backend.load_event_manager(cache_key)
    except SnapshotSchemaVersionError as error:
        logger.info("Extraction cache incompatible (key={}): {}; full re-extraction needed", cache_key, error)
        return CacheLoadResult(
            event_manager=None, metadata=metadata, data_diffs=data_diffs, needs_full_reextraction=True
        )
    all_unchanged = all(data_diff.result == DataDiffResult.UNCHANGED for data_diff in data_diffs.values())
    if all_unchanged and data_diffs:
        logger.info("Extraction cache hit (all sources unchanged, extractor hash: {})", metadata.extractor_hash[:12])
    else:
        changed = [
            f"{data_diff.source}={data_diff.result.value}"
            for data_diff in data_diffs.values()
            if data_diff.result != DataDiffResult.UNCHANGED
        ]
        logger.info("Extraction cache loaded with incremental updates needed: {}", changed)

    return CacheLoadResult(
        event_manager=event_manager, metadata=metadata, data_diffs=data_diffs, needs_full_reextraction=False
    )
