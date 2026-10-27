# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data types for extraction cache metadata and data fingerprinting."""

from dataclasses import dataclass, field
from typing import Any, Optional

_METADATA_FILENAME = "extraction_cache_metadata.json"
_DEFAULT_MAX_AGE_HOURS = 72


@dataclass(frozen=True)
class DataFingerprint:
    """Lightweight fingerprint of an observation source's data.

    Uses spread-sampled hashing: entry count plus hashes of entries at
    evenly spaced indices. This detects middle-content changes that
    boundary-only hashing would miss, while staying O(1) regardless of
    data size.
    """

    count: int
    sample_hashes: tuple[str, ...]

    @property
    def first_hash(self) -> str:
        return self.sample_hashes[0]

    def to_dict(self) -> dict:
        return {"count": self.count, "sample_hashes": list(self.sample_hashes)}

    @classmethod
    def from_dict(cls, data: dict) -> "DataFingerprint":
        if "sample_hashes" in data:
            return cls(count=data["count"], sample_hashes=tuple(data["sample_hashes"]))
        return cls(count=data["count"], sample_hashes=(data["first_hash"], data["last_hash"]))


@dataclass(frozen=True)
class ExtractionCacheMetadata:
    """Records rule versions, job identity, fingerprints, and raw keywords
    associated with a cached extraction."""

    extractor_hash: str
    attribution_hash: str
    job_id: str
    data_fingerprints: dict[str, DataFingerprint] = field(default_factory=dict)
    raw_keywords: Optional[tuple[str, ...]] = ()
    previous_outcome: Optional[str] = None
    attribution_attempt_count: int = 0
    last_attribution_timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        data: dict[str, Any] = {
            "extractor_hash": self.extractor_hash,
            "attribution_hash": self.attribution_hash,
            "job_id": self.job_id,
        }
        if self.data_fingerprints:
            data["data_fingerprints"] = {k: v.to_dict() for k, v in self.data_fingerprints.items()}
        if self.raw_keywords is not None:
            data["raw_keywords"] = list(self.raw_keywords)
        if self.previous_outcome is not None:
            data["previous_outcome"] = self.previous_outcome
        if self.attribution_attempt_count > 0:
            data["attribution_attempt_count"] = self.attribution_attempt_count
        if self.last_attribution_timestamp is not None:
            data["last_attribution_timestamp"] = self.last_attribution_timestamp
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionCacheMetadata":
        raw_fingerprints = data.get("data_fingerprints", {})
        fingerprints = {k: DataFingerprint.from_dict(v) for k, v in raw_fingerprints.items()}
        return cls(
            extractor_hash=data["extractor_hash"],
            attribution_hash=data["attribution_hash"],
            job_id=data["job_id"],
            data_fingerprints=fingerprints,
            raw_keywords=tuple(data["raw_keywords"]) if "raw_keywords" in data else None,
            previous_outcome=data.get("previous_outcome"),
            attribution_attempt_count=data.get("attribution_attempt_count", 0),
            last_attribution_timestamp=data.get("last_attribution_timestamp"),
        )
