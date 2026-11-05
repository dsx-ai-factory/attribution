# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared primitives for reconstructing Kubernetes execution attempts."""

from __future__ import annotations

from datetime import datetime, timezone

RESTART_LABEL = "jobset.sigs.k8s.io/restart-attempt"


def restart_starts(observations: list[dict]) -> dict[str, str]:
    starts = {}
    for observation in observations:
        if observation.get("kind") not in {"Job", "Pod"}:
            continue
        restart = restart_attempt(observation)
        if restart is None:
            continue
        status = observation.get("status") or {}
        timestamp = (
            status.get("startTime")
            or observation.get("sourceTimestamp")
            or observation.get("observedAt")
        )
        if timestamp and (
            restart not in starts
            or parse_time(timestamp) < parse_time(starts[restart])
        ):
            starts[restart] = timestamp
    return starts


def restart_attempt(observation: dict) -> str | None:
    labels = (observation.get("selectedMetadata") or {}).get("labels") or {}
    value = labels.get(RESTART_LABEL)
    return str(value) if value is not None else None


def observations_by_uid(observations) -> dict[str, list[dict]]:
    grouped = {}
    for observation in observations:
        uid = (observation.get("metadata") or {}).get("uid")
        if uid:
            grouped.setdefault(uid, []).append(observation)
    for timeline in grouped.values():
        timeline.sort(key=lambda item: parse_time(item["observedAt"]))
    return grouped


def latest_pod_observations(observations: list[dict]) -> dict[str, dict]:
    return {
        uid: timeline[-1]
        for uid, timeline in observations_by_uid(
            item
            for item in observations
            if item.get("kind") == "Pod"
        ).items()
    }


def controller_owner(metadata: dict) -> dict | None:
    controllers = [
        owner
        for owner in metadata.get("ownerReferences") or []
        if owner.get("controller") is True
        and owner.get("uid")
        and owner.get("kind")
    ]
    return controllers[0] if len(controllers) == 1 else None


def within(
    timestamp: str,
    start: str,
    end: str | None,
    *,
    include_end: bool = False,
) -> bool:
    value = parse_time(timestamp)
    if end is None:
        return value >= parse_time(start)
    return value >= parse_time(start) and (
        value <= parse_time(end)
        if include_end
        else value < parse_time(end)
    )


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


def resource_from_observation(observation: dict) -> dict:
    """Restore the Kubernetes fields selected into a retained observation."""
    metadata = dict(observation.get("metadata") or {})
    selected = observation.get("selectedMetadata") or {}
    for field in ("labels", "annotations", "deletionTimestamp"):
        if selected.get(field):
            metadata[field] = selected[field]
    return {
        "apiVersion": observation.get("apiVersion") or "",
        "kind": observation.get("kind") or "",
        "metadata": metadata,
        "spec": observation.get("spec") or {},
        "status": observation.get("status") or {},
    }


def retained_history_covers_window(
    coverage: dict,
    required_sources: set[str],
    start: datetime,
    end: datetime,
) -> bool:
    """Return whether retained watches continuously cover a time window."""
    disabled = coverage.get("disabledWatches") or {}
    sources = {
        item.get("sourceName"): item
        for item in coverage.get("sources") or []
    }
    for source_name in required_sources:
        if source_name in disabled:
            return False
        source = sources.get(source_name)
        if source is None:
            return False
        try:
            coverage_start = parse_time(source["coverageStartedAt"])
            checkpoint = parse_time(source["checkpointedAt"])
        except (KeyError, TypeError, ValueError):
            return False
        if coverage_start > start or checkpoint < end:
            return False

    for gap in coverage.get("gaps") or []:
        if gap.get("sourceName") not in required_sources:
            continue
        try:
            gap_start = parse_time(gap["startedAt"])
            gap_end = (
                parse_time(gap["recoveredAt"])
                if gap.get("recoveredAt")
                else None
            )
        except (KeyError, TypeError, ValueError):
            return False
        if gap_start <= end and (
            gap_end is None or gap_end >= start
        ):
            return False
    return True
