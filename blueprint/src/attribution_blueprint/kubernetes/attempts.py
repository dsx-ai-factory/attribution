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
