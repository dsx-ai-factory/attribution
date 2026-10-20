# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kueue attempt boundaries reconstructed from retained workload history."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

_RESTART_LABEL = "jobset.sigs.k8s.io/restart-attempt"


@dataclass(frozen=True)
class WorkloadAttempt:
    workload_uid: str
    namespace: str
    workload_name: str
    owner_kind: str | None
    admission_key: str
    restart_attempt: str
    start_time: datetime
    end_time: datetime
    terminal: dict | None
    open: bool
    pod_uids: frozenset[str]

    @property
    def attempt_id(self) -> str:
        canonical = json.dumps(
            {
                "admission": self.admission_key,
                "restartAttempt": self.restart_attempt,
                "workloadUid": self.workload_uid,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return f"kueue:v1:{self.workload_uid}:{digest}"


def attempts_from_workload_history(
    history: dict,
    *,
    workload_name: str,
    now: datetime,
) -> list[WorkloadAttempt]:
    """Split retained Workload/JobSet history into attributable executions."""
    observations = history.get("observations") or []
    workloads = _by_uid(
        item
        for item in observations
        if item.get("kind") == "Workload"
        and (item.get("metadata") or {}).get("name") == workload_name
    )
    restart_starts = _restart_starts(observations)
    terminals = _terminal_events(observations)
    pod_events = _pod_events(observations)
    attempts = []
    for workload_uid, timeline in workloads.items():
        latest_metadata = timeline[-1].get("metadata") or {}
        admissions = _admissions(timeline)
        if not admissions:
            continue
        starts = _starts(admissions, restart_starts)
        owner = _controller_owner(latest_metadata)
        for index, start in enumerate(starts):
            next_start = (
                starts[index + 1]["time"]
                if index + 1 < len(starts)
                else None
            )
            terminal = _terminal_for(
                start,
                next_start,
                terminals,
                workload_uid=workload_uid,
                owner=owner,
            )
            end_time = (
                _parse_time(terminal["time"])
                if terminal is not None
                else _parse_time(next_start)
                if next_start is not None
                else now
            )
            attempts.append(
                WorkloadAttempt(
                    workload_uid=workload_uid,
                    namespace=latest_metadata.get("namespace") or "",
                    workload_name=workload_name,
                    owner_kind=owner.get("kind") if owner else None,
                    admission_key=start["admission_key"],
                    restart_attempt=start["restart_attempt"],
                    start_time=_parse_time(start["time"]),
                    end_time=end_time,
                    terminal=terminal,
                    open=terminal is None and next_start is None,
                    pod_uids=frozenset(
                        uid
                        for uid, event in pod_events.items()
                        if _pod_in_attempt(
                            event,
                            start["restart_attempt"],
                            start["time"],
                            next_start,
                        )
                    ),
                )
            )
    return sorted(
        attempts,
        key=lambda attempt: (
            attempt.start_time,
            attempt.attempt_id,
        ),
    )


def _admissions(timeline: list[dict]) -> list[dict]:
    boundaries = {}
    for observation in timeline:
        for condition in (observation.get("status") or {}).get("conditions") or []:
            transition = condition.get("lastTransitionTime")
            if (
                condition.get("type") == "Admitted"
                and condition.get("status") == "True"
                and transition
            ):
                key = str(condition.get("transitionID") or transition)
                boundaries[key] = {
                    "key": key,
                    "time": transition,
                }
    return sorted(boundaries.values(), key=lambda item: _parse_time(item["time"]))


def _restart_starts(observations: list[dict]) -> dict[str, str]:
    starts = {}
    for observation in observations:
        if observation.get("kind") not in {"Job", "Pod"}:
            continue
        restart = _restart_attempt(observation)
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
            or _parse_time(timestamp) < _parse_time(starts[restart])
        ):
            starts[restart] = timestamp
    return starts


def _starts(admissions: list[dict], restarts: dict[str, str]) -> list[dict]:
    starts = []
    for index, admission in enumerate(admissions):
        next_admission = (
            admissions[index + 1]["time"]
            if index + 1 < len(admissions)
            else None
        )
        scoped_restarts = [
            (restart, timestamp)
            for restart, timestamp in restarts.items()
            if _within(timestamp, admission["time"], next_admission)
        ] or [("0", admission["time"])]
        for restart, timestamp in sorted(
            scoped_restarts,
            key=lambda item: (_parse_time(item[1]), item[0]),
        ):
            starts.append(
                {
                    "admission_key": admission["key"],
                    "restart_attempt": restart,
                    "time": max(
                        admission["time"],
                        timestamp,
                        key=_parse_time,
                    ),
                }
            )
    return starts


def _terminal_events(observations: list[dict]) -> list[dict]:
    events = []
    for observation in observations:
        kind = observation.get("kind")
        if kind not in {"Workload", "Job"}:
            continue
        for condition in (observation.get("status") or {}).get("conditions") or []:
            condition_type = condition.get("type")
            transition = condition.get("lastTransitionTime")
            supported = (
                kind == "Workload"
                and condition_type in {"Evicted", "Finished"}
                or kind == "Job"
                and condition_type in {"Failed", "Complete"}
            )
            if condition.get("status") != "True" or not transition or not supported:
                continue
            events.append(
                {
                    "kind": kind,
                    "uid": (observation.get("metadata") or {}).get("uid"),
                    "type": condition_type,
                    "reason": condition.get("reason"),
                    "time": transition,
                    "restart_attempt": (
                        _restart_attempt(observation)
                        if kind == "Job"
                        else None
                    ),
                }
            )
    return sorted(events, key=lambda item: _parse_time(item["time"]))


def _terminal_for(
    start: dict,
    next_start: str | None,
    terminals: list[dict],
    *,
    workload_uid: str,
    owner: dict | None,
) -> dict | None:
    return next(
        (
            terminal
            for terminal in terminals
            if (
                terminal["kind"] == "Workload"
                and terminal["uid"] == workload_uid
                or owner is not None
                and owner.get("kind") == "Job"
                and terminal["kind"] == "Job"
                and terminal["uid"] == owner.get("uid")
            )
            and _terminal_within(
                terminal["time"],
                start["time"],
                next_start,
            )
            and terminal["restart_attempt"] in {
                None,
                start["restart_attempt"],
            }
        ),
        None,
    )


def _terminal_within(
    timestamp: str,
    start: str,
    end: str | None,
) -> bool:
    value = _parse_time(timestamp)
    return value >= _parse_time(start) and (
        end is None or value <= _parse_time(end)
    )


def _pod_events(observations: list[dict]) -> dict[str, dict]:
    return {
        uid: timeline[-1]
        for uid, timeline in _by_uid(
            item
            for item in observations
            if item.get("kind") == "Pod"
        ).items()
    }


def _pod_in_attempt(
    observation: dict,
    restart_attempt: str,
    start: str,
    end: str | None,
) -> bool:
    observed_restart = _restart_attempt(observation)
    if observed_restart is not None and observed_restart != restart_attempt:
        return False
    status = observation.get("status") or {}
    timestamp = status.get("startTime") or observation.get("observedAt")
    return bool(timestamp and _within(timestamp, start, end))


def _restart_attempt(observation: dict) -> str | None:
    labels = (observation.get("selectedMetadata") or {}).get("labels") or {}
    value = labels.get(_RESTART_LABEL)
    return str(value) if value is not None else None


def _controller_owner(metadata: dict) -> dict | None:
    controllers = [
        owner
        for owner in metadata.get("ownerReferences") or []
        if owner.get("controller") is True
        and owner.get("uid")
        and owner.get("kind")
    ]
    return controllers[0] if len(controllers) == 1 else None


def _by_uid(observations) -> dict[str, list[dict]]:
    grouped = {}
    for observation in observations:
        uid = (observation.get("metadata") or {}).get("uid")
        if uid:
            grouped.setdefault(uid, []).append(observation)
    for timeline in grouped.values():
        timeline.sort(key=lambda item: _parse_time(item["observedAt"]))
    return grouped


def _within(timestamp: str, start: str, end: str | None) -> bool:
    value = _parse_time(timestamp)
    return value >= _parse_time(start) and (
        end is None or value < _parse_time(end)
    )


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )
