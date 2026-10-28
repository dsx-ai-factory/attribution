# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kueue attempt boundaries reconstructed from retained workload history."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

_RESTART_LABEL = "jobset.sigs.k8s.io/restart-attempt"


def attempt_id_for(
    workload_uid: str,
    admission_key: str,
    restart_attempt: str,
) -> str:
    """Return the canonical identity shared by live and retained discovery."""
    canonical = json.dumps(
        {
            "admission": admission_key,
            "restartAttempt": restart_attempt,
            "workloadUid": workload_uid,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"kueue:v1:{workload_uid}:{digest}"


@dataclass(frozen=True)
class LiveAttemptIdentity:
    attempt_id: str
    start_time: datetime
    boundary_resolved: bool


def live_attempt_identity(
    workload: dict,
    resources: list[dict],
) -> LiveAttemptIdentity | None:
    """Resolve the current live attempt using retained-history semantics."""
    metadata = workload.get("metadata") or {}
    workload_uid = metadata.get("uid")
    admissions = _admissions([workload])
    if not workload_uid:
        return None
    if not admissions:
        created = metadata.get("creationTimestamp")
        if not created:
            return None
        return LiveAttemptIdentity(
            attempt_id=f"kueue:pre:v1:{workload_uid}",
            start_time=_parse_time(created),
            boundary_resolved=False,
        )

    admission = admissions[-1]
    restart_values: set[str] = set()
    restart_starts: dict[str, str] = {}
    for resource in resources:
        labels = (resource.get("metadata") or {}).get("labels") or {}
        restart = labels.get(_RESTART_LABEL)
        if restart is None:
            continue
        try:
            restart_number = int(restart)
        except (TypeError, ValueError):
            return None
        if restart_number < 0:
            return None
        restart_key = str(restart_number)
        restart_values.add(restart_key)
        timestamp = (
            (resource.get("status") or {}).get("startTime")
            or (resource.get("metadata") or {}).get("creationTimestamp")
        )
        if timestamp and (
            restart_key not in restart_starts
            or _parse_time(timestamp) < _parse_time(restart_starts[restart_key])
        ):
            restart_starts[restart_key] = timestamp

    if restart_values:
        restart_attempt = str(max(int(value) for value in restart_values))
        restart_start = restart_starts.get(restart_attempt, admission["time"])
    else:
        owner = _controller_owner(metadata)
        if owner is not None and owner.get("kind") == "JobSet":
            return None
        restart_attempt = "0"
        restart_start = admission["time"]
    start = max(admission["time"], restart_start, key=_parse_time)
    return LiveAttemptIdentity(
        attempt_id=attempt_id_for(
            workload_uid,
            admission["key"],
            restart_attempt,
        ),
        start_time=_parse_time(start),
        boundary_resolved=True,
    )


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
    boundary_resolved: bool
    pod_uids: frozenset[str]

    @property
    def attempt_id(self) -> str:
        return attempt_id_for(
            self.workload_uid,
            self.admission_key,
            self.restart_attempt,
        )


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
    workload_boundaries = _workload_boundaries(workloads)
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
            boundary = workload_boundaries[workload_uid]
            workload_end = boundary.end
            attempt_end = min(
                (
                    boundary
                    for boundary in (next_start, workload_end)
                    if boundary is not None
                ),
                key=_parse_time,
                default=None,
            )
            terminal = _terminal_for(
                start,
                next_start,
                terminals,
                workload_uid=workload_uid,
                owner=owner if boundary.resolved else None,
                workload_end=workload_end,
                allow_owner_at_workload_end=boundary.allow_owner_at_end,
            )
            end_time = (
                _parse_time(terminal["time"])
                if terminal is not None
                else _parse_time(attempt_end)
                if attempt_end is not None
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
                    open=(
                        terminal is None
                        and attempt_end is None
                        and not boundary.retired
                    ),
                    boundary_resolved=boundary.resolved,
                    pod_uids=frozenset(
                        uid
                        for uid, event in pod_events.items()
                        if boundary.resolved
                        and _pod_in_attempt(
                            event,
                            start["restart_attempt"],
                            start["time"],
                            attempt_end,
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


@dataclass(frozen=True)
class _WorkloadBoundary:
    end: str | None
    retired: bool
    resolved: bool
    allow_owner_at_end: bool


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


def _workload_boundaries(
    workloads: dict[str, list[dict]],
) -> dict[str, _WorkloadBoundary]:
    """Bound each Workload UID using source lifecycle timestamps only."""
    incarnations = []
    for workload_uid, timeline in workloads.items():
        admissions = _admissions(timeline)
        admitted_at = admissions[0]["time"] if admissions else None
        incarnations.append(
            (
                admitted_at or timeline[0]["observedAt"],
                workload_uid,
                timeline,
                admitted_at,
            )
        )
    incarnations.sort()
    boundaries = {}
    for index, (_first_seen, workload_uid, timeline, _admitted_at) in enumerate(
        incarnations
    ):
        deletion_observations = [
            observation
            for observation in timeline
            if observation.get("eventType") == "DELETED"
        ]
        deletion = min(
            (
                selected_metadata["deletionTimestamp"]
                for observation in deletion_observations
                if (
                    selected_metadata := observation.get("selectedMetadata")
                    or {}
                ).get("deletionTimestamp")
            ),
            key=_parse_time,
            default=None,
        )
        terminal = min(
            (
                condition["lastTransitionTime"]
                for observation in timeline
                for condition in (
                    (observation.get("status") or {}).get("conditions") or []
                )
                if condition.get("status") == "True"
                and condition.get("type") == "Finished"
                and condition.get("lastTransitionTime")
            ),
            key=_parse_time,
            default=None,
        )
        replacement = next(
            (
                admitted_at
                for _sort_time, _uid, _timeline, admitted_at in incarnations[
                    index + 1 :
                ]
                if admitted_at is not None
            ),
            None,
        )
        end = min(
            (
                boundary
                for boundary in (deletion, terminal, replacement)
                if boundary is not None
            ),
            key=_parse_time,
            default=None,
        )
        has_successor = index + 1 < len(incarnations)
        retired = bool(deletion_observations) or has_successor
        boundaries[workload_uid] = _WorkloadBoundary(
            end=end,
            retired=retired,
            resolved=not retired or end is not None,
            allow_owner_at_end=(
                terminal is not None
                and end is not None
                and _parse_time(terminal) == _parse_time(end)
            ),
        )
    return boundaries


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
    ordered_restarts = sorted(
        restarts.items(),
        key=lambda item: (_parse_time(item[1]), item[0]),
    )
    for index, admission in enumerate(admissions):
        next_admission = (
            admissions[index + 1]["time"]
            if index + 1 < len(admissions)
            else None
        )
        active_at_boundary = [
            (restart, timestamp)
            for restart, timestamp in ordered_restarts
            if _parse_time(timestamp) <= _parse_time(admission["time"])
        ]
        subsequent_restarts = [
            (restart, timestamp)
            for restart, timestamp in ordered_restarts
            if _parse_time(timestamp) > _parse_time(admission["time"])
            and (
                next_admission is None
                or _parse_time(timestamp) < _parse_time(next_admission)
            )
        ]
        if active_at_boundary:
            seed_restart = active_at_boundary[-1][0]
            scoped_restarts = [(seed_restart, admission["time"])]
            scoped_restarts.extend(
                (restart, timestamp)
                for restart, timestamp in subsequent_restarts
                if restart != seed_restart
            )
        elif subsequent_restarts:
            scoped_restarts = subsequent_restarts
        else:
            scoped_restarts = [("0", admission["time"])]
        for restart, timestamp in scoped_restarts:
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
        if kind not in {"Workload", "Job", "JobSet"}:
            continue
        status = observation.get("status") or {}
        for condition in status.get("conditions") or []:
            condition_type = condition.get("type")
            transition = condition.get("lastTransitionTime")
            supported = (
                kind == "Workload"
                and condition_type in {"Evicted", "Finished"}
                or kind == "Job"
                and condition_type in {"Failed", "Complete"}
                or kind == "JobSet"
                and condition_type in {
                    "RestartingJobSet",
                    "Completed",
                    "Failed",
                }
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
                        else str(max(int(status.get("restarts") or 0) - 1, 0))
                        if kind == "JobSet"
                        and condition_type == "RestartingJobSet"
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
    workload_end: str | None,
    allow_owner_at_workload_end: bool,
) -> dict | None:
    candidates = [
        terminal
        for terminal in terminals
        if (
                terminal["kind"] == "Workload"
                and terminal["uid"] == workload_uid
                or owner is not None
                and owner.get("kind") == "Job"
                and terminal["kind"] == "Job"
                and terminal["uid"] == owner.get("uid")
                or owner is not None
                and owner.get("kind") == "JobSet"
                and terminal["kind"] == "JobSet"
                and terminal["uid"] == owner.get("uid")
            )
            and _terminal_within(
                terminal["time"],
                start["time"],
                next_start,
            )
            and (
                workload_end is None
                or _parse_time(terminal["time"]) < _parse_time(workload_end)
                or (
                    (
                        terminal["kind"] == "Workload"
                        and terminal["uid"] == workload_uid
                        or allow_owner_at_workload_end
                    )
                    and _parse_time(terminal["time"]) == _parse_time(workload_end)
                )
            )
            and terminal["restart_attempt"] in {
                None,
                start["restart_attempt"],
            }
    ]
    return min(
        candidates,
        key=lambda terminal: (
            -_terminal_priority(terminal),
            _parse_time(terminal["time"]),
        ),
        default=None,
    )


def _terminal_priority(terminal: dict) -> int:
    """Prefer specific owner outcomes over aggregate Workload failures."""
    if (
        terminal["kind"] == "Job"
        and terminal["type"] == "Failed"
        and terminal.get("reason") == "DeadlineExceeded"
    ):
        return 3
    if terminal["kind"] in {"Job", "JobSet"}:
        return 2
    return 1


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
