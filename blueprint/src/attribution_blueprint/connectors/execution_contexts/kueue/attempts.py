# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kueue attempt boundaries reconstructed from retained workload history."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from attribution_blueprint.kubernetes.attempts import (
    RESTART_LABEL as _RESTART_LABEL,
    controller_owner as _controller_owner,
    latest_pod_observations as _pod_events,
    observations_by_uid as _by_uid,
    parse_time as _parse_time,
    restart_attempt as _restart_attempt,
    restart_starts as _restart_starts,
    within as _within,
)


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
    job_uids: frozenset[str] = frozenset()

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
        if item.get("kind") == "Workload" and (item.get("metadata") or {}).get("name") == workload_name
    )
    restart_starts = _restart_starts(observations)
    terminals = _terminal_events(observations)
    pod_events = _pod_events(observations)
    job_events = {
        uid: timeline[-1]
        for uid, timeline in _by_uid(item for item in observations if item.get("kind") == "Job").items()
    }
    workload_boundaries = _workload_boundaries(workloads)
    attempts = []
    for workload_uid, timeline in workloads.items():
        latest_metadata = timeline[-1].get("metadata") or {}
        admissions = _admissions(timeline)
        if not admissions:
            continue
        boundary = workload_boundaries[workload_uid]
        first_admission = admissions[0]["time"]
        scoped_restarts = {
            restart: timestamp
            for restart, timestamp in restart_starts.items()
            if _parse_time(timestamp) >= _parse_time(first_admission)
            and (boundary.end is None or _parse_time(timestamp) < _parse_time(boundary.end))
        }
        prior_restarts = [
            (restart, timestamp)
            for restart, timestamp in restart_starts.items()
            if _parse_time(timestamp) < _parse_time(first_admission)
        ]
        if prior_restarts:
            active_restart, active_timestamp = max(
                prior_restarts,
                key=lambda item: (_parse_time(item[1]), item[0]),
            )
            scoped_restarts[active_restart] = active_timestamp
        starts = _starts(admissions, scoped_restarts)
        owner = _controller_owner(latest_metadata)
        for index, start in enumerate(starts):
            next_start = starts[index + 1]["time"] if index + 1 < len(starts) else None
            workload_end = boundary.end
            attempt_end = min(
                (boundary for boundary in (next_start, workload_end) if boundary is not None),
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
                    open=(terminal is None and attempt_end is None and not boundary.retired),
                    boundary_resolved=boundary.resolved,
                    pod_uids=frozenset(
                        uid
                        for uid, event in pod_events.items()
                        if boundary.resolved
                        and _resource_in_attempt(
                            event,
                            start["restart_attempt"],
                            start["time"],
                            attempt_end,
                        )
                    ),
                    job_uids=frozenset(
                        uid
                        for uid, event in job_events.items()
                        if boundary.resolved
                        and _resource_in_attempt(
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


def attempts_from_live_resources(
    workload: dict,
    pods: list[dict],
    jobs: list[dict],
    *,
    now: datetime,
) -> list[WorkloadAttempt]:
    """Normalize live Kubernetes resources into retained-history evidence."""
    workload_metadata = workload.get("metadata") or {}
    workload_name = workload_metadata.get("name")
    if not workload_name or not workload_metadata.get("uid"):
        return []
    pods = [
        pod
        for pod in pods
        if (pod.get("metadata") or {}).get("uid")
    ]
    jobs = [
        job
        for job in jobs
        if (job.get("metadata") or {}).get("uid")
    ]
    owner = _controller_owner(workload.get("metadata") or {})
    if (
        owner is not None
        and owner.get("kind") == "JobSet"
        and not any(
            ((resource.get("metadata") or {}).get("labels") or {}).get(_RESTART_LABEL) is not None
            for resource in [*pods, *jobs]
        )
    ):
        return []
    observed_at = now.isoformat()
    admissions = _admissions(
        [
            {
                "status": workload.get("status") or {},
            }
        ]
    )
    resource_fallback = admissions[-1]["time"] if admissions else observed_at

    def observation(resource: dict, kind: str) -> dict:
        metadata = resource.get("metadata") or {}
        selected_metadata = {
            field: dict(values) for field in ("annotations", "labels") if (values := metadata.get(field) or {})
        }
        if metadata.get("deletionTimestamp"):
            selected_metadata["deletionTimestamp"] = metadata["deletionTimestamp"]
        return {
            "apiVersion": resource.get("apiVersion") or "",
            "kind": kind,
            "metadata": {
                "namespace": metadata.get("namespace") or "",
                "name": metadata.get("name") or "",
                "uid": metadata["uid"],
                "ownerReferences": metadata.get("ownerReferences") or [],
            },
            "eventType": "MODIFIED",
            "observedAt": (
                (resource.get("status") or {}).get("startTime")
                or metadata.get("creationTimestamp")
                or resource_fallback
            ),
            "selectedMetadata": selected_metadata or None,
            "spec": resource.get("spec") or {},
            "status": resource.get("status") or {},
        }

    history = {
        "observations": [
            observation(workload, "Workload"),
            *(observation(job, "Job") for job in jobs),
            *(observation(pod, "Pod") for pod in pods),
        ]
    }
    attempts = attempts_from_workload_history(
        history,
        workload_name=workload_name,
        now=now,
    )
    return attempts[-1:]


def merge_attempts(
    attempts: list[WorkloadAttempt],
) -> WorkloadAttempt:
    """Union evidence for one canonical attempt before context construction."""
    if not attempts:
        raise ValueError("at least one attempt is required")
    attempt_ids = {attempt.attempt_id for attempt in attempts}
    if len(attempt_ids) != 1:
        raise ValueError("cannot merge different Kueue attempts")
    first = attempts[0]
    terminals = [attempt.terminal for attempt in attempts if attempt.terminal is not None]
    terminal = min(
        terminals,
        key=lambda item: (
            -_terminal_priority(item),
            _parse_time(item["time"]),
        ),
        default=None,
    )
    end_time = _parse_time(terminal["time"]) if terminal is not None else max(attempt.end_time for attempt in attempts)
    return WorkloadAttempt(
        workload_uid=first.workload_uid,
        namespace=first.namespace,
        workload_name=first.workload_name,
        owner_kind=next(
            (attempt.owner_kind for attempt in attempts if attempt.owner_kind is not None),
            None,
        ),
        admission_key=first.admission_key,
        restart_attempt=first.restart_attempt,
        start_time=min(attempt.start_time for attempt in attempts),
        end_time=end_time,
        terminal=terminal,
        open=terminal is None and any(attempt.open for attempt in attempts),
        boundary_resolved=all(attempt.boundary_resolved for attempt in attempts),
        pod_uids=frozenset().union(*(attempt.pod_uids for attempt in attempts)),
        job_uids=frozenset().union(*(attempt.job_uids for attempt in attempts)),
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
            if condition.get("type") == "Admitted" and condition.get("status") == "True" and transition:
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
    for index, (_first_seen, workload_uid, timeline, _admitted_at) in enumerate(incarnations):
        deletion_observations = [observation for observation in timeline if observation.get("eventType") == "DELETED"]
        deletion = min(
            (
                selected_metadata["deletionTimestamp"]
                for observation in deletion_observations
                if (selected_metadata := observation.get("selectedMetadata") or {}).get("deletionTimestamp")
            ),
            key=_parse_time,
            default=None,
        )
        terminal = min(
            (
                condition["lastTransitionTime"]
                for observation in timeline
                for condition in ((observation.get("status") or {}).get("conditions") or [])
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
                for _sort_time, _uid, _timeline, admitted_at in incarnations[index + 1 :]
                if admitted_at is not None
            ),
            None,
        )
        end = min(
            (boundary for boundary in (deletion, terminal, replacement) if boundary is not None),
            key=_parse_time,
            default=None,
        )
        has_successor = index + 1 < len(incarnations)
        retired = bool(deletion_observations) or has_successor
        boundaries[workload_uid] = _WorkloadBoundary(
            end=end,
            retired=retired,
            resolved=not retired or end is not None,
            allow_owner_at_end=(terminal is not None and end is not None and _parse_time(terminal) == _parse_time(end)),
        )
    return boundaries


def _starts(admissions: list[dict], restarts: dict[str, str]) -> list[dict]:
    starts = []
    ordered_restarts = sorted(
        restarts.items(),
        key=lambda item: (_parse_time(item[1]), item[0]),
    )
    for index, admission in enumerate(admissions):
        next_admission = admissions[index + 1]["time"] if index + 1 < len(admissions) else None
        active_at_boundary = [
            (restart, timestamp)
            for restart, timestamp in ordered_restarts
            if _parse_time(timestamp) <= _parse_time(admission["time"])
        ]
        subsequent_restarts = [
            (restart, timestamp)
            for restart, timestamp in ordered_restarts
            if _parse_time(timestamp) > _parse_time(admission["time"])
            and (next_admission is None or _parse_time(timestamp) < _parse_time(next_admission))
        ]
        if active_at_boundary:
            seed_restart = active_at_boundary[-1][0]
            scoped_restarts = [(seed_restart, admission["time"])]
            scoped_restarts.extend(
                (restart, timestamp) for restart, timestamp in subsequent_restarts if restart != seed_restart
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
        requeued_at = max(
            (
                condition["lastTransitionTime"]
                for condition in status.get("conditions") or []
                if condition.get("type") == "Requeued"
                and condition.get("status") == "True"
                and condition.get("lastTransitionTime")
            ),
            key=_parse_time,
            default=None,
        )
        for condition in status.get("conditions") or []:
            condition_type = condition.get("type")
            transition = condition.get("lastTransitionTime")
            if (
                kind == "Workload"
                and condition_type == "Evicted"
                and requeued_at is not None
                and transition is not None
                and _parse_time(requeued_at) >= _parse_time(transition)
            ):
                continue
            supported = (
                kind == "Workload"
                and condition_type in {"Evicted", "Finished"}
                or kind == "Job"
                and condition_type in {"Failed", "Complete"}
                or kind == "JobSet"
                and condition_type
                in {
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
                    "owner_uid": (
                        owner["uid"]
                        if (
                            owner := _controller_owner(
                                observation.get("metadata") or {}
                            )
                        )
                        else None
                    ),
                    "type": condition_type,
                    "reason": condition.get("reason"),
                    "time": transition,
                    "restart_attempt": (
                        _restart_attempt(observation)
                        if kind == "Job"
                        else str(max(int(status.get("restarts") or 0) - 1, 0))
                        if kind == "JobSet" and condition_type == "RestartingJobSet"
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
                or owner is not None
                and owner.get("kind") == "JobSet"
                and terminal["kind"] == "Job"
                and terminal["type"] == "Failed"
                and terminal["owner_uid"] == owner.get("uid")
        )
        and _within(
            terminal["time"],
            start["time"],
            next_start,
            include_end=True,
        )
        and (
            workload_end is None
            or _parse_time(terminal["time"]) < _parse_time(workload_end)
            or (
                (terminal["kind"] == "Workload" and terminal["uid"] == workload_uid or allow_owner_at_workload_end)
                and _parse_time(terminal["time"]) == _parse_time(workload_end)
            )
        )
        and terminal["restart_attempt"]
        in {
            None,
            start["restart_attempt"],
        }
    ]
    if owner is not None and owner.get("kind") == "JobSet":
        child_failures = [
            terminal
            for terminal in candidates
            if terminal["kind"] == "Job"
            and terminal["type"] == "Failed"
        ]
        if child_failures:
            return min(
                child_failures,
                key=lambda terminal: _parse_time(terminal["time"]),
            )
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
    if terminal["kind"] == "Job" and terminal["type"] == "Failed" and terminal.get("reason") == "DeadlineExceeded":
        return 3
    if terminal["kind"] in {"Job", "JobSet"}:
        return 2
    return 1


def _resource_in_attempt(
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
