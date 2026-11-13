# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct KAI attempts from Kubernetes resource snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from attribution_blueprint.connectors.execution_contexts.kai.helpers import (
    POD_GROUP_ANNOTATION,
    resolve_owner,
)
from attribution_blueprint.kubernetes.attempts import (
    controller_owner,
    kai_podgroup_start_timestamp,
    latest_pod_observations,
    observations_by_uid,
    parse_time,
    restart_attempt,
    restart_start_candidates,
    retained_history_covers_window,
)


def attempt_id_for(
    pod_group_uid: str,
    start_key: str,
    restart_attempt: str,
) -> str:
    """Return a stable identity for a PodGroup allocation and restart generation."""
    canonical = json.dumps(
        {
            "boundary": start_key,
            "resourceUid": pod_group_uid,
            "restartAttempt": restart_attempt,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"kai:v1:{pod_group_uid}:{digest}"


@dataclass(frozen=True)
class PodGroupAttempt:
    """Describe one allocation or restart interval and its proven resource membership."""
    pod_group_uid: str
    namespace: str
    pod_group_name: str
    owner_kind: str
    owner_name: str
    owner_uid: str
    start_key: str
    restart_attempt: str
    start_time: datetime
    end_time: datetime
    terminal: dict | None
    open: bool
    boundary_resolved: bool
    membership_resolved: bool
    pod_uids: frozenset[str]
    job_uids: frozenset[str] = frozenset()

    @property
    def attempt_id(self) -> str:
        return attempt_id_for(
            self.pod_group_uid,
            self.start_key,
            self.restart_attempt,
        )


@dataclass(frozen=True)
class _ResolvedEvidence:
    """Share generation inference across boundaries, terminals, and membership."""
    restart_candidates: list[tuple[dict, str, str]]
    terminals: list[dict]
    pods: dict[str, dict]
    jobs: dict[str, dict]
    job_restarts: dict[str, str]


def snapshots_from_live_resources(
    podgroup: dict,
    pods: list[dict],
    owner_jobs: list[dict],
    owner_object: dict,
    *,
    now: datetime,
) -> list[dict]:
    """Project live resources into the daemon snapshot schema, retaining live spec."""
    metadata = podgroup.get("metadata") or {}
    pod_group_name = metadata.get("name")
    if not pod_group_name or not metadata.get("uid"):
        return []
    owner = resolve_owner(metadata)
    if owner is None:
        return []

    fallback = (
        kai_podgroup_start_timestamp(podgroup)
        or metadata.get("creationTimestamp")
        or now.isoformat()
    )

    def snapshot(resource: dict, kind: str) -> dict:
        resource_metadata = resource.get("metadata") or {}
        selected_metadata = {
            field: dict(values)
            for field in ("annotations", "labels")
            if (values := resource_metadata.get(field) or {})
        }
        if resource_metadata.get("deletionTimestamp"):
            selected_metadata["deletionTimestamp"] = resource_metadata[
                "deletionTimestamp"
            ]
        status = resource.get("status") or {}
        if kind == "PodGroup" and (
            last_start := kai_podgroup_start_timestamp(resource)
        ):
            status = {
                **status,
                "lastStartTimestamp": last_start,
            }
        return {
            "apiVersion": resource.get("apiVersion") or "",
            "kind": kind,
            "metadata": {
                "namespace": resource_metadata.get("namespace") or "",
                "name": resource_metadata.get("name") or "",
                "uid": resource_metadata.get("uid"),
                "resourceVersion": resource_metadata.get(
                    "resourceVersion"
                ),
                "ownerReferences": (
                    resource_metadata.get("ownerReferences") or []
                ),
            },
            "eventType": "MODIFIED",
            "observedAt": (
                status.get("startTime")
                or resource_metadata.get("creationTimestamp")
                or fallback
            ),
            "selectedMetadata": selected_metadata or None,
            "spec": resource.get("spec") or {},
            "status": status,
        }

    resources: dict[tuple[str, str], dict] = {}
    for resource, kind in (
        (owner_object, owner["kind"]),
        *((job, "Job") for job in owner_jobs),
        *((pod, "Pod") for pod in pods),
    ):
        uid = (resource.get("metadata") or {}).get("uid")
        if kind and uid:
            resources[(kind, uid)] = snapshot(resource, kind)

    return [
        snapshot(podgroup, "PodGroup"),
        *resources.values(),
    ]


def attempts_from_podgroup_history(
    history: dict,
    *,
    pod_group_name: str,
    now: datetime,
) -> list[PodGroupAttempt]:
    """Reconstruct attempts from scheduling starts, restarts, and exact-owner evidence.

    Missing owners or starts produce no attempt; unresolved boundaries preserve
    uncertainty rather than claim Pod membership."""
    snapshots = history.get("observations") or []
    resolved = _resolve_evidence(snapshots)
    podgroups = observations_by_uid(
        item
        for item in snapshots
        if item.get("kind") == "PodGroup"
        and (item.get("metadata") or {}).get("name") == pod_group_name
    )
    terminals = resolved.terminals
    pod_events = resolved.pods
    job_events = resolved.jobs
    podgroup_boundaries = _podgroup_boundaries(podgroups)
    attempts = []
    for pod_group_uid, timeline in podgroups.items():
        latest = timeline[-1]
        metadata = latest.get("metadata") or {}
        selected = latest.get("selectedMetadata") or {}
        owner = _podgroup_owner(
            {
                **metadata,
                "annotations": selected.get("annotations") or {},
            }
        )
        if owner is None:
            continue
        boundary = podgroup_boundaries[pod_group_uid]
        podgroup_starts = _podgroup_starts(timeline)
        if not podgroup_starts:
            continue
        lifetime_start = podgroup_starts[0]["time"]
        scoped_restarts = _scoped_restart_starts(
            resolved.restart_candidates,
            pod_group_name=pod_group_name,
            owner=owner,
            jobs=job_events,
            lifetime_start=lifetime_start,
            lifetime_end=boundary.end,
            accept_prior_restart=boundary.accept_prior_restart,
        )
        recreated = len(podgroups) > 1
        starts = _attempt_starts(
            podgroup_starts,
            scoped_restarts,
            initial_execution_evidence=_has_initial_execution(
                pod_events,
                job_events,
                pod_group_name=pod_group_name,
                owner=owner,
                inferred_job_restarts=resolved.job_restarts,
                start=lifetime_start,
                end=min(
                    scoped_restarts.values(),
                    key=parse_time,
                    default=None,
                ),
                allow_before_start=not recreated,
            ),
        )
        if owner["kind"] == "JobSet":
            starts = _coalesce_initial_allocation(
                starts,
                snapshots,
                terminals,
                pod_group_name,
                [history.get("coverage") or {}, *(history.get("coverages") or [])],
                lifetime_start,
            )
        assigned_terminals = _assign_terminals(
            starts,
            terminals,
            owner,
            owner_outcome_end=boundary.owner_outcome_end,
        )
        for index, start in enumerate(starts):
            first_attempt_for_restart = not any(
                previous["restart_attempt"] == start["restart_attempt"]
                for previous in starts[:index]
            )
            next_start = (
                starts[index + 1]["time"]
                if index + 1 < len(starts)
                else None
            )
            attempt_end = min(
                (
                    candidate
                    for candidate in (next_start, boundary.end)
                    if candidate is not None
                ),
                key=parse_time,
                default=None,
            )
            terminal = (
                assigned_terminals[index]
                if boundary.resolved
                or boundary.terminal_resolvable
                else None
            )
            boundary_resolved = next_start is not None or boundary.resolved or (
                boundary.terminal_resolvable
                and terminal is not None
            )
            ambiguous_predated_pod = recreated and any(
                _pod_matches_attempt(
                    pod,
                    pod_group_name,
                    start["restart_attempt"],
                    allow_unannotated=False,
                )
                and (timestamp := _pod_timestamp(pod)) is not None
                and parse_time(timestamp) < parse_time(start["time"])
                for pod in pod_events.values()
            )
            attempts.append(
                PodGroupAttempt(
                    pod_group_uid=pod_group_uid,
                    namespace=metadata.get("namespace") or "",
                    pod_group_name=pod_group_name,
                    owner_kind=owner["kind"],
                    owner_name=owner["name"],
                    owner_uid=owner["uid"],
                    start_key=start["start_key"],
                    restart_attempt=start["restart_attempt"],
                    start_time=parse_time(start["time"]),
                    end_time=(
                        parse_time(terminal["time"])
                        if terminal is not None
                        else parse_time(attempt_end)
                        if attempt_end is not None
                        else now
                    ),
                    terminal=terminal,
                    open=(
                        terminal is None
                        and attempt_end is None
                        and not boundary.retired
                    ),
                    boundary_resolved=boundary_resolved,
                    membership_resolved=not ambiguous_predated_pod,
                    pod_uids=frozenset(
                        uid
                        for uid, pod in pod_events.items()
                        if boundary_resolved
                        and _pod_owned_by(pod, owner, job_events)
                        and _pod_in_attempt(
                            pod,
                            pod_group_name,
                            start["restart_attempt"],
                            start["time"],
                            attempt_end,
                            allow_unannotated=not recreated,
                            allow_before_start=(
                                not recreated and index == 0
                            ),
                            allow_restart_before_start=(
                                not recreated
                                and first_attempt_for_restart
                            ),
                        )
                    ),
                    job_uids=frozenset(
                        uid
                        for uid, job in job_events.items()
                        if boundary_resolved
                        and _job_owned_by(job, owner)
                        and _resource_in_attempt(
                            job,
                            start["restart_attempt"],
                            start["time"],
                            attempt_end,
                            inferred_restart=resolved.job_restarts.get(
                                uid
                            ),
                            allow_before_start=(
                                not recreated and index == 0
                            ),
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
class _PodGroupBoundary:
    end: str | None
    owner_outcome_end: str | None
    retired: bool
    resolved: bool
    terminal_resolvable: bool
    accept_prior_restart: bool


def _podgroup_boundaries(
    podgroups: dict[str, list[dict]],
) -> dict[str, _PodGroupBoundary]:
    """Bound membership by deletion or replacement, and owner outcomes by replacement.

    Deletion alone must not discard a later completion from the same owner."""
    incarnations = []
    for pod_group_uid, timeline in podgroups.items():
        starts = _podgroup_starts(timeline)
        started_at = starts[0]["time"] if starts else None
        incarnations.append(
            (
                started_at or timeline[0]["observedAt"],
                pod_group_uid,
                timeline,
                started_at,
            )
        )
    incarnations.sort()

    boundaries = {}
    for index, (_first_seen, pod_group_uid, timeline, _started_at) in enumerate(
        incarnations
    ):
        deletion_observations = [
            snapshot
            for snapshot in timeline
            if snapshot.get("eventType") == "DELETED"
        ]
        deletion = min(
            (
                selected_metadata["deletionTimestamp"]
                for snapshot in deletion_observations
                if (
                    selected_metadata := snapshot.get("selectedMetadata")
                    or {}
                ).get("deletionTimestamp")
            ),
            key=parse_time,
            default=None,
        )
        replacement = next(
            (
                started_at
                for _sort_time, _uid, _timeline, started_at in incarnations[
                    index + 1 :
                ]
                if started_at is not None
            ),
            None,
        )
        end = min(
            (
                candidate
                for candidate in (deletion, replacement)
                if candidate is not None
            ),
            key=parse_time,
            default=None,
        )
        has_successor = index + 1 < len(incarnations)
        retired = bool(deletion_observations) or has_successor
        boundaries[pod_group_uid] = _PodGroupBoundary(
            end=end,
            owner_outcome_end=replacement,
            retired=retired,
            # A standalone deleted PodGroup still has an exact owner UID.
            # Its owner terminal and Pod evidence remain trustworthy even
            # when Kubernetes omitted deletionTimestamp from the retained
            # delete event. A successor UID remains ambiguous without a
            # source timestamp that separates the incarnations.
            resolved=not has_successor or end is not None,
            # An unscheduled successor is not an execution. An exact-owner
            # terminal can still close the last scheduled incarnation and
            # make its retained Pod evidence attributable.
            terminal_resolvable=has_successor and replacement is None,
            # A Pod or child Job may establish the active restart before
            # KAI records the first scheduling timestamp. Carry only the
            # latest such restart into the first PodGroup incarnation.
            accept_prior_restart=index == 0,
        )
    return boundaries


def _podgroup_owner(metadata: dict) -> dict | None:
    """Select the workload owner used to match Pods, restarts, and completion events.

    Prefer the top owner or controller. If neither resolves, accept a sole
    named owner reference with a UID. With multiple unresolved candidates,
    return None because choosing one could attribute the attempt to the wrong
    workload.
    """
    owner = resolve_owner(metadata)
    if owner is None:
        candidates = [
            reference
            for reference in metadata.get("ownerReferences") or []
            if reference.get("name") and reference.get("uid")
        ]
        owner = candidates[0] if len(candidates) == 1 else None
    if owner is None:
        return None

    group = owner.get("group")
    api_version = owner.get("apiVersion")
    if group is None and isinstance(api_version, str) and "/" in api_version:
        group = api_version.split("/", 1)[0]
    if owner.get("kind") == "Job" and group in (None, "batch"):
        return owner
    if (
        owner.get("kind") == "JobSet"
        and group == "jobset.x-k8s.io"
    ):
        return owner
    return None


def _podgroup_starts(timeline: list[dict]) -> list[dict]:
    starts = {}
    for snapshot in timeline:
        timestamp = kai_podgroup_start_timestamp(snapshot)
        if timestamp:
            starts[str(timestamp)] = {
                "key": str(timestamp),
                "time": timestamp,
            }
    return sorted(starts.values(), key=lambda item: parse_time(item["time"]))


def _has_initial_execution(
    pod_events: dict[str, dict],
    job_events: dict[str, dict],
    *,
    pod_group_name: str,
    owner: dict,
    inferred_job_restarts: dict[str, str],
    start: str,
    end: str | None,
    allow_before_start: bool,
) -> bool:
    """Reuse attempt membership rules to verify pre-restart execution."""
    for pod in pod_events.values():
        if _pod_owned_by(
            pod,
            owner,
            job_events,
        ) and _pod_in_attempt(
            pod,
            pod_group_name,
            "0",
            start,
            end,
            allow_unannotated=allow_before_start,
            allow_before_start=allow_before_start,
            allow_restart_before_start=allow_before_start,
        ):
            return True

    for job in job_events.values():
        metadata = job.get("metadata") or {}
        if _job_owned_by(job, owner) and _resource_in_attempt(
            job,
            "0",
            start,
            end,
            inferred_restart=inferred_job_restarts.get(
                metadata.get("uid")
            ),
            allow_before_start=allow_before_start,
        ):
            return True
    return False


def _job_owned_by(job: dict, owner: dict) -> bool:
    """Match the exact Job or JobSet UID to isolate recreated workloads."""
    metadata = job.get("metadata") or {}
    if owner["kind"] == "Job":
        return metadata.get("uid") == owner["uid"]
    controller = controller_owner(metadata)
    return bool(
        controller
        and controller["kind"] == "JobSet"
        and controller["uid"] == owner["uid"]
    )


def _pod_owned_by(
    pod: dict,
    owner: dict,
    jobs: dict[str, dict],
) -> bool:
    """Verify the Pod controller against the resolved workload ownership chain."""
    controller = controller_owner(pod.get("metadata") or {})
    if controller is None:
        return True
    if controller["kind"] != "Job":
        return False
    if owner["kind"] == "Job":
        return controller["uid"] == owner["uid"]
    job = jobs.get(controller["uid"])
    return bool(job and _job_owned_by(job, owner))


def _scoped_restart_starts(
    candidates: list[tuple[dict, str, str]],
    *,
    pod_group_name: str,
    owner: dict,
    jobs: dict[str, dict],
    lifetime_start: str,
    lifetime_end: str | None,
    accept_prior_restart: bool,
) -> dict[str, str]:
    """Collapse restart evidence only after applying incarnation/owner scope."""
    scoped: dict[str, str] = {}
    prior = []
    for snapshot, generation, timestamp in candidates:
        kind = snapshot.get("kind")
        if kind == "Pod":
            annotations = (
                (snapshot.get("selectedMetadata") or {}).get(
                    "annotations"
                )
                or {}
            )
            matches = (
                annotations.get(POD_GROUP_ANNOTATION)
                == pod_group_name
                and _pod_owned_by(snapshot, owner, jobs)
            )
        else:
            matches = _job_owned_by(snapshot, owner)
        if not matches:
            continue

        timestamp_value = parse_time(timestamp)
        if timestamp_value < parse_time(lifetime_start):
            if accept_prior_restart:
                prior.append((generation, timestamp))
            continue
        if (
            lifetime_end is not None
            and timestamp_value >= parse_time(lifetime_end)
        ):
            continue
        if (
            generation not in scoped
            or timestamp_value < parse_time(scoped[generation])
        ):
            scoped[generation] = timestamp

    if prior:
        active_generation, active_timestamp = max(
            prior,
            key=lambda item: (parse_time(item[1]), item[0]),
        )
        scoped[active_generation] = active_timestamp
    return scoped


def _attempt_starts(
    podgroup_starts: list[dict],
    restarts: dict[str, str],
    *,
    initial_execution_evidence: bool = False,
) -> list[dict]:
    """Carry restart generations across scheduling boundaries.

    Defer a first nonzero restart to its evidence time unless earlier execution
    establishes an initial interval."""
    starts = []
    ordered_restarts = sorted(
        restarts.items(),
        key=lambda item: (parse_time(item[1]), item[0]),
    )
    active_restart = "0"
    for index, podgroup_start in enumerate(podgroup_starts):
        boundary_time = podgroup_start["time"]
        next_podgroup_start = (
            podgroup_starts[index + 1]["time"]
            if index + 1 < len(podgroup_starts)
            else None
        )
        active_at_boundary = [
            (restart, timestamp)
            for restart, timestamp in ordered_restarts
            if parse_time(timestamp) <= parse_time(boundary_time)
        ]
        if active_at_boundary:
            active_restart = active_at_boundary[-1][0]
        changes_in_interval = [
            (restart, timestamp)
            for restart, timestamp in ordered_restarts
            if parse_time(timestamp) > parse_time(boundary_time)
            and (
                next_podgroup_start is None
                or parse_time(timestamp)
                < parse_time(next_podgroup_start)
            )
        ]
        defer_initial_nonzero = (
            index == 0
            and not active_at_boundary
            and changes_in_interval
            and changes_in_interval[0][0] != "0"
            and not initial_execution_evidence
        )
        if not defer_initial_nonzero:
            starts.append(
                {
                    "start_key": podgroup_start["key"],
                    "restart_attempt": active_restart,
                    "time": boundary_time,
                }
            )
        for restart, timestamp in changes_in_interval:
            if restart == active_restart:
                continue
            active_restart = restart
            starts.append(
                {
                    "start_key": podgroup_start["key"],
                    "restart_attempt": restart,
                    "time": timestamp,
                }
            )
    return starts


def _coalesce_initial_allocation(
    starts: list[dict],
    snapshots: list[dict],
    terminal_events: list[dict],
    pod_group_name: str,
    coverages: list[dict],
    lifetime_start: str,
) -> list[dict]:
    """Join retry preparation to its first allocation when history proves continuity.

    Require a Pending-to-scheduled Pod transition in the same generation and
    uninterrupted watches. Earlier execution or a terminal blocks the merge;
    missing evidence leaves separate boundaries rather than guessing a retry.
    """
    result = []
    for start in starts:
        previous = result[-1] if result else None
        result.append(start)
        if (
            previous is None
            or previous["restart_attempt"] != start["restart_attempt"]
            or previous["time"] == previous["start_key"]
        ):
            continue
        begin, end = parse_time(previous["time"]), parse_time(start["time"])
        if not any(
            retained_history_covers_window(
                c,
                {"pods", "jobs", "jobsets", "kai-podgroups"},
                parse_time(lifetime_start),
                end,
            )
            for c in coverages
        ):
            continue
        pending, allocated, earlier = False, False, False
        for snapshot in snapshots:
            if (
                snapshot.get("kind") != "Pod"
                or restart_attempt(snapshot) != start["restart_attempt"]
                or not _pod_matches_attempt(
                    snapshot,
                    pod_group_name,
                    start["restart_attempt"],
                    allow_unannotated=False,
                )
            ):
                continue
            status = snapshot.get("status") or {}
            execution_start = status.get("startTime")
            earlier |= bool(execution_start and parse_time(execution_start) < end)
            pending |= (
                status.get("phase") == "Pending"
                and not execution_start
                and begin <= parse_time(snapshot["observedAt"]) < end
            )
            for condition in status.get("conditions") or []:
                if (
                    condition.get("type") == "PodScheduled"
                    and condition.get("status") == "True"
                    and condition.get("lastTransitionTime")
                ):
                    scheduled = parse_time(condition["lastTransitionTime"])
                    earlier |= scheduled < end
                    allocated |= scheduled == end
        if (
            pending
            and allocated
            and not earlier
            and not any(
                begin <= parse_time(event["time"]) <= end
                and event["type"] in {"Failed", "Complete", "Completed"}
                and event.get("restart_attempt") in {None, start["restart_attempt"]}
                for event in terminal_events
            )
        ):
            result.pop()
    return result


def _infer_job_restarts(snapshots: list[dict]) -> dict[str, str]:
    """Infer a Job generation only when its explicitly labeled Pods agree."""
    pod_generations_by_job: dict[str, set[str]] = {}
    for snapshot in snapshots:
        if snapshot.get("kind") != "Pod":
            continue
        generation = restart_attempt(snapshot)
        controller = controller_owner(
            snapshot.get("metadata") or {}
        )
        if (
            generation is not None
            and controller
            and controller["kind"] == "Job"
        ):
            pod_generations_by_job.setdefault(
                controller["uid"],
                set(),
            ).add(generation)
    return {
        job_uid: next(iter(generations))
        for job_uid, generations in pod_generations_by_job.items()
        if len(generations) == 1
    }


def _job_restart(
    snapshot: dict,
    inferred_job_restarts: dict[str, str],
) -> str | None:
    generation = restart_attempt(snapshot)
    if generation is not None:
        return generation
    job_uid = (snapshot.get("metadata") or {}).get("uid")
    return inferred_job_restarts.get(job_uid)


def _terminal_events(
    snapshots: list[dict],
    inferred_job_restarts: dict[str, str],
) -> list[dict]:
    events = []
    for snapshot in snapshots:
        kind = snapshot.get("kind")
        if kind not in {"Job", "JobSet"}:
            continue
        status = snapshot.get("status") or {}
        conditions = status.get("conditions") or []
        has_job_terminal = any(
            condition.get("status") == "True"
            and condition.get("type") in {"Failed", "Complete"}
            and condition.get("lastTransitionTime")
            for condition in conditions
        )
        if (
            kind == "Job"
            and not has_job_terminal
            and status.get("completionTime")
        ):
            events.append(
                {
                    "kind": "Job",
                    "uid": (snapshot.get("metadata") or {}).get(
                        "uid"
                    ),
                    "type": "Complete",
                    "reason": None,
                    "time": status["completionTime"],
                    "restart_attempt": _job_restart(
                        snapshot,
                        inferred_job_restarts,
                    ),
                }
            )
        for condition in conditions:
            condition_type = condition.get("type")
            transition = condition.get("lastTransitionTime")
            supported = (
                kind == "Job"
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
                    "uid": (snapshot.get("metadata") or {}).get("uid"),
                    "type": condition_type,
                    "reason": condition.get("reason"),
                    "time": transition,
                    "restart_attempt": (
                        _job_restart(
                            snapshot,
                            inferred_job_restarts,
                        )
                        if kind == "Job"
                        else str(max(int(status.get("restarts") or 0) - 1, 0))
                        if condition_type == "RestartingJobSet"
                        else None
                    ),
                }
            )
    unique = {
        json.dumps(event, sort_keys=True, separators=(",", ":")): event
        for event in events
    }
    return sorted(
        unique.values(),
        key=lambda item: (
            parse_time(item["time"]),
            json.dumps(item, sort_keys=True, separators=(",", ":")),
        ),
    )


def _resolve_evidence(snapshots: list[dict]) -> _ResolvedEvidence:
    """Resolve snapshots once so boundaries and resource membership use the same generations."""
    job_restarts = _infer_job_restarts(snapshots)
    return _ResolvedEvidence(
        restart_candidates=restart_start_candidates(
            snapshots,
            job_restarts,
        ),
        terminals=_terminal_events(snapshots, job_restarts),
        pods=latest_pod_observations(snapshots),
        jobs={
            uid: timeline[-1]
            for uid, timeline in observations_by_uid(
                snapshot
                for snapshot in snapshots
                if snapshot.get("kind") == "Job"
            ).items()
        },
        job_restarts=job_restarts,
    )


def _assign_terminals(
    starts: list[dict],
    terminals: list[dict],
    owner: dict,
    *,
    owner_outcome_end: str | None,
) -> list[dict | None]:
    """Assign each exact-owner terminal to at most one attempt.

    Starts are ordered. Trying the preceding interval first makes a terminal
    at a shared boundary close that attempt without also closing the next.
    """
    assigned: list[dict | None] = [None] * len(starts)
    for terminal in terminals:
        if (
            terminal["kind"] != owner["kind"]
            or terminal["uid"] != owner["uid"]
        ):
            continue
        terminal_time = parse_time(terminal["time"])
        if (
            owner_outcome_end is not None
            and terminal_time >= parse_time(owner_outcome_end)
        ):
            continue
        for index, start in enumerate(starts):
            start_time = parse_time(start["time"])
            next_time = (
                parse_time(starts[index + 1]["time"])
                if index + 1 < len(starts)
                else None
            )
            if (
                terminal_time >= start_time
                and (next_time is None or terminal_time <= next_time)
                and terminal["restart_attempt"]
                in {None, start["restart_attempt"]}
            ):
                if assigned[index] is None:
                    assigned[index] = terminal
                # This interval owns the transition even when another
                # observation already supplied its terminal details.
                break
    return assigned


def _pod_in_attempt(
    snapshot: dict,
    pod_group_name: str,
    expected_restart: str,
    start: str,
    end: str | None,
    *,
    allow_unannotated: bool,
    allow_before_start: bool,
    allow_restart_before_start: bool,
) -> bool:
    if not _pod_matches_attempt(
        snapshot,
        pod_group_name,
        expected_restart,
        allow_unannotated=allow_unannotated,
    ):
        return False
    observed_restart = restart_attempt(snapshot)
    timestamp = _pod_timestamp(snapshot)
    if timestamp is None:
        return False
    if end is not None and parse_time(timestamp) >= parse_time(end):
        return False
    return (
        allow_before_start
        or (
            allow_restart_before_start
            and observed_restart is not None
        )
        or parse_time(timestamp) >= parse_time(start)
    )


def _resource_in_attempt(
    snapshot: dict,
    expected_restart: str,
    start: str,
    end: str | None,
    *,
    inferred_restart: str | None = None,
    allow_before_start: bool,
) -> bool:
    observed_restart = restart_attempt(snapshot)
    if observed_restart is None:
        observed_restart = inferred_restart
    if (
        observed_restart is not None
        and observed_restart != expected_restart
    ):
        return False
    timestamp = (snapshot.get("status") or {}).get(
        "startTime"
    ) or snapshot.get("observedAt")
    if timestamp is None:
        return False
    if end is not None and parse_time(timestamp) >= parse_time(end):
        return False
    return allow_before_start or parse_time(timestamp) >= parse_time(start)


def _pod_matches_attempt(
    snapshot: dict,
    pod_group_name: str,
    expected_restart: str,
    *,
    allow_unannotated: bool,
) -> bool:
    annotations = (
        (snapshot.get("selectedMetadata") or {}).get("annotations") or {}
    )
    observed_group = annotations.get(POD_GROUP_ANNOTATION)
    if observed_group != pod_group_name and not (
        observed_group is None and allow_unannotated
    ):
        return False
    observed_restart = restart_attempt(snapshot)
    if observed_restart is not None and observed_restart != expected_restart:
        return False
    return True


def _pod_timestamp(snapshot: dict) -> str | None:
    return (snapshot.get("status") or {}).get(
        "startTime"
    ) or snapshot.get("observedAt")
