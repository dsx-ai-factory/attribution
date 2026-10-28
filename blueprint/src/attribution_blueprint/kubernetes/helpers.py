# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-neutral Kubernetes status helpers."""

from __future__ import annotations

from datetime import datetime, timezone

NODE_LOSS_DISRUPTION_REASONS = frozenset({"DeletionByPodGC"})


def earliest_job_failure_reason(jobs: list[dict]) -> str | None:
    """Return the earliest Failed/FailureTarget reason across Jobs."""
    earliest: tuple[datetime, str] | None = None
    for job in jobs:
        conditions = (job.get("status") or {}).get("conditions") or []
        for condition_type in ("Failed", "FailureTarget"):
            matches = [
                condition
                for condition in conditions
                if (condition or {}).get("type") == condition_type
            ]
            if not matches:
                continue
            condition = max(
                matches,
                key=lambda item: item.get("lastTransitionTime", ""),
            )
            if condition.get("status") != "True":
                continue
            reason = (condition.get("reason") or "").strip()
            transition = _parse_dt(
                condition.get("lastTransitionTime")
            )
            if (
                reason
                and transition is not None
                and (
                    earliest is None
                    or transition < earliest[0]
                )
            ):
                earliest = transition, reason
    return earliest[1] if earliest else None


def job_time_limit_minutes(jobs: list[dict]) -> int | None:
    """Return the shortest positive activeDeadlineSeconds in minutes."""
    deadlines = [
        deadline
        for job in jobs
        if isinstance(
            deadline := (job.get("spec") or {}).get(
                "activeDeadlineSeconds"
            ),
            int,
        )
        and not isinstance(deadline, bool)
        and deadline > 0
    ]
    return (min(deadlines) + 59) // 60 if deadlines else None


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return (
        parsed
        if parsed.tzinfo is not None
        else parsed.replace(tzinfo=timezone.utc)
    )


def _most_recent_condition(conditions: list[dict], cond_type: str) -> dict | None:
    matches = [condition for condition in conditions if condition.get("type") == cond_type]
    return max(matches, key=lambda condition: condition.get("lastTransitionTime", "")) if matches else None


def pod_node_failure_reason(pod: dict) -> str | None:
    status = pod.get("status") or {}
    if status.get("reason") == "NodeLost":
        return "NodeLost"
    disruption = _most_recent_condition(status.get("conditions") or [], "DisruptionTarget")
    if disruption and disruption.get("status") == "True" and disruption.get("reason") in NODE_LOSS_DISRUPTION_REASONS:
        return disruption["reason"]
    return None


def pod_container_exit_reason(pod: dict) -> str | None:
    """Return ``ExitCode:<n>`` for the first non-zero container exit."""
    for container_status in (pod.get("status") or {}).get("containerStatuses") or []:
        terminated = (container_status.get("state") or {}).get("terminated") or {}
        exit_code = terminated.get("exitCode")
        if isinstance(exit_code, int) and exit_code != 0:
            return f"ExitCode:{exit_code}"
    return None
