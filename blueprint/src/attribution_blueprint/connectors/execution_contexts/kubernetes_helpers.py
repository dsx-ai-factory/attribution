# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pod-level helpers shared by Kubernetes execution-context adapters."""

from __future__ import annotations

from datetime import datetime, timezone


def earliest_job_failure_reason(jobs: list[dict]) -> str | None:
    """Return the earliest Failed/FailureTarget reason across Jobs."""
    earliest: tuple[datetime, str] | None = None
    for job in jobs:
        conditions = (job.get("status") or {}).get("conditions") or []
        for condition_type in ("Failed", "FailureTarget"):
            matches = [c for c in conditions if (c or {}).get("type") == condition_type]
            if not matches:
                continue
            condition = max(matches, key=lambda c: c.get("lastTransitionTime", ""))
            if condition.get("status") != "True":
                continue
            reason = (condition.get("reason") or "").strip()
            transition = _parse_dt(condition.get("lastTransitionTime"))
            if reason and transition is not None and (earliest is None or transition < earliest[0]):
                earliest = transition, reason
    return earliest[1] if earliest else None


def job_time_limit_minutes(jobs: list[dict]) -> int | None:
    """Return the shortest positive activeDeadlineSeconds, rounded up to minutes."""
    deadlines = [
        deadline
        for job in jobs
        if isinstance((deadline := (job.get("spec") or {}).get("activeDeadlineSeconds")), int)
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
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
