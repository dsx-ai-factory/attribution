# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-neutral Kubernetes status helpers."""

from __future__ import annotations

NODE_LOSS_DISRUPTION_REASONS = frozenset({"DeletionByPodGC"})


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
