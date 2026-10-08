# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Pod-level helpers shared by Kubernetes execution-context adapters."""

from __future__ import annotations

# PodGC sets this only after the Node API object is deleted. Broader
# DisruptionTarget reasons such as DeletionByTaintManager are intentionally
# excluded because they can represent transient unreachability or routine
# eviction rather than node loss.
NODE_LOSS_DISRUPTION_REASONS = frozenset({"DeletionByPodGC"})


def _most_recent_condition(conditions: list[dict], cond_type: str) -> dict | None:
    """Return the most recent (by lastTransitionTime) condition of `cond_type`."""
    matches = [c for c in conditions if c.get("type") == cond_type]
    if not matches:
        return None
    return max(matches, key=lambda c: c.get("lastTransitionTime", ""))


def pod_node_failure_reason(pod: dict) -> str | None:
    """Return the node-loss reason a Pod's status reports, or None.

    Checks the legacy ``NodeLost`` status and PodGC's ``DisruptionTarget``
    condition. Adapters and recorders share this helper so their
    classification cannot drift.
    """
    status = pod.get("status") or {}
    if status.get("reason") == "NodeLost":
        return "NodeLost"
    disruption = _most_recent_condition(status.get("conditions") or [], "DisruptionTarget")
    if disruption and disruption.get("status") == "True" and disruption.get("reason") in NODE_LOSS_DISRUPTION_REASONS:
        return disruption["reason"]
    return None
