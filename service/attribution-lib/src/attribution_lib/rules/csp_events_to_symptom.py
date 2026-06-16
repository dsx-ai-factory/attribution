# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional

ACTIONABLE_CATEGORIES = frozenset(
    {
        "HOST_FAILURE",
        "URGENT_REPAIR",
        "GUEST_PANIC",
        "EMERGENT_MAINTENANCE",
    }
)

SKIP_CATEGORIES = frozenset(
    {
        "CUSTOMER_REPORTED_FAULT",
        "PLANNED_MAINTENANCE",
        "CUSTOMER_TRIGGERED_MAINTENANCE",
    }
)

HOST_FAILURE_REASON_TO_SYMPTOM: dict[str, str] = {
    "BAD_GPU": "CSP Event GPU Fault",
    "BAD_NVSWITCH": "CSP Event NVSwitch Fault",
    "BAD_NIC": "CSP Event NIC Fault",
    "BAD_NV_TRUNK_LINK": "CSP Event NVLink Trunk Fault",
    "POWER_EVENT": "CSP Event Power Loss",
    "ARM_SERR": "CSP Event ARM System Error",
    "BAD_HARDWARE": "CSP Event Hardware Fault",
    "HYPERVISOR_FAILURE": "CSP Event Hypervisor Failure",
    "BAD_SOFTWARE": "CSP Event Software Fault",
    "BAD_CPU": "CSP Event CPU Fault",
    "UNKNOWN": "CSP Event Unrecognized Cause",
    "BAD_MEMORY": "CSP Event Memory Fault",
    "BAD_SSD": "CSP Event Storage Device Fault",
    "BAD_IPU": "CSP Event IPU Fault",
}

HEALTH_EVENT_GENERIC_SYMPTOM = "CSP Event Unrecognized Health Event"

HEALTH_EVENT_REASON_TO_SYMPTOM: dict[str, str] = {
    "CAPACITY_REDUCED": "CSP Event Capacity Reduced",
    "MAINTENANCE_COMPLETED": "CSP Event Maintenance Completed",
    "MAINTENANCE_REQUIRED": "CSP Event Maintenance Required",
    "IMMEDIATE_MAINTENANCE_REQUIRED": "CSP Event Immediate Maintenance Required",
}

# CSP event categories whose symptoms are investigation signals, not blame: the csp_event rule emits
# _interesting (not _bad) for these, so they reach s_interesting_nodes without driving the outcome.
INTERESTING_CATEGORIES = frozenset({"HEALTH_EVENT"})


def _symptom_from_category(category: str, reason: Optional[str]) -> Optional[str]:
    """Map category/reason to a specific symptom string.

    Args:
        category: CSP event category (e.g. HOST_FAILURE, GUEST_PANIC, HEALTH_EVENT).
        reason: Optional reason within the category (e.g. BAD_GPU, CAPACITY_REDUCED).

    Returns:
        Symptom string, or None if the category is neither actionable nor an interesting signal.
    """
    if category in SKIP_CATEGORIES:
        return None

    if category == "HOST_FAILURE":
        return HOST_FAILURE_REASON_TO_SYMPTOM.get(reason or "", "CSP Event Unrecognized Cause")

    if category == "HEALTH_EVENT":
        return HEALTH_EVENT_REASON_TO_SYMPTOM.get(reason or "", HEALTH_EVENT_GENERIC_SYMPTOM)

    if category == "URGENT_REPAIR":
        return "CSP Event Urgent Repair"

    if category == "GUEST_PANIC":
        return "CSP Event Guest OS Panic"

    if category == "EMERGENT_MAINTENANCE":
        return "CSP Event Emergency Maintenance"

    if category in ACTIONABLE_CATEGORIES:
        return "CSP Event Unrecognized Cause"

    return None


def get_symptom(category: str, reason: Optional[str], status: str) -> Optional[str]:
    """Map a CSP event (host termination or HEALTH_EVENT) to a FACT symptom string.

    Args:
        category: CSP event category (e.g. HOST_FAILURE, GUEST_PANIC, HEALTH_EVENT).
        reason: Optional reason within the category (e.g. BAD_GPU, CAPACITY_REDUCED).
        status: Classification status (e.g. CLASSIFICATION_PENDING, CLASSIFICATION_READY).

    Returns:
        Symptom string, or None if the event is neither actionable nor an interesting signal.
    """
    if category in SKIP_CATEGORIES:
        return None

    # PENDING events already carry category/reason from the provider's initial triage.
    # Use the specific symptom when available to reduce attribution delay from
    # ~74h to ~2h. Fall back to generic "pending classification" only when the
    # category/reason doesn't map to a known symptom.
    if status == "CLASSIFICATION_PENDING":
        specific = _symptom_from_category(category, reason)
        return specific if specific is not None else "CSP Event Classification Pending"

    return _symptom_from_category(category, reason)
