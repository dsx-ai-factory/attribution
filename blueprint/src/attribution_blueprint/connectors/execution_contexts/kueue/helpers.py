# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared helpers for Kueue Workloads, Jobs, JobSets, and Pods."""

from __future__ import annotations

from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import (
    OwnerIdentity,
    WorkloadIdentity,
)

POD_OWNER_LABEL = "batch.kubernetes.io/controller-uid"
POD_JOBSET_LABEL = "jobset.sigs.k8s.io/jobset-name"
WORKLOAD_ANNOTATION = "kueue.x-k8s.io/workload"
WORKLOAD_OWNER_UID_LABEL = "kueue.x-k8s.io/job-uid"

# PodGC sets this only after the Node API object is deleted. Broader
# DisruptionTarget reasons such as DeletionByTaintManager are intentionally
# excluded because they can represent transient unreachability or routine
# eviction rather than node loss.
NODE_LOSS_DISRUPTION_REASONS = frozenset({"DeletionByPodGC"})


def controller_owner(resource: dict, kind: str | None = None) -> dict | None:
    """Return the single matching controller owner, rejecting ambiguity."""
    owners = [
        owner
        for owner in (resource.get("metadata") or {}).get("ownerReferences") or []
        if owner.get("controller", True) and (kind is None or owner.get("kind") == kind)
    ]
    return owners[0] if len(owners) == 1 else None


def pod_selector_for(workload: dict) -> tuple[str, str] | None:
    """Return the label selector that identifies Pods for a supported owner."""
    owner = controller_owner(workload)
    if owner is None:
        return None
    if owner.get("kind") == "Job" and owner.get("uid"):
        return POD_OWNER_LABEL, owner["uid"]
    if owner.get("kind") == "JobSet" and owner.get("name"):
        return POD_JOBSET_LABEL, owner["name"]
    return None


def jobs_for_workload(workload: dict, owner_jobs: dict[str, dict]) -> list[dict]:
    """Return direct or JobSet child Jobs associated with a Workload."""
    owner = controller_owner(workload)
    if owner is None:
        return []
    namespace = (workload.get("metadata") or {}).get("namespace") or ""
    owner_name = owner.get("name")
    if not owner_name:
        return []
    if owner.get("kind") == "Job":
        job = owner_jobs.get(f"{namespace}/{owner_name}")
        return [job] if job is not None else []
    if owner.get("kind") == "JobSet":
        return [
            job
            for job in owner_jobs.values()
            if (job.get("metadata") or {}).get("namespace") == namespace
            and ((job.get("metadata") or {}).get("labels") or {}).get(POD_JOBSET_LABEL) == owner_name
        ]
    return []


def owner_job_uids(workload: dict, owner_jobs: dict[str, dict]) -> set[str]:
    """Return exact direct/child Job UIDs belonging to one Workload."""
    owner = controller_owner(workload)
    if owner is None:
        return set()
    if owner.get("kind") == "Job" and owner.get("uid"):
        return {owner["uid"]}
    if owner.get("kind") == "JobSet":
        return {
            uid for job in jobs_for_workload(workload, owner_jobs) if (uid := (job.get("metadata") or {}).get("uid"))
        }
    return set()


def workload_matches_owner(workload: dict, owner: OwnerIdentity) -> bool:
    """Return whether a Workload is authoritatively linked to ``owner``."""
    metadata = workload.get("metadata") or {}
    ref = controller_owner(workload)
    if ref and ref.get("kind") == owner.kind and ref.get("uid") == owner.uid:
        return True
    return (metadata.get("labels") or {}).get(WORKLOAD_OWNER_UID_LABEL) == owner.uid


def _most_recent_condition(conditions: list[dict], cond_type: str) -> dict | None:
    """Return the most recent (by lastTransitionTime) condition of `cond_type`."""
    matches = [c for c in conditions if c.get("type") == cond_type]
    if not matches:
        return None
    return max(matches, key=lambda c: c.get("lastTransitionTime", ""))


def pod_node_failure_reason(pod: dict) -> str | None:
    """Return the node-loss reason a Pod's status reports, or None.

    Checks the legacy ``NodeLost`` status and PodGC's ``DisruptionTarget``
    condition. The adapter and recorder share this helper so their
    classification cannot drift.
    """
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


def identities_from(workload: dict, owner_resource: dict) -> tuple[WorkloadIdentity, OwnerIdentity] | None:
    """Build validated identities from Kubernetes dictionary representations."""
    workload_metadata = workload.get("metadata") or {}
    owner_metadata = owner_resource.get("metadata") or {}
    workload_name = workload_metadata.get("name")
    workload_uid = workload_metadata.get("uid")
    owner_name = owner_metadata.get("name")
    owner_uid = owner_metadata.get("uid")
    owner_kind = owner_resource.get("kind")
    owner_api_version = owner_resource.get("apiVersion")
    if not all((workload_name, workload_uid, owner_name, owner_uid, owner_kind, owner_api_version)):
        return None
    owner = OwnerIdentity(owner_api_version, owner_kind, owner_name, owner_uid)
    if not workload_matches_owner(workload, owner):
        return None
    return WorkloadIdentity(workload_name, workload_uid), owner
