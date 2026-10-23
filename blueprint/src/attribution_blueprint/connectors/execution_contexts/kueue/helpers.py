# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def pod_selector_for(workload: dict) -> tuple[str, str, str] | None:
    """Return the `(namespace, label-key, label-value)` selector that
    identifies Pods for a supported owner.

    Namespace is required alongside the label: the JobSet pod label
    carries only a name, which is unique per namespace but not
    cluster-wide, and cluster-wide discovery can pool pods from every
    namespace before filtering. Always match via `pod_matches_selector`.
    """
    owner = controller_owner(workload)
    if owner is None:
        return None
    namespace = (workload.get("metadata") or {}).get("namespace") or "default"
    if owner.get("kind") == "Job" and owner.get("uid"):
        return namespace, POD_OWNER_LABEL, owner["uid"]
    if owner.get("kind") == "JobSet" and owner.get("name"):
        return namespace, POD_JOBSET_LABEL, owner["name"]
    return None


def pod_matches_selector(pod: dict, selector: tuple[str, str, str]) -> bool:
    """Return whether `pod` matches a `pod_selector_for` selector."""
    namespace, label_key, label_value = selector
    metadata = pod.get("metadata") or {}
    if metadata.get("namespace") != namespace:
        return False
    return (metadata.get("labels") or {}).get(label_key) == label_value


def _jobset_owner_uid(job: dict) -> str | None:
    """Return the uid of the JobSet that owns `job`, from `job`'s own
    ownerReferences, or None if `job` has no JobSet controller reference."""
    for owner in (job.get("metadata") or {}).get("ownerReferences") or []:
        if owner.get("kind") == "JobSet" and owner.get("uid"):
            return owner["uid"]
    return None


def jobs_for_workload(workload: dict, owner_jobs: dict[str, dict]) -> list[dict]:
    """Return direct or JobSet child Jobs associated with a Workload.

    Every returned Job's identity is confirmed by uid, not just the name
    used to look it up: names are reusable after deletion, so a
    namespace/name (or `jobset-name` label) lookup can return a different
    object than the one the ownerReference originally pointed to. A uid
    mismatch is treated as stale evidence, not a valid owner.
    """
    owner = controller_owner(workload)
    if owner is None:
        return []
    namespace = (workload.get("metadata") or {}).get("namespace") or ""
    owner_name = owner.get("name")
    owner_uid = owner.get("uid")
    if not owner_name:
        return []
    if owner.get("kind") == "Job":
        job = owner_jobs.get(f"{namespace}/{owner_name}")
        if job is None:
            return []
        if (job.get("metadata") or {}).get("uid") != owner_uid:
            return []
        return [job]
    if owner.get("kind") == "JobSet":
        return [
            job
            for job in owner_jobs.values()
            if (job.get("metadata") or {}).get("namespace") == namespace
            and ((job.get("metadata") or {}).get("labels") or {}).get(POD_JOBSET_LABEL) == owner_name
            and _jobset_owner_uid(job) == owner_uid
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
