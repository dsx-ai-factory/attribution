# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Kueue Workloads, Jobs, JobSets, and Pods."""

from __future__ import annotations

from attribution_blueprint.kubernetes.daemon.models import ResourceIdentity

POD_OWNER_LABEL = "batch.kubernetes.io/controller-uid"
POD_JOBSET_LABEL = "jobset.sigs.k8s.io/jobset-name"
WORKLOAD_ANNOTATION = "kueue.x-k8s.io/workload"
WORKLOAD_OWNER_UID_LABEL = "kueue.x-k8s.io/job-uid"


def controller_owner(resource: dict, kind: str | None = None) -> dict | None:
    """Return the single matching controller owner, rejecting ambiguity."""
    owners = [
        owner
        for owner in (resource.get("metadata") or {}).get("ownerReferences") or []
        if owner.get("controller", True) and (kind is None or owner.get("kind") == kind)
    ]
    return owners[0] if len(owners) == 1 else None


def workload_matches_owner(workload: dict, owner: ResourceIdentity) -> bool:
    """Return whether a Workload is authoritatively linked to ``owner``."""
    metadata = workload.get("metadata") or {}
    ref = controller_owner(workload)
    if ref and ref.get("kind") == owner.kind and ref.get("uid") == owner.uid:
        return True
    return (metadata.get("labels") or {}).get(WORKLOAD_OWNER_UID_LABEL) == owner.uid
