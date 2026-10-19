# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared KAI PodGroup ownership helpers."""

from __future__ import annotations

import yaml

from attribution_blueprint.kubernetes.daemon.models import (
    OwnerReference,
    ResourceIdentity,
)

POD_GROUP_ANNOTATION = "pod-group-name"
TOP_OWNER_ANNOTATION = "kai.scheduler/top-owner-metadata"


def top_owner(metadata: dict) -> dict | None:
    """Parse KAI's top-owner annotation when it carries a complete identity."""
    raw = (metadata.get("annotations") or {}).get(TOP_OWNER_ANNOTATION)
    if not raw:
        return None
    try:
        owner = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    if not isinstance(owner, dict):
        return None
    return owner if all(owner.get(field) for field in ("kind", "name", "uid")) else None


def controller_owner(metadata: dict) -> dict | None:
    """Return the first complete Kubernetes controller owner reference."""
    references = [
        reference
        for reference in metadata.get("ownerReferences") or []
        if isinstance(reference, dict)
    ]
    for reference in references:
        if (
            reference.get("controller")
            and reference.get("kind")
            and reference.get("name")
            and reference.get("uid")
        ):
            return reference
    return None


def resolve_owner(metadata: dict) -> dict | None:
    """Resolve KAI's annotated owner, falling back to its controller."""
    return top_owner(metadata) or controller_owner(metadata)


def owner_matches(
    reference: dict | OwnerReference | None,
    owner: ResourceIdentity,
) -> bool:
    """Return whether an owner reference identifies the resolved owner."""
    if reference is None:
        return False
    kind = reference.kind if isinstance(reference, OwnerReference) else reference.get("kind")
    uid = reference.uid if isinstance(reference, OwnerReference) else reference.get("uid")
    return kind == owner.kind and uid == owner.uid
