# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from attribution_blueprint.connectors.execution_contexts.kai.helpers import (
    POD_GROUP_ANNOTATION,
    top_owner,
)
from attribution_blueprint.kubernetes.daemon.models import ResourceIdentity
from attribution_blueprint.kubernetes.daemon.resolvers import ResolutionPending, UnsupportedOwner


class KaiResourceResolver:
    """Resolve the KAI PodGroup associated with an owned Pod."""

    def __init__(self, custom_api, *, pod_group_version: str = "v2alpha2"):
        self.custom_api = custom_api
        self.pod_group_version = pod_group_version

    def resolve(self, pod: dict, owner: ResourceIdentity) -> tuple[ResourceIdentity, ...]:
        from kubernetes.client.exceptions import ApiException

        metadata = pod.get("metadata") or {}
        namespace = metadata.get("namespace")
        pod_group_name = (metadata.get("annotations") or {}).get(POD_GROUP_ANNOTATION)
        if not namespace or not pod_group_name:
            raise UnsupportedOwner("Pod has no KAI PodGroup association")
        try:
            pod_group = self.custom_api.get_namespaced_custom_object(
                "scheduling.run.ai",
                self.pod_group_version,
                namespace,
                "podgroups",
                pod_group_name,
            )
        except ApiException as error:
            if error.status == 404:
                raise ResolutionPending("KAI PodGroup is not visible yet") from error
            raise

        pod_group_metadata = pod_group.get("metadata") or {}
        controller = top_owner(
            pod_group_metadata,
            require_complete=False,
        ) or _controller_owner(pod_group)
        if controller is None or controller.get("kind") != owner.kind or controller.get("uid") != owner.uid:
            raise UnsupportedOwner("KAI PodGroup does not belong to the resolved Pod owner")
        if not pod_group_metadata.get("name") or not pod_group_metadata.get("uid"):
            raise ResolutionPending("KAI PodGroup identity is incomplete")
        return (
            ResourceIdentity(
                api_version=pod_group.get("apiVersion")
                or f"scheduling.run.ai/{self.pod_group_version}",
                kind=pod_group.get("kind") or "PodGroup",
                namespace=pod_group_metadata.get("namespace") or namespace,
                name=pod_group_metadata["name"],
                uid=pod_group_metadata["uid"],
            ),
        )


def _controller_owner(resource: dict) -> dict | None:
    owners = [
        owner
        for owner in (resource.get("metadata") or {}).get("ownerReferences") or []
        if owner.get("controller", True)
    ]
    return owners[0] if len(owners) == 1 else None
