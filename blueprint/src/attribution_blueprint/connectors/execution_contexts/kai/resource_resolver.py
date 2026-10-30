# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from attribution_blueprint.connectors.execution_contexts.kai.helpers import (
    POD_GROUP_ANNOTATION,
    owner_matches,
    resolve_owner,
    top_owner,
)
from attribution_blueprint.kubernetes.daemon.models import (
    ResourceIdentity,
    ResourceObservation,
)
from attribution_blueprint.kubernetes.daemon.observation_queries import (
    controller_owner_reference,
    latest_observations_for_kind,
    observation_identity,
)
from attribution_blueprint.kubernetes.daemon.resolvers import ResolutionPending, UnsupportedOwner
from attribution_blueprint.kubernetes.daemon.store import ObservationJournal


class KaiResourceResolver:
    """Resolve the KAI PodGroup associated with an owned Pod."""

    def __init__(
        self,
        custom_api,
        *,
        pod_group_version: str = "v2alpha2",
        journal: ObservationJournal | None = None,
        cluster_id: str = "default",
    ):
        self.custom_api = custom_api
        self.pod_group_version = pod_group_version
        self.journal = journal
        self.cluster_id = cluster_id

    def resolve(self, pod: dict, owner: ResourceIdentity) -> tuple[ResourceIdentity, ...]:
        from kubernetes.client.exceptions import ApiException

        metadata = pod.get("metadata") or {}
        namespace = metadata.get("namespace")
        pod_group_name = (metadata.get("annotations") or {}).get(POD_GROUP_ANNOTATION)
        if not namespace or not pod_group_name:
            raise UnsupportedOwner("Pod has no KAI PodGroup association")
        retained = self._resolve_from_journal(namespace, pod_group_name, owner)
        if retained is not None:
            return (retained,)
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
        if not owner_matches(resolve_owner(pod_group_metadata), owner):
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

    def _resolve_from_journal(
        self,
        namespace: str,
        pod_group_name: str,
        owner: ResourceIdentity,
    ) -> ResourceIdentity | None:
        if self.journal is None:
            return None
        named = latest_observations_for_kind(
            self.journal,
            self.cluster_id,
            namespace,
            "PodGroup",
            name=pod_group_name,
        )
        matches = [
            observation
            for observation in named
            if _observation_matches_owner(observation, owner)
        ]
        if len(matches) == 1:
            return observation_identity(matches[0])
        if len(matches) > 1:
            raise UnsupportedOwner("multiple retained KAI PodGroups match the Pod owner")
        if named:
            raise UnsupportedOwner("retained KAI PodGroup does not belong to the resolved Pod owner")
        return None


def _observation_matches_owner(
    observation: ResourceObservation,
    owner: ResourceIdentity,
) -> bool:
    selected_metadata = observation.selected_metadata or {}
    controller = top_owner(
        {"annotations": selected_metadata.get("annotations") or {}}
    )
    if controller is None:
        controller = controller_owner_reference(observation)
    return owner_matches(controller, owner)
