# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.connectors.execution_contexts.kueue.helpers import (
    WORKLOAD_ANNOTATION,
    WORKLOAD_OWNER_UID_LABEL,
    workload_matches_owner,
)
from attribution_blueprint.kubernetes.daemon.models import (
    ResourceIdentity,
    ResourceObservation,
)
from attribution_blueprint.kubernetes.daemon.observation_queries import (
    latest_observations_for_kind,
    observation_identity,
)
from attribution_blueprint.kubernetes.daemon.resolvers import ResolutionPending, UnsupportedOwner
from attribution_blueprint.kubernetes.daemon.store import ObservationJournal


class KueueResourceResolver:
    """Resolve the Kueue Workload resource related to an owned Pod."""

    def __init__(
        self,
        custom_api,
        *,
        workload_version: str = "v1beta1",
        journal: ObservationJournal | None = None,
        cluster_id: str = "default",
    ):
        self.custom_api = custom_api
        self.workload_version = workload_version
        self.journal = journal
        self.cluster_id = cluster_id

    def resolve(self, pod: dict, owner: ResourceIdentity) -> tuple[ResourceIdentity, ...]:
        from kubernetes.client.exceptions import ApiException

        metadata = pod.get("metadata") or {}
        namespace = metadata.get("namespace")
        if not namespace:
            raise UnsupportedOwner("Pod namespace is unavailable")

        workload_name = (metadata.get("annotations") or {}).get(WORKLOAD_ANNOTATION)
        retained = self._resolve_from_journal(
            namespace,
            workload_name,
            owner,
        )
        if retained is not None:
            return (retained,)
        try:
            if workload_name:
                candidates = [
                    self.custom_api.get_namespaced_custom_object(
                        "kueue.x-k8s.io",
                        self.workload_version,
                        namespace,
                        "workloads",
                        workload_name,
                    )
                ]
            else:
                response = self.custom_api.list_namespaced_custom_object(
                    "kueue.x-k8s.io",
                    self.workload_version,
                    namespace,
                    "workloads",
                    label_selector=f"{WORKLOAD_OWNER_UID_LABEL}={owner.uid}",
                )
                candidates = response.get("items") or []
        except ApiException as error:
            if error.status == 404:
                raise ResolutionPending("Kueue Workload is not visible yet") from error
            raise

        matches = [workload for workload in candidates if _matches_owner(workload, owner)]
        if not matches:
            raise ResolutionPending("matching Kueue Workload is not visible yet")
        if len(matches) != 1:
            raise UnsupportedOwner("multiple Kueue Workloads match the Pod owner")

        workload = matches[0]
        workload_metadata = workload.get("metadata") or {}
        if not workload_metadata.get("name") or not workload_metadata.get("uid"):
            raise ResolutionPending("Kueue Workload identity is incomplete")
        return (
            ResourceIdentity(
                api_version=workload.get("apiVersion") or f"kueue.x-k8s.io/{self.workload_version}",
                kind=workload.get("kind") or "Workload",
                namespace=workload_metadata.get("namespace") or namespace,
                name=workload_metadata["name"],
                uid=workload_metadata["uid"],
            ),
        )

    def _resolve_from_journal(
        self,
        namespace: str,
        workload_name: str | None,
        owner: ResourceIdentity,
    ) -> ResourceIdentity | None:
        if self.journal is None:
            return None
        named = latest_observations_for_kind(
            self.journal,
            self.cluster_id,
            namespace,
            "Workload",
            name=workload_name,
        )
        matches = [
            observation
            for observation in named
            if _observation_matches_owner(observation, owner)
        ]
        if len(matches) == 1:
            return observation_identity(matches[0])
        if len(matches) > 1:
            raise UnsupportedOwner("multiple retained Kueue Workloads match the Pod owner")
        if workload_name is not None and named:
            raise UnsupportedOwner("retained Kueue Workload does not belong to the resolved Pod owner")
        return None


def _matches_owner(workload: dict, owner: ResourceIdentity) -> bool:
    return workload_matches_owner(workload, owner)


def _observation_matches_owner(
    observation: ResourceObservation,
    owner: ResourceIdentity,
) -> bool:
    if any(
        reference.controller
        and reference.kind == owner.kind
        and reference.uid == owner.uid
        for reference in observation.owner_references
    ):
        return True
    selected_metadata = observation.selected_metadata or {}
    return (selected_metadata.get("labels") or {}).get(WORKLOAD_OWNER_UID_LABEL) == owner.uid
