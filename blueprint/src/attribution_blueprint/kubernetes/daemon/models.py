# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-neutral models persisted and served by KubernetesDaemon."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Literal, TypedDict

API_GROUP = "attribution.nvidia.com"
API_VERSION = "v1alpha2"
KIND = "PlacementRecord"
PLURAL = "placementrecords"
MANAGED_BY_LABEL = f"{API_GROUP}/managed-by"


class PodPlacement(TypedDict):
    pod_uid: str
    pod_name: str
    node_name: str
    source: str


@dataclass(frozen=True)
class ResourceIdentity:
    api_version: str
    kind: str
    namespace: str
    name: str
    uid: str


ObservationEventType = Literal["SNAPSHOT", "ADDED", "MODIFIED", "DELETED"]
CoverageGapReason = Literal["RESOURCE_VERSION_EXPIRED", "WATCH_SOURCE_CHANGED"]


@dataclass(frozen=True)
class OwnerReference:
    api_version: str
    kind: str
    name: str
    uid: str
    controller: bool


@dataclass(frozen=True)
class WatchCoverage:
    cluster_id: str
    source_name: str
    api_version: str
    kind: str
    namespace: str
    coverage_started_at: str
    resource_version: str
    checkpointed_at: str


@dataclass(frozen=True)
class CoverageGap:
    id: int
    cluster_id: str
    source_name: str
    api_version: str
    kind: str
    namespace: str
    reason: CoverageGapReason
    start_resource_version: str
    detected_at: str
    end_resource_version: str | None
    recovered_at: str | None


@dataclass(frozen=True)
class ResourceObservation:
    """One durable Kubernetes resource version from a list or watch."""

    cluster_id: str
    api_version: str
    kind: str
    namespace: str
    name: str
    uid: str
    resource_version: str
    event_type: ObservationEventType
    observed_at: str
    owner_references: tuple[OwnerReference, ...] = ()
    selected_metadata: dict | None = None
    source_timestamp: str | None = None
    status: dict | None = None

    @classmethod
    def from_resource(
        cls,
        resource: dict,
        *,
        cluster_id: str,
        event_type: ObservationEventType,
        status: dict | None = None,
        selected_metadata: dict | None = None,
        source_timestamp: str | None = None,
        observed_at: str | None = None,
    ) -> ResourceObservation:
        metadata = resource.get("metadata") or {}
        required = {
            "cluster_id": cluster_id,
            "api_version": resource.get("apiVersion"),
            "kind": resource.get("kind"),
            "namespace": metadata.get("namespace"),
            "name": metadata.get("name"),
            "uid": metadata.get("uid"),
            "resource_version": metadata.get("resourceVersion"),
        }
        missing = [field for field, value in required.items() if not value]
        if missing:
            raise ValueError(f"resource observation is missing {', '.join(missing)}")
        return cls(
            cluster_id=cluster_id,
            api_version=required["api_version"],
            kind=required["kind"],
            namespace=required["namespace"],
            name=required["name"],
            uid=required["uid"],
            resource_version=required["resource_version"],
            event_type=event_type,
            observed_at=observed_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            owner_references=tuple(
                OwnerReference(
                    api_version=owner.get("apiVersion") or "",
                    kind=owner["kind"],
                    name=owner["name"],
                    uid=owner["uid"],
                    controller=bool(owner.get("controller", False)),
                )
                for owner in metadata.get("ownerReferences") or []
                if owner.get("kind") and owner.get("name") and owner.get("uid")
            ),
            selected_metadata=selected_metadata,
            source_timestamp=source_timestamp,
            status=status,
        )


@dataclass(frozen=True)
class PlacementRecord:
    namespace: str
    pod_name: str
    pod_uid: str
    node_name: str
    observed_at: str
    scheduled_at: str | None = None
    node_failure_reason: str | None = None
    owner_references: tuple[ResourceIdentity, ...] = ()
    owner: ResourceIdentity | None = None
    associated_resources: tuple[ResourceIdentity, ...] = ()

    @property
    def name(self) -> str:
        return f"pod-{self.pod_uid}"

    def enriched(
        self,
        *,
        scheduled_at: str | None = None,
        owner: ResourceIdentity | None = None,
        associated_resources: tuple[ResourceIdentity, ...] = (),
        node_failure_reason: str | None = None,
    ) -> PlacementRecord:
        resources = {resource.uid: resource for resource in self.associated_resources}
        resources.update({resource.uid: resource for resource in associated_resources})
        return replace(
            self,
            scheduled_at=self.scheduled_at or scheduled_at,
            owner=owner or self.owner,
            associated_resources=tuple(
                sorted(
                    resources.values(),
                    key=lambda item: (
                        item.uid,
                    ),
                )
            ),
            node_failure_reason=node_failure_reason or self.node_failure_reason,
        )

    def to_resource(self) -> dict:
        spec: dict = {
            "pod": {"name": self.pod_name, "uid": self.pod_uid},
            "node": {"name": self.node_name},
            "observedAt": self.observed_at,
            "ownerReferences": [_identity_to_dict(owner) for owner in self.owner_references],
            "associatedResources": [_identity_to_dict(resource) for resource in self.associated_resources],
        }
        if self.scheduled_at:
            spec["scheduledAt"] = self.scheduled_at
        if self.node_failure_reason:
            spec["nodeFailureReason"] = self.node_failure_reason
        if self.owner:
            spec["owner"] = _identity_to_dict(self.owner)

        return {
            "apiVersion": f"{API_GROUP}/{API_VERSION}",
            "kind": KIND,
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": {MANAGED_BY_LABEL: "kubernetes-daemon"},
            },
            "spec": spec,
        }

    @classmethod
    def from_resource(cls, resource: dict) -> PlacementRecord:
        metadata = resource["metadata"]
        spec = resource["spec"]
        pod = spec["pod"]
        node = spec["node"]
        associated_resources = tuple(
            _identity_from_dict(item, metadata["namespace"]) for item in spec.get("associatedResources") or []
        )

        owner = spec.get("owner")
        return cls(
            namespace=metadata["namespace"],
            pod_name=pod["name"],
            pod_uid=pod["uid"],
            node_name=node["name"],
            observed_at=spec["observedAt"],
            scheduled_at=spec.get("scheduledAt"),
            node_failure_reason=spec.get("nodeFailureReason"),
            owner_references=tuple(
                _identity_from_dict(item, metadata["namespace"]) for item in spec.get("ownerReferences") or []
            ),
            owner=_identity_from_dict(owner, metadata["namespace"]) if owner else None,
            associated_resources=associated_resources,
        )

    @classmethod
    def for_pod(cls, pod: dict, node_failure_reason: str | None = None) -> PlacementRecord:
        metadata = pod.get("metadata") or {}
        node_name = (pod.get("spec") or {}).get("nodeName")
        if not all((metadata.get("namespace"), metadata.get("name"), metadata.get("uid"), node_name)):
            raise ValueError("bound Pod is missing namespace, name, UID, or nodeName")
        scheduled_at = None
        for condition in (pod.get("status") or {}).get("conditions") or []:
            if condition.get("type") == "PodScheduled" and condition.get("status") == "True":
                scheduled_at = condition.get("lastTransitionTime")
                break
        return cls(
            namespace=metadata["namespace"],
            pod_name=metadata["name"],
            pod_uid=metadata["uid"],
            node_name=node_name,
            observed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            scheduled_at=scheduled_at,
            node_failure_reason=node_failure_reason,
            owner_references=tuple(
                _identity_from_dict(owner, metadata["namespace"])
                for owner in metadata.get("ownerReferences") or []
                if owner.get("uid") and owner.get("kind") and owner.get("name")
            ),
        )


def _identity_to_dict(identity: ResourceIdentity) -> dict:
    return {
        "apiVersion": identity.api_version,
        "kind": identity.kind,
        "namespace": identity.namespace,
        "name": identity.name,
        "uid": identity.uid,
    }


def _identity_from_dict(value: dict, default_namespace: str) -> ResourceIdentity:
    return ResourceIdentity(
        api_version=value.get("apiVersion") or "",
        kind=value["kind"],
        namespace=value.get("namespace") or default_namespace,
        name=value["name"],
        uid=value["uid"],
    )
