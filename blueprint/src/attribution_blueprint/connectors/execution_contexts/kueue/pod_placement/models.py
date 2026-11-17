# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PlacementRecord's small, transport-independent representation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypedDict

API_GROUP = "attribution.nvidia.com"
API_VERSION = "v1alpha1"
KIND = "PlacementRecord"
PLURAL = "placementrecords"
WORKLOAD_UID_LABEL = f"{API_GROUP}/workload-uid"
MANAGED_BY_LABEL = f"{API_GROUP}/managed-by"


class PodPlacement(TypedDict):
    pod_uid: str
    pod_name: str
    node_name: str
    source: str


@dataclass(frozen=True)
class PlacementQuery:
    """One Workload's inputs to a shared historical-placement query."""

    workload_uid: str
    namespace: str
    owner_job_uids: frozenset[str]
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True)
class WorkloadIdentity:
    name: str
    uid: str


@dataclass(frozen=True)
class OwnerIdentity:
    api_version: str
    kind: str
    name: str
    uid: str


@dataclass(frozen=True)
class PlacementRecord:
    namespace: str
    workload: WorkloadIdentity
    owner: OwnerIdentity
    pod_name: str
    pod_uid: str
    node_name: str
    observed_at: str
    resolver_source: str
    scheduled_at: str | None = None
    # Node-loss/container-failure evidence read from the Pod's live status at
    # observation time (e.g. "NodeLost", "DeletionByPodGC", "ExitCode:137").
    # None means no such evidence was present *yet* — the recorder updates an
    # existing record in place when a later watch event reports one, since
    # this is the one field expected to change over a Pod's lifetime. See
    # `kubernetes_helpers.pod_node_failure_reason` / `helpers.pod_container_exit_reason`.
    node_failure_reason: str | None = None

    @property
    def name(self) -> str:
        return f"pod-{self.pod_uid}"

    def to_resource(self) -> dict:
        spec = {
            "workload": {"name": self.workload.name, "uid": self.workload.uid},
            "owner": {
                "apiVersion": self.owner.api_version,
                "kind": self.owner.kind,
                "name": self.owner.name,
                "uid": self.owner.uid,
            },
            "pod": {"name": self.pod_name, "uid": self.pod_uid},
            "node": {"name": self.node_name},
            "observedAt": self.observed_at,
            "resolverSource": self.resolver_source,
        }
        if self.scheduled_at:
            spec["scheduledAt"] = self.scheduled_at
        if self.node_failure_reason:
            spec["nodeFailureReason"] = self.node_failure_reason
        return {
            "apiVersion": f"{API_GROUP}/{API_VERSION}",
            "kind": KIND,
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": {
                    WORKLOAD_UID_LABEL: self.workload.uid,
                    MANAGED_BY_LABEL: "pod-placement-recorder",
                },
            },
            "spec": spec,
        }

    @classmethod
    def from_resource(cls, resource: dict) -> PlacementRecord:
        metadata = resource["metadata"]
        spec = resource["spec"]
        workload = spec["workload"]
        owner = spec["owner"]
        pod = spec["pod"]
        node = spec["node"]
        return cls(
            namespace=metadata["namespace"],
            workload=WorkloadIdentity(workload["name"], workload["uid"]),
            owner=OwnerIdentity(owner["apiVersion"], owner["kind"], owner["name"], owner["uid"]),
            pod_name=pod["name"],
            pod_uid=pod["uid"],
            node_name=node["name"],
            observed_at=spec["observedAt"],
            resolver_source=spec["resolverSource"],
            scheduled_at=spec.get("scheduledAt"),
            node_failure_reason=spec.get("nodeFailureReason"),
        )

    @classmethod
    def for_pod(
        cls,
        pod: dict,
        workload: WorkloadIdentity,
        owner: OwnerIdentity,
        resolver_source: str,
        node_failure_reason: str | None = None,
    ) -> PlacementRecord:
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
            workload=workload,
            owner=owner,
            pod_name=metadata["name"],
            pod_uid=metadata["uid"],
            node_name=node_name,
            observed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            resolver_source=resolver_source,
            scheduled_at=scheduled_at,
            node_failure_reason=node_failure_reason,
        )
