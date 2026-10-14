# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import requests

from attribution_blueprint.connectors.auth import Auth
from attribution_blueprint.kubernetes.daemon.client import KubernetesDaemonClient
from attribution_blueprint.kubernetes.daemon.models import PlacementRecord, ResourceIdentity


class KueueDaemonClient:
    """Read from KubernetesDaemon with fallback to the legacy recorder API."""

    def __init__(self, endpoint: str, *, auth: Auth | None = None, timeout_seconds: float = 10.0):
        self.daemon = KubernetesDaemonClient(endpoint, auth=auth, timeout_seconds=timeout_seconds)

    def list_for_resource(self, namespace: str, resource_uid: str) -> list[PlacementRecord]:
        try:
            return self.daemon.list_for_resource(namespace, resource_uid)
        except requests.HTTPError as error:
            # The legacy endpoint rejects `resource_uid` with HTTP 400 because
            # it requires `workload_uid`. Other failures are real outages and
            # must not be hidden by a compatibility retry.
            if error.response is None or error.response.status_code != 400:
                raise
        response = self.daemon.session.get(
            f"{self.daemon.endpoint}/v1/placements",
            params={"namespace": namespace, "workload_uid": resource_uid},
            timeout=self.daemon.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("legacy recorder response must contain an items list")
        records = [_legacy_record(item) for item in payload["items"]]
        if any(record.namespace != namespace for record in records):
            raise ValueError("legacy recorder returned a placement outside the requested namespace")
        if any(not any(resource.uid == resource_uid for resource in record.associated_resources) for record in records):
            raise ValueError("legacy recorder returned a placement outside the requested Workload")
        return records


def _legacy_record(resource: dict) -> PlacementRecord:
    metadata = resource["metadata"]
    spec = resource["spec"]
    workload = spec["workload"]
    owner = spec["owner"]
    pod = spec["pod"]
    node = spec["node"]
    namespace = metadata["namespace"]
    return PlacementRecord(
        namespace=namespace,
        pod_name=pod["name"],
        pod_uid=pod["uid"],
        node_name=node["name"],
        observed_at=spec["observedAt"],
        scheduled_at=spec.get("scheduledAt"),
        node_failure_reason=spec.get("nodeFailureReason"),
        owner=ResourceIdentity(
            api_version=owner["apiVersion"],
            kind=owner["kind"],
            namespace=namespace,
            name=owner["name"],
            uid=owner["uid"],
        ),
        associated_resources=(
            ResourceIdentity(
                api_version="kueue.x-k8s.io/v1beta1",
                kind="Workload",
                namespace=namespace,
                name=workload["name"],
                uid=workload["uid"],
            ),
        ),
    )
