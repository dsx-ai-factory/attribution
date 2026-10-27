# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from attribution_blueprint.connectors.auth import Auth
from attribution_blueprint.kubernetes.daemon.models import PlacementRecord


class KubernetesDaemonClient:
    """Read retained Kubernetes evidence from KubernetesDaemon."""

    def __init__(self, endpoint: str, *, auth: Auth | None = None, timeout_seconds: float = 10.0):
        if not endpoint:
            raise ValueError("endpoint must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=3,
                connect=3,
                read=3,
                status=3,
                backoff_factor=0.25,
                status_forcelist=(429, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
            )
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        if auth is not None:
            kwargs = auth.request_kwargs()
            if "headers" in kwargs:
                self.session.headers.update(kwargs["headers"])
            if "cert" in kwargs:
                self.session.cert = kwargs["cert"]
            if "verify" in kwargs:
                self.session.verify = kwargs["verify"]

    def list_placements(
        self,
        namespace: str,
        *,
        pod_uid: str | None = None,
        owner_uid: str | None = None,
        resource_uid: str | None = None,
    ) -> list[PlacementRecord]:
        if sum(bool(value) for value in (pod_uid, owner_uid, resource_uid)) != 1:
            raise ValueError("exactly one of pod_uid, owner_uid, or resource_uid is required")
        params = {
            "namespace": namespace,
            **({"pod_uid": pod_uid} if pod_uid else {}),
            **({"owner_uid": owner_uid} if owner_uid else {}),
            **({"resource_uid": resource_uid} if resource_uid else {}),
        }
        response = self.session.get(
            f"{self.endpoint}/v1/placements",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("daemon response must contain an items list")

        records = [PlacementRecord.from_resource(item) for item in payload["items"]]
        if any(record.namespace != namespace for record in records):
            raise ValueError("daemon returned a placement outside the requested namespace")
        if pod_uid and any(record.pod_uid != pod_uid for record in records):
            raise ValueError("daemon returned a placement outside the requested Pod")
        if owner_uid and any(record.owner is None or record.owner.uid != owner_uid for record in records):
            raise ValueError("daemon returned a placement outside the requested owner")
        if resource_uid and any(
            not any(resource.uid == resource_uid for resource in record.associated_resources)
            for record in records
        ):
            raise ValueError("daemon returned a placement outside the requested resource")
        return records

    def list_for_pod(self, namespace: str, pod_uid: str) -> list[PlacementRecord]:
        return self.list_placements(namespace, pod_uid=pod_uid)

    def list_for_owner(self, namespace: str, owner_uid: str) -> list[PlacementRecord]:
        return self.list_placements(namespace, owner_uid=owner_uid)

    def list_for_resource(self, namespace: str, resource_uid: str) -> list[PlacementRecord]:
        return self.list_placements(namespace, resource_uid=resource_uid)
