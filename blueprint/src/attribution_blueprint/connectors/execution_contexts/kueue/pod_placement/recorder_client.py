# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from attribution_blueprint.connectors.auth import Auth
from attribution_blueprint.connectors.execution_contexts.kueue.pod_placement.models import PlacementRecord


class RecorderPlacementClient:
    """Read placement records from a recorder's HTTP API."""

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

    def list_for_workload(self, namespace: str, workload_uid: str) -> list[PlacementRecord]:
        response = self.session.get(
            f"{self.endpoint}/v1/placements",
            params={"namespace": namespace, "workload_uid": workload_uid},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("recorder response must contain an items list")

        records = [PlacementRecord.from_resource(item) for item in payload["items"]]
        if any(record.namespace != namespace or record.workload.uid != workload_uid for record in records):
            raise ValueError("recorder returned a placement outside the requested workload")
        return records
