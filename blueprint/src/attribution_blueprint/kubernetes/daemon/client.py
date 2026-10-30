# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime

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

    def list_for_resource(self, namespace: str, resource_uid: str) -> list[PlacementRecord]:
        response = self.session.get(
            f"{self.endpoint}/v1/placements",
            params={"namespace": namespace, "resource_uid": resource_uid},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("daemon response must contain an items list")

        records = [PlacementRecord.from_resource(item) for item in payload["items"]]
        if any(record.namespace != namespace for record in records):
            raise ValueError("daemon returned a placement outside the requested namespace")
        if any(
            not any(resource.uid == resource_uid for resource in record.associated_resources)
            for record in records
        ):
            raise ValueError("daemon returned a placement outside the requested resource")
        return records

    def find_histories(
        self,
        namespace: str,
        kind: str,
        *,
        name: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict]:
        """Find retained controller histories containing a resource kind/name."""
        if not namespace:
            raise ValueError("namespace must not be empty")
        if not kind:
            raise ValueError("kind must not be empty")
        if start is not None and start.tzinfo is None:
            raise ValueError("start must be timezone-aware")
        if end is not None and end.tzinfo is None:
            raise ValueError("end must be timezone-aware")
        if start is not None and end is not None and start > end:
            raise ValueError("start must not be after end")
        params = {
            "namespace": namespace,
            "kind": kind,
        }
        if name:
            params["name"] = name
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()
        response = self.session.get(
            f"{self.endpoint}/v1/history",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(
            payload.get("items"),
            list,
        ):
            raise ValueError(
                "daemon histories response must contain an items list"
            )
        histories = payload["items"]
        for history in histories:
            if not isinstance(history, dict):
                raise ValueError("daemon histories items must be objects")
            if history.get("namespace") != namespace:
                raise ValueError(
                    "daemon returned history outside the requested namespace"
                )
            if not isinstance(history.get("observations"), list):
                raise ValueError(
                    "daemon history response must contain an observations list"
                )
            if "placements" in history and not isinstance(
                history["placements"],
                list,
            ):
                raise ValueError(
                    "daemon history response placements must be a list"
                )
            if "coverage" in history and not isinstance(
                history["coverage"],
                dict,
            ):
                raise ValueError(
                    "daemon history response coverage must be an object"
                )
        return histories
