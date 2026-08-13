# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from datetime import datetime
from typing import NotRequired, TypedDict

import requests


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _serialize(payload: dict) -> str:
    return json.dumps(payload, default=_json_default)


_JSON_HEADERS = {"Content-Type": "application/json"}


# Mirrors the NVIDIA Attribution Service app/v2/models.py AttributorWorkloadInfoV2 /
# AttributorMetadata / AttributorInfoV2. Datetimes are `str` because BP emits ISO
# strings and the service parses them to AwareDatetime.
class AttributorWorkloadDict(TypedDict):
    id: str
    type: str
    job_start_time: str
    timezone: str
    status: str
    nodes: dict[str, list[int]]
    name: str
    username: str
    framework: str
    job_end_time: NotRequired[str]
    interest_start_time: NotRequired[str]
    interest_end_time: NotRequired[str]
    failed_node: NotRequired[str]
    cancelled_by: NotRequired[str]
    time_limit: NotRequired[int]
    srun_cmd: NotRequired[str]
    exit_code_signal: NotRequired[int]
    last_tflops_time: NotRequired[str]


class AttributorMetadataDict(TypedDict):
    cluster: str
    agent: str
    tenant: str
    ruleset: str


class AttributorInfoDict(TypedDict):
    workload: AttributorWorkloadDict
    metadata: AttributorMetadataDict
    parameters: NotRequired[dict]


class AttributionConnector:
    """Thin HTTP client for the NVIDIA Attribution Service v2 API."""

    def __init__(
        self,
        service_url: str,
        timeout_seconds: float = 60.0,
    ):
        self.service_url = service_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()

    def create_attributor(self, info: AttributorInfoDict) -> str:
        """POST /v2/attributor with JSON body. Returns attributor_id."""
        url = f"{self.service_url}/v2/attributor"
        response = self._session.post(url, data=_serialize(info), headers=_JSON_HEADERS, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()["attributor_id"]

    def create_or_replace_attributor(self, info: AttributorInfoDict) -> str:
        """Create an attributor, replacing the stale service record on 409.

        The service returns the existing attributor id as `att_id` on conflict.
        Keeping that HTTP-specific recovery here prevents callers from knowing
        the transport error type or response envelope.
        """
        try:
            return self.create_attributor(info)
        except requests.HTTPError as e:
            resp = e.response
            if resp is None or resp.status_code != 409:
                raise
            try:
                existing = (resp.json() or {}).get("att_id")
            except ValueError:
                existing = None
            if not existing:
                raise
            self.delete_attributor(existing)
            return self.create_attributor(info)

    def ingest_observation(self, attributor_id: str, observation: dict) -> int:
        """POST /v2/attributor/{attributor_id}/observation. Returns observation_id."""
        url = f"{self.service_url}/v2/attributor/{attributor_id}/observation"
        response = self._session.post(
            url, data=_serialize(observation), headers=_JSON_HEADERS, timeout=self.timeout_seconds
        )
        response.raise_for_status()
        return response.json()["observation_id"]

    def get_attribution(self, attributor_id: str) -> dict:
        """GET /v2/attributor/{attributor_id}/attribution. Returns response JSON."""
        url = f"{self.service_url}/v2/attributor/{attributor_id}/attribution"
        response = self._session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def delete_attributor(self, attributor_id: str) -> None:
        """DELETE /v2/attributor/{attributor_id}. 404 is silently tolerated:
        the server's 24h TTL can expire an attributor between create and
        delete, and cleanup should be idempotent for the caller."""
        url = f"{self.service_url}/v2/attributor/{attributor_id}"
        response = self._session.delete(url, timeout=self.timeout_seconds)
        if response.status_code == 404:
            return
        response.raise_for_status()
