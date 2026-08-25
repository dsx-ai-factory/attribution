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
from http import HTTPStatus

from nvidia_attribution_client import (
    ApiClient,
    ApiException,
    AttributorInfoV2,
    Configuration,
    DefaultApi,
    DuplicateAttributorError,
    Observation as ServiceObservation,
)

# Set to "slightly larger than a multiple of 3" as per
# https://requests.readthedocs.io/en/latest/user/advanced/#timeouts
_CONNECT_TIMEOUT_SECONDS = 3.05


class AttributionConnector:
    """Blueprint-facing wrapper around the v2 API of the Attribution Service client."""

    def __init__(
        self,
        service_url: str,
        timeout_seconds: float = 60.0,
    ):
        self.service_url = service_url.rstrip("/")
        self._request_timeout = (_CONNECT_TIMEOUT_SECONDS, timeout_seconds)
        # Retries on idempotent methods only (GET/DELETE)
        configuration = Configuration(host=self.service_url, retries=3)
        self._api = DefaultApi(ApiClient(configuration))

    def create_attributor(self, info: AttributorInfoV2) -> str:
        """POST /v2/attributor. Returns attributor_id."""
        response = self._api.new_attribution_v2_attributor_post(info, _request_timeout=self._request_timeout)
        return response.attributor_id

    def create_or_replace_attributor(self, info: AttributorInfoV2) -> str:
        """Create an attributor, replacing the stale service record on 409.

        The service returns the existing attributor id as `att_id` on conflict.
        Keeping that HTTP-specific recovery here prevents callers from knowing
        the transport error type or response envelope.
        """
        try:
            return self.create_attributor(info)
        except ApiException as e:
            if e.status != HTTPStatus.CONFLICT or not isinstance(e.data, DuplicateAttributorError):
                raise
            self.delete_attributor(e.data.att_id)
            return self.create_attributor(info)

    def ingest_observation(self, attributor_id: str, observation: ServiceObservation) -> int:
        """POST /v2/attributor/{attributor_id}/observation. Returns observation_id."""
        response = self._api.ingest_v2_attributor_att_id_observation_post(
            attributor_id, observation, _request_timeout=self._request_timeout
        )
        return response.observation_id

    def get_attribution(self, attributor_id: str) -> dict:
        """GET /v2/attributor/{attributor_id}/attribution. Returns response JSON."""
        response = self._api.attribute_v2_attributor_att_id_attribution_get(
            attributor_id, _request_timeout=self._request_timeout
        )
        # to_dict() would leave datetimes, which aren't JSON-serializable.
        return json.loads(response.to_json())

    def delete_attributor(self, attributor_id: str) -> None:
        """DELETE /v2/attributor/{attributor_id}. 404 is silently tolerated: the
        server's 24h TTL can expire an attributor before cleanup runs."""
        try:
            self._api.delete_attribution_v2_attributor_att_id_delete(
                attributor_id, _request_timeout=self._request_timeout
            )
        except ApiException as e:
            if e.status != HTTPStatus.NOT_FOUND:
                raise
