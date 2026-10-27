# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from http import HTTPStatus

from nvidia_attribution_client import (
    ApiClient,
    ApiException,
    AttributorInfoV2,
    Configuration,
    DefaultApi,
    DuplicateAttributorError,
    DuplicateObservationError,
    ObservationResponseV2,
    ObservationV2,
)

# Set to "slightly larger than a multiple of 3" as per
# https://requests.readthedocs.io/en/latest/user/advanced/#timeouts
_CONNECT_TIMEOUT_SECONDS = 3.05


class AttributorConflictError(RuntimeError):
    """An existing session prevents creation; its ownership is unchanged."""

    def __init__(self, attributor_id: str):
        self.attributor_id = attributor_id
        super().__init__(
            f"Attributor {attributor_id} already exists; preserving the existing session. "
            "Retry after its owner finishes and deletes it."
        )

    def __reduce__(self):
        """Rebuild from the session ID rather than the formatted message."""
        return type(self), (self.attributor_id,), self.__dict__


class ObservationConflictError(RuntimeError):
    """The attributor already holds an identical observation, so this POST added nothing new."""

    def __init__(self, observation_id: int):
        super().__init__(f"observation {observation_id} already ingested")
        self.observation_id = observation_id


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
        """POST /v2/attributor. Return a newly created attributor's id.

        A conflict does not establish that the existing session is stale or
        owned by this caller. Preserve it and raise AttributorConflictError,
        chaining the original service error. Other service errors propagate.
        """
        try:
            response = self._api.new_attribution_v2_attributor_post(info, _request_timeout=self._request_timeout)
        except ApiException as e:
            if e.status == HTTPStatus.CONFLICT and isinstance(e.data, DuplicateAttributorError):
                raise AttributorConflictError(e.data.att_id) from e
            raise
        return response.attributor_id

    def ingest_observation(self, attributor_id: str, observation: ObservationV2) -> ObservationResponseV2:
        """POST /v2/attributor/{attributor_id}/observation. Returns the service response.

        Raises:
            ObservationConflictError: The service rejected the POST as a duplicate (HTTP 409).
        """
        try:
            response = self._api.ingest_v2_attributor_att_id_observation_post(
                attributor_id, observation, _request_timeout=self._request_timeout
            )
        except ApiException as e:
            if e.status == HTTPStatus.CONFLICT and isinstance(e.data, DuplicateObservationError):
                raise ObservationConflictError(e.data.obs_id) from e
            body = e.body or ""
            raise RuntimeError(f"HTTP {e.status} {e.reason}: {body[:300]}") from e
        return response

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
