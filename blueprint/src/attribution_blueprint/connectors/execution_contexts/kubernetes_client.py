# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only Kubernetes API client shared by Kubernetes adapters."""

from __future__ import annotations

import threading


class KubernetesClientError(RuntimeError):
    """A Kubernetes API read failed.

    `status` carries the HTTP status code when the failure was an API
    response (e.g. 404, 403); it is `None` for connection-level failures
    (timeout, DNS, malformed response, ...).=
    """

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class KubernetesClient:
    """Runs read-only Kubernetes API queries and returns the decoded JSON.

    `kubeconfig` and `context` select a kubeconfig file and context; both
    are optional. When set, they're used exactly as given (no silent
    fallback). When neither is set, credential resolution is delegated
    to `kubernetes.config.load_config()`.

    Every read requires the caller to name the resource it wants: `kind`
    plus either `group` and `api_version`, or a combined
    `api_version="<group>/<version>"` (bare version for the core API).

    `get_one` and `list` raise `KubernetesClientError` for every failure
    (not-found, forbidden, connection error, timeout, malformed response),
    with `error.status` carrying the HTTP status when there is one.
    """

    def __init__(
        self,
        kubeconfig: str | None = None,
        context: str | None = None,
        timeout_seconds: float = 60,
    ):
        self.kubeconfig = kubeconfig
        self.context = context
        self.timeout_seconds = timeout_seconds
        self._client_lock = threading.Lock()
        self._api = None

    def get_one(
        self,
        kind: str,
        name: str,
        *,
        namespace: str,
        group: str | None = None,
        api_version: str | None = None,
        prefix: str | None = None,
    ) -> dict:
        """Read one object, raising `KubernetesClientError` (`.status == 404`
        for not-found) on any failure. The caller already knows whether
        "not found" means "skip this id" or "this object must exist" —
        this never collapses that into a silent `None`.
        """
        resource = self._resource(kind, group=group, api_version=api_version, prefix=prefix)
        try:
            payload = self._client().get(
                resource,
                name=name,
                namespace=namespace or None,
                _request_timeout=self.timeout_seconds,
            )
        except Exception as error:
            # bare exception keeps the raw API response body out of logs.
            status = getattr(error, "status", None)
            reason = getattr(error, "reason", None)
            reason = reason if isinstance(reason, str) and reason else str(error)
            if status is not None:
                raise KubernetesClientError(
                    f"Kubernetes API query failed with status {status}: {reason}", status=status
                ) from error
            raise KubernetesClientError(f"Kubernetes API query failed: {type(error).__name__}: {reason}") from error
        return self._as_dict(payload, kind)

    def list(
        self,
        kind: str,
        *,
        namespace: str | None,
        field_selector: str | None = None,
        label_selector: str | None = None,
        group: str | None = None,
        api_version: str | None = None,
        prefix: str | None = None,
    ) -> list[dict]:
        resource = self._resource(kind, group=group, api_version=api_version, prefix=prefix)
        try:
            payload = self._client().get(
                resource,
                namespace=namespace,
                field_selector=field_selector,
                label_selector=label_selector,
                _request_timeout=self.timeout_seconds,
            )
        except Exception as error:
            # bare exception keeps the raw API response body out of logs.
            status = getattr(error, "status", None)
            reason = getattr(error, "reason", None)
            reason = reason if isinstance(reason, str) and reason else str(error)
            if status is not None:
                raise KubernetesClientError(
                    f"Kubernetes API query failed with status {status}: {reason}", status=status
                ) from error
            raise KubernetesClientError(f"Kubernetes API query failed: {type(error).__name__}: {reason}") from error
        data = self._as_dict(payload, kind)
        items = data.get("items")
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise KubernetesClientError(f"Kubernetes API query for {kind!r} returned an unexpected response shape")
        return items

    def _resource(self, kind: str, *, group: str | None, api_version: str | None, prefix: str | None):
        """Resolve `kind` to a queryable resource; any failure here is a hard
        error, never a skippable one.
        """
        try:
            return self._client().resources.get(kind=kind, group=group, api_version=api_version, prefix=prefix)
        except Exception as error:
            reason = getattr(error, "reason", None)
            reason = reason if isinstance(reason, str) and reason else str(error)
            raise KubernetesClientError(
                f"Kubernetes resource discovery failed for {kind!r}: {type(error).__name__}: {reason}"
            ) from error

    @staticmethod
    def _as_dict(payload, kind: str) -> dict:
        if callable(getattr(payload, "to_dict", None)):
            payload = payload.to_dict()
        if not isinstance(payload, dict):
            raise KubernetesClientError(f"Kubernetes API query for {kind!r} returned an unexpected response shape")
        return payload

    # --- client lifecycle -----------------------------------------------

    def _client(self):
        with self._client_lock:
            if self._api is None:
                self._api = self._build_client()
            return self._api

    def _build_client(self):
        from kubernetes import client, config
        from kubernetes.dynamic import DynamicClient

        configuration = client.Configuration()
        if self.kubeconfig is not None or self.context is not None:
            config.load_kube_config(
                config_file=self.kubeconfig,
                context=self.context,
                client_configuration=configuration,
            )
        else:
            config.load_config(client_configuration=configuration)
        api_client = client.ApiClient(configuration)
        _set_default_request_timeout(api_client, self.timeout_seconds)
        return DynamicClient(api_client)


def _set_default_request_timeout(api_client, timeout_seconds: float) -> None:
    """Apply a timeout to calls, including dynamic discovery requests."""
    call_api = api_client.call_api

    def call_api_with_timeout(*args, **kwargs):
        if kwargs.get("_request_timeout") is None:
            kwargs["_request_timeout"] = timeout_seconds
        return call_api(*args, **kwargs)

    api_client.call_api = call_api_with_timeout
