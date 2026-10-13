# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only Kubernetes API client shared by Kubernetes adapters."""

from __future__ import annotations

import threading


class KubernetesClientError(RuntimeError):
    """A Kubernetes API read failed."""


class KubernetesClientNotFoundError(KubernetesClientError):
    """The API server reported that the requested object does not exist."""


class KubernetesClient:
    """Runs read-only Kubernetes API queries and returns the decoded JSON.

    `kubeconfig` and `context` select a kubeconfig file and context;
    both are optional, and when unset this falls back to the default
    kubeconfig (`KUBECONFIG`, `~/.kube/config`) or, if neither resolves,
    the in-cluster service account. Kubernetes discovery resolves resources
    by kind and selects a group's preferred served version when the caller
    does not require a specific version.

    `get_one` returns `None` only for HTTP 404, and every other failure
    (Forbidden, connection error, timeout, malformed response) raises
    `KubernetesClientError`, so a failed read is never mistaken for an empty
    result.
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
        self._discovery_lock = threading.Lock()
        self._api = None
        self._resources: dict[tuple[str, str | None, str | None], object] = {}

    def get_one(
        self,
        kind: str,
        name: str,
        *,
        namespace: str,
        group: str | None = None,
        api_version: str | None = None,
        prefix: str | None = None,
    ) -> dict | None:
        resource = self._resource(kind, group=group, api_version=api_version, prefix=prefix)
        try:
            payload = self._client().get(
                resource,
                name=name,
                namespace=namespace or None,
                _request_timeout=self.timeout_seconds,
            )
        except Exception as error:
            wrapped = _wrap(error)
            if isinstance(wrapped, KubernetesClientNotFoundError):
                return None
            raise wrapped from error
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
            raise _wrap(error) from error
        data = self._as_dict(payload, kind)
        items = data.get("items")
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise KubernetesClientError(f"Kubernetes API query for {kind!r} returned an unexpected response shape")
        return items

    def _resource(self, kind: str, *, group: str | None, api_version: str | None, prefix: str | None = None):
        key = kind, group, api_version, prefix
        with self._discovery_lock:
            if key in self._resources:
                return self._resources[key]
            try:
                matches = self._client().resources.search(
                    kind=kind,
                    group=group,
                    api_version=api_version,
                    prefix=prefix,
                )
                preferred = [resource for resource in matches if resource.preferred]
                candidates = preferred or matches
                if len(candidates) != 1:
                    raise KubernetesClientError(
                        f"Kubernetes discovery found {len(candidates)} matches for "
                        f"kind={kind!r}, group={group!r}, api_version={api_version!r}, prefix={prefix!r}"
                    )
                self._resources[key] = candidates[0]
                return candidates[0]
            except KubernetesClientError:
                raise
            except Exception as error:
                raise _wrap(error) from error

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

        if self.kubeconfig or self.context:
            api_client = config.new_client_from_config(
                config_file=self.kubeconfig,
                context=self.context,
            )
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                api_client = config.new_client_from_config()
            else:
                api_client = client.ApiClient()
        return DynamicClient(api_client)


def _wrap(error: Exception) -> KubernetesClientError:
    """Classify a client-library exception without leaking its response body."""
    status = getattr(error, "status", None)
    if status == 404:
        return KubernetesClientNotFoundError(f"Kubernetes API query failed with status 404: {_reason(error)}")
    if status is not None:
        return KubernetesClientError(f"Kubernetes API query failed with status {status}: {_reason(error)}")
    return KubernetesClientError(f"Kubernetes API query failed: {type(error).__name__}: {_reason(error)}")


def _reason(error: Exception) -> str:
    reason = getattr(error, "reason", None)
    return reason if isinstance(reason, str) and reason else str(error)
