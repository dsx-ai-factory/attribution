# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from attribution_blueprint.connectors.execution_contexts.kubernetes_client import (
    KubernetesClient,
    KubernetesClientError,
    KubernetesClientNotFoundError,
)


def _client(**kwargs) -> KubernetesClient:
    """A client whose underlying API object is pre-set, so no kubeconfig is
    ever loaded in these unit tests."""
    client = KubernetesClient(**kwargs)
    client._api = _FakeDynamic()
    return client


class _FakeDynamic:
    def __init__(self, result=None, error: Exception | None = None, resources=None):
        self.result = {"items": []} if result is None else result
        self.error = error
        self.calls = []
        self.resources = self
        self._resources = resources or [SimpleNamespace(preferred=True)]

    def search(self, **kwargs):
        self.calls.append({"op": "search", **kwargs})
        return self._resources

    def get(self, resource, **kwargs):
        self.calls.append({"op": "get", "resource": resource, **kwargs})
        if self.error is not None:
            raise self.error
        return self.result


def _stub_call(
    client: KubernetesClient,
    result=None,
    error: Exception | None = None,
    resources=None,
):
    calls: list[dict] = []
    fake = _FakeDynamic(result=result, error=error, resources=resources)
    fake.calls = calls
    client._api = fake
    return calls


def test_get_one_uses_discovered_resource_namespace_and_timeout():
    client = _client(timeout_seconds=7)
    calls = _stub_call(client, result={"kind": "Pod"})

    assert client.get_one("Pod", "p", namespace="ns", api_version="v1") == {"kind": "Pod"}

    assert calls[0] == {"op": "search", "kind": "Pod", "group": None, "api_version": "v1", "prefix": None}
    assert calls[1]["name"] == "p"
    assert calls[1]["namespace"] == "ns"
    assert calls[1]["_request_timeout"] == 7


def test_get_one_without_namespace_passes_none():
    client = _client()
    calls = _stub_call(client, result={"kind": "Job"})

    client.get_one("Job", "j", namespace="", api_version="batch/v1")

    assert calls[1]["namespace"] is None


def test_list_passes_namespace_and_selectors():
    client = _client()
    calls = _stub_call(client, result={"items": [{"a": 1}]})

    items = client.list(
        "Event",
        namespace=None,
        field_selector="reason=Scheduled",
        label_selector="app=train",
        api_version="v1",
    )

    assert items == [{"a": 1}]
    assert calls[1]["namespace"] is None
    assert calls[1]["field_selector"] == "reason=Scheduled"
    assert calls[1]["label_selector"] == "app=train"


def test_discovery_is_cached_for_repeated_reads():
    client = _client()
    calls = _stub_call(client, result={"items": []})

    client.list("Pod", namespace="ns", api_version="v1")
    client.list("Pod", namespace="ns", api_version="v1")

    assert sum(call["op"] == "search" for call in calls) == 1


def test_grouped_resource_uses_discovery_preferred_version():
    client = _client()
    preferred = SimpleNamespace(preferred=True, group_version="example.io/v2")
    old = SimpleNamespace(preferred=False, group_version="example.io/v1")
    calls = _stub_call(client, result={"items": []}, resources=[old, preferred])

    client.list("Widget", namespace=None, group="example.io")

    assert calls[1]["resource"] is preferred


def test_ambiguous_discovery_without_one_preferred_resource_raises():
    client = _client()
    resources = [SimpleNamespace(preferred=True), SimpleNamespace(preferred=True)]
    _stub_call(client, resources=resources)

    with pytest.raises(KubernetesClientError, match="2 matches"):
        client.list("Event", namespace=None)


def test_list_raises_on_an_unexpected_response_shape():
    client = _client()
    _stub_call(client, result={"not": "a list response"})

    with pytest.raises(KubernetesClientError, match="unexpected response shape"):
        client.list("Pod", namespace="ns")


def test_get_one_raises_when_the_response_is_not_an_object():
    client = _client()
    _stub_call(client, result=["not", "an", "object"])

    with pytest.raises(KubernetesClientError, match="unexpected response shape"):
        client.get_one("Pod", "p", namespace="ns")


class _FakeApiException(Exception):
    def __init__(self, status: int, reason: str = "", body: str = ""):
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.body = body


def test_get_one_returns_none_only_for_404():
    client = _client()
    _stub_call(client, error=_FakeApiException(404, "Not Found"))

    assert client.get_one("Pod", "p", namespace="ns") is None


def test_get_one_does_not_treat_discovery_404_as_object_absence():
    client = _client()

    def fail_discovery(**_kwargs):
        raise _FakeApiException(404, "API resource not found")

    client._api.search = fail_discovery

    with pytest.raises(KubernetesClientNotFoundError, match="404"):
        client.get_one("Workload", "wl", namespace="ns", group="example.io")


def test_get_one_raises_on_other_http_statuses():
    client = _client()
    _stub_call(client, error=_FakeApiException(403, "Forbidden"))

    with pytest.raises(KubernetesClientError, match="403"):
        client.get_one("Pod", "p", namespace="ns")


def test_list_raises_on_404():
    client = _client()
    _stub_call(client, error=_FakeApiException(404, "Not Found"))

    with pytest.raises(KubernetesClientNotFoundError):
        client.list("Pod", namespace="ns")


def test_errors_without_a_status_are_wrapped():
    client = _client()
    _stub_call(client, error=TimeoutError("connection timed out"))

    with pytest.raises(KubernetesClientError, match="TimeoutError"):
        client.get_one("Pod", "p", namespace="ns")


def test_error_messages_do_not_leak_response_bodies():
    client = _client()
    _stub_call(client, error=_FakeApiException(403, "Forbidden", body="SECRET-BODY"))

    with pytest.raises(KubernetesClientError) as excinfo:
        client.get_one("Pod", "p", namespace="ns")
    assert "SECRET-BODY" not in str(excinfo.value)


def test_client_is_built_once_and_reused(mocker):
    pytest.importorskip("kubernetes")
    fake = _FakeDynamic()
    dynamic = mocker.patch("kubernetes.dynamic.DynamicClient", return_value=fake)
    build = mocker.patch(
        "kubernetes.config.new_client_from_config",
        return_value=SimpleNamespace(),
    )
    client = KubernetesClient(kubeconfig="/tmp/kc", context="ctx")

    for _ in range(3):
        client.list("Pod", namespace="ns")

    build.assert_called_once_with(config_file="/tmp/kc", context="ctx")
    dynamic.assert_called_once_with(build.return_value)


def test_unset_kubeconfig_and_context_try_incluster_then_default_kubeconfig(mocker):
    pytest.importorskip("kubernetes")
    from kubernetes import config

    mocker.patch("kubernetes.config.load_incluster_config", side_effect=config.ConfigException())
    mocker.patch("kubernetes.dynamic.DynamicClient", return_value=_FakeDynamic())
    build = mocker.patch(
        "kubernetes.config.new_client_from_config",
        return_value=SimpleNamespace(),
    )

    KubernetesClient().list("Pod", namespace="ns")

    build.assert_called_once_with()
