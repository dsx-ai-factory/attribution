# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from attribution_blueprint.connectors.execution_contexts.kubernetes_client import (
    KubernetesClient,
    KubernetesClientError,
)


def _client(**kwargs) -> KubernetesClient:
    """A client whose underlying API object is pre-set, so no kubeconfig is
    ever loaded in these unit tests."""
    client = KubernetesClient(**kwargs)
    client._api = _FakeDynamic()
    return client


class _FakeDynamic:
    """Stands in for `kubernetes.dynamic.DynamicClient`.

    `.resources` is the same object (mirroring the real client's shape), so
    `.resources.get(**kwargs)` records a `"discover"` call and either
    returns a canned resource or raises `discovery_error`; `.get(resource,
    **kwargs)` records a `"get"` call and either returns the canned
    payload or raises `error`. Real discovery caching/preferred-version
    resolution is the upstream library's job, not something this fake
    needs to reproduce — `KubernetesClient` no longer does either itself.
    """

    def __init__(
        self,
        result=None,
        error: Exception | None = None,
        discovery_error: Exception | None = None,
        resource=None,
    ):
        self.result = {"items": []} if result is None else result
        self.error = error
        self.discovery_error = discovery_error
        self.resource = resource if resource is not None else SimpleNamespace(name="fake-resource")
        self.calls: list[dict] = []
        self.resources = self

    def get(self, resource=None, **kwargs):
        if resource is None:
            self.calls.append({"op": "discover", **kwargs})
            if self.discovery_error is not None:
                raise self.discovery_error
            return self.resource
        self.calls.append({"op": "get", "resource": resource, **kwargs})
        if self.error is not None:
            raise self.error
        return self.result


def _stub_call(
    client: KubernetesClient,
    result=None,
    error: Exception | None = None,
    discovery_error: Exception | None = None,
    resource=None,
):
    fake = _FakeDynamic(result=result, error=error, discovery_error=discovery_error, resource=resource)
    client._api = fake
    return fake.calls


def test_get_one_uses_discovered_resource_namespace_and_timeout():
    client = _client(timeout_seconds=7)
    calls = _stub_call(client, result={"kind": "Pod"})

    assert client.get_one("Pod", "p", namespace="ns", api_version="v1") == {"kind": "Pod"}

    assert calls[0] == {"op": "discover", "kind": "Pod", "group": None, "api_version": "v1", "prefix": None}
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


def test_get_one_raises_with_404_status():
    """404 surfaces as `KubernetesClientError` with `.status == 404` — the
    caller decides what "not found" means for its own call site, instead of
    the client collapsing it to `None` or a parallel exception type."""
    client = _client()
    _stub_call(client, error=_FakeApiException(404, "Not Found"))

    with pytest.raises(KubernetesClientError, match="404") as excinfo:
        client.get_one("Pod", "p", namespace="ns")
    assert excinfo.value.status == 404


def test_discovery_404_never_carries_a_status():
    """A 404 while *discovering* a resource type (CRD not installed,
    misconfigured api_version, ...) is not "this object doesn't exist" —
    it's "I can't resolve what this kind even is." Callers skip on
    `error.status == 404` to mean the former, so discovery failures must
    never set `.status`, even when the underlying error happens to be a
    404 — otherwise an unavailable discovery endpoint gets silently
    swallowed as if the requested object were merely absent."""
    client = _client()
    _stub_call(client, discovery_error=_FakeApiException(404, "API resource not found"))

    with pytest.raises(KubernetesClientError, match="discovery failed") as excinfo:
        client.get_one("Workload", "wl", namespace="ns", group="example.io")
    assert excinfo.value.status is None


def test_discovery_errors_without_a_status_are_wrapped():
    """Ambiguous/missing-kind discovery failures (e.g. the upstream
    `ResourceNotFoundError`/`ResourceNotUniqueError`) carry no HTTP status;
    they still get wrapped rather than propagating a bare library type."""
    client = _client()
    _stub_call(client, discovery_error=RuntimeError("Multiple matches found for kind=Event"))

    with pytest.raises(KubernetesClientError, match="Multiple matches") as excinfo:
        client.list("Event", namespace=None)
    assert excinfo.value.status is None


def test_get_one_raises_on_other_http_statuses():
    client = _client()
    _stub_call(client, error=_FakeApiException(403, "Forbidden"))

    with pytest.raises(KubernetesClientError, match="403") as excinfo:
        client.get_one("Pod", "p", namespace="ns")
    assert excinfo.value.status == 403


def test_list_raises_with_404_status():
    client = _client()
    _stub_call(client, error=_FakeApiException(404, "Not Found"))

    with pytest.raises(KubernetesClientError) as excinfo:
        client.list("Pod", namespace="ns")
    assert excinfo.value.status == 404


def test_errors_without_a_status_are_wrapped():
    client = _client()
    _stub_call(client, error=TimeoutError("connection timed out"))

    with pytest.raises(KubernetesClientError, match="TimeoutError") as excinfo:
        client.get_one("Pod", "p", namespace="ns")
    assert excinfo.value.status is None


def test_error_messages_do_not_leak_response_bodies():
    client = _client()
    _stub_call(client, error=_FakeApiException(403, "Forbidden", body="SECRET-BODY"))

    with pytest.raises(KubernetesClientError) as excinfo:
        client.get_one("Pod", "p", namespace="ns")
    assert "SECRET-BODY" not in str(excinfo.value)


def test_client_is_built_once_and_reused(mocker):
    pytest.importorskip("kubernetes")
    configuration = SimpleNamespace()
    api_client = SimpleNamespace(call_api=mocker.Mock(return_value={}))
    fake = _FakeDynamic()
    dynamic = mocker.patch("kubernetes.dynamic.DynamicClient", return_value=fake)
    configuration_cls = mocker.patch(
        "kubernetes.client.Configuration",
        return_value=configuration,
    )
    load = mocker.patch("kubernetes.config.load_kube_config")
    api_client_cls = mocker.patch("kubernetes.client.ApiClient", return_value=api_client)
    client = KubernetesClient(kubeconfig="/tmp/kc", context="ctx")

    for _ in range(3):
        client.list("Pod", namespace="ns")

    configuration_cls.assert_called_once_with()
    load.assert_called_once_with(
        config_file="/tmp/kc",
        context="ctx",
        client_configuration=configuration,
    )
    api_client_cls.assert_called_once_with(configuration)
    dynamic.assert_called_once_with(api_client)


def test_unset_kubeconfig_and_context_delegates_to_load_config(mocker):
    """With no explicit `kubeconfig`/`context`, credential resolution is
    delegated entirely to `kubernetes.config.load_config()`, which already
    implements the `KUBECONFIG`/`~/.kube/config`-then-in-cluster precedence —
    not reimplemented here."""
    pytest.importorskip("kubernetes")
    configuration = SimpleNamespace()
    api_client = SimpleNamespace(call_api=mocker.Mock(return_value={}))
    mocker.patch("kubernetes.client.Configuration", return_value=configuration)
    load_config = mocker.patch("kubernetes.config.load_config")
    load_kube_config = mocker.patch("kubernetes.config.load_kube_config")
    mocker.patch("kubernetes.client.ApiClient", return_value=api_client)
    mocker.patch("kubernetes.dynamic.DynamicClient", return_value=_FakeDynamic())

    KubernetesClient().list("Pod", namespace="ns")

    load_config.assert_called_once_with(client_configuration=configuration)
    load_kube_config.assert_not_called()


def test_explicit_kubeconfig_or_context_bypasses_load_config(mocker):
    """An explicit `kubeconfig`/`context` is honored exactly as given, via
    `load_kube_config` directly — not routed through `load_config()`'s own
    fallback, which can't take a `context` override on its in-cluster path
    anyway (`load_incluster_config` has no `context` parameter)."""
    pytest.importorskip("kubernetes")
    configuration = SimpleNamespace()
    api_client = SimpleNamespace(call_api=mocker.Mock(return_value={}))
    mocker.patch("kubernetes.client.Configuration", return_value=configuration)
    load_kube_config = mocker.patch("kubernetes.config.load_kube_config")
    load_config = mocker.patch("kubernetes.config.load_config")
    mocker.patch("kubernetes.client.ApiClient", return_value=api_client)
    mocker.patch("kubernetes.dynamic.DynamicClient", return_value=_FakeDynamic())

    KubernetesClient(context="kind-kueue-demo").list("Pod", namespace="ns")

    load_kube_config.assert_called_once_with(
        config_file=None, context="kind-kueue-demo", client_configuration=configuration
    )
    load_config.assert_not_called()


def test_explicit_kubeconfig_failure_raises_without_falling_back(mocker):
    """An explicit kubeconfig/context that fails to load must raise, not
    silently fall back to in-cluster credentials — the caller asked for a
    specific cluster, so a broken path to it is an error, not a cue to try
    somewhere else."""
    pytest.importorskip("kubernetes")
    from kubernetes import config

    mocker.patch("kubernetes.client.Configuration", return_value=SimpleNamespace())
    mocker.patch("kubernetes.config.load_kube_config", side_effect=config.ConfigException("invalid"))
    incluster = mocker.patch("kubernetes.config.load_incluster_config")

    with pytest.raises(KubernetesClientError, match="ConfigException"):
        KubernetesClient(kubeconfig="/tmp/broken").list("Pod", namespace="ns")

    incluster.assert_not_called()


def test_default_timeout_is_applied_to_discovery_api_calls(mocker):
    """No explicit kubeconfig/context is given, so this goes through
    `load_config()`, not `load_kube_config()` directly — mock the former.
    Mocking only `load_kube_config()` here would pass or fail depending on
    whether the machine running the test happens to have a `~/.kube/config`
    (it does locally; CI runners typically don't, so `load_config()` falls
    through to a real, unmocked `load_incluster_config()` and errors)."""
    pytest.importorskip("kubernetes")
    configuration = SimpleNamespace()
    call_api = mocker.Mock(return_value={})
    api_client = SimpleNamespace(call_api=call_api)
    mocker.patch("kubernetes.client.Configuration", return_value=configuration)
    mocker.patch("kubernetes.config.load_config")
    mocker.patch("kubernetes.client.ApiClient", return_value=api_client)
    mocker.patch("kubernetes.dynamic.DynamicClient", return_value=_FakeDynamic())

    KubernetesClient(timeout_seconds=7)._client()
    api_client.call_api("/apis", "GET", _request_timeout=None)

    call_api.assert_called_once_with("/apis", "GET", _request_timeout=7)
