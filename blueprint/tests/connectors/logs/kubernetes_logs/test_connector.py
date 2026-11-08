# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KubernetesLogsConnector.

The connector reads each pod listed on the ExecutionContext's `pod_names`
through the Kubernetes API. Tests replace the API client with a fake that
records its calls, and inspect the calls and the emitted Observations.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from attribution_blueprint.connectors.logs.kubectl_logs import KubectlLogsConnector
from attribution_blueprint.connectors.logs.kubernetes_logs import KubernetesLogsConnector
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.kubernetes.client import KubernetesClientError
from attribution_blueprint.types import Source

START = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _text(observation: dict) -> str:
    """Rejoin an APPLOG observation's per-line entries into the log text."""
    return "\n".join(entry["body"] for entry in json.loads(observation["body"]))


def _ctx(**extra) -> dict:
    """A workload execution context for two pods, with fields overridable by keyword."""
    base = {
        "id": "uid-1",
        "job_id": "team-a/train-abc",
        "namespace": "team-a",
        "nodes": ["node-a", "node-b"],
        "start_time": START,
        "end_time": START + timedelta(minutes=30),
        "pod_names": ["train-abc-0", "train-abc-1"],
        "pod_nodes": {"train-abc-0": "node-a", "train-abc-1": "node-b"},
    }
    base.update(extra)
    return base


def _pod(*containers: str, init: tuple[str, ...] = ()) -> dict:
    """A pod manifest listing the given containers, with optional init containers."""
    return {
        "spec": {
            "initContainers": [{"name": n} for n in init],
            "containers": [{"name": n} for n in containers],
        }
    }


class _FakeClient:
    """Stands in for KubernetesClient: pods by name, logs by (pod, container).

    A value in `errors` (by pod, or by (pod, container)) is raised instead.
    """

    def __init__(self, pods: dict[str, dict], logs: dict[tuple[str, str], str], errors: dict | None = None):
        """Store the canned data and start with no recorded calls."""
        self.pods = pods
        self.logs = logs
        self.errors = errors or {}
        self.pod_reads: list[str] = []
        self.log_reads: list[dict] = []

    def get_one(self, kind, name, *, namespace, api_version=None, **_):
        """Return the named pod manifest, or raise the error configured for it."""
        assert kind == "Pod" and api_version == "v1"
        self.pod_reads.append(name)
        if name in self.errors:
            raise self.errors[name]
        return self.pods[name]

    def read_pod_log(self, name, *, namespace, container=None, **kwargs):
        """Record the read and return the configured log, or raise the configured error."""
        self.log_reads.append({"pod": name, "namespace": namespace, "container": container, **kwargs})
        if (name, container) in self.errors:
            raise self.errors[(name, container)]
        return self.logs.get((name, container), "")


def _two_pods(overrides: dict[tuple[str, str], str] | None = None) -> _FakeClient:
    """A fake client for two single-container pods, with optional log overrides."""
    logs = {
        ("train-abc-0", "trainer"): "rank=0\nstep=100\n",
        ("train-abc-1", "trainer"): "rank=1\nCUDA error: device-side assert\nexit 42\n",
    }
    logs.update(overrides or {})
    return _FakeClient({"train-abc-0": _pod("trainer"), "train-abc-1": _pod("trainer")}, logs)


def test_emits_one_observation_per_pod():
    """Emits one observation per pod."""
    conn = KubernetesLogsConnector(client=_two_pods())
    obs = conn.collect(_ctx())

    assert len(obs) == 2
    bodies = {_text(o) for o in obs}
    assert "rank=0\nstep=100\n" in bodies
    assert any("CUDA error: device-side assert" in b for b in bodies)
    for o in obs:
        assert o["context"]["source"] == Source.APPLOG
        assert o["context"]["time_interval"] == {"start": None, "end": None}


def test_stamps_each_entry_with_the_pods_node():
    """Stamps each entry with the pods node."""
    obs = KubernetesLogsConnector(client=_two_pods()).collect(_ctx())
    by_text = {_text(o): json.loads(o["body"]) for o in obs}
    rank0 = by_text["rank=0\nstep=100\n"]
    assert {e["resources"][0]["hostname"] for e in rank0} == {"node-a"}


def test_reads_every_container_init_first_and_joins_without_a_separator():
    """Reads every container init first and joins without a separator."""
    client = _FakeClient(
        {"p": _pod("main", "sidecar", init=("setup",))},
        {("p", "setup"): "init\n", ("p", "main"): "main\n", ("p", "sidecar"): "side\n"},
    )
    obs = KubernetesLogsConnector(client=client).collect(_ctx(pod_names=["p"], pod_nodes={}))

    assert [r["container"] for r in client.log_reads] == ["setup", "main", "sidecar"]
    assert _text(obs[0]) == "init\nmain\nside\n"


def test_names_ephemeral_containers_last():
    """Names ephemeral containers last."""
    pod = _pod("main")
    pod["spec"]["ephemeralContainers"] = [{"name": "debug"}]
    client = _FakeClient({"p": pod}, {("p", "main"): "m\n", ("p", "debug"): "d\n"})
    KubernetesLogsConnector(client=client).collect(_ctx(pod_names=["p"], pod_nodes={}))
    assert [r["container"] for r in client.log_reads] == ["main", "debug"]


def test_a_configured_container_skips_the_pod_read():
    """A configured container skips the pod read."""
    client = _two_pods()
    KubernetesLogsConnector(client=client, container="trainer").collect(_ctx())
    assert client.pod_reads == []
    assert {r["container"] for r in client.log_reads} == {"trainer"}


def test_skips_empty_bodies():
    """Skips empty bodies."""
    client = _two_pods({("train-abc-0", "trainer"): ""})
    obs = KubernetesLogsConnector(client=client).collect(_ctx())
    assert len(obs) == 1
    assert _text(obs[0]).startswith("rank=1")


def test_a_missing_pod_is_skipped_and_the_rest_still_collect():
    """A missing pod is skipped and the rest still collect."""
    client = _two_pods()
    client.errors["train-abc-0"] = KubernetesClientError("not found", status=404)
    obs = KubernetesLogsConnector(client=client).collect(_ctx())
    assert len(obs) == 1
    assert _text(obs[0]).startswith("rank=1")


def test_a_forbidden_pod_is_skipped_with_the_status_in_the_warning():
    """A forbidden pod is skipped with the status in the warning."""
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        client = _two_pods()
        client.errors["train-abc-0"] = KubernetesClientError(
            "Kubernetes API query failed with status 403: Forbidden", status=403
        )
        obs = KubernetesLogsConnector(client=client).collect(_ctx())
    finally:
        logger.remove(sink)
    assert len(obs) == 1
    assert any("403" in m and "train-abc-0" in m for m in messages)


def test_one_container_failing_keeps_the_others():
    """One container failing keeps the others."""
    client = _FakeClient(
        {"p": _pod("a", "b")},
        {("p", "b"): "b-log\n"},
        errors={("p", "a"): KubernetesClientError("container is waiting to start", status=400)},
    )
    obs = KubernetesLogsConnector(client=client).collect(_ctx(pod_names=["p"], pod_nodes={}))
    assert _text(obs[0]) == "b-log\n"


def test_a_pod_without_containers_is_skipped():
    """A pod without containers is skipped."""
    client = _FakeClient({"p": {"spec": {}}}, {})
    assert KubernetesLogsConnector(client=client).collect(_ctx(pod_names=["p"])) == []
    assert client.log_reads == []


def test_returns_nothing_without_pods():
    """Returns nothing without pods."""
    client = _two_pods()
    assert KubernetesLogsConnector(client=client).collect(_ctx(pod_names=[])) == []
    assert client.pod_reads == [] and client.log_reads == []


def test_refuses_to_read_without_a_namespace():
    """Refuses to read without a namespace."""
    client = _two_pods()
    assert KubernetesLogsConnector(client=client).collect(_ctx(namespace="")) == []
    assert client.pod_reads == [] and client.log_reads == []


def test_naive_datetimes_are_rejected():
    """Naive datetimes are rejected."""
    ctx = _ctx(start_time=datetime(2026, 5, 1, 12, 0, 0))
    with pytest.raises(ValueError):
        KubernetesLogsConnector(client=_two_pods()).collect(ctx)


def test_passes_the_read_options_through():
    """Passes the read options through."""
    client = _two_pods()
    conn = KubernetesLogsConnector(client=client, timestamps=False, previous=True, max_bytes_per_pod="1KB")
    conn.collect(_ctx())
    for read in client.log_reads:
        assert read["timestamps"] is False
        assert read["previous"] is True
        assert read["max_bytes"] == 1024
        assert read["since_seconds"] is None


def test_since_context_start_is_the_seconds_back_to_the_start():
    """Since context start is the seconds back to the start."""
    now = START + timedelta(seconds=90, milliseconds=200)
    client = _two_pods()
    KubernetesLogsConnector(client=client, since_context_start=True, now=lambda: now).collect(_ctx())
    assert {r["since_seconds"] for r in client.log_reads} == {91}


def test_since_seconds_is_measured_at_each_read_not_once_per_collection():
    """A slow earlier read must not shift the window for the pods after it."""
    clock = iter(START + timedelta(seconds=90 + 10 * i) for i in range(10))
    client = _two_pods()
    KubernetesLogsConnector(client=client, since_context_start=True, now=lambda: next(clock)).collect(_ctx())
    assert [r["since_seconds"] for r in client.log_reads] == [90, 100]


def test_since_seconds_is_never_below_one():
    """Since seconds is never below one."""
    client = _two_pods()
    # A start time in the future (clock skew) must still be a valid window.
    KubernetesLogsConnector(client=client, since_context_start=True, now=lambda: START - timedelta(seconds=30)).collect(
        _ctx()
    )
    assert {r["since_seconds"] for r in client.log_reads} == {1}


def test_the_pod_total_is_capped_to_the_tail():
    """The pod total is capped to the tail."""
    client = _FakeClient({"p": _pod("a", "b")}, {("p", "a"): "AAAA\n", ("p", "b"): "BBBB\nEND\n"})
    obs = KubernetesLogsConnector(client=client, max_bytes_per_pod=8).collect(_ctx(pod_names=["p"], pod_nodes={}))
    # 4 + 9 bytes joined, tailed to the last 8: "BBBB\nEND\n"[1:]
    assert _text(obs[0]) == "BBB\nEND\n"


def test_emits_the_same_observation_as_kubectl_logs_for_the_same_text(mocker):
    """The API path and the kubectl path must be interchangeable: same text in,
    same entries (timestamps, node stamping, rank prefix handling) out."""
    text = "[rank 0] step=1\n[rank 0] RuntimeError: CUDA out of memory\n"
    ctx = _ctx(pod_names=["p"], pod_nodes={"p": "node-a"})

    api = KubernetesLogsConnector(client=_FakeClient({"p": _pod("trainer")}, {("p", "trainer"): text}))
    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=text, stderr=""),
    )
    kubectl = KubectlLogsConnector()

    assert api.collect(ctx)[0]["body"] == kubectl.collect(ctx)[0]["body"]


def test_source_can_be_syslog():
    """Source can be syslog."""
    client = _two_pods()
    conn = KubernetesLogsConnector(client=client, source=Source.SYSLOG)
    assert conn.source == Source.SYSLOG


def test_mca_registration():
    """Mca registration."""
    from attribution_blueprint import mca_registry

    spec = mca_registry.get_component("log", "kubernetes_logs")
    assert spec.framework == "log"
    assert spec.name == "kubernetes_logs"
    assert {"container", "previous", "since_context_start", "max_bytes_per_pod", "timeout_seconds"}.issubset(
        spec.schema
    )
    # The kubectl binary is not involved, so its knob must not exist here.
    assert "kubectl_path" not in spec.schema


def test_mca_build_defaults_and_validation():
    """Mca build defaults and validation."""
    from attribution_blueprint import mca_registry

    spec = mca_registry.get_component("log", "kubernetes_logs")
    conn = spec.build({"container": "trainer", "timestamps": "false"}, {"source": Source.APPLOG})
    assert isinstance(conn, KubernetesLogsConnector)
    assert conn.source == Source.APPLOG

    with pytest.raises(MCAError):
        spec.build({"timestamps": "maybe"}, {"source": Source.APPLOG})
    with pytest.raises(MCAError):
        spec.build({"timeout_seconds": 0}, {"source": Source.APPLOG})


@pytest.mark.parametrize("value", [1, 1024, "1KB", "16MB", " 2 gb "])
def test_mca_build_accepts_a_positive_size(value):
    """Sizes the shared parser understands are accepted as the byte cap."""
    from attribution_blueprint import mca_registry

    spec = mca_registry.get_component("log", "kubernetes_logs")
    assert isinstance(spec.build({"max_bytes_per_pod": value}, {"source": Source.APPLOG}), KubernetesLogsConnector)


@pytest.mark.parametrize("value", ["abc", "12 parsecs", "", 3.5, 0, "0", "0KB", -1, "-5", [1], {"a": 1}])
def test_mca_build_rejects_an_invalid_size_naming_the_field(value):
    """A bad cap is a configuration error that names its field, not a raw exception."""
    from attribution_blueprint import mca_registry

    spec = mca_registry.get_component("log", "kubernetes_logs")
    with pytest.raises(MCAError, match=r"log\.applog\.kubernetes_logs\.max_bytes_per_pod"):
        spec.build({"max_bytes_per_pod": value}, {"source": Source.APPLOG})


def test_mca_build_names_the_source_in_a_size_error():
    """The error path carries the source the connector was built for."""
    from attribution_blueprint import mca_registry

    spec = mca_registry.get_component("log", "kubernetes_logs")
    with pytest.raises(MCAError, match=r"log\.syslog\.kubernetes_logs\.max_bytes_per_pod"):
        spec.build({"max_bytes_per_pod": "-1"}, {"source": Source.SYSLOG})


def test_a_missing_cap_means_no_cap_when_constructed_directly():
    """`None` keeps the whole log, as for kubectl_logs."""
    client = _two_pods()
    KubernetesLogsConnector(client=client, max_bytes_per_pod=None).collect(_ctx())
    assert {r["max_bytes"] for r in client.log_reads} == {None}


@pytest.mark.parametrize("value", [0, -1, "0", "-3KB"])
def test_a_non_positive_cap_is_rejected_when_constructed_directly(value):
    """The constructor enforces the same rule as the configuration."""
    with pytest.raises(ValueError):
        KubernetesLogsConnector(client=_two_pods(), max_bytes_per_pod=value)


def test_a_credential_failure_skips_the_pod_even_when_the_container_is_configured(mocker):
    """With `container` set there is no earlier pod read, so the first API call is the log read.

    A failure to load credentials must be skipped like any other read failure
    and not abort the whole collection with a raw exception.
    """
    from attribution_blueprint.kubernetes.client import KubernetesClient

    client = KubernetesClient()
    mocker.patch.object(client, "_build_client", side_effect=RuntimeError("no credentials"))
    obs = KubernetesLogsConnector(client=client, container="trainer").collect(_ctx())
    assert obs == []
