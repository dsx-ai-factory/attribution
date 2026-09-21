# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for KubectlLogsConnector.

The connector shells to `kubectl logs <pod> ...` for each pod listed on the
ExecutionContext's `pod_names` field. Tests stub `subprocess.run` and inspect
the resulting argv plus the emitted Observations.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pytest

from attribution_blueprint.connectors.logs.kubectl_logs import KubectlLogsConnector
from attribution_blueprint.types import Format, Source


def _text(observation: dict) -> str:
    """Rejoin an APPLOG observation's per-line entries into the kubectl stdout."""
    return "\n".join(entry["body"] for entry in json.loads(observation["body"]))


def _ctx(**extra) -> dict:
    base = {
        "id": "uid-1",
        "job_id": "kueue-demo/jobset-train-abc",
        "namespace": "kueue-demo",
        "nodes": ["kueue-demo-worker"],
        "start_time": datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 5, 1, 12, 30, 0, tzinfo=timezone.utc),
        "pod_names": ["train-abc-trainer-0-0", "train-abc-trainer-0-1"],
    }
    base.update(extra)
    return base


def _stub_logs(mocker, by_pod: dict[str, str], *, errors: set[str] | None = None):
    """Make `subprocess.run` return per-pod stdout, or raise for pods in
    `errors`."""
    errors = errors or set()

    def _side_effect(cmd, **_kwargs):
        # `kubectl logs <pod> [flags...]` - first positional after "logs"
        # is the pod name.
        argv = list(cmd)
        try:
            pod = argv[argv.index("logs") + 1]
        except (ValueError, IndexError):  # pragma: no cover - defensive
            pod = ""
        if pod in errors:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr=f"pod {pod} not found")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=by_pod.get(pod, ""), stderr="")

    return mocker.patch("subprocess.run", side_effect=_side_effect)


def test_emits_one_observation_per_pod(mocker):
    """Happy path: both pods return non-empty stdout → two Observations,
    body matches kubectl stdout exactly, source/format/resources are set."""
    _stub_logs(
        mocker,
        {
            "train-abc-trainer-0-0": "rank=0\nstep=100\n",
            "train-abc-trainer-0-1": "rank=1\nCUDA error: device-side assert\nexit 42\n",
        },
    )
    conn = KubectlLogsConnector()
    obs = conn.collect(_ctx())

    assert len(obs) == 2
    bodies = {_text(o) for o in obs}
    assert "rank=0\nstep=100\n" in bodies
    assert any("CUDA error: device-side assert" in b for b in bodies)
    for o in obs:
        assert o["context"]["source"] == Source.APPLOG
        assert o["context"]["format"] == Format.ENTRY_TIMESERIES
        assert o["context"]["resources"] == "AllJobResources"
        assert o["context"]["time_interval"] == {"start": None, "end": None}


def test_skips_empty_bodies(mocker):
    """A pod with no output (kubectl returns 0 but empty stdout) must not
    produce an Observation - submitting empty bodies to the attribution
    service is pure noise."""
    _stub_logs(
        mocker,
        {"train-abc-trainer-0-0": "", "train-abc-trainer-0-1": "rank=1\n"},
    )
    conn = KubectlLogsConnector()
    obs = conn.collect(_ctx())
    assert len(obs) == 1
    assert _text(obs[0]) == "rank=1\n"


def test_continues_when_one_pod_errors(mocker):
    """A `kubectl logs` failure on one pod (e.g. NotFound after gc) must
    NOT abort the whole collection - the failing pod is warned-and-skipped
    and the remaining pods still produce Observations."""
    _stub_logs(
        mocker,
        {"train-abc-trainer-0-1": "rank=1 logs survived"},
        errors={"train-abc-trainer-0-0"},
    )
    conn = KubectlLogsConnector()
    obs = conn.collect(_ctx())
    assert len(obs) == 1
    assert _text(obs[0]) == "rank=1 logs survived"


def test_returns_empty_when_pod_names_absent(mocker):
    """If the ECA didn't populate `pod_names` (e.g. all pods gc'd before
    discovery), the connector returns [] without issuing any kubectl call."""
    run_mock = _stub_logs(mocker, {})
    conn = KubectlLogsConnector()
    ctx = _ctx()
    ctx.pop("pod_names")
    obs = conn.collect(ctx)
    assert obs == []
    assert run_mock.call_count == 0


def test_argv_uses_all_containers_and_timestamps_by_default(mocker):
    """Distributed-training pods routinely have a sidecar (init container,
    logging agent, etc.); collecting only the primary container hides
    crash output written to other containers. Default to
    `--all-containers=true` and `--timestamps=true` so the body fed to
    the attribution service has the per-line timestamp prefix rules can
    match on."""
    run_mock = _stub_logs(mocker, {"train-abc-trainer-0-0": "x"})
    KubectlLogsConnector().collect(_ctx(pod_names=["train-abc-trainer-0-0"]))
    argv = run_mock.call_args_list[0].args[0]
    assert "--all-containers=true" in argv
    assert "--timestamps=true" in argv
    assert "-n" in argv and "kueue-demo" in argv


def test_argv_uses_container_filter_when_configured(mocker):
    """When `container` is set, the connector requests only that container's
    logs and drops `--all-containers=true`."""
    run_mock = _stub_logs(mocker, {"train-abc-trainer-0-0": "x"})
    KubectlLogsConnector(container="trainer").collect(_ctx(pod_names=["train-abc-trainer-0-0"]))
    argv = run_mock.call_args_list[0].args[0]
    assert "-c" in argv and "trainer" in argv
    assert "--all-containers=true" not in argv


def test_argv_adds_previous_when_configured(mocker):
    """`previous=True` requests the previous container instance's logs
    (e.g. the one that crashed before a restart). Off by default - most
    Workload failures are terminal and `--previous` would return nothing."""
    run_mock = _stub_logs(mocker, {"train-abc-trainer-0-0": "x"})
    KubectlLogsConnector(previous=True).collect(_ctx(pod_names=["train-abc-trainer-0-0"]))
    argv = run_mock.call_args_list[0].args[0]
    assert "--previous" in argv


def test_argv_passes_kubeconfig_and_context(mocker):
    run_mock = _stub_logs(mocker, {"train-abc-trainer-0-0": "x"})
    conn = KubectlLogsConnector(
        kubectl_path="/usr/local/bin/kubectl",
        kubeconfig="/tmp/kc",
        context="kind-kueue-demo",
    )
    conn.collect(_ctx(pod_names=["train-abc-trainer-0-0"]))
    argv = run_mock.call_args_list[0].args[0]
    assert argv[0] == "/usr/local/bin/kubectl"
    assert "--kubeconfig" in argv and "/tmp/kc" in argv
    assert "--context" in argv and "kind-kueue-demo" in argv


def test_truncates_large_bodies_to_last_n_bytes(mocker):
    """Multi-GB training logs would OOM the blueprint and dwarf any
    actually-useful symptom signal in the upload to the attribution
    service. Cap the per-pod body length, keeping the TAIL (where crash
    symptoms typically land) - matches `local_file`'s discipline."""
    big = "noise\n" * 10000 + "CUDA error: device-side assert\nexit 42\n"
    _stub_logs(mocker, {"train-abc-trainer-0-0": big})
    conn = KubectlLogsConnector(max_bytes_per_pod=64)
    obs = conn.collect(_ctx(pod_names=["train-abc-trainer-0-0"]))
    assert len(obs) == 1
    body = _text(obs[0])
    assert len(body.encode("utf-8")) <= 64
    # The cap keeps the tail, so the crash line survives.
    assert "exit 42" in body
    assert "CUDA error" in body
    # 60000-byte noise prefix is fully discarded; only a tail of "noise"
    # lines that still fit under the cap (alongside the crash output)
    # survives. The byte-length assertion above is the load-bearing check.


def test_since_time_bounds_to_context_start(mocker):
    """`--since-time=<rfc3339>` keeps the upload small by skipping log
    lines from before the workload's evidence window. Default behavior
    uses context.start_time; the connector serializes it without
    sub-second precision (kubectl rejects fractional seconds)."""
    run_mock = _stub_logs(mocker, {"train-abc-trainer-0-0": "x"})
    KubectlLogsConnector(since_context_start=True).collect(_ctx(pod_names=["train-abc-trainer-0-0"]))
    argv = run_mock.call_args_list[0].args[0]
    assert "--since-time=2026-05-01T12:00:00Z" in argv


def test_source_can_be_overridden_to_syslog(mocker):
    """`source` defaults to APPLOG (container stdout = application log).
    A user pointing the connector at a node-agent's stdout (e.g. a
    DaemonSet that tails kernel ring buffer) can switch the source to
    SYSLOG so the right service-side rules fire."""
    _stub_logs(mocker, {"train-abc-trainer-0-0": "x"})
    conn = KubectlLogsConnector(source=Source.SYSLOG)
    obs = conn.collect(_ctx(pod_names=["train-abc-trainer-0-0"]))
    assert obs[0]["context"]["source"] == Source.SYSLOG
    assert obs[0]["context"]["format"] == Format.RAW_LOG
    assert obs[0]["context"]["time_interval"]["start"] == datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert obs[0]["body"] == "x"


def test_kubectl_timeout_returns_empty_for_that_pod(mocker):
    """If kubectl hangs and is killed by `subprocess.run(..., timeout=N)`,
    the TimeoutExpired must surface as a per-pod warn-and-skip - not as a
    crash that aborts the whole collection."""

    def _side_effect(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    mocker.patch("subprocess.run", side_effect=_side_effect)
    conn = KubectlLogsConnector(timeout_seconds=5)
    obs = conn.collect(_ctx(pod_names=["p1", "p2"]))
    assert obs == []


def test_rejects_naive_datetimes():
    """`assert_aware` discipline - naive start/end would break the
    `--since-time` rendering and the service rejects naive observations
    anyway."""
    conn = KubectlLogsConnector()
    ctx = _ctx()
    ctx["start_time"] = datetime(2026, 5, 1, 12, 0, 0)  # naive
    with pytest.raises(ValueError):
        conn.collect(ctx)


def test_namespace_absent_returns_empty(mocker):
    """`kubectl logs <pod>` without `-n <ns>` would target the default
    namespace and almost certainly fetch the wrong logs. If the ECA
    didn't stamp a namespace, refuse to issue any kubectl call rather
    than producing misleading evidence."""
    run_mock = _stub_logs(mocker, {})
    conn = KubectlLogsConnector()
    ctx = _ctx()
    ctx["namespace"] = ""
    obs = conn.collect(ctx)
    assert obs == []
    assert run_mock.call_count == 0


def test_mca_registration(mocker):
    """The MCA registry must wire `log.kubectl_logs` to the build function
    so configs can select this connector via TOML / CLI / env."""
    from attribution_blueprint import mca_registry

    spec = mca_registry.get_component("log", "kubectl_logs")
    assert spec.framework == "log"
    assert spec.name == "kubectl_logs"
    # Schema must expose the configurable knobs so the help-discovery and
    # env-var passthrough machinery can route values to the builder.
    assert {"kubectl_path", "container", "previous", "since_context_start"}.issubset(spec.schema)
