# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest

from attribution_blueprint.connectors.execution_contexts.kubectl_client import KubectlClient


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_command_carries_binary_flags_and_json_output(mocker):
    run = mocker.patch("subprocess.run", return_value=_completed("{}"))

    KubectlClient("/bin/kubectl", kubeconfig="/tmp/kc", context="ctx", timeout_seconds=7).run_json(["get", "pods"])

    cmd = run.call_args.args[0]
    assert cmd == ["/bin/kubectl", "--kubeconfig", "/tmp/kc", "--context", "ctx", "get", "pods", "-o", "json"]
    assert run.call_args.kwargs["timeout"] == 7


def test_unset_kubeconfig_and_context_add_no_flags(mocker):
    run = mocker.patch("subprocess.run", return_value=_completed("{}"))

    KubectlClient().run_json(["get", "pods"])

    assert run.call_args.args[0] == ["kubectl", "get", "pods", "-o", "json"]


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(returncode=1, cmd=["kubectl"], stderr="NotFound"),
        subprocess.TimeoutExpired(cmd=["kubectl"], timeout=1),
    ],
    ids=["exit-nonzero", "timeout"],
)
def test_failures_are_swallowed_unless_required(mocker, failure):
    mocker.patch("subprocess.run", side_effect=failure)
    client = KubectlClient()

    assert client.run_json(["get", "pods"]) is None
    with pytest.raises(RuntimeError, match="required kubectl query"):
        client.run_json(["get", "pods"], required=True)


def test_invalid_json_is_swallowed_unless_required(mocker):
    mocker.patch("subprocess.run", return_value=_completed("not-json{"))
    client = KubectlClient()

    assert client.run_json(["get", "pods"]) is None
    with pytest.raises(RuntimeError, match="invalid JSON"):
        client.run_json(["get", "pods"], required=True)


def test_get_one_scopes_to_namespace(mocker):
    run = mocker.patch("subprocess.run", return_value=_completed(json.dumps({"kind": "Pod"})))

    assert KubectlClient().get_one("pod", "p", namespace="ns") == {"kind": "Pod"}

    assert run.call_args.args[0][1:] == ["get", "pod", "p", "-n", "ns", "-o", "json"]


def test_list_uses_all_namespaces_when_unscoped_and_field_selector(mocker):
    run = mocker.patch("subprocess.run", return_value=_completed(json.dumps({"items": [{"a": 1}]})))

    items = KubectlClient().list("events", namespace=None, field_selector="reason=Scheduled")

    assert items == [{"a": 1}]
    assert run.call_args.args[0][1:] == ["get", "events", "--field-selector", "reason=Scheduled", "-A", "-o", "json"]


@pytest.mark.parametrize("stdout", ["{}", json.dumps({"items": None}), json.dumps({"items": {}})])
def test_list_returns_empty_without_a_list_of_items(mocker, stdout):
    mocker.patch("subprocess.run", return_value=_completed(stdout))

    assert KubectlClient().list("pods", namespace="ns") == []
