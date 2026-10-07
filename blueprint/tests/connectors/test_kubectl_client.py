# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest

from attribution_blueprint.connectors.execution_contexts.kubectl_client import (
    KubectlClient,
    KubectlError,
    KubectlNotFoundError,
)


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_command_carries_binary_flags_and_json_output(mocker):
    run = mocker.patch("subprocess.run", return_value=_completed('{"items": []}'))

    KubectlClient("/bin/kubectl", kubeconfig="/tmp/kc", context="ctx", timeout_seconds=7).list("pods", namespace="ns")

    cmd = run.call_args.args[0]
    assert cmd == [
        "/bin/kubectl",
        "--kubeconfig",
        "/tmp/kc",
        "--context",
        "ctx",
        "get",
        "pods",
        "-n",
        "ns",
        "-o",
        "json",
    ]
    assert run.call_args.kwargs["timeout"] == 7


def test_unset_kubeconfig_and_context_add_no_flags(mocker):
    run = mocker.patch("subprocess.run", return_value=_completed('{"items": []}'))

    KubectlClient().list("pods", namespace="ns")

    assert run.call_args.args[0] == ["kubectl", "get", "pods", "-n", "ns", "-o", "json"]


def test_get_one_scopes_to_namespace(mocker):
    run = mocker.patch("subprocess.run", return_value=_completed(json.dumps({"kind": "Pod"})))

    assert KubectlClient().get_one("pod", "p", namespace="ns") == {"kind": "Pod"}

    assert run.call_args.args[0][1:] == ["get", "pod", "p", "-n", "ns", "-o", "json"]


def test_list_uses_all_namespaces_and_selectors_when_unscoped(mocker):
    run = mocker.patch("subprocess.run", return_value=_completed(json.dumps({"items": [{"a": 1}]})))

    items = KubectlClient().list(
        "events",
        namespace=None,
        field_selector="reason=Scheduled",
        label_selector="app=train",
    )

    assert items == [{"a": 1}]
    assert run.call_args.args[0][1:] == [
        "get",
        "events",
        "--field-selector",
        "reason=Scheduled",
        "-l",
        "app=train",
        "-A",
        "-o",
        "json",
    ]


@pytest.mark.parametrize(
    "stdout",
    ["{}", "null", "[]", json.dumps({"items": None}), json.dumps({"items": {}})],
    ids=["empty-object", "null", "array", "null-items", "object-items"],
)
def test_list_raises_on_an_unexpected_response_shape(mocker, stdout):
    mocker.patch("subprocess.run", return_value=_completed(stdout))

    with pytest.raises(KubectlError, match="unexpected response shape"):
        KubectlClient().list("pods", namespace="ns")


def test_list_returns_an_empty_items_list(mocker):
    mocker.patch("subprocess.run", return_value=_completed(json.dumps({"items": []})))

    assert KubectlClient().list("pods", namespace="ns") == []


@pytest.mark.parametrize("stdout", ["null", "[]", '"text"'], ids=["null", "array", "string"])
def test_get_one_raises_when_the_response_is_not_an_object(mocker, stdout):
    mocker.patch("subprocess.run", return_value=_completed(stdout))

    with pytest.raises(KubectlError, match="unexpected response shape"):
        KubectlClient().get_one("pod", "p", namespace="ns")


def _fail(stderr):
    return subprocess.CalledProcessError(returncode=1, cmd=["kubectl"], stderr=stderr)


def test_get_one_returns_none_only_for_not_found(mocker):
    mocker.patch("subprocess.run", side_effect=_fail('Error from server (NotFound): pods "p" not found'))

    assert KubectlClient().get_one("pod", "p", namespace="ns") is None


def test_get_one_does_not_classify_other_not_found_text_as_absence(mocker):
    mocker.patch("subprocess.run", side_effect=_fail("proxy failed: upstream returned NotFound"))

    with pytest.raises(KubectlError):
        KubectlClient().get_one("pod", "p", namespace="ns")


@pytest.mark.parametrize(
    "failure",
    [
        _fail("Error from server (Forbidden): pods is forbidden"),
        subprocess.TimeoutExpired(cmd=["kubectl"], timeout=1),
    ],
    ids=["forbidden", "timeout"],
)
def test_get_one_and_list_raise_on_other_failures(mocker, failure):
    mocker.patch("subprocess.run", side_effect=failure)
    client = KubectlClient()

    with pytest.raises(KubectlError):
        client.get_one("pod", "p", namespace="ns")
    with pytest.raises(KubectlError):
        client.list("pods", namespace="ns")


def test_get_one_and_list_raise_on_malformed_output(mocker):
    mocker.patch("subprocess.run", return_value=_completed("not-json{"))
    client = KubectlClient()

    with pytest.raises(KubectlError, match="invalid JSON"):
        client.get_one("pod", "p", namespace="ns")
    with pytest.raises(KubectlError, match="invalid JSON"):
        client.list("pods", namespace="ns")


def test_list_raises_on_not_found(mocker):
    mocker.patch("subprocess.run", side_effect=_fail("Error from server (NotFound)"))

    with pytest.raises(KubectlNotFoundError):
        KubectlClient().list("pods", namespace="ns")
