# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from attribution_blueprint.kubernetes.daemon.cli import (
    _API_REQUEST_TIMEOUT_SECONDS,
    _parser,
    _set_default_api_timeout,
)


def test_kubernetes_api_calls_have_bounded_connect_and_read_timeouts():
    calls = []
    api_client = SimpleNamespace(
        call_api=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    _set_default_api_timeout(api_client)

    api_client.call_api("/api/v1/pods")
    api_client.call_api("/api/v1/jobs", _request_timeout=(2, 3))

    assert calls[0][1]["_request_timeout"] == _API_REQUEST_TIMEOUT_SECONDS
    assert calls[1][1]["_request_timeout"] == (2, 3)


def test_scheduler_integrations_are_explicitly_disabled_by_default():
    args = _parser().parse_args(["--namespace", "team-a"])

    assert args.enable_kueue is False
    assert args.enable_kai is False


def test_scheduler_integrations_and_api_versions_are_configurable():
    args = _parser().parse_args(
        [
            "--namespace",
            "team-a",
            "--enable-kueue",
            "--kueue-workload-version",
            "v1beta1",
            "--enable-kai",
            "--kai-pod-group-version",
            "v2alpha2",
        ]
    )

    assert args.enable_kueue is True
    assert args.kueue_workload_version == "v1beta1"
    assert args.enable_kai is True
    assert args.kai_pod_group_version == "v2alpha2"
