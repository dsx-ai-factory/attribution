# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.kubernetes.daemon.cli import _parser


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
