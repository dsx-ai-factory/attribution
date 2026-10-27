# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from attribution_blueprint.connectors.execution_contexts.kueue.helpers import (
    pod_container_exit_reason,
)


def _pod(status=None):
    return {"status": status or {}}


class TestPodContainerExitReason:
    def test_no_status_returns_none(self):
        assert pod_container_exit_reason({}) is None

    def test_zero_exit_code_returns_none(self):
        pod = _pod({"containerStatuses": [{"state": {"terminated": {"exitCode": 0}}}]})
        assert pod_container_exit_reason(pod) is None

    def test_nonzero_exit_code_returns_formatted_reason(self):
        pod = _pod({"containerStatuses": [{"state": {"terminated": {"exitCode": 137}}}]})
        assert pod_container_exit_reason(pod) == "ExitCode:137"

    def test_non_terminated_container_ignored(self):
        pod = _pod({"containerStatuses": [{"state": {"running": {}}}]})
        assert pod_container_exit_reason(pod) is None

    def test_first_nonzero_exit_code_among_multiple_containers(self):
        pod = _pod(
            {
                "containerStatuses": [
                    {"state": {"terminated": {"exitCode": 0}}},
                    {"state": {"terminated": {"exitCode": 1}}},
                ]
            }
        )
        assert pod_container_exit_reason(pod) == "ExitCode:1"
