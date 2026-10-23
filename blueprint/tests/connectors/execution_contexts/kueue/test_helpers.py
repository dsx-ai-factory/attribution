# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_blueprint.connectors.execution_contexts.kueue.helpers import (
    pod_container_exit_reason,
    pod_node_failure_reason,
)


def _pod(status=None):
    return {"status": status or {}}


class TestPodNodeFailureReason:
    def test_no_status_returns_none(self):
        assert pod_node_failure_reason({}) is None

    def test_legacy_node_lost_reason(self):
        pod = _pod({"reason": "NodeLost"})
        assert pod_node_failure_reason(pod) == "NodeLost"

    def test_disruption_target_pod_gc_is_node_loss(self):
        pod = _pod(
            {
                "conditions": [
                    {
                        "type": "DisruptionTarget",
                        "status": "True",
                        "reason": "DeletionByPodGC",
                        "lastTransitionTime": "t1",
                    },
                ]
            }
        )
        assert pod_node_failure_reason(pod) == "DeletionByPodGC"

    @pytest.mark.parametrize("reason", ["PreemptionByScheduler", "DeletionByTaintManager"])
    def test_disruption_target_non_node_loss_reason_ignored(self, reason):
        pod = _pod(
            {
                "conditions": [
                    {
                        "type": "DisruptionTarget",
                        "status": "True",
                        "reason": reason,
                        "lastTransitionTime": "t1",
                    },
                ]
            }
        )
        assert pod_node_failure_reason(pod) is None

    def test_disruption_target_false_status_ignored(self):
        pod = _pod(
            {
                "conditions": [
                    {
                        "type": "DisruptionTarget",
                        "status": "False",
                        "reason": "DeletionByPodGC",
                        "lastTransitionTime": "t1",
                    },
                ]
            }
        )
        assert pod_node_failure_reason(pod) is None

    def test_most_recent_disruption_target_condition_wins(self):
        pod = _pod(
            {
                "conditions": [
                    {
                        "type": "DisruptionTarget",
                        "status": "True",
                        "reason": "PreemptionByScheduler",
                        "lastTransitionTime": "2026-05-01T00:00:00Z",
                    },
                    {
                        "type": "DisruptionTarget",
                        "status": "True",
                        "reason": "DeletionByPodGC",
                        "lastTransitionTime": "2026-05-01T00:05:00Z",
                    },
                ]
            }
        )
        assert pod_node_failure_reason(pod) == "DeletionByPodGC"


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
