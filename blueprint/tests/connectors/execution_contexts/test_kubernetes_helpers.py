# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_blueprint.connectors.execution_contexts.kubernetes_helpers import (
    earliest_job_failure_reason,
    job_time_limit_minutes,
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


def test_earliest_job_failure_reason():
    jobs = [
        {
            "status": {
                "conditions": [
                    {
                        "type": "Failed",
                        "status": "True",
                        "reason": "BackoffLimitExceeded",
                        "lastTransitionTime": "2026-05-01T00:05:00Z",
                    }
                ]
            }
        },
        {
            "status": {
                "conditions": [
                    {
                        "type": "FailureTarget",
                        "status": "True",
                        "reason": "DeadlineExceeded",
                        "lastTransitionTime": "2026-05-01T00:01:00Z",
                    }
                ]
            }
        },
    ]

    assert earliest_job_failure_reason(jobs) == "DeadlineExceeded"


@pytest.mark.parametrize(
    "deadlines, expected",
    [
        ([30], 1),
        ([61, 120], 2),
        ([None, 0, -1, True], None),
    ],
)
def test_job_time_limit_minutes(deadlines, expected):
    jobs = [{"spec": {"activeDeadlineSeconds": deadline}} for deadline in deadlines]

    assert job_time_limit_minutes(jobs) == expected
