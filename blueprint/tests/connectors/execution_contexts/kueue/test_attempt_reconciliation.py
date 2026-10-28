# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

from attribution_blueprint.connectors.execution_contexts.kueue.connector import (
    KueueExecutionContextAdapter,
)


def test_live_context_with_different_attempt_id_is_preserved():
    start = datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)
    previous = {
        "workload_uid": "workload-uid",
        "state": "COMPLETED",
        "id": "previous-attempt",
        "attempt_id": "previous-attempt",
        "start_time": datetime(2026, 10, 14, 9, 0, tzinfo=timezone.utc),
        "end_time": start,
        "kueue_state": "FINISHED_SUCCEEDED",
    }
    retained = {
        "workload_uid": "workload-uid",
        "state": "FAILED",
        "id": "retained-attempt",
        "attempt_id": "retained-attempt",
        "start_time": start,
        "end_time": datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
        "kueue_state": "FINISHED_FAILED",
    }
    live = {
        "workload_uid": "workload-uid",
        "state": "TIMEOUT",
        "id": "workload-uid",
        "attempt_id": "workload-uid",
        "start_time": start,
        "end_time": datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
        "kueue_state": "FINISHED_FAILED",
    }

    reconciled = KueueExecutionContextAdapter._reconcile_live_contexts(
        [previous, retained],
        [live],
    )

    assert len(reconciled) == 3
    assert reconciled[0]["attempt_id"] == "previous-attempt"
    assert reconciled[0]["state"] == "COMPLETED"
    assert reconciled[1]["attempt_id"] == "retained-attempt"
    assert reconciled[1]["state"] == "FAILED"
    assert reconciled[2]["attempt_id"] == "workload-uid"
    assert reconciled[2]["state"] == "TIMEOUT"


def test_live_context_reconciles_only_matching_attempt_id():
    start = datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)
    retained = {
        "workload_uid": "workload-uid",
        "state": "FAILED",
        "id": "attempt-1",
        "attempt_id": "attempt-1",
        "start_time": start,
        "end_time": datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
        "kueue_state": "FINISHED_FAILED",
        "nodes": ["old-node"],
        "pod_names": ["old-pod"],
    }
    live = {
        **retained,
        "state": "TIMEOUT",
        "end_time": datetime(2026, 10, 14, 11, 30, tzinfo=timezone.utc),
        "nodes": ["new-node"],
        "pod_names": ["new-pod"],
    }

    [reconciled] = KueueExecutionContextAdapter._reconcile_live_contexts(
        [retained],
        [live],
    )

    assert reconciled["state"] == "TIMEOUT"
    assert reconciled["end_time"] == live["end_time"]
    # Retained membership is preserved because it is scoped from history.
    assert reconciled["nodes"] == ["old-node"]
    assert reconciled["pod_names"] == ["old-pod"]


def test_live_running_does_not_downgrade_retained_node_fail():
    start = datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)
    retained = {
        "workload_uid": "workload-uid",
        "state": "NODE_FAIL",
        "failed_node": "node-failed",
        "id": "attempt-1",
        "attempt_id": "attempt-1",
        "start_time": start,
        "end_time": datetime(2026, 10, 14, 10, 30, tzinfo=timezone.utc),
        "kueue_state": "FINISHED_FAILED",
    }
    live = {
        **retained,
        "state": "RUNNING",
        "kueue_state": "RUNNING",
        "end_time": datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc),
    }

    [reconciled] = KueueExecutionContextAdapter._reconcile_live_contexts(
        [retained],
        [live],
    )

    assert reconciled["state"] == "NODE_FAIL"
    assert reconciled["failed_node"] == "node-failed"
    assert reconciled["end_time"] == live["end_time"]
