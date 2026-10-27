# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

from attribution_blueprint.connectors.execution_contexts.kueue.connector import (
    KueueExecutionContextAdapter,
)


def test_live_timeout_reconciles_same_execution_without_duplicate():
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

    assert len(reconciled) == 2
    assert reconciled[0]["attempt_id"] == "previous-attempt"
    assert reconciled[0]["state"] == "COMPLETED"
    assert reconciled[1]["attempt_id"] == "retained-attempt"
    assert reconciled[1]["state"] == "TIMEOUT"
