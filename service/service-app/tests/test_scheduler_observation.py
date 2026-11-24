# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that scheduler facts reach the rules through a `scheduler` observation over the v2 API."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import make_up_attributor_info, post_attributor


@pytest.mark.parametrize(
    ("facts", "outcome", "cancellation_actors"),
    [
        ([], "Fail InfraSW", []),
        (
            [
                "allocation failure: allocation=allocation state=NODE_FAIL node=node-a",
                "execution cancellation: execution=execution cancelled_by=user-a",
            ],
            "Cancelled by User",
            ["User"],
        ),
    ],
    ids=["no-scheduler-observation", "cancelled-execution-in-failed-allocation"],
)
def test_scheduler_observation_supplies_facts_that_the_request_cannot_carry(
    client, facts, outcome, cancellation_actors
):
    """A cancellation of the execution replaces the node failure that the allocation reported."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    info = make_up_attributor_info(
        "v2",
        id="execution",
        nodes={"node-b": [0]},
        status="NODE_FAIL",
        failed_node="node-a",
        exit_code_signal=0,
        job_start_time=start,
        job_end_time=end,
        timezone="UTC",
    )
    att_id = post_attributor(client, info)
    if facts != []:
        response = client.post(
            f"/v2/attributor/{att_id}/observation",
            json={
                "context": {
                    "source": "scheduler",
                    "time_interval": {"start": start.isoformat(), "end": (end + timedelta(minutes=1)).isoformat()},
                },
                "body": json.dumps([{"body": fact, "ts": int(end.timestamp() * 1e9)} for fact in facts]),
            },
        )
        assert response.status_code == 201, response.text
        assert response.json()["ingest"] == {"accepted": 2, "dropped": {}}

    response = client.get(f"/v2/attributor/{att_id}/attribution")

    assert response.status_code == 200, response.text
    verdict = response.json()["verdict"]
    assert verdict["attribution_outcome"] == outcome
    assert verdict["cancellation_actors"] == cancellation_actors
