# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from attribution_lib.bundled_rules.common.scheduler_formatting import (
    CANCELLED_BY_SOURCE,
    LAST_RESORT_SOURCE,
    NODE_FAILURE,
    NODE_FAILURE_IDS,
    PREEMPTED_ROOT_CAUSE,
    PREEMPTED_RULE,
    SHORTER_TIMEOUT_ROOT_CAUSE,
    SHORTER_TIMEOUT_RULE,
    STATE_SOURCE,
    TIMEOUT_ROOT_CAUSE,
    render,
)
from attribution_lib.job_info import Scheduler


def test_slurm_wording_remains_byte_compatible() -> None:
    assert render(NODE_FAILURE, Scheduler.SLURM) == "Node failure reported by slurm"
    assert render(STATE_SOURCE, Scheduler.SLURM) == "slurm state"
    assert render(LAST_RESORT_SOURCE, Scheduler.SLURM) == "last resort; slurm status"
    assert render(CANCELLED_BY_SOURCE, Scheduler.SLURM) == "slurmdb cancelled_by"
    assert render(PREEMPTED_ROOT_CAUSE, Scheduler.SLURM) == "SLURM preempted"
    assert render(PREEMPTED_RULE, Scheduler.SLURM) == "reclassified: slurm preempted"
    assert render(SHORTER_TIMEOUT_ROOT_CAUSE, Scheduler.SLURM) == (
        "SLURM timeout with shorter user distributed timeout"
    )
    assert render(SHORTER_TIMEOUT_RULE, Scheduler.SLURM) == "reclassified: pytorch timeout < slurm limit"
    assert render(TIMEOUT_ROOT_CAUSE, Scheduler.SLURM) == "SLURM timeout (no other errors)"


@pytest.mark.parametrize("scheduler", [Scheduler.UNKNOWN, Scheduler.KUEUE, Scheduler.KAI, Scheduler.RUNAPI])
def test_non_slurm_wording_uses_lowercase_scheduler_token(scheduler: Scheduler) -> None:
    token = scheduler.value or "unknown"

    assert render(NODE_FAILURE, scheduler) == f"Node failure reported by {token}"
    assert render(STATE_SOURCE, scheduler) == f"{token} state"
    assert render(LAST_RESORT_SOURCE, scheduler) == f"last resort; {token} status"
    assert render(CANCELLED_BY_SOURCE, scheduler) == f"{token} cancelled_by"
    assert render(PREEMPTED_ROOT_CAUSE, scheduler) == f"{token} preempted"
    assert render(PREEMPTED_RULE, scheduler) == f"reclassified: {token} preempted"
    assert render(SHORTER_TIMEOUT_ROOT_CAUSE, scheduler) == (
        f"{token} timeout with shorter user distributed timeout"
    )
    assert render(SHORTER_TIMEOUT_RULE, scheduler) == f"reclassified: pytorch timeout < {token} limit"
    assert render(TIMEOUT_ROOT_CAUSE, scheduler) == f"{token} timeout (no other errors)"


def test_node_failure_variants_cover_every_scheduler() -> None:
    assert NODE_FAILURE_IDS == (
        "Node failure reported by unknown",
        "Node failure reported by slurm",
        "Node failure reported by kueue",
        "Node failure reported by kai",
        "Node failure reported by runapi",
    )
