# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render scheduler-aware strings used by bundled rules."""

from attribution_lib.job_info import Scheduler

NODE_FAILURE_PREFIX = "Node failure reported by"
NODE_FAILURE = f"{NODE_FAILURE_PREFIX} {{scheduler}}"
STATE_SOURCE = "{scheduler} state"
LAST_RESORT_SOURCE = "last resort; {scheduler} status"
CANCELLED_BY_SOURCE = "{scheduler} cancelled_by"
PREEMPTED_ROOT_CAUSE = "{scheduler} preempted"
PREEMPTED_RULE = "reclassified: {scheduler} preempted"
TIMEOUT_ROOT_CAUSE = "{scheduler} timeout (no other errors)"
SHORTER_TIMEOUT_ROOT_CAUSE = "{scheduler} timeout with shorter user distributed timeout"
SHORTER_TIMEOUT_RULE = "reclassified: pytorch timeout < {scheduler} limit"

# Slurm strings are existing serialized contracts. Templates can render new
# schedulers uniformly, but these historical exceptions must remain byte-identical.
_SLURM_COMPATIBILITY = {
    CANCELLED_BY_SOURCE: "slurmdb cancelled_by",
    PREEMPTED_ROOT_CAUSE: "SLURM preempted",
    TIMEOUT_ROOT_CAUSE: "SLURM timeout (no other errors)",
    SHORTER_TIMEOUT_ROOT_CAUSE: "SLURM timeout with shorter user distributed timeout",
}


def render(template: str, scheduler: Scheduler) -> str:
    """Render a bundled-rule phrase without changing legacy Slurm output.

    Args:
        template: Named scheduler template declared in this module.
        scheduler: Normalized scheduler identity from ``JobInfo``.

    Returns:
        The legacy fixed string for a Slurm compatibility exception. Otherwise,
        the template populated with the scheduler's lowercase wire token, or
        ``unknown`` when no scheduler identity was supplied.
    """
    if scheduler is Scheduler.SLURM and template in _SLURM_COMPATIBILITY:
        return _SLURM_COMPATIBILITY[template]
    return template.format(scheduler=scheduler.value or "unknown")


NODE_FAILURE_IDS = tuple(render(NODE_FAILURE, scheduler) for scheduler in Scheduler)
STATE_SOURCES = tuple(render(STATE_SOURCE, scheduler) for scheduler in Scheduler)
LAST_RESORT_SOURCES = tuple(render(LAST_RESORT_SOURCE, scheduler) for scheduler in Scheduler)
CANCELLED_BY_SOURCES = tuple(render(CANCELLED_BY_SOURCE, scheduler) for scheduler in Scheduler)
PREEMPTED_RULES = tuple(render(PREEMPTED_RULE, scheduler) for scheduler in Scheduler)
SHORTER_TIMEOUT_RULES = tuple(render(SHORTER_TIMEOUT_RULE, scheduler) for scheduler in Scheduler)
