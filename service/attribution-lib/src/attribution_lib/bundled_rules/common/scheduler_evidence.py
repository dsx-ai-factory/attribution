# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler facts that a client sends in a SCHEDULER observation.

The extractors in ``scheduler_leaf_rules.toml`` turn each entry of the observation
into a generic event. The refiners read the events through this module.
"""

import logging

from attribution_lib.event_helpers import query_events_with_source
from attribution_lib.events import EventStream
from attribution_lib.execution_cancellation import ExecutionCancellation
from attribution_lib.job_info import JobInfo
from attribution_lib.rule_types.definitions import RefinerContext
from attribution_lib.scheduler_failure import SchedulerFailure
from attribution_lib.sources import Source

JOB_TAG_EVENT = "Scheduler.job.tag"
JOB_EXIT_CODE_EVENT = "Scheduler.job.exit_code"
JOB_GPU_COUNT_EVENT = "Scheduler.job.gpu_count"
ALLOCATION_FAILURE_EVENT = "Scheduler.allocation.failure"
EXECUTION_CANCELLATION_EVENT = "Scheduler.execution.cancellation"
SCHEDULER_EVENT_TYPES = (
    JOB_TAG_EVENT,
    JOB_EXIT_CODE_EVENT,
    JOB_GPU_COUNT_EVENT,
    ALLOCATION_FAILURE_EVENT,
    EXECUTION_CANCELLATION_EVENT,
)

# Observations in the order that they arrived, then entries in the order of each observation.
# The observation ID is a hash, so the first event ID of each observation gives its arrival.
_ARRIVAL_ORDER = "MIN(e.id) OVER (PARTITION BY e.obs_context_id), e.line_num, e.id"

logger = logging.getLogger(__name__)


def scheduler_job_info(context: RefinerContext) -> JobInfo | None:
    """Return the job of a refiner context with its scheduler observation applied.

    Args:
        context: Refiner context with the job and the event stream.

    Returns:
        The job with each fact from the scheduler observation, or None without a job.
    """
    if context.job_info is None:
        return None
    return job_with_scheduler_evidence(context.job_info, context.event_stream)


def job_with_scheduler_evidence(job_info: JobInfo, event_stream: EventStream | None) -> JobInfo:
    """Return the job with the facts from its scheduler observation.

    A fact in the observation replaces the JobInfo input of the same name. Tags are
    added to the JobInfo tags. A later entry of the same kind replaces an earlier entry,
    also when the entries arrive in different observations. A cancellation entry for
    another execution or without a timestamp is ignored with a warning.
    A client that calls the library directly can still fill the JobInfo inputs, so a
    job without a scheduler observation is returned unchanged.

    Args:
        job_info: Job as the client described it.
        event_stream: Event stream of the job, or None when the caller has none.

    Returns:
        The job with ``tags``, ``exit_code``, ``num_gpus``, ``allocation_failure`` and
        ``execution_cancellation`` from the scheduler observation.
    """
    if event_stream is None:
        return job_info
    names = {event_stream.mgr.types.get_id(name): name for name in SCHEDULER_EVENT_TYPES}
    events = query_events_with_source(event_stream, list(names), order_by=_ARRIVAL_ORDER)
    for _event_id, event_type, payload, source in events:
        job_info = _with_fact(job_info, names[event_type], payload, source)
    return job_info


def _with_fact(job_info: JobInfo, event: str, payload: dict[str, str], source: Source) -> JobInfo:
    """Apply one scheduler event to the job.

    Args:
        job_info: Job to update.
        event: Event type name of the fact.
        payload: Payload that the extractor captured.
        source: Provenance of the observation entry.

    Returns:
        The job with the fact applied.
    """
    if event == JOB_TAG_EVENT:
        return job_info._replace(tags=job_info.tags | {payload["tag"]})
    if event == JOB_EXIT_CODE_EVENT:
        return job_info._replace(exit_code=int(payload["exit_code"]))
    if event == JOB_GPU_COUNT_EVENT:
        return job_info._replace(num_gpus=int(payload["gpu_count"]))
    if event == ALLOCATION_FAILURE_EVENT:
        failure = SchedulerFailure(payload["allocation_id"], payload["state"], payload["node"], source)
        return job_info._replace(allocation_failure=failure)
    if payload["execution_id"] != job_info.id:
        logger.warning(
            "Ignoring a cancellation of execution %s in the observation of %s", payload["execution_id"], job_info.id
        )
        return job_info
    try:
        cancellation = ExecutionCancellation(payload["execution_id"], payload["cancelled_by"], source)
    except ValueError as error:
        logger.warning("Ignoring a cancellation entry of %s: %s", job_info.id, error)
        return job_info
    return job_info._replace(execution_cancellation=cancellation)
