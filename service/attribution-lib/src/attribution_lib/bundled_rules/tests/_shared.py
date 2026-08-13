# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared test fixtures for the bundled_rules test suite."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from attribution_lib.job_info import JobInfo, JobInfoBuilder

from test_support.helpers import make_node_to_ranks

_DEFAULT_START = datetime(2024, 1, 1, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
_DEFAULT_END = _DEFAULT_START + timedelta(hours=1)


def default_job_info(
    *,
    nodes: list[str] | None = None,
    ranks_per_node: int = 1,
    state: str = "FAILED",
    job_end_time: datetime | None = None,
    time_limit: int | None = 3600,
    exit_code_signal: int | None = 1,
    cancelled_by: str | None = None,
    exit_code: int | None = None,
    tags: frozenset[str] = frozenset(),
) -> JobInfo:
    """Canonical JobInfo for bundled_rules tests, built through the real JobInfoBuilder.

    Return a JobInfo containing fixed boilerplate values and knobs for frequently-varying
    fields in tests.
    """
    nodes_ = nodes if nodes is not None else ["node01"]
    interest_end = job_end_time if job_end_time is not None else _DEFAULT_END

    builder = JobInfoBuilder(
        id="123",
        type="batch",
        state=state,
        cluster="test",
        framework="pytorch",
        source_name="test",
        timezone=ZoneInfo("UTC"),
        job_start_time=_DEFAULT_START,
        node_to_ranks=make_node_to_ranks(nodes_, ranks_per_node),
    )
    builder.srun_cmd("srun dummy")
    if exit_code_signal is not None:
        builder.exit_code_signal(exit_code_signal)
    if exit_code is not None:
        builder.exit_code(exit_code)
    builder.tags(tags)
    if cancelled_by is not None:
        builder.cancelled_by(cancelled_by)
    builder.interest_end_time(interest_end)
    if job_end_time is not None:
        builder.job_end_time(job_end_time)
    if time_limit is not None:
        builder.time_limit(time_limit)
    builder.num_gpus(8)
    return builder.build()
