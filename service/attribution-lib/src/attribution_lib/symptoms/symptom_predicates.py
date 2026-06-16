# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from attribution_lib.job_info import JobInfo
from attribution_lib.sources import SymptomSet


def _elapsed_hours(job_info: Optional[JobInfo]) -> Optional[float]:
    """Return the elapsed job duration in hours, if both endpoints are known."""
    if job_info is None:
        return None
    end_time = job_info.job_end_time or job_info.interest_end_time
    if end_time is None:
        return None
    delta = end_time - job_info.job_start_time
    return delta.total_seconds() / 3600


def job_elapsed_under_2_hours(symptoms: SymptomSet[str], job_info: Optional[JobInfo]) -> bool:
    """Return True when the job ran for less than two hours."""
    elapsed = _elapsed_hours(job_info)
    return elapsed is not None and elapsed < 2.0


def job_exceeded_time_limit(symptoms: SymptomSet[str], job_info: Optional[JobInfo]) -> bool:
    """Return True when the job exceeded its time limit window."""
    if job_info is None or job_info.time_limit is None:
        return False
    duration = job_info.interest_end_time - job_info.interest_start_time
    return duration > timedelta(minutes=job_info.time_limit)


def contains_ibv_modify_qp_failure(symptoms: SymptomSet[str], job_info: Optional[JobInfo]) -> bool:
    """Check SymptomSet entries for the ibv_modify_qp failure signature."""
    pattern = "Call to ibv_modify_qp failed"
    return any(pattern in str(sym) for sym in symptoms.symptoms())
