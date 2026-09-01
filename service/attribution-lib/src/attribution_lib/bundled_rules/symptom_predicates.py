# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditional-discard predicates (discrete).

Cascade rules reference these by name in ``conditional_discards``. Each registers
onto the loading RuleEngine under its function name via ``@symptom_predicate`` when
this rule file loads; the compiled symptom registry resolves the name to the
callable. Moved here from the core library so the domain signatures/thresholds live
with the rules.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

from attribution_lib.job_info import JobInfo
from attribution_lib.rules import symptom_predicate
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


@symptom_predicate
def job_elapsed_below_eap_observation_window(symptoms: SymptomSet[Any], job_info: Optional[JobInfo]) -> bool:
    """Return True when the job is shorter than the EAP observation window."""
    elapsed = _elapsed_hours(job_info)
    return elapsed is not None and elapsed < 2.0


@symptom_predicate
def job_exceeded_time_limit(symptoms: SymptomSet[Any], job_info: Optional[JobInfo]) -> bool:
    """Return True when the job exceeded its time limit window."""
    if job_info is None or job_info.time_limit is None:
        return False
    duration = job_info.interest_end_time - job_info.interest_start_time
    return duration > timedelta(minutes=job_info.time_limit)


@symptom_predicate
def contains_ibv_modify_qp_failure(symptoms: SymptomSet[Any], job_info: Optional[JobInfo]) -> bool:
    """Check SymptomSet entries for the ibv_modify_qp failure signature."""
    pattern = "Call to ibv_modify_qp failed"
    return any(pattern in str(sym) for sym in symptoms.symptoms())
