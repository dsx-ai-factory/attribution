# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler evidence owned by an allocation, with optional caller-resolved cycle ownership."""

from dataclasses import dataclass
from typing import Optional

from attribution_lib.sources import Source


@dataclass(frozen=True)
class SchedulerFailure:
    """Preserve the original scheduler report and, optionally, the cycle that owns it.

    ``source.time`` is the scheduler's reporting time, not failure onset.
    Supplying this on a JobInfo for an allocation's participation in an NVRx
    cycle makes allocation ownership explicit; neither reporting time nor
    node membership in one record establishes which cycle the failure belongs to.

    ``owner_execution_id`` names the execution record that the caller has
    established as the owner of this failure, for example by comparing the
    failed node's membership in consecutive cycles. ``None`` keeps ownership
    unresolved, and the failure then changes no attribution on its own.
    """

    allocation_id: str
    state: str
    node: str
    source: Source
    owner_execution_id: Optional[str] = None
