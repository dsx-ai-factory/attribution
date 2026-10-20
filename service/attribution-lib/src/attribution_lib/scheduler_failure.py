# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler evidence owned by an allocation, with cycle ownership unresolved."""

from dataclasses import dataclass

from attribution_lib.sources import Source


@dataclass(frozen=True)
class SchedulerFailure:
    """Preserve the original scheduler report without asserting which cycle failed.

    ``source.time`` is the scheduler's reporting time, not failure onset.
    Supplying this on a JobInfo for an allocation's participation in an NVRx
    cycle makes allocation ownership explicit; neither reporting time nor
    node membership establishes which cycle the failure belongs to.
    """

    allocation_id: str
    state: str
    node: str
    source: Source
