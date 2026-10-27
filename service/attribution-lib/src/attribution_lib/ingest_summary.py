# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-ingest tally of the entries an observation kept and the ones it dropped."""

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum


class DropReason(StrEnum):
    """Why an entry was discarded before the extraction rules."""

    OUTSIDE_WINDOW = "outside_window"
    """Entry was outside the (start, end) window of the job."""

    NO_TIMESTAMP = "no_timestamp"
    """Entry had no timestamp and one was required."""

    NO_NODE = "no_node"
    """Entry had no node and one was required."""

    UNKNOWN_NODE = "unknown_node"
    """Entry names a node or rank outside the job."""

    SOURCE_SKIPPED_NO_JOB_NODES = "source_skipped_no_job_nodes"
    """The source requires job nodes and the job has none."""

    PRE_FILTERED = "pre_filtered"
    """Entry was removed by the source's pre-filters."""


@dataclass
class IngestSummary:
    """Entries an ingestion kept, and how many it dropped per reason."""

    accepted: int = 0
    """Number of entries ingested."""

    dropped: Counter[DropReason] = field(default_factory=Counter)
    """Number of entries dropped per reason."""
