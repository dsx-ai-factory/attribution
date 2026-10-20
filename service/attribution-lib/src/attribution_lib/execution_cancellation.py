# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancellation evidence associated with the execution being attributed."""

from dataclasses import dataclass

from attribution_lib.sources import Source


@dataclass(frozen=True)
class ExecutionCancellation:
    """A caller-established cancellation, independent of allocation status.

    The caller must establish that the source describes ``execution_id``.
    A containing allocation's final state alone does not establish this.
    ``cancelled_by`` uses the same identity contract as JobInfo: ``root``,
    ``site-automation-account``, or a nonempty scheduler identity. Missing
    or conflicting identities must remain unresolved at the caller.
    """

    execution_id: str
    cancelled_by: str
    source: Source

    def __post_init__(self) -> None:
        if not self.execution_id or not self.cancelled_by or not self.cancelled_by.strip():
            raise ValueError("Execution cancellation requires an execution ID and cancellation identity")
        if self.source.time is None or self.source.time.utcoffset() is None:
            raise ValueError("Execution cancellation requires an aware cancellation timestamp")
