# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from attribution_lib.events import EventStream
from attribution_lib.exceptions import ParseStateError, UndeclaredEventTypeError
from attribution_lib.sources import Source

from .rankpulse_all_common import (
    Callbacks,
    RankpulseData,
    process_rankpulse_header,
    process_rankpulse_line,
)


@dataclass
class RankpulseParseState:
    """Container for rankpulse callstack parsing state."""

    source: Optional[Source] = None
    callstack_from_rank: dict[int, Callbacks] = field(default_factory=dict)

    def process_line(self, line: str, source: Source) -> None:
        """Process a single line for rankpulse callstack data."""
        if self.source is None:
            self.source = source.derived(rule="Initial line for Rankpulse")
        process_rankpulse_header(line, self.callstack_from_rank)
        process_rankpulse_line(line, self.callstack_from_rank)

    def finalize_and_extract(self) -> Optional[RankpulseData]:
        """Finalize parsing and extract rankpulse callstack data.

        Returns:
            RankpulseData with rankpulse_callstacks or None if no data collected.

        Raises:
            ParseStateError: If method is called before any lines were processed.
        """
        if self.source is None:
            raise ParseStateError(
                "RankpulseParseState.finalize_and_extract() called with no source. Was process_line() ever called?"
            )

        # Add rank to rankpulse_callstacks.ranks list
        for rank, callstack in self.callstack_from_rank.items():
            if rank not in callstack.ranks:
                callstack.ranks.append(rank)

        if len(self.callstack_from_rank) == 0:
            return None

        logger.debug(f"Collected rankpulse callstacks from {len(self.callstack_from_rank)} rank(s)")

        return RankpulseData(
            source=self.source,
            rankpulse_callstacks=self.callstack_from_rank,
        )

    @classmethod
    def collect(cls, event_stream: EventStream) -> list[RankpulseData]:
        try:
            typ_id = event_stream.mgr.types.get_id("Application.RankPulse.report")
        except UndeclaredEventTypeError:
            return []

        rows = event_stream.query_raw(
            "SELECT payload FROM events WHERE context_id=? AND event_type=? AND status=1",
            [event_stream.ctx_id, typ_id],
        )

        results = []
        for (payload_json,) in rows:
            if payload_json is None:
                continue
            payload = json.loads(payload_json)
            if (data := payload.get("data")) is not None:
                results.append(RankpulseData.from_dict(data))
        return results
