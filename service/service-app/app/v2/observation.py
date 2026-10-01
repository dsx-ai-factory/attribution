# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""V2 observation wire model. The body is always an `entry_timeseries` entry list."""

from zoneinfo import ZoneInfo

from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as ObservationFormat,
    Observation as LibObservation,
    Source as ObservationSource,
    TimeInterval as LibTimeInterval,
)
from attribution_lib.resources import AllJobResources
from pydantic import ConfigDict, Field, model_validator

from app.entry_timeseries import decode_timeseries_entries
from app.util import HashableModel, TimeInterval


class TimeIntervalV2(TimeInterval):
    """The window the entries were collected over. A null bound is unbounded on that side."""

    @model_validator(mode="after")
    def _start_must_not_follow_end(self) -> "TimeIntervalV2":
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("time_interval.start must not be later than time_interval.end")
        return self


class ObservationContextV2(HashableModel):
    """Describes the scope of an `entry_timeseries` observation."""

    model_config = ConfigDict(extra="forbid")

    time_interval: TimeIntervalV2 = Field(
        description=(
            "The collection window: the interval the entries were deliberately gathered over. "
            "The service filters entries by it; an entry whose timestamp falls outside the window is "
            "discarded and never reaches a rule. Specifying `None` for either bound is allowed, and "
            "treats the observation as unbounded on that side. `start` must not be later than `end`."
        )
    )
    source: ObservationSource
    is_prefiltered: bool = False

    def to_lib_context(self, timezone: ZoneInfo) -> InboundContext:
        """Convert the observation context to a lib context."""
        return InboundContext(
            LibTimeInterval(start=self.time_interval.start, end=self.time_interval.end),
            AllJobResources,
            self.source,
            ObservationFormat.ENTRY_TIMESERIES,
            timezone,
            pre_filtered=self.is_prefiltered,
        )


class ObservationV2(HashableModel):
    """An `entry_timeseries` entry list and the context it was collected in."""

    model_config = ConfigDict(extra="forbid")

    context: ObservationContextV2
    body: str = Field(
        description=(
            "JSON-encoded list of entries. Each entry has a `body` string, an optional `ts` "
            "(unix nanoseconds; null when the producer has none) and optional `resources` "
            "naming what the entry is about. Null handling varies by source."
        )
    )

    def to_lib_observation(self, timezone: ZoneInfo) -> LibObservation:
        """Convert the observation to a lib observation."""
        return LibObservation(self.context.to_lib_context(timezone), decode_timeseries_entries(self.body))
