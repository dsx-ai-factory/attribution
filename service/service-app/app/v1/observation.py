# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V1 observation wire model. Accepts every legacy `Format`; kept as-is for existing V1 consumers."""

from zoneinfo import ZoneInfo

from attribution_lib.job_info import JobInfo
from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format as LibFormat,
    Observation as LibObservation,
    Source as ObservationSource,
    TimeInterval as LibTimeInterval,
)
from pydantic import model_validator

from app.entry_timeseries import decode_timeseries_entries
from app.exceptions import UnknownObservationFormatError
from app.util import HashableModel, TimeInterval
from app.v1.legacy_formats import (
    Format as ObservationFormat,
    legacy_window,
    to_entry_timeseries,
)


class ObservationContextV1(HashableModel):
    """Describes the scope of an observation."""

    time_interval: TimeInterval
    resources: str = "AllJobResources"  # Accepted for wire compatibility; the lib context has no resources.
    source: ObservationSource
    format: ObservationFormat
    is_prefiltered: bool = False

    @model_validator(mode="after")
    def _only_entry_timeseries_may_declare_an_open_interval(self) -> "ObservationContextV1":
        # Every other format has the bounds read out from under it during extraction.
        # This validator ensures other formats are silently affected during migration to entry_timeseries.
        if self.format is not ObservationFormat.ENTRY_TIMESERIES and (
            self.time_interval.start is None or self.time_interval.end is None
        ):
            raise ValueError(f"a null time_interval bound is only accepted for {ObservationFormat.ENTRY_TIMESERIES}")
        return self

    @model_validator(mode="after")
    def _start_must_not_follow_end(self) -> "ObservationContextV1":
        # Legacy formats are re-windowed by `legacy_window` and never use the declared interval.
        if (
            self.format in (ObservationFormat.ENTRY_TIMESERIES, ObservationFormat.STRING_LIST)
            and self.time_interval.start is not None
            and self.time_interval.end is not None
            and self.time_interval.start > self.time_interval.end
        ):
            raise ValueError("time_interval.start must not be later than time_interval.end")
        return self

    def to_lib_context(self, timezone: ZoneInfo) -> InboundContext:
        """Convert the observation context to a lib context."""
        return self._lib_context(
            LibTimeInterval(start=self.time_interval.start, end=self.time_interval.end),
            LibFormat(self.format.value),
            timezone,
        )

    def _lib_context(self, time_interval: LibTimeInterval, format: LibFormat, timezone: ZoneInfo) -> InboundContext:
        return InboundContext(
            time_interval,
            self.source,
            format,
            timezone,
            pre_filtered=self.is_prefiltered,
        )


class ObservationV1(HashableModel):
    """Describes association of some observation data with a context."""

    context: ObservationContextV1
    body: str

    def to_lib_observation(self, job_info: JobInfo) -> LibObservation:
        """Convert the observation to a lib observation, decoding a legacy body into `entry_timeseries`."""
        context = self.context
        match context.format:
            case ObservationFormat.UNDEFINED:
                raise UnknownObservationFormatError.new(context.format.value).to_http_exception()
            case ObservationFormat.STRING_LIST:
                return LibObservation(context.to_lib_context(job_info.timezone), self.body.split("\n"))
            case ObservationFormat.ENTRY_TIMESERIES:
                return LibObservation(context.to_lib_context(job_info.timezone), decode_timeseries_entries(self.body))
            case _:
                entries = to_entry_timeseries(
                    context.source, context.format, self.body, job_info, context.time_interval.end
                )
                lib_context = context._lib_context(
                    legacy_window(context.source, job_info), LibFormat.ENTRY_TIMESERIES, job_info.timezone
                )
                return LibObservation(lib_context, entries)
