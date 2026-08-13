# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Self, TypeVar
from zoneinfo import ZoneInfo

from attribution_lib.observation.inbound import (
    Context as InboundContext,
    Format,
    Id,
    Observation as InboundObservation,
    Source as ObsSource,
    TimeInterval,
)
from attribution_lib.resources.resource_set import validate_resource_set


@dataclass
class Context(InboundContext):
    id: Id

    def __hash__(self) -> int:
        return hash((self.time_interval, self.resources, self.source, self.format, self.timezone, self.id))

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Reconstruct ObservationContext from dict."""
        ti = data["time_interval"]
        start = datetime.fromisoformat(ti["start"]) if isinstance(ti.get("start"), str) else ti.get("start")
        end = datetime.fromisoformat(ti["end"]) if isinstance(ti.get("end"), str) else ti.get("end")
        time_interval = TimeInterval(start, end)

        obs_source = ObsSource(data["source"])
        obs_format = Format(data["format"])
        timezone = ZoneInfo(data["timezone"])

        return cls(
            time_interval=time_interval,
            resources=validate_resource_set(data.get("resources") or "AllJobResources"),
            source=obs_source,
            format=obs_format,
            timezone=timezone,
            id=data["id"],
        )


V = TypeVar("V", bound=Iterable)


class Observation(Generic[V]):
    context: Context
    lines: V

    def __init__(self, context: Context, lines: V):
        self.context = context
        self.lines = lines

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    @classmethod
    def from_inbound(cls, context: InboundContext, lines: V) -> "Observation[V]":
        id = Id(hash(InboundObservation(context, lines)))
        return cls(Context(**vars(context), id=id), lines)

    @classmethod
    def from_inbound_observation(cls, obs: InboundObservation[V]) -> "Observation[V]":
        id = Id(hash(obs))
        return cls(Context(**vars(obs.context), id=id), obs.lines)
