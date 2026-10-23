# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi.encoders import jsonable_encoder
from pydantic import AwareDatetime, BaseModel


def to_zoneinfo(dt: datetime) -> ZoneInfo:
    """Convert a datetime's tzinfo to ZoneInfo (required by attribution_lib).

    If the datetime already carries a ZoneInfo, return it directly.
    Otherwise derive a fixed-offset IANA key like 'Etc/GMT+7' from the UTC offset.
    Falls back to UTC if the offset is not on an hour boundary.
    """
    tz = dt.tzinfo
    if isinstance(tz, ZoneInfo):
        return tz
    if tz is None:
        return ZoneInfo("UTC")
    offset = tz.utcoffset(dt)
    if offset is None:
        return ZoneInfo("UTC")
    total_seconds = int(offset.total_seconds())
    if total_seconds == 0:
        return ZoneInfo("UTC")
    hours, remainder = divmod(abs(total_seconds), 3600)
    if remainder != 0:
        return ZoneInfo("UTC")
    # POSIX Etc/GMT sign convention is inverted: UTC-7 → Etc/GMT+7
    sign = "-" if total_seconds > 0 else "+"
    return ZoneInfo(f"Etc/GMT{sign}{hours}")


class HashableModel(BaseModel):
    def __hash__(self):
        # Sort the dictionary to ensure consistent ordering
        sorted_dict = json.dumps(jsonable_encoder(self), sort_keys=True)
        # Create a hash of the sorted JSON string
        return int(hashlib.md5(sorted_dict.encode()).hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, HashableModel):
            return self.__hash__() == other.__hash__()
        return False


class TimeInterval(HashableModel):
    """The window an observation declares it covers.

    Both keys are required, so a payload that omits one is a validation error rather than unbounded.
    A null bound means unbounded on that side, and only the `entry_timeseries` format accepts one.
    """

    start: Optional[AwareDatetime]
    end: Optional[AwareDatetime]
