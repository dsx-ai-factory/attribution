# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `entry_timeseries` body shared by every API version."""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, NotRequired, Optional, TypedDict

from attribution_lib.observation.inbound import (
    Format as ObservationFormat,
)
from attribution_lib.resources import ResourceSet
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, with_config

from app.exceptions import InvalidObservationBodyError

# One day inside datetime's range so any accepted ts survives `datetime.fromtimestamp(ts, tz)` in every timezone.
_TS_NS_MIN = int((datetime.min.replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp() * 1e9)
_TS_NS_MAX = int((datetime.max.replace(tzinfo=timezone.utc) - timedelta(days=1)).timestamp() * 1e9)


@with_config(ConfigDict(extra="forbid"))
class TimeseriesEntry(TypedDict):
    """One entry of an `entry_timeseries` body."""

    body: str
    ts: NotRequired[Optional[Annotated[int, Field(ge=_TS_NS_MIN, le=_TS_NS_MAX)]]]
    resources: NotRequired[Optional[ResourceSet]]


_TIMESERIES_ENTRY_DICTS = TypeAdapter(list[TimeseriesEntry])
_RESOURCE_SET = TypeAdapter(ResourceSet)


def decode_timeseries_entries(body: str) -> list[dict[str, Any]]:
    """Decode an `entry_timeseries` body into the entry list the library receives, or raise a 400."""
    # Validate straight into plain dicts, then give each the shape a
    # pydantic model_dump() gives: "ts" and "resources" always present,
    # "resources" serialized. Building a model per entry and then a dict
    # from each held two full copies of the entries at once: about seven
    # times the body for a 2 GiB application log.
    try:
        entries = _TIMESERIES_ENTRY_DICTS.validate_json(body)
    except ValidationError as e:
        raise InvalidObservationBodyError.new(ObservationFormat.ENTRY_TIMESERIES.value).to_http_exception() from e
    for entry in entries:
        entry.setdefault("ts", None)
        resources = entry.get("resources")
        entry["resources"] = None if resources is None else _RESOURCE_SET.dump_python(resources)
    return entries
