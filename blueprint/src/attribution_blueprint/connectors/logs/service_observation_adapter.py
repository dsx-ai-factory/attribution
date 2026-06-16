# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Adapt log observations to the attribution-service observation contract."""

import json
from datetime import datetime
from typing import Any

from attribution_blueprint.types import Format, Observation, Source

_ALL_JOB_RESOURCES = "AllJobResources"


def adapt_log_observation_for_service(observation: Observation) -> Observation:
    """Return a service-compatible observation without mutating the collector output.

    Args:
        observation: Log observation emitted by a connector.

    Returns:
        Observation in the shape accepted by the attribution service.
    """
    context = observation["context"]
    if _value(context["source"]) == Source.SYSLOG.value and _value(context["format"]) == Format.RAW_LOG.value:
        return _raw_syslog_to_loki_dict_list(observation)
    return observation


def _raw_syslog_to_loki_dict_list(observation: Observation) -> Observation:
    """Convert raw syslog text to the flat Loki entry list accepted by the service.

    Args:
        observation: Raw syslog observation emitted by a connector.

    Returns:
        Service-compatible observation with a JSON-encoded Loki entry list.
    """
    context = observation["context"]
    node = _resource_node(context.get("resources"))
    fallback_time = _fallback_time(context["time_interval"]["start"])
    entries: list[dict[str, int | str]] = []
    for line in observation["body"].splitlines():
        if line == "":
            continue
        entry_node = node if node is not None and node != "" else _line_node(line)
        entries.append(
            {
                "timestamp": _line_timestamp_ns(line, fallback_time),
                "body": line,
                "node": entry_node,
            }
        )

    adapted_context = dict(context)
    adapted_context["format"] = Format.LOKI_DICT_LIST
    adapted_context["resources"] = _ALL_JOB_RESOURCES
    return {
        "context": adapted_context,
        "body": json.dumps(entries),
    }


def _fallback_time(value: datetime | str) -> datetime:
    """Normalize an observation start time to a datetime.

    Args:
        value: Observation start time as a datetime or ISO-formatted string.

    Returns:
        Parsed datetime value.
    """
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _line_timestamp_ns(line: str, fallback: datetime) -> int:
    """Return a nanosecond timestamp parsed from a syslog line.

    Args:
        line: Raw syslog line.
        fallback: Timestamp to use when the line does not start with ISO time.

    Returns:
        Unix timestamp in nanoseconds.
    """
    parts = line.split(maxsplit=1)
    if parts == []:
        return int(fallback.timestamp() * 1e9)

    try:
        parsed = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
    except ValueError:
        return int(fallback.timestamp() * 1e9)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=fallback.tzinfo)
    return int(parsed.timestamp() * 1e9)


def _line_node(line: str) -> str:
    """Infer a node name from a raw syslog line.

    Args:
        line: Raw syslog line.

    Returns:
        Node token inferred from the line, or an empty string when unavailable.
    """
    parts = line.split(maxsplit=2)
    if len(parts) >= 2:
        try:
            datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
        except ValueError:
            return parts[0]
        return parts[1]
    if parts != []:
        return parts[0]
    return ""


def _resource_node(resources: Any) -> str | None:
    """Extract a concrete node name from an observation resource value.

    Args:
        resources: Observation resource field value.

    Returns:
        Node name when the resource is node-specific, otherwise None.
    """
    if isinstance(resources, str) is False:
        return None
    if resources == "" or resources == _ALL_JOB_RESOURCES:
        return None
    return resources


def _value(value: Any) -> str:
    """Normalize enum-like and string values for observation context comparisons.

    Args:
        value: Enum-like value, string, or other comparable object.

    Returns:
        String value suitable for comparison with service enum values.
    """
    if hasattr(value, "value"):
        return str(value.value)
    if isinstance(value, str):
        return value.lower()
    return str(value)


__all__ = ["adapt_log_observation_for_service"]
