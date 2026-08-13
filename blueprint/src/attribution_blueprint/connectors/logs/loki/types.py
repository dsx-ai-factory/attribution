# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Body-element TypedDicts for the Loki connector's emitted Observations.

The Service consumes:

  (Source.SYSLOG, Format.LOKI_DICT_LIST)   -> list[LokiSyslogEntry]
  (Source.APPLOG, Format.RAW_LOKI_STREAMS) -> raw Loki stream objects; the
                                              Service-side parser flattens
                                              each value into a LokiAppEntry

These types live next to the connector that owns them (Architectural Rule
#3) and are intentionally redefined here rather than imported from
`attribution_lib` (Architectural Rule #2).
"""

from typing import NotRequired, TypedDict


class LokiSyslogEntry(TypedDict):
    """Flat per-entry shape the connector emits for SYSLOG + LOKI_DICT_LIST.

    Constructed by `_encode_body` from each `values` row in each Loki stream.
    Hostname (`node`) is read from the stream labels via `node_label` (default
    `host_name`); rows with no such label carry `node=None`.
    """

    timestamp: int  # nanoseconds since epoch (cast from Loki's string value)
    body: str  # the log line as Loki returned it
    node: str | None  # cluster hostname; the spec requires host-specifiers for SYSLOG


class LokiAppEntry(TypedDict):
    """Per-entry shape the Service expects when APPLOG arrives pre-flattened.

    The reference connector emits Format.RAW_LOKI_STREAMS today (the raw
    stream objects, deferring flattening to the Service-side parser), so this
    type is the documented contract for the flattened variant rather than a
    shape this connector currently produces. Retained next to its sibling
    `LokiSyslogEntry` so future connector evolution (or downstream tooling)
    has a single source of truth for the wire shape.

    `timestamp` is a string of nanoseconds as Loki's API returns it in
    `result[*].values[i][0]`; `body` is the raw log line. `line` is reserved
    for the parsed JSON-encoded `line` field rank-prefixed APPLOG entries
    carry. `node` is optional because the APPLOG spec accepts rank-prefixed
    lines without host-specifiers (see the inbound design doc, page 2).
    """

    timestamp: str  # nanoseconds-as-string per Loki's response shape
    body: str
    line: NotRequired[str]
    node: NotRequired[str]


__all__ = ["LokiSyslogEntry", "LokiAppEntry"]
