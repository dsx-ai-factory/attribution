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
  (Source.APPLOG, Format.ENTRY_TIMESERIES) -> list[TimeseriesEntry], the
                                              shared shape in
                                              `attribution_blueprint.types`

These types live next to the connector that owns them (Architectural Rule
#3) and are intentionally redefined here rather than imported from
`attribution_lib` (Architectural Rule #2).
"""

from typing import TypedDict


class LokiSyslogEntry(TypedDict):
    """Flat per-entry shape the connector emits for SYSLOG + LOKI_DICT_LIST.

    Constructed by `_encode_body` from each `values` row in each Loki stream.
    Hostname (`node`) is read from the stream labels via `node_label` (default
    `host_name`); rows with no such label carry `node=None`.
    """

    timestamp: int  # nanoseconds since epoch (cast from Loki's string value)
    body: str  # the log line as Loki returned it
    node: str | None  # cluster hostname; the spec requires host-specifiers for SYSLOG


__all__ = ["LokiSyslogEntry"]
