# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class Source(str, Enum):
    """
    The classes of observations that can be ingested.  - `slurm_info`      Automatically attached to JobInfo objects; not supported for other Observations. No relevant Format.  - `job_keywords`      Formats: `[string_list]`      Keywords extracted from preprocessing some logs.  - `applog`      Formats: `[raw_log, loki_dict_list, raw_loki_streams]`. Timestamps should in in nanoseconds, entries must have rank-like prefixes.      Application's stdout/stderr.  - `syslog`      Formats: `[loki_dict_list, raw_loki_streams, entry_timeseries]`. Timestamps should be in nanoseconds.     Host-specifiers required.      Data from dmesg/syslog  - `health_check`      Formats: `[loki_dict_list, raw_loki_streams, entry_timeseries]`. Timestamps should be in milliseconds for     `loki_dict_list`, nanoseconds otherwise. Host-specifiers required.      Data from health checks.  - `node_metrics`     Formats: `[prometheus_metrics, entry_timeseries]`. Timestamps should be in seconds for `prometheus_metrics` and    nanoseconds for `entry_timeseries`, entries must have node information.     Prometheus metrics data for node attribution.  - `nvlink_metrics`     Formats: `[prometheus_metrics, entry_timeseries]`. Timestamps should be in seconds for `prometheus_metrics` and    nanoseconds for `entry_timeseries`, entries must have node information.     Prometheus metrics data for nvlink attribution.  - `ib_metrics`     Not implemented.  - `eth_metrics`     Not implemented.  - `network_syslog`     Not implemented.  - `csp_event`     Formats: `[csp_event_list]`. Timestamps should be in seconds.     CSP event events indicating infrastructure-level host failures,    repairs, or maintenance actions reported by the cloud service provider.
    """

    """
    allowed enum values
    """
    EMPTY = ''
    SLURM_INFO = 'slurm_info'
    JOB_KEYWORDS = 'job_keywords'
    APPLOG = 'applog'
    SYSLOG = 'syslog'
    HEALTH_CHECK = 'health_check'
    NODE_METRICS = 'node_metrics'
    NVLINK_METRICS = 'nvlink_metrics'
    IB_METRICS = 'ib_metrics'
    ETH_METRICS = 'eth_metrics'
    NETWORK_SYSLOG = 'network_syslog'
    UFMLOG = 'ufmlog'
    CSP_EVENT = 'csp_event'

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of Source from a JSON string"""
        return cls(json.loads(json_str))


