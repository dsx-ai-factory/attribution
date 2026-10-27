# coding: utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
import json
from enum import Enum
from typing_extensions import Self


class Source(str, Enum):
    """
    The classes of observations that can be ingested.  `job_keywords` takes the `string_list` format; every other implemented source takes `entry_timeseries`. See `Format` for how each source reads its entries.  - `slurm_info`      Automatically attached to JobInfo objects; not supported for other Observations. No relevant Format.  - `job_keywords`      Keywords extracted from preprocessing some logs.  - `applog`      Application's stdout/stderr.  - `syslog`      Data from dmesg/syslog.  - `health_check`      Data from health checks.  - `ufmlog`      Fabric manager events.  - `node_metrics`     Prometheus metrics data for node attribution.  - `nvlink_metrics`     Prometheus metrics data for nvlink attribution.  - `ib_metrics`     Not implemented.  - `eth_metrics`     Not implemented.  - `network_syslog`     Not implemented.  - `csp_event`     CSP event events indicating infrastructure-level host failures,    repairs, or maintenance actions reported by the cloud service provider.
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


