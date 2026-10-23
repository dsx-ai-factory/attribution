# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.types import Source


def test_source_enum_matches_attribution_service_contract_values():
    """Blueprint Source values mirror the service observation API contract."""
    assert {source.value for source in Source} == {
        "",
        "slurm_info",
        "job_keywords",
        "applog",
        "syslog",
        "health_check",
        "node_metrics",
        "nvlink_metrics",
        "ib_metrics",
        "eth_metrics",
        "network_syslog",
        "ufmlog",
        "csp_event",
    }


def test_source_enum_matches_attribution_service_contract_names():
    """Blueprint Source names mirror the service observation API contract."""
    assert set(Source.__members__) == {
        "EMPTY",
        "SLURM_INFO",
        "JOB_KEYWORDS",
        "APPLOG",
        "SYSLOG",
        "HEALTH_CHECK",
        "NODE_METRICS",
        "NVLINK_METRICS",
        "IB_METRICS",
        "ETH_METRICS",
        "NETWORK_SYSLOG",
        "UFMLOG",
        "CSP_EVENT",
    }


def test_observation_contract_keeps_existing_blueprint_names():
    """Current Blueprint connectors keep their service wire values."""
    assert Source.APPLOG.value == "applog"
    assert Source.SYSLOG.value == "syslog"
    assert Source.NODE_METRICS.value == "node_metrics"
    assert Source.NVLINK_METRICS.value == "nvlink_metrics"
