# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from attribution_blueprint.types import Format, Source


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


def test_format_enum_matches_attribution_service_contract_values():
    """Blueprint Format values mirror the service observation API contract."""
    assert {fmt.value for fmt in Format} == {
        "",
        "string_list",
        "raw_log",
        "raw_loki_streams",
        "loki_dict_list",
        "csv_metric_response",
        "ufmlog",
        "prometheus_metrics",
        "csp_event_list",
    }


def test_format_enum_matches_attribution_service_contract_names():
    """Blueprint Format names mirror the service observation API contract."""
    assert set(Format.__members__) == {
        "EMPTY",
        "STRING_LIST",
        "RAW_LOG",
        "RAW_LOKI_STREAMS",
        "LOKI_DICT_LIST",
        "CSV_METRIC_RESPONSE",
        "UFMLOG",
        "PROMETHEUS_METRICS",
        "CSP_EVENT_LIST",
    }


def test_observation_contract_keeps_existing_blueprint_names():
    """Current Blueprint connectors keep their service wire values."""
    assert Source.APPLOG.value == "applog"
    assert Source.SYSLOG.value == "syslog"
    assert Source.NODE_METRICS.value == "node_metrics"
    assert Source.NVLINK_METRICS.value == "nvlink_metrics"
    assert Format.LOKI_DICT_LIST.value == "loki_dict_list"
    assert Format.RAW_LOKI_STREAMS.value == "raw_loki_streams"
    assert Format.RAW_LOG.value == "raw_log"
    assert Format.PROMETHEUS_METRICS.value == "prometheus_metrics"
