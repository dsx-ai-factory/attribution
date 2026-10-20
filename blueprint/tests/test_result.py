# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from attribution_blueprint.result import enrich_attribution_result, service_outcome


def test_service_outcome_preserves_explicit_empty_outcome():
    result = {
        "verdict": {
            "attribution_outcome": "",
        }
    }

    assert service_outcome(result) == ""


def test_enrich_attribution_result_adds_contract_fields():
    result = enrich_attribution_result(
        {
            "verdict": {
                "attribution_outcome": "Fail InfraHW",
                "root_symptom": {
                    "symptom": "XID-79",
                    "source": {"source": "SYSLOG", "rule": "xid_79"},
                },
            },
            "symptoms": {
                "XID-79": [
                    {
                        "type": "NodeSource",
                        "node": "node-a",
                        "source": {"source": "SYSLOG", "rule": "xid_79"},
                    }
                ]
            },
        },
        [{"connector": "LokiConnector", "source": "syslog", "status": "ok", "reason": "", "observation_count": 1}],
        [{"connector": "LokiConnector", "source": "syslog", "status": "ok", "attempted": 1, "failed": 0, "reason": ""}],
    )

    assert result["terminal_state"] == "fail"
    assert result["sub_domain"] == "hardware"
    assert result["tenant_sre_action"] == ""
    assert result["cloud_operator_sre_action"] == ""
    assert result["connector_reports"][0]["observation_count"] == 1
    assert result["evidence_status"] == "complete"
    assert result["verdict"]["root_symptom"]["source_channel"] == "syslog"
    assert result["symptoms"]["XID-79"][0]["source_channel"] == "syslog"


@pytest.mark.parametrize(
    "evidence",
    [
        {},
        {"symptoms": None},
        {"symptoms": []},
        {"symptoms": "invalid"},
        {"symptoms": {"XID-79": None}},
        {"symptoms": {"XID-79": "invalid"}},
        {"symptoms": {"XID-79": []}},
        {"symptoms": {"XID-79": [None, 0, "invalid", {"source": None}]}},
    ],
)
def test_enrich_attribution_result_preserves_partial_evidence(evidence):
    service_result = {
        "verdict": {
            "attribution_outcome": "Fail InfraHW",
            "root_symptom": None,
            "decision_graph": {"nodes": [], "edges": []},
        },
        **evidence,
    }
    original = deepcopy(service_result)

    result = enrich_attribution_result(service_result, [], [])

    assert service_result == original
    assert result["verdict"] == original["verdict"]
    assert ("symptoms" in result) == ("symptoms" in original)
    if "symptoms" in original:
        assert result["symptoms"] == original["symptoms"]
    assert result["terminal_state"] == "fail"
    assert result["sub_domain"] == "hardware"


def test_enrich_attribution_result_enriches_valid_sites_among_malformed_entries():
    service_result = {
        "symptoms": {
            "XID-79": [None, {"source": {"source": "SYSLOG", "pos": 0}}],
        },
    }
    original = deepcopy(service_result)

    result = enrich_attribution_result(service_result, [], [])

    assert service_result == original
    assert result["symptoms"]["XID-79"] == [
        None,
        {"source": {"source": "SYSLOG", "pos": 0}, "source_channel": "syslog"},
    ]


def _connector(source: str, status: str, observation_count: int) -> dict:
    reason = "" if status == "ok" else "boom"
    return {
        "connector": "C",
        "source": source,
        "status": status,
        "reason": reason,
        "observation_count": observation_count,
    }


def _ingestion(source: str, status: str, attempted: int, failed: int) -> dict:
    reason = "" if failed == 0 else "boom"
    return {
        "connector": "C",
        "source": source,
        "status": status,
        "attempted": attempted,
        "failed": failed,
        "reason": reason,
    }


@pytest.mark.parametrize(
    ("connector_reports", "ingestion_reports", "expected"),
    [
        ([_connector("applog", "ok", 3)], [_ingestion("applog", "partial", 3, 1)], "partial"),
        (
            [_connector("applog", "error", 0), _connector("syslog", "ok", 2)],
            [_ingestion("applog", "ok", 0, 0), _ingestion("syslog", "ok", 2, 0)],
            "partial",
        ),
        ([_connector("applog", "ok", 2)], [_ingestion("applog", "error", 2, 2)], "none"),
        ([_connector("applog", "error", 0)], [_ingestion("applog", "ok", 0, 0)], "none"),
        ([_connector("applog", "ok", 0)], [_ingestion("applog", "ok", 0, 0)], "none"),
    ],
    ids=[
        "partial-delivery",
        "collector-error-with-other-delivery",
        "nothing-delivered",
        "collector-error-only",
        "nothing-collected-no-errors",
    ],
)
def test_enrich_attribution_result_flags_incomplete_evidence(connector_reports, ingestion_reports, expected):
    result = enrich_attribution_result({}, connector_reports, ingestion_reports)

    assert result["evidence_status"] == expected
