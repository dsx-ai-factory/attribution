# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
    )

    assert result["terminal_state"] == "fail"
    assert result["sub_domain"] == "hardware"
    assert result["tenant_sre_action"] == ""
    assert result["cloud_operator_sre_action"] == ""
    assert result["connector_reports"][0]["observation_count"] == 1
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

    result = enrich_attribution_result(service_result, [])

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

    result = enrich_attribution_result(service_result, [])

    assert service_result == original
    assert result["symptoms"]["XID-79"] == [
        None,
        {"source": {"source": "SYSLOG", "pos": 0}, "source_channel": "syslog"},
    ]
