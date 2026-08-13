# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
