# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum


class BlueprintTaxonomy(str, Enum):
    PASS = "Pass"
    HARDWARE_FAULT = "Hardware Fault"
    SYSTEM_SOFTWARE_FAULT = "System Software Fault"
    APPLICATION_FAULT = "Application Fault"
    AMBIGUOUS_FAULT = "Ambiguous Fault"
    UNATTRIBUTED = "Unattributed"
    CANCELLED_WORKLOAD_OWNER = "Cancelled, Workload Owner"
    CANCELLED_OPERATOR = "Cancelled, Operator"
    EXPECTED_FAILURE = "Expected Failure"


@dataclass(frozen=True)
class TaxonomyRow:
    terminal_state: str
    sub_domain: str
    blueprint_taxonomy: BlueprintTaxonomy


SERVICE_ATTRIBUTION_OUTCOMES = frozenset(
    {
        "",
        "Pass",
        "Fail InfraHW",
        "Fail InfraSW",
        "Fail User",
        "Fail Multi",
        "Fail MultiCategory",
        "Fail Undetermined",
        "Cancelled by User",
        "Cancelled by cluster operators",
        "Expected Fail",
        "Artificial",
        "Incomplete Info",
    }
)

TAXONOMY_MAP: dict[str, TaxonomyRow] = {
    "": TaxonomyRow("unknown", "unclassified", BlueprintTaxonomy.UNATTRIBUTED),
    "Pass": TaxonomyRow("pass", "none", BlueprintTaxonomy.PASS),
    "Fail InfraHW": TaxonomyRow("fail", "hardware", BlueprintTaxonomy.HARDWARE_FAULT),
    "Fail InfraSW": TaxonomyRow("fail", "system_software", BlueprintTaxonomy.SYSTEM_SOFTWARE_FAULT),
    "Fail User": TaxonomyRow("fail", "application", BlueprintTaxonomy.APPLICATION_FAULT),
    "Fail Multi": TaxonomyRow("fail", "ambiguous", BlueprintTaxonomy.AMBIGUOUS_FAULT),
    "Fail MultiCategory": TaxonomyRow("fail", "ambiguous", BlueprintTaxonomy.AMBIGUOUS_FAULT),
    "Fail Undetermined": TaxonomyRow("fail", "unattributed", BlueprintTaxonomy.UNATTRIBUTED),
    "Cancelled by User": TaxonomyRow(
        "cancelled",
        "workload_owner",
        BlueprintTaxonomy.CANCELLED_WORKLOAD_OWNER,
    ),
    "Cancelled by cluster operators": TaxonomyRow(
        "cancelled",
        "operator",
        BlueprintTaxonomy.CANCELLED_OPERATOR,
    ),
    "Expected Fail": TaxonomyRow("expected_failure", "expected", BlueprintTaxonomy.EXPECTED_FAILURE),
    "Artificial": TaxonomyRow("expected_failure", "artificial", BlueprintTaxonomy.EXPECTED_FAILURE),
    "Incomplete Info": TaxonomyRow(
        "unknown",
        "unattributed",
        BlueprintTaxonomy.UNATTRIBUTED,
    ),
}

_MISSING_TAXONOMY_ROWS = SERVICE_ATTRIBUTION_OUTCOMES - TAXONOMY_MAP.keys()
if _MISSING_TAXONOMY_ROWS:
    missing = ", ".join(sorted(_MISSING_TAXONOMY_ROWS))
    raise RuntimeError(f"TAXONOMY_MAP missing service outcome rows: {missing}")

_SERVICE_TO_BLUEPRINT: dict[str, BlueprintTaxonomy] = {
    service_value: row.blueprint_taxonomy for service_value, row in TAXONOMY_MAP.items()
}


def translate_service_taxonomy(service_value: str) -> BlueprintTaxonomy:
    return _SERVICE_TO_BLUEPRINT.get(service_value, BlueprintTaxonomy.UNATTRIBUTED)


def taxonomy_for_service_value(service_value: str) -> TaxonomyRow:
    return TAXONOMY_MAP.get(service_value, TAXONOMY_MAP[""])


__all__ = [
    "BlueprintTaxonomy",
    "SERVICE_ATTRIBUTION_OUTCOMES",
    "TAXONOMY_MAP",
    "TaxonomyRow",
    "taxonomy_for_service_value",
    "translate_service_taxonomy",
]
