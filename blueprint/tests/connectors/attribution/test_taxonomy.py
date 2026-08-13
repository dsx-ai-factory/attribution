# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest

from attribution_blueprint.taxonomy import (
    SERVICE_ATTRIBUTION_OUTCOMES,
    TAXONOMY_MAP,
    BlueprintTaxonomy,
    taxonomy_for_service_value,
    translate_service_taxonomy,
)

EXPECTED_MAPPING = [
    ("", BlueprintTaxonomy.UNATTRIBUTED),
    ("Pass", BlueprintTaxonomy.PASS),
    ("Fail InfraHW", BlueprintTaxonomy.HARDWARE_FAULT),
    ("Fail InfraSW", BlueprintTaxonomy.SYSTEM_SOFTWARE_FAULT),
    ("Fail User", BlueprintTaxonomy.APPLICATION_FAULT),
    ("Fail Multi", BlueprintTaxonomy.AMBIGUOUS_FAULT),
    ("Fail MultiCategory", BlueprintTaxonomy.AMBIGUOUS_FAULT),
    ("Fail Undetermined", BlueprintTaxonomy.UNATTRIBUTED),
    ("Cancelled by User", BlueprintTaxonomy.CANCELLED_WORKLOAD_OWNER),
    ("Cancelled by cluster operators", BlueprintTaxonomy.CANCELLED_OPERATOR),
    ("Expected Fail", BlueprintTaxonomy.EXPECTED_FAILURE),
    ("Artificial", BlueprintTaxonomy.EXPECTED_FAILURE),
    ("Incomplete Info", BlueprintTaxonomy.UNATTRIBUTED),
]


@pytest.mark.parametrize("service_value,expected", EXPECTED_MAPPING)
def test_translate_known_values(service_value, expected):
    assert translate_service_taxonomy(service_value) is expected


@pytest.mark.parametrize(
    "service_value",
    ["Unknown", "fail infrahw", "Fail Infrahw", "PASS", "Fail HW", "null"],
)
def test_unknown_values_fall_back_to_unattributed(service_value):
    assert translate_service_taxonomy(service_value) is BlueprintTaxonomy.UNATTRIBUTED


def test_mapping_covers_every_expected_row():
    assert set(TAXONOMY_MAP) == SERVICE_ATTRIBUTION_OUTCOMES
    assert set(TAXONOMY_MAP) == {src for src, _ in EXPECTED_MAPPING}


def test_taxonomy_rows_include_terminal_state_and_sub_domain():
    row = taxonomy_for_service_value("Fail InfraHW")
    assert row.terminal_state == "fail"
    assert row.sub_domain == "hardware"


def test_blueprint_taxonomy_rendered_strings():
    assert BlueprintTaxonomy.PASS.value == "Pass"
    assert BlueprintTaxonomy.HARDWARE_FAULT.value == "Hardware Fault"
    assert BlueprintTaxonomy.SYSTEM_SOFTWARE_FAULT.value == "System Software Fault"
    assert BlueprintTaxonomy.APPLICATION_FAULT.value == "Application Fault"
    assert BlueprintTaxonomy.AMBIGUOUS_FAULT.value == "Ambiguous Fault"
    assert BlueprintTaxonomy.UNATTRIBUTED.value == "Unattributed"
    assert BlueprintTaxonomy.CANCELLED_WORKLOAD_OWNER.value == "Cancelled, Workload Owner"
    assert BlueprintTaxonomy.CANCELLED_OPERATOR.value == "Cancelled, Operator"
    assert BlueprintTaxonomy.EXPECTED_FAILURE.value == "Expected Failure"


def test_taxonomy_members_are_strings():
    assert BlueprintTaxonomy.HARDWARE_FAULT == "Hardware Fault"
    assert str(BlueprintTaxonomy.HARDWARE_FAULT.value) == "Hardware Fault"
