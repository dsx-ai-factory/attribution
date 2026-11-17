# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check documentation mappings for representative bundled runtime variants."""

import pytest

from attribution_lib.core import get_bundled_rules_path
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.symptoms.registry_generation import registry_from_declarations


@pytest.mark.parametrize(
    ("symptom", "expected_doc"),
    [
        ("NV_ERR_RESET_REQUIRED kernel_channel.c", "gpu-reset-required"),
        ("NV_ERR_NO_MEMORY mem_multicast_fabric.c", "nv-err-no-memory"),
        ("NV_ERR_INVALID_STATE nv_gpu_ops.c", "nv-err-invalid-state"),
        ("NV_ERR_INSUFFICIENT_RESOURCES mem_multicast_fabric.c", "nv-err-insufficient-resources"),
        ("exit code signal 15", "exit-code-signal"),
        ("IB Error-12 (IBV_WC_RETRY_EXC_ERR)", "ib-error-12"),
        ("IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)", "ib-error-12-vendor-129"),
    ],
)
def test_bundled_runtime_variants_resolve_to_expected_document(symptom: str, expected_doc: str) -> None:
    """Keep specific bundled-rule mapping examples within the library suite."""
    loader = RuleLoader()
    loader.load(get_bundled_rules_path() / "symptom_defaults.py")
    registry = registry_from_declarations(loader.symptom_declarations)
    match = registry.outcome_for([symptom])

    assert match is not None
    assert match.doc == expected_doc
