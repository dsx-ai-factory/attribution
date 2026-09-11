# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Regression tests for bundled symptom defaults."""

from attribution_lib.core import get_bundled_rules_path
from attribution_lib.rules_engine import RuleLoader
from attribution_lib.symptoms.declarations import DagDiscardBehavior
from attribution_lib.symptoms.registry_generation import registry_from_declarations


def test_row_remap_does_not_drive_outcome():
    """Keep Row Remap in the vocabulary without mapping it to a failure."""
    loader = RuleLoader()
    loader.load(str(get_bundled_rules_path() / "symptom_defaults.py"))
    registry = registry_from_declarations(loader.symptom_declarations)

    assert registry.contains("Row Remap")
    assert registry.outcome_for(["Row Remap"]) is None


def test_cuda_illegal_address_resolves_its_own_guidance():
    """Preserve CUDA outcome selection and KEEP behavior, including mixed NCCL evidence."""
    loader = RuleLoader()
    loader.load(str(get_bundled_rules_path() / "symptom_defaults.py"))
    registry = registry_from_declarations(loader.symptom_declarations)

    assert registry.dag_discard_for("cudaErrorIllegalAddress") == DagDiscardBehavior.KEEP
    for symptoms, expected_symptom, expected_outcome, expected_doc in (
        (["cudaErrorIllegalAddress"], "cudaErrorIllegalAddress", "Fail User", "cudaerrorillegaladdress"),
        (["ncclUnhandledCudaError"], "ncclUnhandledCudaError", "Fail MultiCategory", "ncclunhandledcudaerror"),
        (
            ["cudaErrorIllegalAddress", "ncclUnhandledCudaError"],
            "cudaErrorIllegalAddress",
            "Fail User",
            "cudaerrorillegaladdress",
        ),
        (
            ["ncclUnhandledCudaError", "cudaErrorIllegalAddress"],
            "cudaErrorIllegalAddress",
            "Fail User",
            "cudaerrorillegaladdress",
        ),
    ):
        match = registry.outcome_for(symptoms)
        assert match is not None
        assert match.symptom == expected_symptom
        assert match.outcome == expected_outcome
        assert match.doc == expected_doc
