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
from attribution_lib.symptoms.registry_generation import registry_from_declarations


def test_row_remap_does_not_drive_outcome():
    """Keep Row Remap in the vocabulary without mapping it to a failure."""
    loader = RuleLoader()
    loader.load(str(get_bundled_rules_path() / "symptom_defaults.py"))
    registry = registry_from_declarations(loader.symptom_declarations)

    assert registry.contains("Row Remap")
    assert registry.outcome_for(["Row Remap"]) is None
