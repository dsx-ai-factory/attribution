# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shallow exports for the symptom interface.

Exports are resolved lazily so pure symptom modules can be imported during
rules-engine initialization without pulling in runtime pruning dependencies.
"""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "BaseSymptom": "attribution_lib.symptoms.base",
    "CascadeScope": "attribution_lib.symptoms.declarations",
    "OutcomePriority": "attribution_lib.symptoms.declarations",
    "SymptomCategory": "attribution_lib.symptoms.declarations",
    "SymptomFactory": "attribution_lib.symptoms.factory",
    "SymptomDeclaration": "attribution_lib.symptoms.declarations",
    "INFORMATIONAL_XIDS": "attribution_lib.symptoms.symptom_enums",
    "RegisteredSymptoms": "attribution_lib.symptoms.symptom_enums",
    "cascade_prune": "attribution_lib.symptoms.symptom_set_operator",
    "project_node_symptoms_to_bad_nodes": "attribution_lib.symptoms.event_to_symptom",
    "translate_and_prune_node_attributions": "attribution_lib.symptoms.event_to_symptom",
    "translate_node_attributions_to_symptoms": "attribution_lib.symptoms.event_to_symptom",
}


def __getattr__(name: str) -> Any:
    """Resolve public symptom exports on first access."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "BaseSymptom",
    "CascadeScope",
    "OutcomePriority",
    "SymptomCategory",
    "SymptomFactory",
    "SymptomDeclaration",
    "cascade_prune",
    "INFORMATIONAL_XIDS",
    "RegisteredSymptoms",
    "translate_and_prune_node_attributions",
    "translate_node_attributions_to_symptoms",
    "project_node_symptoms_to_bad_nodes",
]
