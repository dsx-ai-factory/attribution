# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Explicit bootstrap for the built-in rule-type implementations."""

from importlib import import_module

_BUILTIN_RULE_TYPE_MODULES = (
    "attribution_lib.rule_types.collector",
    "attribution_lib.rule_types.extractor",
    "attribution_lib.rule_types.filter",
    "attribution_lib.rule_types.refiner",
)


def load_builtin_rule_types() -> None:
    """Import built-in implementations so their ``@rule_type`` decorators run."""
    for module in _BUILTIN_RULE_TYPE_MODULES:
        import_module(module)
