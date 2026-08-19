# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "attribution_lib.prefilters",
        "attribution_lib.rule_types.registration",
    ],
)
def test_registration_modules_can_be_imported_first(module):
    """Registration modules do not depend on an earlier loader import."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_declarative_facade_loads_loader_on_demand():
    """Importing the package alone does not initialize its dependency graph."""
    code = """
import sys
import attribution_lib.declarative

assert "attribution_lib.declarative.loader" not in sys.modules
from attribution_lib.declarative import load_toml_rules
assert callable(load_toml_rules)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_rule_types_load_only_when_rule_loader_is_constructed():
    """The base package is inert; RuleLoader explicitly loads built-in implementations."""
    code = """
import sys
import attribution_lib.rule_types

builtins = {
    "attribution_lib.rule_types.collector",
    "attribution_lib.rule_types.extractor",
    "attribution_lib.rule_types.filter",
    "attribution_lib.rule_types.refiner",
}
assert builtins.isdisjoint(sys.modules)

from attribution_lib.rules_engine import RuleLoader
loader = RuleLoader()
assert builtins <= sys.modules.keys()
assert len(loader._loaders) == 4
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
