# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards on the rule-authoring facade."""

import ast
import pathlib

import attribution_lib.rules as rules
import attribution_lib.rules_engine as rules_engine

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ROOTS = (_PROJECT_ROOT / "src", _PROJECT_ROOT / "tests")


def _names_imported_from_rules() -> tuple[set[str], list[pathlib.Path]]:
    """Every name any module imports from attribution_lib.rules."""
    missing_roots = [root for root in _ROOTS if root.is_dir() is False]
    assert missing_roots == [], f"source roots do not exist: {missing_roots}"

    found: set[str] = set()
    unparsable: list[pathlib.Path] = []
    for root in _ROOTS:
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                unparsable.append(path)
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "attribution_lib.rules":
                    found.update(alias.name for alias in node.names)
    return found, unparsable


def test_every_exported_name_resolves():
    missing = sorted(name for name in rules.__all__ if hasattr(rules, name) is False)
    assert missing == [], f"rules.__all__ names that do not resolve: {missing}"


def test_every_imported_name_is_exported():
    declared = set(rules.__all__)
    imported, unparsable = _names_imported_from_rules()
    assert unparsable == [], f"Python sources could not be parsed: {unparsable}"
    undeclared = sorted(imported - declared)
    assert undeclared == [], f"imported from rules but absent from __all__: {undeclared}"


def test_rules_engine_is_not_a_facade():
    assert hasattr(rules_engine, "__all__") is False
