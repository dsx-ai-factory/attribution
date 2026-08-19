# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Guards on the rules_engine facade.

rules_engine re-exports names that now live in rule_types, declarative,
prefilters and symptoms. Ruff cannot distinguish a deliberate re-export from a
dead import, so an autofix pass will happily delete one and break every rule
module at import time. These two tests make that a named failure instead.
"""

import ast
import pathlib
import subprocess
import sys

import attribution_lib.rules_engine as rules_engine

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ROOTS = (_PROJECT_ROOT / "src", _PROJECT_ROOT / "tests")


def _names_imported_from_rules_engine() -> tuple[set[str], list[pathlib.Path]]:
    """Every name any module imports from attribution_lib.rules_engine."""
    missing_roots = [root for root in _ROOTS if root.is_dir() is False]
    assert missing_roots == [], f"source roots do not exist: {missing_roots}"

    found: set[str] = set()
    unparsable: list[pathlib.Path] = []
    for root in _ROOTS:
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                unparsable.append(path)
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "attribution_lib.rules_engine":
                    found.update(alias.name for alias in node.names)
    return found, unparsable


def test_importing_package_does_not_eagerly_import_rules_engine():
    """The package root must remain usable without loading the engine module."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import attribution_lib, sys; assert 'attribution_lib.rules_engine' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_every_exported_name_resolves():
    """__all__ is the deliberate surface; every name in it must exist.

    If this fails because a name was deliberately moved or removed, update
    __all__ in the same commit that moves it, so the removal is visible in
    review rather than surfacing as an ImportError somewhere else.
    """
    missing = sorted(name for name in rules_engine.__all__ if hasattr(rules_engine, name) is False)
    assert missing == [], f"rules_engine.__all__ names that do not resolve: {missing}"


def test_every_imported_name_is_exported():
    """Anything another module imports from rules_engine must be declared in __all__.

    This is what keeps __all__ from drifting into fiction: a caller cannot
    quietly start depending on a name the facade does not claim to offer.
    Underscore-prefixed names are private by convention and so are not in
    __all__; test_every_imported_private_name_resolves covers those instead.
    """
    declared = set(rules_engine.__all__)
    imported, unparsable = _names_imported_from_rules_engine()
    assert unparsable == [], f"Python sources could not be parsed: {unparsable}"
    undeclared = sorted(n for n in imported if n.startswith("_") is False and n not in declared)
    assert undeclared == [], f"imported from rules_engine but absent from __all__: {undeclared}"


def test_every_imported_private_name_resolves():
    """Private names other modules import from rules_engine must still resolve.

    __all__ deliberately excludes underscore-prefixed names, so the guard above
    cannot see them. Several modules import them anyway (_PRE_FILTERS,
    _METRIC_SPECS, _EXPR_HELPER_RESERVED), and deleting one of
    those re-exports as apparently-unused surfaces as an ImportError somewhere
    unrelated.

    Known blind spot: some tests write a rule file at runtime from a string
    literal, and its imports are invisible to any static scan. _get_caller_location
    is imported that way by test_symptom_registry_generation and by nothing
    else, so only running the suite catches its removal. Treat a name being
    absent here as weak evidence, not proof that it is unused.
    """
    imported, unparsable = _names_imported_from_rules_engine()
    assert unparsable == [], f"Python sources could not be parsed: {unparsable}"
    missing = sorted(name for name in imported if name.startswith("_") and hasattr(rules_engine, name) is False)
    assert missing == [], f"private names imported from rules_engine that do not resolve: {missing}"
