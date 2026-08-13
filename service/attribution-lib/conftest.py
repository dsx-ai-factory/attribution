# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support.fixtures import *  # noqa: E402,F401,F403


def _iter_co_located_test_paths() -> list[Path]:
    configured_roots = [
        Path(root).expanduser() for root in os.environ.get("ATTRIBUTION_RULE_TEST_ROOTS", "").split(":") if root
    ]
    if not configured_roots:
        configured_roots = [ROOT / "src" / "attribution_lib" / "bundled_rules"]

    module_paths: list[Path] = []
    for root in configured_roots:
        if not root.exists():
            continue
        module_paths.extend(sorted(root.rglob("*_test.py")))
    return module_paths


class CoLocatedModuleCollector(pytest.File):
    def collect(self):
        for path in _iter_co_located_test_paths():
            yield pytest.Module.from_parent(self, path=path)


def pytest_collect_file(file_path: Path, parent):
    if file_path.name == "test_co_located_modules.py":
        return CoLocatedModuleCollector.from_parent(parent, path=file_path)
    return None
