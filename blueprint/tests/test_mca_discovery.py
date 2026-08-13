# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Verify that dropping a Python package into connectors/ is enough to enable
a new MCA component. No mca.py edits required."""

from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def dropin_package():
    import attribution_blueprint.connectors.execution_contexts as eca_pkg

    target = Path(eca_pkg.__file__).parent / "dummy_test_only"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir()
    (target / "__init__.py").write_text(
        textwrap.dedent(
            """
            from attribution_blueprint import mca_registry as _registry
            from attribution_blueprint.exceptions import MCAError
            from attribution_blueprint.mca_registry import _PARAM


            class _DummyConnector:
                def __init__(self, marker: str = "default"):
                    self._marker = marker

                def discover_by_ids(self, ids):
                    return []

                def discover_by_time_range(self, since, until, nodelist):
                    return []


            def _build(spec, ctx):
                return _DummyConnector(marker=spec.get("marker", "default"))


            _registry.register_component(
                framework="eca",
                name="dummy_test_only",
                schema={"marker": _PARAM},
                build=_build,
            )
            """
        )
    )

    yield target

    shutil.rmtree(target)
    from attribution_blueprint import (
        mca,
        mca_registry as r,
    )

    r._COMPONENTS.pop(("eca", "dummy_test_only"), None)
    for mod_name in list(sys.modules):
        if "dummy_test_only" in mod_name:
            del sys.modules[mod_name]
    mca.SCHEMA = r.build_schema()
    mca._build_env_indexes()


def test_dropin_package_discovered_by_mca(dropin_package, capsys):
    """A package dropped under connectors/ is picked up by discover() + build_schema()."""
    from attribution_blueprint import cli, mca, mca_registry

    mca_registry.discover("attribution_blueprint.connectors")
    # Rebuild derived caches in place: SCHEMA is computed once by mca's
    # module-scope build_schema() and never recomputed after a later
    # registration.
    mca.SCHEMA = mca_registry.build_schema()
    mca._build_env_indexes()

    rc = cli.main(["--connector", "eca", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dummy_test_only" in out

    rc = cli.main(["--connector", "eca", "dummy_test_only", "help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "marker" in out
