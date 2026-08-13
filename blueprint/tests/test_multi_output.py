# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Config-time tests for the multi-instance output framework.

Covers cases that the existing test_config.py single-output assertions
don't reach: more than one instance, two instances of the same component,
zero-config default, and reserved-name / missing-key errors.
"""

import pytest

from attribution_blueprint.connectors.outputs.sqlite import SQLiteOutput
from attribution_blueprint.connectors.outputs.stdout import StdoutOutput
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca import build_config, load_toml, resolve_cli_tokens, settings_from_cli


def _write(tmp_path, body: str):
    p = tmp_path / "c.toml"
    p.write_text(body)
    return p


def test_multiple_instances_built(tmp_path):
    p = _write(
        tmp_path,
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.primary]\ncomponent = "sqlite"\n'
        '[output.primary.sqlite]\npath = "./a.db"\n'
        '[output.console]\ncomponent = "stdout"\n',
    )
    cfg = build_config(load_toml(p))
    assert len(cfg.outputs) == 2
    types = {type(o).__name__ for o in cfg.outputs}
    assert types == {"SQLiteOutput", "StdoutOutput"}


def test_two_instances_same_component_get_distinct_params(tmp_path):
    p = _write(
        tmp_path,
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.db_a]\ncomponent = "sqlite"\n'
        '[output.db_a.sqlite]\npath = "./a.db"\n'
        '[output.db_b]\ncomponent = "sqlite"\n'
        '[output.db_b.sqlite]\npath = "./b.db"\n',
    )
    cfg = build_config(load_toml(p))
    paths = sorted(o._path for o in cfg.outputs if isinstance(o, SQLiteOutput))
    assert paths == ["./a.db", "./b.db"]


def test_zero_output_section_defaults_to_stdout():
    cfg = build_config({"attribution": {"service_url": "http://x"}, "eca": {"component": "slurm"}})
    assert len(cfg.outputs) == 1
    assert isinstance(cfg.outputs[0], StdoutOutput)


def test_reserved_instance_name_component_rejected(tmp_path):
    p = _write(
        tmp_path,
        '[attribution]\nservice_url = "http://x"\n'
        '[eca]\ncomponent = "slurm"\n'
        '[output.component]\ncomponent = "stdout"\n',
    )
    with pytest.raises(MCAError, match="reserved"):
        load_toml(p)


def test_instance_missing_component_key_errors(tmp_path):
    # An instance section with no `component` key isn't useful and should
    # be a loud failure rather than a silent skip (instances are explicit
    # user choices, unlike sourced frameworks which skip absent sources).
    p = _write(
        tmp_path,
        '[attribution]\nservice_url = "http://x"\n[eca]\ncomponent = "slurm"\n[output.empty]\n',
    )
    with pytest.raises(MCAError, match="requires a `component` key"):
        build_config(load_toml(p))


def test_instance_with_unknown_component_errors():
    merged = {
        "attribution": {"service_url": "http://x"},
        "eca": {"component": "slurm"},
        "output": {"primary": {"component": "carrier-pigeon"}},
    }
    with pytest.raises(MCAError, match="unknown output component"):
        build_config(merged)


def test_cli_resolves_instance_component_selection():
    path, value = resolve_cli_tokens(["output", "console", "stdout"])
    assert path == ("output", "console", "component")
    assert value == "stdout"


def test_cli_resolves_param_under_instance():
    path, value = resolve_cli_tokens(["output", "primary", "sqlite", "path", "/tmp/x.db"])
    assert path == ("output", "primary", "sqlite", "path")
    assert value == "/tmp/x.db"


def test_cli_param_under_instance_selects_component():
    settings = settings_from_cli([["output", "primary", "sqlite", "path", "/tmp/x.db"]])
    assert settings == {"output": {"primary": {"component": "sqlite", "sqlite": {"path": "/tmp/x.db"}}}}


def test_cli_rejects_reserved_instance_name_component():
    with pytest.raises(MCAError, match="reserved"):
        resolve_cli_tokens(["output", "component", "stdout"])


def test_cli_rejects_instance_name_starting_with_underscore():
    with pytest.raises(MCAError, match="reserved"):
        resolve_cli_tokens(["output", "_weird", "stdout"])


def test_cli_instance_without_component_errors():
    with pytest.raises(MCAError, match="requires a component selection"):
        resolve_cli_tokens(["output", "console"])
