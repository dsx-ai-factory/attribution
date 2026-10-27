# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TOML configuration loader.

The TOML loader is intentionally thin: it now delegates to ``mca.load_toml`` /
``mca.build_config``, which power CLI, env-var, and TOML configuration uniformly.
"""

from __future__ import annotations

from pathlib import Path

from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca import Config, build_config, load_toml

__all__ = ["Config", "load_config", "MCAError"]


def load_config(path: str | Path) -> Config:
    """Load a TOML file and return a fully-built ``Config``."""
    return build_config(load_toml(path))
