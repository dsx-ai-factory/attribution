# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
