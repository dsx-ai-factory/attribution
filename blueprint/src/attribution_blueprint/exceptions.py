# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exception types shared by the MCA config layer and the connector packages.

They live here rather than in `mca.py` so a connector module can bind them at
module scope: importing `mca.py` runs its module-scope `discover()` and
`build_schema()` bootstrap, which drops any connector still mid-initialization.
"""


class BlueprintComponentError(Exception):
    """Base class for the MCA config layer and component registry errors."""


class MCAError(BlueprintComponentError):
    """Any user-facing MCA validation / dispatch error."""


class RegistryError(BlueprintComponentError):
    """Registration of a framework or component is malformed or duplicated."""


__all__ = ["BlueprintComponentError", "MCAError", "RegistryError"]
