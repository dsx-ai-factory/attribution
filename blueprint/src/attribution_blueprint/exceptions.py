# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exception types shared by configuration, collection and the connector packages.

They live here rather than in `mca.py` so a connector module can bind them at
module scope: importing `mca.py` runs its module-scope `discover()` and
`build_schema()` bootstrap, which drops any connector still mid-initialization.
"""

from attribution_blueprint.types import Observation


class BlueprintComponentError(Exception):
    """Base class for the MCA config layer and component registry errors."""


class MCAError(BlueprintComponentError):
    """Any user-facing MCA validation / dispatch error."""


class RegistryError(BlueprintComponentError):
    """Registration of a framework or component is malformed or duplicated."""


class PartialCollectionError(RuntimeError):
    """Collection was incomplete; the available observations can still be ingested."""

    def __init__(self, reason: str, observations: list[Observation]):
        """Retain normalized evidence alongside the reason collection was incomplete.

        Args:
            reason: Description of the collection limit or failure.
            observations: Available observations, using the normal connector shape.
        """
        super().__init__(reason)
        self.observations = observations

    def __reduce__(self):
        """Preserve the partial observations when crossing a process boundary."""
        return type(self), (str(self), self.observations)


__all__ = ["BlueprintComponentError", "MCAError", "RegistryError", "PartialCollectionError"]
