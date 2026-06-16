# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json

from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.types import ExecutionContext


class StdoutOutput:
    """Prints one JSON line per (context, attribution) pair to stdout."""

    def write(self, context: ExecutionContext, attribution: dict) -> None:
        print(json.dumps({"context": context, "attribution": attribution}, default=str))

    def close(self) -> None:
        pass


def _build(spec: dict, ctx: dict) -> StdoutOutput:
    return StdoutOutput()


_registry.register_component(framework="output", name="stdout", schema={}, build=_build)


__all__ = ["StdoutOutput"]
