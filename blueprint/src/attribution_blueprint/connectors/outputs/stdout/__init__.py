# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
