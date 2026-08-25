# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Dump the service's OpenAPI schema to a file.

Usage: extract_spec.py <output-path>

Runs the app in-process, so no port is bound and no readiness polling is
needed. The lifespan handler still has to run first: create_app() defers some
route registration to it -- Instrumentator.expose() adds /metrics there, see
app/factory.py -- and asking for the schema before startup would silently omit
those routes from the spec, and therefore from the generated client.
"""

import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI

from app.factory import create_app


async def _build_schema(app: FastAPI) -> dict:
    """Run the app's startup, then snapshot the schema."""
    async with app.router.lifespan_context(app):
        return app.openapi()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <output-path>")

    destination = Path(sys.argv[1])
    schema = asyncio.run(_build_schema(create_app()))

    # trailing newline so the file is diff-friendly and POSIX-clean
    destination.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
