# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point: `nvidia-attribution-blueprint-web --db PATH`."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from attribution_blueprint.webui.app import WebAssetsMissingError, create_app
from attribution_blueprint.webui.db import DatabaseSchemaError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nvidia-attribution-blueprint-web",
        description="Browse the attribution SQLite database in a web UI.",
    )
    parser.add_argument("--db", required=True, help="Path to the attribution SQLite database.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    try:
        app = create_app(args.db)
    except (FileNotFoundError, WebAssetsMissingError, DatabaseSchemaError) as e:
        print(str(e), file=sys.stderr)
        return 2

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
