# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""FastAPI app for browsing the attribution SQLite database."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from attribution_blueprint.webui import db


class WebAssetsMissingError(RuntimeError):
    """Raised when the web UI starts without built React assets."""


_HERE = Path(__file__).parent
_STATIC = _HERE / "static"
_INDEX = _STATIC / "index.html"
# Exact root files Vite emits outside /assets/ (favicon/logo/etc). is_file()
# guards entries that a given build may not produce.
_ROOT_FILES = {"favicon.ico", "logo.svg", "vite.svg", "robots.txt"}


def create_app(db_path: str) -> FastAPI:
    """Build the FastAPI app bound to a single backend, configured at bringup.

    `db_path` today is a SQLite file path; future backends (an attribution
    service, Elasticsearch, a SQL DB) plug in here. The dashboard never
    exposes backend selection in the URL — operators choose at startup.
    """
    backend = Path(db_path)
    if not backend.exists():
        raise FileNotFoundError(f"sqlite db not found: {db_path}")
    db.assert_schema_current(str(backend))
    if not _INDEX.is_file():
        raise WebAssetsMissingError(
            f"React assets missing: {_INDEX} not found. In a source checkout, run "
            "`scripts/webui.sh --db PATH`, build the frontend assets, or install a built wheel."
        )

    app = FastAPI(title="Attribution Blueprint Dashboard")
    app.mount("/assets", StaticFiles(directory=str(_STATIC / "assets")), name="assets")

    @app.get("/api/v1/meta")
    def meta() -> dict[str, Any]:
        summary = db.db_summary(str(backend))
        lo, hi = db.date_range_bounds(str(backend))
        n_runs = summary.get("n_runs", 0)
        return {
            "versions": {
                "schema": summary.get("schema_version"),
                "pipeline": None,
                "service": None,
            },
            "dateBounds": {"min": lo[:10] if lo is not None else None, "max": hi[:10] if hi is not None else None},
            "windowedRuns": n_runs,
            "dbTotalRuns": n_runs,
            "database": backend.name,
            "returnedRuns": 0,
        }

    @app.get("/api/v1/dashboard")
    def dashboard(
        from_: date | None = Query(default=None, alias="from"),
        to: date | None = None,
        outcome: str | None = None,
    ) -> dict[str, Any]:
        return db.dashboard_data(
            str(backend),
            since=from_.isoformat() if from_ is not None else None,
            until=to.isoformat() if to is not None else None,
            ledger_outcome=outcome,
        )

    @app.get("/api/v1/runs")
    def runs(
        from_: date | None = Query(default=None, alias="from"),
        to: date | None = None,
        outcome: str | None = None,
        q: str | None = None,
        status: str | None = None,
        symptom: str | None = None,
        node: str | None = None,
        rule: str | None = None,
        user: str | None = None,
        page: int = 0,
        page_size: int = Query(default=10, alias="pageSize"),
    ) -> dict[str, Any]:
        rows, total = db.list_runs_page(
            str(backend),
            outcome=outcome,
            since=from_.isoformat() if from_ is not None else None,
            until=to.isoformat() if to is not None else None,
            job_query=q,
            status=status,
            symptom=symptom,
            node=node,
            rule=rule,
            user=user,
            page=page,
            page_size=page_size,
        )
        from attribution_blueprint.webui import bootstrap

        return {
            "runs": [bootstrap.run_to_react(row) for row in rows],
            "total": total,
            "page": page if page >= 0 else 0,
            "pageSize": min(max(page_size, db.PAGE_SIZE_MIN), db.PAGE_SIZE_MAX),
        }

    @app.get("/api/v1/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        row = db.get_run(str(backend), run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        from attribution_blueprint.webui import bootstrap

        return bootstrap.run_to_react(row)

    @app.get("/{path:path}", response_class=HTMLResponse)
    def spa(path: str):
        # /assets/* is handled (and 404'd) by the StaticFiles mount above.
        if path in _ROOT_FILES and (_STATIC / path).is_file():
            return FileResponse(str(_STATIC / path))
        return HTMLResponse(_INDEX.read_text(encoding="utf-8"))

    return app
