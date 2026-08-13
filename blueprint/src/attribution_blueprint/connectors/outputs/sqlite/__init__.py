# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone

from attribution_blueprint import (
    __version__ as PIPELINE_VERSION,
    mca_registry as _registry,
)
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM
from attribution_blueprint.result import service_outcome
from attribution_blueprint.types import ExecutionContext

# Schema version persisted as POC metadata. The current shape stores
# Blueprint-facing taxonomy columns while keeping the raw service outcome in
# `outcome`.
SCHEMA_VERSION = "1"
SCHEMA_DESCRIPTION = (
    "execution_context + attribution_run normalized schema; "
    "scheduler_type and attempt_id read from connector-supplied context; "
    "attribution_run carries terminal_state, sub_domain, and responsible node taxonomy"
)
_RESPONSIBLE_NODE_MARKER = re.compile(r"\bresponsible_node=([A-Za-z0-9_.:-]+)")

_SCHEMA_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_version ("
    "    version     TEXT PRIMARY KEY,"
    "    description TEXT,"
    "    applied_at  TEXT NOT NULL"
    ")",
    "CREATE TABLE IF NOT EXISTS execution_context ("
    "    id                 TEXT PRIMARY KEY,"
    "    job_id             TEXT NOT NULL,"
    "    attempt_id         TEXT,"
    "    scheduler_type     TEXT NOT NULL,"
    "    start_time         TEXT NOT NULL,"
    "    end_time           TEXT NOT NULL,"
    "    nodes_json         TEXT NOT NULL,"
    "    status             TEXT,"
    "    workload_framework TEXT,"
    "    user_name          TEXT,"
    "    raw_scheduler_json TEXT NOT NULL DEFAULT '{}',"
    "    created_at         TEXT NOT NULL,"
    "    UNIQUE(job_id, attempt_id, scheduler_type)"
    ")",
    "CREATE TABLE IF NOT EXISTS attribution_run ("
    "    id                   TEXT PRIMARY KEY,"
    "    execution_context_id TEXT NOT NULL REFERENCES execution_context(id),"
    "    schema_version       TEXT NOT NULL,"
    "    pipeline_version     TEXT,"
    "    service_api_version  TEXT,"
    "    ruleset_version      TEXT,"
    "    started_at           TEXT NOT NULL,"
    "    completed_at         TEXT,"
    "    outcome              TEXT NOT NULL,"
    "    terminal_state       TEXT,"
    "    sub_domain           TEXT,"
    "    raw_result_json      TEXT NOT NULL,"
    "    responsible_nodes_json TEXT NOT NULL DEFAULT '[]'"
    ")",
    "CREATE INDEX IF NOT EXISTS idx_execution_context_job_id ON execution_context(job_id)",
    "CREATE INDEX IF NOT EXISTS idx_execution_context_end_time ON execution_context(end_time)",
    "CREATE INDEX IF NOT EXISTS idx_run_outcome              ON attribution_run(outcome)",
    "CREATE INDEX IF NOT EXISTS idx_run_execution_context    ON attribution_run(execution_context_id)",
)


def _short_hash(*parts: str) -> str:
    h = hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()
    return h[:32]


def _iso(value) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bump_iso_microsecond(value: str, offset: int) -> str:
    if offset == 0:
        return value
    try:
        return (datetime.fromisoformat(value) + timedelta(microseconds=offset)).isoformat()
    except ValueError:
        return f"{value}.{offset}"


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_column(conn: sqlite3.Connection, table: str, name: str, ddl: str) -> None:
    if name not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "attribution_run", "terminal_state", "terminal_state TEXT")
    _ensure_column(conn, "attribution_run", "sub_domain", "sub_domain TEXT")
    _ensure_column(
        conn,
        "attribution_run",
        "responsible_nodes_json",
        "responsible_nodes_json TEXT NOT NULL DEFAULT '[]'",
    )


def _responsible_node_from_site(site: dict) -> str | None:
    """Return a node from a symptom site.

    Args:
        site: Symptom site from the attribution service response.

    Returns:
        Node name from the site, or from an explicit ``responsible_node=``
        evidence marker when the service returns a resourceless site.
    """

    node = site.get("node")
    if isinstance(node, str) and node != "":
        return node

    source = site.get("source")
    if not isinstance(source, dict):
        return None

    line = source.get("line")
    if not isinstance(line, str) or line == "":
        return None

    match = _RESPONSIBLE_NODE_MARKER.search(line)
    if match is None:
        return None
    return match.group(1)


def _responsible_nodes_from(attribution: dict | None) -> list[str]:
    """Return the sorted union of nodes correlated with verdict evidence.

    Args:
        attribution: Attribution service response after result enrichment.

    Returns:
        Sorted, deduplicated responsible nodes from symptom sites.
    """

    if attribution is None:
        return []
    seen: set[str] = set()
    symptoms = attribution.get("symptoms")
    if not isinstance(symptoms, dict):
        return []

    for sites in symptoms.values():
        if not isinstance(sites, list):
            continue
        for site in sites:
            if not isinstance(site, dict):
                continue
            node = _responsible_node_from_site(site)
            if node is not None and node != "":
                seen.add(node)
    return sorted(seen)


class SQLiteOutput:
    """Persists execution contexts and attribution runs per docs/output_schema.md.

    Connection is opened lazily on the first write. Schema is created on first
    connect and the schema_version row is upserted.
    """

    def __init__(self, path: str, scheduler_type: str | None = None):
        """Create a SQLite output writer.

        Args:
            path: SQLite database path.
            scheduler_type: Optional default scheduler type when the context
                does not provide one.
        """
        self._path = path
        self._scheduler_type_fallback = scheduler_type
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        # WAL allows the webui (reader) and the orchestrator (writer) to share
        # the file without `OperationalError: database is locked`. Set per
        # connection because pragma scope is the connection.
        conn.execute("PRAGMA journal_mode=WAL")
        for ddl in _SCHEMA_DDL:
            conn.execute(ddl)
        _migrate_schema(conn)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version(version, description, applied_at) VALUES (?, ?, ?)",
            (SCHEMA_VERSION, SCHEMA_DESCRIPTION, _now()),
        )
        conn.commit()
        return conn

    def write(self, context: ExecutionContext, attribution: dict) -> None:
        if self._conn is None:
            self._conn = self._connect()

        ctx = dict(context)
        job_id = ctx.get("job_id") or ctx.get("id", "")
        # `attempt_id` is connector-supplied (Slurm sets it = sluid, Kueue
        # sets it = the Workload's uid).
        attempt_id = ctx.get("attempt_id", "")
        # `scheduler_type` is the BP-internal connector identity. Prefer the
        # connector-owned `scheduler` value, then use the service wire
        # `workload_type`, then the constructor-time default.
        scheduler_type = ctx.get("scheduler") or ctx.get("workload_type") or self._scheduler_type_fallback or ""
        ec_id = _short_hash(job_id, attempt_id, scheduler_type)
        now = _now()

        # UPSERT (not OR IGNORE): re-running an ExecutionContext whose metadata changed
        # between runs (end_time corrected, status RUNNING→COMPLETED) must
        # land the new values. created_at is preserved on conflict; everything
        # mutable is refreshed.
        self._conn.execute(
            "INSERT INTO execution_context"
            "(id, job_id, attempt_id, scheduler_type, start_time, end_time, "
            "nodes_json, status, workload_framework, user_name, raw_scheduler_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "start_time=excluded.start_time, "
            "end_time=excluded.end_time, "
            "nodes_json=excluded.nodes_json, "
            "status=excluded.status, "
            "workload_framework=excluded.workload_framework, "
            "user_name=excluded.user_name, "
            "raw_scheduler_json=excluded.raw_scheduler_json",
            (
                ec_id,
                job_id,
                attempt_id or None,
                scheduler_type,
                _iso(ctx.get("start_time")),
                _iso(ctx.get("end_time")),
                json.dumps(ctx.get("nodes") or []),
                ctx.get("state"),
                ctx.get("workload_framework"),
                ctx.get("user_name"),
                json.dumps(ctx, default=str),
                now,
            ),
        )

        # `attribution_outcome` is the raw lib outcome string ("Fail InfraHW",
        # "Fail InfraSW", "Pass", "Fail MultiCategory", "Fail User", etc.) as
        # surfaced by the NVIDIA Attribution Service v2 API.
        outcome = service_outcome(attribution)
        terminal_state = attribution.get("terminal_state") if attribution is not None else None
        sub_domain = attribution.get("sub_domain") if attribution is not None else None
        started_at_base = now
        completed_at = _now()
        responsible_nodes = _responsible_nodes_from(attribution)

        for offset in range(1000):
            started_at = _bump_iso_microsecond(started_at_base, offset)
            run_id = _short_hash(ec_id, started_at)
            try:
                self._conn.execute(
                    "INSERT INTO attribution_run"
                    "(id, execution_context_id, schema_version, pipeline_version, "
                    "service_api_version, ruleset_version, started_at, completed_at, "
                    "outcome, terminal_state, sub_domain, raw_result_json, responsible_nodes_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        ec_id,
                        SCHEMA_VERSION,
                        PIPELINE_VERSION,
                        attribution.get("model_type") if attribution is not None else None,
                        None,
                        started_at,
                        completed_at,
                        outcome,
                        terminal_state,
                        sub_domain,
                        json.dumps(attribution, default=str),
                        json.dumps(responsible_nodes),
                    ),
                )
                break
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" not in str(e) or "attribution_run" not in str(e):
                    raise
                if offset == 999:
                    raise
                continue
        self._conn.commit()

    def close(self) -> None:
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None


def migrate(path: str) -> str:
    """Create missing tables and apply additive SQLite schema updates."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        for ddl in _SCHEMA_DDL:
            conn.execute(ddl)
        _migrate_schema(conn)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version(version, description, applied_at) VALUES (?, ?, ?)",
            (SCHEMA_VERSION, SCHEMA_DESCRIPTION, _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return SCHEMA_VERSION


def _build(spec: dict, ctx: dict) -> SQLiteOutput:
    path = spec.get("path")
    if not path:
        raise MCAError("output=sqlite requires sqlite.path")
    return SQLiteOutput(path=path)


_registry.register_component(
    framework="output",
    name="sqlite",
    schema={"path": _PARAM},
    build=_build,
)


__all__ = ["SQLiteOutput", "SCHEMA_VERSION", "migrate"]
