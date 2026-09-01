# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Read-only SQLite access for the web dashboard.

Queries the schema defined in docs/output_schema.md
(schema_version, execution_context, attribution_run).
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

_REQUIRED_ATTRIBUTION_RUN_COLUMNS = frozenset({"terminal_state", "sub_domain"})
_RESPONSIBLE_NODES_COLUMN = "responsible_nodes_json"
RAW_OUTCOME_BY_KEY = {
    "hardware": "Fail InfraHW",
    "software": "Fail InfraSW",
    "multi": "Fail MultiCategory",
    "user": "Fail User",
    "undetermined": "Fail Undetermined",
    "pass": "Pass",
}
OUTCOME_KEY_BY_RAW = {value: key for key, value in RAW_OUTCOME_BY_KEY.items()}
COMBINED_FAILURE_OUTCOMES = ("Fail InfraHW", "Fail InfraSW")
PAGE_SIZE_MIN = 1
PAGE_SIZE_MAX = 100
PARETO_LEDGER_LIMIT = 10
NODE_LEDGER_LIMIT = PARETO_LEDGER_LIMIT


class DatabaseSchemaError(RuntimeError):
    """Raised when the dashboard needs a newer SQLite schema than the DB has."""


@dataclass(frozen=True)
class RunRow:
    run_id: str
    context_id: str
    job_id: str
    attempt_id: str | None
    user_name: str | None
    status: str | None
    workload_framework: str | None
    nodes: list[str]
    start_time: str
    end_time: str
    started_at: str
    completed_at: str | None
    outcome: str
    pipeline_version: str | None
    service_api_version: str | None
    schema_version: str
    terminal_state: str | None
    sub_domain: str | None
    raw_result: dict[str, Any]
    raw_scheduler: dict[str, Any]
    responsible_nodes: list[str]


def _row_to_run(row: sqlite3.Row) -> RunRow:
    raw_result = json.loads(row["raw_result_json"] or "{}")
    responsible_nodes = json.loads(row["responsible_nodes_json"] or "[]")
    return RunRow(
        run_id=row["run_id"],
        context_id=row["context_id"],
        job_id=row["job_id"],
        attempt_id=row["attempt_id"],
        user_name=row["user_name"],
        status=row["status"],
        workload_framework=row["workload_framework"],
        nodes=json.loads(row["nodes_json"] or "[]"),
        start_time=row["start_time"],
        end_time=row["end_time"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        outcome=row["outcome"],
        pipeline_version=row["pipeline_version"],
        service_api_version=row["service_api_version"],
        schema_version=row["schema_version"],
        terminal_state=row["terminal_state"],
        sub_domain=row["sub_domain"],
        raw_result=raw_result,
        raw_scheduler=json.loads(row["raw_scheduler_json"] or "{}"),
        responsible_nodes=responsible_nodes,
    )


_RUN_SELECT_TEMPLATE = """
SELECT
    r.id                AS run_id,
    r.execution_context_id AS context_id,
    r.outcome,
    r.started_at,
    r.completed_at,
    r.pipeline_version,
    r.service_api_version,
    r.schema_version,
    r.terminal_state,
    r.sub_domain,
    r.raw_result_json,
    {responsible_nodes_column},
    c.job_id,
    c.attempt_id,
    c.user_name,
    c.status,
    c.workload_framework,
    c.nodes_json,
    c.start_time,
    c.end_time,
    c.raw_scheduler_json
FROM attribution_run r
JOIN execution_context c ON c.id = r.execution_context_id
"""


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return SQLite column names for a table.

    Args:
        conn: SQLite connection.
        table: Table name.

    Returns:
        Column names present in the table.
    """

    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _attribution_run_columns(conn: sqlite3.Connection) -> set[str]:
    return _table_columns(conn, "attribution_run")


def _responsible_nodes_column_expr(conn: sqlite3.Connection) -> str:
    """Return a SQL expression aliased as the canonical responsible-node column.

    Args:
        conn: SQLite connection.

    Returns:
        SQL column expression for the current database layout.
    """

    columns = _attribution_run_columns(conn)
    if _RESPONSIBLE_NODES_COLUMN in columns:
        return f"r.{_RESPONSIBLE_NODES_COLUMN} AS responsible_nodes_json"
    raise DatabaseSchemaError(
        "SQLite attribution database is missing required responsible node column. "
        "Regenerate the database before opening it in the webui."
    )


def _responsible_nodes_filter_column(conn: sqlite3.Connection) -> str:
    """Return the concrete SQL column used for responsible-node filtering.

    Args:
        conn: SQLite connection.

    Returns:
        Qualified SQL column name for the current database layout.
    """

    columns = _attribution_run_columns(conn)
    if _RESPONSIBLE_NODES_COLUMN in columns:
        return f"r.{_RESPONSIBLE_NODES_COLUMN}"
    raise DatabaseSchemaError(
        "SQLite attribution database is missing required responsible node column. "
        "Regenerate the database before opening it in the webui."
    )


def _run_select_sql(conn: sqlite3.Connection) -> str:
    """Return the run SELECT with current responsible-node storage.

    Args:
        conn: SQLite connection.

    Returns:
        SQL SELECT statement for run rows.
    """

    return _RUN_SELECT_TEMPLATE.format(responsible_nodes_column=_responsible_nodes_column_expr(conn))


def assert_schema_current(path: str) -> None:
    """Raise a migration hint before read-only webui queries touch current columns."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        columns = _attribution_run_columns(conn)
        missing = _REQUIRED_ATTRIBUTION_RUN_COLUMNS - columns
        has_responsible_nodes = _RESPONSIBLE_NODES_COLUMN in columns
    if len(missing) > 0:
        cols = ", ".join(sorted(missing))
        raise DatabaseSchemaError(
            f"SQLite attribution database is missing required column(s): {cols}. "
            "User-facing migration tooling is a v1.1 TODO and is not available in this build; "
            "regenerate the database with the current SQLite layout before opening it in the webui."
        )
    if has_responsible_nodes is False:
        raise DatabaseSchemaError(
            "SQLite attribution database is missing required responsible node column. "
            "Regenerate the database before opening it in the webui."
        )


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        columns = _attribution_run_columns(conn)
        missing = _REQUIRED_ATTRIBUTION_RUN_COLUMNS - columns
        has_responsible_nodes = _RESPONSIBLE_NODES_COLUMN in columns
    except Exception:
        conn.close()
        raise
    if len(missing) > 0:
        conn.close()
        cols = ", ".join(sorted(missing))
        raise DatabaseSchemaError(
            f"SQLite attribution database is missing required column(s): {cols}. "
            "User-facing migration tooling is a v1.1 TODO and is not available in this build; "
            "regenerate the database with the current SQLite layout before opening it in the webui."
        )
    if has_responsible_nodes is False:
        conn.close()
        raise DatabaseSchemaError(
            "SQLite attribution database is missing required responsible node column. "
            "Regenerate the database before opening it in the webui."
        )
    return conn


# Soft cap on the page-load query so a large DB doesn't OOM the webui. The UI
# paginates client-side within this window; raise via the `limit` kwarg when
# more headroom is needed.
DEFAULT_RUN_LIMIT = 5000


def _escape_like(value: str) -> str:
    """Escape SQLite LIKE wildcard characters.

    Args:
        value: User search input.

    Returns:
        Escaped value safe for a LIKE pattern.
    """

    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _normalize_outcome(outcome: str | None) -> str | None:
    """Normalize a frontend outcome key or raw outcome value.

    Args:
        outcome: Frontend outcome key or raw NVIDIA Attribution outcome.

    Returns:
        Raw NVIDIA Attribution outcome, or None when no outcome filter is requested.
    """

    if outcome is None:
        return None
    stripped = outcome.strip()
    if stripped == "":
        return None
    return RAW_OUTCOME_BY_KEY.get(stripped, stripped)


def _filtered_clauses(
    *,
    responsible_nodes_column: str = "r.responsible_nodes_json",
    outcome: str | None = None,
    since: str | None = None,
    until: str | None = None,
    job_query: str | None = None,
    status: str | None = None,
    symptom: str | None = None,
    node: str | None = None,
    rule: str | None = None,
    user: str | None = None,
) -> tuple[list[str], list[Any]]:
    """Build WHERE clauses shared by run list/count endpoints.

    Args:
        responsible_nodes_column: Qualified SQLite responsible-node column.
        outcome: Frontend outcome key or raw NVIDIA Attribution outcome.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        job_query: Free-text search query.
        status: ``failed`` excludes pass rows, ``all`` keeps every row.
        symptom: Root symptom text filter.
        node: Responsible-node or allocated-node text filter.
        rule: Root rule text filter.
        user: Exact user filter.

    Returns:
        SQL clause fragments and bound parameters.
    """

    clauses: list[str] = []
    params: list[Any] = []
    raw_outcome = _normalize_outcome(outcome)
    if raw_outcome is not None:
        clauses.append("r.outcome = ?")
        params.append(raw_outcome)
    if since is not None:
        clauses.append("c.end_time >= ?")
        params.append(since)
    if until is not None:
        clauses.append("c.end_time < ?")
        params.append(_iso_next_day(until))
    if status == "failed":
        clauses.append("r.outcome != ?")
        params.append("Pass")
    if job_query is not None and job_query.strip() != "":
        escaped = _escape_like(job_query.strip())
        clauses.append(
            "(c.job_id LIKE ? ESCAPE '\\' OR c.user_name LIKE ? ESCAPE '\\' "
            f"OR r.raw_result_json LIKE ? ESCAPE '\\' OR {responsible_nodes_column} LIKE ? ESCAPE '\\')"
        )
        like = f"%{escaped}%"
        params.extend([like, like, like, like])
    if symptom is not None and symptom.strip() != "":
        clauses.append("r.raw_result_json LIKE ? ESCAPE '\\'")
        params.append(f"%{_escape_like(symptom.strip())}%")
    if rule is not None and rule.strip() != "":
        clauses.append("r.raw_result_json LIKE ? ESCAPE '\\'")
        params.append(f"%{_escape_like(rule.strip())}%")
    if node is not None and node.strip() != "":
        like = f"%{_escape_like(node.strip())}%"
        clauses.append(f"({responsible_nodes_column} LIKE ? ESCAPE '\\' OR c.nodes_json LIKE ? ESCAPE '\\')")
        params.extend([like, like])
    if user is not None and user.strip() != "":
        clauses.append("c.user_name = ?")
        params.append(user.strip())
    return clauses, params


def _append_where(sql: str, clauses: list[str]) -> str:
    """Append WHERE clauses to a SQL statement.

    Args:
        sql: Base SQL statement.
        clauses: WHERE clause fragments.

    Returns:
        SQL with a WHERE section when clauses are present.
    """

    if len(clauses) == 0:
        return sql
    return sql + " WHERE " + " AND ".join(clauses)


def _gpu_hours_sql() -> str:
    """Return the SQL expression used for NVIDIA Attribution GPU-hours.

    Returns:
        SQL expression preferring copied NVIDIA Attribution ``d_gpu_hours`` with a duration
        fallback for older DBs.
    """

    return (
        "CASE "
        "WHEN json_extract(c.raw_scheduler_json, '$.d_gpu_hours') IS NOT NULL "
        "THEN CAST(json_extract(c.raw_scheduler_json, '$.d_gpu_hours') AS REAL) "
        "ELSE MAX((julianday(c.end_time) - julianday(c.start_time)) * 24.0, 0.0) * "
        "CAST(COALESCE(json_extract(c.raw_scheduler_json, '$.gpu_count'), "
        "json_array_length(c.nodes_json) * COALESCE(json_extract(c.raw_scheduler_json, '$.gpus_per_node'), 4), 0) AS REAL) "
        "END"
    )


def _gpu_count_sql() -> str:
    """Return the SQL expression used for allocated GPU count.

    Returns:
        SQL expression for copied or fallback GPU count.
    """

    return (
        "CAST(COALESCE(json_extract(c.raw_scheduler_json, '$.gpu_count'), "
        "json_array_length(c.nodes_json) * COALESCE(json_extract(c.raw_scheduler_json, '$.gpus_per_node'), 4), 0) AS INTEGER)"
    )


def _gpus_per_node_sql(context_alias: str) -> str:
    """Return a positive GPUs-per-node SQL expression.

    Args:
        context_alias: SQL alias for ``execution_context``.

    Returns:
        SQL expression with a default for historical rows.
    """

    value = f"CAST(json_extract({context_alias}.raw_scheduler_json, '$.gpus_per_node') AS INTEGER)"
    return f"CASE WHEN {value} > 0 THEN {value} ELSE 4 END"


def _window_gpu_count_conn(conn: sqlite3.Connection, clauses: list[str], params: list[Any]) -> int:
    """Return unique observed nodes multiplied by GPUs per node.

    Args:
        conn: SQLite connection.
        clauses: Normalized attribution/execution WHERE clauses.
        params: Bound parameters for ``clauses``.

    Returns:
        Window GPU capacity based on unique nodes in the filtered period.
    """

    gpus_per_node = _gpus_per_node_sql("c")
    node_clauses = [*clauses, "node.value IS NOT NULL", "node.value != ''"]
    inner = _append_where(
        "SELECT node.value AS node_name, MAX(" + gpus_per_node + ") AS gpus_per_node "
        "FROM attribution_run r "
        "JOIN execution_context c ON c.id = r.execution_context_id "
        "JOIN json_each(c.nodes_json) AS node",
        node_clauses,
    )
    sql = "SELECT COALESCE(SUM(gpus_per_node), 0) AS gpu_count FROM (" + inner + " GROUP BY node.value)"
    row = conn.execute(sql, tuple(params)).fetchone()
    return int(row["gpu_count"] if row is not None and row["gpu_count"] is not None else 0)


def list_runs(
    path: str,
    outcome: str | None = None,
    since: str | None = None,
    until: str | None = None,
    job_query: str | None = None,
    limit: int = DEFAULT_RUN_LIMIT,
) -> list[RunRow]:
    """Return runs ordered by attempt end_time descending — the user is reasoning
    about attempts, not about when attribution happened.

    `since`/`until` are ISO-date strings (YYYY-MM-DD) applied as inclusive
    bounds on `c.end_time` IN SQL, before the `LIMIT` is reached. Filtering
    in Python after a capped fetch would silently drop matching rows when
    the date window falls outside the most recent `limit` runs.

    `job_query` is a substring matched against `c.job_id` with SQL LIKE.
    Used by the workload search input on the dashboard. Slurm jobids are
    integers and Kueue job_ids are `<namespace>/<name>` strings; both are
    searchable through the same column. The query is escaped to treat
    `%` and `_` from user input as literals via SQLite's ESCAPE clause.

    Capped at `limit` rows (default DEFAULT_RUN_LIMIT) to keep the
    page-load query bounded.
    """
    with _connect(path) as conn:
        responsible_nodes_column = _responsible_nodes_filter_column(conn)
        sql = _run_select_sql(conn)
        clauses, params = _filtered_clauses(
            responsible_nodes_column=responsible_nodes_column,
            outcome=outcome,
            since=since,
            until=until,
            job_query=job_query,
        )
        sql = _append_where(sql, clauses)
        sql += " ORDER BY c.end_time DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [_row_to_run(r) for r in rows]


def count_runs(
    path: str,
    outcome: str | None = None,
    since: str | None = None,
    until: str | None = None,
    job_query: str | None = None,
) -> int:
    """Pre-cap COUNT(*) with the same filters as list_runs. Used for honest
    'showing N of M' when list_runs caps at DEFAULT_RUN_LIMIT."""
    with _connect(path) as conn:
        responsible_nodes_column = _responsible_nodes_filter_column(conn)
        sql = "SELECT COUNT(*) AS n FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id"
        clauses, params = _filtered_clauses(
            responsible_nodes_column=responsible_nodes_column,
            outcome=outcome,
            since=since,
            until=until,
            job_query=job_query,
        )
        sql = _append_where(sql, clauses)
        return conn.execute(sql, tuple(params)).fetchone()["n"]


def list_runs_page(
    path: str,
    *,
    outcome: str | None = None,
    since: str | None = None,
    until: str | None = None,
    job_query: str | None = None,
    status: str | None = None,
    symptom: str | None = None,
    node: str | None = None,
    rule: str | None = None,
    user: str | None = None,
    page: int = 0,
    page_size: int = 10,
) -> tuple[list[RunRow], int]:
    """Return one page of runs and the matching total.

    Args:
        path: SQLite database path.
        outcome: Frontend outcome key or raw NVIDIA Attribution outcome.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        job_query: Free-text search query.
        status: ``failed`` excludes pass rows, ``all`` keeps every row.
        symptom: Root symptom text filter.
        node: Responsible-node or allocated-node text filter.
        rule: Root rule text filter.
        user: Exact user filter.
        page: Zero-based page number.
        page_size: Requested page size.

    Returns:
        Tuple of run rows and pre-page total.
    """

    safe_page = page if page >= 0 else 0
    safe_page_size = min(max(page_size, PAGE_SIZE_MIN), PAGE_SIZE_MAX)
    with _connect(path) as conn:
        responsible_nodes_column = _responsible_nodes_filter_column(conn)
        clauses, params = _filtered_clauses(
            responsible_nodes_column=responsible_nodes_column,
            outcome=outcome,
            since=since,
            until=until,
            job_query=job_query,
            status=status,
            symptom=symptom,
            node=node,
            rule=rule,
            user=user,
        )
        count_sql = _append_where(
            "SELECT COUNT(*) AS n FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id",
            clauses,
        )
        list_sql = _append_where(_run_select_sql(conn), clauses)
        list_sql += " ORDER BY c.end_time DESC LIMIT ? OFFSET ?"
        total = conn.execute(count_sql, tuple(params)).fetchone()["n"]
        rows = conn.execute(
            list_sql,
            tuple([*params, safe_page_size, safe_page * safe_page_size]),
        ).fetchall()
    return ([_row_to_run(r) for r in rows], total)


def _iso_next_day(iso_date: str) -> str:
    """Add one day to an ISO date string for half-open upper bounds."""
    from datetime import date, timedelta

    d = date.fromisoformat(iso_date)
    return (d + timedelta(days=1)).isoformat()


def date_range_bounds(path: str) -> tuple[str | None, str | None]:
    """Return (min, max) of c.end_time in the DB as ISO strings, or (None, None)
    when empty. Used by the webui to populate the date picker without pulling
    the entire row set."""
    with _connect(path) as conn:
        row = conn.execute("SELECT MIN(end_time) AS lo, MAX(end_time) AS hi FROM execution_context").fetchone()
    return (row["lo"], row["hi"]) if row else (None, None)


def get_run(path: str, run_id: str) -> RunRow | None:
    with _connect(path) as conn:
        row = conn.execute(_run_select_sql(conn) + " WHERE r.id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row else None


def _outcome_counts_conn(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT outcome, COUNT(*) AS n FROM attribution_run GROUP BY outcome").fetchall()
    return {(r["outcome"] or ""): r["n"] for r in rows}


def outcome_counts(path: str) -> dict[str, int]:
    """Returns {outcome: count}. Empty-string outcomes are kept as "" (not coerced)
    so the template can branch with `if outcome` and produce a valid CSS class."""
    with _connect(path) as conn:
        return _outcome_counts_conn(conn)


def db_summary(path: str) -> dict[str, Any]:
    # Single connection for all summary queries so the four results come from
    # one snapshot (also halves file-handle churn on every dashboard load).
    with _connect(path) as conn:
        sv = conn.execute("SELECT version, applied_at FROM schema_version ORDER BY applied_at DESC LIMIT 1").fetchone()
        n_contexts = conn.execute("SELECT COUNT(*) AS n FROM execution_context").fetchone()["n"]
        n_runs = conn.execute("SELECT COUNT(*) AS n FROM attribution_run").fetchone()["n"]
        by_outcome = _outcome_counts_conn(conn)
    return {
        "schema_version": sv["version"] if sv else None,
        "schema_applied_at": sv["applied_at"] if sv else None,
        "n_contexts": n_contexts,
        "n_runs": n_runs,
        "by_outcome": by_outcome,
    }


def runs_grouped_by_outcome(path: str) -> dict[str, list[RunRow]]:
    runs = list_runs(path)
    grouped: dict[str, list[RunRow]] = {}
    for r in runs:
        grouped.setdefault(r.outcome or "(empty)", []).append(r)
    return grouped


def symptom_summary(run: RunRow) -> list[tuple[str, int]]:
    """Return [(symptom_name, occurrence_count), ...] sorted by count desc."""
    symptoms = run.raw_result.get("symptoms") or {}
    counts = Counter()
    for name, sites in symptoms.items():
        counts[name] = len(sites or [])
    return counts.most_common()


def _parse_iso_date(value: str | None) -> date | None:
    """Parse an ISO calendar date.

    Args:
        value: Date string.

    Returns:
        Parsed date, or None when absent or invalid.
    """

    if value is None:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _date_spine(since: str | None, until: str | None, lo: str | None, hi: str | None) -> list[str]:
    """Build an inclusive day spine for dashboard charts.

    Args:
        since: Requested lower date bound.
        until: Requested upper date bound.
        lo: Dataset lower bound.
        hi: Dataset upper bound.

    Returns:
        ISO date strings.
    """

    start = _parse_iso_date(since)
    if start is None:
        start = _parse_iso_date(lo)
    end = _parse_iso_date(until)
    if end is None:
        end = _parse_iso_date(hi)
    if start is None or end is None:
        return []
    if start > end:
        start, end = end, start
    days: list[str] = []
    cursor = start
    while cursor <= end:
        days.append(cursor.isoformat())
        cursor = cursor + timedelta(days=1)
    return days


def _daily_breakdown_conn(
    conn: sqlite3.Connection,
    *,
    outcome: str,
    days: list[str],
    since: str | None,
    until: str | None,
) -> dict[str, Any]:
    """Return per-day top root-symptom breakdown for one outcome.

    Args:
        conn: SQLite connection.
        outcome: Raw NVIDIA Attribution outcome.
        days: Inclusive chart day spine.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.

    Returns:
        React ``DailyBreakdown`` shaped dictionary.
    """

    clauses, params = _filtered_clauses(outcome=outcome, since=since, until=until)
    sql = _append_where(
        "SELECT substr(c.end_time, 1, 10) AS day, r.raw_result_json "
        "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id",
        clauses,
    )
    rows = conn.execute(sql, tuple(params)).fetchall()
    day_index = {day: i for i, day in enumerate(days)}
    by_symptom: dict[str, list[int]] = {}
    for row in rows:
        day = row["day"]
        if day not in day_index:
            continue
        try:
            raw_result = json.loads(row["raw_result_json"] or "{}")
        except json.JSONDecodeError:
            raw_result = {}
        verdict = raw_result.get("verdict") if isinstance(raw_result, dict) else {}
        root = verdict.get("root_symptom") if isinstance(verdict, dict) else {}
        symptom = root.get("symptom") if isinstance(root, dict) else None
        label = symptom if isinstance(symptom, str) and symptom != "" else "(unnamed)"
        values = by_symptom.setdefault(label, [0 for _ in days])
        values[day_index[day]] += 1

    ranked = sorted(
        ({"symptom": symptom, "data": values, "total": sum(values)} for symptom, values in by_symptom.items()),
        key=lambda item: (-item["total"], item["symptom"]),
    )
    top = ranked[:7]
    rest = ranked[7:]
    if len(rest) > 0:
        other = [0 for _ in days]
        for item in rest:
            for idx, value in enumerate(item["data"]):
                other[idx] += value
        top.append({"symptom": "Other", "data": other, "total": sum(other)})
    return {"days": days, "series": top, "total": sum(item["total"] for item in ranked)}


def _table_exists_conn(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return whether a SQLite table exists.

    Args:
        conn: SQLite connection.
        table_name: Table name.

    Returns:
        True when the table exists.
    """

    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1", (table_name,)
    ).fetchone()
    return row is not None


def _rollup_clauses(
    *,
    outcome: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> tuple[list[str], list[Any]]:
    """Build WHERE clauses for ``webui_run_fact``.

    Args:
        outcome: Raw NVIDIA Attribution outcome.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.

    Returns:
        SQL clause fragments and bound parameters.
    """

    clauses: list[str] = []
    params: list[Any] = []
    if outcome is not None:
        clauses.append("outcome = ?")
        params.append(outcome)
    if since is not None:
        clauses.append("end_time >= ?")
        params.append(since)
    if until is not None:
        clauses.append("end_time < ?")
        params.append(_iso_next_day(until))
    return clauses, params


def _append_rollup_where(sql: str, clauses: list[str]) -> str:
    """Append rollup WHERE clauses to a SQL statement.

    Args:
        sql: Base SQL statement.
        clauses: WHERE clause fragments.

    Returns:
        SQL with a WHERE section when clauses are present.
    """

    if len(clauses) == 0:
        return sql
    return sql + " WHERE " + " AND ".join(clauses)


def _rollup_window_gpu_count_conn(
    conn: sqlite3.Connection,
    clauses: list[str],
    params: list[Any],
    *,
    fallback_gpu_count: int,
) -> int:
    """Return unique-node GPU capacity for a rollup-filtered window.

    Args:
        conn: SQLite connection.
        clauses: Rollup WHERE clauses.
        params: Bound parameters for ``clauses``.
        fallback_gpu_count: Value to use for rollup-only fixtures without
            matching normalized rows.

    Returns:
        Window GPU capacity based on unique nodes when node data is available.
    """

    rollup_filter = _append_rollup_where("SELECT run_id FROM webui_run_fact", clauses)
    gpus_per_node = _gpus_per_node_sql("c")
    sql = (
        "SELECT COALESCE(SUM(gpus_per_node), 0) AS gpu_count FROM ("
        "SELECT node.value AS node_name, MAX(" + gpus_per_node + ") AS gpus_per_node "
        "FROM (" + rollup_filter + ") f "
        "JOIN attribution_run r ON r.id = f.run_id "
        "JOIN execution_context c ON c.id = r.execution_context_id "
        "JOIN json_each(c.nodes_json) AS node "
        "WHERE node.value IS NOT NULL AND node.value != '' "
        "GROUP BY node.value)"
    )
    row = conn.execute(sql, tuple(params)).fetchone()
    gpu_count = int(row["gpu_count"] if row is not None and row["gpu_count"] is not None else 0)
    if gpu_count > 0:
        return gpu_count
    return fallback_gpu_count


def _rollup_node_day_gpu_count_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
) -> int | None:
    """Return window GPU capacity from the node-day rollup table.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.

    Returns:
        Window GPU capacity when the rollup table exists, otherwise None.
    """

    if _table_exists_conn(conn, "webui_node_day_fact") is False:
        return None

    clauses: list[str] = []
    params: list[Any] = []
    if since is not None:
        clauses.append("end_day >= ?")
        params.append(since)
    if until is not None:
        clauses.append("end_day < ?")
        params.append(_iso_next_day(until))
    inner = _append_rollup_where(
        "SELECT node_name, MAX(gpus_per_node) AS gpus_per_node FROM webui_node_day_fact",
        clauses,
    )
    sql = "SELECT COALESCE(SUM(gpus_per_node), 0) AS gpu_count FROM (" + inner + " GROUP BY node_name)"
    row = conn.execute(sql, tuple(params)).fetchone()
    return int(row["gpu_count"] if row is not None and row["gpu_count"] is not None else 0)


def _daily_rollup_breakdown_conn(
    conn: sqlite3.Connection,
    *,
    outcome: str,
    days: list[str],
    since: str | None,
    until: str | None,
) -> dict[str, Any]:
    """Return per-day root-symptom breakdown from the rollup table.

    Args:
        conn: SQLite connection.
        outcome: Raw NVIDIA Attribution outcome.
        days: Inclusive chart day spine.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.

    Returns:
        React ``DailyBreakdown`` shaped dictionary.
    """

    clauses, params = _rollup_clauses(outcome=outcome, since=since, until=until)
    sql = _append_rollup_where(
        "SELECT end_day AS day, root_symptom AS symptom, COUNT(*) AS n FROM webui_run_fact",
        clauses,
    )
    sql += " GROUP BY day, symptom"
    rows = conn.execute(sql, tuple(params)).fetchall()
    day_index = {day: i for i, day in enumerate(days)}
    by_symptom: dict[str, list[int]] = {}
    for row in rows:
        day = row["day"]
        if day not in day_index:
            continue
        raw_symptom = row["symptom"]
        if isinstance(raw_symptom, str) and raw_symptom != "":
            symptom = raw_symptom
        else:
            symptom = "(unnamed)"
        values = by_symptom.setdefault(symptom, [0 for _ in days])
        values[day_index[day]] += int(row["n"])

    ranked = sorted(
        ({"symptom": symptom, "data": values, "total": sum(values)} for symptom, values in by_symptom.items()),
        key=lambda item: (-item["total"], item["symptom"]),
    )
    top = ranked[:7]
    rest = ranked[7:]
    if len(rest) > 0:
        other = [0 for _ in days]
        for item in rest:
            for idx, value in enumerate(item["data"]):
                other[idx] += value
        top.append({"symptom": "Other", "data": other, "total": sum(other)})
    return {"days": days, "series": top, "total": sum(item["total"] for item in ranked)}


def _rollup_symptom_ledger_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
    outcome: str | None = None,
    limit: int = PARETO_LEDGER_LIMIT,
) -> list[dict[str, Any]]:
    """Return failure symptom ledger entries from the rollup table.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        outcome: Optional frontend outcome key or raw NVIDIA Attribution outcome.
        limit: Maximum number of symptoms to return.

    Returns:
        Symptom ledger entries keyed for React.
    """

    clauses, params = _rollup_clauses(outcome=_normalize_outcome(outcome), since=since, until=until)
    clauses.append("outcome != ?")
    params.append("Pass")
    sql = _append_rollup_where(
        "SELECT root_symptom, outcome, COUNT(*) AS n, SUM(gpu_hours) AS gpu_hours FROM webui_run_fact",
        clauses,
    )
    sql += " GROUP BY root_symptom, outcome"
    rows = conn.execute(sql, tuple(params)).fetchall()
    by_symptom: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_outcome = row["outcome"]
        outcome_key = OUTCOME_KEY_BY_RAW.get(raw_outcome if isinstance(raw_outcome, str) else "")
        if outcome_key is None:
            continue
        raw_symptom = row["root_symptom"]
        if isinstance(raw_symptom, str) and raw_symptom != "":
            symptom = raw_symptom
        else:
            symptom = "(unnamed)"
        entry = by_symptom.setdefault(
            symptom,
            {
                "symptom": symptom,
                "count": 0,
                "nodeHours": 0.0,
                "outcomes": {},
            },
        )
        n = int(row["n"])
        entry["count"] += n
        entry["nodeHours"] += float(row["gpu_hours"] if row["gpu_hours"] is not None else 0.0)
        entry["outcomes"][outcome_key] = entry["outcomes"].get(outcome_key, 0) + n
    return sorted(by_symptom.values(), key=lambda item: (-item["count"], item["symptom"]))[:limit]


def _rollup_rule_ledger_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
    outcome: str | None = None,
    rollup_columns: set[str] | None = None,
    limit: int = PARETO_LEDGER_LIMIT,
) -> list[dict[str, Any]]:
    """Return the top root-rule ledger entries from the rollup table.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        outcome: Optional frontend outcome key or raw NVIDIA Attribution outcome.
        rollup_columns: Columns present on ``webui_run_fact``.
        limit: Maximum number of rules to return.

    Returns:
        Rule ledger entries keyed for React.
    """

    if rollup_columns is None:
        rollup_columns = _table_columns(conn, "webui_run_fact")
    if "root_rule" in rollup_columns:
        clauses, params = _rollup_clauses(outcome=_normalize_outcome(outcome), since=since, until=until)
        clauses.append("outcome != ?")
        params.append("Pass")
        sql = _append_rollup_where(
            "SELECT root_rule, outcome, COUNT(*) AS n, SUM(gpu_hours) AS gpu_hours FROM webui_run_fact",
            clauses,
        )
        sql += " GROUP BY root_rule, outcome"
    else:
        raw_clauses, params = _rollup_clauses(outcome=_normalize_outcome(outcome), since=since, until=until)
        clauses = [clause.replace("end_time", "f.end_time").replace("outcome", "f.outcome") for clause in raw_clauses]
        clauses.append("f.outcome != ?")
        params.append("Pass")
        where_sql = " WHERE " + " AND ".join(clauses)
        sql = (
            "SELECT json_extract(r.raw_result_json, '$.verdict.root_symptom.source.rule') AS root_rule, "
            "f.outcome, COUNT(*) AS n, SUM(f.gpu_hours) AS gpu_hours "
            "FROM webui_run_fact f "
            "LEFT JOIN attribution_run r ON r.id = f.run_id " + where_sql + " GROUP BY root_rule, f.outcome"
        )
    rows = conn.execute(sql, tuple(params)).fetchall()
    by_rule: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_outcome = row["outcome"]
        outcome_key = OUTCOME_KEY_BY_RAW.get(raw_outcome if isinstance(raw_outcome, str) else "")
        if outcome_key is None:
            continue
        raw_rule = row["root_rule"]
        if isinstance(raw_rule, str) and raw_rule != "":
            rule = raw_rule
        else:
            rule = "(unnamed)"
        entry = by_rule.setdefault(
            rule,
            {
                "rule": rule,
                "count": 0,
                "nodeHours": 0.0,
                "outcomes": {},
            },
        )
        n = int(row["n"])
        entry["count"] += n
        entry["nodeHours"] += float(row["gpu_hours"] if row["gpu_hours"] is not None else 0.0)
        entry["outcomes"][outcome_key] = entry["outcomes"].get(outcome_key, 0) + n
    return sorted(by_rule.values(), key=lambda item: (-item["count"], item["rule"]))[:limit]


def _rollup_user_ledger_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
    outcome: str | None = None,
    rollup_columns: set[str] | None = None,
    limit: int = PARETO_LEDGER_LIMIT,
) -> list[dict[str, Any]]:
    """Return the top user ledger entries from the rollup table.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        outcome: Optional frontend outcome key or raw NVIDIA Attribution outcome.
        rollup_columns: Columns present on ``webui_run_fact``.
        limit: Maximum number of users to return.

    Returns:
        User ledger entries keyed for React.
    """

    if rollup_columns is None:
        rollup_columns = _table_columns(conn, "webui_run_fact")
    if "user_name" in rollup_columns:
        clauses, params = _rollup_clauses(outcome=_normalize_outcome(outcome), since=since, until=until)
        clauses.append("outcome != ?")
        params.append("Pass")
        sql = _append_rollup_where(
            "SELECT user_name, outcome, COUNT(*) AS n, SUM(gpu_hours) AS gpu_hours FROM webui_run_fact",
            clauses,
        )
        sql += " GROUP BY user_name, outcome"
    else:
        raw_clauses, params = _rollup_clauses(outcome=_normalize_outcome(outcome), since=since, until=until)
        clauses = [clause.replace("end_time", "f.end_time").replace("outcome", "f.outcome") for clause in raw_clauses]
        clauses.append("f.outcome != ?")
        params.append("Pass")
        where_sql = " WHERE " + " AND ".join(clauses)
        sql = (
            "SELECT c.user_name, f.outcome, COUNT(*) AS n, SUM(f.gpu_hours) AS gpu_hours "
            "FROM webui_run_fact f "
            "LEFT JOIN attribution_run r ON r.id = f.run_id "
            "LEFT JOIN execution_context c ON c.id = r.execution_context_id "
            + where_sql
            + " GROUP BY c.user_name, f.outcome"
        )
    rows = conn.execute(sql, tuple(params)).fetchall()
    by_user: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_outcome = row["outcome"]
        outcome_key = OUTCOME_KEY_BY_RAW.get(raw_outcome if isinstance(raw_outcome, str) else "")
        if outcome_key is None:
            continue
        raw_user = row["user_name"]
        if isinstance(raw_user, str) and raw_user != "":
            user = raw_user
        else:
            user = "(unknown)"
        entry = by_user.setdefault(
            user,
            {
                "user": user,
                "count": 0,
                "nodeHours": 0.0,
                "outcomes": {},
            },
        )
        n = int(row["n"])
        entry["count"] += n
        entry["nodeHours"] += float(row["gpu_hours"] if row["gpu_hours"] is not None else 0.0)
        entry["outcomes"][outcome_key] = entry["outcomes"].get(outcome_key, 0) + n
    return sorted(by_user.values(), key=lambda item: (-item["count"], item["user"]))[:limit]


def _rollup_outcome_ledger_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
) -> list[dict[str, Any]]:
    """Return failure outcome ledger entries from the rollup table.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.

    Returns:
        Outcome ledger entries keyed for React.
    """

    clauses, params = _rollup_clauses(since=since, until=until)
    clauses.append("outcome != ?")
    params.append("Pass")
    sql = _append_rollup_where(
        "SELECT outcome, COUNT(*) AS n, SUM(gpu_hours) AS gpu_hours FROM webui_run_fact",
        clauses,
    )
    sql += " GROUP BY outcome"
    rows = conn.execute(sql, tuple(params)).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        raw_outcome = row["outcome"]
        outcome_key = OUTCOME_KEY_BY_RAW.get(raw_outcome if isinstance(raw_outcome, str) else "")
        if outcome_key is None:
            continue
        out.append(
            {
                "outcome": outcome_key,
                "count": int(row["n"]),
                "nodeHours": float(row["gpu_hours"] if row["gpu_hours"] is not None else 0.0),
            }
        )
    return sorted(out, key=lambda item: (-item["count"], item["outcome"]))


def _rollup_node_ledger_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
    outcome: str | None = None,
    limit: int = NODE_LEDGER_LIMIT,
) -> list[dict[str, Any]]:
    """Return the top responsible-node ledger entries from the rollup table.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        outcome: Optional frontend outcome key or raw NVIDIA Attribution outcome.
        limit: Maximum number of nodes to return.

    Returns:
        Node ledger entries keyed for React.
    """

    raw_clauses, params = _rollup_clauses(outcome=_normalize_outcome(outcome), since=since, until=until)
    clauses = [clause.replace("end_time", "f.end_time").replace("outcome", "f.outcome") for clause in raw_clauses]
    clauses.append("f.outcome != ?")
    params.append("Pass")
    responsible_nodes_column = _responsible_nodes_filter_column(conn)
    clauses.append(f"{responsible_nodes_column} != ?")
    params.append("[]")
    clauses.append(f"{responsible_nodes_column} != ?")
    params.append("")
    where_sql = " WHERE " + " AND ".join(clauses)
    sql = (
        f"SELECT {responsible_nodes_column} AS responsible_nodes_json, "
        "c.nodes_json, f.outcome, f.root_symptom, f.gpu_hours "
        "FROM webui_run_fact f "
        "JOIN attribution_run r ON r.id = f.run_id "
        "JOIN execution_context c ON c.id = r.execution_context_id " + where_sql
    )
    rows = conn.execute(sql, tuple(params)).fetchall()
    by_node: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_outcome = row["outcome"]
        outcome_key = OUTCOME_KEY_BY_RAW.get(raw_outcome if isinstance(raw_outcome, str) else "")
        if outcome_key is None:
            continue
        try:
            responsible_nodes = json.loads(row["responsible_nodes_json"] or "[]")
        except json.JSONDecodeError:
            responsible_nodes = []
        if not isinstance(responsible_nodes, list):
            continue
        try:
            allocated_nodes = json.loads(row["nodes_json"] or "[]")
        except json.JSONDecodeError:
            allocated_nodes = []
        if isinstance(allocated_nodes, list) and len(allocated_nodes) > 0:
            node_count = len(allocated_nodes)
        else:
            node_count = 1
        raw_symptom = row["root_symptom"]
        if isinstance(raw_symptom, str) and raw_symptom != "":
            symptom = raw_symptom
        else:
            symptom = "(unnamed)"
        node_hours = float(row["gpu_hours"] if row["gpu_hours"] is not None else 0.0) / node_count
        for raw_node in responsible_nodes:
            node = raw_node if isinstance(raw_node, str) else ""
            if node == "":
                continue
            entry = by_node.setdefault(
                node,
                {
                    "node": node,
                    "count": 0,
                    "outcomes": {},
                    "symptoms": {},
                    "nodeHours": 0.0,
                },
            )
            entry["count"] += 1
            entry["nodeHours"] += node_hours
            entry["outcomes"][outcome_key] = entry["outcomes"].get(outcome_key, 0) + 1
            entry["symptoms"][symptom] = entry["symptoms"].get(symptom, 0) + 1
    return sorted(by_node.values(), key=lambda item: (-item["count"], item["node"]))[:limit]


def _schema_meta_conn(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return schema metadata for dashboard responses.

    Args:
        conn: SQLite connection.

    Returns:
        Dashboard version metadata.
    """

    row = conn.execute("SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1").fetchone()
    return {
        "schema": row["version"] if row is not None else None,
        "pipeline": None,
        "service": None,
    }


def _db_total_runs_conn(conn: sqlite3.Connection) -> int:
    """Return the number of runs held by the whole database.

    Args:
        conn: SQLite connection.

    Returns:
        Count of runs across the database.
    """

    # Not count_runs(): that one exists to match list_runs' filters, and this denominator
    # must stay unfiltered while the date window moves.
    row = conn.execute("SELECT COUNT(*) AS n FROM attribution_run").fetchone()
    return int(row["n"] if row is not None else 0)


def _db_date_bounds_conn(conn: sqlite3.Connection) -> tuple[str | None, str | None]:
    """Return the first and last run end days held by the whole database.

    Args:
        conn: SQLite connection.

    Returns:
        Inclusive ``(min, max)`` ISO day pair; both None when the database holds no runs.
    """

    row = conn.execute(
        "SELECT MIN(c.end_time) AS lo, MAX(c.end_time) AS hi "
        "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id"
    ).fetchone()
    if row is None:
        return None, None
    lo = row["lo"][:10] if row["lo"] is not None else None
    hi = row["hi"][:10] if row["hi"] is not None else None
    return lo, hi


def _dashboard_data_from_rollup_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
    database: str,
    ledger_outcome: str | None = None,
) -> dict[str, Any]:
    """Return dashboard data from the denormalized quickstart rollup table.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        database: Display name of the database being served.
        ledger_outcome: Optional outcome filter for concentration ledgers.

    Returns:
        Dashboard payload consumed by the React fleet view.
    """

    clauses, params = _rollup_clauses(since=since, until=until)
    where_sql = " WHERE " + " AND ".join(clauses) if len(clauses) > 0 else ""
    metric_sql = (
        "SELECT COUNT(*) AS windowed_runs, "
        "SUM(gpu_hours) AS gpu_hours, "
        "SUM(gpu_count) AS gpu_count, "
        "SUM(CASE WHEN outcome = 'Fail InfraHW' THEN 1 ELSE 0 END) AS hw_failures, "
        "SUM(CASE WHEN outcome = 'Fail InfraSW' THEN 1 ELSE 0 END) AS sw_failures, "
        "SUM(CASE WHEN outcome = 'Fail MultiCategory' THEN 1 ELSE 0 END) AS mix_failures "
        "FROM webui_run_fact" + where_sql
    )
    # Daily buckets only cover the outcomes the day-by-day charts plot;
    # 'Fail Undetermined' is intentionally omitted (matches DailyBucket in
    # the React interfaces). Window totals come from windowed_runs.
    daily_sql = (
        "SELECT end_day AS day, "
        "SUM(gpu_hours) AS gpu_hours, "
        "SUM(CASE WHEN outcome = 'Fail InfraHW' THEN 1 ELSE 0 END) AS hw, "
        "SUM(CASE WHEN outcome = 'Fail InfraSW' THEN 1 ELSE 0 END) AS sw, "
        "SUM(CASE WHEN outcome = 'Fail MultiCategory' THEN 1 ELSE 0 END) AS mix, "
        "SUM(CASE WHEN outcome = 'Fail User' THEN 1 ELSE 0 END) AS usr, "
        "SUM(CASE WHEN outcome = 'Pass' THEN 1 ELSE 0 END) AS pass "
        "FROM webui_run_fact" + where_sql + " GROUP BY day ORDER BY day"
    )
    metric_row = conn.execute(metric_sql, tuple(params)).fetchone()
    fallback_gpu_count = int(
        metric_row["gpu_count"] if metric_row is not None and metric_row["gpu_count"] is not None else 0
    )
    window_gpu_count = _rollup_node_day_gpu_count_conn(conn, since=since, until=until)
    if window_gpu_count is None:
        window_gpu_count = _rollup_window_gpu_count_conn(
            conn,
            clauses,
            params,
            fallback_gpu_count=fallback_gpu_count,
        )
    rollup_lo_hi = conn.execute("SELECT MIN(end_time) AS lo, MAX(end_time) AS hi FROM webui_run_fact").fetchone()
    db_total_runs = _db_total_runs_conn(conn)
    db_min, db_max = _db_date_bounds_conn(conn)
    days = _date_spine(
        since,
        until,
        rollup_lo_hi["lo"] if rollup_lo_hi is not None else None,
        rollup_lo_hi["hi"] if rollup_lo_hi is not None else None,
    )
    daily_rows = conn.execute(daily_sql, tuple(params)).fetchall()
    hw_breakdown = _daily_rollup_breakdown_conn(conn, outcome="Fail InfraHW", days=days, since=since, until=until)
    sw_breakdown = _daily_rollup_breakdown_conn(conn, outcome="Fail InfraSW", days=days, since=since, until=until)
    rollup_columns = _table_columns(conn, "webui_run_fact")
    symptom_ledger = _rollup_symptom_ledger_conn(conn, since=since, until=until, outcome=ledger_outcome)
    node_ledger = _rollup_node_ledger_conn(conn, since=since, until=until, outcome=ledger_outcome)
    rule_ledger = _rollup_rule_ledger_conn(
        conn, since=since, until=until, outcome=ledger_outcome, rollup_columns=rollup_columns
    )
    user_ledger = _rollup_user_ledger_conn(
        conn, since=since, until=until, outcome=ledger_outcome, rollup_columns=rollup_columns
    )
    outcome_ledger = _rollup_outcome_ledger_conn(conn, since=since, until=until)

    by_day = {row["day"]: row for row in daily_rows}
    daily: list[dict[str, Any]] = []
    for day in days:
        row = by_day.get(day)
        daily.append(
            {
                "date": day,
                "hw": int(row["hw"] if row is not None else 0),
                "sw": int(row["sw"] if row is not None else 0),
                "mix": int(row["mix"] if row is not None else 0),
                "usr": int(row["usr"] if row is not None else 0),
                "pass": int(row["pass"] if row is not None else 0),
                "nodeHours": float(row["gpu_hours"] if row is not None and row["gpu_hours"] is not None else 0.0),
            }
        )

    gpu_hours = float(
        metric_row["gpu_hours"] if metric_row is not None and metric_row["gpu_hours"] is not None else 0.0
    )
    hw_failures = int(
        metric_row["hw_failures"] if metric_row is not None and metric_row["hw_failures"] is not None else 0
    )
    sw_failures = int(
        metric_row["sw_failures"] if metric_row is not None and metric_row["sw_failures"] is not None else 0
    )
    mix_failures = int(
        metric_row["mix_failures"] if metric_row is not None and metric_row["mix_failures"] is not None else 0
    )
    combined_failures = hw_failures + sw_failures
    return {
        "meta": {
            "versions": _schema_meta_conn(conn),
            "dateBounds": {"min": db_min, "max": db_max},
            "windowedRuns": int(metric_row["windowed_runs"] if metric_row is not None else 0),
            "dbTotalRuns": db_total_runs,
            "database": database,
            "returnedRuns": 0,
        },
        "metrics": {
            "windowedRuns": int(metric_row["windowed_runs"] if metric_row is not None else 0),
            "gpuHoursRaw": gpu_hours,
            "gpuCount": window_gpu_count,
            "hwFailures": hw_failures,
            "swFailures": sw_failures,
            "mixFailures": mix_failures,
            "combinedFailures": combined_failures,
            "hwMtbi": 0.0 if hw_failures == 0 else gpu_hours / hw_failures / 1000.0,
            "combinedMtbi": 0.0 if combined_failures == 0 else gpu_hours / combined_failures / 1000.0,
        },
        "daily": daily,
        "hwBreakdown": hw_breakdown,
        "swBreakdown": sw_breakdown,
        "ledgers": {
            "symptom": symptom_ledger,
            "node": node_ledger,
            "user": user_ledger,
            "rule": rule_ledger,
            "outcome": outcome_ledger,
        },
        "movers": [],
    }


def _outcome_ledger_conn(conn: sqlite3.Connection, clauses: list[str], params: list[Any]) -> list[dict[str, Any]]:
    """Return outcome counts and GPU-hours for the dashboard Pareto donut.

    Args:
        conn: SQLite connection.
        clauses: WHERE clause fragments.
        params: Bound parameters.

    Returns:
        Outcome ledger entries keyed for React.
    """

    gpu_expr = _gpu_hours_sql()
    sql = _append_where(
        "SELECT r.outcome AS outcome, COUNT(*) AS n, SUM(" + gpu_expr + ") AS gpu_hours "
        "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id",
        clauses,
    )
    sql += " GROUP BY r.outcome"
    rows = conn.execute(sql, tuple(params)).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        outcome_key = OUTCOME_KEY_BY_RAW.get(row["outcome"] or "")
        if outcome_key is None:
            continue
        out.append(
            {
                "outcome": outcome_key,
                "count": row["n"],
                "nodeHours": float(row["gpu_hours"] or 0.0),
            }
        )
    return out


def _normalized_symptom_ledger_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
    outcome: str | None = None,
    limit: int = PARETO_LEDGER_LIMIT,
) -> list[dict[str, Any]]:
    """Return top root-symptom ledgers from normalized schema tables.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        outcome: Optional frontend outcome key or raw NVIDIA Attribution outcome.
        limit: Maximum number of entries to return.

    Returns:
        Symptom ledger entries keyed for React.
    """

    gpu_expr = _gpu_hours_sql()
    clauses, params = _filtered_clauses(outcome=outcome, since=since, until=until, status="failed")
    sql = _append_where(
        "SELECT json_extract(r.raw_result_json, '$.verdict.root_symptom.symptom') AS root_symptom, "
        "r.outcome AS outcome, COUNT(*) AS n, SUM(" + gpu_expr + ") AS gpu_hours "
        "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id",
        clauses,
    )
    sql += " GROUP BY root_symptom, r.outcome"
    rows = conn.execute(sql, tuple(params)).fetchall()
    by_symptom: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_outcome = row["outcome"]
        outcome_key = OUTCOME_KEY_BY_RAW.get(raw_outcome if isinstance(raw_outcome, str) else "")
        if outcome_key is None:
            continue
        raw_symptom = row["root_symptom"]
        symptom = raw_symptom if isinstance(raw_symptom, str) and raw_symptom != "" else "(unnamed)"
        entry = by_symptom.setdefault(
            symptom,
            {
                "symptom": symptom,
                "count": 0,
                "nodeHours": 0.0,
                "outcomes": {},
            },
        )
        n = int(row["n"])
        entry["count"] += n
        entry["nodeHours"] += float(row["gpu_hours"] if row["gpu_hours"] is not None else 0.0)
        entry["outcomes"][outcome_key] = entry["outcomes"].get(outcome_key, 0) + n
    return sorted(by_symptom.values(), key=lambda item: (-item["count"], item["symptom"]))[:limit]


def _normalized_rule_ledger_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
    outcome: str | None = None,
    limit: int = PARETO_LEDGER_LIMIT,
) -> list[dict[str, Any]]:
    """Return top root-rule ledgers from normalized schema tables.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        outcome: Optional frontend outcome key or raw NVIDIA Attribution outcome.
        limit: Maximum number of entries to return.

    Returns:
        Rule ledger entries keyed for React.
    """

    gpu_expr = _gpu_hours_sql()
    clauses, params = _filtered_clauses(outcome=outcome, since=since, until=until, status="failed")
    sql = _append_where(
        "SELECT json_extract(r.raw_result_json, '$.verdict.root_symptom.source.rule') AS root_rule, "
        "r.outcome AS outcome, COUNT(*) AS n, SUM(" + gpu_expr + ") AS gpu_hours "
        "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id",
        clauses,
    )
    sql += " GROUP BY root_rule, r.outcome"
    rows = conn.execute(sql, tuple(params)).fetchall()
    by_rule: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_outcome = row["outcome"]
        outcome_key = OUTCOME_KEY_BY_RAW.get(raw_outcome if isinstance(raw_outcome, str) else "")
        if outcome_key is None:
            continue
        raw_rule = row["root_rule"]
        rule = raw_rule if isinstance(raw_rule, str) and raw_rule != "" else "(unnamed)"
        entry = by_rule.setdefault(
            rule,
            {
                "rule": rule,
                "count": 0,
                "nodeHours": 0.0,
                "outcomes": {},
            },
        )
        n = int(row["n"])
        entry["count"] += n
        entry["nodeHours"] += float(row["gpu_hours"] if row["gpu_hours"] is not None else 0.0)
        entry["outcomes"][outcome_key] = entry["outcomes"].get(outcome_key, 0) + n
    return sorted(by_rule.values(), key=lambda item: (-item["count"], item["rule"]))[:limit]


def _normalized_user_ledger_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
    outcome: str | None = None,
    limit: int = PARETO_LEDGER_LIMIT,
) -> list[dict[str, Any]]:
    """Return top user ledgers from normalized schema tables.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        outcome: Optional frontend outcome key or raw NVIDIA Attribution outcome.
        limit: Maximum number of entries to return.

    Returns:
        User ledger entries keyed for React.
    """

    gpu_expr = _gpu_hours_sql()
    clauses, params = _filtered_clauses(outcome=outcome, since=since, until=until, status="failed")
    sql = _append_where(
        "SELECT c.user_name AS user_name, r.outcome AS outcome, COUNT(*) AS n, "
        "SUM(" + gpu_expr + ") AS gpu_hours "
        "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id",
        clauses,
    )
    sql += " GROUP BY c.user_name, r.outcome"
    rows = conn.execute(sql, tuple(params)).fetchall()
    by_user: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_outcome = row["outcome"]
        outcome_key = OUTCOME_KEY_BY_RAW.get(raw_outcome if isinstance(raw_outcome, str) else "")
        if outcome_key is None:
            continue
        raw_user = row["user_name"]
        user = raw_user if isinstance(raw_user, str) and raw_user != "" else "(unknown)"
        entry = by_user.setdefault(
            user,
            {
                "user": user,
                "count": 0,
                "nodeHours": 0.0,
                "outcomes": {},
            },
        )
        n = int(row["n"])
        entry["count"] += n
        entry["nodeHours"] += float(row["gpu_hours"] if row["gpu_hours"] is not None else 0.0)
        entry["outcomes"][outcome_key] = entry["outcomes"].get(outcome_key, 0) + n
    return sorted(by_user.values(), key=lambda item: (-item["count"], item["user"]))[:limit]


def _normalized_node_ledger_conn(
    conn: sqlite3.Connection,
    *,
    since: str | None,
    until: str | None,
    outcome: str | None = None,
    limit: int = NODE_LEDGER_LIMIT,
) -> list[dict[str, Any]]:
    """Return top responsible-node ledgers from normalized schema tables.

    Args:
        conn: SQLite connection.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        outcome: Optional frontend outcome key or raw NVIDIA Attribution outcome.
        limit: Maximum number of entries to return.

    Returns:
        Node ledger entries keyed for React.
    """

    gpu_expr = _gpu_hours_sql()
    responsible_nodes_column = _responsible_nodes_filter_column(conn)
    clauses, params = _filtered_clauses(
        responsible_nodes_column=responsible_nodes_column,
        outcome=outcome,
        since=since,
        until=until,
        status="failed",
    )
    clauses.append(f"{responsible_nodes_column} != ?")
    params.append("[]")
    clauses.append(f"{responsible_nodes_column} != ?")
    params.append("")
    sql = _append_where(
        f"SELECT {responsible_nodes_column} AS responsible_nodes_json, c.nodes_json, "
        "r.outcome AS outcome, "
        "json_extract(r.raw_result_json, '$.verdict.root_symptom.symptom') AS root_symptom, "
        "(" + gpu_expr + ") AS gpu_hours "
        "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id",
        clauses,
    )
    rows = conn.execute(sql, tuple(params)).fetchall()
    by_node: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_outcome = row["outcome"]
        outcome_key = OUTCOME_KEY_BY_RAW.get(raw_outcome if isinstance(raw_outcome, str) else "")
        if outcome_key is None:
            continue
        try:
            responsible_nodes = json.loads(row["responsible_nodes_json"] or "[]")
        except json.JSONDecodeError:
            responsible_nodes = []
        if not isinstance(responsible_nodes, list):
            continue
        try:
            allocated_nodes = json.loads(row["nodes_json"] or "[]")
        except json.JSONDecodeError:
            allocated_nodes = []
        node_count = len(allocated_nodes) if isinstance(allocated_nodes, list) and len(allocated_nodes) > 0 else 1
        raw_symptom = row["root_symptom"]
        symptom = raw_symptom if isinstance(raw_symptom, str) and raw_symptom != "" else "(unnamed)"
        node_hours = float(row["gpu_hours"] if row["gpu_hours"] is not None else 0.0) / node_count
        for raw_node in responsible_nodes:
            node = raw_node if isinstance(raw_node, str) else ""
            if node == "":
                continue
            entry = by_node.setdefault(
                node,
                {
                    "node": node,
                    "count": 0,
                    "outcomes": {},
                    "symptoms": {},
                    "nodeHours": 0.0,
                },
            )
            entry["count"] += 1
            entry["nodeHours"] += node_hours
            entry["outcomes"][outcome_key] = entry["outcomes"].get(outcome_key, 0) + 1
            entry["symptoms"][symptom] = entry["symptoms"].get(symptom, 0) + 1
    return sorted(by_node.values(), key=lambda item: (-item["count"], item["node"]))[:limit]


def dashboard_data(
    path: str,
    *,
    since: str | None = None,
    until: str | None = None,
    ledger_outcome: str | None = None,
) -> dict[str, Any]:
    """Return full-window dashboard aggregates without materializing all runs.

    Args:
        path: SQLite database path.
        since: Inclusive lower date bound.
        until: Inclusive upper date bound.
        ledger_outcome: Optional outcome filter for concentration ledgers.

    Returns:
        Dashboard payload consumed by the React fleet view.
    """

    with _connect(path) as conn:
        if _table_exists_conn(conn, "webui_run_fact") is True:
            return _dashboard_data_from_rollup_conn(
                conn,
                since=since,
                until=until,
                database=Path(path).name,
                ledger_outcome=ledger_outcome,
            )

    gpu_expr = _gpu_hours_sql()
    clauses, params = _filtered_clauses(since=since, until=until)
    where_sql = " WHERE " + " AND ".join(clauses) if len(clauses) > 0 else ""
    metric_sql = (
        "SELECT COUNT(*) AS windowed_runs, "
        f"SUM({gpu_expr}) AS gpu_hours, "
        "SUM(CASE WHEN r.outcome = 'Fail InfraHW' THEN 1 ELSE 0 END) AS hw_failures, "
        "SUM(CASE WHEN r.outcome = 'Fail InfraSW' THEN 1 ELSE 0 END) AS sw_failures, "
        "SUM(CASE WHEN r.outcome = 'Fail MultiCategory' THEN 1 ELSE 0 END) AS mix_failures "
        "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id" + where_sql
    )
    # Same intentional omission of 'Fail Undetermined' as the rollup daily_sql above.
    daily_sql = (
        "SELECT substr(c.end_time, 1, 10) AS day, "
        f"SUM({gpu_expr}) AS gpu_hours, "
        "SUM(CASE WHEN r.outcome = 'Fail InfraHW' THEN 1 ELSE 0 END) AS hw, "
        "SUM(CASE WHEN r.outcome = 'Fail InfraSW' THEN 1 ELSE 0 END) AS sw, "
        "SUM(CASE WHEN r.outcome = 'Fail MultiCategory' THEN 1 ELSE 0 END) AS mix, "
        "SUM(CASE WHEN r.outcome = 'Fail User' THEN 1 ELSE 0 END) AS usr, "
        "SUM(CASE WHEN r.outcome = 'Pass' THEN 1 ELSE 0 END) AS pass "
        "FROM attribution_run r JOIN execution_context c ON c.id = r.execution_context_id"
        + where_sql
        + " GROUP BY day ORDER BY day"
    )
    with _connect(path) as conn:
        metric_row = conn.execute(metric_sql, tuple(params)).fetchone()
        window_gpu_count = _window_gpu_count_conn(conn, clauses, params)
        lo_hi = conn.execute("SELECT MIN(end_time) AS lo, MAX(end_time) AS hi FROM execution_context").fetchone()
        db_total_runs = _db_total_runs_conn(conn)
        db_min, db_max = _db_date_bounds_conn(conn)
        summary = db_summary(path)
        days = _date_spine(since, until, lo_hi["lo"] if lo_hi else None, lo_hi["hi"] if lo_hi else None)
        daily_rows = conn.execute(daily_sql, tuple(params)).fetchall()
        hw_breakdown = _daily_breakdown_conn(conn, outcome="Fail InfraHW", days=days, since=since, until=until)
        sw_breakdown = _daily_breakdown_conn(conn, outcome="Fail InfraSW", days=days, since=since, until=until)
        symptom_ledger = _normalized_symptom_ledger_conn(conn, since=since, until=until, outcome=ledger_outcome)
        node_ledger = _normalized_node_ledger_conn(conn, since=since, until=until, outcome=ledger_outcome)
        user_ledger = _normalized_user_ledger_conn(conn, since=since, until=until, outcome=ledger_outcome)
        rule_ledger = _normalized_rule_ledger_conn(conn, since=since, until=until, outcome=ledger_outcome)
        outcome_ledger = _outcome_ledger_conn(conn, clauses + ["r.outcome != ?"], [*params, "Pass"])

    by_day = {row["day"]: row for row in daily_rows}
    daily: list[dict[str, Any]] = []
    for day in days:
        row = by_day.get(day)
        daily.append(
            {
                "date": day,
                "hw": int(row["hw"] if row is not None else 0),
                "sw": int(row["sw"] if row is not None else 0),
                "mix": int(row["mix"] if row is not None else 0),
                "usr": int(row["usr"] if row is not None else 0),
                "pass": int(row["pass"] if row is not None else 0),
                "nodeHours": float(row["gpu_hours"] if row is not None and row["gpu_hours"] is not None else 0.0),
            }
        )

    gpu_hours = float(
        metric_row["gpu_hours"] if metric_row is not None and metric_row["gpu_hours"] is not None else 0.0
    )
    hw_failures = int(
        metric_row["hw_failures"] if metric_row is not None and metric_row["hw_failures"] is not None else 0
    )
    sw_failures = int(
        metric_row["sw_failures"] if metric_row is not None and metric_row["sw_failures"] is not None else 0
    )
    mix_failures = int(
        metric_row["mix_failures"] if metric_row is not None and metric_row["mix_failures"] is not None else 0
    )
    combined_failures = hw_failures + sw_failures
    return {
        "meta": {
            "versions": {
                "schema": summary.get("schema_version"),
                "pipeline": None,
                "service": None,
            },
            "dateBounds": {"min": db_min, "max": db_max},
            "windowedRuns": int(metric_row["windowed_runs"] if metric_row is not None else 0),
            "dbTotalRuns": db_total_runs,
            "database": Path(path).name,
            "returnedRuns": 0,
        },
        "metrics": {
            "windowedRuns": int(metric_row["windowed_runs"] if metric_row is not None else 0),
            "gpuHoursRaw": gpu_hours,
            "gpuCount": window_gpu_count,
            "hwFailures": hw_failures,
            "swFailures": sw_failures,
            "mixFailures": mix_failures,
            "combinedFailures": combined_failures,
            "hwMtbi": 0.0 if hw_failures == 0 else gpu_hours / hw_failures / 1000.0,
            "combinedMtbi": 0.0 if combined_failures == 0 else gpu_hours / combined_failures / 1000.0,
        },
        "daily": daily,
        "hwBreakdown": hw_breakdown,
        "swBreakdown": sw_breakdown,
        "ledgers": {
            "symptom": symptom_ledger,
            "node": node_ledger,
            "user": user_ledger,
            "rule": rule_ledger,
            "outcome": outcome_ledger,
        },
        "movers": [],
    }
